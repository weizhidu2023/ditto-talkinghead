#!/usr/bin/env python3
"""
Render a single frame with all expression channels zeroed.

This script registers the source image to obtain source frame data, builds
an x_d_info (either from a provided kp_info NPZ or by copying x_s_info),
sets the entire `exp` tensor to zeros, then runs the rendering pipeline
(motion_stitch -> warp_f3d -> decode_f3d -> putback) and saves the final
frame as a PNG for inspection.

Usage:
  python tools/zero_exp_render.py --image face.jpg --cfg_pkl ./checkpoints/cfg.pkl --data_root ./checkpoints --out out/zero_exp.png

If `--kp_npz` is provided and contains `x_d_info` or a compatible dict with
`exp`, that will be used as the base for zeroing; otherwise `x_s_info` (from
avatar registration) will be used.
"""
import argparse
import os
import sys
import numpy as np
from PIL import Image

# Ensure repo root on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_kp_info(npz_path):
    import numpy as _np
    data = _np.load(npz_path, allow_pickle=True)
    if 'kp_info' in data:
        return data['kp_info'].item()
    keys = data.files
    if set(['kp', 'exp']) <= set(keys):
        kp_info = {k: data[k] for k in keys}
        return kp_info
    for k in keys:
        val = data[k]
        if isinstance(val, _np.ndarray) and val.dtype == object and val.size == 1:
            maybe = val.item()
            if isinstance(maybe, dict):
                return maybe
    raise RuntimeError(f"Cannot interpret NPZ file {npz_path}; expected 'kp_info' or keys 'kp','exp' etc.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Source image (face)')
    parser.add_argument('--out', required=True, help='Output PNG path')
    parser.add_argument('--kp_npz', default=None, help='Optional kp_info NPZ to use as x_d_info')
    parser.add_argument('--cfg_pkl', default=None, help='Path to pipeline cfg pkl (required)')
    parser.add_argument('--data_root', default=None, help='Data root for model files (required)')
    args = parser.parse_args()

    if args.cfg_pkl is None or args.data_root is None:
        raise RuntimeError('--cfg_pkl and --data_root are required')

    # Lazy import of pipeline components
    try:
        from core.atomic_components.cfg import parse_cfg
        from core.atomic_components.avatar_registrar import AvatarRegistrar
        from core.atomic_components.motion_stitch import MotionStitch
        from core.atomic_components.warp_f3d import WarpF3D
        from core.atomic_components.decode_f3d import DecodeF3D
        from core.atomic_components.putback import PutBackNumpy
    except Exception as e:
        raise RuntimeError('Failed to import rendering components: ' + str(e))

    # parse cfg and instantiate components
    cfgs = parse_cfg(args.cfg_pkl, args.data_root)
    avatar_registrar_cfg = cfgs[0]
    stitch_network_cfg = cfgs[3]
    warp_network_cfg = cfgs[4]
    decoder_cfg = cfgs[5]

    avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)
    motion_stitch = MotionStitch(stitch_network_cfg)
    warp_f3d = WarpF3D(warp_network_cfg)
    decode_f3d = DecodeF3D(decoder_cfg)
    putback = PutBackNumpy()

    # Register source image (one-frame)
    source_info = avatar_registrar.register(args.image, max_dim=1920, n_frames=1)
    x_s_info = source_info['x_s_info_lst'][0]
    f_s = source_info['f_s_lst'][0]
    M_c2o = source_info['M_c2o_lst'][0]
    frame_rgb = source_info['img_rgb_lst'][0]

    # Build x_d_info_base
    if args.kp_npz:
        kp_info = load_kp_info(args.kp_npz)
        # prefer x_d_info if present
        if isinstance(kp_info, dict) and 'x_d_info' in kp_info and kp_info['x_d_info']:
            x_d_info_base = kp_info['x_d_info']
        elif isinstance(kp_info, dict) and ('exp' in kp_info or 'kp' in kp_info):
            x_d_info_base = kp_info
        else:
            x_d_info_base = x_s_info.copy()
    else:
        # copy source info to use as base
        x_d_info_base = dict(x_s_info)

    # Ensure exp exists; if not, create zeros matching expected shape
    if 'exp' in x_d_info_base and x_d_info_base['exp'] is not None:
        zero_exp = np.zeros_like(x_d_info_base['exp'])
    else:
        # assume (1, num_kp, 3) using x_s_info if available
        if 'exp' in x_s_info and x_s_info['exp'] is not None:
            zero_exp = np.zeros_like(x_s_info['exp'])
        else:
            raise RuntimeError('Cannot determine shape for exp; provide kp_npz or use image with detectable face')

    x_d_info_base['exp'] = zero_exp

    # Setup motion_stitch with source info so pose_s etc. are available
    motion_stitch.setup(is_image_flag=source_info['is_image_flag'], x_s_info=x_s_info)

    # Run motion_stitch to obtain x_s and x_d (motion_stitch may also apply other adjustments)
    x_s_res, x_d_res = motion_stitch(x_s_info, x_d_info_base)

    # Warp, decode and put back to original frame
    f_3d = warp_f3d(f_s, x_s_res, x_d_res)
    render_img = decode_f3d(f_3d)
    # decode may return batch dim
    if isinstance(render_img, np.ndarray) and render_img.ndim == 4 and render_img.shape[0] == 1:
        render_img = render_img[0]

    # Normalize to uint8
    if isinstance(render_img, np.ndarray):
        if np.issubdtype(render_img.dtype, np.floating):
            mx = float(render_img.max())
            if mx <= 1.1:
                render_img = render_img * 255.0
            render_img = np.clip(render_img, 0, 255).astype(np.uint8)
        else:
            render_img = np.clip(render_img, 0, 255).astype(np.uint8)
    else:
        render_img = np.array(render_img, dtype=np.uint8)

    final_frame = putback(frame_rgb, render_img, M_c2o)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    Image.fromarray(final_frame).save(args.out)
    print('Saved zero-exp render to', args.out)


if __name__ == '__main__':
    main()
