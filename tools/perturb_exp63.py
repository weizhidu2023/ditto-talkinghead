#!/usr/bin/env python3
"""
Perturb each dimension of the 63-D expression vector and visualize effect.

This script loads a kp_info NPZ (same format as visualize_keypoints), and for
each keypoint index (0..20) and coord (x/y/z) applies a small additive delta to
`exp` and computes transformed keypoints via
`core.atomic_components.motion_stitch.transform_keypoint` (so the repo must be
importable). It saves side-by-side images (original / perturbed) and a CSV
report describing each perturbation.

Usage:
  python tools/perturb_exp63.py --kp_npz kp_info.npz --image face.jpg --out_dir out --delta 0.05

Note: This script does NOT run the full rendering/inference pipeline; it
visualizes how the implied 3D keypoint positions change when you perturb each
component of the 63-D `exp` vector. This helps map each flattened index to a
facial region.
"""
import argparse
import os
import sys
import csv
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so we can import `core` package when running
# this script directly. This mirrors running `PYTHONPATH=. python ...`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# Optional: rendering pipeline components
from typing import Optional


def load_kp_info(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if 'kp_info' in data:
        return data['kp_info'].item()
    keys = data.files
    if set(['kp', 'exp']) <= set(keys):
        kp_info = {k: data[k] for k in keys}
        return kp_info
    for k in keys:
        val = data[k]
        if isinstance(val, np.ndarray) and val.dtype == object and val.size == 1:
            maybe = val.item()
            if isinstance(maybe, dict):
                return maybe
    raise RuntimeError(f"Cannot interpret NPZ file {npz_path}; expected 'kp_info' or keys 'kp','exp' etc.")


def side_by_side_plot(image_path, kps_orig, kps_pert, labels, out_path, title=None):
    img = Image.open(image_path).convert('RGB')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Normalize keypoints to image pixel coordinates so differences are visible.
    img_w, img_h = img.size

    def map_to_image(kps_a, kps_b, img_w, img_h):
        """
        Map canonical 3D keypoint coordinates to image pixels.

        Ditto keypoints are in a canonical face-local coordinate system
        (approximately centered at 0, with typical x/y in ~[-0.5,0.5]).
        Use a fixed orthographic mapping: screen_x = (x + 0.5) * img_w,
        screen_y = (0.5 - y) * img_h. This avoids min/max scaling which can
        shift the points away from the face when the image contains extra
        padding or different crop.
        """
        import numpy as _np

        def _map(kps):
            xs = (kps[:, 0] + 0.5) * img_w
            ys = (0.5 - kps[:, 1]) * img_h
            return _np.stack([xs, ys], axis=1)

        return _map(kps_a), _map(kps_b)

    kps_orig_mapped, kps_pert_mapped = map_to_image(kps_orig, kps_pert, img_w, img_h)

    for ax, kps, sub in zip(axes, (kps_orig_mapped, kps_pert_mapped), ('original', 'perturbed')):
        ax.imshow(img)
        x = kps[:, 0]
        y = kps[:, 1]
        ax.scatter(x, y, c='r')
        for i, (xx, yy) in enumerate(kps):
            ax.text(xx + 2, yy + 2, str(labels[i]), color='yellow', fontsize=8, weight='bold')
        ax.set_title(sub)
        # Lock axes to image pixel coordinates to avoid autoscaling when some
        # keypoints fall outside the image (which would shrink the displayed image).
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_aspect('equal')
        ax.axis('off')
    if title:
        fig.suptitle(title)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kp_npz', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out_dir', default='outputs/perturb_exp', help='Output directory')
    parser.add_argument('--delta', type=float, default=0.05, help='Perturbation magnitude (additive)')
    parser.add_argument('--axis', choices=['x', 'y', 'z', 'all'], default='all', help='Which axis(es) to perturb')
    parser.add_argument('--render', action='store_true', help='Also run the rendering pipeline to produce full frames for each perturbation')
    parser.add_argument('--cfg_pkl', type=str, default=None, help='Path to pipeline cfg pkl (required if --render)')
    parser.add_argument('--data_root', type=str, default=None, help='Data root for model files (required if --render)')
    args = parser.parse_args()

    kp_info = load_kp_info(args.kp_npz)

    # Handle nested kp_info saved as {'x_s_info': {...}, 'x_d_info': {...}}
    # Prefer source (`x_s_info`) if present, otherwise use `x_d_info`.
    if isinstance(kp_info, dict) and ('x_s_info' in kp_info or 'x_d_info' in kp_info):
        if 'x_s_info' in kp_info and kp_info['x_s_info']:
            print('Using nested `x_s_info` from kp_info')
            kp_info = kp_info['x_s_info']
        else:
            print('Using nested `x_d_info` from kp_info')
            kp_info = kp_info['x_d_info']

    try:
        from core.atomic_components.motion_stitch import transform_keypoint
    except Exception as e:
        raise RuntimeError('Failed to import transform_keypoint: ' + str(e))

    # If rendering is requested, import and prepare pipeline components lazily
    if args.render:
        if not args.cfg_pkl or not args.data_root:
            raise RuntimeError('--render requires --cfg_pkl and --data_root')
        try:
            from core.atomic_components.cfg import parse_cfg
            from core.atomic_components.avatar_registrar import AvatarRegistrar
            from core.atomic_components.motion_stitch import MotionStitch
            from core.atomic_components.warp_f3d import WarpF3D
            from core.atomic_components.decode_f3d import DecodeF3D
            # Import both putback implementations; prefer PutBackNumpy to avoid Cython dtype issues
            from core.atomic_components.putback import PutBack, PutBackNumpy
        except Exception as e:
            raise RuntimeError('Failed to import rendering pipeline components: ' + str(e))

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
        # Use the numpy-based PutBack implementation to avoid Cython dtype mismatch
        putback = PutBackNumpy()

        # register the source image to get f_s, f_s_lst, x_s_info etc.
        # Use one-frame registration (n_frames=1)
        source_info = avatar_registrar.register(args.image, max_dim=1920, n_frames=1)
        x_s_info = source_info['x_s_info_lst'][0]
        f_s = source_info['f_s_lst'][0]
        M_c2o = source_info['M_c2o_lst'][0]
        frame_rgb = source_info['img_rgb_lst'][0]

        # setup motion_stitch with the detected source info
        motion_stitch.setup(is_image_flag=source_info['is_image_flag'], x_s_info=x_s_info)

    # Ensure exp exists
    exp = kp_info.get('exp')
    if exp is None:
        raise RuntimeError('kp_info does not contain "exp"; cannot perturb')
    exp = np.array(exp)

    # call transform_keypoint to get original positions
    kp_trans_orig = transform_keypoint(kp_info)
    if kp_trans_orig.ndim == 3:
        kps_orig = kp_trans_orig[0]
    else:
        kps_orig = kp_trans_orig

    num_kp = kps_orig.shape[0]
    labels = list(range(1, num_kp + 1))

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'perturb_report.csv')

    rows = []

    axes_map = {'x': 0, 'y': 1, 'z': 2}
    axes_to_run = [0, 1, 2] if args.axis == 'all' else [axes_map[args.axis]]

    for kp_idx in range(num_kp):
        for ax in axes_to_run:
            # build delta_exp of shape matching flattened exp
            exp_flat = exp.reshape(1, -1).copy()
            idx = kp_idx * 3 + ax
            exp_flat[0, idx] = exp_flat[0, idx] + args.delta

            kp_info_pert = copy.deepcopy(kp_info)
            # put back into shape expected by transform_keypoint
            # transform_keypoint expects kp_info['exp'] shaped (bs, num_kp, 3) or flattened
            try:
                kp_info_pert['exp'] = exp_flat.reshape(1, num_kp, 3)
            except Exception:
                kp_info_pert['exp'] = exp_flat

            kp_trans_pert = transform_keypoint(kp_info_pert)
            if kp_trans_pert.ndim == 3:
                kps_pert = kp_trans_pert[0]
            else:
                kps_pert = kp_trans_pert

            fname = f'kp{kp_idx+1:02d}_axis{ax}_d{args.delta:.3f}.png'
            out_path = os.path.join(out_dir, fname)
            side_by_side_plot(args.image, kps_orig[:, :2], kps_pert[:, :2], labels, out_path,
                              title=f'kp={kp_idx+1}, axis={ax}, delta={args.delta}')

            desc = ('x' if ax == 0 else 'y' if ax == 1 else 'z')
            rows.append({'kp_index': kp_idx + 1, 'axis': desc, 'delta': args.delta, 'image': out_path})

            # Optional: produce rendered frame showing the perturbation
            if args.render:
                try:
                    # Build a base x_d_info. Prefer kp_info if it contains 'exp', else copy x_s_info
                    if isinstance(kp_info, dict) and kp_info.get('exp') is not None:
                        x_d_info_base = copy.deepcopy(kp_info)
                    else:
                        x_d_info_base = copy.deepcopy(x_s_info)

                    # delta_exp should match shape of x_d_info_base['exp']
                    try:
                        exp_shape = x_d_info_base['exp'].shape
                        delta_exp = np.zeros_like(x_d_info_base['exp'])
                        # idx corresponds to flattened index into (1, num_kp, 3)
                        # convert flattened idx to (b, kp, axis)
                        if exp_shape and len(exp_shape) >= 2:
                            # reshape idx to (1, num_kp, 3) assumption
                            kp_n = exp_shape[1] if len(exp_shape) > 1 else exp_shape[0]
                            # compute kp index and axis
                            kp_axis_idx = kp_idx * 3 + ax
                            kp_i = kp_axis_idx // 3
                            axis_i = kp_axis_idx % 3
                            # handle batch dim
                            if len(exp_shape) == 3:
                                delta_exp[0, kp_i, axis_i] = args.delta
                            elif len(exp_shape) == 2:
                                delta_exp[0, kp_i * 3 + axis_i] = args.delta
                            else:
                                # fallback: try flattened
                                delta_exp = np.zeros_like(x_d_info_base['exp']).reshape(1, -1)
                                delta_exp[0, kp_axis_idx] = args.delta
                        else:
                            # fallback: flattened
                            delta_exp = np.zeros_like(np.array(x_d_info_base['exp']).reshape(1, -1))
                            delta_exp[0, kp_idx * 3 + ax] = args.delta
                    except Exception:
                        delta_exp = None

                    # Call motion_stitch to compute x_s and x_d after applying delta
                    # Pass delta_exp as kwargs so ctrl_motion will add it
                    if delta_exp is not None:
                        x_s_res, x_d_res = motion_stitch(x_s_info, x_d_info_base, delta_exp=delta_exp)
                    else:
                        x_s_res, x_d_res = motion_stitch(x_s_info, x_d_info_base)

                    # Warp, decode, and put back to the original frame
                    f_3d = warp_f3d(f_s, x_s_res, x_d_res)
                    render_img = decode_f3d(f_3d)
                    # decode may return batch dim
                    if isinstance(render_img, np.ndarray) and render_img.ndim == 4 and render_img.shape[0] == 1:
                        render_img = render_img[0]

                    # Normalize render_img to uint8 (0..255) which PutBackNumpy handles robustly.
                    if isinstance(render_img, np.ndarray):
                        if np.issubdtype(render_img.dtype, np.floating):
                            mx = float(render_img.max())
                            # if values in [0,1], scale up
                            if mx <= 1.1:
                                render_img = render_img * 255.0
                            render_img = np.clip(render_img, 0, 255).astype(np.uint8)
                        else:
                            render_img = np.clip(render_img, 0, 255).astype(np.uint8)
                    else:
                        render_img = np.array(render_img, dtype=np.uint8)

                    # frame_rgb is from AvatarRegistrar and typically uint8; keep as-is
                    final_frame = putback(frame_rgb, render_img, M_c2o)

                    render_fname = f'kp{kp_idx+1:02d}_axis{ax}_d{args.delta:.3f}_render.png'
                    render_out = os.path.join(out_dir, render_fname)
                    # save with PIL
                    Image.fromarray(final_frame).save(render_out)
                    # add to csv row
                    rows[-1]['render_image'] = render_out
                except Exception as e:
                    # don't fail the whole script on render errors; just warn and continue
                    print('Warning: render failed for', fname, 'error:', e)

    # write CSV
    csv_fieldnames = ['kp_index', 'axis', 'delta', 'image']
    if args.render:
        csv_fieldnames.append('render_image')

    with open(csv_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print('Saved report:', csv_path)


if __name__ == '__main__':
    main()
