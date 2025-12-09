#!/usr/bin/env python3
"""
visualize_kp_on_image.py

Usage examples:
  # using a saved kp_info npz that contains x_s_info or kp_info and M_c2o
  python scripts/visualize_kp_on_image.py \
      --image ./example/image.png \
      --kp_npz ./outputs/kp_info_sample.npz \
      --out ./outputs/kp_on_image.png

This script does NOT depend on `tools/`. It expects a .npz that contains
either a dict-like `kp_info` or `x_s_info` saved as numpy arrays. It will:
  - load kp_info
  - call the project's `transform_keypoint` to get 3D kp in crop space
  - map kp xy to crop pixels (two mapping options supported)
  - apply `M_c2o` (if available in npz or provided) to get original image pixels
  - draw keypoints on the original image and save

The script is defensive: if model-specific artifacts are missing, it will
print helpful errors and guidance.
"""
import argparse
import os
import sys
import numpy as np
import cv2

# Ensure repo root is on sys.path so imports like `stream_pipeline_offline` work
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.atomic_components.motion_stitch import transform_keypoint
from core.atomic_components.cfg import parse_cfg
from core.atomic_components.avatar_registrar import AvatarRegistrar


def load_npz_as_dict(path):
    d = np.load(path, allow_pickle=True)
    # if saved using savez_compressed with an object, it may be under 'arr_0' or 'kp_info'
    out = {}
    for k in d.files:
        out[k] = d[k]
    return out


def extract_kp_info(npz_dict):
    # try common keys
    if 'x_s_info' in npz_dict:
        v = npz_dict['x_s_info']
        try:
            return v.item()
        except Exception:
            return v
    if 'kp_info' in npz_dict:
        v = npz_dict['kp_info']
        try:
            return v.item()
        except Exception:
            return v
    # fallback: try to assemble from flat arrays
    # look for arrays named kp, exp, pitch, yaw, roll, scale, t
    keys = set(npz_dict.keys())
    need = {'kp', 'exp', 'pitch', 'yaw', 'roll', 'scale', 't'}
    if need.issubset(keys):
        return {k: npz_dict[k] for k in need}
    raise ValueError('Cannot find kp_info or x_s_info in npz. Available keys: %s' % list(npz_dict.keys()))


def map_to_crop_pixels(pts_xy, mode='model', model_dsize=256, crop_dsize=512, center_scale=None):
    """
    pts_xy: (N,2) coordinates from transform_keypoint (assumed normalized with center at 0)
    mode: 'model' uses (kp + 0.5) * MODEL_DSIZE -> scaled to crop
          'center' uses center + kp * center_scale
    """
    pts = pts_xy.copy()
    if mode == 'model':
        pts_model_px = (pts + 0.5) * model_dsize
        pts_crop_px = pts_model_px * (crop_dsize / float(model_dsize))
        return pts_crop_px
    elif mode == 'center':
        if center_scale is None:
            raise ValueError('center_scale must be provided for center mode')
        center = np.array([crop_dsize / 2.0, crop_dsize / 2.0], dtype=np.float32)
        scale_xy = np.array(center_scale, dtype=np.float32)
        return center + pts * scale_xy
    else:
        raise ValueError('unknown mode')


def project_points_to_original(pts_crop_px, M_c2o):
    """Apply affine 3x3 M_c2o to crop pixel coordinates (Nx2) -> original image pixels (Nx2)"""
    M = np.asarray(M_c2o)
    A = M[:2, :2]
    b = M[:2, 2]
    pts = np.asarray(pts_crop_px)
    return pts @ A.T + b


def draw_keypoints_on_image(img_path, pts_orig_px, out_path, radius=3, color=(0, 255, 0)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    for i, (x, y) in enumerate(pts_orig_px):
        if np.isnan(x) or np.isnan(y):
            continue
        pt = (int(round(x)), int(round(y)))
        cv2.circle(img, pt, radius, color, -1)
        cv2.putText(img, str(i+1), (pt[0]+4, pt[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, help='original image path')
    p.add_argument('--kp_npz', required=False, help='(optional) npz file that contains kp_info or x_s_info and optionally M_c2o')
    p.add_argument('--cfg_pkl', required=False, help='(optional) pipeline cfg pkl for end-to-end extraction')
    p.add_argument('--data_root', required=False, help='(optional) data_root to resolve model paths when using cfg_pkl')
    p.add_argument('--out', required=True, help='output image path')
    p.add_argument('--map_mode', choices=['model', 'center'], default='model', help='mapping mode from kp->pixels')
    p.add_argument('--center_scale_x', type=float, default=300.0)
    p.add_argument('--center_scale_y', type=float, default=300.0)
    p.add_argument('--model_dsize', type=int, default=256)
    p.add_argument('--crop_dsize', type=int, default=512)
    p.add_argument('--M_c2o_npy', type=str, default=None, help='optional: .npy file containing M_c2o (3x3) overrides npz')
    args = p.parse_args()

    ki = None
    M_c2o = None

    # Priority: if user provided a kp_npz, use it; else if cfg_pkl+data_root provided, run end-to-end extraction
    if args.kp_npz:
        npz_dict = load_npz_as_dict(args.kp_npz)
        try:
            kp_info = extract_kp_info(npz_dict)
        except Exception as e:
            print('Error loading kp_info from npz:', e)
            return

        # format handling
        if isinstance(kp_info, dict) and 'kp' in kp_info:
            ki = kp_info
        else:
            if all(k in npz_dict for k in ('kp','exp','pitch','yaw','roll','scale','t')):
                ki = {k: npz_dict[k] for k in ('kp','exp','pitch','yaw','roll','scale','t')}
            else:
                print('kp_info format not recognized in npz. Expect dict-like with keys kp,exp,pitch,yaw,roll,scale,t')
                return

        # extract M_c2o if present
        if 'M_c2o' in npz_dict:
            M_c2o = npz_dict['M_c2o']
        elif 'M_o2c' in npz_dict:
            M_c2o = np.linalg.inv(npz_dict['M_o2c'])

    elif args.cfg_pkl and args.data_root:
        # End-to-end: construct AvatarRegistrar using parse_cfg and extract kp_info from the input image
        print('Running end-to-end extraction using cfg_pkl and data_root. This may load models and take time...')
        cfgs = parse_cfg(args.cfg_pkl, args.data_root)
        avatar_registrar_cfg = cfgs[0]
        # avatar_registrar_cfg contains the configs for Source2Info
        registrar = AvatarRegistrar(
            avatar_registrar_cfg['insightface_det_cfg'],
            avatar_registrar_cfg['landmark106_cfg'],
            avatar_registrar_cfg['landmark203_cfg'],
            avatar_registrar_cfg['landmark478_cfg'],
            avatar_registrar_cfg['appearance_extractor_cfg'],
            avatar_registrar_cfg['motion_extractor_cfg'],
        )

        # register returns source_info with lists; use first frame
        source_info = registrar.register(args.image)
        # pick first frame info
        ki = source_info['x_s_info_lst'][0]
        M_c2o = source_info['M_c2o_lst'][0]
    else:
        print('Either --kp_npz or (--cfg_pkl and --data_root) must be provided for input extraction')
        return

    # compute transformed 3D kps (crop-space normalized coords)
    try:
        kp3d = transform_keypoint(ki)
    except Exception as e:
        print('transform_keypoint failed:', e)
        return

    kp3d = np.asarray(kp3d)
    if kp3d.ndim == 3 and kp3d.shape[0] == 1:
        kp3d = kp3d[0]

    pts_crop_xy = kp3d[:, :2]

    # mapping to crop pixels
    if args.map_mode == 'center':
        center_scale = (args.center_scale_x, args.center_scale_y)
        pts_crop_px = map_to_crop_pixels(pts_crop_xy, mode='center', crop_dsize=args.crop_dsize, center_scale=center_scale)
    else:
        pts_crop_px = map_to_crop_pixels(pts_crop_xy, mode='model', model_dsize=args.model_dsize, crop_dsize=args.crop_dsize)

    # Allow override of M_c2o via npy. Otherwise, M_c2o should have been set
    # earlier when loading from npz or from the end-to-end registrar.
    if args.M_c2o_npy is not None:
        M_c2o = np.load(args.M_c2o_npy)

    if M_c2o is None:
        print('Warning: M_c2o not found. Cannot project to original image.')
        print('Ensure kp_npz contains M_c2o, or provide --cfg_pkl/--data_root for end-to-end extraction, or pass --M_c2o_npy')
        return

    pts_orig = project_points_to_original(pts_crop_px, M_c2o)

    draw_keypoints_on_image(args.image, pts_orig, args.out)
    print('Saved visualization to', args.out)


if __name__ == '__main__':
    main()
