#!/usr/bin/env python3
"""
Extract `exp` from a source image by running the avatar registration step.

This script requires the pipeline cfg pkl and data_root so it can construct
the `AvatarRegistrar` with the proper configs. It registers the image (one
frame) and extracts the `x_s_info['exp']` tensor, saving it to an NPZ file.

Usage:
  python tools/extract_exp.py --image face.jpg --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl --data_root ./checkpoints/ditto_trt_Ampere_Plus --out out/face_exp.npz

Output NPZ contains key `exp` (the x_s_info['exp']), and also `x_s_info` saved as an object for convenience.
"""
import argparse
import os
import sys
import numpy as np

# ensure repo root on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to source image')
    parser.add_argument('--cfg_pkl', required=True, help='Path to pipeline cfg pkl')
    parser.add_argument('--data_root', required=True, help='Path to data_root containing models')
    parser.add_argument('--out', required=True, help='Output NPZ path to save exp')
    parser.add_argument('--max_dim', type=int, default=1920, help='Max dimension for registration crop')
    parser.add_argument('--n_frames', type=int, default=1, help='Number of frames to register (default 1)')
    args = parser.parse_args()

    try:
        from core.atomic_components.cfg import parse_cfg
        from core.atomic_components.avatar_registrar import AvatarRegistrar
    except Exception as e:
        raise RuntimeError('Failed to import AvatarRegistrar: ' + str(e))

    # parse cfg and get avatar registrar cfg
    cfgs = parse_cfg(args.cfg_pkl, args.data_root)
    avatar_registrar_cfg = cfgs[0]

    avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)

    # register image
    source_info = avatar_registrar.register(args.image, max_dim=args.max_dim, n_frames=args.n_frames)

    # prefer first frame x_s_info
    if 'x_s_info_lst' in source_info and len(source_info['x_s_info_lst']) > 0:
        x_s_info = source_info['x_s_info_lst'][0]
    else:
        raise RuntimeError('AvatarRegistrar did not return x_s_info_lst')

    if 'exp' not in x_s_info or x_s_info['exp'] is None:
        raise RuntimeError('Registered x_s_info does not contain exp')

    exp = np.array(x_s_info['exp'])

    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # Write exp to a plain text file, one value per line (vertical).
    # Include a header with the original shape so downstream tools can reshape.
    exp_flat = np.array(exp).reshape(-1)
    header = f"# exp shape: {np.array(exp).shape} | flattened length: {exp_flat.size}"
    # Save one value per line for easier inspection.
    np.savetxt(args.out, exp_flat, fmt='%.8e', header=header, comments='')

    # Also save x_s_info for debugging convenience (object array) next to the txt
    xs_out = os.path.splitext(args.out)[0] + '_x_s_info.npz'
    np.savez_compressed(xs_out, x_s_info=np.array(x_s_info, dtype=object))

    print('Saved exp (txt) to', args.out)
    print('Saved x_s_info (npz) to', xs_out)
    print('exp shape:', exp.shape)


if __name__ == '__main__':
    main()
