#!/usr/bin/env python3
"""
visualize_kp.py

提取单张源图像的关键点参数，计算 3D 关键点并将它们投影回原图进行可视化。

用法示例：
  python tools/visualize_kp.py --image ./example/image.png \
    --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \
    --data_root ./checkpoints/ditto_trt_Ampere_Plus

输出：
  - image_overlay.png: 原图上叠加的 2D 关键点
  - kp3d_scatter.png: 关键点的 3D 散点图（以 crop 空间坐标为基准）
"""
import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so imports like `stream_pipeline_offline` work
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stream_pipeline_offline import StreamSDK
from core.atomic_components.motion_stitch import transform_keypoint


def project_points_to_original(pts_crop_xy, M_c2o):
    """把 crop 空间的 2D 点投影回原始图像坐标。
    pts_crop_xy: (N,2)
    M_c2o: 3x3 矩阵（crop->original）
    返回 (N,2)
    """
    pts = np.asarray(pts_crop_xy, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[None, :]
    # 使用仿射变换: pts @ A.T + b
    A = M_c2o[:2, :2]
    b = M_c2o[:2, 2]
    pts_orig = pts @ A.T + b
    return pts_orig


def draw_overlay(image_bgr, pts2d, out_path):
    img = image_bgr.copy()
    for i, p in enumerate(pts2d):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(img, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(out_path, img)


def plot_3d(kp3d, out_path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    x = kp3d[:, 0]
    y = kp3d[:, 1]
    z = kp3d[:, 2]
    ax.scatter(x, y, z, c='r', s=30)
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], str(i), color='black', fontsize=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=-60)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='path to input image')
    parser.add_argument('--cfg_pkl', type=str, default='./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl')
    parser.add_argument('--data_root', type=str, default='./checkpoints/ditto_trt_Ampere_Plus')
    parser.add_argument('--frame', type=int, default=0, help='frame index in case of video source')
    parser.add_argument('--out_dir', type=str, default='./tools/visualize_out')
    parser.add_argument('--use_center_origin', action='store_true', help='use image-center-as-origin mapping for coords (alternative mapping)')
    parser.add_argument('--center_scale_x', type=float, default=300.0, help='scale factor for x when using center-origin mapping')
    parser.add_argument('--center_scale_y', type=float, default=300.0, help='scale factor for y when using center-origin mapping')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 初始化 StreamSDK（会加载模型/权重，根据 cfg_pkl）
    SDK = StreamSDK(args.cfg_pkl, args.data_root)

    # 使用 AvatarRegistrar 注册 source（复用 SDK 的流程以保证一致性）
    source_info = SDK.avatar_registrar.register(args.image, max_dim=SDK.max_size if hasattr(SDK, 'max_size') else 1920, n_frames=1)

    # 读取原始 crop 信息
    frame_idx = args.frame
    x_s_info = source_info['x_s_info_lst'][frame_idx]
    M_c2o = source_info['M_c2o_lst'][frame_idx]
    img_rgb = source_info['img_rgb_lst'][frame_idx]

    # transform_keypoint 返回 (bs, num_kp, 3) 或 (num_kp,3)
    kp3d = transform_keypoint(x_s_info)
    kp3d = np.asarray(kp3d)
    if kp3d.ndim == 3 and kp3d.shape[0] == 1:
        kp3d = kp3d[0]

    # crop 空间的 2D（x,y）
    pts_crop_xy = kp3d[:, :2]

    # NOTE:
    # motion_extractor 的输入是被 resize 到 256x256（见 _img_crop_to_bchw256），
    # 模型输出的 kp/exp/t 等是基于该规范化坐标系（中心可能在 0，数值范围通常在 -0.5..0.5）。
    # 因此要把模型输出映射为 crop 图像像素坐标，需要做：
    #   pts_model_px = (pts_crop_xy + 0.5) * MODEL_DSIZE
    # 如果 pipeline 的 crop 原始尺寸为 512（Source2Info 使用的 dsize），
    # 则需要把 model 尺寸放缩到 crop 尺寸再用 M_c2o 投影：
    MODEL_DSIZE = 256
    CROP_DSIZE = 512

    # ----- 默认映射（保持脚本原有逻辑） -----
    pts_model_px_default = (pts_crop_xy + 0.5) * MODEL_DSIZE
    pts_crop_px_default = pts_model_px_default * (CROP_DSIZE / MODEL_DSIZE)

    # ----- 可选映射：以图像中心为原点（按你的想法直接使用带符号的 coords） -----
    # 如果 kp_xy 的数值已经在以 0 为中心的归一化坐标系（通常范围约 [-0.5,0.5]），
    # 则在中心为 (CROP_DSIZE/2) 的像素空间中：
    #   px = center + kp * CROP_DSIZE
    # 等价变形（center = CROP_DSIZE/2）： px = (kp + 0.5) * CROP_DSIZE
    # 这里实现直接按你描述的“以中心为原点”的形式，便于对比
    center = CROP_DSIZE / 2.0
    # 支持对 x,y 分别设置缩放因子（以图像中心为原点）
    scale_xy = np.array([args.center_scale_x, args.center_scale_y], dtype=np.float32)
    center_arr = np.array([center, center], dtype=np.float32)
    pts_crop_px_center = center_arr + pts_crop_xy * scale_xy

    # 选择最终使用哪种映射用于后续可视化（默认继续使用原有逻辑）
    if args.use_center_origin:
        pts_crop_px = pts_crop_px_center
        pts_model_px = (pts_crop_px / (CROP_DSIZE / MODEL_DSIZE))
    else:
        pts_crop_px = pts_crop_px_default
        pts_model_px = pts_model_px_default

    # 保存两个版本以便对比（npz）

    # 投影回原图（将 crop 像素坐标映射到原图像素坐标）
    pts_orig = project_points_to_original(pts_crop_px, M_c2o)

    # debug: 打印与保存关键点数值范围
    print('kp3d min/max:', np.nanmin(kp3d, axis=0), np.nanmax(kp3d, axis=0))
    print('pts_orig min/max:', np.nanmin(pts_orig, axis=0), np.nanmax(pts_orig, axis=0))
    np.savez_compressed(
        os.path.join(args.out_dir, 'kp_debug.npz'),
        kp3d=kp3d,
        pts_model_px_default=pts_model_px_default,
        pts_crop_px_default=pts_crop_px_default,
        pts_crop_px_center=pts_crop_px_center,
        pts_model_px=pts_model_px,
        pts_crop_px=pts_crop_px,
        pts_orig=pts_orig,
    )

    # 保存 overlay（只画在原图内的点；超出边界的点会标为红色）
    image_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    pts_clamped = []
    for p in pts_orig:
        x, y = p[0], p[1]
        if not (np.isfinite(x) and np.isfinite(y)):
            pts_clamped.append((None, None))
            continue
        inside = 0 <= x < w and 0 <= y < h
        if inside:
            pts_clamped.append((int(round(x)), int(round(y))))
        else:
            # clamp to image for visualization
            xc = int(min(max(round(x), 0), w - 1))
            yc = int(min(max(round(y), 0), h - 1))
            pts_clamped.append((xc, yc))

    overlay_path = os.path.join(args.out_dir, 'image_overlay.png')
    # draw: inside points green, out-of-bounds red, nan gray
    img = image_bgr.copy()
    for i, p in enumerate(pts_clamped):
        px, py = p
        if px is None:
            # invalid
            continue
        orig = pts_orig[i]
        x_f, y_f = orig[0], orig[1]
        inside = (0 <= x_f < w) and (0 <= y_f < h)
        color = (0, 255, 0) if inside else (0, 0, 255)
        cv2.circle(img, (px, py), 4, color, -1)
        cv2.putText(img, str(i), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(overlay_path, img)

    # 保存 3D 散点（以 crop 坐标为基准）
    kp3d_path = os.path.join(args.out_dir, 'kp3d_scatter.png')
    plot_3d(kp3d, kp3d_path)

    print('Saved overlay:', overlay_path)
    print('Saved 3D scatter:', kp3d_path)

    # 另外：重建 crop 图像并在 crop 上可视化（crop->original 的逆矩阵 M_o2c）
    try:
        M_c2o_arr = np.asarray(M_c2o)
        M_o2c = np.linalg.inv(M_c2o_arr)
        # pipeline 的 crop 原始尺寸通常为 512（Source2Info 中的 dsize），
        # 所以在重建 crop 图像时应使用该尺寸
        CROP_DSIZE = 512
        img_crop = cv2.warpAffine(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), M_o2c[:2, :], dsize=(CROP_DSIZE, CROP_DSIZE))
        # 我们已经把 kp 转换为 crop (px) 坐标 pts_crop_px（对应 CROP_DSIZE），直接用于绘制
        kp_px = []
        h_c, w_c = CROP_DSIZE, CROP_DSIZE
        for p in pts_crop_px:
            if not (np.isfinite(p[0]) and np.isfinite(p[1])):
                kp_px.append((None, None))
                continue
            x = int(round(p[0]))
            y = int(round(p[1]))
            x = min(max(x, 0), w_c - 1)
            y = min(max(y, 0), h_c - 1)
            kp_px.append((x, y))

        crop_overlay = img_crop.copy()
        for i, (x, y) in enumerate(kp_px):
            if x is None:
                continue
            cv2.circle(crop_overlay, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(crop_overlay, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        crop_overlay_path = os.path.join(args.out_dir, 'crop_overlay.png')
        cv2.imwrite(crop_overlay_path, crop_overlay)
        print('Saved crop overlay:', crop_overlay_path)
    except Exception as e:
        print('Failed to produce crop overlay:', e)

    # --------------- 额外调试：直接使用 Source2Info._crop 获取并保存 crop 与 landmark203 ---------------
    try:
        src2info = SDK.avatar_registrar.source2info
        # 调用内部 _crop（它返回 img_crop, M_c2o, lmk203）
        img_crop2, M_c2o2, lmk203 = src2info._crop(img_rgb, last_lmk=None)
        if img_crop2 is not None:
            crop_path = os.path.join(args.out_dir, 'raw_crop.png')
            cv2.imwrite(crop_path, cv2.cvtColor(img_crop2, cv2.COLOR_RGB2BGR))
            print('Saved raw crop from Source2Info._crop:', crop_path)

            # lmk203 为 (N,2)，在 crop 空间内，绘制以验证定位
            try:
                lmk = lmk203
                lmk_img = cv2.cvtColor(img_crop2, cv2.COLOR_RGB2BGR).copy()
                for i, p in enumerate(lmk):
                    x, y = int(round(p[0])), int(round(p[1]))
                    cv2.circle(lmk_img, (x, y), 2, (255, 0, 0), -1)
                lmk_path = os.path.join(args.out_dir, 'lmk203_overlay.png')
                cv2.imwrite(lmk_path, lmk_img)
                print('Saved lmk203 overlay:', lmk_path)
            except Exception as e:
                print('Failed to draw lmk203:', e)

        print('M_c2o (from register):\n', M_c2o)
        print('M_c2o (from _crop):\n', M_c2o2)
    except Exception as e:
        print('Failed to run Source2Info._crop debug:', e)


if __name__ == '__main__':
    main()
