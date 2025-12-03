#!/usr/bin/env python3
# debug_coordinates.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

def explore_coordinates(npz_path):
    """探索关键点的坐标系统"""
    data = np.load(npz_path, allow_pickle=True)
    kp_info = data['kp_info'].item()
    
    x_s_info = kp_info['x_s_info']
    
    print("=== 坐标探索 ===")
    print(f"kp shape: {x_s_info['kp'].shape}")
    print(f"kp dtype: {x_s_info['kp'].dtype}")
    
    # 原始扁平化关键点
    kp_flat = x_s_info['kp'][0]  # shape: (63,)
    print(f"\n扁平化关键点前10个值: {kp_flat[:10]}")
    
    # 重塑为21x3
    kp_3d = kp_flat.reshape(21, 3)
    print(f"\n重塑后形状: {kp_3d.shape}")
    
    # 分析坐标范围
    print("\n坐标统计分析:")
    print(f"X (列0)范围: [{kp_3d[:, 0].min():.4f}, {kp_3d[:, 0].max():.4f}]")
    print(f"Y (列1)范围: [{kp_3d[:, 1].min():.4f}, {kp_3d[:, 1].max():.4f}]")
    print(f"Z (列2)范围: [{kp_3d[:, 2].min():.4f}, {kp_3d[:, 2].max():.4f}]")
    
    print(f"X均值: {kp_3d[:, 0].mean():.4f}, 标准差: {kp_3d[:, 0].std():.4f}")
    print(f"Y均值: {kp_3d[:, 1].mean():.4f}, 标准差: {kp_3d[:, 1].std():.4f}")
    print(f"Z均值: {kp_3d[:, 2].mean():.4f}, 标准差: {kp_3d[:, 2].std():.4f}")
    
    # 查看具体的关键点
    print("\n关键点示例 (前5个):")
    for i in range(5):
        print(f"  {i}: X={kp_3d[i, 0]:.4f}, Y={kp_3d[i, 1]:.4f}, Z={kp_3d[i, 2]:.4f}")
    
    # 检查是否是常见的3D人脸关键点范围
    # 典型的3DDFA关键点：X/Y在±100左右，Z在±50左右
    # 或者可能是相机空间坐标
    
    return kp_3d

def test_different_projections(kp_3d, image_path):
    """测试不同的投影方法"""
    img = Image.open(image_path)
    img_width, img_height = img.size
    print(f"\n=== 图像信息 ===")
    print(f"图像尺寸: {img_width}x{img_height}")
    
    # 方法1: 直接使用XY作为像素坐标（可能已经缩放）
    print("\n方法1: 直接使用XY坐标")
    kp_xy = kp_3d[:, :2]
    print(f"XY坐标范围: X[{kp_xy[:, 0].min():.1f}, {kp_xy[:, 0].max():.1f}], "
          f"Y[{kp_xy[:, 1].min():.1f}, {kp_xy[:, 1].max():.1f}]")
    
    # 方法2: 假设坐标在[-1, 1]范围，转换到[0, 1]
    print("\n方法2: 假设[-1,1]归一化")
    kp_normalized = (kp_xy + 1) / 2
    print(f"归一化后: X[{kp_normalized[:, 0].min():.3f}, {kp_normalized[:, 0].max():.3f}], "
          f"Y[{kp_normalized[:, 1].min():.3f}, {kp_normalized[:, 1].max():.3f}]")
    kp_pixel = kp_normalized * np.array([[img_width, img_height]])
    
    # 方法3: 使用正交投影（忽略Z）
    print("\n方法3: 直接作为像素坐标（假设已经是正确范围）")
    
    # 绘制测试图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 方法1
    ax = axes[0]
    ax.imshow(img)
    ax.scatter(kp_xy[:, 0], kp_xy[:, 1], c='red', s=30)
    for i, (x, y) in enumerate(kp_xy):
        ax.text(x, y, str(i), color='yellow', fontsize=8)
    ax.set_title("方法1: 直接使用XY")
    
    # 方法2
    ax = axes[1]
    ax.imshow(img)
    ax.scatter(kp_pixel[:, 0], kp_pixel[:, 1], c='green', s=30)
    for i, (x, y) in enumerate(kp_pixel):
        ax.text(x, y, str(i), color='white', fontsize=8)
    ax.set_title("方法2: [-1,1]归一化")
    
    # 方法3: 尝试找到合适的缩放
    print("\n方法3: 自动寻找合适缩放")
    
    # 查看是否有t（平移）和scale（缩放）参数
    # 这些通常用于将3D模型坐标转换到图像坐标
    
    # 绘制3D视图
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(kp_3d[:, 0], kp_3d[:, 1], kp_3d[:, 2])
    for i, (x, y, z) in enumerate(kp_3d):
        ax3d.text(x, y, z, str(i), fontsize=8)
    ax3d.set_title("3D关键点视图")
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    npz_path = sys.argv[1] if len(sys.argv) > 1 else "./outputs/kp_npz/frame_000000_kpinfo.npz"
    image_path = sys.argv[2] if len(sys.argv) > 2 else "./example/image.png"
    
    kp_3d = explore_coordinates(npz_path)
    test_different_projections(kp_3d, image_path)