#!/usr/bin/env python3
"""
Project 3D keypoints to 2D image for Ditto
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

def load_kp_info(npz_path):
    """加载kp_info.npz文件"""
    data = np.load(npz_path, allow_pickle=True)
    kp_info = data['kp_info'].item()
    return kp_info['x_s_info']

def calculate_3d_keypoints(kp_info):
    """计算3D关键点位置，基于 LivePortrait 的变换：s * (R * kp + exp) + t

    兼容多种输入形状：flat、(N,3)、(bs,N,3)。如果 exp 是 63-D 表情向量，尝试 reshape 为 (21,3)。
    返回: kp_3d (N,3) 以及 pose angles (pitch,yaw,roll)（以度为单位）
    """
    def _to_batch_kp(a):
        a = np.asarray(a)
        if a.ndim == 1:
            return a.reshape(1, a.size // 3, 3)
        if a.ndim == 2:
            if a.shape[1] == 3:
                return a[np.newaxis, ...]
            if a.shape[1] % 3 == 0:
                return a.reshape(a.shape[0], a.shape[1] // 3, 3)
            return a.reshape(1, a.size // 3, 3)
        return a

    def _rotation_matrices_from_degrees(pitch, yaw, roll):
        p = np.radians(np.asarray(pitch).reshape(-1))
        y = np.radians(np.asarray(yaw).reshape(-1))
        r = np.radians(np.asarray(roll).reshape(-1))
        bs = max(len(p), 1)
        mats = np.zeros((bs, 3, 3), dtype=float)
        for i in range(bs):
            cp, sp = np.cos(p[i]), np.sin(p[i])
            cy, sy = np.cos(y[i]), np.sin(y[i])
            cr, sr = np.cos(r[i]), np.sin(r[i])
            Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
            mats[i] = Rz @ Ry @ Rx
        return mats

    kp = kp_info['kp']
    exp = kp_info['exp']

    kp_b = _to_batch_kp(kp)
    try:
        exp_b = _to_batch_kp(exp)
    except Exception:
        exp_arr = np.asarray(exp)
        if exp_arr.size == 63:
            exp_b = exp_arr.reshape(1, 21, 3)
        else:
            exp_b = np.zeros_like(kp_b)

    # broadcast batch if needed
    if exp_b.shape[0] != kp_b.shape[0]:
        exp_b = np.repeat(exp_b, kp_b.shape[0], axis=0)

    # adjust keypoint count
    if exp_b.shape[1] != kp_b.shape[1]:
        n = min(exp_b.shape[1], kp_b.shape[1])
        exp_b = exp_b[:, :n, :]
        kp_b = kp_b[:, :n, :]

    pitch = kp_info.get('pitch', None)
    yaw = kp_info.get('yaw', None)
    roll = kp_info.get('roll', None)
    t = kp_info.get('t', None)
    scale = kp_info.get('scale', None)

    # compute pose angles for return
    def get_angle_from_distribution(dist):
        bins = np.linspace(-99, 99, 66)
        max_bin = np.argmax(dist)
        return bins[max_bin]

    if hasattr(pitch, 'flatten') and hasattr(yaw, 'flatten') and hasattr(roll, 'flatten'):
        try:
            pitch_angle = get_angle_from_distribution(np.asarray(pitch).flatten())
            yaw_angle = get_angle_from_distribution(np.asarray(yaw).flatten())
            roll_angle = get_angle_from_distribution(np.asarray(roll).flatten())
        except Exception:
            pitch_angle = yaw_angle = roll_angle = 0.0
    else:
        pitch_angle = yaw_angle = roll_angle = 0.0

    rot_mats = _rotation_matrices_from_degrees(pitch_angle, yaw_angle, roll_angle)

    kp_rot = np.einsum('bnc,bcd->bnd', kp_b, rot_mats)
    kp_transformed = kp_rot + exp_b

    if scale is not None:
        try:
            s = float(np.asarray(scale).reshape(-1)[0])
            kp_transformed = kp_transformed * s
        except Exception:
            pass

    if t is not None:
        t_arr = np.asarray(t)
        if t_arr.size >= 2:
            kp_transformed[:, :, 0:2] += t_arr.reshape(1, -1)[:, None, 0:2]

    # return first batch if single
    if kp_transformed.shape[0] == 1:
        return kp_transformed[0], (pitch_angle, yaw_angle, roll_angle)
    return kp_transformed, (pitch_angle, yaw_angle, roll_angle)

def project_3d_to_2d(kp_3d, image_shape, pose_angles=None):
    """
    将3D关键点投影到2D图像平面
    """
    h, w = image_shape[:2]

    # 先把 kp_3d 转为 numpy array 确保可操作
    pts = np.asarray(kp_3d)
    # 打印3D边界信息
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    print(f"3D bounds: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}], Z[{z_min:.3f}]")

    # 如果提供姿态角则应用旋转（向量化），否则直接使用原坐标
    if pose_angles is None:
        x_final = pts[:, 0].copy()
        y_final = pts[:, 1].copy()
    else:
        pitch, yaw, roll = pose_angles
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        roll_rad = np.radians(roll)

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        # yaw around Y axis (左右)
        x_rot = x * np.cos(yaw_rad) - z * np.sin(yaw_rad)
        z_rot = x * np.sin(yaw_rad) + z * np.cos(yaw_rad)

        # pitch around X axis (上下)
        y_rot = y * np.cos(pitch_rad) - z_rot * np.sin(pitch_rad)
        # z_final = y * np.sin(pitch_rad) + z_rot * np.cos(pitch_rad)

        # apply roll as in-plane rotation
        x_final = x_rot * np.cos(roll_rad) - y_rot * np.sin(roll_rad)
        y_final = x_rot * np.sin(roll_rad) + y_rot * np.cos(roll_rad)

    # 中心化并按范围自适应缩放，使关键点群居中且落在面部区域
    cx = x_final.mean()
    cy = y_final.mean()

    rx = x_final.max() - x_final.min()
    ry = y_final.max() - y_final.min()
    # 防止除零
    rx = max(rx, 1e-6)
    ry = max(ry, 1e-6)

    # 使用图像较小边的比例来决定缩放，保证不会超出图像
    scale = min(w, h) * 0.4 / max(rx, ry)

    screen_x = (w / 2.0) + (x_final - cx) * scale
    screen_y = (h / 2.0) + (y_final - cy) * scale  # 图像 y 向下为正，所以减号将 3D 向上映射到图像向上？

    kp_2d = np.stack([screen_x, screen_y], axis=1)
    return kp_2d

def visualize_on_image(image_path, kp_2d, save_path, title="2D Keypoints Projection"):
    """在图像上可视化2D关键点"""
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 子图1：带关键点的完整图像
    ax1.imshow(image_rgb)
    
    # 绘制关键点
    colors = plt.cm.tab20(np.arange(len(kp_2d)) / len(kp_2d))
    
    for i, (x, y) in enumerate(kp_2d):
        # 确保坐标在图像范围内
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        
        # 绘制点
        ax1.scatter(x, y, c=[colors[i]], s=300, alpha=0.8, 
                   edgecolors='white', linewidth=2, zorder=5)
        
        # 标记序号
        ax1.text(x + 15, y + 15, f'{i+1}', 
                color='yellow', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                zorder=6)
    
    ax1.set_title(f'{title}\n(All Keypoints)', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # 子图2：面部区域特写
    ax2.imshow(image_rgb)
    
    # 只绘制面部的关键区域
    face_keypoints = []
    face_indices = []
    
    # 根据区域分组绘制
    regions = {
        'Eyes': [11, 12, 13, 14, 15, 16, 17, 18, 19],  # KP12-KP20
        'Mouth': [19, 20, 21],  # KP20-KP21
        'Nose': [7, 8, 9, 10, 11],  # KP8-KP12
        'Brows': [0, 1, 2, 3, 4, 5]  # KP1-KP6
    }
    
    region_colors = {
        'Eyes': 'red',
        'Mouth': 'blue', 
        'Nose': 'green',
        'Brows': 'purple'
    }
    
    for region_name, indices in regions.items():
        for idx in indices:
            if idx < len(kp_2d):
                x, y = kp_2d[idx]
                x = max(0, min(x, w-1))
                y = max(0, min(y, h-1))
                
                ax2.scatter(x, y, c=region_colors[region_name], s=200, 
                          alpha=0.7, edgecolors='white', linewidth=2, 
                          label=region_name if idx == indices[0] else "")
                
                ax2.text(x + 10, y + 10, f'KP{idx+1}', 
                        color='white', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=region_colors[region_name], alpha=0.7))
    
    # 添加图例
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax2.set_title('Facial Regions (Colored by Function)', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 2D visualization saved to: {save_path}")
    
    return kp_2d

def create_keypoint_mapping_image(image_path, kp_2d, save_dir):
    """创建关键点映射图（带详细标注）"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # 创建大图
    fig = plt.figure(figsize=(24, 12))
    
    # 左侧：完整图像带所有关键点
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image_rgb)
    
    # 右侧：分区域详细图
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(image_rgb)
    
    # 定义区域和描述
    regions = {
        'Brow/Forehead': {
            'indices': [0, 1, 2, 3, 4, 5],  # KP1-KP6
            'color': 'purple',
            'description': 'Controls forehead wrinkles and brow movements'
        },
        'Nose/Cheek': {
            'indices': [6, 7, 8, 9, 10],  # KP7-KP11
            'color': 'green',
            'description': 'Controls nose shape and cheek movements'
        },
        'Right Eye': {
            'indices': [11, 12, 13, 14, 15, 16],  # KP12-KP17
            'color': 'red',
            'description': 'Right eye opening/closing (dim34=KP12-x controls eye open)'
        },
        'Left Eye': {
            'indices': [17, 18],  # KP18-KP19
            'color': 'orange',
            'description': 'Left eye movements'
        },
        'Mouth/Jaw': {
            'indices': [19, 20],  # KP20-KP21
            'color': 'blue',
            'description': 'Mouth opening (dim58=KP20-z controls jaw open)'
        }
    }
    
    # 在左侧图上绘制所有关键点（按区域颜色）
    for region_name, region_info in regions.items():
        color = region_info['color']
        
        for idx in region_info['indices']:
            if idx < len(kp_2d):
                x, y = kp_2d[idx]
                x = max(0, min(x, w-1))
                y = max(0, min(y, h-1))
                
                # 绘制点
                ax1.scatter(x, y, c=color, s=250, alpha=0.7, 
                          edgecolors='white', linewidth=2, zorder=5)
                
                # 标记序号
                ax1.text(x + 20, y + 20, f'{idx+1}', 
                        color='white', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                        zorder=6)
    
    ax1.set_title('Complete Facial Keypoints (Color-coded by Region)', 
                 fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # 在右侧图上绘制区域边界和描述
    ax2.imshow(image_rgb)
    
    # 为每个区域绘制边界框和描述
    y_text_offset = 0.05
    for region_name, region_info in regions.items():
        color = region_info['color']
        
        # 收集该区域所有点
        points = []
        for idx in region_info['indices']:
            if idx < len(kp_2d):
                x, y = kp_2d[idx]
                points.append([x, y])
        
        if points:
            points = np.array(points)
            
            # 计算边界框
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            # 绘制边界框
            rect = plt.Rectangle((x_min-10, y_min-10), 
                                x_max-x_min+20, y_max-y_min+20,
                                linewidth=3, edgecolor=color, 
                                facecolor='none', alpha=0.7)
            ax2.add_patch(rect)
            
            # 添加区域标签
            ax2.text(x_min, y_min-30, region_name, 
                    color=color, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 添加关键点
            for idx in region_info['indices']:
                if idx < len(kp_2d):
                    x, y = kp_2d[idx]
                    ax2.scatter(x, y, c=color, s=150, alpha=0.8, 
                              edgecolors='white', linewidth=1.5)
                    ax2.text(x+8, y+8, f'{idx+1}', 
                            color='white', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
    
    ax2.set_title('Facial Regions with Functional Descriptions', 
                 fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # 添加图例
    from matplotlib.patches import Patch
    
    legend_elements = []
    for region_name, region_info in regions.items():
        legend_elements.append(
            Patch(facecolor=region_info['color'], alpha=0.7, 
                  label=f"{region_name}: {region_info['description']}")
        )
    
    plt.figlegend(handles=legend_elements, loc='lower center', 
                 ncol=2, fontsize=10, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    save_path = os.path.join(save_dir, 'keypoint_mapping_detailed.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Detailed mapping saved to: {save_path}")

def create_alignment_grid(image_path, kp_2d, save_dir):
    """创建对齐网格，显示3D关键点如何对应到2D面部"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建4x3的网格
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()
    
    # 不同的可视化方式
    visualizations = [
        ("All Keypoints with Numbers", 'tab20', 21, False),
        ("Eyes Region (KP12-KP19)", 'Reds', 8, [11, 12, 13, 14, 15, 16, 17, 18]),
        ("Mouth/Jaw (KP20-KP21)", 'Blues', 2, [19, 20]),
        ("Brows (KP1-KP6)", 'Purples', 6, list(range(6))),
        ("Nose/Cheek (KP7-KP11)", 'Greens', 5, list(range(6, 11))),
        ("Paper Special Points", 'Set1', 2, [11, 19]),  # KP12和KP20
        ("X-Coordinate Heatmap", 'hot', 21, False),
        ("Y-Coordinate Heatmap", 'cool', 21, False),
        ("Distance from Center", 'viridis', 21, False),
        ("Connected Facial Features", 'tab20', 21, True),
        ("Keypoint Size by Z-depth", 'plasma', 21, False),
        ("Region Connectivity", 'Set2', 21, True)
    ]
    
    for idx, (title, cmap, num_points, specific_indices) in enumerate(visualizations):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        ax.imshow(image_rgb)
        
        # specific_indices can be False (use all), True (use all), a list of indices, or False
        if specific_indices is False or specific_indices is True:
            # 使用所有点
            indices = range(len(kp_2d))
        else:
            indices = specific_indices
        
        # 计算颜色
        if cmap in ['hot', 'cool', 'viridis', 'plasma']:
            if cmap == 'hot':
                values = kp_2d[indices, 0]  # X坐标
            elif cmap == 'cool':
                values = kp_2d[indices, 1]  # Y坐标
            elif cmap == 'viridis':
                # 距离中心的距离
                center = np.array([image.shape[1]/2, image.shape[0]/2])
                distances = np.linalg.norm(kp_2d[indices] - center, axis=1)
                values = distances
            elif cmap == 'plasma':
                # 使用索引作为值
                values = list(range(len(indices)))
            
            values = np.asarray(values)
            vmin, vmax = float(values.min()), float(values.max())
            if vmin == vmax:
                # avoid zero-range Normalize
                vmin -= 1e-6
                vmax += 1e-6
            norm = plt.Normalize(vmin, vmax)
            cmap_func = plt.get_cmap(cmap)
            colors = cmap_func(norm(values))
        else:
            colors = plt.cm.get_cmap(cmap)(np.arange(len(indices)) / max(1, len(indices)-1))
        
        # 绘制点
        for i, point_idx in enumerate(indices):
            if point_idx < len(kp_2d):
                x, y = kp_2d[point_idx]
                x = max(0, min(x, image.shape[1]-1))
                y = max(0, min(y, image.shape[0]-1))
                
                if cmap == 'plasma':
                    size = 50 + (point_idx * 20)  # 根据索引调整大小
                else:
                    size = 150
                
                ax.scatter(x, y, c=[colors[i]], s=size, alpha=0.8, 
                          edgecolors='white', linewidth=1.5)
                
                if title not in ['X-Coordinate Heatmap', 'Y-Coordinate Heatmap', 
                                'Distance from Center', 'Keypoint Size by Z-depth']:
                    ax.text(x+8, y+8, f'{point_idx+1}', 
                           color='white', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # 如果需要连线
        if specific_indices is True or (isinstance(specific_indices, list) and len(specific_indices) > 1):
            if isinstance(specific_indices, list):
                points = kp_2d[specific_indices]
            else:
                points = kp_2d
            
            # 简单的面部连线（示意）
            face_connections = [
                [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],  # 眉毛
                [6, 7], [7, 8], [8, 9], [9, 10],  # 鼻子
                [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],  # 右眼
                [17, 18],  # 左眼
                [19, 20]  # 嘴巴
            ]
            
            for conn in face_connections:
                if conn[0] < len(points) and conn[1] < len(points):
                    x1, y1 = points[conn[0]]
                    x2, y2 = points[conn[1]]
                    ax.plot([x1, x2], [y1, y2], 'w-', linewidth=1, alpha=0.5)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'keypoint_alignment_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Alignment grid saved to: {save_path}")

def main():
    """主函数"""
    # 配置路径
    npz_path = "./outputs/kp_npz/frame_000000_kpinfo.npz"
    image_path = input("请输入原始2D图像路径: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ 图像不存在: {image_path}")
        print("请提供原始的2D人脸图像路径")
        return
    
    output_dir = "./outputs/visualization_2d"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("🎭 2D KEYPOINT PROJECTION VISUALIZATION")
    print("="*80)
    
    # 1. 加载数据
    print("📂 Loading kp_info data...")
    kp_info = load_kp_info(npz_path)
    
    # 2. 计算3D关键点
    print("🧮 Calculating 3D keypoints...")
    kp_3d, pose_angles = calculate_3d_keypoints(kp_info)
    
    # 3. 加载图像
    print(f"🖼️  Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 无法加载图像")
        return
    
    h, w = image.shape[:2]
    print(f"   Image size: {w}x{h}")
    
    # 4. 投影到2D
    print("📐 Projecting 3D to 2D...")
    kp_2d = project_3d_to_2d(kp_3d, image.shape, pose_angles)
    
    # 打印2D坐标
    print("\n📊 2D Keypoint Coordinates:")
    print("-" * 50)
    for i, (x, y) in enumerate(kp_2d):
        print(f"KP{i+1:2d}: ({x:7.1f}, {y:7.1f})")
    
    # 5. 可视化
    print("\n🎨 Creating visualizations...")
    
    # 基本可视化
    save_path1 = os.path.join(output_dir, 'keypoints_2d_projection.png')
    kp_2d = visualize_on_image(image_path, kp_2d, save_path1, 
                              "Ditto 3D Keypoints Projected to 2D Image")
    
    # 详细映射图
    create_keypoint_mapping_image(image_path, kp_2d, output_dir)
    
    # 对齐网格
    create_alignment_grid(image_path, kp_2d, output_dir)
    
    # 6. 保存坐标数据
    csv_path = os.path.join(output_dir, 'keypoints_2d_coordinates.csv')
    with open(csv_path, 'w') as f:
        f.write("KP_ID,Image_X,Image_Y,3D_X,3D_Y,3D_Z,Flat_Index,Region\n")
        
        # 区域定义
        regions = {
            'Brow': list(range(6)),
            'Nose': list(range(6, 11)),
            'RightEye': list(range(11, 17)),
            'LeftEye': [17, 18],
            'Mouth': [19, 20]
        }
        
        for i in range(len(kp_2d)):
            # 确定区域
            region = "Other"
            for reg_name, reg_indices in regions.items():
                if i in reg_indices:
                    region = reg_name
                    break
            
            f.write(f"{i+1},{kp_2d[i,0]:.2f},{kp_2d[i,1]:.2f},"
                   f"{kp_3d[i,0]:.6f},{kp_3d[i,1]:.6f},{kp_3d[i,2]:.6f},"
                   f"{i*3},{region}\n")
    
    print(f"✅ Coordinates saved to CSV: {csv_path}")
    
    print("\n" + "="*80)
    print("✅ 2D VISUALIZATION COMPLETE!")
    print(f"   Output directory: {output_dir}")
    print("   Files created:")
    print("     - keypoints_2d_projection.png")
    print("     - keypoint_mapping_detailed.png") 
    print("     - keypoint_alignment_grid.png")
    print("     - keypoints_2d_coordinates.csv")
    print("="*80)

if __name__ == "__main__":
    main()