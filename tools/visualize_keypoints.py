#!/usr/bin/env python3
"""
Complete 3D Keypoint Visualization for Ditto
Updated to handle nested dict structure
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

def load_kp_info(npz_path):
    """加载kp_info.npz文件，处理嵌套字典结构"""
    print(f"📂 Loading kp_info from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # 检查数据格式
    if 'kp_info' in data:
        # 如果保存为字典对象
        kp_info = data['kp_info'].item()
        print("  Format: dict saved under 'kp_info' key")
        
        # 检查是否是嵌套字典（x_s_info, x_d_info）
        if isinstance(kp_info, dict):
            print(f"  Nested keys: {list(kp_info.keys())}")
            
            # 提取 x_s_info 和 x_d_info
            x_s_info = kp_info.get('x_s_info', {})
            x_d_info = kp_info.get('x_d_info', {})
            
            print(f"\n  x_s_info keys: {list(x_s_info.keys()) if isinstance(x_s_info, dict) else 'Not a dict'}")
            print(f"  x_d_info keys: {list(x_d_info.keys()) if isinstance(x_d_info, dict) else 'Not a dict'}")
            
            # 返回主要的信息（通常x_s_info包含源图像的关键点）
            return x_s_info
    else:
        # 如果直接保存为数组
        kp_info = {k: data[k] for k in data.files}
        print(f"  Format: direct arrays, keys: {list(kp_info.keys())}")
    
    return kp_info

def extract_kp_and_exp(kp_info):
    """从kp_info字典中提取kp和exp"""
    print("\n🔍 Extracting kp and exp from kp_info...")
    
    # 方法1：直接查找
    if 'kp' in kp_info and 'exp' in kp_info:
        print("  Found kp and exp directly")
        kp = kp_info['kp']
        exp = kp_info['exp']
        extra = {k: kp_info.get(k, None) for k in ('pitch', 'yaw', 'roll', 't', 'scale')}
        return kp, exp, extra
    
    # 方法2：检查常见的关键名称
    key_mapping = {
        'kp': ['kp', 'keypoints', 'canonical_keypoints', 'c_ref'],
        'exp': ['exp', 'expression', 'deformation', 'delta', 'δ']
    }
    
    for target_key, possible_keys in key_mapping.items():
        for key in possible_keys:
            if key in kp_info:
                print(f"  Found {target_key} as '{key}'")
                if target_key == 'kp':
                    kp = kp_info[key]
                else:
                    exp = kp_info[key]
                break
    
    # 方法3：检查数组形状
    print("\n  All arrays in kp_info:")
    for key, value in kp_info.items():
        if isinstance(value, np.ndarray):
            shape = value.shape
            flat_len = value.size
            print(f"    {key}: shape={shape}, flattened={flat_len}")
            
            # 根据形状猜测
            if flat_len == 63 or (shape[-1] == 63 and len(shape) <= 2):
                print(f"      → Possible expression vector (63-D)")
                exp = value
            elif flat_len == 66 or (shape[-1] == 66 and len(shape) <= 2):
                print(f"      → Possible pose distribution (66-bin)")
            elif flat_len == 3:
                print(f"      → Possible translation vector")
            elif flat_len == 1:
                print(f"      → Possible scale")
            elif flat_len % 3 == 0 and flat_len >= 21:
                num_kp = flat_len // 3
                print(f"      → Possible keypoints: {num_kp} points")
                if 'kp' not in locals():
                    kp = value
    
    if 'kp' not in locals() or 'exp' not in locals():
        print("\n❌ Could not find both kp and exp")
        print("   Available keys:", list(kp_info.keys()))
        
        # 尝试从所有数据中重构
        all_data = []
        for key, value in kp_info.items():
            if isinstance(value, np.ndarray):
                all_data.append(value.flatten())
        
        if all_data:
            combined = np.concatenate(all_data)
            print(f"\n   Total flattened data length: {len(combined)}")
            
            # 尝试推断：63维的exp + 63维的kp + 其他
            if len(combined) >= 126:  # 至少kp和exp
                print("   Attempting to extract kp and exp from combined data...")
                exp = combined[:63].reshape(1, -1)
                kp = combined[63:126].reshape(1, -1)
                print(f"   Extracted exp shape: {exp.shape}")
                print(f"   Extracted kp shape: {kp.shape}")
            else:
                # 如果只有63维，假设是exp
                exp = combined.reshape(1, -1)
                kp = np.zeros_like(exp)
                print(f"   Assuming exp only, shape: {exp.shape}")
    extra = {k: kp_info.get(k, None) for k in ('pitch', 'yaw', 'roll', 't', 'scale')}
    return kp, exp, extra

def _to_batch_kp(a):
    a = np.asarray(a)
    if a.ndim == 1:
        if a.size % 3 != 0:
            raise ValueError('Flat keypoint array length is not divisible by 3')
        return a.reshape(1, a.size // 3, 3)
    if a.ndim == 2:
        # cases: (N,3) -> treat as (1,N,3); (bs, N*3) -> reshape to (bs, N,3)
        if a.shape[1] == 3:
            return a[np.newaxis, ...]
        if a.shape[1] % 3 == 0:
            return a.reshape(a.shape[0], a.shape[1] // 3, 3)
        # fallback: flatten and try
        if a.size % 3 == 0:
            return a.reshape(1, a.size // 3, 3)
    if a.ndim == 3:
        return a
    raise ValueError('Unsupported keypoint array shape: ' + str(a.shape))


def _rotation_matrices_from_degrees(pitch, yaw, roll):
    """输入角度（deg）标量或数组（bs,），返回 (bs,3,3) 旋转矩阵。顺序: R = Rz(roll) @ Ry(yaw) @ Rx(pitch)"""
    pitch = np.asarray(pitch) if pitch is not None else None
    yaw = np.asarray(yaw) if yaw is not None else None
    roll = np.asarray(roll) if roll is not None else None

    # Determine batch size
    bs = 1
    for arr in (pitch, yaw, roll):
        if arr is not None:
            if arr.ndim == 0:
                bs = max(bs, 1)
            else:
                bs = max(bs, arr.shape[0])

    def to_arr(x):
        if x is None:
            return np.zeros((bs,))
        xa = np.asarray(x).reshape(-1)
        if xa.shape[0] == 1 and bs > 1:
            return np.repeat(xa, bs)
        return xa

    p = np.radians(to_arr(pitch))
    y = np.radians(to_arr(yaw))
    r = np.radians(to_arr(roll))

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


def transform_keypoints(kp, exp, extra=None):
    """基于 LivePortrait 的 Eqn.2: s * (R * x_c + exp) + t
    kp, exp 可以是多种形状（flat, (N,3), (bs,N,3)），extra 可包含 'pitch','yaw','roll','t','scale'
    返回形状 (bs, N, 3) 或 (N,3)（如果 bs==1 会 squeeze）
    """
    print("\n🧮 Calculating 3D positions using transform (R * kp + exp)...")

    kp_b = _to_batch_kp(kp)

    # Try to interpret exp as per-keypoint displacements
    try:
        exp_b = _to_batch_kp(exp)
    except Exception:
        exp_arr = np.asarray(exp)
        if exp_arr.size == 63:
            try:
                exp_b = exp_arr.reshape(1, 21, 3)
                exp_b = np.repeat(exp_b, kp_b.shape[0], axis=0)
                print("  Note: exp is 63-D, reshaped to (21,3) and broadcasted to batch")
            except Exception:
                exp_b = np.zeros_like(kp_b)
                print("  Warning: cannot reshape 63-D exp to per-keypoint displacements; using zeros")
        else:
            exp_b = np.zeros_like(kp_b)
            print("  Warning: exp not per-keypoint, using zero displacements")

    bs = kp_b.shape[0]
    num_kp = kp_b.shape[1]

    # If exp has different num_kp, trim or pad
    if exp_b.shape[1] != num_kp:
        min_k = min(exp_b.shape[1], num_kp)
        if exp_b.shape[1] < num_kp:
            pad = np.zeros((bs, num_kp - exp_b.shape[1], 3), dtype=exp_b.dtype)
            exp_b = np.concatenate([exp_b, pad], axis=1)
        else:
            exp_b = exp_b[:, :num_kp, :]
        print(f"  Adjusted exp to match keypoint count: {exp_b.shape}")

    # pose info
    pitch = yaw = roll = None
    t = None
    scale = None
    if extra is not None:
        pitch = extra.get('pitch', None)
        yaw = extra.get('yaw', None)
        roll = extra.get('roll', None)
        t = extra.get('t', None)
        scale = extra.get('scale', None)

    # rotation matrices
    if pitch is None or yaw is None or roll is None:
        rot_mats = np.repeat(np.eye(3)[None, ...], bs, axis=0)
        print("  Note: pitch/yaw/roll not provided, using identity rotation")
    else:
        rot_mats = _rotation_matrices_from_degrees(pitch, yaw, roll)

    # Apply rotation (kp @ R) and add exp
    kp_rot = np.einsum('bnc,bcd->bnd', kp_b, rot_mats)
    kp_transformed = kp_rot + exp_b

    # scale
    if scale is not None:
        s = np.asarray(scale).reshape(-1)
        if s.size == 1:
            kp_transformed = kp_transformed * s.item()
        else:
            kp_transformed = kp_transformed * s[:, None, None]

    # translation (only tx,ty)
    if t is not None:
        t_arr = np.asarray(t)
        if t_arr.ndim == 1 and t_arr.size >= 2:
            if t_arr.size == 2:
                t_arr = t_arr.reshape(1, 2)
            if t_arr.shape[0] == 1 and bs > 1:
                t_arr = np.repeat(t_arr, bs, axis=0)
            kp_transformed[:, :, 0:2] += t_arr[:, None, 0:2]

    # Enforce convention: keypoint 1's Y must be positive. If KP1 Y is negative,
    # flip the Y axis for all keypoints in that batch (handles both bs>1 and bs==1).
    try:
        if kp_transformed.ndim == 3:
            for bi in range(kp_transformed.shape[0]):
                if kp_transformed[bi, 0, 1] < 0:
                    kp_transformed[bi, :, 1] = -kp_transformed[bi, :, 1]
        elif kp_transformed.ndim == 2:
            if kp_transformed[0, 1] < 0:
                kp_transformed[:, 1] = -kp_transformed[:, 1]
    except Exception:
        # If any unexpected shape issue arises, skip enforcement silently
        pass

    if kp_transformed.shape[0] == 1:
        return kp_transformed[0]
    return kp_transformed

def visualize_3d_keypoints(kp_3d, save_dir='./outputs/visualization'):
    """3D可视化关键点"""
    os.makedirs(save_dir, exist_ok=True)
    
    if kp_3d.ndim == 3:
        kp_3d = kp_3d[0]  # 取第一个batch
    
    num_kp = kp_3d.shape[0]
    print(f"\n🎨 Visualizing {num_kp} 3D keypoints")
    
    # 设置颜色映射
    colors = plt.cm.tab20(np.arange(num_kp) / max(num_kp, 1))
    
    # 创建多子图
    fig = plt.figure(figsize=(20, 6))
    
    # 1. 3D散点图
    ax1 = fig.add_subplot(131, projection='3d')
    xs, ys, zs = kp_3d[:, 0], kp_3d[:, 1], kp_3d[:, 2]
    
    scatter = ax1.scatter(xs, ys, zs, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # 标记序号
    for i in range(num_kp):
        ax1.text(xs[i], ys[i], zs[i], f'{i+1}', 
                fontsize=12, fontweight='bold', 
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('X', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax1.set_title('3D Facial Keypoints', fontsize=14, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. XY平面（正面视图）
    ax2 = fig.add_subplot(132)
    ax2.scatter(xs, ys, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    for i in range(num_kp):
        ax2.text(xs[i], ys[i], f'{i+1}', 
                fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel('X', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax2.set_title('XY Plane (Front View)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. XZ平面（侧视图）
    ax3 = fig.add_subplot(133)
    ax3.scatter(xs, zs, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    for i in range(num_kp):
        ax3.text(xs[i], zs[i], f'{i+1}', 
                fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    ax3.set_xlabel('X', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Z', fontsize=12, fontweight='bold')
    ax3.set_title('XZ Plane (Side View)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, '3d_keypoints_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 3D visualization saved to: {save_path}")
    
    # 显示图像
    plt.show()
    
    return fig

def create_keypoint_table(kp_3d):
    """创建关键点信息表格"""
    if kp_3d.ndim == 3:
        kp_3d = kp_3d[0]
    
    num_kp = kp_3d.shape[0]
    
    print("\n" + "="*80)
    print("📋 KEYPOINT INFORMATION TABLE")
    print("="*80)
    print(f"{'ID':<4} {'X':<12} {'Y':<12} {'Z':<12} {'Flat Index':<20} {'Possible Region':<25}")
    print("-"*80)
    
    for i in range(num_kp):
        kp_id_1based = i + 1
        kp_id_0based = i
        
        # 计算扁平索引
        flat_x = i * 3
        flat_y = i * 3 + 1
        flat_z = i * 3 + 2
        
        # 确定区域（基于常见的21点布局）
        region = "Unknown"
        
        # 尝试推断区域
        if i < 6:
            region = "Brow/Forehead"
        elif i < 11:
            region = "Nose/Cheek"
        elif i == 11:  # KP12
            region = "Right Eye Outer 👁"
        elif i == 12:  # KP13
            region = "Right Eye"
        elif i == 13:  # KP14
            region = "Right Eye"
        elif i == 14:  # KP15
            region = "Right Eye Pupil 👁"
        elif i == 15:  # KP16
            region = "Right Eye"
        elif i == 16:  # KP17
            region = "Right Eye Inner"
        elif i == 17:  # KP18
            region = "Left Eye Inner"
        elif i == 18:  # KP19
            region = "Left Eye 👁"
        elif i == 19:  # KP20
            region = "Chin/Jaw 👄"
        elif i == 20:  # KP21
            region = "Lip Corner 👄"
        
        # 特殊标注（基于论文）
        special_note = ""
        if kp_id_1based == 12:  # KP12
            special_note = f" [Paper: dim34={flat_x} controls eye open]"
        elif kp_id_1based == 20:  # KP20
            special_note = f" [Paper: dim58={flat_z} controls mouth open]"
        
        print(f"{kp_id_1based:<4} "
              f"{kp_3d[i, 0]:<12.6f} {kp_3d[i, 1]:<12.6f} {kp_3d[i, 2]:<12.6f} "
              f"x:{flat_x:<3} y:{flat_y:<3} z:{flat_z:<3}  "
              f"{region:<25}{special_note}")

def analyze_pose_parameters(kp_info):
    """分析姿态参数"""
    print("\n" + "="*80)
    print("🧭 POSE PARAMETER ANALYSIS")
    print("="*80)
    
    pose_keys = ['pitch', 'yaw', 'roll', 't', 'scale']
    
    for key in pose_keys:
        if key in kp_info:
            value = kp_info[key]
            if isinstance(value, np.ndarray):
                print(f"\n{key.upper()}:")
                print(f"  Shape: {value.shape}")
                print(f"  Values: {value.flatten()}")
                
                if key in ['pitch', 'yaw', 'roll'] and value.size == 66:
                    # 分析66-bin分布
                    bins = np.linspace(-99, 99, 66)
                    max_bin = np.argmax(value)
                    max_angle = bins[max_bin]
                    print(f"  Max bin: {max_bin} (angle ≈ {max_angle:.1f}°)")
            else:
                print(f"{key}: {value}")

def debug_kp_info_structure(kp_info):
    """深入调试kp_info结构"""
    print("\n" + "="*80)
    print("🔬 DEEP DEBUG OF KP_INFO STRUCTURE")
    print("="*80)
    
    def print_dict_structure(d, indent=0, max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            print("  " * indent + "... (max depth reached)")
            return
            
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}: dict with keys: {list(value.keys())}")
                print_dict_structure(value, indent+1, max_depth, current_depth+1)
            elif isinstance(value, np.ndarray):
                print("  " * indent + f"{key}: np.ndarray shape={value.shape}, dtype={value.dtype}")
                if value.size <= 10:
                    print("  " * (indent+1) + f"values: {value.flatten()}")
            elif isinstance(value, (list, tuple)):
                print("  " * indent + f"{key}: {type(value).__name__} length={len(value)}")
                if len(value) <= 5 and current_depth < max_depth-1:
                    for i, item in enumerate(value[:3]):
                        if isinstance(item, dict):
                            print("  " * (indent+1) + f"[{i}]: dict with keys: {list(item.keys())}")
                        elif isinstance(item, np.ndarray):
                            print("  " * (indent+1) + f"[{i}]: np.ndarray shape={item.shape}")
                        else:
                            print("  " * (indent+1) + f"[{i}]: {type(item).__name__}")
                    if len(value) > 3:
                        print("  " * (indent+1) + f"... and {len(value)-3} more")
            else:
                print("  " * indent + f"{key}: {type(value).__name__} = {value}")
    
    print_dict_structure(kp_info)

def save_raw_data(kp_info, save_dir='./outputs/visualization'):
    """保存原始数据以便进一步分析"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存为文本文件
    txt_path = os.path.join(save_dir, 'kp_info_raw.txt')
    with open(txt_path, 'w') as f:
        f.write("RAW KP_INFO DATA\n")
        f.write("="*50 + "\n")
        
        def write_dict(d, indent=0, f=f):
            for key, value in d.items():
                if isinstance(value, dict):
                    f.write("  " * indent + f"{key}:\n")
                    write_dict(value, indent+1, f)
                elif isinstance(value, np.ndarray):
                    f.write("  " * indent + f"{key}: shape={value.shape}, dtype={value.dtype}\n")
                    if value.size <= 20:
                        f.write("  " * (indent+1) + f"values: {value.flatten()}\n")
                else:
                    f.write("  " * indent + f"{key}: {type(value)} = {value}\n")
        
        write_dict(kp_info)
    
    print(f"✅ Raw data saved to: {txt_path}")

def main():
    """主函数"""
    # 配置
    npz_path = "./outputs/kp_npz/frame_000000_kpinfo.npz"
    output_dir = "./outputs/visualization_3d"
    
    print("="*80)
    print("🎭 DITTO 3D KEYPOINT VISUALIZATION TOOL")
    print("="*80)
    
    # 1. 加载数据
    kp_info = load_kp_info(npz_path)
    
    # 2. 调试数据结构
    debug_kp_info_structure(kp_info)
    
    # 3. 提取kp和exp（以及可能的pose/scale/t）
    kp, exp, extra = extract_kp_and_exp(kp_info)
    
    # 4. 分析姿态参数
    analyze_pose_parameters(kp_info)
    
    # 5. 计算3D关键点位置
    print("\n🔧 Calculating 3D keypoint positions...")
    kp_3d = transform_keypoints(kp, exp, extra)
    
    # 6. 创建关键点信息表格
    create_keypoint_table(kp_3d)
    
    # 7. 3D可视化
    print("\n🎨 Generating visualizations...")
    visualize_3d_keypoints(kp_3d, output_dir)
    
    # 8. 保存原始数据
    save_raw_data(kp_info, output_dir)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print(f"   Output directory: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()