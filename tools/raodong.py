#!/usr/bin/env python3
"""
修复版单帧扰动实验 - 正确处理关键点坐标
"""
import argparse
import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import csv
import pickle

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.atomic_components.cfg import parse_cfg
from core.atomic_components.avatar_registrar import AvatarRegistrar
from core.atomic_components.motion_stitch import transform_keypoint


def extract_single_frame_info(image_path, cfg_pkl, data_root, **kwargs):
    """
    提取单帧的运动参数
    """
    # 解析配置
    cfg_list = parse_cfg(cfg_pkl, data_root, kwargs)
    avatar_registrar_cfg = cfg_list[0]
    
    # 初始化AvatarRegistrar
    registrar = AvatarRegistrar(**avatar_registrar_cfg)
    
    # 提取信息
    crop_kwargs = {
        "crop_scale": kwargs.get("crop_scale", 2.3),
        "crop_vx_ratio": kwargs.get("crop_vx_ratio", 0),
        "crop_vy_ratio": kwargs.get("crop_vy_ratio", -0.125),
        "crop_flag_do_rot": kwargs.get("crop_flag_do_rot", True),
    }
    
    max_dim = kwargs.get("max_size", 512)
    
    source_info = registrar(
        image_path, 
        max_dim=max_dim,
        n_frames=1, 
        **crop_kwargs,
    )
    
    # 提取第一帧的x_s_info
    if "x_s_info_lst" in source_info and len(source_info["x_s_info_lst"]) > 0:
        x_s_info = source_info["x_s_info_lst"][0]
        
        # 调试：打印关键信息
        print(f"提取到的x_s_info字段: {list(x_s_info.keys())}")
        for key in ['kp', 'exp', 'pitch', 'yaw', 'roll', 't', 'scale']:
            if key in x_s_info:
                val = x_s_info[key]
                if hasattr(val, 'shape'):
                    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"  {key}: {val}")
        
        return x_s_info, source_info
    else:
        raise ValueError("无法提取运动参数")


def get_image_crop_info(image_path, source_info):
    """
    获取图像裁剪信息，用于正确绘制关键点
    """
    if "img_rgb_lst" in source_info and len(source_info["img_rgb_lst"]) > 0:
        # 获取裁剪后的图像
        cropped_img = source_info["img_rgb_lst"][0]
        print(f"裁剪后图像形状: {cropped_img.shape}")
        return cropped_img
    
    # 如果没有裁剪后的图像，加载原图
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def transform_and_scale_keypoints(x_s_info, image_shape):
    """
    转换关键点并缩放到图像坐标
    """
    # 转换关键点
    kps_3d = transform_keypoint(x_s_info)
    
    # 处理batch维度
    if kps_3d.ndim == 3:
        kps_3d = kps_3d[0]  # (21, 3)
    
    print(f"转换后关键点形状: {kps_3d.shape}")
    print(f"关键点范围: x[{kps_3d[:, 0].min():.2f}, {kps_3d[:, 0].max():.2f}], "
          f"y[{kps_3d[:, 1].min():.2f}, {kps_3d[:, 1].max():.2f}], "
          f"z[{kps_3d[:, 2].min():.2f}, {kps_3d[:, 2].max():.2f}]")
    
    # 获取x,y坐标
    kps_2d = kps_3d[:, :2]  # (21, 2)
    
    # 关键点可能是在归一化坐标或原始坐标中
    # 尝试自动检测和缩放
    
    # 如果坐标范围在[-1, 1]或[0, 1]之间，需要缩放到图像尺寸
    x_range = kps_2d[:, 0].max() - kps_2d[:, 0].min()
    y_range = kps_2d[:, 1].max() - kps_2d[:, 1].min()
    
    print(f"x范围: {x_range:.4f}, y范围: {y_range:.4f}")
    
    # 常见的归一化范围检测
    if x_range < 2.0 and y_range < 2.0:
        print("检测到归一化坐标，进行缩放...")
        # 可能是在[-1, 1]范围内
        if kps_2d[:, 0].min() >= -1 and kps_2d[:, 0].max() <= 1:
            # 从[-1, 1]缩放到[0, image_width]
            kps_2d[:, 0] = (kps_2d[:, 0] + 1) * 0.5 * image_shape[1]
            kps_2d[:, 1] = (kps_2d[:, 1] + 1) * 0.5 * image_shape[0]
        elif kps_2d[:, 0].min() >= 0 and kps_2d[:, 0].max() <= 1:
            # 从[0, 1]缩放到图像尺寸
            kps_2d[:, 0] = kps_2d[:, 0] * image_shape[1]
            kps_2d[:, 1] = kps_2d[:, 1] * image_shape[0]
    
    print(f"缩放后关键点范围: x[{kps_2d[:, 0].min():.2f}, {kps_2d[:, 0].max():.2f}], "
          f"y[{kps_2d[:, 1].min():.2f}, {kps_2d[:, 1].max():.2f}]")
    
    return kps_2d


def visualize_keypoint_changes_direct(image_path, x_s_info_orig, x_s_info_pert, 
                                     kp_idx, axis_idx, delta, out_path, 
                                     source_info=None):
    """
    直接可视化关键点变化（使用更直接的方法）
    """
    try:
        # 获取裁剪后的图像用于显示
        if source_info and "img_rgb_lst" in source_info and len(source_info["img_rgb_lst"]) > 0:
            img_rgb = source_info["img_rgb_lst"][0]
            print(f"使用裁剪后图像: {img_rgb.shape}")
        else:
            # 加载原图
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法加载图像: {image_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"使用原始图像: {img_rgb.shape}")
        
        # 获取图像尺寸
        img_height, img_width = img_rgb.shape[:2]
        
        # 转换并缩放关键点
        kps_orig_2d = transform_and_scale_keypoints(x_s_info_orig, (img_height, img_width))
        kps_pert_2d = transform_and_scale_keypoints(x_s_info_pert, (img_height, img_width))
        
        # 检查关键点是否在图像范围内
        print(f"原始关键点位置示例: KP1={kps_orig_2d[0]}")
        print(f"扰动后关键点位置示例: KP1={kps_pert_2d[0]}")
        
        # 计算变化
        delta_2d = kps_pert_2d - kps_orig_2d
        print(f"关键点变化: KP{kp_idx+1} Δ={delta_2d[kp_idx]}")
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # 左侧：原始关键点
        axes[0].imshow(img_rgb)
        # 绘制所有关键点
        axes[0].scatter(kps_orig_2d[:, 0], kps_orig_2d[:, 1], 
                       c='red', s=30, alpha=0.7, label='Keypoints')
        # 标注关键点编号
        for i, (x, y) in enumerate(kps_orig_2d):
            axes[0].text(x+3, y+3, str(i+1), 
                        color='yellow', fontsize=9, weight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
        axes[0].set_title('Original Keypoints')
        axes[0].legend(loc='upper right')
        axes[0].axis('off')
        
        # 右侧：扰动后关键点
        axes[1].imshow(img_rgb)
        # 绘制所有关键点
        axes[1].scatter(kps_pert_2d[:, 0], kps_pert_2d[:, 1], 
                       c='red', s=30, alpha=0.7, label='Keypoints')
        
        # 特别标注被扰动的关键点
        changed_kp = kp_idx
        axes[1].scatter(kps_pert_2d[changed_kp, 0], kps_pert_2d[changed_kp, 1], 
                       c='cyan', s=150, marker='*', edgecolors='white', 
                       linewidth=2, label=f'Perturbed KP{changed_kp+1}')
        
        # 绘制变化向量（如果变化明显）
        dx, dy = delta_2d[changed_kp]
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            axes[1].arrow(kps_orig_2d[changed_kp, 0], kps_orig_2d[changed_kp, 1],
                         dx, dy, color='lime', width=3, 
                         head_width=15, head_length=15, 
                         length_includes_head=True,
                         label='Change vector')
        
        # 标注关键点编号
        for i, (x, y) in enumerate(kps_pert_2d):
            axes[1].text(x+3, y+3, str(i+1), 
                        color='yellow', fontsize=9, weight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
        
        axis_name = ['x', 'y', 'z'][axis_idx]
        axes[1].set_title(f'Perturbed: KP{kp_idx+1} {axis_name} (+{delta:.3f})')
        axes[1].legend(loc='upper right')
        axes[1].axis('off')
        
        # 添加整体标题
        dim_idx = kp_idx * 3 + axis_idx
        title = f'Dimension {dim_idx+1}: KP{kp_idx+1}-{axis_name} (Δ={delta})'
        if dim_idx + 1 == 34:
            title += ' - Controls right eye open/close'
        elif dim_idx + 1 == 58:
            title += ' - Controls mouth open'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"图像已保存: {out_path}")
        
        return kps_orig_2d, kps_pert_2d
        
    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def debug_keypoint_transform(x_s_info):
    """
    调试函数：查看transform_keypoint的详细信息
    """
    print("\n" + "="*60)
    print("关键点转换调试信息")
    print("="*60)
    
    # 打印输入信息
    for key in ['kp', 'exp', 'pitch', 'yaw', 'roll', 't', 'scale']:
        if key in x_s_info:
            val = x_s_info[key]
            if hasattr(val, 'shape'):
                print(f"{key}: shape={val.shape}")
                if val.size < 10:  # 小数组打印全部值
                    print(f"  values: {val}")
            else:
                print(f"{key}: {val}")
    
    # 尝试转换
    try:
        kps_transformed = transform_keypoint(x_s_info)
        print(f"\n转换后关键点形状: {kps_transformed.shape}")
        
        if kps_transformed.ndim == 3:
            kps = kps_transformed[0]  # 取第一个batch
        else:
            kps = kps_transformed
        
        print(f"关键点坐标 (前5个):")
        for i in range(min(5, len(kps))):
            print(f"  KP{i+1}: ({kps[i, 0]:.4f}, {kps[i, 1]:.4f}, {kps[i, 2]:.4f})")
        
        print(f"\n坐标范围:")
        print(f"  X: [{kps[:, 0].min():.4f}, {kps[:, 0].max():.4f}]")
        print(f"  Y: [{kps[:, 1].min():.4f}, {kps[:, 1].max():.4f}]")
        print(f"  Z: [{kps[:, 2].min():.4f}, {kps[:, 2].max():.4f}]")
        
        return kps
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="修复版单帧扰动实验")
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--cfg_pkl', required=True, help='配置文件路径')
    parser.add_argument('--data_root', required=True, help='模型数据根目录')
    parser.add_argument('--out_dir', default='outputs/debug_perturb', help='输出目录')
    parser.add_argument('--delta', type=float, default=0.5, help='扰动幅度（增大以看到明显变化）')
    parser.add_argument('--dimension', type=int, default=34, help='处理特定维度（默认34）')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 60)
    print("修复版单帧扰动实验")
    print(f"输入图像: {args.image}")
    print(f"处理维度: {args.dimension}")
    print(f"扰动幅度: {args.delta}")
    print("=" * 60)
    
    # 1. 提取运动参数
    print("\n[1/3] 提取图像运动参数...")
    try:
        kwargs = {
            "max_size": 512,
            "crop_scale": 2.3,
        }
        
        x_s_info, source_info = extract_single_frame_info(
            args.image, args.cfg_pkl, args.data_root, **kwargs
        )
        print("✓ 成功提取运动参数")
        
        # 调试模式：查看详细信息
        if args.debug:
            debug_keypoint_transform(x_s_info)
        
        # 检查和处理exp
        if 'exp' not in x_s_info:
            print("错误: x_s_info中没有exp字段")
            return
        
        exp_original = x_s_info['exp']
        print(f"原始exp形状: {exp_original.shape}")
        
        # 确保exp是正确形状 (1, 21, 3)
        if exp_original.ndim == 1 and exp_original.shape[0] == 63:
            exp_original = exp_original.reshape(1, 21, 3)
        elif exp_original.ndim == 2 and exp_original.shape[1] == 63:
            exp_original = exp_original.reshape(1, 21, 3)
        elif exp_original.ndim == 3 and exp_original.shape[1] == 21 and exp_original.shape[2] == 3:
            pass  # 已经是正确形状
        else:
            print(f"警告: exp形状异常，尝试修复...")
            try:
                # 展平并取前63个值
                exp_flat = exp_original.flatten()
                if len(exp_flat) >= 63:
                    exp_original = exp_flat[:63].reshape(1, 21, 3)
                else:
                    print(f"无法修复: exp只有{len(exp_flat)}个值")
                    return
            except:
                print("无法修复exp形状")
                return
        
        x_s_info['exp'] = exp_original
        print(f"修复后exp形状: {exp_original.shape}")
        
        # 打印一些exp值
        print(f"exp值示例:")
        print(f"  KP12 (右眼上眼睑): x={exp_original[0, 11, 0]:.4f}, y={exp_original[0, 11, 1]:.4f}, z={exp_original[0, 11, 2]:.4f}")
        print(f"  KP20 (下唇中部): x={exp_original[0, 19, 0]:.4f}, y={exp_original[0, 19, 1]:.4f}, z={exp_original[0, 19, 2]:.4f}")
            
    except Exception as e:
        print(f"✗ 提取运动参数失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 进行扰动
    print(f"\n[2/3] 进行维度 {args.dimension} 的扰动...")
    
    dim_idx = args.dimension - 1  # 转换为0-based
    kp_idx = dim_idx // 3
    axis_idx = dim_idx % 3
    axis_name = ['x', 'y', 'z'][axis_idx]
    
    print(f"维度{args.dimension} → KP{kp_idx+1} {axis_name}")
    
    # 创建扰动后的exp
    exp_perturbed = exp_original.copy()
    original_value = exp_perturbed[0, kp_idx, axis_idx]
    exp_perturbed[0, kp_idx, axis_idx] = original_value + args.delta
    
    print(f"KP{kp_idx+1} {axis_name}: {original_value:.4f} → {exp_perturbed[0, kp_idx, axis_idx]:.4f}")
    
    # 创建扰动后的x_s_info
    x_s_info_pert = x_s_info.copy()
    x_s_info_pert['exp'] = exp_perturbed
    
    # 3. 可视化
    print(f"\n[3/3] 可视化关键点变化...")
    
    out_img_path = os.path.join(
        args.out_dir, 
        f"dim{args.dimension:03d}_kp{kp_idx+1}_{axis_name}_delta{args.delta}.png"
    )
    
    kps_orig, kps_pert = visualize_keypoint_changes_direct(
        args.image, x_s_info, x_s_info_pert,
        kp_idx, axis_idx, args.delta, out_img_path, source_info
    )
    
    if kps_orig is not None and kps_pert is not None:
        # 分析变化
        delta_2d = kps_pert - kps_orig
        print(f"\n关键点变化分析:")
        print(f"被扰动关键点 KP{kp_idx+1}: Δ({delta_2d[kp_idx, 0]:.2f}, {delta_2d[kp_idx, 1]:.2f})")
        
        # 找出变化最大的关键点
        changes_magnitude = np.linalg.norm(delta_2d, axis=1)
        max_change_idx = np.argmax(changes_magnitude)
        max_change = changes_magnitude[max_change_idx]
        
        print(f"最大变化: KP{max_change_idx+1}, 幅度={max_change:.2f}")
        
        # 面部区域描述
        facial_regions = {
            1: "额头", 2: "左眉外侧", 3: "左眉内侧", 4: "右眉内侧", 5: "右眉外侧",
            6: "左眼外侧", 7: "左眼上睑", 8: "左眼下睑", 9: "左眼内侧", 10: "右眼内侧",
            11: "右下睑", 12: "右上睑", 13: "右眼外侧", 14: "鼻梁", 15: "鼻尖",
            16: "左鼻翼", 17: "右鼻翼", 18: "左上唇", 19: "右上唇", 20: "嘴中部",
            21: "下巴"
        }
        
        # 记录有明显变化的关键点
        significant_changes = np.where(changes_magnitude > 1.0)[0]  # 变化大于1像素
        if len(significant_changes) > 0:
            print(f"有明显变化的关键点:")
            for idx in significant_changes:
                region = facial_regions.get(idx+1, f"KP{idx+1}")
                dx, dy = delta_2d[idx]
                print(f"  {region}: Δ({dx:.2f}, {dy:.2f})")
        
        # 保存结果到CSV
        csv_path = os.path.join(args.out_dir, 'perturbation_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['维度', '关键点', '轴', '扰动值', '图像路径', '影响区域'])
            
            affected_regions = []
            for idx in significant_changes:
                region = facial_regions.get(idx+1, f"KP{idx+1}")
                if region not in affected_regions:
                    affected_regions.append(region)
            
            writer.writerow([
                args.dimension,
                f"KP{kp_idx+1}",
                axis_name,
                args.delta,
                out_img_path,
                ', '.join(affected_regions) if affected_regions else "无明显变化"
            ])
        
        print(f"\n✓ 完成!")
        print(f"图像保存到: {out_img_path}")
        print(f"结果保存到: {csv_path}")
        
        # 特别提示
        if args.dimension == 34:
            print(f"\n注意：维度34应该控制右眼的睁开/闭合")
            print(f"请检查图像中KP12（右眼上眼睑）的变化")
        elif args.dimension == 58:
            print(f"\n注意：维度58应该控制嘴巴张开")
            print(f"请检查图像中KP20（下唇中部）的变化")
    else:
        print("✗ 可视化失败")


def quick_verify():
    """
    快速验证：测试多个扰动值
    """
    print("快速验证：测试不同扰动值对维度34的影响")
    print("=" * 60)
    
    # 测试不同的delta值
    test_deltas = [0.1, 0.3, 0.5, 1.0, 2.0]
    
    for delta in test_deltas:
        print(f"\n测试 delta = {delta}")
        
        # 这里模拟一个x_s_info进行测试
        # 在实际中，你需要先提取x_s_info
        
        print(f"建议: 使用 delta={delta} 运行完整脚本查看效果")


if __name__ == '__main__':
    main()
    
    # 或者运行快速验证
    # quick_verify()