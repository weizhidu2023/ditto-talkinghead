#!/usr/bin/env python3
"""
修复版单帧扰动对比实验
"""
import argparse
import os
import sys
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.atomic_components.cfg import parse_cfg
from core.atomic_components.avatar_registrar import AvatarRegistrar
from core.atomic_components.motion_stitch import MotionStitch, transform_keypoint
from core.atomic_components.warp_f3d import WarpF3D
from core.atomic_components.decode_f3d import DecodeF3D
from core.atomic_components.putback import PutBack


class SingleFramePerturbRenderer:
    """单帧扰动渲染器 - 修复版"""
    
    def __init__(self, cfg_pkl, data_root):
        # 解析配置
        cfg_list = parse_cfg(cfg_pkl, data_root, {})
        
        self.avatar_registrar_cfg = cfg_list[0]
        self.stitch_network_cfg = cfg_list[3]
        self.warp_network_cfg = cfg_list[4]
        self.decoder_cfg = cfg_list[5]
        
        # 初始化组件
        self.avatar_registrar = AvatarRegistrar(**self.avatar_registrar_cfg)
        self.motion_stitch = MotionStitch(self.stitch_network_cfg)
        self.warp_f3d = WarpF3D(self.warp_network_cfg)
        self.decode_f3d = DecodeF3D(self.decoder_cfg)
        self.putback = PutBack()
    
    def extract_frame_info(self, image_path):
        """提取单帧信息"""
        crop_kwargs = {
            "crop_scale": 2.3,
            "crop_vx_ratio": 0,
            "crop_vy_ratio": -0.125,
            "crop_flag_do_rot": True,
        }
        
        source_info = self.avatar_registrar(
            image_path, 
            max_dim=512,
            n_frames=1, 
            **crop_kwargs,
        )
        
        # 提取第一帧信息
        if "x_s_info_lst" in source_info and len(source_info["x_s_info_lst"]) > 0:
            x_s_info = source_info["x_s_info_lst"][0]
            f_s = source_info["f_s_lst"][0]
            img_rgb = source_info["img_rgb_lst"][0]
            M_c2o = source_info["M_c2o_lst"][0]
            
            # 确保x_s_info中的exp是正确形状
            # 统一为扁平形式 (1, 63)，以匹配 motion_stitch 中的 a1/a2 形状
            self.fix_exp_shape(x_s_info)
            
            print(f"提取信息成功:")
            print(f"  x_s_info keys: {list(x_s_info.keys())}")
            print(f"  exp shape: {x_s_info['exp'].shape}")
            print(f"  f_s shape: {f_s.shape}")
            print(f"  img shape: {img_rgb.shape}")
            
            return {
                'x_s_info': x_s_info,
                'f_s': f_s,
                'img_rgb': img_rgb,
                'M_c2o': M_c2o,
                'source_info': source_info
            }
        else:
            raise ValueError("无法提取帧信息")
    
    def fix_exp_shape(self, x_info):
        """修复/归一化 exp 形状为扁平 (1, 63)。
        motion_stitch 在内部使用系数 a1/a2/a3，它们被 reshape 为 (1,63)，
        因此在进入 motion_stitch 前将 exp 统一为 (1,63) 可以避免广播错误。
        """
        if 'exp' in x_info:
            exp = np.array(x_info['exp'])
            try:
                print(f"原始exp形状: {exp.shape}")
            except Exception:
                pass

            # 如果是 (1,21,3) -> 扁平为 (1,63)
            if exp.ndim == 3 and exp.shape[1] == 21 and exp.shape[2] == 3:
                exp = exp.reshape(1, 63)
            # 如果是 (1,63) 或 (bs,63) 保持为 (bs,63)，否则尝试扁平截取前63
            elif exp.ndim == 2 and exp.shape[1] == 63:
                pass
            elif exp.ndim == 1 and exp.shape[0] == 63:
                exp = exp.reshape(1, 63)
            else:
                print(f"警告: exp形状异常 {exp.shape}, 尝试修复...")
                try:
                    exp = exp.flatten()[:63].reshape(1, 63)
                except Exception:
                    raise ValueError(f"无法修复exp形状: {exp.shape}")

            x_info['exp'] = exp
            print(f"修复后exp形状: {exp.shape}")
    
    def render_single_frame(self, frame_info, x_d_info):
        """渲染单帧"""
        x_s_info = frame_info['x_s_info']
        f_s = frame_info['f_s']
        original_img = frame_info['img_rgb']
        M_c2o = frame_info['M_c2o']
        
        # 关键修复：motion_stitch 的 a1/a2 是 (1,63)，因此把 exp 也统一为扁平 (1,63)
        print(f"修复前: x_s_info exp shape = {x_s_info['exp'].shape}, x_d_info exp shape = {x_d_info['exp'].shape}")

        def ensure_flat63(exp):
            exp = np.array(exp)
            if exp.ndim == 3 and exp.shape[1] == 21 and exp.shape[2] == 3:
                return exp.reshape(1, 63)
            if exp.ndim == 2 and exp.shape[1] == 63:
                return exp
            if exp.ndim == 1 and exp.shape[0] == 63:
                return exp.reshape(1, 63)
            raise ValueError(f"无法处理的exp形状: {exp.shape}")

        x_s_info['exp'] = ensure_flat63(x_s_info['exp'])
        x_d_info['exp'] = ensure_flat63(x_d_info['exp'])

        print(f"修复后: x_s_info exp shape = {x_s_info['exp'].shape}, x_d_info exp shape = {x_d_info['exp'].shape}")
        
        # 设置motion_stitch - 简化参数避免问题
        try:
            # 启用 drive_eye=True，这样对 eye 相关维度的扰动才会生效（如右眼开合）
            self.motion_stitch.setup(
                N_d=1,
                use_d_keys=("exp",),  # 只使用exp
                relative_d=False,
                drive_eye=True,  # 启用 eye 驱动以让扰动生效
                flag_stitching=False,  # 先关闭stitching
                is_image_flag=True,
                x_s_info=x_s_info,
                d0=None,
            )
        except Exception as e:
            print(f"motion_stitch.setup 失败: {e}")
            # 尝试更简化的设置
            self.motion_stitch.setup(
                N_d=1,
                use_d_keys=("exp",),
                relative_d=False,
                drive_eye=False,
                flag_stitching=False,
                is_image_flag=True,
                x_s_info=x_s_info,
                d0=None,
                ch_info=None,
                overall_ctrl_info={}
            )
        
        # 获取关键点
        print("调用motion_stitch...")
        # 传入 delta_exp 保证在 motion_stitch 内部可能的合并/复写后，
        # 通过 ctrl_motion 将我们的表达扰动加回去，避免被 _fix_exp_for_x_d_info_v2 覆盖。
        try:
            delta_exp = x_d_info['exp'] - x_s_info['exp']
        except Exception:
            delta_exp = None

        if delta_exp is not None:
            x_s, x_d = self.motion_stitch(x_s_info, x_d_info, delta_exp=delta_exp)
        else:
            x_s, x_d = self.motion_stitch(x_s_info, x_d_info)
        print(f"x_s shape: {x_s.shape}, x_d shape: {x_d.shape}")
        
        # 渲染
        print("调用warp_f3d...")
        f_3d = self.warp_f3d(f_s, x_s, x_d)
        print(f"f_3d shape: {f_3d.shape}")
        
        print("调用decode_f3d...")
        render_img = self.decode_f3d(f_3d)
        print(f"render_img shape: {render_img.shape}")
        
        print("调用putback...")
        result_img = self.putback(original_img, render_img, M_c2o)
        print(f"result_img shape: {result_img.shape}")
        
        return result_img
    
    def create_perturbed_x_d_info(self, x_s_info, dimension, delta):
        """创建扰动后的x_d_info"""
        # 深度复制
        import copy
        x_d_info = copy.deepcopy(x_s_info)
        
        # 确保exp为扁平 (1,63)
        self.fix_exp_shape(x_d_info)
        exp = x_d_info['exp']

        # 为便于索引，转换为 (1,21,3)
        if exp.ndim == 2 and exp.shape[1] == 63:
            exp3 = exp.reshape(1, 21, 3).copy()
        else:
            exp3 = np.array(exp)

        # 计算关键点索引
        dim_idx = dimension - 1  # 0-based
        kp_idx = dim_idx // 3
        axis_idx = dim_idx % 3
        
        # 获取面部区域描述
        facial_regions = {
            1: "额头", 2: "左眉外侧", 3: "左眉内侧", 4: "右眉内侧", 5: "右眉外侧",
            6: "左眼外侧", 7: "左眼上睑", 8: "左眼下睑", 9: "左眼内侧", 10: "右眼内侧",
            11: "右下睑", 12: "右上睑", 13: "右眼外侧", 14: "鼻梁", 15: "鼻尖",
            16: "左鼻翼", 17: "右鼻翼", 18: "左上唇", 19: "右上唇", 20: "嘴中部",
            21: "下巴"
        }
        
        region = facial_regions.get(kp_idx + 1, f"关键点{kp_idx+1}")
        axis_name = ['x', 'y', 'z'][axis_idx]
        axis_desc = ['左右', '上下', '前后深度'][axis_idx]
        
        # 应用扰动
        # 对于眼睛相关的维度，使用与 motion_stitch._eye_delta 相同的缩放策略。
        # 这样外部传入的 delta 值（如 0.005, 0.5）对视觉变化的影响更直观。
        region_name = region
        original_value = exp3[0, kp_idx, axis_idx]

        if '眼' in region_name or '睑' in region_name:
            # 对眼区域也直接在对应通道上相加（与其它区域一致），
            # 这样 dimension/ delta 的语义对所有通道都是线性的：
            # new = original + delta
            new_value = original_value + delta
            exp3[0, kp_idx, axis_idx] = new_value
        else:
            # 非眼区域，直接在对应通道添加 delta
            new_value = original_value + delta
            exp3[0, kp_idx, axis_idx] = new_value
        
        print(f"\n扰动信息:")
        print(f"  维度: {dimension}")
        print(f"  关键点: KP{kp_idx+1} ({region})")
        print(f"  轴: {axis_name} ({axis_desc})")
        print(f"  原始值: {original_value:.6f}")
        print(f"  扰动值: {delta}")
        print(f"  新值: {new_value:.6f}")
        
        # 特别标注已知维度
        if dimension == 34:
            print(f"  预期效果: 控制右眼的睁开/闭合")
        elif dimension == 58:
            print(f"  预期效果: 控制嘴巴张开")
        
        # 存回扁平形式 (1,63)
        x_d_info['exp'] = exp3.reshape(1, 63)
        
        return x_d_info


def create_comparison_image(original_img, perturbed_img, title_info, output_path, save_diff=False):
    """创建对比图像"""
    # 确保图像是uint8
    if original_img.dtype != np.uint8:
        original_img = (np.clip(original_img, 0, 1) * 255).astype(np.uint8)
    if perturbed_img.dtype != np.uint8:
        perturbed_img = (np.clip(perturbed_img, 0, 1) * 255).astype(np.uint8)
    
    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # 左侧：原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # 右侧：扰动后图像
    axes[1].imshow(perturbed_img)
    
    # 添加扰动信息
    dim = title_info.get('dimension', 'N/A')
    delta = title_info.get('delta', 0)
    kp_idx = title_info.get('kp_idx', 'N/A')
    axis = title_info.get('axis', 'N/A')
    region = title_info.get('region', '')
    
    perturb_title = f'Perturbed Image\nDimension {dim}: KP{kp_idx} {axis}'
    if region:
        perturb_title += f'\n({region})'
    perturb_title += f'\nΔ = {delta}'
    
    axes[1].set_title(perturb_title, fontsize=14, fontweight='bold', pad=10)
    axes[1].axis('off')
    
    # 添加整体标题
    fig.suptitle(f'Expression Vector Perturbation Experiment', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 添加信息文本
    info_text = f"Dimension {dim}: Controls {region} {axis}-axis movement"
    if dim == 34:
        info_text += " (Right eye open/close)"
    elif dim == 58:
        info_text += " (Mouth open)"
    
    # 添加颜色框突出差异
    if original_img.shape == perturbed_img.shape:
        # 计算差异图
        diff = np.abs(original_img.astype(float) - perturbed_img.astype(float))
        diff_sum = np.sum(diff) / (original_img.size / 3)  # 平均每通道差异
        
        info_text += f"\nMean difference: {diff_sum:.2f}"
    
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.9)  # 调整边距
    
    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"对比图已保存: {output_path}")
    
    # 单独保存差异图（默认仅在差异明显时保存；当 save_diff=True 时强制保存）
    if 'diff' in locals():
        diff_path = output_path.replace('.png', '_diff.png')
        max_val = diff.max() if diff.max() > 0 else 1.0
        diff_normalized = (diff / max_val * 255).astype(np.uint8)
        if save_diff or diff_sum > 5:
            Image.fromarray(diff_normalized).save(diff_path)
            print(f"差异图已保存: {diff_path}")


def simple_render_test():
    """简单渲染测试：验证渲染管道是否工作"""
    print("\n" + "="*60)
    print("简单渲染测试")
    print("="*60)
    
    cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
    data_root = "./checkpoints/ditto_trt_Ampere_Plus"
    image_path = "./example/image.png"
    
    try:
        # 初始化
        cfg_list = parse_cfg(cfg_pkl, data_root, {})
        avatar_registrar_cfg = cfg_list[0]
        
        registrar = AvatarRegistrar(**avatar_registrar_cfg)
        
        # 提取图像
        crop_kwargs = {
            "crop_scale": 2.3,
            "crop_vx_ratio": 0,
            "crop_vy_ratio": -0.125,
            "crop_flag_do_rot": True,
        }
        
        source_info = registrar(
            image_path, 
            max_dim=512,
            n_frames=1, 
            **crop_kwargs,
        )
        
        print(f"提取成功!")
        print(f"Keys: {list(source_info.keys())}")
        
        if "x_s_info_lst" in source_info:
            x_s_info = source_info["x_s_info_lst"][0]
            print(f"\nx_s_info keys: {list(x_s_info.keys())}")
            
            for key in ['kp', 'exp', 'pitch', 'yaw', 'roll', 't', 'scale']:
                if key in x_s_info:
                    val = x_s_info[key]
                    if hasattr(val, 'shape'):
                        print(f"  {key}: shape={val.shape}")
                    else:
                        print(f"  {key}: {val}")
            
            # 测试transform_keypoint
            print("\n测试transform_keypoint:")
            kps = transform_keypoint(x_s_info)
            print(f"  关键点形状: {kps.shape}")
            print(f"  关键点范围: x[{kps[0,:,0].min():.2f}, {kps[0,:,0].max():.2f}], "
                  f"y[{kps[0,:,1].min():.2f}, {kps[0,:,1].max():.2f}]")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="修复版单帧扰动对比实验")
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--cfg_pkl', required=True, help='配置文件路径')
    parser.add_argument('--data_root', required=True, help='模型数据根目录')
    parser.add_argument('--dimension', type=int, default=34, help='扰动维度 (1-63)')
    parser.add_argument('--delta', type=float, default=1.0, help='扰动幅度')
    parser.add_argument('--batch_all', action='store_true', help='对1..63所有维度分别生成扰动图（使用 --batch_delta）')
    parser.add_argument('--batch_delta', type=float, default=-0.02, help='批量扰动时每个维度使用的 delta 值')
    parser.add_argument('--batch_deltas', type=str, default='',
                        help='批量模式下使用的逗号分隔 delta 列表，例如 "-0.2,-0.1,-0.05". 如果提供，将覆盖 --batch_delta')
    parser.add_argument('--output_dir', default='./outputs/perturb_images', help='输出目录')
    parser.add_argument('--simple_test', action='store_true', help='运行简单测试')
    args = parser.parse_args()
    
    if args.simple_test:
        simple_render_test()
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("修复版单帧扰动对比实验")
    print(f"输入图像: {args.image}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    # 先运行简单测试
    print("\n[1/4] 运行基础测试...")
    if not simple_render_test():
        print("基础测试失败，请检查配置")
        return
    
    # 初始化渲染器
    print("\n[2/4] 初始化渲染器...")
    try:
        renderer = SingleFramePerturbRenderer(args.cfg_pkl, args.data_root)
    except Exception as e:
        print(f"初始化渲染器失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 提取帧信息
    print("\n[3/4] 提取图像信息...")
    try:
        frame_info = renderer.extract_frame_info(args.image)
        x_s_info = frame_info['x_s_info']
    except Exception as e:
        print(f"提取图像信息失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 进行扰动实验
    print("\n[4/4] 进行扰动实验...")

    # 如果批量标志被设置，则对 1..63 的每个维度分别应用相同的 delta，生成 63 张扰动图
    if args.batch_all:
        # 解析 batch_deltas 优先于单一 batch_delta
        if args.batch_deltas:
            try:
                batch_deltas = [float(x.strip()) for x in args.batch_deltas.split(',') if x.strip()]
            except Exception:
                print(f"无法解析 --batch_deltas: {args.batch_deltas}. 使用 --batch_delta {args.batch_delta} 作为回退。")
                batch_deltas = [args.batch_delta]
        else:
            batch_deltas = [args.batch_delta]

        print(f"批量模式: 对 1..63 每个维度分别应用 Δ 列表 {batch_deltas} 并生成图片")

        try:
            # 先渲染并保存一次原始图（后续重复使用）
            print("\n渲染原始图（仅一次）...")
            original_img = renderer.render_single_frame(frame_info, x_s_info)
            original_path = os.path.join(args.output_dir, f"original.png")
            Image.fromarray(original_img).save(original_path)

            # 对每个 delta 值循环处理
            for batch_delta in batch_deltas:
                print(f"\n开始处理 Δ={batch_delta} 的所有维度...")
                for dim in range(1, 64):
                    print(f"\n--- 处理维度 {dim} (Δ={batch_delta}) ---")
                    try:
                        x_d_info_pert = renderer.create_perturbed_x_d_info(x_s_info, dim, batch_delta)
                        perturbed_img = renderer.render_single_frame(frame_info, x_d_info_pert)

                        dim_idx = dim - 1
                        kp_idx = (dim_idx // 3) + 1
                        axis = ['x', 'y', 'z'][dim_idx % 3]

                        output_path = os.path.join(
                            args.output_dir,
                            f"perturb_dim{dim:03d}_kp{kp_idx}_{axis}_delta{batch_delta:.4f}.png"
                        )

                        title_info = {
                            'dimension': dim,
                            'delta': batch_delta,
                            'kp_idx': kp_idx,
                            'axis': axis,
                            'region': ''
                        }

                        create_comparison_image(original_img, perturbed_img, title_info, output_path, save_diff=True)

                        perturbed_path = os.path.join(args.output_dir, f"perturbed_dim{dim:03d}_delta{batch_delta:.4f}.png")
                        Image.fromarray(perturbed_img).save(perturbed_path)
                        print(f"已保存: {perturbed_path}")

                    except Exception as e:
                        print(f"维度 {dim} 处理失败: {e}")
                        import traceback
                        traceback.print_exc()
                        # 继续下一个维度
                        continue

            print("\n批量处理完成。所有输出保存到: {}".format(args.output_dir))
        except Exception as e:
            print(f"批量处理失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 批量完成后退出
        return

    # 否则按单一维度运行（保持原有行为）
    dim = args.dimension
    delta = args.delta
    
    print(f"\n{'='*40}")
    print(f"测试维度 {dim} (Δ={delta})")
    
    try:
        # 1. 渲染原始图像
        print("\n渲染原始图像...")
        original_img = renderer.render_single_frame(frame_info, x_s_info)
        
        # 2. 创建扰动后的x_d_info
        x_d_info_pert = renderer.create_perturbed_x_d_info(x_s_info, dim, delta)
        
        # 3. 渲染扰动后图像
        print("\n渲染扰动后图像...")
        perturbed_img = renderer.render_single_frame(frame_info, x_d_info_pert)
        
        # 4. 创建对比图
        dim_idx = dim - 1
        kp_idx = (dim_idx // 3) + 1
        axis = ['x', 'y', 'z'][dim_idx % 3]
        
        # 获取区域描述
        facial_regions = {
            1: "额头", 2: "左眉外侧", 3: "左眉内侧", 4: "右眉内侧", 5: "右眉外侧",
            6: "左眼外侧", 7: "左眼上睑", 8: "左眼下睑", 9: "左眼内侧", 10: "右眼内侧",
            11: "右下睑", 12: "右上睑", 13: "右眼外侧", 14: "鼻梁", 15: "鼻尖",
            16: "左鼻翼", 17: "右鼻翼", 18: "左上唇", 19: "右上唇", 20: "嘴中部",
            21: "下巴"
        }
        region = facial_regions.get(kp_idx, f"关键点{kp_idx}")
        
        output_path = os.path.join(
            args.output_dir, 
            f"perturb_dim{dim:03d}_kp{kp_idx}_{axis}_delta{delta}.png"
        )
        
        title_info = {
            'dimension': dim,
            'delta': delta,
            'kp_idx': kp_idx,
            'axis': axis,
            'region': region
        }
        
        create_comparison_image(original_img, perturbed_img, title_info, output_path, save_diff=True)
        
        # 5. 保存单独图像
        original_path = os.path.join(args.output_dir, f"original.png")
        perturbed_path = os.path.join(args.output_dir, f"perturbed_dim{dim}.png")
        
        Image.fromarray(original_img).save(original_path)
        Image.fromarray(perturbed_img).save(perturbed_path)
        
        print(f"\n原始图像: {original_path}")
        print(f"扰动图像: {perturbed_path}")
        print(f"对比图像: {output_path}")
        
    except Exception as e:
        print(f"测试维度 {dim} 失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"所有输出文件保存在: {args.output_dir}")


if __name__ == '__main__':
    # 先运行简单测试
    # python single_frame_perturb_fixed.py --simple_test
    
    # 然后运行实际实验
    # python single_frame_perturb_fixed.py \
    #   --image ./example/image.png \
    #   --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \
    #   --data_root ./checkpoints/ditto_trt_Ampere_Plus \
    #   --dimension 34 \
    #   --delta 2.0
    
    main()