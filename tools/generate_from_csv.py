#!/usr/bin/env python3
"""
根据 CSV 批量生成扰动图像（使用 single_frame_perturb.py 中的渲染器）

CSV 规则：每行至少三列，第二列为维度（int），第三列为扰动值（float）。
示例 CSV 行（逗号分隔）：
img1.png,34,0.2

用法示例：
python scripts/generate_from_csv.py \
  --csv inputs/perturb_list.csv \
  --image ./example/image.png \
  --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \
  --data_root ./checkpoints/ditto_trt_Ampere_Plus \
  --output_dir ./outputs/perturb_csv
"""
import argparse
import csv
import os
import sys
from PIL import Image

# 确保能 import repo 内模块
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from single_frame_perturb import SingleFramePerturbRenderer, create_comparison_image


def parse_csv(csv_path, skip_header=False):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for i, r in enumerate(reader):
            if i == 0 and skip_header:
                continue
            if not r or len(r) < 3:
                continue
            # 默认第二列 index 1 是 dimension，第三列 index 2 是 delta
            try:
                dim = int(r[1].strip())
                delta = float(r[2].strip())
                rows.append((dim, delta))
            except Exception:
                # 跳过无法解析的行
                continue
    return rows


def main():
    parser = argparse.ArgumentParser(description="从 CSV 生成扰动图像（批量）")
    parser.add_argument('--csv', required=True, help='CSV 文件路径（第2列为维度，第3列为扰动值）')
    parser.add_argument('--image', required=True, help='输入图像路径（用于提取源）')
    parser.add_argument('--cfg_pkl', required=True, help='配置 pkl 路径')
    parser.add_argument('--data_root', required=True, help='模型数据根目录')
    parser.add_argument('--output_dir', default='./outputs/perturb_csv', help='输出目录')
    parser.add_argument('--skip_header', action='store_true', help='CSV 包含表头，跳过第一行')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"读取 CSV: {args.csv}")
    rows = parse_csv(args.csv, skip_header=args.skip_header)
    if not rows:
        print("未解析到有效行，请检查 CSV 格式（第二列=维度，第三列=扰动值）。")
        return

    print(f"解析到 {len(rows)} 条任务。初始化渲染器...")

    try:
        renderer = SingleFramePerturbRenderer(args.cfg_pkl, args.data_root)
    except Exception as e:
        print(f"初始化渲染器失败: {e}")
        raise

    print("提取源帧信息...")
    frame_info = renderer.extract_frame_info(args.image)
    x_s_info = frame_info['x_s_info']

    print("渲染并保存原始图（仅一次）...")
    original_img = renderer.render_single_frame(frame_info, x_s_info)
    original_path = os.path.join(args.output_dir, 'original.png')
    Image.fromarray(original_img).save(original_path)
    print(f"保存: {original_path}")

    for idx, (dim, delta) in enumerate(rows, start=1):
        print(f"\n[{idx}/{len(rows)}] 维度 {dim}, Δ={delta}")
        try:
            x_d_info = renderer.create_perturbed_x_d_info(x_s_info, dim, delta)
            perturbed_img = renderer.render_single_frame(frame_info, x_d_info)

            dim_idx = dim - 1
            kp_idx = (dim_idx // 3) + 1
            axis = ['x', 'y', 'z'][dim_idx % 3]

            safe_delta = str(delta).replace('.', '_').replace('-', 'm')
            cmp_name = f"cmp_dim{dim:03d}_kp{kp_idx}_{axis}_delta{safe_delta}.png"
            cmp_path = os.path.join(args.output_dir, cmp_name)

            title_info = {
                'dimension': dim,
                'delta': delta,
                'kp_idx': kp_idx,
                'axis': axis,
                'region': ''
            }

            create_comparison_image(original_img, perturbed_img, title_info, cmp_path, save_diff=True)

            pert_name = f"perturbed_dim{dim:03d}_delta{safe_delta}.png"
            pert_path = os.path.join(args.output_dir, pert_name)
            Image.fromarray(perturbed_img).save(pert_path)
            print(f"已保存对比图: {cmp_path}")
            print(f"已保存扰动图: {pert_path}")

        except Exception as e:
            print(f"处理 维度 {dim} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print('\n全部完成。输出目录: ' + args.output_dir)


if __name__ == '__main__':
    main()
