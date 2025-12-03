#!/usr/bin/env python3
"""
修改版 inference.py：支持exp扰动实验
"""
import librosa
import math
import os
import numpy as np
import random
import torch
import pickle
import argparse

from stream_pipeline_offline import StreamSDK


def perturb_exp_vector(exp_vector, dimension, delta):
    """
    扰动exp向量中的特定维度
    
    Args:
        exp_vector: 形状为 (1, 21, 3) 或 (63,)
        dimension: 1-63
        delta: 扰动值
    """
    if exp_vector.ndim == 1 and exp_vector.shape[0] == 63:
        exp_vector = exp_vector.reshape(1, 21, 3)
    elif exp_vector.ndim == 2 and exp_vector.shape[1] == 63:
        exp_vector = exp_vector.reshape(1, 21, 3)
    
    dim_idx = dimension - 1  # 0-based
    kp_idx = dim_idx // 3
    axis_idx = dim_idx % 3
    
    print(f"扰动维度 {dimension}: KP{kp_idx+1} {['x','y','z'][axis_idx]}")
    print(f"原始值: {exp_vector[0, kp_idx, axis_idx]:.6f} → {exp_vector[0, kp_idx, axis_idx] + delta:.6f}")
    
    # 应用扰动
    exp_vector[0, kp_idx, axis_idx] += delta
    
    return exp_vector


def run_with_perturbation(SDK, audio_path, source_path, output_path, 
                          dimension, delta, more_kwargs=None):
    """
    运行推理，但在中间注入exp扰动
    """
    if more_kwargs is None:
        more_kwargs = {}
    
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})
    
    SDK.setup(source_path, output_path, **setup_kwargs)
    
    # 加载音频
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    
    # 设置参数
    SDK.setup_Nd(N_d=num_f)
    
    # 关键：我们需要在生成过程中修改exp
    # 由于SDK架构，我们需要修改 motion_stitch 的行为
    
    print("\n=== 扰动实验 ===")
    print(f"将在运动生成过程中扰动维度 {dimension}")
    print(f"扰动值: {delta}")
    
    # 这里需要修改SDK内部的x_d_info
    # 由于架构限制，这需要修改 motion_stitch.py
    
    # 临时方案：通过ctrl_info传递扰动
    ctrl_info = run_kwargs.get("ctrl_info", {})
    
    # 计算扰动向量
    delta_exp = np.zeros(63, dtype=np.float32)
    delta_exp[dimension-1] = delta
    
    # 添加到所有帧
    for i in range(num_f):
        if i not in ctrl_info:
            ctrl_info[i] = {}
        ctrl_info[i]["delta_exp"] = delta_exp.reshape(1, 21, 3)
    
    run_kwargs["ctrl_info"] = ctrl_info
    
    # 运行
    if SDK.online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    
    SDK.close()
    
    # 添加音频
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)
    
    print(f"生成完成: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dimension", type=int, default=34, help="扰动维度 (1-63)")
    parser.add_argument("--delta", type=float, default=1.0, help="扰动幅度")
    parser.add_argument("--test_mode", action="store_true", help="测试模式：生成多个delta值")
    args = parser.parse_args()
    
    # 初始化SDK
    SDK = StreamSDK(args.cfg_pkl, args.data_root)
    
    if args.test_mode:
        # 测试多个delta值
        test_deltas = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for delta in test_deltas:
            output_path = args.output_path.replace('.mp4', f'_dim{args.dimension}_delta{delta}.mp4')
            print(f"\n测试 delta={delta}")
            
            run_with_perturbation(
                SDK, 
                args.audio_path, 
                args.source_path, 
                output_path,
                args.dimension,
                delta
            )
            
            # 需要重新初始化SDK
            SDK = StreamSDK(args.cfg_pkl, args.data_root)
    else:
        run_with_perturbation(
            SDK,
            args.audio_path,
            args.source_path,
            args.output_path,
            args.dimension,
            args.delta
        )


if __name__ == "__main__":
    main()