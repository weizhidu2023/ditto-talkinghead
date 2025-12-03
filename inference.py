import librosa
import math
import os
import numpy as np
import random
import torch
import pickle

from stream_pipeline_offline import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    SDK.setup(source_path, output_path, **setup_kwargs)

    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    online_mode = SDK.online_mode
    if online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    SDK.close()

    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)

    print(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus", help="path to trt data_root")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", help="path to cfg_pkl")

    parser.add_argument("--audio_path", type=str, help="path to input wav")
    parser.add_argument("--source_path", type=str, help="path to input image")
    parser.add_argument("--output_path", type=str, help="path to output mp4")
    parser.add_argument("--compare_gaze", action="store_true", help="生成有/无 gaze control 的对比输出")
    parser.add_argument("--gaze_amp_yaw", type=float, default=15.0, help="头部 yaw 振幅 (度)，用以制造明显头部运动以便比较")
    parser.add_argument("--gaze_amp_pitch", type=float, default=5.0, help="头部 pitch 振幅 (度)")
    parser.add_argument("--gaze_freq", type=float, default=0.4, help="头部运动频率 (Hz)")
    parser.add_argument("--gaze_comp_sign", type=float, default=1.0, help="gaze 补偿符号，设置为 -1 或 1 来翻转补偿方向")
    parser.add_argument("--dump_kp_dir", type=str, default=None, help="(optional) directory to dump per-frame kp_info NPZ files")
    args = parser.parse_args()

    # init sdk
    data_root = args.data_root   # model dir
    cfg_pkl = args.cfg_pkl     # cfg pkl
    # Note: when comparing gaze we will create separate SDK instances per run
    SDK = StreamSDK(cfg_pkl, data_root)

    # input args
    audio_path = args.audio_path    # .wav
    source_path = args.source_path   # video|image
    output_path = args.output_path   # .mp4

    # run
    # seed_everything(1024)
    if not args.compare_gaze:
        more_kwargs = {}
        if args.dump_kp_dir:
            more_kwargs = {"setup_kwargs": {"dump_kp_dir": args.dump_kp_dir}}
        run(SDK, audio_path, source_path, output_path, more_kwargs)
    else:
        # load audio to compute frame count (same logic as in run)
        audio, sr = librosa.core.load(audio_path, sr=16000)
        num_f = math.ceil(len(audio) / 16000 * 25)

        # build head motion ctrl_info: per-frame sinusoidal yaw/pitch offsets in degrees
        amp_yaw = args.gaze_amp_yaw  # degrees
        amp_pitch = args.gaze_amp_pitch  # degrees
        freq = args.gaze_freq
        ctrl_info = {}
        for i in range(num_f):
            t = i / 25.0
            yaw = amp_yaw * math.sin(2 * math.pi * freq * t)
            pitch = amp_pitch * math.sin(2 * math.pi * freq * t + math.pi / 6)
            ctrl_info[i] = {"delta_yaw": float(yaw), "delta_pitch": float(pitch)}

        # no-gaze output
        out_nogaze = os.path.splitext(args.output_path)[0] + "_no_gaze.mp4"
        SDK1 = StreamSDK(cfg_pkl, data_root)
        more_kwargs_nogaze = {
            "setup_kwargs": {"overall_ctrl_info": {"gaze_fixed": False, "gaze_comp_sign": args.gaze_comp_sign}},
            "run_kwargs": {"ctrl_info": ctrl_info}
        }
        if args.dump_kp_dir:
            more_kwargs_nogaze.setdefault('setup_kwargs', {})['dump_kp_dir'] = os.path.join(args.dump_kp_dir, 'nogaze')
        run(SDK1, audio_path, source_path, out_nogaze, more_kwargs_nogaze)

        # gaze output
        out_gaze = os.path.splitext(args.output_path)[0] + "_gaze.mp4"
        SDK2 = StreamSDK(cfg_pkl, data_root)
        more_kwargs_gaze = {
            "setup_kwargs": {"overall_ctrl_info": {"gaze_fixed": True, "gaze_comp_sign": args.gaze_comp_sign}},
            "run_kwargs": {"ctrl_info": ctrl_info}
        }
        if args.dump_kp_dir:
            more_kwargs_gaze.setdefault('setup_kwargs', {})['dump_kp_dir'] = os.path.join(args.dump_kp_dir, 'gaze')
        run(SDK2, audio_path, source_path, out_gaze, more_kwargs_gaze)
