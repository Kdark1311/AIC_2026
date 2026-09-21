#!/usr/bin/env python3
"""
Cắt keyframe từ video bằng OmniShotCut — cùng cách cắt với hệ thống cũ.

Quy tắc (giữ nguyên như hệ thống cũ):
  * Model OmniShotCut v1.5 (uva-cv-lab/OmniShotCut_v1.5), mode "clean_shot",
    overlap giữa các cửa sổ suy luận = 20 frame.
  * Mỗi shot [start, end] lấy 3 keyframe: đầu (start), giữa ((start+end)//2), cuối (end).
    Shot quá ngắn thì các frame trùng nhau được gộp lại.
  * "clean_shot" bỏ các đoạn chuyển cảnh (dissolve, wipe, ...) nên giữa hai shot
    có thể có khoảng hở không có keyframe — đúng như hệ thống cũ.
  * Tên file = số thứ tự frame, 6 chữ số: 000123.webp.

Đầu ra cho mỗi video <VIDEO_ID>:
  <output_dir>/keyframes/<LOT>/<VIDEO_ID>/<frame_id:06d>.webp
  <output_dir>/map-keyframes/<VIDEO_ID>.csv     cột: frame_id,fps,timestamp
  <output_dir>/shots/<VIDEO_ID>.json            danh sách shot (để kiểm tra / debug)

  timestamp (giây) = frame_id / fps.

Chạy lại lệnh sẽ bỏ qua các video đã có file CSV (CSV được ghi cuối cùng,
nên video nào đã có CSV là đã xong trọn vẹn).
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".flv"}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config():
    p = argparse.ArgumentParser(description="Cắt keyframe từ video bằng OmniShotCut")
    p.add_argument("--config", default=str(HERE / "config.yaml"), help="đường dẫn file config.yaml")
    p.add_argument("--video_dir", help="ghi đè video_dir trong config")
    p.add_argument("--output_dir", help="ghi đè output_dir trong config")
    p.add_argument("--gpu", type=int, help="chỉ số GPU dùng (mặc định 0)")
    p.add_argument("--num_shards", type=int, default=1, help="chia danh sách video thành N phần (chạy song song nhiều GPU)")
    p.add_argument("--shard", type=int, default=0, help="phần thứ mấy (0..num_shards-1)")
    p.add_argument("--limit", type=int, help="chỉ chạy N video đầu (để thử)")
    p.add_argument("--overwrite", action="store_true", help="chạy lại cả video đã xong")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("video_dir", "output_dir", "gpu"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    cfg.setdefault("gpu", 0)
    cfg.setdefault("checkpoint", "uva-cv-lab/OmniShotCut_v1.5")
    cfg.setdefault("mode", "clean_shot")
    cfg.setdefault("overlap", 20)
    cfg.setdefault("image_format", "webp")
    cfg.setdefault("image_quality", 80)
    cfg.setdefault("group_by_lot", True)

    for key in ("video_dir", "output_dir"):
        if not cfg.get(key):
            sys.exit(f"[lỗi] Chưa khai báo '{key}' trong {args.config}")
        # Đường dẫn tương đối được tính từ thư mục chứa config
        path = Path(os.path.expanduser(str(cfg[key])))
        if not path.is_absolute():
            path = (Path(args.config).resolve().parent / path).resolve()
        cfg[key] = path

    if cfg["image_format"] not in ("webp", "jpg", "png"):
        sys.exit("[lỗi] image_format phải là webp | jpg | png")
    if cfg["mode"] not in ("clean_shot", "default"):
        sys.exit("[lỗi] mode phải là clean_shot | default")
    return args, cfg


# --------------------------------------------------------------------------- #
# ffmpeg: dùng ffmpeg của hệ thống nếu có, không thì dùng bản đi kèm imageio-ffmpeg
# --------------------------------------------------------------------------- #
def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_small(video_path, width, height):
    """Giải mã toàn bộ video ở độ phân giải xử lý của model (vd 128x96), RGB.
    Tham số giống hệt omnishotcut.datasets.utils._decode_video để đánh số frame
    và kết quả cắt khớp với hệ thống cũ.
    """
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(video_path),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-vsync", "passthrough", "pipe:"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg không giải mã được {video_path}:\n{proc.stderr.decode('utf-8', 'ignore')}")
    frames = np.frombuffer(proc.stdout, np.uint8).reshape(-1, height, width, 3)
    if len(frames) == 0:
        raise RuntimeError(f"giải mã được 0 frame: {video_path}")
    return frames


def video_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise RuntimeError(f"không đọc được fps của {video_path}")
    return float(fps)


# --------------------------------------------------------------------------- #
# Shot -> keyframe
# --------------------------------------------------------------------------- #
def ranges_to_shots(ranges):
    """OmniShotCut trả về [start, end) (end không bao gồm) -> shot với end_frame bao gồm."""
    shots = []
    for start, end in ranges:
        start, last = int(start), int(end) - 1
        if last < start:
            continue
        shots.append({
            "shot_index": len(shots) + 1,
            "start_frame": start,
            "middle_frame": (start + last) // 2,
            "end_frame": last,
        })
    return shots


def shots_to_keyframes(shots):
    frames = set()
    for s in shots:
        frames.update((s["start_frame"], s["middle_frame"], s["end_frame"]))
    return sorted(frames)


def save_frames(video_path, frame_ids, out_dir, fmt, quality):
    """Đọc tuần tự video (không seek, nên chính xác từng frame) và lưu các frame cần."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(frame_ids)
    last = max(frame_ids)
    if fmt == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    elif fmt == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    else:
        params = []

    cap = cv2.VideoCapture(str(video_path))
    idx, saved = 0, 0
    while idx <= last:
        if not cap.grab():
            break
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok:
                cv2.imwrite(str(out_dir / f"{idx:06d}.{fmt}"), frame, params)
                saved += 1
        idx += 1
    cap.release()
    return saved


def write_metadata_csv(path, frame_ids, fps):
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "fps", "timestamp"])
        for fid in frame_ids:
            w.writerow([fid, round(fps, 4), round(fid / fps, 4)])
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def lot_of(video_id):
    m = re.match(r"^(L\d+)_", video_id)
    return m.group(1) if m else None


def list_videos(video_dir):
    videos = sorted(p for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    seen = {}
    for v in videos:
        if v.stem in seen:
            sys.exit(f"[lỗi] Trùng tên video: {seen[v.stem]} và {v}")
        seen[v.stem] = v
    return videos


def main():
    args, cfg = load_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    import torch
    if not torch.cuda.is_available():
        sys.exit("[lỗi] Không thấy GPU CUDA. OmniShotCut bắt buộc chạy trên GPU NVIDIA.")
    import omnishotcut

    video_dir, out_root = cfg["video_dir"], cfg["output_dir"]
    if not video_dir.is_dir():
        sys.exit(f"[lỗi] Không tìm thấy thư mục video: {video_dir}")
    kf_root = out_root / "keyframes"
    meta_root = out_root / "map-keyframes"
    shots_root = out_root / "shots"
    for d in (kf_root, meta_root, shots_root):
        d.mkdir(parents=True, exist_ok=True)

    videos = list_videos(video_dir)
    videos = videos[args.shard::args.num_shards]
    todo = [v for v in videos if args.overwrite or not (meta_root / f"{v.stem}.csv").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Video: {video_dir}\nOutput: {out_root}\n"
          f"{len(videos)} video (shard {args.shard}/{args.num_shards}), cần xử lý {len(todo)}")
    if not todo:
        return

    model = omnishotcut.load(cfg["checkpoint"])
    margs = model._model_args
    width, height = margs.process_width, margs.process_height

    def finish(video_path, fps, shots, keyframes):
        """Chạy trên thread riêng để GPU xử lý video kế tiếp trong lúc lưu ảnh."""
        vid = video_path.stem
        lot = lot_of(vid) if cfg["group_by_lot"] else None
        out_dir = kf_root / lot / vid if lot else kf_root / vid
        if args.overwrite and out_dir.exists():
            shutil.rmtree(out_dir)
        saved = save_frames(video_path, keyframes, out_dir, cfg["image_format"], cfg["image_quality"])
        if saved != len(keyframes):
            print(f"  [cảnh báo] {vid}: cần {len(keyframes)} frame nhưng chỉ lưu được {saved}", flush=True)
            keyframes = sorted(int(p.stem) for p in out_dir.glob(f"*.{cfg['image_format']}"))
        with open(shots_root / f"{vid}.json", "w", encoding="utf-8") as f:
            json.dump({"video": vid, "source": str(video_path), "fps": fps,
                       "checkpoint": cfg["checkpoint"], "mode": cfg["mode"], "overlap": cfg["overlap"],
                       "shots": shots}, f, ensure_ascii=False)
        write_metadata_csv(meta_root / f"{vid}.csv", keyframes, fps)
        return vid, len(shots), len(keyframes)

    t0 = time.time()
    pending = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, video_path in enumerate(todo, 1):
            vid = video_path.stem
            try:
                fps = video_fps(video_path)
                frames = decode_small(video_path, width, height)
                result = model.inference(frames, mode=cfg["mode"], overlap=cfg["overlap"])
                ranges = result if cfg["mode"] == "clean_shot" else result[0]
                del frames
            except Exception as e:
                print(f"[{i}/{len(todo)}] {vid}: LỖI {e}", flush=True)
                continue
            shots = ranges_to_shots(ranges)
            keyframes = shots_to_keyframes(shots)
            if not keyframes:
                print(f"[{i}/{len(todo)}] {vid}: không có shot nào", flush=True)
                continue
            pending.append((i, pool.submit(finish, video_path, fps, shots, keyframes)))
            # Giữ tối đa 2 video đang lưu ảnh để không tốn RAM
            while len(pending) >= 2:
                report(pending.pop(0), len(todo), t0)
        while pending:
            report(pending.pop(0), len(todo), t0)
    print(f"Xong. Tổng thời gian {(time.time() - t0) / 60:.1f} phút")


def report(item, total, t0):
    i, fut = item
    try:
        vid, n_shots, n_kf = fut.result()
        print(f"[{i}/{total}] {vid}: {n_shots} shot -> {n_kf} keyframe "
              f"({(time.time() - t0) / 60:.1f} phút)", flush=True)
    except Exception as e:
        print(f"[{i}/{total}] LỖI khi lưu ảnh: {e}", flush=True)


if __name__ == "__main__":
    main()
