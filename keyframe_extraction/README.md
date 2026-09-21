# Cắt keyframe bằng OmniShotCut

Cắt keyframe từ video theo đúng cách của hệ thống cũ, đồng thời tạo file metadata
(`frame_id, fps, timestamp`) cho từng video.

## Cách cắt

- Phát hiện shot bằng [OmniShotCut](https://github.com/UVA-Computer-Vision-Lab/OmniShotCut),
  checkpoint `uva-cv-lab/OmniShotCut_v1.5`, mode `clean_shot`, overlap 20 frame.
- Mỗi shot lấy **3 keyframe: đầu, giữa, cuối** (`giữa = (đầu + cuối) // 2`). Shot quá ngắn
  thì frame trùng được gộp.
- `clean_shot` bỏ các đoạn chuyển cảnh (dissolve, wipe, ...), nên giữa 2 shot có thể không có keyframe.
- Tên ảnh là số thứ tự frame 6 chữ số, ví dụ `000123.webp` = frame 123 của video.

## Yêu cầu

- GPU NVIDIA có CUDA (OmniShotCut chỉ chạy trên GPU).
- Python 3.9 trở lên.
- Không cần cài ffmpeg: nếu máy không có sẵn thì dùng bản đi kèm `imageio-ffmpeg`.

## Cách dùng

```bash
git clone https://github.com/Kdark1311/AIC_2026.git
cd AIC_2026/keyframe_extraction

./setup.sh            # Windows: setup.bat   (chỉ cần chạy 1 lần)
```

Mở `config.yaml` và sửa **2 đường dẫn**:

```yaml
video_dir: /duong/dan/toi/Videos     # thư mục chứa video, tìm cả trong thư mục con
output_dir: /duong/dan/toi/output    # thư mục chính để ghi keyframe + metadata
```

Chạy:

```bash
./run.sh              # Windows: run.bat
./run.sh --limit 1    # chạy thử 1 video trước
```

Lần chạy đầu sẽ tự tải checkpoint OmniShotCut từ HuggingFace.
Chạy lại lệnh thì các video đã xong được bỏ qua, nên dừng giữa chừng cũng không mất gì.

## Kết quả

```
output_dir/
├── keyframes/
│   └── L21/
│       └── L21_V001/
│           ├── 000000.webp
│           ├── 000020.webp
│           └── ...
├── map-keyframes/
│   └── L21_V001.csv          # mỗi video một file
└── shots/
    └── L21_V001.json         # danh sách shot (đầu/giữa/cuối), để kiểm tra
```

`map-keyframes/L21_V001.csv`:

```csv
frame_id,fps,timestamp
0,30.0,0.0
20,30.0,0.6667
40,30.0,1.3333
```

- `frame_id`: số thứ tự frame trong video, trùng tên file ảnh.
- `fps`: fps của video.
- `timestamp`: thời điểm tính bằng giây, `= frame_id / fps`.

Video tên dạng `Lxx_Vyyy` được gom theo lô (`keyframes/L21/L21_V001/`). Muốn để phẳng
(`keyframes/<video>/`) thì đặt `group_by_lot: false` trong `config.yaml`.

## Tuỳ chọn

| Tham số | Ý nghĩa |
|---|---|
| `--config file.yaml` | Dùng file config khác |
| `--video_dir`, `--output_dir` | Ghi đè đường dẫn trong config |
| `--gpu 1` | Chọn GPU |
| `--num_shards 2 --shard 0` | Chia video thành N phần để chạy song song, mỗi GPU một phần |
| `--limit N` | Chỉ chạy N video (để thử) |
| `--overwrite` | Chạy lại cả video đã xong |

Chạy 2 GPU song song:

```bash
./run.sh --gpu 0 --num_shards 2 --shard 0 &
./run.sh --gpu 1 --num_shards 2 --shard 1 &
```

Các tuỳ chọn trong `config.yaml` như `mode`, `overlap`, `checkpoint` nên để nguyên để cắt
giống hệ thống cũ. `image_format` (`webp`/`jpg`/`png`) và `image_quality` có thể đổi tuỳ ý.
