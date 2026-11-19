"""Benchmark a pretrained Ultralytics YOLO model for basketball tracking.

Runs detection + ByteTrack on video files, filtered to COCO class 32 ("sports ball").
For each video it writes:
  - an annotated video (ball box, track ID, confidence)
  - a per-frame CSV (video, frame, timestamp, track_id, conf, box)
and for the whole run a summary CSV with detection rate, track-ID count and longest gap.

Example:
    python ball_benchmark.py pre_train_videos --model yolo11m.pt
    python ball_benchmark.py clip1.mp4 clip2.mp4 --model yolo11n.pt --imgsz 960
"""

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

SPORTS_BALL = 32  # COCO class index
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def collect_videos(inputs):
    videos = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            videos += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS)
        elif p.is_file():
            videos.append(p)
        else:
            print(f"warning: {p} not found, skipping")
    return videos


def process_video(model, video, args, out_dir):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(
        str(out_dir / f"{video.stem}_annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / args.stride,
        (width, height),
    )

    rows = []
    frames_total = 0
    frames_with_ball = 0
    ids_seen = set()
    gap = longest_gap = 0

    results = model.track(
        source=str(video),
        stream=True,
        persist=True,
        tracker=args.tracker,
        classes=[SPORTS_BALL],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        vid_stride=args.stride,
        verbose=False,
    )

    for i, r in enumerate(results):
        frame_idx = i * args.stride  # index in the original video
        frames_total += 1
        writer.write(r.plot())  # draws boxes, track IDs and confidences

        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            frames_with_ball += 1
            gap = 0
            ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
            for tid, conf, xyxy in zip(ids, boxes.conf.tolist(), boxes.xyxy.tolist()):
                if tid is not None:
                    ids_seen.add(tid)
                rows.append(
                    [video.name, frame_idx, round(frame_idx / fps, 4), tid, round(conf, 4)]
                    + [round(v, 1) for v in xyxy]
                )
        else:
            gap += 1
            longest_gap = max(longest_gap, gap)

    writer.release()

    with open(out_dir / f"{video.stem}_detections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "timestamp_s", "track_id", "conf", "x1", "y1", "x2", "y2"])
        w.writerows(rows)

    return {
        "video": video.name,
        "frames": frames_total,
        "fps": round(fps, 2),
        "frames_with_ball": frames_with_ball,
        "detection_rate": round(frames_with_ball / max(frames_total, 1), 3),
        "distinct_track_ids": len(ids_seen),
        "longest_gap_frames": longest_gap,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="video files and/or folders of videos")
    ap.add_argument("--model", default="yolo11m.pt", help="pretrained weights (auto-downloaded)")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size; larger helps small balls")
    ap.add_argument("--conf", type=float, default=0.15, help="low on purpose, to see misses vs false hits")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (CPU speed-up)")
    ap.add_argument("--device", default=None, help="e.g. cpu, 0 (default: auto)")
    ap.add_argument("--out", default="runs/ball_benchmark", help="output root")
    args = ap.parse_args()

    videos = collect_videos(args.inputs)
    if not videos:
        raise SystemExit("no videos found")

    out_dir = Path(args.out) / f"{Path(args.model).stem}_imgsz{args.imgsz}_conf{args.conf}_stride{args.stride}"
    if Path(args.tracker).stem != "bytetrack":  # keep default-tracker folder names unchanged
        out_dir = out_dir.with_name(out_dir.name + f"_{Path(args.tracker).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    summary = []
    for v in videos:
        print(f"[{len(summary) + 1}/{len(videos)}] {v}")
        s = process_video(model, v, args, out_dir)
        print(f"    {s}")
        summary.append(s)
        model.predictor = None  # reset tracker state between videos

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nresults in {out_dir}")


if __name__ == "__main__":
    main()
