"""Sample frames from split-organised videos for ball-detection labeling.

Expects:   <root>/train/*.MOV   <root>/val/*.MOV   <root>/test/*.MOV
Writes:    <out>/train/<video>_f000120.jpg ...   (same split as the source video)
           <out>/sampling_manifest.csv           (video, split, frame, reason, file)

Base sample = every Nth frame. Extra "targeted" frames (bottom of dribble, blur,
occlusion, ...) can be added with --extra extra.csv (columns: video,frame,reason),
where `video` is the file name or stem and `frame` is the ORIGINAL frame index.
Extra frames are ignored for the test split, so test scores stay unbiased.

Example:
    python sample_frames.py videos --out frames --every 8
    python sample_frames.py videos --out frames --every 8 --extra extra.csv
"""

import argparse
import csv
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
SPLITS = ["train", "val", "test"]
MIN_SPACING = 4  # drop extra frames closer than this to an already-chosen frame


def load_extra(path):
    extra = {}
    if not path:
        return extra
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = Path(row["video"]).stem
            extra.setdefault(key, []).append((int(row["frame"]), row["reason"].strip() or "extra"))
    return extra


def choose_frames(total, every, extras):
    """Return {frame_index: reason}. Base frames win; extras too close to a chosen frame are dropped."""
    chosen = {i: "base" for i in range(0, total, every)}
    for frame, reason in sorted(extras):
        if frame in chosen or frame >= total:
            continue
        if any(abs(frame - c) < MIN_SPACING for c in chosen):
            continue
        chosen[frame] = reason
    return chosen


def resize(frame, max_side):
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if max_side <= 0 or scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("root", help="folder containing train/ val/ test/ video folders")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--every", type=int, default=8, help="base sample: every Nth frame")
    ap.add_argument("--max-side", type=int, default=1920, help="downscale longest side (0 = keep full size)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap.add_argument("--extra", help="CSV of targeted frames: video,frame,reason")
    ap.add_argument("--meta", help="optional CSV: video,player,setting,lighting,ball,move (any columns) for the balance summary")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    extra = load_extra(args.extra)
    manifest = []

    for split in SPLITS:
        vids = sorted(v for v in (root / split).glob("*") if v.suffix.lower() in VIDEO_EXTS)
        if not vids:
            print(f"warning: no videos in {root / split}")
            continue
        (out / split).mkdir(parents=True, exist_ok=True)
        for v in vids:
            cap = cv2.VideoCapture(str(v))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            extras = [] if split == "test" else extra.get(v.stem, [])
            chosen = choose_frames(total, args.every, extras)
            saved = 0
            for idx in range(total):
                ok, frame = cap.read()  # sequential read: seeking is unreliable on phone video
                if not ok:
                    break
                if idx not in chosen:
                    continue
                name = f"{v.stem}_f{idx:06d}.jpg"
                cv2.imwrite(
                    str(out / split / name),
                    resize(frame, args.max_side),
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality],
                )
                manifest.append([v.name, split, idx, chosen[idx], f"{split}/{name}"])
                saved += 1
            cap.release()
            print(f"{split}/{v.name}: {saved} frames (of {total})")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sampling_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "split", "frame", "reason", "file"])
        w.writerows(manifest)
    print(f"\n{len(manifest)} frames -> {out}  (manifest: {out / 'sampling_manifest.csv'})")
    print_summary(manifest, args.meta)


def print_summary(manifest, meta_path):
    """Balance check: frames per split, per video, per reason, and (with --meta) per attribute."""
    from collections import Counter

    print("\nFrames per split:")
    by_split = Counter(r[1] for r in manifest)
    for s in SPLITS:
        print(f"  {s:<6}{by_split.get(s, 0):>6}")

    print("\nFrames per video (split, video, total, extras):")
    per_video = Counter((r[1], r[0]) for r in manifest)
    extras = Counter((r[1], r[0]) for r in manifest if r[3] != "base")
    for (split, video), n in sorted(per_video.items(), key=lambda kv: (SPLITS.index(kv[0][0]), kv[0][1])):
        print(f"  {split:<6}{video:<22}{n:>5}{extras.get((split, video), 0):>5}")

    print("\nFrames per reason:")
    for reason, n in Counter(r[3] for r in manifest).most_common():
        print(f"  {reason:<12}{n:>6}")

    if not meta_path:
        return
    # meta CSV: video,<any attribute columns, e.g. player,setting,lighting,ball,move>
    with open(meta_path, newline="") as f:
        meta = {Path(row["video"]).stem: row for row in csv.DictReader(f)}
    attrs = [c for c in next(iter(meta.values())).keys() if c != "video"] if meta else []
    total = max(len(manifest), 1)
    for attr in attrs:
        counts = Counter(meta.get(Path(r[0]).stem, {}).get(attr, "(missing)") for r in manifest)
        print(f"\nFrames per {attr}:")
        for value, n in counts.most_common():
            flag = "  <-- under 15%" if n / total < 0.15 and value != "(missing)" else ""
            print(f"  {value:<20}{n:>6}  {100 * n / total:4.0f}%{flag}")


if __name__ == "__main__":
    main()
