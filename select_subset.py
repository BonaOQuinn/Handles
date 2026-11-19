"""Pick a balanced subset of the sampled frames to label first.

Reads frames/sampling_manifest.csv and keeps at most N frames per video, so long clips
don't dominate. Within a video the choice is deterministic:
  1. targeted frames (reason != base) are always kept first,
  2. the rest of the budget is filled with base frames spaced evenly across the clip.

Writes <out>/<split>/*.jpg (copies, originals untouched) and <out>/labeling_subset.csv.

Example:
    python select_subset.py
    python select_subset.py --train-per-video 40 --val-per-video 20 --test-per-video 30
"""

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def even_pick(items, k):
    """k items spread evenly across a list (first and last included when k >= 2)."""
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    if k == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--frames", default="frames", help="folder with the sampled frames and manifest")
    ap.add_argument("--out", default=None, help="output folder (default: <frames>/to_label)")
    ap.add_argument("--train-per-video", type=int, default=30)
    ap.add_argument("--val-per-video", type=int, default=30)
    ap.add_argument("--test-per-video", type=int, default=30)
    ap.add_argument("--no-copy", action="store_true", help="only write the CSV, don't copy images")
    args = ap.parse_args()

    frames = Path(args.frames)
    out = Path(args.out) if args.out else frames / "to_label"
    caps = {"train": args.train_per_video, "val": args.val_per_video, "test": args.test_per_video}

    rows = list(csv.DictReader(open(frames / "sampling_manifest.csv", newline="")))
    by_video = defaultdict(list)
    for r in rows:
        by_video[(r["split"], r["video"])].append(r)

    chosen = []
    for (split, video), items in sorted(by_video.items()):
        items.sort(key=lambda r: int(r["frame"]))
        cap = caps[split]
        targeted = [r for r in items if r["reason"] != "base"]
        base = [r for r in items if r["reason"] == "base"]
        keep = targeted + even_pick(base, cap - len(targeted))
        chosen += sorted(keep, key=lambda r: int(r["frame"]))

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labeling_subset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(chosen)

    if not args.no_copy:
        for r in chosen:
            dst = out / r["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(frames / r["file"], dst)

    per_split = Counter(r["split"] for r in chosen)
    total_split = Counter(r["split"] for r in rows)
    print("Selected / available per split:")
    for s in ("train", "val", "test"):
        print(f"  {s:<6}{per_split.get(s, 0):>5} / {total_split.get(s, 0)}")
    print(f"  total {len(chosen):>5} / {len(rows)}")
    per_video = Counter((r["split"], r["video"]) for r in chosen)
    print(f"\nFrames per video: min {min(per_video.values())}, max {max(per_video.values())}, "
          f"videos {len(per_video)}")
    print(f"\nlist: {out / 'labeling_subset.csv'}" + ("" if args.no_copy else f"\nimages: {out}"))


if __name__ == "__main__":
    main()
