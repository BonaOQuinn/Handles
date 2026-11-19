"""Step through videos and mark targeted frames for ball-detection labeling.

Writes/updates extra.csv (video,frame,reason), which sample_frames.py reads via --extra.
Marks are saved after every keypress, so closing the window never loses work.

Keys
  d / a      next / previous frame            l / j    forward / back 10 frames
  space      play / pause (plays at ~1/4 speed)
  1 bottom   2 blur   4 edge   5 second_ball
  6 body_occluded (ball hidden by the player's body or another object)
  7 out_of_frame  (ball fully out of view: a negative example, label with NO box;
                   mark only 1-2 per video, ideally with something ball-like in view)
                                    (the number keys mark the current frame)
  = / -      make the picture larger / smaller (if it is cut off, press -)
  u          undo the last mark on this video
  n          next video          x   quit

Test-split videos are refused: targeted frames must never go into the test set.

Example:
    python frame_picker.py videos/train videos/val --out extra.csv
"""

import argparse
import csv
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
REASONS = {ord("1"): "bottom", ord("2"): "blur", ord("4"): "edge", ord("5"): "second_ball", ord("6"): "body_occluded", ord("7"): "out_of_frame"}


def default_view_height():
    """Picture height that fits the screen: usable screen height minus title bar and taskbar."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()  # real pixels, not scaled ones
        except Exception:
            pass
        return max(400, user32.GetSystemMetrics(1) - 170)
    except Exception:
        return 800  # non-Windows or no display info


def collect(inputs):
    vids = []
    for item in inputs:
        p = Path(item)
        found = sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS) if p.is_dir() else [p]
        for v in found:
            if "test" in {part.lower() for part in v.parts}:
                print(f"skipping {v}: test-split videos get no targeted frames")
                continue
            vids.append(v)
    return vids


def load_marks(path):
    marks = {}
    if Path(path).exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                marks.setdefault(row["video"], []).append((int(row["frame"]), row["reason"]))
    return marks


def save_marks(path, marks):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "reason"])
        for video, items in marks.items():
            for frame, reason in sorted(set(items)):
                w.writerow([video, frame, reason])


def show(frame, idx, total, fps, video, marks_here, view_h):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (int(w * view_h / h), view_h), interpolation=cv2.INTER_AREA)
    here = [r for f, r in marks_here if f == idx]
    lines = [
        f"{video}  frame {idx}/{total - 1}  t={idx / fps:.2f}s  marks: {len(marks_here)}",
        "MARKED: " + ", ".join(here) if here else "d/a step  j/l x10  space play  1 bottom 2 blur 4 edge 5 2nd ball 6 body 7 no ball  u undo  n next  x quit",
    ]
    for i, text in enumerate(lines):
        color = (0, 255, 0) if here and i == 1 else (255, 255, 255)
        cv2.putText(img, text, (12, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, text, (12, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("frame_picker", img)
    cv2.moveWindow("frame_picker", 0, 0)  # top-left, so the bottom is not pushed off screen


def pick(video, marks, out_path, view):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    idx, frame, playing = 0, None, False  # view[0] = picture height in pixels
    marks_here = marks.setdefault(video.name, [])

    def goto(target, sequential=False):
        nonlocal idx, frame
        target = max(0, min(total - 1, target))
        if not (sequential and target == idx + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, f = cap.read()
        if ok:
            idx, frame = target, f

    goto(0)
    while True:
        show(frame, idx, total, fps, video.name, marks_here, view[0])
        key = cv2.waitKey(66 if playing else 0) & 0xFF
        if key == 255:  # timeout while playing
            goto(idx + 1, sequential=True)
            if idx >= total - 1:
                playing = False
            continue
        if key == ord(" "):
            playing = not playing
        elif key == ord("d"):
            playing = False
            goto(idx + 1, sequential=True)
        elif key == ord("a"):
            playing = False
            goto(idx - 1)
        elif key in (ord("="), ord("+")):
            view[0] = int(view[0] * 1.1)
        elif key in (ord("-"), ord("_")):
            view[0] = max(300, int(view[0] / 1.1))
        elif key == ord("l"):
            goto(idx + 10)
        elif key == ord("j"):
            goto(idx - 10)
        elif key in REASONS:
            marks_here.append((idx, REASONS[key]))
            save_marks(out_path, marks)
        elif key == ord("u") and marks_here:
            marks_here.pop()
            save_marks(out_path, marks)
        elif key == ord("n"):
            cap.release()
            return True
        elif key == ord("x"):
            cap.release()
            return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="video files or folders (train/val only)")
    ap.add_argument("--out", default="extra.csv")
    ap.add_argument("--height", type=int, default=0, help="picture height in pixels (default: fit the screen)")
    args = ap.parse_args()

    marks = load_marks(args.out)
    view = [args.height or default_view_height()]
    print(f"picture height: {view[0]} px (press - / = in the window to resize, or use --height)")
    for v in collect(args.inputs):
        print(f"{v}  (existing marks: {len(marks.get(v.name, []))})")
        if not pick(v, marks, args.out, view):
            break
    cv2.destroyAllWindows()
    save_marks(args.out, marks)
    print(f"saved {sum(len(x) for x in marks.values())} marks -> {args.out}")


if __name__ == "__main__":
    main()
