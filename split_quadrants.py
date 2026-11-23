"""
CatchCam utility: split a 4-way CCTV split-screen video into 4 separate
single-camera video files (top-left, top-right, bottom-left, bottom-right).

Usage:
    python split_quadrants.py --source split_screen.mp4 --out_dir .
"""
import argparse
import cv2
from pathlib import Path


def split_quadrants(source: str, out_dir: str = "."):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_w, half_h = w // 2, h // 2

    names = ["cam1_topleft", "cam2_topright", "cam3_bottomleft", "cam4_bottomright"]
    writers = {
        name: cv2.VideoWriter(
            str(out_path / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (half_w, half_h)
        )
        for name in names
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        quadrants = {
            "cam1_topleft": frame[0:half_h, 0:half_w],
            "cam2_topright": frame[0:half_h, half_w:w],
            "cam3_bottomleft": frame[half_h:h, 0:half_w],
            "cam4_bottomright": frame[half_h:h, half_w:w],
        }
        for name, crop in quadrants.items():
            writers[name].write(crop)

        frame_idx += 1

    cap.release()
    for writer in writers.values():
        writer.release()

    print(f"[CatchCam] Split {frame_idx} frames into 4 camera files in {out_path}/")
    for name in names:
        print(f"  -> {out_path / (name + '.mp4')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()
    split_quadrants(args.source, args.out_dir)
