"""
Calibration helper: save one frame of a camera video with a labelled pixel grid, so you can read off the (x, y) pixel
position of floor landmarks (tile corners, door frames at floor level) without any GUI tool.

    python -m floorplan.grid_still --video cameras/cam1_topleft.mp4 --out cam1_grid.png [--frame 0] [--step 100]

Open cam1_grid.png, pick 4-8 spread-out points on the FLOOR, note their pixel coordinates from the grid labels, measure the
same spots on your floor plan (in metres), and put both lists in floorplan.json as image_points / floor_points.
Then check your clicks:   python -m floorplan.core check floorplan.json
"""
import argparse

import cv2


def grid_still(video, out, frame=0, step=100):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame} of {video}")
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(img, str(x), (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, str(x), (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(img, str(y), (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, str(y), (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), img)
    print(f"[CatchCam] Saved {w}x{h} frame with {step}px grid -> {out}")
    return w, h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--out", default="grid.png")
    ap.add_argument("--frame", type=int, default=0); ap.add_argument("--step", type=int, default=100)
    a = ap.parse_args()
    grid_still(a.video, a.out, a.frame, a.step)
