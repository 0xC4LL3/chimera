# Overlay the rendered rig on the generated clip and measure silhouette overlap.
# Usage: python pipeline/measure.py <video.mp4> <fit_dir> <out.mp4> [turned_dir]
# The video mask is "differs from the plain background"; the rig mask is the render's alpha.
# Prints mean IoU and writes a clip: video | rig overlay (rig red, video creature blue)
# | the rig from the turned camera, when given.
import sys, glob, cv2, numpy as np

video, fit, out = sys.argv[1:4]
turned = sorted(glob.glob(f"{sys.argv[4]}/*.png")) if len(sys.argv) > 4 else []
cap = cv2.VideoCapture(video)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
renders = sorted(glob.glob(f"{fit}/*.png"))
n = min(len(frames), len(renders))
h, w = frames[0].shape[:2]
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 24, (w * (3 if turned else 2), h))
ious = []
for i in range(n):
    f = frames[i]
    bg = np.median(np.concatenate([f[:20].reshape(-1, 3), f[-20:].reshape(-1, 3)]), 0)
    vmask = np.abs(f.astype(int) - bg).sum(2) > 60
    vmask = cv2.morphologyEx(vmask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    r = cv2.imread(renders[i], cv2.IMREAD_UNCHANGED)
    rmask = r[..., 3] > 127
    ious.append((vmask & rmask).sum() / max((vmask | rmask).sum(), 1))
    over = (f * 0.5).astype(np.uint8)
    over[vmask] = over[vmask] // 2 + np.array([120, 40, 0], np.uint8)
    over[rmask] = over[rmask] // 2 + np.array([0, 40, 120], np.uint8)
    cv2.putText(over, f"IoU {ious[-1]:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    panels = [f, over]
    if turned:
        t = cv2.resize(cv2.imread(turned[i], cv2.IMREAD_UNCHANGED), (w, h)).astype(float)
        a = t[..., 3:] / 255
        panels.append((t[..., :3] * a + 90 * (1 - a)).astype(np.uint8))
    writer.write(np.hstack(panels))
writer.release()
print(f"frames={n} mean_iou={np.mean(ious):.3f} min_iou={np.min(ious):.3f} first={ious[0]:.3f}")
print("per-frame:", " ".join(f"{x:.2f}" for x in ious))
