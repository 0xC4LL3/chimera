# Dense point tracks from frame 0 with AllTracker, kept for pixels on the creature.
# Usage: python pipeline/track.py <video.mp4> <out_tracks.npz> [alltracker_dir]
# Writes query (N,2) frame-0 pixels, tracks (T,N,2) pixel positions and visible (T,N) bools.
# Tracking runs on a crop around everywhere the creature appears in the clip: the tracker's memory
# and time grow with the image area, and the plain background around the creature carries nothing.
import sys, os
import cv2, numpy as np, torch

video, out = sys.argv[1:3]
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "third_party", "alltracker"))
from nets.alltracker import Net

STRIDE, VIS_THR, MARGIN = 2, 0.5, 32

cap = cv2.VideoCapture(video)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
H, W = frames[0].shape[:2]


def mask_of(f):
    bg = np.median(np.concatenate([f[:20].reshape(-1, 3), f[-20:].reshape(-1, 3)]), 0)
    return cv2.morphologyEx((np.abs(f.astype(int) - bg).sum(2) > 60).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


mask = mask_of(frames[0])
ys, xs = np.nonzero(np.max([mask_of(f) for f in frames], 0))
x0, y0 = max(xs.min() - MARGIN, 0) // 8 * 8, max(ys.min() - MARGIN, 0) // 8 * 8
x1, y1 = min(-(-(xs.max() + MARGIN) // 8) * 8, W), min(-(-(ys.max() + MARGIN) // 8) * 8, H)

model = Net(16)
url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location="cpu")["model"])
model.cuda().eval()
rgbs = torch.stack([torch.from_numpy(f[y0:y1, x0:x1, ::-1].copy()).permute(2, 0, 1) for f in frames])[None].float().cuda()
with torch.no_grad():
    flows, visconf, _, _ = model.forward_sliding(rgbs, iters=4, sw=None, is_training=False)

ys, xs = np.nonzero(mask[y0:y1:STRIDE, x0:x1:STRIDE])
ys, xs = ys * STRIDE, xs * STRIDE  # in crop pixels
flows, visconf = flows[0].cpu().numpy(), visconf[0].cpu().numpy()  # T,2,h,w
tracks = np.stack([x0 + xs + flows[:, 0, ys, xs], y0 + ys + flows[:, 1, ys, xs]], -1)
visible = (visconf[:, 0, ys, xs] > VIS_THR) & (visconf[:, 1, ys, xs] > VIS_THR)
xs, ys = xs + x0, ys + y0
np.savez(out, query=np.stack([xs, ys], 1), tracks=tracks, visible=visible)
print(f"tracked {len(xs)} points over {len(frames)} frames, mean visible {visible.mean():.2f}")
