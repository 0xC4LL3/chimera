# Dense point tracks from frame 0 with AllTracker, kept for pixels on the creature.
# Usage: python pipeline/track.py <video.mp4> <out_tracks.npz> [alltracker_dir]
# Writes query (N,2) frame-0 pixels, tracks (T,N,2) pixel positions and visible (T,N) bools.
import sys, os
import cv2, numpy as np, torch

video, out = sys.argv[1:3]
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/alltracker"))
from nets.alltracker import Net

STRIDE, VIS_THR = 2, 0.5

cap = cv2.VideoCapture(video)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
H, W = frames[0].shape[:2]

f0 = frames[0]
bg = np.median(np.concatenate([f0[:20].reshape(-1, 3), f0[-20:].reshape(-1, 3)]), 0)
mask = cv2.morphologyEx((np.abs(f0.astype(int) - bg).sum(2) > 60).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

model = Net(16)
url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location="cpu")["model"])
model.cuda().eval()
rgbs = torch.stack([torch.from_numpy(f[..., ::-1].copy()).permute(2, 0, 1) for f in frames])[None].float().cuda()
with torch.no_grad():
    flows, visconf, _, _ = model.forward_sliding(rgbs, iters=4, sw=None, is_training=False)

ys, xs = np.nonzero(mask[::STRIDE, ::STRIDE])
ys, xs = ys * STRIDE, xs * STRIDE
flows, visconf = flows[0].cpu().numpy(), visconf[0].cpu().numpy()  # T,2,H,W
tracks = np.stack([xs + flows[:, 0, ys, xs], ys + flows[:, 1, ys, xs]], -1)
visible = (visconf[:, 0, ys, xs] > VIS_THR) & (visconf[:, 1, ys, xs] > VIS_THR)
np.savez(out, query=np.stack([xs, ys], 1), tracks=tracks, visible=visible)
print(f"tracked {len(xs)} points over {len(frames)} frames, mean visible {visible.mean():.2f}")
