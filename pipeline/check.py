# Check how well a captured animation explains the video it came from.
# Usage: python pipeline/check.py <run_dir> <pose.npz> <video.mp4>
# Renders the fitted mesh's silhouette through the reference camera on every frame and compares
# it with the creature's mask in the video. Poor overlap means the fit could not follow the clip
# (an action the rig cannot make, the camera seeing it end-on, a broken generation). Also reports
# how far any bone turned towards or away from the camera: the fitter keeps that small, so a large
# turn means a limb was swung towards the lens to fake the 2D shape, which the camera cannot see.
# Prints a JSON report.
import json, sys, os
import cv2, numpy as np, torch

sys.path.append(os.path.dirname(__file__))
from rigtorch import Rig, T

MEAN_LIMIT, MIN_LIMIT, TURN_LIMIT = 0.8, 0.6, 45.0

run, pose, video = sys.argv[1:4]
rig = Rig(run)
basis = T(np.load(pose)["basis"])
axis = -rig.w2c[2, :3]
faces = rig.faces.cpu().numpy()

cap = cv2.VideoCapture(video)
ious = []
with torch.no_grad():
    verts, mats = rig.skin(basis)
    pix = rig.project(verts).cpu().numpy()
    for t in range(len(basis)):
        ok, f = cap.read()
        if not ok:
            break
        bg = np.median(np.concatenate([f[:20].reshape(-1, 3), f[-20:].reshape(-1, 3)]), 0)
        video_mask = cv2.morphologyEx((np.abs(f.astype(int) - bg).sum(2) > 60).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fit_mask = np.zeros_like(video_mask)
        for tri in np.round(pix[t][faces]).astype(np.int32):  # one by one: overlapping triangles in one fillPoly cancel out
            cv2.fillConvexPoly(fit_mask, tri, 1)
        ious.append((video_mask & fit_mask).sum() / max((video_mask | fit_mask).sum(), 1))
    deforming = (rig.weights.sum(0) > 0) & torch.tensor([p >= 0 for p in rig.parents], device=rig.rest.device)
    turn = ((mats[:, deforming, :3, 1] - rig.fk(rig.ref_basis)[deforming, :3, 1]) @ axis).abs().max().clamp(max=1)

ious = np.array(ious)
turn_deg = float(np.degrees(np.arcsin(turn.item())))
print(json.dumps({"ok": bool(ious.mean() >= MEAN_LIMIT and ious.min() >= MIN_LIMIT and turn_deg <= TURN_LIMIT),
                  "mean_overlap": round(float(ious.mean()), 3), "min_overlap": round(float(ious.min()), 3),
                  "worst_frame": int(ious.argmin()), "max_turn_deg": round(turn_deg, 1)}))
