# Plausibility check of a captured animation, for failures the camera view cannot show.
# Usage: python pipeline/check.py <run_dir> <pose.npz>
# One camera cannot see motion towards or away from it, so a fit can match the video well while
# a limb swings out sideways. When the rig comes with clips, the check measures how far each
# frame's pose lies outside every combination of the rig's own clip poses (the same directions
# the fitter's pose prior uses), in degrees: a pose its animator never made is suspect. Without
# clips it falls back to how far any part of the mesh moved along the camera axis, in units of
# the creature's size. Prints a JSON report.
import json, sys, os
import numpy as np, torch
from scipy.spatial.transform import Rotation

sys.path.append(os.path.dirname(__file__))
from rigtorch import Rig, T

POSE_LIMIT, DEPTH_LIMIT, POSE_VAR = 10.0, 0.10, 0.999

run, pose = sys.argv[1:3]
rig = Rig(run)
data = np.load(os.path.join(run, "rig.npz"))
basis = np.load(pose)["basis"]
report = {}

axis = -rig.w2c[2, :3]
v0 = rig.skin(rig.ref_basis)[0]
size = (v0.max(0).values - v0.min(0).values).norm()
with torch.no_grad():
    shift = np.array([(((rig.skin(b)[0] - v0) @ axis) / size).abs().max().item() for b in T(basis)])
report["max_depth_shift"] = round(float(shift.max()), 3)

clips = data["clip_poses"]
if len(clips) > 1:
    X = clips[:, 1:].reshape(len(clips), -1)
    mean = X.mean(0)
    _, sv, Vt = np.linalg.svd(X - mean, full_matrices=False)
    dirs = Vt[:int(((sv**2).cumsum() / (sv**2).sum() < POSE_VAR).sum()) + 1].T
    inv_ref = np.linalg.inv(data["ref_basis"][1:, :3, :3])
    residual = []
    for frame in basis:
        x = Rotation.from_matrix(inv_ref @ frame[1:, :3, :3]).as_rotvec().reshape(-1) - mean
        residual.append(np.degrees(np.linalg.norm(x - dirs @ (dirs.T @ x))))
    residual = np.array(residual)
    report["max_pose_residual_deg"] = round(float(residual.max()), 1)
    bad = np.nonzero(residual > POSE_LIMIT)[0]
else:
    bad = np.nonzero(shift > DEPTH_LIMIT)[0]
report["ok"] = len(bad) == 0
report["bad_frames"] = [int(bad[0]), int(bad[-1])] if len(bad) else None
print(json.dumps(report))
