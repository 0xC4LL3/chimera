# Clean up a fitted animation: smoothing, loop closing and a root-motion split.
# Usage: python pipeline/cleanup.py <run_dir> <pose.npz> <out.npz> [--loop]
# One optimisation over the whole clip adjusts every bone's rotation (and each root's position) a
# little away from the fit, trading closeness to the fit (joint positions and bone orientations)
# against:
#   - smoothness: low acceleration of the same quantities. With both in the same units the pair
#     acts as a low-pass filter; W_ACC / W_FIT = 0.5 keeps motion up to about 5 Hz at 24 fps;
#   - loop closing (--loop): the last frame matches the first, so an attack chains into idle.
# Feet are kept flat and planted by the fitter, which weighs them against the video; locking them
# here, without the video, pulled the body away from the clip.
# Writes basis (root motion kept) and basis_inplace (horizontal root travel removed).
import sys, os, argparse
import numpy as np, torch

sys.path.append(os.path.dirname(__file__))
from rigtorch import Rig, T, dev, rodrigues

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("pose")
ap.add_argument("out")
ap.add_argument("--loop", action="store_true")
args = ap.parse_args()

ITERS = 200
W_FIT, W_ACC, W_LOOP = 1.0, 0.5, 100.0

rig = Rig(args.run)
fit = T(np.load(args.pose)["basis"])
N, B = fit.shape[:2]
with torch.no_grad():
    mats0 = rig.fk(fit)
p0, R0 = mats0[..., :3, 3], mats0[..., :3, :3]

size = (p0[0].max(0).values - p0[0].min(0).values).norm()

rot = torch.zeros(N, B, 3, device=dev, requires_grad=True)
roots = [i for i, p in enumerate(rig.parents) if p < 0]
trans = torch.zeros(N, len(roots), 3, device=dev, requires_grad=True)
root_rest = rig.rest[roots]


def move_roots(basis, trans):
    # Translate every root bone (extra roots are usually IK controls that carry the feet).
    Tm = torch.eye(4, device=dev).repeat(N, len(roots), 1, 1)
    Tm[..., :3, 3] = trans
    basis = basis.clone()
    basis[:, roots] = torch.linalg.inv(root_rest) @ Tm @ root_rest @ basis[:, roots]
    return basis


def basis_of(rot, trans):
    R = torch.eye(4, device=dev).repeat(N, B, 1, 1)
    R[..., :3, :3] = rodrigues(rot)
    return move_roots(fit @ R, trans)


def stats(mats):
    p = mats[..., :3, 3]
    acc = (p[2:] - 2 * p[1:-1] + p[:-2]).norm(dim=-1).mean() / size
    gap = (p[-1] - p[0]).norm(dim=-1).mean() / size
    return f"jitter {acc.item() * 100:.2f}%  loop gap {gap.item() * 100:.2f}%"


print("before:", stats(mats0))
opt = torch.optim.Adam([rot, trans], lr=0.0125)
for it in range(ITERS):
    mats = rig.fk(basis_of(rot, trans))
    p, Rm = mats[..., :3, 3] / size, mats[..., :3, :3]
    loss = W_FIT * ((p - p0 / size).pow(2).sum(-1).mean() + (Rm - R0).pow(2).sum((-1, -2)).mean())
    loss = loss + W_ACC * ((p[2:] - 2 * p[1:-1] + p[:-2]).pow(2).sum(-1).mean()
                           + (Rm[2:] - 2 * Rm[1:-1] + Rm[:-2]).pow(2).sum((-1, -2)).mean())
    if args.loop:
        loss = loss + W_LOOP * ((p[-1] - p[0]).pow(2).sum(-1).mean() + (Rm[-1] - Rm[0]).pow(2).sum((-1, -2)).mean())
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    basis = basis_of(rot, trans)
    mats = rig.fk(basis)
    print("after: ", stats(mats))
    # In-place variant: remove the main root's horizontal travel since frame 0 from every root.
    drift = mats[:, 0, :3, 3] - mats[0, 0, :3, 3]
    drift[:, 2] = 0
    inplace = move_roots(basis, -drift[:, None].expand(N, len(roots), 3))
np.savez(args.out, basis=basis.cpu().numpy(), basis_inplace=inplace.cpu().numpy(),
         joints=mats[..., :3, 3].cpu().numpy(), names=rig.names)
print("wrote", args.out)
