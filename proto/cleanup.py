# Clean up a fitted animation: smoothing, loop closing, foot locking and a root-motion split.
# Usage: python proto/cleanup.py <run_dir> <pose.npz> <out.npz> [--loop] [--contacts auto|Bone1,Bone2]
# One optimisation over the whole clip adjusts every bone's rotation (and the root position) a
# little away from the fit, trading closeness to the fit (joint positions and bone orientations)
# against:
#   - smoothness: low acceleration of the same quantities. With both in the same units the pair
#     acts as a low-pass filter; W_ACC / W_FIT = 0.5 keeps motion up to about 5 Hz at 24 fps;
#   - loop closing (--loop): the last frame matches the first, so an attack chains into idle;
#   - foot locking (--contacts): contact joints hold still while planted and stay above ground.
# Contacts follow the UniMate recipe: bones picked by name (or listed), in contact when low
# and slow, cleaned with a median filter. A bone's contact point is the lowest mesh vertex
# mostly skinned to it in the reference pose (a toe tip, the sole of a paw), since bones often
# end above the sole. Ground is the lowest contact point on frame 0, the known reference pose
# (asking for contacts means the creature starts on the ground).
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
ap.add_argument("--contacts", default="")
args = ap.parse_args()

ITERS = 500
W_FIT, W_ACC, W_LOOP, W_LOCK, W_GROUND = 1.0, 0.5, 100.0, 1000.0, 1000.0
CONTACT_NAMES = ("foot", "feet", "toe", "paw", "hoof")
H_THR, V_THR, MEDIAN = 0.03, 0.01, 5  # height and per-frame speed, in units of skeleton size

rig = Rig(args.run)
fit = T(np.load(args.pose)["basis"])
N, B = fit.shape[:2]
with torch.no_grad():
    mats0 = rig.fk(fit)
p0, R0 = mats0[..., :3, 3], mats0[..., :3, :3]

size = (p0[0].max(0).values - p0[0].min(0).values).norm()

# Contact joints and per-frame contact flags.
if args.contacts == "auto":
    contact = [i for i, n in enumerate(rig.names) if any(k in n.lower() for k in CONTACT_NAMES)]
else:
    contact = [rig.names.index(n) for n in args.contacts.split(",") if n]
locks = []  # (contact index, frame indices, target position)
ground = None
cvert = [int(torch.where(rig.weights[:, j] > 0.5, rig.ref_verts[:, 2], torch.inf).argmin()) for j in contact]
contact_points = lambda mats: rig.skin_points(mats, cvert)
if contact:
    c0 = contact_points(mats0)
    z = c0[..., 2]
    ground = z[0].min()
    speed = torch.cat([torch.zeros_like(z[:1]), (c0[1:] - c0[:-1]).norm(dim=-1)])
    flag = ((z - ground < H_THR * size) & (speed < V_THR * size)).cpu().numpy()
    pad = np.pad(flag, ((MEDIAN // 2, MEDIAN // 2), (0, 0)), mode="edge")
    flag = np.stack([np.median(pad[t:t + MEDIAN], 0) for t in range(N)]) > 0.5
    for c in range(len(contact)):
        t = 0
        while t < N:
            if flag[t, c]:
                s = t
                while t < N and flag[t, c]:
                    t += 1
                frames = torch.arange(s, t, device=dev)
                target = c0[frames, c].mean(0)
                target[2] = ground
                locks.append((c, frames, target))
            t += 1
    print(f"contacts: {[rig.names[j] for j in contact]}, {len(locks)} planted intervals")

rot = torch.zeros(N, B, 3, device=dev, requires_grad=True)
trans = torch.zeros(N, 3, device=dev, requires_grad=True)
root_rest = rig.rest[0]


def basis_of(rot, trans):
    R = torch.eye(4, device=dev).repeat(N, B, 1, 1)
    R[..., :3, :3] = rodrigues(rot)
    basis = fit @ R
    Tm = torch.eye(4, device=dev).repeat(N, 1, 1)
    Tm[:, :3, 3] = trans
    root = torch.linalg.inv(root_rest) @ Tm @ root_rest @ basis[:, 0]
    return torch.cat([root[:, None], basis[:, 1:]], 1)


def stats(mats):
    p, tp = mats[..., :3, 3], contact_points(mats)
    acc = (p[2:] - 2 * p[1:-1] + p[:-2]).norm(dim=-1).mean() / size
    gap = (p[-1] - p[0]).norm(dim=-1).mean() / size
    slide = max([(tp[f, j] - tp[f[0], j]).norm(dim=-1).max().item() / size.item() for j, f, _ in locks], default=0)
    return f"jitter {acc.item() * 100:.2f}%  loop gap {gap.item() * 100:.2f}%  max planted slide {slide * 100:.2f}%"


print("before:", stats(mats0))
opt = torch.optim.Adam([rot, trans], lr=0.005)
for it in range(ITERS):
    mats = rig.fk(basis_of(rot, trans))
    p, Rm = mats[..., :3, 3] / size, mats[..., :3, :3]
    loss = W_FIT * ((p - p0 / size).pow(2).sum(-1).mean() + (Rm - R0).pow(2).sum((-1, -2)).mean())
    loss = loss + W_ACC * ((p[2:] - 2 * p[1:-1] + p[:-2]).pow(2).sum(-1).mean()
                           + (Rm[2:] - 2 * Rm[1:-1] + Rm[:-2]).pow(2).sum((-1, -2)).mean())
    if args.loop:
        loss = loss + W_LOOP * ((p[-1] - p[0]).pow(2).sum(-1).mean() + (Rm[-1] - Rm[0]).pow(2).sum((-1, -2)).mean())
    tp = contact_points(mats) / size
    for j, f, target in locks:
        loss = loss + W_LOCK * ((tp[f, j] - target / size).pow(2).sum(-1)).mean()
    if ground is not None:
        loss = loss + W_GROUND * torch.relu(ground / size - tp[..., 2]).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    basis = basis_of(rot, trans)
    mats = rig.fk(basis)
    print("after: ", stats(mats))
    # In-place variant: remove the root's horizontal travel relative to frame 0.
    drift = mats[:, 0, :3, 3] - mats[0, 0, :3, 3]
    drift[:, 2] = 0
    Tm = torch.eye(4, device=dev).repeat(N, 1, 1)
    Tm[:, :3, 3] = -drift
    inplace = basis.clone()
    inplace[:, 0] = torch.linalg.inv(root_rest) @ Tm @ root_rest @ basis[:, 0]
np.savez(args.out, basis=basis.cpu().numpy(), basis_inplace=inplace.cpu().numpy(),
         joints=mats[..., :3, 3].cpu().numpy(), names=rig.names)
print("wrote", args.out)
