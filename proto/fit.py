# Fit the rig to a generated clip by matching silhouettes through the known camera.
# Usage: python proto/fit.py <run_dir> <video.mp4> <out_pose.npz> [tracks.npz]
# Reads rig.npz and camera.json from run_dir. Solves frames in order, each starting from the
# previous solution (frame 0 starts at the known reference pose). Per frame it optimises one
# rotation offset per bone plus a root translation, against a two-way silhouette chamfer:
# projected surface samples outside the mask are pulled in, and mask pixels pull the nearest
# sample towards them. A soft left/right symmetry prior keeps limbs the camera cannot see
# (the far wing, or one pointing at the lens) mirroring their visible twin. It assumes the rig
# is mirror-symmetric about world x=0 at rest and pairs bones by .L/.R names. A depth prior
# penalises joints moving along the camera axis, which silhouettes cannot see; the camera is
# chosen perpendicular to the action, so most real motion is across the view. With point tracks
# (from track.py), each tracked frame-0 pixel is tied to the visible surface point under it in
# the known reference pose, and that point is pulled along its track wherever it stays visible.
# Writes per-frame bone basis matrices (N,B,4,4) in Blender's convention and joint positions.
import sys, json, os, time
import cv2, numpy as np, torch

run, video, out = sys.argv[1:4]
tracks_path = sys.argv[4] if len(sys.argv) > 4 else None
dev = "cuda"
ITERS, N_SURF, N_MASK, N_TRACK = 300, 4000, 1500, 1500
W_COVER, W_SMOOTH, W_PRIOR, W_SYM, W_DEPTH, W_TRACK = 1.0, 3.0, 3.0, 3.0, 30.0, 0.3
if tracks_path:
    W_DEPTH = 300.0  # tracks pull hard in 2D; without a firm depth prior bones tilt towards the lens

rig = np.load(os.path.join(run, "rig.npz"))
cam = json.load(open(os.path.join(run, "camera.json")))
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
parents = rig["parents"].tolist()
rest, ref_basis = T(rig["rest"]), T(rig["ref_basis"])
verts, faces, weights = T(rig["verts"]), torch.tensor(rig["faces"], device=dev), T(rig["weights"])
B = len(parents)
local_rest = torch.stack([rest[i] if p < 0 else torch.linalg.inv(rest[p]) @ rest[i] for i, p in enumerate(parents)])
inv_rest = torch.linalg.inv(rest)
w2c = torch.linalg.inv(T(cam["cam_to_world"]))
K = T(cam["K"])

# Symmetry pairs, each compared in the rest frame of the pair's nearest common ancestor.
names = rig["names"].tolist()
chain = lambda i: [i] + chain(parents[i]) if i >= 0 else []
pairs = []
for i, n in enumerate(names):
    if n.endswith(".L") and n[:-2] + ".R" in names:
        j = names.index(n[:-2] + ".R")
        anc = next(a for a in chain(i) if a in chain(j))
        R0 = rest[anc, :3, :3]
        mirror = R0.T @ torch.diag(T([-1.0, 1.0, 1.0])) @ R0
        pairs.append((i, j, anc, mirror))

# Fixed random surface samples (area-weighted), stored as face index plus barycentric weights.
tri = verts[faces]
area = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=1)


def surface_samples(n, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    fi = torch.multinomial(area, n, replacement=True, generator=g)
    uv = torch.rand(n, 2, device=dev, generator=g)
    uv = torch.where(uv.sum(1, keepdim=True) > 1, 1 - uv, uv)
    return fi, torch.stack([1 - uv.sum(1), uv[:, 0], uv[:, 1]], 1)


fidx, bary = surface_samples(N_SURF, 0)
at = lambda v, fi, b: (v[faces[fi]] * b[:, :, None]).sum(1)


def rodrigues(r):
    th = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = r / th
    kx = torch.zeros(*r.shape[:-1], 3, 3, device=dev)
    kx[..., 0, 1], kx[..., 0, 2], kx[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    kx = kx - kx.transpose(-1, -2)
    th = th[..., None]
    return torch.eye(3, device=dev) + torch.sin(th) * kx + (1 - torch.cos(th)) * kx @ kx


def basis_of(rot, trans):
    # Reference basis with a rotation offset per bone; root translation folded into the root basis.
    R = torch.eye(4, device=dev).repeat(B, 1, 1)
    R[:, :3, :3] = rodrigues(rot)
    basis = ref_basis @ R
    Tm = torch.eye(4, device=dev)
    Tm[:3, 3] = trans
    root = torch.linalg.inv(rest[0]) @ Tm @ rest[0] @ basis[0]
    return torch.cat([root[None], basis[1:]])


def skin(basis):
    mats = []
    for i, p in enumerate(parents):
        mats.append(local_rest[i] @ basis[i] if p < 0 else mats[p] @ local_rest[i] @ basis[i])
    mats = torch.stack(mats)
    deform = mats @ inv_rest
    v = torch.einsum("vb,bij,vj->vi", weights, deform[:, :3], torch.cat([verts, torch.ones_like(verts[:, :1])], 1))
    return v, mats


def symmetry(mats):
    # Rotation change of each bone since rest, seen from its pair's common ancestor; L vs mirrored R.
    R, R0 = mats[:, :3, :3], rest[:, :3, :3]
    change = lambda b, a: (R[a].T @ R[b]) @ (R0[a].T @ R0[b]).T
    return sum((change(i, a) - m @ change(j, a) @ m).pow(2).sum() for i, j, a, m in pairs)


def project(p):
    c = (w2c[:3, :3] @ p.T + w2c[:3, 3:]).T  # Blender camera looks down -Z
    return torch.stack([K[0, 0] * c[:, 0] / -c[:, 2] + K[0, 2], K[1, 2] - K[1, 1] * c[:, 1] / -c[:, 2]], 1)


def mask_of(f):
    bg = np.median(np.concatenate([f[:20].reshape(-1, 3), f[-20:].reshape(-1, 3)]), 0)
    m = (np.abs(f.astype(int) - bg).sum(2) > 60).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


ref_joints = skin(ref_basis)[1][:, :3, 3]
axis = -w2c[2, :3]  # camera viewing direction in world space
size = (ref_joints.max(0).values - ref_joints.min(0).values).norm()

# Check: torch skinning in the reference pose must match Blender's.
err = (skin(ref_basis)[0] - T(rig["ref_verts"])).norm(dim=1).max().item()
print(f"skinning check: max vertex error {err:.5f}")
assert err < 1e-3

if tracks_path:
    # Tie each tracked frame-0 pixel to the front-most surface point under it in the reference pose.
    tr = np.load(tracks_path)
    pick = np.random.default_rng(0).choice(len(tr["query"]), min(N_TRACK, len(tr["query"])), replace=False)
    query, track_xy, track_vis = T(tr["query"][pick]), T(tr["tracks"][:, pick]), torch.tensor(tr["visible"][:, pick], device=dev)
    dfi, dbary = surface_samples(60000, 1)
    dense = at(skin(ref_basis)[0], dfi, dbary)
    dproj, ddepth = project(dense), (w2c[2, :3] @ dense.T + w2c[2, 3]).neg()
    near = torch.cdist(query, dproj) < 1.5  # dense points under each query pixel
    best = torch.where(near, ddepth[None], torch.inf).argmin(1)
    keep = near.any(1)
    tfidx, tbary = dfi[best[keep]], dbary[best[keep]]
    track_xy, track_vis = track_xy[:, keep], track_vis[:, keep]
    print(f"tracks tied to the surface: {int(keep.sum())} of {len(pick)}")

cap = cv2.VideoCapture(video)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
H, W = frames[0].shape[:2]

rot = torch.zeros(B, 3, device=dev)
trans = torch.zeros(3, device=dev)
result, joints, t0 = [], [], time.time()
for i, f in enumerate(frames):
    m = mask_of(f)
    dt = torch.tensor(cv2.distanceTransform(1 - m, cv2.DIST_L2, 5), device=dev)[None, None]
    ys, xs = np.nonzero(m)
    sel = np.random.default_rng(i).choice(len(xs), min(N_MASK, len(xs)), replace=False)
    mpix = T(np.stack([xs[sel], ys[sel]], 1))
    prev_rot, prev_trans = rot.clone(), trans.clone()
    rot.requires_grad_(True)
    trans.requires_grad_(True)
    opt = torch.optim.Adam([rot, trans], lr=0.01)
    for it in range(ITERS):
        v, mats = skin(basis_of(rot, trans))
        s = project(at(v, fidx, bary))
        grid = torch.stack([s[:, 0] / (W - 1), s[:, 1] / (H - 1)], 1) * 2 - 1
        l_out = torch.nn.functional.grid_sample(dt, grid[None, None], align_corners=True).mean()
        l_cover = torch.cdist(mpix, s).min(1).values.mean()
        l_smooth = (rot - prev_rot).pow(2).sum() + (trans - prev_trans).pow(2).sum()
        l_prior = rot.pow(2).sum()
        l_sym = symmetry(mats)
        l_depth = (((mats[:, :3, 3] - ref_joints) @ axis) / size).pow(2).mean()
        loss = (l_out + W_COVER * l_cover + W_SMOOTH * l_smooth + W_PRIOR * l_prior + W_SYM * l_sym
                + W_DEPTH * l_depth)
        if tracks_path:
            d = project(at(v, tfidx, tbary))[track_vis[i]] - track_xy[i][track_vis[i]]
            loss = loss + W_TRACK * torch.nn.functional.huber_loss(d, torch.zeros_like(d), delta=3.0, reduction="sum") / max(len(d), 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    rot, trans = rot.detach(), trans.detach()
    result.append(basis_of(rot, trans).cpu().numpy())
    joints.append(mats[:, :3, 3].detach().cpu().numpy())
    if i % 12 == 0:
        print(f"frame {i}: out {l_out.item():.2f}px cover {l_cover.item():.2f}px sym {l_sym.item():.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

np.savez(out, basis=np.array(result), joints=np.array(joints), names=rig["names"])
print("wrote", out)
