# Fit the rig to a generated clip through the known camera.
# Usage: python pipeline/fit.py <run_dir> <video.mp4> <out_pose.npz> [tracks.npz] [--grounded]
# Reads rig.npz and camera.json from run_dir. Solves windows of WIN frames in order, all frames of
# a window together (one batch on the GPU, much faster than frame by frame) and each window starting
# from the last solved frame (frame 0 starts at the known reference pose). Per frame it optimises
# one rotation offset per bone plus a translation per root bone. Data terms:
#   - silhouette: a two-way chamfer; projected surface samples outside the mask are pulled in,
#     and mask pixels pull the nearest sample towards them;
#   - point tracks (from track.py): each tracked frame-0 pixel is tied to the visible surface
#     point under it in the known reference pose, and pulled along its track while visible.
# One camera leaves many poses that match the video equally well, so priors pick a plausible one:
#   - bone direction: bones that carry mesh keep pointing as much towards or away from the camera
#     as in the reference pose. The camera is chosen perpendicular to the action, so most real
#     motion stays across the view, while a fit that cannot match the 2D shape otherwise would
#     turn a limb towards the lens to shorten its projection (a leg splayed sideways looks fine
#     from the side). Needs nothing but the rig, so it works for a brand-new character;
#   - reference: a weak pull towards the reference pose, for bones nothing else constrains;
#   - attachment: extra root bones that carry mesh (IK foot controls) stay at their reference
#     offset from the bone they share most mesh with, so a paw cannot leave its leg;
#   - depth: joints moving along the camera axis, which neither data term can see, cost a little;
#   - symmetry: left/right bone pairs mirror each other a little, mostly for the far-side limbs
#     the camera barely sees. Pairs are found from the rest pose (a bone whose head and direction
#     mirror another's across x=0), so any bone names work; assumes a rig symmetric about x=0.
#     Bones on the centre line (spine, neck, tail) are their own mirror image, so relative to their
#     parent they may bend up and down but not roll or turn sideways, a motion the side camera
#     barely sees (a rolled body raises one wing and drops the other);
#   - floor (--grounded): no part of the mesh may sink below the lowest point of the reference;
#   - feet (--grounded): a foot near the floor lies flat as in the reference pose and does not
#     slide, and no sole sinks below its reference height. Feet are the bones with at least half
#     of their mostly-skinned vertices touching the floor in the reference pose; their sole is
#     those touching vertices. Flatness and sliding fade in as the sole gets within FOOT_H of its
#     reference height, so a lifted foot is free and the video decides when it is down.
# Writes per-frame bone basis matrices (N,B,4,4) in Blender's convention and joint positions.
import json, sys, os, time
import cv2, numpy as np, torch

sys.path.append(os.path.dirname(__file__))
from rigtorch import Rig, T, dev, rodrigues

args = [a for a in sys.argv[1:] if not a.startswith("--")]
run, video, out = args[:3]
tracks_path = args[3] if len(args) > 3 else None
grounded = "--grounded" in sys.argv
WIN, ITERS, N_SURF, N_MASK, N_TRACK = 8, 150, 4000, 1500, 1500
FOOT_H = 0.03  # in units of skeleton size
W = dict(cover=1.0, track=0.5, smooth=3.0, bdir=10.0, ref=1.0, attach=1000.0, depth=300.0, sym=3.0, floor=1e4,
         flat=1e4, slide=1e4)
W.update(json.loads(os.environ.get("FIT_WEIGHTS", "{}")))  # for tuning experiments

rig = Rig(run)
parents, names, B = rig.parents, rig.names, rig.B
rest, ref_basis, verts, faces, w2c = rig.rest, rig.ref_basis, rig.verts, rig.faces, rig.w2c
skin, project = rig.skin, rig.project
roots = [i for i, p in enumerate(parents) if p < 0]
chain = lambda i: [i] + chain(parents[i]) if i >= 0 else []
ref_v, ref_mats = skin(ref_basis)
ref_joints = ref_mats[:, :3, 3]
axis = -w2c[2, :3]  # camera viewing direction in world space
size = (ref_joints.max(0).values - ref_joints.min(0).values).norm()
floor = ref_v[:, 2].min()

# Check: torch skinning in the reference pose must match Blender's.
err = (ref_v - rig.ref_verts).norm(dim=1).max().item()
print(f"skinning check: max vertex error {err:.5f}")
assert err < 1e-3

# Bone direction: how much each mesh-carrying bone (root excluded) points along the camera axis.
deforming = (rig.weights.sum(0) > 0) & torch.tensor([p >= 0 for p in parents], device=dev)
ref_dir_depth = ref_mats[deforming, :3, 1] @ axis  # a Blender bone points along its local Y

# Attachments: extra roots that carry mesh, glued to the bone they share most mesh with.
att_root, att_bone, att_off = [], [], []
for r in roots[1:]:
    sub = [i for i in range(B) if r in chain(i)]
    w_sub = rig.weights[:, sub].sum(1)
    if w_sub.sum() > 0:
        share = (rig.weights * w_sub[:, None]).sum(0)
        share[sub] = 0
        a = int(share.argmax())
        att_root.append(r), att_bone.append(a)
        att_off.append((torch.linalg.inv(ref_mats[a]) @ ref_mats[r])[:3, 3])
att_off = torch.stack(att_off) if att_off else None

# Feet: per contact bone, the sole vertices touching the floor in the reference pose.
mostly = rig.weights > 0.5
near_floor = rig.ref_verts[:, 2] < floor + FOOT_H * size
soles = [torch.nonzero(mostly[:, j] & near_floor).flatten() for j in range(B)] if grounded else []
soles = [v for j, v in enumerate(soles) if len(v) and len(v) >= mostly[:, j].sum() / 2]  # not shins grazing the floor
print(f"feet: {len(soles)}")
# All soles in one (feet, longest sole) index array, padded by repeating each sole's first vertex.
sole_idx = torch.stack([torch.cat([v, v[:1].repeat(max(map(len, soles)) - len(v))]) for v in soles]) if soles else None
sole_on = torch.stack([torch.arange(sole_idx.shape[1], device=dev) < len(v) for v in soles]).float() if soles else None
sole_n = sole_on.sum(1) if soles else None


def feet(v, prev):
    # Flat and planted feet for frames v (K,V,3); prev: sole positions on the frame before (F,S,3).
    p = v[:, sole_idx]  # K,F,S,3
    lift = (p[..., 2] - ref_v[sole_idx, 2]) / size  # sole height change since the reference pose
    mean = (lift * sole_on).sum(-1) / sole_n
    g = (1 - torch.where(sole_on > 0, lift, torch.inf).detach().min(-1).values / FOOT_H).clamp(0, 1)  # 1 down, 0 lifted
    tilt = ((lift - mean[..., None]).pow(2) * sole_on).sum(-1) / sole_n  # the sole tips away from its reference attitude
    flat = (g * tilt).sum() + torch.relu(-mean).pow(2).sum()  # and never sinks below its reference height
    p_all, g_all = torch.cat([prev[None], p]), torch.cat([g[:1], g])
    step = (((p_all[1:, ..., :2] - p_all[:-1, ..., :2]) / size).pow(2).sum(-1) * sole_on).sum(-1) / sole_n
    return flat, (g_all[1:] * g_all[:-1] * step).sum()


# Symmetry pairs, each compared in the rest frame of the pair's nearest common ancestor.
heads, dirs = rest[:, :3, 3], rest[:, :3, 1]
flip = T([-1.0, 1.0, 1.0])
mismatch = torch.cdist(heads * flip, heads) / size + 1 - (dirs * flip) @ dirs.T
mid = torch.diagonal(mismatch) < 0.03
mismatch.fill_diagonal_(torch.inf)
sym_l, sym_r, sym_a = [], [], []
for i in range(B):
    if mid[i]:  # a centre bone, compared with itself in its parent's frame (the world for a root)
        sym_l.append(i), sym_r.append(i), sym_a.append(parents[i])
    j = int(mismatch[i].argmin())
    if heads[i, 0] > 0.01 * size and mismatch[i, j] < 0.03:  # i on the +x side, j its mirror image
        anc = next((a for a in chain(i) if a in chain(j)), None)
        if anc is not None:  # pairs in separate hierarchies are IK controls; attachment covers them
            sym_l.append(i), sym_r.append(j), sym_a.append(anc)
eye = torch.eye(3, device=dev)
R0 = torch.cat([rest[:, :3, :3], eye[None]])  # index -1 (no parent) is the world
sym_rest_l = (R0[sym_a].transpose(1, 2) @ R0[sym_l]).transpose(1, 2)
sym_rest_r = (R0[sym_a].transpose(1, 2) @ R0[sym_r]).transpose(1, 2)
sym_mirror = R0[sym_a].transpose(1, 2) @ torch.diag(T([-1.0, 1.0, 1.0])) @ R0[sym_a]


def symmetry(mats):
    # Rotation change of each bone since rest, seen from its pair's common ancestor; L vs mirrored R.
    R = torch.cat([mats[..., :3, :3], eye.expand(*mats.shape[:-3], 1, 3, 3)], -3)
    Ra = R[..., sym_a, :, :].transpose(-1, -2)
    cl, cr = Ra @ R[..., sym_l, :, :] @ sym_rest_l, Ra @ R[..., sym_r, :, :] @ sym_rest_r
    return (cl - sym_mirror @ cr @ sym_mirror).pow(2).sum()


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
at = lambda v, fi, b: (v[..., faces[fi], :] * b[..., None]).sum(-2)


def basis_of(rot, trans):
    # Reference basis with a rotation offset per bone; root translations folded into root bases.
    # Works on one frame (B,3)/(R,3) or many (K,B,3)/(K,R,3).
    R = torch.eye(4, device=dev).repeat(*rot.shape[:-1], 1, 1)
    R[..., :3, :3] = rodrigues(rot)
    basis = ref_basis @ R
    Tm = torch.eye(4, device=dev).repeat(*trans.shape[:-1], 1, 1)
    Tm[..., :3, 3] = trans
    basis[..., roots, :, :] = torch.linalg.inv(rest[roots]) @ Tm @ rest[roots] @ basis[..., roots, :, :]
    return basis


def mask_of(f):
    bg = np.median(np.concatenate([f[:20].reshape(-1, 3), f[-20:].reshape(-1, 3)]), 0)
    m = (np.abs(f.astype(int) - bg).sum(2) > 60).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


if tracks_path:
    # Tie each tracked frame-0 pixel to the front-most surface point under it in the reference pose.
    tr = np.load(tracks_path)
    pick = np.random.default_rng(0).choice(len(tr["query"]), min(N_TRACK, len(tr["query"])), replace=False)
    query, track_xy, track_vis = T(tr["query"][pick]), T(tr["tracks"][:, pick]), torch.tensor(tr["visible"][:, pick], device=dev)
    dfi, dbary = surface_samples(60000, 1)
    dense = at(ref_v, dfi, dbary)
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
H, Wd = frames[0].shape[:2]

huber = torch.nn.functional.huber_loss
rot = torch.zeros(B, 3, device=dev)
trans = torch.zeros(len(roots), 3, device=dev)
prev_soles = ref_v[sole_idx] if soles else None
result, joints, t0 = [], [], time.time()
for w0 in range(0, len(frames), WIN):
    ids = list(range(w0, min(w0 + WIN, len(frames))))
    K = len(ids)
    dts, mpix = [], []
    for i in ids:
        m = mask_of(frames[i])
        dts.append(torch.tensor(cv2.distanceTransform(1 - m, cv2.DIST_L2, 5), device=dev)[None])
        ys, xs = np.nonzero(m)
        sel = np.random.default_rng(i).choice(len(xs), N_MASK, replace=len(xs) < N_MASK)
        mpix.append(T(np.stack([xs[sel], ys[sel]], 1)))
    dts, mpix = torch.stack(dts), torch.stack(mpix)
    prev_rot, prev_trans = rot, trans
    rot = prev_rot.expand(K, B, 3).clone().requires_grad_(True)  # each window starts at the last solved frame
    trans = prev_trans.expand(K, len(roots), 3).clone().requires_grad_(True)
    if tracks_path:
        txy, tvis = track_xy[ids], track_vis[ids].float()[..., None]
    opt = torch.optim.Adam([rot, trans], lr=0.01)
    for it in range(ITERS):
        v, mats = skin(basis_of(rot, trans))
        s3 = at(v, fidx, bary)
        s = project(s3)
        grid = torch.stack([s[..., 0] / (Wd - 1), s[..., 1] / (H - 1)], -1) * 2 - 1
        loss = torch.nn.functional.grid_sample(dts, grid[:, None], align_corners=True).mean((1, 2, 3)).sum()
        loss = loss + W["cover"] * torch.cdist(mpix, s).min(2).values.mean(1).sum()
        all_rot, all_trans = torch.cat([prev_rot[None], rot]), torch.cat([prev_trans[None], trans])
        loss = loss + W["smooth"] * ((all_rot[1:] - all_rot[:-1]).pow(2).sum() + (all_trans[1:] - all_trans[:-1]).pow(2).sum())
        loss = loss + W["depth"] * (((mats[..., :3, 3] - ref_joints) @ axis) / size).pow(2).mean(1).sum()
        loss = loss + W["sym"] * symmetry(mats)
        loss = loss + W["bdir"] * ((mats[..., deforming, :3, 1] @ axis) - ref_dir_depth).pow(2).sum()
        loss = loss + W["ref"] * rot[:, 1:].pow(2).sum()
        if att_off is not None:
            pos = (mats[:, att_bone, :3, :3] @ att_off[..., None])[..., 0] + mats[:, att_bone, :3, 3]
            loss = loss + W["attach"] * ((pos - mats[:, att_root, :3, 3]) / size).pow(2).sum()
        if grounded:
            loss = loss + W["floor"] * (torch.relu(floor - s3[..., 2]) / size).pow(2).mean(1).sum()
        if soles:
            flat, slide = feet(v, prev_soles)
            loss = loss + W["flat"] * flat + W["slide"] * slide
        if tracks_path:
            d = project(at(v, tfidx, tbary)) - txy
            per = (huber(d, torch.zeros_like(d), delta=3.0, reduction="none") * tvis).sum((1, 2))
            loss = loss + W["track"] * (per / tvis.sum((1, 2)).clamp(min=1)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    rot, trans = rot.detach(), trans.detach()
    result += list(basis_of(rot, trans).cpu().numpy())
    joints += list(mats[..., :3, 3].detach().cpu().numpy())
    rot, trans = rot[-1], trans[-1]
    prev_soles = v[-1, sole_idx].detach() if soles else None
    print(f"frames {ids[0]}-{ids[-1]}: loss {loss.item() / K:.2f} per frame ({time.time() - t0:.0f}s)", flush=True)

np.savez(out, basis=np.array(result), joints=np.array(joints), names=names)
print("wrote", out)
