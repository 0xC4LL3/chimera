# Torch version of a rig dumped by dump_rig.py: forward kinematics, skinning and projection
# through the reference camera, all differentiable. Shared by the fitter and the cleanup.
import json, os
import numpy as np, torch

dev = "cuda"
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)


def rodrigues(r):
    th = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = r / th
    kx = torch.zeros(*r.shape[:-1], 3, 3, device=dev)
    kx[..., 0, 1], kx[..., 0, 2], kx[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    kx = kx - kx.transpose(-1, -2)
    th = th[..., None]
    return torch.eye(3, device=dev) + torch.sin(th) * kx + (1 - torch.cos(th)) * kx @ kx


class Rig:
    def __init__(self, run):
        rig = np.load(os.path.join(run, "rig.npz"))
        cam = json.load(open(os.path.join(run, "camera.json")))
        self.names = rig["names"].tolist()
        self.parents = rig["parents"].tolist()
        self.B = len(self.parents)
        self.rest, self.ref_basis = T(rig["rest"]), T(rig["ref_basis"])
        self.verts, self.weights = T(rig["verts"]), T(rig["weights"])
        self.faces = torch.tensor(rig["faces"], device=dev)
        self.ref_verts = T(rig["ref_verts"])
        self.local_rest = torch.stack([self.rest[i] if p < 0 else torch.linalg.inv(self.rest[p]) @ self.rest[i]
                                       for i, p in enumerate(self.parents)])
        self.inv_rest = torch.linalg.inv(self.rest)
        depth = [0] * self.B
        for i, p in enumerate(self.parents):  # parents come before children
            depth[i] = 0 if p < 0 else depth[p] + 1
        self.levels = [(torch.tensor([i for i in range(self.B) if depth[i] == d], device=dev),
                        torch.tensor([self.parents[i] for i in range(self.B) if depth[i] == d], device=dev))
                       for d in range(1, max(depth) + 1)]
        self.cam = cam
        # One view, or several side by side in the frame (render_ref.py --split): camera, intrinsics, x range.
        views = cam.get("views", [{"K": cam["K"], "cam_to_world": cam["cam_to_world"], "box": [0, cam["width"]]}])
        self.views = [(torch.linalg.inv(T(v["cam_to_world"])), T(v["K"]), v["box"]) for v in views]
        self.w2c, self.K = self.views[0][:2]

    def fk(self, basis):
        # World matrix of every bone from basis matrices, for one frame (B,4,4) or many (N,B,4,4).
        # Bones at the same depth in the hierarchy are done together, one step per level.
        mats = self.local_rest @ basis
        for idx, par in self.levels:
            mats = mats.index_copy(-3, idx, mats[..., par, :, :] @ mats[..., idx, :, :])
        return mats

    def skin_points(self, mats, idx):
        # Skinned positions of selected vertices, given world bone matrices from fk() (any batch shape).
        v1 = torch.cat([self.verts[idx], torch.ones_like(self.verts[idx, :1])], 1)
        return torch.einsum("vb,...bij,vj->...vi", self.weights[idx], (mats @ self.inv_rest)[..., :3, :], v1)

    def skin(self, basis):
        # Skinned vertices and bone matrices, for one frame of basis matrices or many.
        mats = self.fk(basis)
        deform = mats @ self.inv_rest
        v1 = torch.cat([self.verts, torch.ones_like(self.verts[:, :1])], 1)
        return torch.einsum("vb,...bij,vj->...vi", self.weights, deform[..., :3, :], v1), mats

    def project(self, p, view=0):
        # Pixel positions of points (..., 3) in one view; the Blender camera looks down -Z.
        w2c, K, _ = self.views[view]
        c = p @ w2c[:3, :3].T + w2c[:3, 3]
        return torch.stack([K[0, 0] * c[..., 0] / -c[..., 2] + K[0, 2], K[1, 2] - K[1, 1] * c[..., 1] / -c[..., 2]], -1)
