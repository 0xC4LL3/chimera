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
        self.cam = cam
        self.w2c = torch.linalg.inv(T(cam["cam_to_world"]))
        self.K = T(cam["K"])

    def fk(self, basis):
        # World matrix of every bone for one frame of basis matrices (B,4,4).
        mats = []
        for i, p in enumerate(self.parents):
            mats.append(self.local_rest[i] @ basis[i] if p < 0 else mats[p] @ self.local_rest[i] @ basis[i])
        return torch.stack(mats)

    def skin(self, basis):
        mats = self.fk(basis)
        deform = mats @ self.inv_rest
        v1 = torch.cat([self.verts, torch.ones_like(self.verts[:, :1])], 1)
        return torch.einsum("vb,bij,vj->vi", self.weights, deform[:, :3], v1), mats

    def project(self, p):
        c = (self.w2c[:3, :3] @ p.T + self.w2c[:3, 3:]).T  # Blender camera looks down -Z
        K = self.K
        return torch.stack([K[0, 0] * c[:, 0] / -c[:, 2] + K[0, 2], K[1, 2] - K[1, 1] * c[:, 1] / -c[:, 2]], 1)
