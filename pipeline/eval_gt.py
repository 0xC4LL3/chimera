# Score a fitted pose against ground-truth joints from render_clip.py.
# Usage: python pipeline/eval_gt.py <run_dir> <pose.npz>
# Errors are in % of the skeleton's size; "depth" is along the camera axis, "image" across it.
import sys, os, json
import numpy as np

run, pose = sys.argv[1:3]
gt = np.load(os.path.join(run, "gt.npz"))["joints"]
fit = np.load(pose)["joints"][:len(gt)]
cam = json.load(open(os.path.join(run, "camera.json")))
axis = -np.array(cam["cam_to_world"])[:3, 2]  # Blender camera looks down -Z
size = np.linalg.norm(gt[0].max(0) - gt[0].min(0)) / 100


def report(name, pred):
    d = pred - gt
    depth = np.abs(d @ axis)
    image = np.linalg.norm(d - np.outer(d @ axis, axis).reshape(d.shape), axis=-1)
    print(f"{name:7s} mean {np.linalg.norm(d, axis=-1).mean() / size:5.1f}%  "
          f"depth {depth.mean() / size:5.1f}%  image {image.mean() / size:5.1f}%  "
          f"worst frame {np.linalg.norm(d, axis=-1).mean(1).max() / size:5.1f}%")


report("fit", fit)
report("static", np.repeat(gt[:1], len(gt), 0))
