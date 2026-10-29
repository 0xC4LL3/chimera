# Chimera

Describe a move. Capture it onto your creature's own skeleton.

Chimera turns a 3D model and a text prompt into an editable skeletal animation. LTX-2.5 generates a motion video from a render of the model. Chimera then fits the original rig to that video and exports GLB, FBX and BVH.

The input can be a new character with no animation clips. Models without a skeleton can be rigged with SkinTokens. The web app uses the Comfy SDK with a local ComfyUI backend. All inference in the shipped workflows uses open-weight models.

## Run locally

Follow [SETUP.md](SETUP.md) and [MODELS.md](MODELS.md), then run:

```bash
./run.sh
```

Open <http://127.0.0.1:8000>. Upload a GLB or FBX, describe a short move and click **Animate**. Each take shows the generated video and the captured animation. Open a result to orbit the model, compare root motion with the in-place version and download it.

The development machine uses an RTX 5080 with 16 GB VRAM, 64 GB RAM and Linux under WSL2. Setup documentation lists the exact dependencies and model files. Other hardware configurations still need validation.

## How it works

1. **Prepare the model.** Keep its rig or create one with SkinTokens. Estimate its facing direction with SigLIP 2.
2. **Generate a take.** Render a side view. Qwen rewrites the action using the visible anatomy. LTX-2.5 animates the reference image.
3. **Fit the rig.** AllTracker follows points through the video. Chimera optimizes bone poses to match those tracks and the silhouette. Symmetry, ground contact and smoothness guide the fit.
4. **Check the result.** Render the captured motion over the video to measure geometric agreement. If a side-view take fits poorly, the app retries with two synchronized three-quarter views.
5. **Export.** Smooth and close the motion into a loop. GLB and FBX contain root-motion and in-place actions. BVH contains the root-motion action.

The central contribution is fitting generated motion back onto the supplied mesh and skeleton through a known camera. The model keeps its own topology and bone names. A new character can gain animation without an existing motion library.

## What to expect

Short actions with a clear silhouette work best. Review the captured animation from several angles before using it in a game. The overlap check measures geometric agreement with the video. It cannot judge whether the video performed the requested action.

Fast limbs, camera-facing turns and invented anatomy can produce poor captures. A single view also leaves hidden motion ambiguous. Auto-rigging may miss small parts or place joints poorly. A carefully authored rig gives more control over unusual anatomy.

## Repository

- `app/`: the upload, take selection, 3D preview and download interface.
- `nodes.py` and `pipeline/`: ComfyUI nodes and the rig fitting pipeline.
- `workflows/`: API-format workflows for preview, rigging, generation and capture, plus a combined workflow.
- [CREDITS.md](CREDITS.md): model, asset and software credits.

## License

Chimera's code is [GPL-3.0](LICENSE). Third-party models and sample assets retain their own licenses, listed in [CREDITS.md](CREDITS.md) and [MODELS.md](MODELS.md).
