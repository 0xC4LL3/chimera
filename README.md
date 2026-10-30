# Chimera

Describe a move. Capture it onto your creature's own skeleton.

Chimera turns a 3D model and a text prompt into an editable skeletal animation. LTX-2.5 generates a motion video from a render of the model. Chimera then fits the original rig to that video and exports GLB, FBX and BVH.

The input can be a new character with no animation clips. Models without a skeleton can be rigged with SkinTokens. The web app uses the Comfy SDK with a local ComfyUI backend. All inference in the shipped workflows uses open-weight models.

## Try the captured example

The [chimera input](app/samples/chimera.glb) has a custom 35-bone rig and no animation clips. The example turns a rear-and-roar prompt into a four-second skeletal animation.

Download the [animated GLB](examples/chimera-rear.glb), [FBX](examples/chimera-rear.fbx) or [BVH](examples/chimera-rear.bvh). GLB and FBX include root-motion and in-place clips. BVH contains the root-motion clip. These files can be inspected without downloading inference models.

The [generated reference video](examples/chimera-reference.mp4) and [capture record](examples/chimera-rear.json) preserve the prompt, video seed and geometric check. The chosen result was also reviewed from four angles. [CREDITS.md](CREDITS.md) covers its source asset.

An original four-bone desk lamp demonstrates mechanical articulation. Try its [input model](examples/desk-lamp-input.glb), [captured bow as GLB](examples/desk-lamp-bow.glb) or [FBX](examples/desk-lamp-bow.fbx). The [reference video](examples/desk-lamp-reference.mp4) and [capture record](examples/desk-lamp-bow.json) show the same pipeline applied to rigid parts.

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
