# Local setup

The development machine runs Linux under WSL2 with an RTX 5080 (16 GB VRAM), 64 GB system RAM and 60 GB available to WSL. ComfyUI uses Python 3.13 and PyTorch 2.14.1 with CUDA 13.0. Other hardware configurations have not been verified. The capture scripts require CUDA.

## 1. Prepare ComfyUI

Install [ComfyUI](https://github.com/Comfy-Org/ComfyUI#manual-install-windows-linux) with a Python 3.13 environment at `ComfyUI/.venv`. Follow its NVIDIA instructions for PyTorch. Chimera was developed against ComfyUI commit `bc5f880` from 23 October 2026. It needs the native LTX-2.5 and Qwen text generation nodes included in that version.

Put this repository at `ComfyUI/custom_nodes/chimera`. Run the following commands from the Chimera directory. The commands use [uv](https://docs.astral.sh/uv/) to install packages into the existing environments.

```bash
export COMFYUI_DIR="$HOME/ComfyUI"
uv pip install --python "$COMFYUI_DIR/.venv/bin/python" -r requirements.txt
```

Set `COMFYUI_DIR` to your actual ComfyUI directory if it differs. Blender's Python module is installed by `requirements.txt`; a separate Blender application is optional.

## 2. Install the rigging and tracking code

```bash
mkdir -p third_party
git clone https://github.com/VAST-AI-Research/SkinTokens.git third_party/SkinTokens
git -C third_party/SkinTokens checkout 273b691
git clone https://github.com/aharley/alltracker.git third_party/alltracker
git -C third_party/alltracker checkout e755313
(
  cd third_party/SkinTokens
  "$COMFYUI_DIR/.venv/bin/python" download.py --model
)
```

The SkinTokens download installs its rigging checkpoints and the Qwen3-0.6B configuration. Chimera supplies the inference dependencies and a PyTorch attention adapter in `pipeline/shims`. AllTracker and SigLIP 2 download their weights on first use, so the first run needs internet access.

## 3. Download the video and prompt models

Follow [MODELS.md](MODELS.md) for the exact files and destinations. The workflows require both LTX-2.5 and the local Qwen prompt rewriter.

## 4. Install the app

```bash
uv venv --python 3.12 app/.venv
uv pip install --python app/.venv/bin/python -r app/requirements.txt
```

## 5. Run

```bash
./run.sh
```

Open [Chimera](http://127.0.0.1:8000). The launcher starts ComfyUI on port 8188, the Comfy SDK proxy on 8189 and the app on 8000. It reuses services already answering on those ports. Ctrl+C stops the processes this invocation started. Logs are saved in `runs/`.

ComfyUI uses `--cache-ram 12` to leave 12 GB of RAM headroom when managing cached models. Run one GPU workload at a time. Closing another application window may leave its GPU process running.

Upload a GLB or FBX, wait for the rig preview, enter a short action and generate takes. Inspect the generated video and the captured animation before exporting. The overlap check measures how closely the rig follows the video. You judge whether it performs the requested action.

## Files and troubleshooting

- Uploads and downloaded results are in `app/data/`. Pipeline intermediates are in `ComfyUI/output/chimera/`. Export animations before removing old run folders.
- A rigged GLB contains its textures. FBX files with external texture images need those images packed into the file for a textured result.
- If the app cannot connect, inspect `runs/app.log`, `runs/proxy.log` and `runs/comfyui.log`.
- Restart ComfyUI after changing node or pipeline code. Cached node results can otherwise preserve an earlier result.
- A missing `TextGenerate` or LTX node means ComfyUI needs updating. A missing model error names the required file in [MODELS.md](MODELS.md).
- This setup has been checked against the installed development environment. A clean machine installation has not yet been verified.
