# Model files

The shipped workflows run open-weight models locally through ComfyUI. They use native ComfyUI loaders and Chimera's custom nodes. The app does not call a hosted model API.

## ComfyUI models

Download these exact files into the indicated folders under `ComfyUI/models`. The five files total about 53.2 GB. Links point to the publishers' repositories.

LTX-2.5 requires a Hugging Face account. Sign in and accept the access terms on the [LTX-2.5 model page](https://huggingface.co/Lightricks/LTX-2.5) before downloading its files.

| Folder | File and download |
| --- | --- |
| `diffusion_models` | [ltx-2.5-22b-distilled-transformer-nvfp4.safetensors](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors) |
| `vae` | [ltx-2.5-video-vae-bf16.safetensors](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors) |
| `vae` | [ltx-2.5-audio-vae-bf16.safetensors](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors) |
| `text_encoders` | [gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors) |
| `text_encoders` | [qwen3.8_27b_w4a8.safetensors](https://huggingface.co/Comfy-Org/Qwen3.8-27B/resolve/main/text_encoders/qwen3.8_27b_w4a8.safetensors) |

LTX-2.5 generates the motion video. Gemma encodes its prompt. Qwen looks at the rendered creature and expands the requested move into an action description. Both video and audio VAEs are required by the LTX workflow, although the output video is silent.

These file choices have been used on an RTX 5080 with 16 GB VRAM and 64 GB system RAM. The NVFP4 transformer targets Blackwell GPUs. Other GPU configurations need separate validation. Keep the `--cache-ram 12` launch option so ComfyUI can evict cached models under memory pressure.

The [LTX-2.5 model card](https://huggingface.co/Lightricks/LTX-2.5) links its community license. The [ComfyUI Qwen package](https://huggingface.co/Comfy-Org/Qwen3.8-27B) is marked Apache-2.0. Model terms are separate from Chimera's GPL code license.

## Rigging, tracking and facing

| Model | Setup | Source |
| --- | --- | --- |
| SkinTokens / TokenRig | Run `download.py --model` inside `third_party/SkinTokens` as described in [SETUP.md](SETUP.md). This downloads the two checkpoints and Qwen3-0.6B configuration. | [VAST-AI/SkinTokens](https://huggingface.co/VAST-AI/SkinTokens), MIT |
| AllTracker | The capture script downloads its checkpoint on first use. | [aharley/alltracker](https://huggingface.co/aharley/alltracker), MIT |
| SigLIP 2 | The facing script downloads `google/siglip2-so400m-patch14-384` on first use. | [Google's model card](https://huggingface.co/google/siglip2-so400m-patch14-384), Apache-2.0 |

Allow extra disk space for these models, Python dependencies and generated takes. First use needs internet access for model downloads. The browser also loads its model viewer and fonts from public CDNs.
