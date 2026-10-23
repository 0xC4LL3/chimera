# Generate first-last-frame clips with LTX-2.5 on a local ComfyUI.
# Usage: python3 pipeline/gen_video.py <run_dir> "<prompt>" [n_candidates]
# Reads <run_dir>/ref.png (used as first and last frame), writes <run_dir>/cand_<i>.mp4.
import json, sys, time, uuid, urllib.request, os, random

URL = "http://127.0.0.1:8188"
run, prompt = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
W, H, FPS, FRAMES = 1024, 576, 24, 97  # frames % 8 == 1
NEG = ("blurry, low quality, camera motion, camera shake, zoom, background change, "
       "extra limbs, extra wings, extra heads, deformed anatomy, morphing, text, watermark")


def post(path, data, headers):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL + path, data=data, headers=headers)))


def upload(path):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{os.path.basename(path)}"\r\n'
            f"Content-Type: image/png\r\n\r\n").encode() + open(path, "rb").read() + f"\r\n--{b}--\r\n".encode()
    return post("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"})["name"]


def graph(image, seed, prefix):
    # API-format port of ComfyUI's video_ltx2_5_flf2v template, without the prompt enhancer.
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors", "weight_dtype": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "type": "ltxv"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image}},
        "pre": {"class_type": "LTXVPreprocess", "inputs": {"image": ["img", 0], "img_compression": 18}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["clip", 0]}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "frame_rate": FPS}},
        "lat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": W, "height": H, "length": FRAMES, "batch_size": 1}},
        "g0": {"class_type": "LTXVAddGuide", "inputs": {"positive": ["cond", 0], "negative": ["cond", 1], "vae": ["vae", 0],
                                                        "latent": ["lat", 0], "image": ["pre", 0], "frame_idx": 0, "strength": 1.0}},
        "g1": {"class_type": "LTXVAddGuide", "inputs": {"positive": ["g0", 0], "negative": ["g0", 1], "vae": ["vae", 0],
                                                        "latent": ["g0", 2], "image": ["pre", 0], "frame_idx": -1, "strength": 0.7}},
        "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": FRAMES, "frame_rate": FPS, "batch_size": 1, "audio_vae": ["avae", 0]}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["g1", 2], "audio_latent": ["alat", 0]}},
        "guider": {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["unet", 0], "positive": ["g1", 0], "negative": ["g1", 1], "video_cfg": 1, "audio_cfg": 1}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "SamplerEulerAncestral", "inputs": {"eta": 0, "s_noise": 1}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                                                     "sigmas": ["sigmas", 0], "latent_image": ["av", 0]}},
        "sep": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sample", 1]}},
        "crop": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["g1", 0], "negative": ["g1", 1], "latent": ["sep", 0]}},
        "dec": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["crop", 2], "vae": ["vae", 0], "tile_size": 512, "overlap": 64,
                                                           "temporal_size": 64, "temporal_overlap": 16}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": FPS}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix, "format": "mp4"}},
    }


image = upload(os.path.join(run, "ref.png"))
for i in range(n):
    t = time.time()
    seed = random.randint(0, 2**48)
    pid = post("/prompt", json.dumps({"prompt": graph(image, seed, f"chimera/cand_{i}")}).encode(),
               {"Content-Type": "application/json"})["prompt_id"]
    while True:
        h = json.load(urllib.request.urlopen(f"{URL}/history/{pid}")).get(pid)
        if h and h["status"].get("completed") is not None:
            break
        time.sleep(2)
    if not h["status"]["completed"]:
        sys.exit(json.dumps(h["status"], indent=1))
    f = h["outputs"]["save"]["images"][0]
    q = f"filename={f['filename']}&subfolder={f['subfolder']}&type={f['type']}"
    open(os.path.join(run, f"cand_{i}.mp4"), "wb").write(urllib.request.urlopen(f"{URL}/view?{q}").read())
    print(f"cand_{i}.mp4 seed={seed} {time.time() - t:.0f}s", flush=True)
