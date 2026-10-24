# Rig an unrigged mesh with SkinTokens (TokenRig, MIT): skeleton and skin weights from the mesh alone.
# Usage: python pipeline/autorig.py <in_mesh.glb|obj|fbx> <out_rigged.glb> [skintokens_dir]
# SkinTokens wants flash-attn, which has no build for this PyTorch; shims/flash_attn_interface.py
# routes its attention calls to PyTorch's own, for it and its Blender helper process. Its demo's
# command line is reused as is, with gradio (only used by the demo's web UI) stubbed out.
import os, sys, types
from argparse import Namespace

src, dst = (os.path.abspath(p) for p in sys.argv[1:3])
root = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "third_party", "SkinTokens"))
shims = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shims")
sys.path[:0] = [shims, root]
os.environ["PYTHONPATH"] = os.pathsep.join([shims, os.environ.get("PYTHONPATH", "")])  # for its Blender helper too
sys.modules["gradio"] = types.ModuleType("gradio")
os.chdir(root)  # the demo loads checkpoints and its Blender helper by relative path

import transformers
_from_config = transformers.AutoModelForCausalLM.from_config
transformers.AutoModelForCausalLM.from_config = lambda *a, **kw: _from_config(*a, **{**kw, "attn_implementation": "sdpa"})

import demo
demo.start_bpy_server()
demo.wait_for_bpy_server(timeout=120)
demo.run_cli(Namespace(input=src, output=dst, top_k=5, top_p=0.95, temperature=1.0, repetition_penalty=2.0,
                       num_beams=10, use_skeleton=False, use_transfer=True, use_postprocess=False,
                       model_ckpt=demo.MODEL_CKPTS[0], hf_path=None))
print("rigged", dst)
