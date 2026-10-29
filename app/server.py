# Chimera web app: a thin front end over the two ComfyUI workflows, through the Comfy SDK.
# The same code runs against local ComfyUI (via comfy-api-proxy), Comfy Cloud or a Comfy API
# deployment; only COMFY_BASE_URL (and COMFY_API_KEY for hosted backends) changes.
# Run: COMFY_BASE_URL=http://127.0.0.1:8189 app/.venv/bin/uvicorn server:app --app-dir app --port 8000
import json, random, shutil, struct, uuid
import httpx
from pathlib import Path
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from comfy_sdk import Comfy
from comfy_sdk.exceptions import JobFailed

HERE = Path(__file__).parent
WORKFLOWS = HERE.parent / "workflows"
DATA = HERE / "data"  # uploaded creatures and downloaded results
DATA.mkdir(exist_ok=True)
SAMPLES = HERE / "samples"  # Quaternius (CC0) and Scenario (CC BY 4.0) creatures

comfy = Comfy(timeout=120)
app = FastAPI()
jobs = {}  # job id -> Job


@app.exception_handler(httpx.ConnectError)
async def backend_unavailable(request, error):
    return JSONResponse(status_code=503, content={"detail": "Cannot reach ComfyUI. Start the backend with ./run.sh and try again."})


MODELS = (".glb", ".fbx")


def creature_path(creature):
    if (DATA / creature / "rigged.glb").exists():  # auto-rigged after upload (/api/rig)
        return DATA / creature / "rigged.glb"
    files = [p for p in (DATA / creature).glob("*") if p.suffix.lower() in MODELS]
    if not files:
        raise HTTPException(404, "unknown creature")
    return files[0]


# The camera is the app's choice, so the app words it; the user's prompt only describes the action, which the
# workflow first rewrites for this creature's body (a local vision-language model looking at the first frame).
FRAMING = {
    True: ("Split screen showing two synchronized views of the same creature at the same moment: a three-quarter "
           "view from the front right on the left and a three-quarter view from the front left on the right. {action} "
           "Both views stay perfectly in sync. Static locked cameras, plain studio background, 3D render."),
    False: "{action} Side view, static locked camera, plain studio background, 3D render.",
}


def workflow(name, creature, action, frame, grounded, auto_rig, split_screen):
    wf = comfy.workflows.from_file(WORKFLOWS / name)
    wf.set_input("creature", "model_file", comfy.assets.from_file(creature_path(creature)))
    wf.set_input("creature", "start_action", action)
    wf.set_input("creature", "start_frame", frame)
    wf.set_input("creature", "grounded", grounded)
    wf.set_input("creature", "auto_rig", auto_rig)
    wf.set_input("creature", "split_screen", split_screen)
    return wf


def submit(wf):
    job = comfy.submit(wf)
    jobs[job.id] = job
    return job.id


def model_info(path):
    # Whether the model has a skeleton, its animations (possible starting poses) and a GLB the browser
    # can show. A glTF says all this in its JSON; any other format is read by ComfyUI (Blender).
    if path.suffix.lower() != ".fbx":
        data = path.read_bytes()
        doc = json.loads(data[20:20 + struct.unpack_from("<I", data, 12)[0]] if data[:4] == b"glTF" else data)
        return {"bones": sum(len(s["joints"]) for s in doc.get("skins", [])),
                "animations": [a.get("name", f"Animation {i + 1}") for i, a in enumerate(doc.get("animations", []))],
                "preview": f"/data/{path.parent.name}/{path.name}"}
    try:
        out = preview(path)
    except JobFailed:
        raise HTTPException(400, "Can't read this FBX. Try binary FBX 7.1+ or GLB.")
    return {**json.loads((out / "info.json").read_text()), "preview": url(out / "preview.glb"), "skeleton": skeleton_url(out)}


def url(path):
    return "/" + str(path.relative_to(HERE))


def skeleton_url(folder):
    return url(folder / "skeleton.glb") if (folder / "skeleton.glb").exists() else None


def run_into(wf, out, timeout=600):
    # Run a workflow and save its output files into a folder.
    out.mkdir(exist_ok=True)
    for o in comfy.run(wf, timeout=timeout).outputs:
        o.to_file(out / o.name)
    return out


def preview(path):
    # A GLB the browser can show, the model's bone count and animations, and for a rigged model its bones.
    wf = comfy.workflows.from_file(WORKFLOWS / "preview_api.json")
    wf.set_input("preview", "model_file", comfy.assets.from_file(path))
    return run_into(wf, path.parent / "preview")


@app.post("/api/skeleton/{creature}")
def skeleton(creature: str):
    # The bones of a model that came with a rig, for the preview.
    return {"skeleton": skeleton_url(preview(creature_path(creature)))}


@app.post("/api/rig/{creature}")
def rig(creature: str):
    # Auto-rig an unrigged model once, right after upload: every take then uses this same rig.
    wf = comfy.workflows.from_file(WORKFLOWS / "rig_api.json")
    wf.set_input("rig", "model_file", comfy.assets.from_file(creature_path(creature)))
    try:
        out = run_into(wf, DATA / creature / "rig", timeout=1800)
    except JobFailed:
        raise HTTPException(500, "Auto-rigging failed. Try again, or upload a rigged model.")
    (out / "rigged.glb").rename(DATA / creature / "rigged.glb")
    return {"bones": json.loads((out / "info.json").read_text())["bones"], "skeleton": skeleton_url(out)}


def add_creature(name, src):
    if Path(name).suffix.lower() not in MODELS:
        raise HTTPException(400, "Use a .glb or .fbx model")
    creature = uuid.uuid4().hex[:8]
    (DATA / creature).mkdir()
    with open(DATA / creature / Path(name).name, "wb") as f:
        shutil.copyfileobj(src, f)
    return {"creature": creature, **model_info(creature_path(creature))}


@app.post("/api/creature")
def upload_creature(file: UploadFile):
    return add_creature(file.filename, file.file)


@app.post("/api/sample/{name}")
def sample_creature(name: str):
    path = SAMPLES / f"{Path(name).name}.glb"
    if not path.exists():
        raise HTTPException(404, "unknown sample")
    with open(path, "rb") as f:
        return add_creature(path.name, f)


@app.post("/api/generate")
def generate(creature: str = Form(), prompt: str = Form(), action: str = Form(""), frame: int = Form(0),
             grounded: bool = Form(True), auto_rig: bool = Form(False), split_screen: bool = Form(False),
             count: int = Form(3, ge=1, le=4)):
    ids = []
    for _ in range(count):
        wf = workflow("generate_api.json", creature, action, frame, grounded, auto_rig, split_screen)
        wf.set_input("rewrite", "prompt", "Action: " + prompt.strip())
        wf.set_input("framing", "value", FRAMING[split_screen])
        wf.set_input("noise", "noise_seed", random.randrange(2**48))
        ids.append(submit(wf))
    return {"jobs": ids}


@app.post("/api/capture")
def capture(creature: str = Form(), clip: str = Form(), action: str = Form(""), frame: int = Form(0),
            grounded: bool = Form(True), auto_rig: bool = Form(False), split_screen: bool = Form(False),
            name: str = Form("attack"), loop: bool = Form(True)):
    video = next((DATA / clip).glob("save_clip_*"), None)
    if video is None:
        raise HTTPException(404, "unknown clip")
    wf = workflow("capture_api.json", creature, action, frame, grounded, auto_rig, split_screen)
    wf.set_input("clip", "file", comfy.assets.from_file(video))
    wf.set_input("cleanup", "loop", loop)
    wf.set_input("export", "clip_name", name)
    return {"job": submit(wf)}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    job = jobs[job_id].refresh()
    files = []
    if job.status == "succeeded":
        folder = DATA / job_id
        folder.mkdir(exist_ok=True)
        for o in job.outputs:
            path = folder / f"{o.node_id}_{o.name}"
            if not path.exists():
                o.to_file(path)
            files.append(f"/data/{job_id}/{path.name}")
    return {"status": job.status, "error": str(job.error) if job.error else None, "files": files}


app.mount("/data", StaticFiles(directory=DATA), name="data")
app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
