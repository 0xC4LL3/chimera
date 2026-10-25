# Chimera web app: a thin front end over the two ComfyUI workflows, through the Comfy SDK.
# The same code runs against local ComfyUI (via comfy-api-proxy), Comfy Cloud or a Comfy API
# deployment; only COMFY_BASE_URL (and COMFY_API_KEY for hosted backends) changes.
# Run: COMFY_BASE_URL=http://127.0.0.1:8189 app/.venv/bin/uvicorn server:app --app-dir app --port 8000
import json, random, shutil, struct, uuid
from pathlib import Path
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from comfy_sdk import Comfy

HERE = Path(__file__).parent
WORKFLOWS = HERE.parent / "workflows"
DATA = HERE / "data"  # uploaded creatures and downloaded results
DATA.mkdir(exist_ok=True)
SAMPLES = HERE / "samples"  # Quaternius (CC0) and Scenario (CC BY 4.0) creatures

comfy = Comfy(timeout=120)
app = FastAPI()
jobs = {}  # job id -> Job


def creature_path(creature):
    files = list((DATA / creature).glob("*.gl*"))
    if not files:
        raise HTTPException(404, "unknown creature")
    return files[0]


# The camera is the app's choice, so the app words it; the user's prompt only describes the action.
FRAMING = {
    True: ("Split screen showing two synchronized views of the same creature at the same moment: a three-quarter "
           "view from the front right on the left and a three-quarter view from the front left on the right. {} "
           "Both views stay perfectly in sync. Static locked cameras, plain studio background, 3D render."),
    False: "{} Side view, static locked camera, plain studio background, 3D render.",
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
    # Whether the model has a skeleton, and its animations (possible starting poses), from the glTF JSON.
    data = path.read_bytes()
    doc = json.loads(data[20:20 + struct.unpack_from("<I", data, 12)[0]] if data[:4] == b"glTF" else data)
    return {"bones": sum(len(s["joints"]) for s in doc.get("skins", [])),
            "animations": [a.get("name", f"Animation {i + 1}") for i, a in enumerate(doc.get("animations", []))]}


def add_creature(name, src):
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
             grounded: bool = Form(True), auto_rig: bool = Form(False), split_screen: bool = Form(True),
             count: int = Form(3)):
    ids = []
    for _ in range(count):
        wf = workflow("generate_api.json", creature, action, frame, grounded, auto_rig, split_screen)
        wf.set_input("pos", "text", FRAMING[split_screen].format(prompt.strip()))
        wf.set_input("noise", "noise_seed", random.randrange(2**48))
        ids.append(submit(wf))
    return {"jobs": ids}


@app.post("/api/capture")
def capture(creature: str = Form(), clip: str = Form(), action: str = Form(""), frame: int = Form(0),
            grounded: bool = Form(True), auto_rig: bool = Form(False), split_screen: bool = Form(True),
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
