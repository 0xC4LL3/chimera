# Chimera web app: a thin front end over the two ComfyUI workflows, through the Comfy SDK.
# The same code runs against local ComfyUI (via comfy-api-proxy), Comfy Cloud or a Comfy API
# deployment; only COMFY_BASE_URL (and COMFY_API_KEY for hosted backends) changes.
# Run: COMFY_BASE_URL=http://127.0.0.1:8189 app/.venv/bin/uvicorn server:app --app-dir app --port 8000
import random, shutil, uuid
from pathlib import Path
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from comfy_sdk import Comfy

HERE = Path(__file__).parent
WORKFLOWS = HERE.parent / "workflows"
DATA = HERE / "data"  # uploaded creatures and downloaded results
DATA.mkdir(exist_ok=True)

comfy = Comfy(timeout=120)
app = FastAPI()
jobs = {}  # job id -> Job


def creature_path(creature):
    files = list((DATA / creature).glob("*.gl*"))
    if not files:
        raise HTTPException(404, "unknown creature")
    return files[0]


def workflow(name, creature, action, frame, grounded, auto_rig):
    wf = comfy.workflows.from_file(WORKFLOWS / name)
    wf.set_input("creature", "model_file", comfy.assets.from_file(creature_path(creature)))
    wf.set_input("creature", "start_action", action)
    wf.set_input("creature", "start_frame", frame)
    wf.set_input("creature", "grounded", grounded)
    wf.set_input("creature", "auto_rig", auto_rig)
    return wf


def submit(wf):
    job = comfy.submit(wf)
    jobs[job.id] = job
    return job.id


@app.post("/api/creature")
def upload_creature(file: UploadFile):
    creature = uuid.uuid4().hex[:8]
    (DATA / creature).mkdir()
    with open(DATA / creature / Path(file.filename).name, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"creature": creature}


@app.post("/api/generate")
def generate(creature: str = Form(), prompt: str = Form(), action: str = Form(""), frame: int = Form(0),
             grounded: bool = Form(True), auto_rig: bool = Form(False), count: int = Form(3)):
    ids = []
    for _ in range(count):
        wf = workflow("generate_api.json", creature, action, frame, grounded, auto_rig)
        wf.set_input("pos", "text", prompt)
        wf.set_input("noise", "noise_seed", random.randrange(2**48))
        ids.append(submit(wf))
    return {"jobs": ids}


@app.post("/api/capture")
def capture(creature: str = Form(), clip: str = Form(), action: str = Form(""), frame: int = Form(0),
            grounded: bool = Form(True), auto_rig: bool = Form(False), name: str = Form("attack"), loop: bool = Form(True)):
    video = next((DATA / clip).glob("save_clip_*"), None)
    if video is None:
        raise HTTPException(404, "unknown clip")
    wf = workflow("capture_api.json", creature, action, frame, grounded, auto_rig)
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
app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
