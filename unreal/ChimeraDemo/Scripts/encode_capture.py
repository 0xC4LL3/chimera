"""Encode a completed camera capture using the local ffmpeg installation.

Run from WSL: python3 unreal/ChimeraDemo/Scripts/encode_capture.py <capture-folder>
"""
import json
from pathlib import Path
import subprocess
import sys

source = Path(sys.argv[1]).resolve()
info = json.loads((source / "capture.json").read_text())
status = json.loads((source / "status.json").read_text())
if status["status"] != "complete":
    raise RuntimeError("The Unreal camera capture is not complete.")
out = Path(__file__).resolve().parents[3] / "runs/submission/unreal" / source.name
out.mkdir(parents=True, exist_ok=True)
for shot in info["shots"]:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(info["fps"]),
                    "-i", str(source / shot / "%04d.png"), "-an", "-c:v", "libx264", "-threads", "4",
                    "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(out / (shot + ".mp4"))], check=True)
(out / "capture.json").write_text(json.dumps(info, indent=2))
print(out)
