from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from main import analyze_video


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "web_runtime"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
RESULTS_DIR = RUNTIME_DIR / "results"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _result_asset_urls(job_id: str, assets: dict[str, str]) -> dict[str, str | None]:
    urls: dict[str, str | None] = {}
    for key, value in assets.items():
        path = Path(value)
        urls[key] = f"/results/{job_id}/{path.name}" if path.exists() else None
    return urls


def _update_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def _run_analysis_job(job_id: str, video_path: Path, result_dir: Path) -> None:
    _update_job(job_id, status="running", started_at=_timestamp(), message="正在执行视频分析，请稍候...")
    try:
        result = analyze_video(video_file=video_path, output_dir=result_dir)
        result["assets"] = _result_asset_urls(job_id, result["assets"])
        _update_job(
            job_id,
            status="completed",
            completed_at=_timestamp(),
            message="分析完成。",
            result=result,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            completed_at=_timestamp(),
            message=str(exc),
            error=traceback.format_exc(),
        )


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def create_analysis():
    uploaded_file = request.files.get("video")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "请先选择一个视频文件。"}), 400

    job_id = uuid.uuid4().hex[:12]
    upload_dir = UPLOADS_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(uploaded_file.filename) or f"{job_id}.mp4"
    video_path = upload_dir / filename
    uploaded_file.save(video_path)

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": _timestamp(),
            "message": "任务已创建，准备开始分析...",
            "result": None,
            "error": None,
        }

    thread = threading.Thread(target=_run_analysis_job, args=(job_id, video_path, result_dir), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "任务不存在。"}), 404

    return jsonify(job)


@app.get("/results/<job_id>/<path:filename>")
def serve_result_file(job_id: str, filename: str):
    job_dir = RESULTS_DIR / job_id
    file_path = job_dir / filename
    if not file_path.exists():
        return jsonify({"error": "结果文件不存在。"}), 404

    mimetype = "video/mp4" if file_path.suffix.lower() == ".mp4" else None
    response = send_file(file_path, mimetype=mimetype, conditional=True, as_attachment=False)
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Content-Disposition"] = f'inline; filename="{file_path.name}"'
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
