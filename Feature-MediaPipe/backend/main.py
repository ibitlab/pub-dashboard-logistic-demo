"""Warehouse Activity Intelligence — FastAPI Backend."""

import asyncio
import base64
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from services.database import Database
from services.classifier import ActivityClassifier
from services.kpi_engine import KPIEngine


db = Database()
classifier = ActivityClassifier()
kpi_engine = KPIEngine(db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init()
    classifier.load()
    yield
    await db.close()


app = FastAPI(
    title="Warehouse Activity Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- REST endpoints ----------

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/activities")
async def get_activity_classes():
    """Return supported activity classes."""
    return {"classes": classifier.get_classes()}


@app.get("/api/events")
async def get_events(limit: int = 100, offset: int = 0):
    """Return recent events."""
    events = await db.get_events(limit=limit, offset=offset)
    return {"events": events, "total": await db.count_events()}


@app.get("/api/metrics")
async def get_metrics():
    """Return current KPI metrics."""
    return await kpi_engine.compute_metrics()


@app.get("/api/metrics/timeline")
async def get_timeline(minutes: int = 30):
    """Return activity timeline for the last N minutes."""
    return await kpi_engine.get_timeline(minutes=minutes)


@app.post("/api/train/upload")
async def upload_training_video(
    file: UploadFile = File(...),
    label: str = Form(...),
):
    """Upload a labeled video clip for training.

    IMPORTANT: Videos must be H.264 encoded for reliable browser playback and seeking.
    VP9/AV1 codecs cause frame seeking issues in the Pose Preview slider.
    Re-encode if needed: ffmpeg -i input.mp4 -c:v libx264 -movflags +faststart output.mp4
    """
    contents = await file.read()
    sample_id = await classifier.add_training_sample(contents, label)
    return {"sample_id": sample_id, "label": label, "status": "stored"}


@app.post("/api/train/start")
async def start_training():
    """Trigger model (re-)training on collected samples."""
    result = classifier.train()
    return result


@app.get("/api/train/status")
async def training_status():
    """Get training status and model info."""
    return classifier.get_status()


@app.post("/api/predict")
async def predict_frame(file: UploadFile = File(...)):
    """Classify a single frame / image."""
    contents = await file.read()
    result = classifier.predict_image(contents)
    return result


@app.post("/api/predict/landmarks")
async def predict_with_landmarks(file: UploadFile = File(...)):
    """Classify a single frame and return full pose landmarks for visualization."""
    contents = await file.read()
    result = classifier.predict_image(contents, include_landmarks=True)
    return result


# ---------- Pose model management ----------

@app.get("/api/models")
async def list_pose_models():
    """List all available pose .task models."""
    return {"models": classifier.list_pose_models()}


@app.post("/api/models/switch")
async def switch_pose_model(filename: str = Form(...)):
    """Switch the active pose model."""
    try:
        result = classifier.switch_pose_model(filename)
        return result
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})


@app.post("/api/models/upload")
async def upload_pose_model(file: UploadFile = File(...)):
    """Upload a custom .task model file."""
    if not file.filename.endswith(".task"):
        return JSONResponse(status_code=400, content={"error": "File must be a .task model"})

    # Same directory the classifier lists and loads from, or the upload never appears
    from services.classifier import MODELS_DIR
    dest = os.path.join(MODELS_DIR, os.path.basename(file.filename))

    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    size_mb = round(len(contents) / (1024 * 1024), 1)
    return {"status": "uploaded", "filename": file.filename, "size_mb": size_mb}


# ---------- Annotation endpoints ----------

ANNOTATION_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "annotations")
os.makedirs(ANNOTATION_DIR, exist_ok=True)

# MediaPipe 33-landmark names (standard ordering)
MP_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

MP_SKELETON = [
    [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
    [9,10],[11,12],[11,13],[13,15],[12,14],[14,16],
    [11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],
    [15,17],[15,19],[15,21],[16,18],[16,20],[16,22],
    [27,29],[27,31],[28,30],[28,32],
]


@app.post("/api/annotations/projects")
async def create_annotation_project(
    file: UploadFile = File(...),
    name: str = Form(""),
):
    """Create an annotation project from an uploaded video."""
    project_id = str(uuid.uuid4())[:8]
    video_filename = file.filename or f"{project_id}.mp4"
    video_path = os.path.join(ANNOTATION_DIR, f"{project_id}_{video_filename}")

    contents = await file.read()
    with open(video_path, "wb") as f:
        f.write(contents)

    # Extract video metadata
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    project_name = name or video_filename.rsplit(".", 1)[0]
    result = await db.create_annotation_project(
        project_id, project_name, video_filename, video_path,
        frame_width, frame_height, total_frames, fps,
    )
    return result


@app.post("/api/annotations/projects/demo")
async def create_annotation_project_from_demo(
    video_url: str = Form(...),
    name: str = Form(""),
):
    """Create an annotation project from a demo video already on disk."""
    project_id = str(uuid.uuid4())[:8]
    # video_url is like /demo/warehouse_walking.mp4
    filename = video_url.split("/")[-1]
    video_path = os.path.join(demo_path, filename)

    if not os.path.exists(video_path):
        return JSONResponse(status_code=404, content={"error": "Demo video not found"})

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    project_name = name or filename.rsplit(".", 1)[0]
    result = await db.create_annotation_project(
        project_id, project_name, filename, video_path,
        frame_width, frame_height, total_frames, fps,
    )
    return result


@app.get("/api/annotations/projects")
async def list_annotation_projects():
    """List all annotation projects."""
    projects = await db.list_annotation_projects()
    return {"projects": projects}


@app.get("/api/annotations/projects/{project_id}")
async def get_annotation_project(project_id: str):
    project = await db.get_annotation_project(project_id)
    if not project:
        return JSONResponse(status_code=404, content={"error": "Project not found"})
    return project


@app.delete("/api/annotations/projects/{project_id}")
async def delete_annotation_project(project_id: str):
    project = await db.get_annotation_project(project_id)
    if not project:
        return JSONResponse(status_code=404, content={"error": "Project not found"})
    await db.delete_annotation_project(project_id)
    return {"status": "deleted"}


@app.get("/api/annotations/projects/{project_id}/frame/{frame_index}")
async def get_frame_image(project_id: str, frame_index: int):
    """Return a single frame as JPEG and its saved keypoints (if any)."""
    project = await db.get_annotation_project(project_id)
    if not project:
        return JSONResponse(status_code=404, content={"error": "Project not found"})

    cap = cv2.VideoCapture(project["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JSONResponse(status_code=404, content={"error": "Frame not found"})

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode()

    saved = await db.get_annotation_frame(project_id, frame_index)

    return {
        "frame_index": frame_index,
        "image": f"data:image/jpeg;base64,{b64}",
        "width": project["frame_width"],
        "height": project["frame_height"],
        "keypoints": saved["keypoints"] if saved else None,
    }


@app.post("/api/annotations/projects/{project_id}/detect/{frame_index}")
async def auto_detect_keypoints(project_id: str, frame_index: int):
    """Run MediaPipe on a single frame and return detected landmarks."""
    project = await db.get_annotation_project(project_id)
    if not project:
        return JSONResponse(status_code=404, content={"error": "Project not found"})

    cap = cv2.VideoCapture(project["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JSONResponse(status_code=404, content={"error": "Frame not found"})

    landmarks = classifier.detect_landmarks(frame)
    if landmarks is None:
        return {"frame_index": frame_index, "detected": False, "keypoints": []}

    # Return as list of {x, y, visibility, name} — normalized 0..1
    keypoints = []
    for i, lm in enumerate(landmarks):
        keypoints.append({
            "x": lm["x"],
            "y": lm["y"],
            "visibility": lm["visibility"],
            "name": MP_LANDMARK_NAMES[i] if i < len(MP_LANDMARK_NAMES) else f"point_{i}",
        })

    return {"frame_index": frame_index, "detected": True, "keypoints": keypoints}


@app.post("/api/annotations/projects/{project_id}/save")
async def save_frame_keypoints(project_id: str, frame_index: int = Form(...), keypoints: str = Form(...)):
    """Save corrected keypoints for a frame."""
    project = await db.get_annotation_project(project_id)
    if not project:
        return JSONResponse(status_code=404, content={"error": "Project not found"})

    kp_list = json.loads(keypoints)
    frame_id = str(uuid.uuid4())[:8]
    result = await db.save_annotation_frame(frame_id, project_id, frame_index, kp_list)
    return result


@app.get("/api/annotations/projects/{project_id}/frames")
async def list_annotated_frames(project_id: str):
    """List all annotated frames for a project."""
    frames = await db.get_annotation_frames(project_id)
    return {"frames": frames}


@app.get("/api/annotations/projects/{project_id}/export")
async def export_coco_keypoints(project_id: str):
    """Export annotations in COCO keypoint format (33 MediaPipe landmarks).

    MediaPipe pose models cannot be fine-tuned, so this is for evaluating
    detections or training a different pose model on the corrected keypoints.
    """
    project = await db.get_annotation_project(project_id)
    if not project:
        return JSONResponse(status_code=404, content={"error": "Project not found"})

    frames = await db.get_annotation_frames(project_id)

    images = []
    annotations = []

    for i, frame_data in enumerate(frames):
        image_id = i + 1
        ann_id = i + 1
        frame_idx = frame_data["frame_index"]

        images.append({
            "id": image_id,
            "file_name": f"frame_{frame_idx:06d}.jpg",
            "width": project["frame_width"],
            "height": project["frame_height"],
        })

        kps = frame_data["keypoints"]
        # COCO format: [x1, y1, v1, x2, y2, v2, ...]
        # v: 0=not labeled, 1=labeled not visible, 2=labeled and visible
        coco_kp = []
        num_visible = 0
        x_coords = []
        y_coords = []
        for kp in kps:
            px = kp["x"] * project["frame_width"]
            py = kp["y"] * project["frame_height"]
            vis = kp.get("visibility", 0)
            if vis > 0.3:
                v_flag = 2
                num_visible += 1
                x_coords.append(px)
                y_coords.append(py)
            else:
                v_flag = 1
            coco_kp.extend([round(px, 1), round(py, 1), v_flag])

        # Compute bounding box from visible keypoints
        if x_coords:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            margin = 20
            bbox = [
                max(0, x_min - margin),
                max(0, y_min - margin),
                min(project["frame_width"], x_max - x_min + 2 * margin),
                min(project["frame_height"], y_max - y_min + 2 * margin),
            ]
        else:
            bbox = [0, 0, project["frame_width"], project["frame_height"]]

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,
            "keypoints": coco_kp,
            "num_keypoints": num_visible,
            "bbox": [round(b, 1) for b in bbox],
            "area": round(bbox[2] * bbox[3], 1),
            "iscrowd": 0,
        })

    skeleton_1indexed = [[a + 1, b + 1] for a, b in MP_SKELETON]

    coco = {
        "info": {
            "description": f"Pose keypoint annotations — {project['name']}",
            "version": "1.0",
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": MP_LANDMARK_NAMES,
            "skeleton": skeleton_1indexed,
        }],
    }

    return coco


@app.post("/api/annotations/projects/{project_id}/export-frames")
async def export_annotated_frames(project_id: str):
    """Extract annotated frames as JPEGs into a zip-ready directory. Returns the path."""
    project = await db.get_annotation_project(project_id)
    if not project:
        return JSONResponse(status_code=404, content={"error": "Project not found"})

    frames = await db.get_annotation_frames(project_id)
    export_dir = os.path.join(ANNOTATION_DIR, f"export_{project_id}")
    os.makedirs(export_dir, exist_ok=True)

    cap = cv2.VideoCapture(project["video_path"])
    exported = []
    for frame_data in frames:
        idx = frame_data["frame_index"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            fname = f"frame_{idx:06d}.jpg"
            cv2.imwrite(os.path.join(export_dir, fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            exported.append(fname)
    cap.release()

    return {"export_dir": export_dir, "frames_exported": len(exported), "files": exported}


# ---------- WebSocket for live stream ----------

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)



manager = ConnectionManager()


@app.websocket("/ws/live")
async def live_analysis(ws: WebSocket):
    """
    WebSocket endpoint for live video analysis.
    Client sends base64-encoded frames, server responds with classifications.
    """
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                frame_data = msg["data"]
                result = classifier.predict_base64(frame_data)

                # Generate event if activity changed
                event = await kpi_engine.process_prediction(result)

                response = {
                    "type": "prediction",
                    "timestamp": time.time(),
                    "prediction": result,
                }
                if event:
                    response["event"] = event

                await ws.send_json(response)

            elif msg.get("type") == "ping":
                await ws.send_json({"type": "pong", "timestamp": time.time()})

    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.websocket("/ws/dashboard")
async def dashboard_feed(ws: WebSocket):
    """
    WebSocket for dashboard — pushes KPI updates and events in real-time.
    """
    await manager.connect(ws)
    try:
        while True:
            metrics = await kpi_engine.compute_metrics()
            await ws.send_json({"type": "metrics", "data": metrics, "timestamp": time.time()})
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------- Demo videos ----------
demo_path = os.path.join(os.path.dirname(__file__), "..", "data", "demo")
# Create it up front: the /demo mount below only exists if the folder does at startup
os.makedirs(demo_path, exist_ok=True)


@app.get("/api/demo/videos")
async def list_demo_videos():
    """List available demo video files.

    Videos must be H.264 encoded for reliable browser seeking (Pose Preview slider).
    VP9/AV1 codecs have sparse keyframes causing the video to freeze on seek.
    Re-encode: ffmpeg -i input.mp4 -c:v libx264 -movflags +faststart output.mp4
    """
    if not os.path.exists(demo_path):
        return {"videos": []}
    files = [f for f in os.listdir(demo_path) if f.endswith((".mp4", ".avi", ".mov", ".webm"))]
    return {"videos": [{"name": f, "url": f"/demo/{f}"} for f in sorted(files)]}


# Serve demo videos (StaticFiles handles Range/HEAD correctly for browser video seeking)
app.mount("/demo", StaticFiles(directory=demo_path), name="demo")

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
