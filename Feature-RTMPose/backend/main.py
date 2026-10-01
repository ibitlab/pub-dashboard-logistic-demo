"""Warehouse Activity Intelligence — FastAPI Backend."""

import asyncio
import json
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from services.database import Database
from services.classifier import ActivityClassifier
from services.reid import PersonReID
from services.kpi_engine import KPIEngine


db = Database()
classifier = ActivityClassifier()
kpi_engine = KPIEngine(db)


def _startup_migration():
    """Ensure required directories exist and migrate legacy data if needed."""
    data_base = os.path.join(os.path.dirname(__file__), "..", "data")

    # Ensure base directories
    for d in ["projects", "workers", "models", "demo"]:
        os.makedirs(os.path.join(data_base, d), exist_ok=True)

    # Migrate global gallery to "Default" collection if no collections exist yet
    from services.workers import WORKERS_BASE, list_collections, create_collection
    from services.reid import DEFAULT_GALLERY_PATH, DEFAULT_THUMBS_DIR

    if not list_collections() and os.path.exists(DEFAULT_GALLERY_PATH):
        try:
            c = create_collection("Default (migrated)")
            cdir = os.path.join(WORKERS_BASE, c["id"])
            # Copy gallery
            shutil.copy2(DEFAULT_GALLERY_PATH, os.path.join(cdir, "reid_gallery.json"))
            # Copy thumbnails
            if os.path.isdir(DEFAULT_THUMBS_DIR):
                dst_thumbs = os.path.join(cdir, "reid_thumbs")
                if os.path.isdir(dst_thumbs):
                    shutil.rmtree(dst_thumbs)
                shutil.copytree(DEFAULT_THUMBS_DIR, dst_thumbs)
            print(f"[Migration] Global gallery migrated to collection '{c['name']}' ({c['id']})")
        except Exception as e:
            print(f"[Migration] Warning: could not migrate global gallery: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_migration()
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


# ---------- Training Projects ----------

from services import projects as projects_svc


@app.get("/api/projects")
async def list_projects():
    """List all training projects."""
    return {"projects": projects_svc.list_projects()}


@app.post("/api/projects")
async def create_project(name: str = Form(...), pose_mode: str = Form(None)):
    """Create a new training project."""
    return projects_svc.create_project(name, pose_mode or None)


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get project info."""
    p = projects_svc.get_project(project_id)
    if not p:
        return JSONResponse(status_code=404, content={"error": "Project not found"})
    return p


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project and all its data."""
    if classifier.active_project_id == project_id:
        # Deactivate first
        classifier.set_project_paths(None)
        classifier._load_model_state()
        classifier.active_project_id = None
    return projects_svc.delete_project(project_id)


@app.patch("/api/projects/{project_id}")
async def update_project(project_id: str, request: Request):
    """Update project metadata (link collection, rename)."""
    body = await request.json()
    result = projects_svc.update_project(project_id, **body)
    if not result:
        return JSONResponse(status_code=404, content={"error": "Project not found"})
    return result


@app.post("/api/projects/{project_id}/activate")
async def activate_project(project_id: str):
    """Switch the active training project. Reloads model + dataset for this project."""
    project = projects_svc.get_project(project_id)
    if not project:
        return JSONResponse(status_code=404, content={"error": "Project not found"})

    project_dir = os.path.join(projects_svc.PROJECTS_BASE, project_id)
    classifier.set_project_paths(project_dir)
    classifier._load_model_state()
    classifier.active_project_id = project_id

    # If project has a pose_mode, verify/switch
    if project.get("pose_mode") and project["pose_mode"] != classifier.pose_mode:
        try:
            classifier.switch_pose_model(project["pose_mode"])
        except Exception:
            pass  # model may need downloading, don't block

    # If project links to a workers collection, load that collection's ReID
    collection_id = project.get("collection_id")
    if collection_id:
        # Same cached instance the enroll endpoints write to, so a worker
        # enrolled after activation is recognized without re-activating.
        collection_reid = _get_collection_reid(collection_id)
        if collection_reid:
            classifier.reid = collection_reid
    else:
        # No linked collection: fall back to the global gallery rather than
        # keeping whichever project's workers were active before
        classifier.reid = PersonReID()
        classifier.reid.load()

    return {
        "status": "activated",
        "project_id": project_id,
        "name": project["name"],
        "is_trained": classifier.is_trained,
        "classes": classifier.get_classes(),
        "pose_mode": classifier.pose_mode,
        "collection_id": collection_id,
    }


@app.post("/api/projects/deactivate")
async def deactivate_project():
    """Deactivate current project, return to global mode."""
    classifier.set_project_paths(None)
    classifier._load_model_state()
    classifier.active_project_id = None

    # Restore global ReID
    classifier.reid = PersonReID()
    classifier.reid.load()

    return {"status": "deactivated"}


@app.get("/api/projects/{project_id}/labels")
async def list_project_labels(project_id: str):
    """List all videos with saved labels in this project."""
    return {"videos": projects_svc.list_labeled_videos(project_id)}


@app.get("/api/projects/{project_id}/labels/{video_name:path}")
async def get_project_labels(project_id: str, video_name: str):
    """Get saved labels for a specific video in this project."""
    data = projects_svc.load_labels(project_id, video_name)
    if not data:
        return {"video_name": video_name, "labels": [], "label_count": 0}
    return data


@app.post("/api/projects/{project_id}/labels/{video_name:path}")
async def save_project_labels(project_id: str, video_name: str, request: Request):
    """Save activity labels for a specific video in this project."""
    body = await request.json()
    labels = body.get("labels", [])
    return projects_svc.save_labels(project_id, video_name, labels)


# ---------- Label Stability Analysis ----------

@app.post("/api/train/analyze-label-stability")
async def analyze_label_stability(request: Request):
    """Analyze person detection stability within a frame range.

    Samples frames, detects persons, identifies break points where
    person count or positions change significantly.
    Returns proposed splits for stable sub-ranges.
    """
    body = await request.json()
    video_url = body.get("video_url", "")
    start_time = body.get("start_time", 0)
    end_time = body.get("end_time", 0)

    # Resolve video path
    filename = video_url.split("/")[-1].split("?")[0]
    video_path = os.path.join(demo_path, filename)
    if not os.path.exists(video_path):
        return JSONResponse(status_code=400, content={"error": f"Video not found: {filename}"})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return JSONResponse(status_code=400, content={"error": "Cannot open video"})

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Sample every 5th frame
    sample_interval = 5
    frame_data = []  # [{frame, person_count, bboxes}]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % sample_interval == 0:
            persons = classifier._detect_all_poses(frame)
            person_count = len(persons)

            # Compute bounding boxes for IoU tracking
            bboxes = []
            for kps, scs in persons:
                visible = scs > 0.3
                if np.any(visible):
                    pts = kps[visible]
                    bboxes.append((float(pts[:, 0].min()), float(pts[:, 1].min()),
                                   float(pts[:, 0].max()), float(pts[:, 1].max())))

            frame_data.append({
                "frame": frame_idx,
                "time": frame_idx / fps,
                "person_count": person_count,
                "bboxes": bboxes,
            })

        frame_idx += 1

    cap.release()

    if len(frame_data) < 2:
        return {"stable": True, "splits": [], "samples": len(frame_data)}

    # Find break points: person count changes or significant position jumps
    IOU_THRESHOLD = 0.3
    break_points = []

    for i in range(1, len(frame_data)):
        prev = frame_data[i - 1]
        curr = frame_data[i]

        is_break = False
        reason = ""

        # Person count changed
        if curr["person_count"] != prev["person_count"]:
            is_break = True
            reason = f"person count {prev['person_count']}→{curr['person_count']}"

        # Position jump (low IoU between consecutive frames)
        elif prev["bboxes"] and curr["bboxes"]:
            max_iou = 0
            for ba in prev["bboxes"]:
                for bb in curr["bboxes"]:
                    x1 = max(ba[0], bb[0]); y1 = max(ba[1], bb[1])
                    x2 = min(ba[2], bb[2]); y2 = min(ba[3], bb[3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    area_a = (ba[2] - ba[0]) * (ba[3] - ba[1])
                    area_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
                    union = area_a + area_b - inter
                    iou = inter / union if union > 0 else 0
                    max_iou = max(max_iou, iou)

            if max_iou < IOU_THRESHOLD:
                is_break = True
                reason = f"position jump (IoU={max_iou:.2f})"

        if is_break:
            break_points.append({
                "frame": curr["frame"],
                "time": round(curr["time"], 2),
                "reason": reason,
            })

    # Build splits from break points
    if not break_points:
        return {"stable": True, "splits": [], "break_points": [], "samples": len(frame_data)}

    splits = []
    split_start = start_time
    for bp in break_points:
        if bp["time"] > split_start + 0.1:  # min 0.1s split
            splits.append({
                "start_time": round(split_start, 2),
                "end_time": round(bp["time"], 2),
                "person_count": frame_data[0]["person_count"],  # approximate
            })
        split_start = bp["time"]

    # Final split
    if end_time > split_start + 0.1:
        splits.append({
            "start_time": round(split_start, 2),
            "end_time": round(end_time, 2),
            "person_count": frame_data[-1]["person_count"],
        })

    return {
        "stable": False,
        "splits": splits,
        "break_points": break_points,
        "samples": len(frame_data),
    }


# ---------- Training from Labels ----------

@app.post("/api/train/from-labels")
async def train_from_labels(request: Request):
    """Train activity classifier from labeled video intervals.

    Accepts JSON with video_url and labels array. Each label has:
    start_time, end_time, activity, person (all or #N).
    Extracts pose features from labeled intervals and trains the classifier.
    """
    body = await request.json()
    video_url = body.get("video_url", "")
    labels = body.get("labels", [])

    if not labels:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No labels provided"})

    # Resolve video path from URL
    # video_url is like "http://localhost:8001/demo/warehouse_walking.mp4" or "/demo/..."
    video_path = None
    filename = video_url.split("/")[-1].split("?")[0]
    # Check demo path
    candidate = os.path.join(demo_path, filename)
    if os.path.exists(candidate):
        video_path = candidate
    # Check annotations path
    if not video_path:
        for f in os.listdir(os.path.join(os.path.dirname(__file__), "..", "data", "annotations")):
            if f.endswith(filename):
                video_path = os.path.join(os.path.dirname(__file__), "..", "data", "annotations", f)
                break

    if not video_path:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Video not found: {filename}"})

    append = body.get("append_to_existing", True)
    # Save raw labels for later review/re-use
    raw_labels_info = {
        "video_url": video_url,
        "video_file": filename,
        "labels": labels,
    }
    result = classifier.train_from_labels(video_path, labels, append_to_existing=append, raw_labels=raw_labels_info)

    # Lock pose_mode in project after first successful training
    if result.get("status") == "success" and classifier.active_project_id:
        project = projects_svc.get_project(classifier.active_project_id)
        if project and not project.get("pose_mode"):
            projects_svc.update_project(classifier.active_project_id, pose_mode=classifier.pose_mode)

    return result


@app.get("/api/train/dataset")
async def get_training_dataset_info():
    """Get info about the persisted training dataset (accumulated features)."""
    return classifier.get_dataset_info()


@app.delete("/api/train/dataset")
async def clear_training_dataset():
    """Clear the persisted training dataset. Next train starts fresh."""
    return classifier.clear_dataset()


@app.get("/api/train/progress")
async def training_progress():
    """Get live training progress (poll during training)."""
    return classifier.train_progress or {"status": "idle", "percent": 0}


@app.get("/api/train/status")
async def training_status():
    """Get training status and model info."""
    return classifier.get_status()


# ---------- Training Data & Model Versioning ----------

@app.get("/api/train/versions/{version_id}/labels")
async def get_version_labels(version_id: str):
    """Get the raw activity labels used to train a specific version."""
    entry = next((h for h in classifier.training_history if h["version_id"] == version_id), None)
    if not entry:
        return JSONResponse(status_code=404, content={"error": "Version not found"})
    raw = entry.get("raw_labels")
    if not raw:
        return {"version_id": version_id, "has_labels": False, "labels": []}
    return {"version_id": version_id, "has_labels": True, **raw}


@app.delete("/api/train/dataset/source/{source_name:path}")
async def remove_dataset_source(source_name: str):
    """Remove all samples from a specific source video in the dataset."""
    return classifier.remove_dataset_source(source_name)


@app.post("/api/train/retrain")
async def retrain_on_dataset():
    """Re-train the model on the full accumulated dataset."""
    result = classifier.retrain_on_dataset()
    if result.get("status") == "error":
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/api/train/switch-version")
async def switch_model_version(version_id: str = Form(...)):
    """Switch to a previously trained model version."""
    result = classifier.switch_model_version(version_id)
    if result["status"] == "error":
        return JSONResponse(status_code=404, content=result)
    return result


@app.delete("/api/train/versions/{version_id}")
async def delete_model_version(version_id: str):
    """Delete a saved model version. Cannot delete the active version."""
    result = classifier.delete_version(version_id)
    if result["status"] == "error":
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/api/train/ensemble")
async def toggle_ensemble(enabled: bool = Form(...)):
    """Toggle ensemble mode. When ON, predictions use ALL saved model versions."""
    return classifier.set_ensemble(enabled)


@app.post("/api/pose-model/switch")
async def switch_pose_model(mode: str = Form(...)):
    """Switch RTMPose model size at runtime. Models auto-download on first use.

    Modes: lightweight, balanced, performance
    """
    try:
        result = classifier.switch_pose_model(mode)
        return result
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/api/predict")
async def predict_frame(file: UploadFile = File(...)):
    """Classify a single frame / image."""
    contents = await file.read()
    result = classifier.predict_image(contents)
    return result


# ---------- Workers Collections ----------

from services import workers as workers_svc

# Cache for collection ReID instances (avoid reloading gallery on every request)
_collection_reid_cache: dict[str, PersonReID] = {}


def _get_collection_reid(collection_id: str):
    """Get or create a cached PersonReID for a collection."""
    if collection_id not in _collection_reid_cache:
        reid = workers_svc.get_reid_for_collection(collection_id)
        if reid is None:
            return None
        _collection_reid_cache[collection_id] = reid
    return _collection_reid_cache[collection_id]


@app.get("/api/workers/collections")
async def list_worker_collections():
    """List all workers collections."""
    return {"collections": workers_svc.list_collections()}


@app.post("/api/workers/collections")
async def create_worker_collection(name: str = Form(...)):
    """Create a new workers collection."""
    return workers_svc.create_collection(name)


@app.get("/api/workers/collections/{collection_id}")
async def get_worker_collection(collection_id: str):
    """Get collection info."""
    c = workers_svc.get_collection(collection_id)
    if not c:
        return JSONResponse(status_code=404, content={"error": "Collection not found"})
    return c


@app.delete("/api/workers/collections/{collection_id}")
async def delete_worker_collection(collection_id: str):
    """Delete a workers collection and all its data."""
    _collection_reid_cache.pop(collection_id, None)
    return workers_svc.delete_collection(collection_id)


@app.get("/api/workers/collections/{collection_id}/gallery")
async def collection_gallery(collection_id: str):
    """List persons in a collection."""
    reid = _get_collection_reid(collection_id)
    if not reid:
        return JSONResponse(status_code=404, content={"error": "Collection not found"})
    return {"gallery": reid.get_gallery(), "enabled": reid.enabled}


@app.post("/api/workers/collections/{collection_id}/enroll")
async def collection_enroll(collection_id: str, file: UploadFile = File(...), name: str = Form(...)):
    """Enroll a person into a collection from a cropped person image."""
    reid = _get_collection_reid(collection_id)
    if not reid:
        return JSONResponse(status_code=404, content={"error": "Collection not found"})
    if not reid.enabled:
        return JSONResponse(status_code=400, content={"error": "ReID model not loaded (see README: export OSNet-AIN)"})

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    embedding = reid.extract_embedding(image)
    return reid.enroll(name, embedding, bgr_crop=image)


@app.post("/api/workers/collections/{collection_id}/enroll-from-frame")
async def collection_enroll_from_frame(collection_id: str, file: UploadFile = File(...), name: str = Form(...), person_index: int = Form(0)):
    """Enroll a person from a full video frame into a collection."""
    reid = _get_collection_reid(collection_id)
    if not reid:
        return JSONResponse(status_code=404, content={"error": "Collection not found"})
    if not reid.enabled:
        return JSONResponse(status_code=400, content={"error": "ReID model not loaded (see README: export OSNet-AIN)"})

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    persons = classifier._detect_all_poses(image)
    if not persons:
        return JSONResponse(status_code=400, content={"error": "No person detected"})
    if person_index >= len(persons):
        return JSONResponse(status_code=400, content={"error": f"Only {len(persons)} persons detected"})

    persons.sort(key=lambda p: float(np.mean(p[0][:, 0])))
    kps, scs = persons[person_index]
    crop = reid.crop_person(image, kps, scs)
    if crop is None:
        return JSONResponse(status_code=400, content={"error": "Could not crop person"})

    embedding = reid.extract_embedding(crop)
    return reid.enroll(name, embedding, bgr_crop=crop)


@app.delete("/api/workers/collections/{collection_id}/gallery/{name}")
async def collection_remove_worker(collection_id: str, name: str):
    """Remove a person from a collection."""
    reid = _get_collection_reid(collection_id)
    if not reid:
        return JSONResponse(status_code=404, content={"error": "Collection not found"})

    # Also remove thumbnails
    thumb_dir = reid.get_thumbs_dir_for(name)
    if os.path.isdir(thumb_dir):
        shutil.rmtree(thumb_dir, ignore_errors=True)

    return reid.remove(name)


@app.get("/api/workers/collections/{collection_id}/gallery/{name}/thumbnails")
async def collection_thumbnails(collection_id: str, name: str):
    """List thumbnails for a person in a collection."""
    reid = _get_collection_reid(collection_id)
    if not reid:
        return JSONResponse(status_code=404, content={"error": "Collection not found"})
    thumbs = reid.get_thumbnails(name)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return {"name": name, "thumbnails": [f"/api/workers/collections/{collection_id}/thumb/{safe_name}/{t}" for t in thumbs]}


@app.get("/api/workers/collections/{collection_id}/thumb/{safe_name}/{filename}")
async def collection_thumb_image(collection_id: str, safe_name: str, filename: str):
    """Serve a thumbnail image from a collection."""
    reid = _get_collection_reid(collection_id)
    if not reid:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    filepath = os.path.join(reid.thumbs_dir, safe_name, filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return FileResponse(filepath, media_type="image/jpeg")


# ---------- ReID Gallery Management (global/legacy) ----------

@app.get("/api/reid/status")
async def reid_status():
    """Get ReID model status and gallery info."""
    if classifier.reid:
        return classifier.reid.get_status()
    return {"enabled": False, "model": "not loaded", "gallery_size": 0}


@app.get("/api/reid/gallery")
async def reid_gallery():
    """List all enrolled persons in the gallery."""
    if not classifier.reid or not classifier.reid.enabled:
        return {"gallery": [], "enabled": False}
    return {"gallery": classifier.reid.get_gallery(), "enabled": True}


@app.post("/api/reid/enroll")
async def reid_enroll(file: UploadFile = File(...), name: str = Form(...)):
    """Enroll a person from a cropped person image. Saves embedding + thumbnail."""
    if not classifier.reid or not classifier.reid.enabled:
        return JSONResponse(status_code=400, content={"error": "ReID not enabled"})

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    embedding = classifier.reid.extract_embedding(image)
    result = classifier.reid.enroll(name, embedding, bgr_crop=image)
    return result


@app.post("/api/reid/enroll-from-frame")
async def reid_enroll_from_frame(file: UploadFile = File(...), name: str = Form(...), person_index: int = Form(0)):
    """Enroll a person from a full video frame. Detects, crops, saves embedding + thumbnail."""
    if not classifier.reid or not classifier.reid.enabled:
        return JSONResponse(status_code=400, content={"error": "ReID not enabled"})

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    persons = classifier._detect_all_poses(image)
    if not persons:
        return JSONResponse(status_code=400, content={"error": "No person detected"})
    if person_index >= len(persons):
        return JSONResponse(status_code=400, content={"error": f"Only {len(persons)} persons detected, index {person_index} out of range"})

    persons.sort(key=lambda p: float(np.mean(p[0][:, 0])))
    kps, scs = persons[person_index]
    crop = classifier.reid.crop_person(image, kps, scs)
    if crop is None:
        return JSONResponse(status_code=400, content={"error": "Could not crop person — too small or occluded"})

    embedding = classifier.reid.extract_embedding(crop)
    result = classifier.reid.enroll(name, embedding, bgr_crop=crop)
    return result


@app.get("/api/reid/gallery/{name}/thumbnails")
async def reid_thumbnails(name: str):
    """List thumbnail images for a person."""
    if not classifier.reid or not classifier.reid.enabled:
        return JSONResponse(status_code=400, content={"error": "ReID not enabled"})
    thumbs = classifier.reid.get_thumbnails(name)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return {"name": name, "thumbnails": [f"/api/reid/thumb/{safe_name}/{t}" for t in thumbs]}


@app.get("/api/reid/thumb/{safe_name}/{filename}")
async def reid_thumb_image(safe_name: str, filename: str):
    """Serve a thumbnail image."""
    if not classifier.reid or not classifier.reid.enabled:
        return JSONResponse(status_code=400, content={"error": "ReID not enabled"})
    filepath = os.path.join(classifier.reid.thumbs_dir, safe_name, filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return FileResponse(filepath, media_type="image/jpeg")


@app.delete("/api/reid/gallery/{name}")
async def reid_remove(name: str):
    """Remove a person and their thumbnails from the gallery."""
    if not classifier.reid or not classifier.reid.enabled:
        return JSONResponse(status_code=400, content={"error": "ReID not enabled"})
    # Also remove thumbnails
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    thumb_dir = os.path.join(classifier.reid.thumbs_dir, safe_name)
    if os.path.isdir(thumb_dir):
        shutil.rmtree(thumb_dir, ignore_errors=True)
    return classifier.reid.remove(name)


@app.post("/api/predict/landmarks")
async def predict_with_landmarks(file: UploadFile = File(...), video_time: float = Form(None)):
    """Classify a single frame and return full pose landmarks for visualization.

    video_time (s) is the frame's position in its video; motion features use it
    to measure speed and reset tracking on a seek.
    """
    contents = await file.read()
    result = classifier.predict_image(contents, include_landmarks=True, video_time=video_time)
    return result


# ---------- Annotation endpoints ----------

ANNOTATION_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "annotations")
os.makedirs(ANNOTATION_DIR, exist_ok=True)

# COCO 17-point landmark names (RTMPose output)
from services.classifier import COCO_KEYPOINT_NAMES, COCO_SKELETON

LANDMARK_NAMES = COCO_KEYPOINT_NAMES
SKELETON = COCO_SKELETON


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
    import base64
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
    """Run RTMPose on a single frame and return detected landmarks.

    NOTE: This endpoint uses OpenCV's cap.set(CAP_PROP_POS_FRAMES) to seek to
    the requested frame, which is unreliable for many codecs — it may land on a
    different frame than the one the user sees in the browser. The frontend now
    uses /api/predict/landmarks instead (sending the displayed image directly),
    but this endpoint is kept for API completeness and non-browser clients.
    """
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
            "name": LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"point_{i}",
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
    """Export annotations in COCO keypoint format for MMPose/RTMPose fine-tuning."""
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

    skeleton_1indexed = [[a + 1, b + 1] for a, b in SKELETON]

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
            "keypoints": LANDMARK_NAMES,
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
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".webm")


@app.get("/api/demo/videos")
async def list_demo_videos():
    """List available demo video files.

    Videos must be H.264 encoded for reliable browser seeking (Pose Preview slider).
    VP9/AV1 codecs have sparse keyframes causing the video to freeze on seek.
    Re-encode: ffmpeg -i input.mp4 -c:v libx264 -movflags +faststart output.mp4
    """
    if not os.path.exists(demo_path):
        return {"videos": []}
    files = [f for f in os.listdir(demo_path) if f.endswith(VIDEO_EXTENSIONS)]
    return {"videos": [{"name": f, "url": f"/demo/{f}"} for f in sorted(files)]}



@app.post("/api/demo/upload")
async def upload_demo_video(file: UploadFile = File(...)):
    """Store an uploaded clip in data/demo/ next to the demo videos.

    Training from labels reads the video on the server, so a clip picked in the
    Training tab's Upload source has to land here first. Never overwrites: a
    name clash gets a numeric suffix.
    """
    stem, ext = os.path.splitext(os.path.basename(file.filename or ""))
    ext = ext.lower()
    if ext not in VIDEO_EXTENSIONS:
        return JSONResponse(status_code=400, content={"error": f"Unsupported video type: {ext or 'none'}"})

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_") or "upload"
    os.makedirs(demo_path, exist_ok=True)
    name, n = f"{safe}{ext}", 1
    while os.path.exists(os.path.join(demo_path, name)):
        name, n = f"{safe}_{n}{ext}", n + 1

    with open(os.path.join(demo_path, name), "wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"name": name, "url": f"/demo/{name}"}


# Serve demo videos with HTTP Range support (required for browser video seeking)
@app.get("/demo/{filename}")
async def serve_demo_video(filename: str, request: Request):
    filepath = os.path.join(demo_path, filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "Not found"})

    file_size = os.path.getsize(filepath)
    range_header = request.headers.get("range")

    if range_header:
        # Parse Range: bytes=start-end
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        def iter_file():
            with open(filepath, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iter_file(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
        )

    return FileResponse(filepath, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
