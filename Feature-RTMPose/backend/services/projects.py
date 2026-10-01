"""Training Projects — isolated workspaces for training activity classifiers.

Each project has its own dataset, model versions, and persisted labels per video.
The pose model is locked after first training to ensure feature consistency.

Storage: data/projects/{project_id}/
"""

import json
import os
import shutil
import time
import uuid


PROJECTS_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "projects")


def _ensure_base():
    os.makedirs(PROJECTS_BASE, exist_ok=True)


def _project_dir(project_id: str) -> str:
    return os.path.join(PROJECTS_BASE, project_id)


def _meta_path(project_id: str) -> str:
    return os.path.join(_project_dir(project_id), "project.json")


def list_projects() -> list[dict]:
    """List all training projects with basic info."""
    _ensure_base()
    projects = []
    for entry in sorted(os.listdir(PROJECTS_BASE)):
        pdir = _project_dir(entry)
        if not os.path.isdir(pdir):
            continue
        meta = _read_meta(entry)
        if not meta:
            continue

        # Count labeled videos
        labels_dir = os.path.join(pdir, "labels")
        video_count = 0
        if os.path.isdir(labels_dir):
            video_count = len([f for f in os.listdir(labels_dir) if f.endswith(".json")])

        # Check if trained
        has_model = os.path.exists(os.path.join(pdir, "activity_model.pkl"))

        # Dataset size
        dataset_path = os.path.join(pdir, "dataset.json")
        dataset_samples = 0
        if os.path.exists(dataset_path):
            with open(dataset_path) as f:
                d = json.load(f)
            dataset_samples = len(d.get("features", []))

        projects.append({
            "id": entry,
            "name": meta.get("name", entry),
            "pose_mode": meta.get("pose_mode"),
            "collection_id": meta.get("collection_id"),
            "created_at": meta.get("created_at", ""),
            "video_count": video_count,
            "dataset_samples": dataset_samples,
            "has_model": has_model,
        })
    return projects


def create_project(name: str, pose_mode: str | None = None) -> dict:
    """Create a new training project."""
    _ensure_base()
    project_id = str(uuid.uuid4())[:8]
    pdir = _project_dir(project_id)
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(os.path.join(pdir, "versions"), exist_ok=True)
    os.makedirs(os.path.join(pdir, "labels"), exist_ok=True)

    meta = {
        "name": name,
        "pose_mode": pose_mode,  # None until first training, then locked
        "collection_id": None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(_meta_path(project_id), "w") as f:
        json.dump(meta, f, indent=2)

    return {"id": project_id, **meta}


def get_project(project_id: str) -> dict | None:
    """Get project metadata."""
    meta = _read_meta(project_id)
    if not meta:
        return None
    meta["id"] = project_id
    return meta


def update_project(project_id: str, **kwargs) -> dict | None:
    """Update project metadata (link collection, rename, lock pose_mode)."""
    meta = _read_meta(project_id)
    if not meta:
        return None
    for key in ["name", "pose_mode", "collection_id"]:
        if key in kwargs and kwargs[key] is not None:
            meta[key] = kwargs[key]
    with open(_meta_path(project_id), "w") as f:
        json.dump(meta, f, indent=2)
    meta["id"] = project_id
    return meta


def delete_project(project_id: str) -> dict:
    """Delete a project and all its data."""
    pdir = _project_dir(project_id)
    if not os.path.isdir(pdir):
        return {"status": "not_found"}
    shutil.rmtree(pdir, ignore_errors=True)
    return {"status": "deleted", "id": project_id}


# ===== Per-video label persistence =====

def save_labels(project_id: str, video_name: str, labels: list[dict]) -> dict:
    """Save activity labels for a specific video within a project."""
    labels_dir = os.path.join(_project_dir(project_id), "labels")
    os.makedirs(labels_dir, exist_ok=True)

    # Sanitize video name for filename
    safe_name = video_name.replace("/", "_").replace("\\", "_")
    filepath = os.path.join(labels_dir, f"{safe_name}.json")

    data = {
        "video_name": video_name,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label_count": len(labels),
        "labels": labels,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return {"status": "saved", "video_name": video_name, "label_count": len(labels)}


def load_labels(project_id: str, video_name: str) -> dict | None:
    """Load saved labels for a specific video."""
    safe_name = video_name.replace("/", "_").replace("\\", "_")
    filepath = os.path.join(_project_dir(project_id), "labels", f"{safe_name}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)


def list_labeled_videos(project_id: str) -> list[dict]:
    """List all videos that have saved labels in this project."""
    labels_dir = os.path.join(_project_dir(project_id), "labels")
    if not os.path.isdir(labels_dir):
        return []

    videos = []
    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith(".json"):
            continue
        filepath = os.path.join(labels_dir, fname)
        with open(filepath) as f:
            data = json.load(f)
        videos.append({
            "video_name": data.get("video_name", fname.replace(".json", "")),
            "label_count": data.get("label_count", 0),
            "saved_at": data.get("saved_at", ""),
        })
    return videos


# ===== Helpers =====

def _read_meta(project_id: str) -> dict | None:
    path = _meta_path(project_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
