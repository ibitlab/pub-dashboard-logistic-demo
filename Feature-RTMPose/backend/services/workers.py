"""Workers Collections — manages separate ReID galleries per location/shift.

Each collection has its own gallery (embeddings + thumbnails), stored in
data/workers/{collection_id}/. The ONNX model is shared globally.
"""

import json
import os
import shutil
import time
import uuid

from services.reid import PersonReID


WORKERS_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "workers")


def _ensure_base():
    os.makedirs(WORKERS_BASE, exist_ok=True)


def list_collections() -> list[dict]:
    """List all workers collections with basic info."""
    _ensure_base()
    collections = []
    for entry in sorted(os.listdir(WORKERS_BASE)):
        cdir = os.path.join(WORKERS_BASE, entry)
        if not os.path.isdir(cdir):
            continue
        meta_path = os.path.join(cdir, "collection.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {"name": entry, "created_at": ""}

        # Count gallery size
        gallery_path = os.path.join(cdir, "reid_gallery.json")
        gallery_size = 0
        if os.path.exists(gallery_path):
            with open(gallery_path) as f:
                gallery_size = len(json.load(f))

        collections.append({
            "id": entry,
            "name": meta.get("name", entry),
            "created_at": meta.get("created_at", ""),
            "gallery_size": gallery_size,
        })
    return collections


def create_collection(name: str) -> dict:
    """Create a new workers collection."""
    _ensure_base()
    collection_id = str(uuid.uuid4())[:8]
    cdir = os.path.join(WORKERS_BASE, collection_id)
    os.makedirs(cdir, exist_ok=True)
    os.makedirs(os.path.join(cdir, "reid_thumbs"), exist_ok=True)

    meta = {
        "name": name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(cdir, "collection.json"), "w") as f:
        json.dump(meta, f)

    return {"id": collection_id, "name": name, **meta}


def get_collection(collection_id: str) -> dict | None:
    """Get collection metadata."""
    cdir = os.path.join(WORKERS_BASE, collection_id)
    meta_path = os.path.join(cdir, "collection.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    meta["id"] = collection_id
    return meta


def delete_collection(collection_id: str) -> dict:
    """Delete a workers collection and all its data."""
    cdir = os.path.join(WORKERS_BASE, collection_id)
    if not os.path.isdir(cdir):
        return {"status": "not_found"}
    shutil.rmtree(cdir, ignore_errors=True)
    return {"status": "deleted", "id": collection_id}


def get_reid_for_collection(collection_id: str) -> PersonReID | None:
    """Create a PersonReID instance pointed at a collection's gallery.

    Returns None if collection doesn't exist.
    The ONNX model is shared globally — only the gallery path differs.
    """
    cdir = os.path.join(WORKERS_BASE, collection_id)
    if not os.path.isdir(cdir):
        return None

    gallery_path = os.path.join(cdir, "reid_gallery.json")
    thumbs_dir = os.path.join(cdir, "reid_thumbs")

    reid = PersonReID(gallery_path=gallery_path, thumbs_dir=thumbs_dir)
    reid.load()
    return reid
