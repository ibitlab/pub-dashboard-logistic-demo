"""Person Re-Identification using OSNet-AIN via ONNX Runtime.

Extracts 512-d appearance embeddings from cropped person images and matches
them against a gallery of known workers using cosine similarity.

Gallery stores MULTIPLE embeddings per person (different angles, cameras, outfits)
and matches against the best one. This is more robust than averaging.

License: MIT (OSNet/torchreid) — free for commercial use.
Dependencies: onnxruntime, opencv-python-headless, numpy (NO PyTorch needed).
"""

import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort


# Default similarity threshold for "same person" decision.
DEFAULT_SIMILARITY_THRESHOLD = 0.65

# Max embeddings stored per person (different angles/cameras/outfits)
MAX_EMBEDDINGS_PER_PERSON = 10

# ImageNet normalization (same as torchreid training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

INPUT_HEIGHT = 256
INPUT_WIDTH = 128

# Global defaults — used when no custom paths are provided
DEFAULT_REID_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "osnet_ain_x1_0.onnx")
DEFAULT_GALLERY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "reid_gallery.json")
DEFAULT_THUMBS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "reid_thumbs")

# Shared ONNX session — loaded once, used by all PersonReID instances
_shared_session: ort.InferenceSession | None = None
_shared_input_name: str = ""


def _get_shared_session() -> tuple[ort.InferenceSession | None, str]:
    """Get or create the shared ONNX session (one model, many galleries)."""
    global _shared_session, _shared_input_name
    if _shared_session is None:
        if os.path.exists(DEFAULT_REID_MODEL_PATH):
            _shared_session = ort.InferenceSession(DEFAULT_REID_MODEL_PATH, providers=["CPUExecutionProvider"])
            _shared_input_name = _shared_session.get_inputs()[0].name
            print(f"[ReID] Loaded OSNet-AIN from {DEFAULT_REID_MODEL_PATH}")
        else:
            print(f"[ReID] Model not found at {DEFAULT_REID_MODEL_PATH}")
    return _shared_session, _shared_input_name


class PersonReID:
    """Lightweight person re-identification using OSNet-AIN ONNX model.

    Accepts optional gallery_path and thumbs_dir to support multiple galleries
    (Workers Collections). The ONNX model is shared globally.
    """

    def __init__(self, gallery_path: str | None = None, thumbs_dir: str | None = None):
        self.gallery_path = gallery_path or DEFAULT_GALLERY_PATH
        self.thumbs_dir = thumbs_dir or DEFAULT_THUMBS_DIR
        self.session: ort.InferenceSession | None = None
        self.input_name: str = ""
        self.gallery: dict[str, dict] = {}
        self.enabled = False

    def load(self):
        """Load the shared ONNX model and this instance's gallery."""
        self.session, self.input_name = _get_shared_session()
        if self.session is None:
            self.enabled = False
            return

        self.enabled = True
        os.makedirs(self.thumbs_dir, exist_ok=True)
        self._load_gallery()

    def _load_gallery(self):
        """Load saved gallery from disk."""
        if not os.path.exists(self.gallery_path):
            return

        with open(self.gallery_path) as f:
            data = json.load(f)

        self.gallery = {}
        for name, entry in data.items():
            if "embeddings" in entry:
                self.gallery[name] = {
                    "embeddings": [np.array(e, dtype=np.float32) for e in entry["embeddings"]],
                    "labels": entry.get("labels", []),
                    "enrolled_at": entry.get("enrolled_at", ""),
                    "updated_at": entry.get("updated_at", ""),
                }
            elif "embedding" in entry:
                self.gallery[name] = {
                    "embeddings": [np.array(entry["embedding"], dtype=np.float32)],
                    "labels": ["migrated"],
                    "enrolled_at": entry.get("enrolled_at", ""),
                    "updated_at": entry.get("enrolled_at", ""),
                }

        print(f"[ReID] Gallery loaded: {len(self.gallery)} persons from {self.gallery_path}")

    def _save_gallery(self):
        """Persist gallery to disk."""
        os.makedirs(os.path.dirname(self.gallery_path), exist_ok=True)
        data = {}
        for name, entry in self.gallery.items():
            data[name] = {
                "embeddings": [e.tolist() for e in entry["embeddings"]],
                "labels": entry.get("labels", []),
                "enrolled_at": entry.get("enrolled_at", ""),
                "updated_at": entry.get("updated_at", ""),
            }
        with open(self.gallery_path, "w") as f:
            json.dump(data, f)

    def preprocess(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Preprocess a cropped person image for OSNet."""
        resized = cv2.resize(bgr_crop, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]
        blob = (blob - MEAN) / STD
        return blob

    def extract_embedding(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Extract a 512-d L2-normalized embedding from a cropped person image."""
        if not self.enabled:
            return np.zeros(512, dtype=np.float32)

        blob = self.preprocess(bgr_crop)
        outputs = self.session.run(None, {self.input_name: blob})
        embedding = outputs[0].flatten().astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def crop_person(self, image: np.ndarray, kps: np.ndarray, scs: np.ndarray,
                    padding: float = 0.15) -> np.ndarray | None:
        """Crop a person from the full image using their keypoints."""
        visible = scs > 0.3
        if np.sum(visible) < 4:
            return None

        pts = kps[visible]
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()

        w, h = x2 - x1, y2 - y1
        x1 -= w * padding
        y1 -= h * padding
        x2 += w * padding
        y2 += h * padding

        ih, iw = image.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(iw, int(x2)), min(ih, int(y2))

        if x2 - x1 < 20 or y2 - y1 < 40:
            return None

        return image[y1:y2, x1:x2]

    @staticmethod
    def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(emb_a, emb_b))

    def match(self, embedding: np.ndarray, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> dict:
        """Match an embedding against the gallery."""
        if not self.gallery or not self.enabled:
            return {"name": "unknown", "similarity": 0.0, "matched": False}

        best_name = "unknown"
        best_sim = -1.0

        for name, entry in self.gallery.items():
            for stored_emb in entry["embeddings"]:
                sim = self.cosine_similarity(embedding, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

        if best_sim >= threshold:
            return {"name": best_name, "similarity": round(best_sim, 3), "matched": True}

        return {"name": "unknown", "similarity": round(best_sim, 3), "matched": False}

    def _save_thumbnail(self, name: str, index: int, bgr_crop: np.ndarray):
        """Save a person crop as a JPEG thumbnail."""
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        person_dir = os.path.join(self.thumbs_dir, safe_name)
        os.makedirs(person_dir, exist_ok=True)

        h, w = bgr_crop.shape[:2]
        scale = min(128 / h, 96 / w)
        thumb = cv2.resize(bgr_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(person_dir, f"{index}.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def get_thumbnails(self, name: str) -> list[str]:
        """Return list of thumbnail filenames for a person."""
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        person_dir = os.path.join(self.thumbs_dir, safe_name)
        if not os.path.isdir(person_dir):
            return []
        return sorted(f for f in os.listdir(person_dir) if f.endswith(".jpg"))

    def get_thumbs_dir_for(self, name: str) -> str:
        """Return the thumbnail directory path for a person."""
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return os.path.join(self.thumbs_dir, safe_name)

    def enroll(self, name: str, embedding: np.ndarray, bgr_crop: np.ndarray | None = None, label: str = "") -> dict:
        """Add a new embedding for a person."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        source_label = label or now

        if name in self.gallery:
            entry = self.gallery[name]
            embeddings = entry["embeddings"]
            labels = entry["labels"]

            if len(embeddings) >= MAX_EMBEDDINGS_PER_PERSON:
                sims = [self.cosine_similarity(embedding, e) for e in embeddings]
                most_similar_idx = int(np.argmax(sims))
                embeddings[most_similar_idx] = embedding
                labels[most_similar_idx] = source_label
                thumb_idx = most_similar_idx
            else:
                embeddings.append(embedding)
                labels.append(source_label)
                thumb_idx = len(embeddings) - 1

            entry["updated_at"] = now
            status = "updated"
            count = len(embeddings)
        else:
            self.gallery[name] = {
                "embeddings": [embedding],
                "labels": [source_label],
                "enrolled_at": now,
                "updated_at": now,
            }
            status = "enrolled"
            count = 1
            thumb_idx = 0

        if bgr_crop is not None:
            self._save_thumbnail(name, thumb_idx, bgr_crop)

        self._save_gallery()
        return {
            "name": name,
            "status": status,
            "embeddings_count": count,
            "gallery_size": len(self.gallery),
        }

    def remove(self, name: str) -> dict:
        """Remove a person from the gallery."""
        if name in self.gallery:
            del self.gallery[name]
            self._save_gallery()
            return {"name": name, "status": "removed", "gallery_size": len(self.gallery)}
        return {"name": name, "status": "not_found", "gallery_size": len(self.gallery)}

    def get_gallery(self) -> list[dict]:
        """Return gallery info (without raw embeddings)."""
        return [
            {
                "name": name,
                "enrolled_at": entry.get("enrolled_at", ""),
                "updated_at": entry.get("updated_at", ""),
                "embeddings_count": len(entry["embeddings"]),
                "labels": entry.get("labels", []),
            }
            for name, entry in self.gallery.items()
        ]

    def get_status(self) -> dict:
        total_embeddings = sum(len(e["embeddings"]) for e in self.gallery.values())
        return {
            "enabled": self.enabled,
            "model": "OSNet-AIN-x1.0 (512-d, MIT license)" if self.enabled else "not loaded",
            "gallery_size": len(self.gallery),
            "total_embeddings": total_embeddings,
            "max_per_person": MAX_EMBEDDINGS_PER_PERSON,
            "threshold": DEFAULT_SIMILARITY_THRESHOLD,
        }
