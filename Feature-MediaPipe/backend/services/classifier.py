"""Activity classifier using MediaPipe PoseLandmarker (Tasks API) + sklearn."""

import base64
import json
import os
import time
import uuid

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TRAINING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "training")
MODEL_PATH = os.path.join(MODELS_DIR, "activity_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
# Features + labels + source clip of every training sample, in the same layout
# as Feature-RTMPose, so Feature-RTMPose/scripts/eval_leave_one_clip_out.py
# can score this model too.
DATASET_PATH = os.path.join(TRAINING_DIR, "dataset.json")

# Suggested labels for the Training tab only. The model learns nothing until it
# is trained on real clips: there is no synthetic baseline.

# Known MediaPipe pose model variants (ordered by accuracy)
POSE_MODEL_VARIANTS = {
    "lite":  {"file": "pose_landmarker_lite.task",  "size_mb": 5.5,  "speed": "fastest",  "accuracy": "lowest"},
    "full":  {"file": "pose_landmarker_full.task",  "size_mb": 9.0,  "speed": "balanced", "accuracy": "medium"},
    "heavy": {"file": "pose_landmarker_heavy.task", "size_mb": 29.2, "speed": "slowest",  "accuracy": "highest"},
}


class ActivityClassifier:
    def __init__(self):
        self.pose_landmarker = None
        self.active_pose_model: str = ""  # filename of currently loaded .task
        self.model: RandomForestClassifier | None = None
        self.label_encoder: LabelEncoder | None = None
        self.is_trained = False
        self.training_samples_count = 0
        self.last_trained: float | None = None

    def load(self):
        """Initialize MediaPipe PoseLandmarker and load trained model if exists."""
        # Find the best available pose model (prefer lite as default)
        pose_model_path = self._find_default_pose_model()
        self._load_pose_model(pose_model_path)

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(TRAINING_DIR, exist_ok=True)

        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
            self.is_trained = True

    def _find_default_pose_model(self) -> str:
        """Find the best available .task file, preferring lite."""
        for variant in ["lite", "full", "heavy"]:
            path = os.path.join(MODELS_DIR, POSE_MODEL_VARIANTS[variant]["file"])
            if os.path.exists(path):
                return path
        # Fallback: any .task file
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".task"):
                return os.path.join(MODELS_DIR, f)
        raise FileNotFoundError("No pose .task model found in " + MODELS_DIR)

    def _load_pose_model(self, model_path: str):
        """Load (or hot-swap) a MediaPipe PoseLandmarker from a .task file."""
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        self.active_pose_model = os.path.basename(model_path)

    def list_pose_models(self) -> list[dict]:
        """Return all available .task files with metadata."""
        models = []
        for f in sorted(os.listdir(MODELS_DIR)):
            if not f.endswith(".task"):
                continue
            path = os.path.join(MODELS_DIR, f)
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)

            # Match to known variant
            variant = None
            for v, info in POSE_MODEL_VARIANTS.items():
                if info["file"] == f:
                    variant = v
                    break

            models.append({
                "filename": f,
                "size_mb": size_mb,
                "variant": variant or "custom",
                "speed": POSE_MODEL_VARIANTS.get(variant, {}).get("speed", "unknown"),
                "accuracy": POSE_MODEL_VARIANTS.get(variant, {}).get("accuracy", "unknown"),
                "active": f == self.active_pose_model,
            })
        return models

    def switch_pose_model(self, filename: str) -> dict:
        """Switch to a different .task model. Returns new active model info."""
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path) or not filename.endswith(".task"):
            raise FileNotFoundError(f"Model not found: {filename}")

        if filename == self.active_pose_model:
            return {"status": "already_active", "model": filename}

        self._load_pose_model(path)
        return {"status": "switched", "model": self.active_pose_model}

    def _detect_pose(self, image: np.ndarray):
        """Run MediaPipe pose detection once and return raw landmarks or None."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose_landmarker.detect(mp_image)
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None
        return result.pose_landmarks[0]

    def detect_landmarks(self, image: np.ndarray) -> list[dict] | None:
        """Detect pose landmarks and return normalized coordinates."""
        lm = self._detect_pose(image)
        if lm is None:
            return None
        return [
            {"x": round(pt.x, 4), "y": round(pt.y, 4), "visibility": round(float(pt.visibility), 3)}
            for pt in lm
        ]

    def _features_from_landmarks(self, lm) -> list[float]:
        """Compute 7D feature vector from raw MediaPipe landmarks."""
        # Key landmarks (NormalizedLandmark objects with .x, .y, .z, .visibility)
        l_shoulder = lm[11]
        r_shoulder = lm[12]
        l_wrist = lm[15]
        r_wrist = lm[16]
        l_hip = lm[23]
        r_hip = lm[24]
        l_ankle = lm[27]
        r_ankle = lm[28]

        # Feature 1: Torso angle (forward lean)
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        mid_hip_y = (l_hip.y + r_hip.y) / 2
        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
        mid_hip_x = (l_hip.x + r_hip.x) / 2
        torso_angle = abs(mid_shoulder_x - mid_hip_x) + abs(mid_shoulder_y - mid_hip_y) * 0.5

        # Feature 2: Arm spread
        arm_spread = abs(l_wrist.x - r_wrist.x) + abs(l_wrist.y - r_wrist.y)

        # Feature 3: Vertical movement (wrist height relative to shoulder)
        vertical_movement = abs(
            ((l_wrist.y + r_wrist.y) / 2) - mid_shoulder_y
        )

        # Feature 4: Hand height (inverse — higher hands = higher value)
        hand_height = 1.0 - ((l_wrist.y + r_wrist.y) / 2)

        # Feature 5: Stride length
        stride_length = abs(l_ankle.x - r_ankle.x) + abs(l_ankle.y - r_ankle.y)

        # Feature 6: Body speed proxy (visibility-based stability)
        vis_values = [l_shoulder.visibility, r_shoulder.visibility,
                      l_hip.visibility, r_hip.visibility]
        body_speed = 1.0 - min(float(np.mean(vis_values)), 1.0)

        # Feature 7: Symmetry
        left_arm_len = np.sqrt(
            (l_shoulder.x - l_wrist.x) ** 2 + (l_shoulder.y - l_wrist.y) ** 2
        )
        right_arm_len = np.sqrt(
            (r_shoulder.x - r_wrist.x) ** 2 + (r_shoulder.y - r_wrist.y) ** 2
        )
        symmetry = 1.0 - abs(left_arm_len - right_arm_len)

        return [
            float(np.clip(torso_angle, 0, 1)),
            float(np.clip(arm_spread, 0, 1)),
            float(np.clip(vertical_movement, 0, 1)),
            float(np.clip(hand_height, 0, 1)),
            float(np.clip(stride_length, 0, 1)),
            float(np.clip(body_speed, 0, 1)),
            float(np.clip(symmetry, 0, 1)),
        ]

    def extract_features(self, image: np.ndarray) -> list[float] | None:
        """Extract pose-based features from an image frame using PoseLandmarker."""
        lm = self._detect_pose(image)
        if lm is None:
            return None
        return self._features_from_landmarks(lm)

    def predict_image(self, image_bytes: bytes, include_landmarks: bool = False) -> dict:
        """Classify activity from raw image bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self._predict(image, include_landmarks=include_landmarks)

    def predict_base64(self, b64_data: str) -> dict:
        """Classify activity from a base64-encoded image."""
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self._predict(image)

    def _predict(self, image: np.ndarray, include_landmarks: bool = False) -> dict:
        """Run prediction pipeline on an OpenCV image."""
        start = time.time()

        # Single pose detection pass for both features and landmarks
        raw_lm = self._detect_pose(image)
        features = self._features_from_landmarks(raw_lm) if raw_lm is not None else None
        landmarks = (
            [{"x": round(pt.x, 4), "y": round(pt.y, 4), "visibility": round(float(pt.visibility), 3)} for pt in raw_lm]
            if include_landmarks and raw_lm is not None else None
        )

        if features is None:
            return {
                "activity": "no_person",
                "confidence": 0.0,
                "all_scores": {},
                "features": [],
                "landmarks": [],
                "latency_ms": round((time.time() - start) * 1000, 1),
                "pose_detected": False,
            }

        if not self.is_trained:
            # Pose is still detected and shown; there is just nothing to classify with
            return {
                "activity": "untrained",
                "confidence": 0.0,
                "all_scores": {},
                "features": [round(f, 4) for f in features],
                "landmarks": landmarks or [],
                "latency_ms": round((time.time() - start) * 1000, 1),
                "pose_detected": True,
            }

        X = np.array([features])
        probabilities = self.model.predict_proba(X)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(probabilities[predicted_idx])

        all_scores = {
            self.label_encoder.inverse_transform([i])[0]: round(float(p), 3)
            for i, p in enumerate(probabilities)
        }

        latency = round((time.time() - start) * 1000, 1)

        result = {
            "activity": predicted_label,
            "confidence": round(confidence, 3),
            "all_scores": all_scores,
            "features": [round(f, 4) for f in features],
            "landmarks": landmarks or [],
            "latency_ms": latency,
            "pose_detected": True,
        }
        return result

    async def add_training_sample(self, video_bytes: bytes, label: str) -> str:
        """Store a training video under data/training/{id}_{label}.mp4. Features are extracted by train()."""
        sample_id = str(uuid.uuid4())[:8]
        filepath = os.path.join(TRAINING_DIR, f"{sample_id}_{label}.mp4")

        with open(filepath, "wb") as f:
            f.write(video_bytes)

        self.training_samples_count += 1
        return sample_id

    def train(self) -> dict:
        """Re-train the model on all collected samples."""
        training_files = [
            f for f in os.listdir(TRAINING_DIR)
            if f.endswith((".mp4", ".avi", ".mov", ".webm"))
        ]

        if not training_files:
            return {"status": "error", "message": "No training samples found"}

        features_list = []
        labels = []
        sources = []

        for fname in training_files:
            label = fname.split("_", 1)[1].rsplit(".", 1)[0]
            filepath = os.path.join(TRAINING_DIR, fname)

            cap = cv2.VideoCapture(filepath)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % 5 == 0:
                    feats = self.extract_features(frame)
                    if feats:
                        features_list.append(feats)
                        labels.append(label)
                        sources.append(fname)
                frame_count += 1
            cap.release()

        if not features_list:
            return {"status": "error", "message": "Could not extract features from training data"}

        uploaded_classes = sorted(set(labels))
        with open(DATASET_PATH, "w") as f:
            json.dump({"features": features_list, "label_sets": [[l] for l in labels],
                       "metadata": [{"source": s} for s in sources]}, f)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(uploaded_classes)

        X = np.array(features_list)
        y = self.label_encoder.transform(labels)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        self.last_trained = time.time()
        self.training_samples_count = len(features_list)

        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.label_encoder, ENCODER_PATH)

        return {
            "status": "success",
            "samples": len(features_list),
            "classes": uploaded_classes,
        }

    def get_classes(self) -> list[str]:
        if self.label_encoder:
            return list(self.label_encoder.classes_)
        return []

    def get_status(self) -> dict:
        return {
            "is_trained": self.is_trained,
            "model_type": "RandomForestClassifier",
            "classes": self.get_classes(),
            "training_samples": self.training_samples_count,
            "last_trained": self.last_trained,
            "pose_model": self.active_pose_model,
            "features": [
                "torso_angle", "arm_spread", "vertical_movement",
                "hand_height", "stride_length", "body_speed", "symmetry"
            ],
        }
