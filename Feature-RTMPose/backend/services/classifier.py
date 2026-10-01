"""Activity classifier using RTMPose (via rtmlib/ONNX) + sklearn.

Replaces MediaPipe with RTMPose for pose estimation.
Key difference: COCO 17-point skeleton instead of MediaPipe 33-point.
License: Apache 2.0 (rtmlib + MMPose) — free for commercial use.
"""

import base64
import os
import time

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from services.motion import MOTION_FEATURE_NAMES, WINDOW_SEC, PoseTracker, body_frame, motion_features


import json as _json

# Default global paths — used when no project is active
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
MAX_VERSIONS = 5

# RTMPose model paths (ONNX) — set to None for auto-download on first use
# To use custom models, set these to the ONNX file paths in MODELS_DIR
POSE_MODEL_PATH = None  # auto-downloads RTMPose default model
DET_MODEL_PATH = None   # auto-downloads the YOLOX person detector (humanart weights)

# No default activity classes — model starts empty, learns only from real labeled data.
# Previously had synthetic baseline ["sorting", "carrying", ...] with fake feature vectors
# that confused the classifier and gave false confidence. Removed.

# COCO 17-point keypoint names
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO skeleton connections (pairs of keypoint indices)
COCO_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],         # face
    [5, 6],                                    # shoulders
    [5, 7], [7, 9], [6, 8], [8, 10],         # arms
    [5, 11], [6, 12], [11, 12],              # torso
    [11, 13], [13, 15], [12, 14], [14, 16],  # legs
]

# COCO landmark index mapping for feature extraction
# (equivalent to MediaPipe indices 11,12,15,16,23,24,27,28)
COCO_L_SHOULDER = 5
COCO_R_SHOULDER = 6
COCO_L_WRIST = 9
COCO_R_WRIST = 10
COCO_L_HIP = 11
COCO_R_HIP = 12
COCO_L_ANKLE = 15
COCO_R_ANKLE = 16


POSE_FEATURE_NAMES = [
    "torso_angle", "arm_spread", "vertical_movement",
    "hand_height", "stride_length", "body_speed", "symmetry",
]
# Pose features come first, so a model trained on the original 7 still works:
# it is fed the first n_features_in_ values of the full vector.
FEATURE_NAMES = POSE_FEATURE_NAMES + MOTION_FEATURE_NAMES

# Sample every Nth frame of a labeled interval during training
TRAIN_FRAME_STEP = 3


def _n_features(model) -> int:
    """Number of features a fitted (MultiOutput) model expects."""
    n = getattr(model, "n_features_in_", None)
    if n is None and hasattr(model, "estimators_"):
        n = model.estimators_[0].n_features_in_
    return int(n) if n else len(POSE_FEATURE_NAMES)


POSE_MODES = {
    "lightweight": "RTMPose-s + YOLOX-tiny (fastest)",
    "balanced": "RTMPose-m + YOLOX-m (recommended)",
    "performance": "RTMPose-x + YOLOX-x (best accuracy)",
}


class ActivityClassifier:
    def __init__(self):
        self.body = None  # rtmlib Body pipeline
        self.pose_mode = "balanced"
        self.model: MultiOutputClassifier | None = None
        self.train_progress: dict = {}
        self.activity_labels: list[str] = []
        self.is_trained = False
        self.training_samples_count = 0
        self.last_trained: float | None = None
        self.active_version: str | None = None
        self.active_project_id: str | None = None
        self.ensemble_mode = False
        self._ensemble_models: list[tuple] = []
        self.training_history: list[dict] = []
        self.reid = None
        # Separate trackers: Pose Preview seeks through a video, Live Analysis streams it
        self.trackers = {"preview": PoseTracker(), "live": PoseTracker()}

        # Configurable paths — default to global, overridden per-project
        self._set_paths(DEFAULT_MODELS_DIR)

    def _set_paths(self, base_dir: str):
        """Configure all storage paths relative to a base directory."""
        self.models_dir = base_dir
        self.versions_dir = os.path.join(base_dir, "versions")
        self.model_path = os.path.join(base_dir, "activity_model.pkl")
        self.encoder_path = os.path.join(base_dir, "label_encoder.pkl")
        self.history_path = os.path.join(base_dir, "training_history.json")
        self.dataset_path = os.path.join(base_dir, "dataset.json")

    def set_project_paths(self, project_dir: str | None):
        """Switch storage paths to a project directory, or back to global."""
        if project_dir:
            self._set_paths(project_dir)
        else:
            self._set_paths(DEFAULT_MODELS_DIR)

    def load(self, mode: str = "balanced"):
        """Initialize RTMPose pipeline and load trained activity model if exists."""
        self._load_pose_model(mode)

        # Initialize ReID (optional — silently disabled if model not found)
        from services.reid import PersonReID
        self.reid = PersonReID()
        self.reid.load()

        self._load_model_state()

    def _load_model_state(self):
        """Load classifier model + history from current paths. Called on startup and project switch."""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.versions_dir, exist_ok=True)

        self._load_history()

        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            self.model = joblib.load(self.model_path)
            labels_data = joblib.load(self.encoder_path)
            if isinstance(labels_data, list):
                self.activity_labels = labels_data
            elif hasattr(labels_data, 'classes_'):
                self.activity_labels = list(labels_data.classes_)
            self.is_trained = True
            if self.training_history:
                last = self.training_history[-1]
                self.active_version = last.get("version_id")
                self.training_samples_count = last.get("total_samples") or last.get("samples") or 0
        else:
            # Untrained — reset state
            self.model = None
            self.activity_labels = []
            self.is_trained = False
            self.active_version = None
            self.training_samples_count = 0
        # Ensemble models belong to the previous project; reload them for this one
        if self.ensemble_mode:
            self._load_ensemble_models()

    def _load_pose_model(self, mode: str):
        """Load or reload the RTMPose pipeline with the given mode."""
        from rtmlib import Body

        if mode not in POSE_MODES:
            raise ValueError(f"Unknown mode: {mode}. Choose from: {list(POSE_MODES.keys())}")

        self.body = Body(
            det=DET_MODEL_PATH,
            pose=POSE_MODEL_PATH,
            mode=mode,
            backend="onnxruntime",
            device="cpu",
        )
        self.pose_mode = mode

    def switch_pose_model(self, mode: str) -> dict:
        """Switch to a different RTMPose model size at runtime.
        Models are auto-downloaded on first use (~50-150MB).
        """
        if mode == self.pose_mode:
            return {"status": "unchanged", "mode": mode, "description": POSE_MODES[mode]}

        self._load_pose_model(mode)
        return {"status": "switched", "mode": mode, "description": POSE_MODES[mode]}

    @staticmethod
    def _bbox_from_keypoints(kps: np.ndarray, scs: np.ndarray) -> tuple[float, float, float, float]:
        """Compute bounding box (x1, y1, x2, y2) from visible keypoints."""
        visible = scs > 0.3
        if not np.any(visible):
            return (0, 0, 1, 1)
        pts = kps[visible]
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        """IoU between two (x1,y1,x2,y2) boxes."""
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def _detect_all_poses(self, image: np.ndarray) -> list[tuple]:
        """Run RTMPose detection. Returns list of (keypoints, scores) for all detected persons.

        Each entry: (kps: (17,2) pixel coords, scs: (17,) confidence scores)
        Filtered: only persons with >= 4 visible keypoints.
        Deduplicated: overlapping detections (IoU > 0.5) merged — keeps higher confidence.
        """
        keypoints, scores = self.body(image)

        if keypoints is None or len(keypoints) == 0:
            return []

        persons = []
        for i in range(len(keypoints)):
            kps = keypoints[i]  # (17, 2)
            scs = scores[i]     # (17,)
            visible = np.sum(scs > 0.3)
            if visible >= 4:
                persons.append((kps, scs))

        # NMS: remove duplicate detections of the same person
        if len(persons) <= 1:
            return persons

        # Sort by average keypoint confidence (descending) — keep best first
        persons.sort(key=lambda p: float(np.mean(p[1])), reverse=True)

        kept = []
        suppressed = set()
        for i, (kps_i, scs_i) in enumerate(persons):
            if i in suppressed:
                continue
            kept.append((kps_i, scs_i))
            bbox_i = self._bbox_from_keypoints(kps_i, scs_i)
            for j in range(i + 1, len(persons)):
                if j in suppressed:
                    continue
                bbox_j = self._bbox_from_keypoints(persons[j][0], persons[j][1])
                if self._iou(bbox_i, bbox_j) > 0.5:
                    suppressed.add(j)

        return kept

    def _detect_pose(self, image: np.ndarray):
        """Run RTMPose detection. Returns (keypoints, scores) for first person, or (None, None).
        Kept for backward compatibility with single-person pipeline (live analysis, training).
        """
        persons = self._detect_all_poses(image)
        if not persons:
            return None, None
        return persons[0]

    def detect_landmarks(self, image: np.ndarray) -> list[dict] | None:
        """Detect pose landmarks and return normalized coordinates (0..1) for first person."""
        kps, scs = self._detect_pose(image)
        if kps is None:
            return None

        h, w = image.shape[:2]
        return [
            {
                "x": round(float(kps[i, 0]) / w, 4),
                "y": round(float(kps[i, 1]) / h, 4),
                "visibility": round(float(scs[i]), 3),
            }
            for i in range(len(kps))
        ]

    def _features_from_landmarks(self, kps, scs, img_h, img_w) -> list[float]:
        """Compute 7D feature vector from COCO 17-point landmarks.

        kps: (17, 2) pixel coordinates
        scs: (17,) confidence scores
        img_h, img_w: image dimensions for normalization
        """
        # Normalize to 0..1 range
        def nx(i): return float(kps[i, 0]) / img_w
        def ny(i): return float(kps[i, 1]) / img_h
        def vis(i): return float(scs[i])

        l_shoulder_x, l_shoulder_y = nx(COCO_L_SHOULDER), ny(COCO_L_SHOULDER)
        r_shoulder_x, r_shoulder_y = nx(COCO_R_SHOULDER), ny(COCO_R_SHOULDER)
        l_wrist_x, l_wrist_y = nx(COCO_L_WRIST), ny(COCO_L_WRIST)
        r_wrist_x, r_wrist_y = nx(COCO_R_WRIST), ny(COCO_R_WRIST)
        l_hip_x, l_hip_y = nx(COCO_L_HIP), ny(COCO_L_HIP)
        r_hip_x, r_hip_y = nx(COCO_R_HIP), ny(COCO_R_HIP)
        l_ankle_x, l_ankle_y = nx(COCO_L_ANKLE), ny(COCO_L_ANKLE)
        r_ankle_x, r_ankle_y = nx(COCO_R_ANKLE), ny(COCO_R_ANKLE)

        # Feature 1: Torso angle (forward lean)
        mid_shoulder_y = (l_shoulder_y + r_shoulder_y) / 2
        mid_hip_y = (l_hip_y + r_hip_y) / 2
        mid_shoulder_x = (l_shoulder_x + r_shoulder_x) / 2
        mid_hip_x = (l_hip_x + r_hip_x) / 2
        torso_angle = abs(mid_shoulder_x - mid_hip_x) + abs(mid_shoulder_y - mid_hip_y) * 0.5

        # Feature 2: Arm spread
        arm_spread = abs(l_wrist_x - r_wrist_x) + abs(l_wrist_y - r_wrist_y)

        # Feature 3: Vertical movement (wrist height relative to shoulder)
        vertical_movement = abs(
            ((l_wrist_y + r_wrist_y) / 2) - mid_shoulder_y
        )

        # Feature 4: Hand height (inverse — higher hands = higher value)
        hand_height = 1.0 - ((l_wrist_y + r_wrist_y) / 2)

        # Feature 5: Stride length
        stride_length = abs(l_ankle_x - r_ankle_x) + abs(l_ankle_y - r_ankle_y)

        # Feature 6: Body speed proxy (visibility-based stability)
        vis_values = [
            vis(COCO_L_SHOULDER), vis(COCO_R_SHOULDER),
            vis(COCO_L_HIP), vis(COCO_R_HIP),
        ]
        body_speed = 1.0 - min(float(np.mean(vis_values)), 1.0)

        # Feature 7: Symmetry
        left_arm_len = np.sqrt(
            (l_shoulder_x - l_wrist_x) ** 2 + (l_shoulder_y - l_wrist_y) ** 2
        )
        right_arm_len = np.sqrt(
            (r_shoulder_x - r_wrist_x) ** 2 + (r_shoulder_y - r_wrist_y) ** 2
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

    def _person_features(self, persons: list[tuple], t: float, tracker: PoseTracker,
                         img_h: int, img_w: int) -> list[tuple[list[float], int]]:
        """Full feature vector (pose + motion) and track id for every person in a frame.

        persons must be the frame's full detection list: the tracker needs all of
        them to keep identities apart, and nearest_person needs the neighbours.
        """
        tracks = tracker.update(t, persons)
        frames = [body_frame(k, s) for k, s in persons]
        out = []
        for i, (kps, scs) in enumerate(persons):
            others = frames[:i] + frames[i + 1:]
            feats = self._features_from_landmarks(kps, scs, img_h, img_w) + motion_features(tracks[i], others)
            out.append((feats, tracks[i].id))
        return out

    def predict_image(self, image_bytes: bytes, include_landmarks: bool = False,
                      video_time: float | None = None, stream: str = "preview") -> dict:
        """Classify activity from raw image bytes.

        video_time: position of this frame in the video (s). Motion features need
        it to measure speed; without it, wall-clock time is used.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self._predict(image, include_landmarks=include_landmarks,
                             video_time=video_time, stream=stream)

    def predict_base64(self, b64_data: str) -> dict:
        """Classify activity from a base64-encoded image (live stream)."""
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self._predict(image, stream="live")

    def _classify_person(self, kps, scs, img_h, img_w, features: list[float] | None = None) -> dict:
        """Classify a single person's activities (multi-label) from their keypoints.

        Returns:
            activities: list of detected activities (e.g. ["walking", "carrying"])
            activity: primary activity (highest confidence, for backward compat)
            all_scores: {activity: probability} for each known label
        """
        if features is None:
            features = self._features_from_landmarks(kps, scs, img_h, img_w)

        # Untrained model — return features but no classification
        if not self.is_trained or self.model is None or not self.activity_labels:
            landmarks = [
                {
                    "x": round(float(np.clip(kps[i, 0] / img_w, 0, 1)), 4),
                    "y": round(float(np.clip(kps[i, 1] / img_h, 0, 1)), 4),
                    "visibility": round(float(scs[i]), 3),
                }
                for i in range(len(kps))
            ]
            return {
                "activity": "untrained",
                "activities": [],
                "confidence": 0.0,
                "all_scores": {},
                "features": [round(f, 4) for f in features],
                "landmarks": landmarks,
            }

        # Ensemble mode: run all saved versions, merge by averaging
        if self.ensemble_mode and self._ensemble_models:
            all_scores, active_labels = self._ensemble_classify(features)
            if all_scores:
                primary = max(all_scores, key=all_scores.get)
                primary_conf = all_scores[primary]
            else:
                primary = "unknown"
                primary_conf = 0.0

            landmarks = [
                {
                    "x": round(float(np.clip(kps[i, 0] / img_w, 0, 1)), 4),
                    "y": round(float(np.clip(kps[i, 1] / img_h, 0, 1)), 4),
                    "visibility": round(float(scs[i]), 3),
                }
                for i in range(len(kps))
            ]

            return {
                "activity": primary,
                "activities": active_labels,
                "confidence": round(primary_conf, 3),
                "all_scores": all_scores,
                "features": [round(f, 4) for f in features],
                "landmarks": landmarks,
                "ensemble": True,
            }

        X = np.array([features[:_n_features(self.model)]])

        # MultiOutputClassifier: predict_proba returns list of arrays, one per label
        # Each array shape: (1, 2) for binary [prob_no, prob_yes]
        try:
            proba_list = self.model.predict_proba(X)
            all_scores = {}
            active_labels = []

            for i, label_name in enumerate(self.activity_labels):
                if i < len(proba_list):
                    proba = proba_list[i][0]  # probabilities for this label
                    if len(proba) == 2:
                        prob_yes = float(proba[1])  # [prob_no, prob_yes]
                    elif len(proba) == 1:
                        # Single class — estimator only saw one class during training
                        # Check which class it is: if classes_=[1], it's always "yes"
                        est_classes = self.model.estimators_[i].classes_
                        prob_yes = float(proba[0]) if est_classes[0] == 1 else 0.0
                    else:
                        prob_yes = 0.0
                else:
                    prob_yes = 0.0
                all_scores[label_name] = round(prob_yes, 3)
                if prob_yes >= 0.5:
                    active_labels.append(label_name)

            # Primary activity = highest probability (for backward compatibility)
            if all_scores:
                primary = max(all_scores, key=all_scores.get)
                primary_conf = all_scores[primary]
            else:
                primary = "unknown"
                primary_conf = 0.0

        except Exception:
            # Fallback for old single-label model or any error
            try:
                probabilities = self.model.predict_proba(X)[0]
                predicted_idx = int(np.argmax(probabilities))
                primary = self.activity_labels[predicted_idx] if predicted_idx < len(self.activity_labels) else "unknown"
                primary_conf = float(probabilities[predicted_idx])
                all_scores = {
                    self.activity_labels[i]: round(float(p), 3)
                    for i, p in enumerate(probabilities) if i < len(self.activity_labels)
                }
                active_labels = [primary] if primary != "unknown" else []
            except Exception:
                primary = "error"
                primary_conf = 0.0
                all_scores = {}
                active_labels = []

        landmarks = [
            {
                "x": round(float(np.clip(kps[i, 0] / img_w, 0, 1)), 4),
                "y": round(float(np.clip(kps[i, 1] / img_h, 0, 1)), 4),
                "visibility": round(float(scs[i]), 3),
            }
            for i in range(len(kps))
        ]

        return {
            "activity": primary,                    # backward compat: single top activity
            "activities": active_labels,            # NEW: all active labels
            "confidence": round(primary_conf, 3),
            "all_scores": all_scores,
            "features": [round(f, 4) for f in features],
            "landmarks": landmarks,
        }

    def _identify_person(self, image: np.ndarray, kps: np.ndarray, scs: np.ndarray) -> dict:
        """Run ReID on a detected person crop. Returns identity match info."""
        if not self.reid or not self.reid.enabled:
            return {"name": "unknown", "similarity": 0.0, "matched": False}

        crop = self.reid.crop_person(image, kps, scs)
        if crop is None:
            return {"name": "unknown", "similarity": 0.0, "matched": False}

        embedding = self.reid.extract_embedding(crop)
        return self.reid.match(embedding)

    def _predict(self, image: np.ndarray, include_landmarks: bool = False,
                 video_time: float | None = None, stream: str = "preview") -> dict:
        """Run prediction pipeline on an OpenCV image.

        When include_landmarks=True, returns all detected persons in a 'persons' array.
        The top-level activity/confidence/landmarks are for the primary (first) person
        for backward compatibility with live analysis and training.
        """
        start = time.time()
        h, w = image.shape[:2]
        tracker = self.trackers[stream]
        t = video_time if video_time is not None else start

        all_persons = self._detect_all_poses(image)

        if not all_persons:
            tracker.update(t, [])
            return {
                "activity": "no_person",
                "confidence": 0.0,
                "all_scores": {},
                "features": [],
                "landmarks": [],
                "persons": [],
                "person_count": 0,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "pose_detected": False,
            }

        # Classify all detected persons
        # person_id is detection order per frame (left-to-right by x-center), which is
        # what the labeling UI's "#N" refers to. track_id follows a person across
        # consecutive frames (services/motion.py) and resets on a seek.
        all_persons.sort(key=lambda p: float(np.mean(p[0][:, 0])))
        per_person = self._person_features(all_persons, t, tracker, h, w)

        persons = []
        for idx, (kps, scs) in enumerate(all_persons):
            feats, track_id = per_person[idx]
            person_data = self._classify_person(kps, scs, h, w, features=feats)
            person_data["person_id"] = idx + 1
            person_data["track_id"] = track_id

            # ReID: identify person if enabled and landmarks requested
            if include_landmarks:
                identity = self._identify_person(image, kps, scs)
                person_data["identity"] = identity
            else:
                person_data["identity"] = {"name": "unknown", "similarity": 0.0, "matched": False}

            persons.append(person_data)

        # Primary person = leftmost (the list is sorted by x above); Live Analysis
        # reports only this one, Pose Preview returns everyone
        primary = persons[0]
        latency = round((time.time() - start) * 1000, 1)

        result = {
            "activity": primary["activity"],
            "confidence": primary["confidence"],
            "all_scores": primary["all_scores"],
            "features": primary["features"],
            "landmarks": primary["landmarks"] if include_landmarks else [],
            "persons": persons if include_landmarks else [],
            "person_count": len(persons),
            "latency_ms": latency,
            "pose_detected": True,
        }
        return result

    def _fit_and_save(self, features_list: list, label_sets: list[list[str]],
                       source_info: dict) -> dict:
        """Shared multi-label training core.

        Args:
            features_list: list of feature vectors (7 pose + 8 motion values)
            label_sets: list of label lists per sample, e.g. [["walking","carrying"], ["idle"]]
            source_info: metadata for history entry (training_files, label_count, etc.)

        Returns: result dict with status, version_id, samples, classes
        """
        if not features_list:
            return {"status": "error", "message": "No features extracted"}

        real_sample_count = len(features_list)

        # Collect all unique activity labels from real data only
        all_labels = set()
        for ls in label_sets:
            all_labels.update(ls)
        self.activity_labels = sorted(all_labels)

        if not self.activity_labels:
            return {"status": "error", "message": "No activity labels found in training data"}

        # Build multi-label Y matrix: each row has 0/1 per activity
        label_to_idx = {l: i for i, l in enumerate(self.activity_labels)}
        Y = np.zeros((len(features_list), len(self.activity_labels)), dtype=int)
        for row_idx, ls in enumerate(label_sets):
            for lbl in ls:
                if lbl in label_to_idx:
                    Y[row_idx, label_to_idx[lbl]] = 1

        X = np.array(features_list)

        self.model = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        self.model.fit(X, Y)
        self.is_trained = True
        self.last_trained = time.time()
        self.training_samples_count = real_sample_count

        # Save active
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.activity_labels, self.encoder_path)

        # Save versioned
        version_id = time.strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(self.versions_dir, version_id)
        os.makedirs(version_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(version_dir, "model.pkl"))
        joblib.dump(self.activity_labels, os.path.join(version_dir, "encoder.pkl"))
        self.active_version = version_id

        entry = {
            "version_id": version_id,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": real_sample_count,
            "classes": sorted(all_labels),
            **source_info,
        }
        self.training_history.append(entry)
        self._save_history()
        self._cleanup_old_versions()
        if self.ensemble_mode:
            self._load_ensemble_models()

        return {
            "status": "success",
            "version_id": version_id,
            "samples": real_sample_count,
            "classes": sorted(all_labels),
        }

    # ===== Model versioning =====

    def _load_history(self):
        if os.path.exists(self.history_path):
            with open(self.history_path) as f:
                self.training_history = _json.load(f)
        else:
            self.training_history = []

    def _save_history(self):
        with open(self.history_path, "w") as f:
            _json.dump(self.training_history, f, indent=2)

    def _cleanup_old_versions(self):
        """Keep only the last MAX_VERSIONS versioned models."""
        if len(self.training_history) <= MAX_VERSIONS:
            return
        to_remove = self.training_history[:-MAX_VERSIONS]
        self.training_history = self.training_history[-MAX_VERSIONS:]
        self._save_history()
        for entry in to_remove:
            vdir = os.path.join(self.versions_dir, entry["version_id"])
            if os.path.isdir(vdir):
                import shutil
                shutil.rmtree(vdir, ignore_errors=True)

    def switch_model_version(self, version_id: str) -> dict:
        """Load a previously trained model version."""
        version_dir = os.path.join(self.versions_dir, version_id)
        model_path = os.path.join(version_dir, "model.pkl")
        encoder_path = os.path.join(version_dir, "encoder.pkl")

        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            return {"status": "error", "message": f"Version {version_id} not found"}

        self.model = joblib.load(model_path)
        labels_data = joblib.load(encoder_path)

        # Handle both old LabelEncoder and new activity_labels list
        if isinstance(labels_data, list):
            self.activity_labels = labels_data
        elif hasattr(labels_data, 'classes_'):
            self.activity_labels = list(labels_data.classes_)
        else:
            self.activity_labels = []

        self.active_version = version_id
        self.is_trained = True

        # Copy to active model path for persistence across restarts
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.activity_labels, self.encoder_path)

        entry = next((h for h in self.training_history if h["version_id"] == version_id), None)
        return {
            "status": "switched",
            "version_id": version_id,
            "classes": list(self.activity_labels),
            "samples": entry["samples"] if entry else 0,
        }

    def delete_version(self, version_id: str) -> dict:
        """Delete a saved model version. Cannot delete the active version."""
        if version_id == self.active_version:
            return {"status": "error", "message": "Cannot delete the active version. Switch to another first."}

        version_dir = os.path.join(self.versions_dir, version_id)
        if not os.path.isdir(version_dir):
            return {"status": "error", "message": f"Version {version_id} not found"}

        import shutil
        shutil.rmtree(version_dir, ignore_errors=True)

        self.training_history = [h for h in self.training_history if h["version_id"] != version_id]
        self._save_history()

        return {"status": "deleted", "version_id": version_id}

    # ===== Persistent training dataset =====
    # Stores extracted features + labels + source metadata per sample.
    # Accumulates across labeling sessions so the model improves over time.

    # Dataset path is now an instance attribute set by _set_paths()

    def _load_dataset(self) -> tuple[list, list, list]:
        """Load persisted dataset. Returns (features, label_sets, metadata).
        Handles old format (no metadata) and old filename gracefully.
        """
        # Try current path first, then legacy path
        path = self.dataset_path
        if not os.path.exists(path):
            legacy = os.path.join(self.models_dir, "training_dataset.json")
            if os.path.exists(legacy):
                path = legacy

        if os.path.exists(path):
            with open(path) as f:
                data = _json.load(f)
            features = data.get("features", [])
            label_sets = data.get("label_sets", [])
            metadata = data.get("metadata", [{"source": "unknown"}] * len(features))
            return features, label_sets, metadata
        return [], [], []

    def _save_dataset(self, features: list, label_sets: list, metadata: list):
        """Persist training dataset with metadata."""
        with open(self.dataset_path, "w") as f:
            _json.dump({"features": features, "label_sets": label_sets, "metadata": metadata}, f)

    def get_dataset_info(self) -> dict:
        """Return detailed breakdown of the persisted training dataset."""
        features, label_sets, metadata = self._load_dataset()

        # Count samples per activity
        activity_counts = {}
        for ls in label_sets:
            for lbl in ls:
                activity_counts[lbl] = activity_counts.get(lbl, 0) + 1

        # Count samples per source video
        source_counts = {}
        for m in metadata:
            src = m.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        return {
            "sample_count": len(features),
            "has_data": len(features) > 0,
            "activities": [
                {"name": name, "count": count}
                for name, count in sorted(activity_counts.items(), key=lambda x: -x[1])
            ],
            "sources": [
                {"name": name, "count": count}
                for name, count in sorted(source_counts.items(), key=lambda x: -x[1])
            ],
        }

    def clear_dataset(self) -> dict:
        """Clear the persisted training dataset."""
        if os.path.exists(self.dataset_path):
            os.remove(self.dataset_path)
        return {"status": "cleared"}

    def remove_dataset_source(self, source_name: str) -> dict:
        """Remove all samples from a specific source video."""
        features, label_sets, metadata = self._load_dataset()
        new_features = []
        new_labels = []
        new_meta = []
        removed = 0
        for f, ls, m in zip(features, label_sets, metadata):
            if m.get("source") == source_name:
                removed += 1
            else:
                new_features.append(f)
                new_labels.append(ls)
                new_meta.append(m)
        self._save_dataset(new_features, new_labels, new_meta)
        return {"status": "removed", "removed_samples": removed, "remaining": len(new_features)}

    def retrain_on_dataset(self) -> dict:
        """Re-train the model on the full persisted dataset."""
        features, label_sets, metadata = self._load_dataset()
        if not features:
            return {"status": "error", "message": "Dataset is empty. Label some video intervals first."}
        return self._fit_and_save(
            list(features), list(label_sets),
            {"source": "full_dataset", "total_samples": len(features)}
        )

    def train_from_labels(self, video_path: str, labels: list[dict],
                          append_to_existing: bool = True,
                          raw_labels: dict | None = None) -> dict:
        """Train from labeled time intervals. Multi-label supported.

        Each label: {start_time, end_time, activities: ["walking","carrying"], person}
        If activities key is missing, falls back to activity (single string).

        When append_to_existing=True, replaces this video's earlier samples in the
        persisted dataset with the new ones, keeps every other clip, and retrains
        on everything. Re-training the same video therefore never duplicates it.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.train_progress = {}
            return {"status": "error", "message": f"Cannot open video: {video_path}"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_labels = len(labels)

        new_features = []
        new_label_sets = []
        new_metadata = []
        video_name = os.path.basename(video_path)

        # Calculate total frames to process for progress tracking
        total_frames_to_process = 0
        for li in labels:
            sf = int(li.get("start_time", 0) * fps)
            ef = int(li.get("end_time", 0) * fps)
            total_frames_to_process += max(0, (ef - sf) // TRAIN_FRAME_STEP + 1)

        # Motion features need history, so tracking starts this many frames
        # before each labeled interval; those warm-up frames produce no samples.
        warmup_frames = int(WINDOW_SEC * fps)

        frames_processed = 0
        self.train_progress = {
            "status": "extracting",
            "label_current": 0,
            "label_total": total_labels,
            "frames_processed": 0,
            "frames_total": total_frames_to_process,
            "features_extracted": 0,
            "percent": 0,
        }

        for label_idx, label_info in enumerate(labels):
            start_t = label_info.get("start_time", 0)
            end_t = label_info.get("end_time", 0)
            activities = label_info.get("activities")
            if not activities:
                act = label_info.get("activity", "unknown")
                activities = [act] if act else ["unknown"]
            person_filter = label_info.get("person", "all")

            start_frame = int(start_t * fps)
            end_frame = int(end_t * fps)

            self.train_progress["label_current"] = label_idx + 1
            self.train_progress["current_activity"] = "+".join(activities)

            tracker = PoseTracker()
            frame_idx = max(0, start_frame - warmup_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            while frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_idx - start_frame) % TRAIN_FRAME_STEP == 0:
                    all_persons = self._detect_all_poses(frame)
                    all_persons.sort(key=lambda p: float(np.mean(p[0][:, 0])))
                    h, w = frame.shape[:2]
                    per_person = self._person_features(all_persons, frame_idx / fps, tracker, h, w)

                    if frame_idx >= start_frame:
                        frames_processed += 1
                        self.train_progress["frames_processed"] = frames_processed
                        self.train_progress["percent"] = round(frames_processed / max(1, total_frames_to_process) * 90)

                        if person_filter == "all":
                            chosen = range(len(per_person))
                        else:
                            person_idx = int(person_filter.replace("#", "")) - 1
                            chosen = [person_idx] if person_idx < len(per_person) else []

                        for pi in chosen:
                            feats, track_id = per_person[pi]
                            new_features.append(feats)
                            new_label_sets.append(activities)
                            new_metadata.append({"source": video_name, "activities": activities,
                                                 "frame": frame_idx, "person": pi + 1, "track": track_id})

                        self.train_progress["features_extracted"] = len(new_features)

                frame_idx += 1

        cap.release()

        if not new_features:
            self.train_progress = {}
            return {"status": "error", "message": "Could not extract features from labeled intervals"}

        self.train_progress["status"] = "training"
        self.train_progress["percent"] = 92

        # Merge with existing dataset if appending
        if append_to_existing:
            existing_features, existing_labels, existing_meta = self._load_dataset()
            # Rows from before the motion features (7 values) cannot share a
            # matrix with the current vectors; drop them rather than fail.
            n = len(new_features[0])
            keep = [i for i, f in enumerate(existing_features) if len(f) == n]
            if len(keep) < len(existing_features):
                print(f"[Train] Dropped {len(existing_features) - len(keep)} dataset rows "
                      f"with an old feature layout")
            # The UI sends every label of the current video on each Add & Train,
            # so this video's previous samples are replaced rather than appended to
            keep = [i for i in keep if existing_meta[i].get("source") != video_name]
            existing_features = [existing_features[i] for i in keep]
            existing_labels = [existing_labels[i] for i in keep]
            existing_meta = [existing_meta[i] for i in keep]
            all_features = existing_features + new_features
            all_label_sets = existing_labels + new_label_sets
            all_metadata = existing_meta + new_metadata
        else:
            all_features = new_features
            all_label_sets = new_label_sets
            all_metadata = new_metadata

        # Persist the combined dataset
        self._save_dataset(all_features, all_label_sets, all_metadata)

        # Train on the full dataset
        self.train_progress["status"] = "fitting"
        self.train_progress["percent"] = 95

        source_info = {
            "training_files": [os.path.basename(video_path)],
            "label_count": len(labels),
            "new_samples": len(new_features),
            "total_samples": len(all_features),
            "source": "labeled_intervals",
        }
        if raw_labels:
            source_info["raw_labels"] = raw_labels

        result = self._fit_and_save(
            list(all_features), list(all_label_sets), source_info
        )

        self.train_progress = {"status": "done", "percent": 100}
        return result

    def get_classes(self) -> list[str]:
        return list(self.activity_labels)

    # ===== Ensemble mode =====

    def set_ensemble(self, enabled: bool) -> dict:
        """Enable/disable ensemble mode. When enabled, predictions use ALL saved versions."""
        self.ensemble_mode = enabled
        if enabled:
            self._load_ensemble_models()
            return {
                "status": "enabled",
                "models_loaded": len(self._ensemble_models),
                "versions": [v["version_id"] for v in self.training_history],
            }
        else:
            self._ensemble_models = []
            return {"status": "disabled"}

    def _load_ensemble_models(self):
        """Load all saved model versions for ensemble prediction."""
        self._ensemble_models = []
        for entry in self.training_history:
            vid = entry["version_id"]
            vdir = os.path.join(self.versions_dir, vid)
            model_path = os.path.join(vdir, "model.pkl")
            encoder_path = os.path.join(vdir, "encoder.pkl")
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                model = joblib.load(model_path)
                labels = joblib.load(encoder_path)
                if not isinstance(labels, list):
                    labels = list(labels.classes_) if hasattr(labels, 'classes_') else []
                self._ensemble_models.append((model, labels))

    def _ensemble_classify(self, features: list[float]) -> tuple[dict, list[str]]:
        """Run all ensemble models and merge results by averaging probabilities.

        Returns: (all_scores: {activity: avg_prob}, active_labels: [activities above 0.5])
        """
        # Collect votes: {activity_name: [prob_from_model1, prob_from_model2, ...]}
        votes: dict[str, list[float]] = {}

        for model, labels in self._ensemble_models:
            try:
                proba_list = model.predict_proba(np.array([features[:_n_features(model)]]))
                for i, label_name in enumerate(labels):
                    if i < len(proba_list):
                        proba = proba_list[i][0]
                        if len(proba) == 2:
                            p = float(proba[1])
                        elif len(proba) == 1:
                            est_classes = model.estimators_[i].classes_
                            p = float(proba[0]) if est_classes[0] == 1 else 0.0
                        else:
                            p = 0.0
                    else:
                        p = 0.0
                    if label_name not in votes:
                        votes[label_name] = []
                    votes[label_name].append(p)
            except Exception:
                continue

        if not votes:
            return {}, []

        # Average probabilities across models
        all_scores = {name: round(float(np.mean(probs)), 3) for name, probs in votes.items()}
        active_labels = [name for name, avg in all_scores.items() if avg >= 0.5]

        return all_scores, active_labels

    def get_status(self) -> dict:
        status = {
            "is_trained": self.is_trained,
            "model_type": "MultiOutputClassifier" if not self.ensemble_mode else f"Ensemble ({len(self._ensemble_models)} models)",
            "pose_model": POSE_MODES.get(self.pose_mode, self.pose_mode),
            "pose_mode": self.pose_mode,
            "active_project": self.active_project_id,
            "classes": self.get_classes(),
            "training_samples": self.training_samples_count,
            "last_trained": self.last_trained,
            "active_version": self.active_version,
            "versions": self.training_history,
            "ensemble": self.ensemble_mode,
            "ensemble_count": len(self._ensemble_models) if self.ensemble_mode else 0,
            "features": FEATURE_NAMES,
            "model_features": _n_features(self.model) if self.model is not None else None,
        }
        if self.reid:
            status["reid"] = self.reid.get_status()
        return status
