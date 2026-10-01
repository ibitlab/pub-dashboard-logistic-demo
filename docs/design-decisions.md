# Design decisions

Why the pipeline is built the way it is. Measured results: [README → Demo results](../README.md#demo-results).

## 1. Two stages: pose model → small classifier

```
frame ─► Stage 1: pose model (frozen)   ─► keypoints ─► features ─► Stage 2: RandomForest ─► activity
         MediaPipe or RTMPose                                       (retrained by "Train Model")
```

| | Feature-MediaPipe | Feature-RTMPose |
|---|---|---|
| Stage 1 | PoseLandmarker `.task` (lite / full / heavy), 33 landmarks | rtmlib `Body` (YOLOX person detector + RTMPose) on ONNX Runtime, 17 COCO keypoints |
| Features | 7 per-frame pose features | same 7 + 8 motion features over a 1.2 s per-person track window |
| Stage 2 | `RandomForestClassifier`, one label per frame | `MultiOutputClassifier(RandomForest)`, several labels per person |

- **Why pose, not boxes.** The activities differ by what the arms, hands and legs do (lifting, carrying, pushing, phone at ear), not by where a person is. A person box gives position and size only; keypoints give posture. Boxes are still used, but only for detection and for IoU matching between frames.
- **Why a small classifier.** A feature vector of 7 to 15 values trains in seconds on CPU from a few labelled clips, so users can retrain from the dashboard.
- **Consequence.** "Train Model" retrains Stage 2 only. It cannot fix missing people, misplaced joints or jitter from Stage 1: bad keypoints give bad features. The Pose Preview tool lets you check the skeleton before you train.

## 2. MediaPipe limits

- **One person per frame.** `_detect_pose()` returns `pose_landmarks[0]`, and `num_poses` is left at its default of 1. When there are several workers, only one is analysed.
- **"No person" when only the legs are visible.** The detector needs to see the upper body, so side or low-angle views miss people entirely. Retraining Stage 2 cannot fix this.
- **Cannot be fine-tuned.** `mediapipe-model-maker` has no pose-landmarker task: it covers only image classification, object detection, text classification, gesture recognition and face stylizing. The only way to improve Stage 1 is to switch between lite, full and heavy.
- **Result.** The Annotate tab's COCO export cannot improve MediaPipe. It is useful only for fine-tuning or evaluating in MMPose/RTMPose. The MediaPipe build exports 33 MediaPipe landmarks, not COCO-17, so MMPose would need a custom dataset definition.

## 3. Pose model choice

Requirements: multi-person, fine-tunable, COCO input, near-real-time on CPU, Python server-side, permissive licence.

| Model | Fine-tune | CPU speed | Licence | Verdict |
|---|---|---|---|---|
| MediaPipe PoseLandmarker | No | fast | Apache 2.0 | kept as the simple reference build |
| **RTMPose (MMPose)** | Yes, COCO native | ~0.3–0.4 s/frame with detector and ReID (balanced mode) | Apache 2.0 | **picked** |
| YOLO11-Pose (Ultralytics) | Yes, easiest | fast | AGPL-3.0 / paid | rejected: licence |
| MoveNet | No (training code not released) | fast | Apache 2.0 | rejected |
| ViTPose, AlphaPose | Yes | too slow without a GPU | — | rejected |
| OpenPose | Not practical | 1–3 FPS | non-commercial | rejected |

**RTMPose.** Apache 2.0, and runs through `rtmlib` with only ONNX Runtime, so there is no PyTorch or MMCV at inference. It is multi-person (with dedup at IoU > 0.5) and has three speed/accuracy modes. It can be fine-tuned in MMPose on the Annotate tab's COCO-17 export. The feature code only needed index remapping (MediaPipe 11/12/15/16/23/24/27/28 → COCO 5/6/9/10/11/12/15/16), because every landmark the features use exists in COCO-17.

**YOLO11-Pose rejected.** Its fine-tuning is the simplest of all the options, but it is AGPL-3.0. The network-use clause (section 13) means that serving the dashboard to users would require releasing the whole application under AGPL. Ultralytics also treats fine-tuned and ONNX-exported weights as covered. The only way out is a paid Enterprise licence, so we did not take the dependency.

## 4. Annotate tab

- **Workflow.** Create a project from a video. Scrub to a frame (the server extracts it with OpenCV) and press Space to auto-detect keypoints. Drag any wrong points, toggle occluded points to not-visible, and save with Ctrl/Cmd+S. Saves go to SQLite and overwrite by (project, frame).
- **Auto-detect differs between the builds.** Feature-MediaPipe calls `/api/annotations/projects/{id}/detect/{frame}`, which seeks the frame again on the server. Feature-RTMPose sends the displayed image to `/api/predict/landmarks`, so the keypoints match exactly what is on screen.
- **Scope.** One person per annotated frame: MediaPipe's single pose, or the leftmost person for RTMPose.
- **Export.** `GET /api/annotations/projects/{id}/export` returns COCO keypoint JSON. `POST …/export-frames` writes the matching JPEGs to `data/annotations/export_{id}/`.

| COCO field | Content |
|---|---|
| `images[]` | `file_name: frame_{index:06d}.jpg`, video width and height |
| `annotations[].keypoints` | `[x, y, v] × N` in pixels. `v = 2` if confidence > 0.3, else `v = 1` (`0` is never written) |
| `annotations[].bbox` | box around the visible keypoints plus a 20 px margin. `area = w × h`, `num_keypoints` = number visible |
| `categories[0]` | `person`, keypoint names (17 COCO or 33 MediaPipe), skeleton as 1-indexed pairs |

## 5. Out of scope

- **GPU training in the dashboard.** MMPose fine-tuning needs its own Linux + CUDA setup. Loading its ONNX output is not wired up yet: `POSE_MODEL_PATH` is `None`, so rtmlib downloads its default model.
- **One-click pose-model fine-tuning.** Not offered: MediaPipe cannot be fine-tuned (section 2), and RTMPose fine-tuning needs the GPU setup above.
