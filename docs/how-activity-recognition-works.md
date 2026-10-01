# How activity recognition works

What Feature-RTMPose actually learns during training, why it stays in 2D, and how to evaluate it honestly on a handful of videos.

Related: [objects-and-context.md](objects-and-context.md) (phones, carts, scanners) · [cross-camera-reid.md](cross-camera-reid.md) (same person on other cameras) · [design-decisions.md](design-decisions.md).

## Pipeline

```
frame ─► YOLOX (find people) ─► RTMPose (17 keypoints per person)
      ─► IoU tracker (same person across frames)
      ─► 7 pose features + 8 motion features over the last 1.2 s
      ─► MultiOutput RandomForest ─► probability per activity
```

| Step | What happens | Code |
|---|---|---|
| Person detection | YOLOX (`*_humanart` weights from `rtmlib.Body`) finds boxes of one class: person. Nothing else exists for the pipeline | [classifier.py](../Feature-RTMPose/backend/services/classifier.py) `_load_pose_model` |
| Skeleton | RTMPose places 17 COCO keypoints as `(x, y)` **pixels plus a confidence** — 2D, no depth. Persons with < 4 visible points are dropped; of two overlapping detections (keypoint boxes, IoU > 0.5) the less confident one is dropped | `_detect_all_poses` |
| Tracking | Greedy IoU matching of boxes between consecutive frames. A track unseen for > 1 s is dropped; a seek resets the tracker | [motion.py](../Feature-RTMPose/backend/services/motion.py) `PoseTracker` |
| Pose features (7) | One frame, coordinates normalized by image size: `torso_angle`, `arm_spread`, `vertical_movement`, `hand_height`, `stride_length`, `body_speed`, `symmetry` | `_features_from_landmarks` |
| Motion features (8) | Over the track's last 1.2 s, in torso lengths around the body center | `motion_features` |
| Classification | `MultiOutputClassifier(RandomForest, 100 trees)`: an independent yes/no per activity, so several can fire at once (`carrying` + `walking`) | `_fit_and_save`, `_classify_person` |

Things that are easy to get wrong:

- **`body_speed` is not speed.** It is `1 − mean confidence` of shoulders and hips. Real speed is the motion feature `body_velocity`.
- **`person_id` ≠ `track_id`.** `person_id` is left-to-right order in the current frame (what the labeling UI's `#N` means); `track_id` follows a person across frames.
- **`person = all`** in a label means every person in the frame becomes a sample; `#1`, `#2`, … select one person by left-to-right order.
- **Older models keep working.** Pose features come first, and a model is fed only as many features as it was trained on (7 for pre-motion models). When appending to a dataset, old 7-value rows are dropped.

## Motion features

Normalization (`body_frame`): center = mid-hip (falls back to mid-shoulder, then the center of the box around the visible keypoints — hips are often hidden from above); scale = shoulder-to-hip length (falls back to that box's height / 3). No rotation. The five window features (`body_velocity`, `wrist_motion`, `gait_rhythm`, `hands_ahead`, `wrist_rise`) need ≥ 3 frames spanning ≥ 0.3 s and yield 0 otherwise; `hands_ahead` also needs `body_velocity` > 0.3. The other three (`wrist_near_head`, `head_down`, `nearest_person`) use the current frame only; when the needed points are missing, `wrist_near_head` defaults to 3.0 and `nearest_person` to 10.0 (far away), `head_down` to 0.

| Feature | Meaning | Targets |
|---|---|---|
| `body_velocity` | body-center displacement over the window, torso lengths / s | walking vs standing |
| `wrist_motion` | mean speed of the hands relative to the body | handling, scanning |
| `gait_rhythm` | spread of the ankle-to-ankle distance (steps oscillate) | walking |
| `hands_ahead` | hands in front (+) or behind (−) along the walking direction; only when moving | pushing vs pulling a cart |
| `wrist_near_head` | closest wrist-to-head distance | phone at the ear |
| `head_down` | nose relative to shoulders | reading, scanning |
| `wrist_rise` | net upward hand movement over the window | lifting |
| `nearest_person` | distance to the closest other person | conversation |

Not implemented: joint angles, per-window min/max statistics, temporal smoothing of the output (each frame is classified independently, so labels can flicker; a median filter or "hold for N windows" rule is the usual fix), stronger trackers (ByteTrack, OC-SORT add Kalman prediction and survive occlusion better).

Classifier options:

| Approach | Data needed | Notes |
|---|---|---|
| Window features + RandomForest / gradient boosting (current) | tens of samples per class | interpretable, cheap |
| Network on skeleton sequences: ST-GCN, PoseC3D (MMAction2, Apache-2.0) | hundreds–thousands of segments; ~50–200 when fine-tuning | learns features itself; overkill for minutes of video |

## 2D vs 3D

| Option | How | Verdict |
|---|---|---|
| **2D (current)** | one camera, RTMPose | works |
| 3D from one camera | lift 2D→3D (VideoPose3D, MotionBERT) | depth is *guessed*; large errors on overhead views, since these models are trained mostly at eye level |
| 3D from several cameras | triangulate the same point from calibrated cameras | needs synchronized recordings with known camera poses |

Stay in 2D: normalized 2D trajectories are enough for walk / stand / carry / push / pull. 3D pays off when true geometry matters (ergonomics, back angle while lifting).

Viewpoint still matters in 2D. From above the torso is foreshortened and legs are hidden by the body or racks, so `stride_length` and `gait_rhythm` are weaker. Someone walking straight toward an eye-level camera barely moves in 2D, so `body_velocity` stays near zero. Measure this per camera. Don't assume it.

## Evaluating honestly: split by clip

Adjacent frames are near-duplicates. Shuffle all frames and hold out 20%, and frame 101 lands in the test set while frames 100 and 102 are in training. The model recognizes an almost identical frame and scores near-perfectly. That says nothing about a new video.

Split levels, weakest to strongest:

1. **By clip**: train on some clips, test on others.
2. **By person**: test on people the model has never seen.
3. **By camera / viewpoint**: test on a camera absent from training.

**Leave-one-clip-out** ([eval_leave_one_clip_out.py](../Feature-RTMPose/scripts/eval_leave_one_clip_out.py)): hold out one whole source video, train the same model as the Training tab on the rest, score the held-out clip, rotate through all clips. Per-label TP/FP/FN are summed over folds; the result is macro F1 over testable labels. `--features 7` scores pose-only for comparison.

- A label present **only** in the held-out clip is reported as *untestable*, not counted as a miss. False positives it causes on other clips still count.
- An activity that appears in a single clip can never be tested this way. Show it in a demo, but don't quote an accuracy for it.
- If the same people appear in most clips, a by-clip split is not a by-person split. State that next to any number.

Rough data needs (rules of thumb, not measured here): RandomForest on window features needs ~20–50 segments per class from **several** clips and **several** people.

Measured results on the demo clips: [README](../README.md#demo-results).

## Activities by signal

What each signal should be able to carry, in principle. The measured results above show how far the demo data is from that: `standing` scored 0.00 with pose + motion, and `carrying` and `talking` could not be tested at all.

| Activity | Pose only | Pose + motion | Needs objects / context |
|---|---|---|---|
| Walking | partial | yes | — |
| Standing | confused with walking | yes | — |
| Carrying a box | yes (hands at chest) | yes | — |
| Lifting | partial | yes (`wrist_rise`) | — |
| Pulling a cart / pallet jack | no | partial (`hands_ahead` < 0) | cart detector helps |
| Pushing a cart | no | partial (`hands_ahead` > 0) | cart detector helps |
| Reading a tablet / paperwork | partial | partial (`head_down`) | tablet detector |
| Talking to others | no | yes, from pairs of tracks (`nearest_person`) | — |
| On the phone | no | partial (`wrist_near_head`) | phone detector |
| Scanning | no | partial | scanner detector, rack zone |
| Packing | no | partial | packing-table zone |

How a person-only model can learn about phones and carts: [objects-and-context.md](objects-and-context.md).
