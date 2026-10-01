**English** | [Українська](README.uk.md)

# Warehouse Activity Intelligence

**Can a laptop tell what warehouse workers are doing from ordinary camera video — no GPU, no cloud?**

Pose estimation → a small activity classifier → worker re-identification, built twice (MediaPipe and RTMPose) and measured on held-out clips.

## Background

This started at the Nova Poshta × DEV Challenge hackathon (March 2026), where a small team built a prototype for analysing activity at a parcel terminal. Hackathon rules don't allow publishing that code or footage, so this repository is an independent rebuild. It uses different design choices to try other approaches to recognising human movement and activity. It shares no code, data or video with the hackathon entry and is not affiliated with Nova Poshta or DEV Challenge.

## What works, what doesn't

| | Status | Why |
|---|---|---|
| Skeletons for several people per frame | ✅ RTMPose: 57–100% of frames, most camera angles | top-down detector + per-person pose |
| Label → train → live loop in the browser | ✅ both apps | interval labels (RTMPose) or whole-clip labels (MediaPipe) |
| Recognising activities on a clip the model never saw | ❌ macro F1 0.24 | too few clips per activity; 2D poses only |
| Recognising a worker from a new camera angle | ❌ 0 of 134 detections | ReID uses ImageNet weights; every frame matched on its own |
| Real-time on CPU | ⚠️ MediaPipe 11–21 ms/frame, RTMPose ~300–415 ms | RTMPose runs pose + ReID per person |

## Demo results

This is the only page with demo data; the apps and the other docs work with any video you put into `data/demo/`.

**The clips.** Realistic CCTV footage that may be published is hard to find, so the demo uses the closest stock clips available: nine [Pexels](https://www.pexels.com/license/) videos, renamed as six fixed warehouse cameras. They are staged, not surveillance footage, and the camera setup is simulated. Seven show the same four workers from above and at eye level. The clips are not in the repo; the GIFs and screenshots are modified derivatives. Workers are labelled neutrally (`Worker A`–`D`).

| Camera | Pexels clip | View | Content |
|---|---|---|---|
| cam-01 (3 clips) | [4284181](https://www.pexels.com/video/4284181/), [4280746](https://www.pexels.com/video/4280746/), [4280748](https://www.pexels.com/video/4280748/) | overhead | crew walks, stops, talks |
| cam-02 | [4281405](https://www.pexels.com/video/4281405/) | overhead | three walk together |
| cam-03 | [4284182](https://www.pexels.com/video/4284182/) | near top-down | pallet jack, picking |
| cam-04 (2 clips) | [4293960](https://www.pexels.com/video/4293960/), [4292902](https://www.pexels.com/video/4292902/) | eye level | carrying boxes; scanning |
| cam-05 | [6170059](https://www.pexels.com/video/6170059/) | overhead, outdoors | another person, hand truck |
| cam-06 | [13422071](https://www.pexels.com/video/13422071/) | side, mostly legs | another person, floor scrubber |

**Same clips, same frames.** Frames where an engine found nobody are kept.

| | MediaPipe (lite) | RTMPose (balanced) |
|---|---|---|
| cam-01, overhead | <img src="docs/mediapipe-cam01-overhead.gif" width="360" alt="MediaPipe overhead: one skeleton at most, some frames empty"> | <img src="docs/rtmpose-cam01-overhead.gif" width="360" alt="RTMPose overhead: three named skeletons"> |
| cam-04, eye level | <img src="docs/mediapipe-cam04-floor-level.gif" width="360" alt="MediaPipe eye level: one of two workers"> | <img src="docs/rtmpose-cam04-floor-level.gif" width="360" alt="RTMPose eye level: both workers, no names"> |

Workers were enrolled from cam-01, so names appear there and not at eye level. Activity labels in the GIFs come from a model that saw these clips in training: they show the pipeline, not accuracy.

### Pose detection (every 5th frame, CPU, Apple silicon)

The MediaPipe and RTMPose columns give the share of frames with at least one person found. People leave some clips, so compare the engines, not against 100%. The persons average counts only frames where someone was found.

| Clip | View | MediaPipe | RTMPose | RTMPose persons, avg (max) |
|---|---|---|---|---|
| cam-01 #1 / #2 / #3 | overhead | 77% / **7%** / 63% | 98% / 80% / 88% | 2.7 (3) / 2.6 (3) / 1.5 (3) |
| cam-02 | overhead | 53% | 77% | 2.6 (3) |
| cam-03 | near top-down | **16%** | 92% | 1.8 (3) |
| cam-04 #1 / #2 | eye level | 81% / 66% | 91% / 87% | 2.0 (3) / 1.0 (2) |
| cam-05 | outdoors | 54% | 100% | 1.0 (2) |
| cam-06 | side | 35% | 57% | 1.4 (**4**, at most 2 real) |

MediaPipe returns one person per frame by design and loses people seen from above. RTMPose also makes mistakes: it reports four people on cam-06, where there are at most two.

### Activity classification (leave-one-clip-out)

Train on all clips but one, test on that one, repeat for every clip. No synthetic data in either app.

| | RTMPose: 9 clips, interval labels, 15 features | MediaPipe: 7 clips, whole-clip labels, 7 features |
|---|---|---|
| Random 80/20 frame split (**invalid**: neighbouring frames leak) | 0.94 | 0.77 |
| Leave-one-clip-out, macro F1 | **0.24** | **0.17** |
| `walking` F1 | 0.72 (fails when people walk towards the camera) | 0.34 |
| `standing`, `push-cart` | 0.00 | — / 0.00 |
| untestable (in one clip only) | carrying, picking, pull-cart, scanning, talking | carrying, scanning |

Motion features (8 values over 1.2 s per tracked person) did not help here: `walking` scored 0.72 with them and 0.76 without. The bottleneck is too few clips per activity, not the model.

### Worker re-identification

Gallery: 4 workers, 8 crops, all from cam-01.

| Test | Result |
|---|---|
| Other overhead clips | names assigned, but 8–12 frames per clip give one name to two people |
| Eye level (cam-04), same workers | **0 / 134** and 0 / 81 detections recognised (one false name) |
| Strangers (cam-05, cam-06) | 7 / 140 wrongly named |

Raw output: [benchmark-cameras.json](docs/benchmark-cameras.json), [eval-leave-one-clip-out.json](docs/eval-leave-one-clip-out.json), [reid-cross-camera.json](docs/reid-cross-camera.json). The pose-detection numbers are reproduced with [bench.py](.claude/skills/run-warehouse-dashboards/bench.py) against a running app (`python3 .claude/skills/run-warehouse-dashboards/bench.py --app rtmpose --every 5 --video <clip>`); the F1 numbers with the leave-one-clip-out script described in the RTMPose README.

## Two implementations

| | [Feature-RTMPose](Feature-RTMPose/) (recommended) | [Feature-MediaPipe](Feature-MediaPipe/) |
|---|---|---|
| Pose model | RTMPose + YOLOX, 17 COCO keypoints | MediaPipe PoseLandmarker, 33 landmarks |
| People per frame | all | one |
| Classifier input | 7 pose + 8 motion features, multi-label | 7 pose features, one label |
| Training labels | time intervals per person, projects | one label per uploaded clip |
| Worker ReID | OSNet-AIN, per-project worker collections | — |
| Port | 8001 | 8000 |

Both have the same four tabs: **Live Analysis**, **Training**, **Annotate** (correct keypoints and export them as COCO) and **Events Log**.

![RTMPose Pose Preview with three workers and their features](docs/rtmpose-pose-preview.png)

## Quick start

Python 3.10–3.12, macOS or Linux. The first start creates a venv and downloads the pose models. Run each app from its own folder:

```bash
cd Feature-RTMPose && bash run.sh      # http://localhost:8001
```

```bash
cd Feature-MediaPipe && bash run.sh    # http://localhost:8000
```

Put H.264 MP4 clips into `data/demo/` of either app, or upload them in the Training tab. Setup, training and the API are in each app's README.

## Docs

- [How activity recognition works](docs/how-activity-recognition-works.md): features, 2D vs 3D, honest evaluation
- [Objects and context](docs/objects-and-context.md): phones, carts, scanners, and what a skeleton can't see
- [Cross-camera ReID](docs/cross-camera-reid.md): how workers are matched across cameras and why it fails
- [Design decisions](docs/design-decisions.md): why RTMPose and not MediaPipe or YOLO

## Licensing

Code: [MIT](LICENSE). Models: MediaPipe, RTMPose and YOLOX under Apache 2.0, OSNet-AIN under MIT. The pose models are downloaded on first run, OSNet-AIN is exported by hand (see the RTMPose README); none are shipped. scikit-learn is BSD. YOLO11-Pose was not used because of AGPL-3.0. Demo clips: [Pexels License](https://www.pexels.com/license/), used only for the derivatives in `docs/`.
