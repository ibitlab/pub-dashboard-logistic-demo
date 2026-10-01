# Feature-MediaPipe

Single-person activity recognition from video. MediaPipe PoseLandmarker (33 keypoints) gives 7 pose features per frame, and a RandomForest trained on your own labeled clips names the activity. It is the simpler baseline next to [Feature-RTMPose](../Feature-RTMPose/README.md), which handles several people, adds motion features and ReID.

Measured results on the demo clips: [../README.md#demo-results](../README.md#demo-results)

## Run

Requires Python 3.10–3.12 (mediapipe 0.10.33 may not install on 3.13+), macOS or Linux (on Windows, use WSL2), and `curl`.

```bash
./run.sh          # http://localhost:8000, API docs at /docs
./run.sh 9000     # another port
```

`run.sh` creates `venv/` and installs `backend/requirements.txt`. It also downloads any missing `pose_landmarker_{lite,full,heavy}.task` from Google's MediaPipe model storage into `backend/models/`, which is git-ignored, so the models are not in the repository. Then it starts uvicorn with `--reload`. To download a model by hand:

```bash
curl -fsSL -o backend/models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

| Pose model (header dropdown) | Size | Use |
|---|---|---|
| Lite (default) | 5.5 MB | fastest |
| Full | 9.0 MB | middle ground |
| Heavy | 29.2 MB | most accurate on difficult poses, slowest |

Switching models takes effect on the next frame, with no restart. The `+` button uploads a custom `.task` file (`POST /api/models/upload`).

**Videos:** put `.mp4/.avi/.mov/.webm` files in `data/demo/` (the server creates the folder at startup). Use H.264 for reliable seeking: `ffmpeg -i in.mp4 -c:v libx264 -movflags +faststart out.mp4`.

## Tabs

| Tab | What it does |
|---|---|
| Live Analysis | Webcam, a `data/demo/` clip or a local file (played in the browser), streamed over `/ws/live`. Shows the activity badge, class scores, KPI cards and a timeline. |
| Training | Upload labeled clips, train, and use Pose Preview to scrub, run *Analyze Frame*, or *Play* frame by frame with an FPS counter. Pose Preview shows whether the pose is found before you train on a clip. |
| Annotate | Correct 33 keypoints (Space = auto-detect, A/D or arrow keys = previous/next frame, Ctrl/Cmd+S = save), then export COCO JSON for external tools. Standalone: training does not use it. |
| Events Log | Activity-change events with CSV/JSON export. |

## Training

The classifier starts **untrained**: no bundled model, no synthetic data. Until you train it, pose detection and the 7 features still work, but every frame with a pose comes back as activity `untrained` with empty scores (frames without a pose are `no_person`), and `/api/activities` returns `{"classes": []}`.

1. **Training** tab → **Upload File** (one or more videos; they all get the same label) or **Demo Video** (the browser fetches a `data/demo/` clip and uploads it the same way).
2. Pick a label: the suggestions are `sorting`, `carrying`, `loading`, `idle` and `walking`, or type your own. **One label covers the whole clip**, so use clips that show a single activity.
3. **Upload Sample.** The server stores a copy as `data/training/{id}_{label}.mp4`, where `{id}` is 8 hex characters. No manual file naming is needed.
4. Repeat for at least two labels, then click **Train Model**. It reads every video in `data/training/` (label taken from the file name), runs every 5th frame through MediaPipe and uses **one pose per frame**. With several people in view, MediaPipe picks one and you can't choose which. Frames without a pose are skipped. It extracts the 7 features and fits `RandomForestClassifier(n_estimators=100)` on these real samples only.
5. Output: `backend/models/activity_model.pkl` + `label_encoder.pkl`, and `data/training/dataset.json` (`features`, `label_sets`, `metadata[].source`).

Each training run starts over from all stored clips. To remove a sample, delete its file from `data/training/` and train again. The features are the same 7 as Feature-RTMPose's pose features: `torso_angle`, `arm_spread`, `vertical_movement`, `hand_height`, `stride_length`, `body_speed` (1 minus shoulder/hip visibility, not speed) and `symmetry`. They are computed from single frames, with no motion across frames. Details: [../docs/how-activity-recognition-works.md](../docs/how-activity-recognition-works.md).

## Evaluation

Adjacent frames of one clip are near-duplicates, so a random frame split overstates accuracy. Score by held-out clip instead, from the repository root:

```bash
Feature-MediaPipe/venv/bin/python Feature-RTMPose/scripts/eval_leave_one_clip_out.py \
    Feature-MediaPipe/data/training/dataset.json
```

The script trains on all clips but one, tests on the held-out clip, and rotates. With one label per clip, a label needs at least two clips before it can be scored.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/train/upload` | POST | Form `file`, `label` → `data/training/{id}_{label}.mp4` |
| `/api/train/start`, `/api/train/status` | POST / GET | Train on all stored clips; model info |
| `/api/predict`, `/api/predict/landmarks` | POST | Frame → activity (+ 33 landmarks); `no_person` / `untrained` special cases |
| `/api/models`, `/api/models/switch`, `/api/models/upload` | GET / POST / POST | List `.task` files, switch (form `filename`), add a custom one |
| `/api/demo/videos`, `/api/activities` | GET | Clips in `data/demo/`; trained classes |
| `/ws/live`, `/ws/dashboard` | WS | Live frames `{"type":"frame","data":"<base64>"}`; KPI push every 2 s |
| `/api/annotations/projects/…` | various | Annotate tab, including `export` (COCO JSON) |
| `/api/events`, `/api/metrics`, `/api/metrics/timeline`, `/api/health` | GET | Event log, KPIs, KPI history, health check |

`run.sh` binds to `0.0.0.0`, so the API is reachable from the network; there is no authentication.

## Layout

```
backend/main.py        FastAPI app, all endpoints
backend/services/      classifier.py (MediaPipe + features + RandomForest), database.py, kpi_engine.py
backend/models/        git-ignored: *.task (from run.sh), activity_model.pkl, label_encoder.pkl
frontend/              index.html, app.js, annotate.js, styles.css (vanilla JS)
data/                  git-ignored: demo/, training/, annotations/, warehouse.db
```

## Limitations

- One pose per frame: only one person is classified, and with several people in view you don't choose which one.
- If the upper body or face is not visible (legs-only frames, top-down or overhead cameras, occlusion), MediaPipe often returns no pose. Lite, Full and Heavy share this blind spot, and retraining the classifier can't fix it. See [../docs/design-decisions.md](../docs/design-decisions.md).
- There are no motion features, so activities that look alike in a single frame (walking with or without a load, push vs. pull) are hard to separate.
- Clip-level labels: a clip that mixes activities teaches the model wrong labels. For interval labels, use Feature-RTMPose.
- Objects and context are not modelled; see [../docs/objects-and-context.md](../docs/objects-and-context.md).
