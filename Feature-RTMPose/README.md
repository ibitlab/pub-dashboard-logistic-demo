# Feature-RTMPose

Multi-person activity recognition from video. RTMPose finds every person, a per-person tracker adds motion features, and a multi-label RandomForest that you train on your own labeled clips names the activities. Optional OSNet-AIN ReID attaches an enrolled name to each person. Everything runs on CPU through ONNX Runtime. PyTorch is needed only for the one-time ReID export.

Measured results on the demo clips: [../README.md#demo-results](../README.md#demo-results)

## Run

```bash
bash run.sh          # http://localhost:8001, API docs at /docs
bash run.sh 8002     # another port
```

- `run.sh` creates `venv/`, installs `backend/requirements.txt` and starts uvicorn with `--reload`.
- The server loads the `balanced` pose model at startup, so the first start downloads it (RTMPose-m + YOLOX-m, ~150 MB) into the rtmlib cache `~/.cache/rtmlib/`. The other modes download the first time you switch to them.
- Videos go in `data/demo/` (the server creates the folder at startup). H.264 MP4 seeks reliably in the browser. Re-encode anything else: `ffmpeg -i in.mp4 -c:v libx264 -movflags +faststart out.mp4`.
- `data/` is git-ignored and holds everything the app writes.

| Pose mode (header dropdown) | Detector + pose model | Note |
|---|---|---|
| Lightweight | YOLOX-tiny + RTMPose-s | fastest |
| Balanced (default) | YOLOX-m + RTMPose-m | |
| Performance | YOLOX-x + RTMPose-x | not real-time on CPU |

## Tabs

| Tab | What it does |
|---|---|
| Live Analysis | Webcam, a `data/demo/` clip or a local file (played in the browser, not uploaded), streamed over `/ws/live`. Shows KPI cards, scores and a timeline. It classifies **one person per frame** (the leftmost) and shows no ReID. |
| Training | Projects, interval labeling, training, model versions, the Workers (ReID) panel, and Pose Preview. Pose Preview shows every person with skeleton, `#N`, identity, activities and all 15 features. |
| Annotate | Correct the 17 COCO keypoints frame by frame, then export COCO JSON (MMPose-ready) or the annotated frames as JPEGs. Standalone: training does not use it. |
| Events Log | Activity-change events from SQLite, with CSV/JSON export. |

## Training

The classifier starts empty: no bundled model, no synthetic data. Until you train a model, every person comes back as activity `untrained` with no scores. Pose detection and features still work.

1. **Project.** Pick or create a training project (`+ New`). Each project keeps its own labels, `dataset.json`, active model and last 5 versions under `data/projects/<project_id>/`. The first training records the project's pose mode and the panel shows it as locked, because features from different pose models do not mix. The lock is informational: the header dropdown still switches models, so keep the mode yourself. `-- No Project (Global) --` uses `data/models/` and does not save labels.
2. **Video.** *Demo* lists `data/demo/`. *Upload* copies your file to the server (`POST /api/demo/upload`) and selects it. It never overwrites: a name clash gets a numeric suffix. Training reads the video on the server, so a clip that is only open in the browser cannot be trained on.
3. **Label intervals.** Scrub the video and mark the interval with `[` (start) and `]` (end). Tick one or more activities, or type a new one. Then pick the person: `all` means every detected person, and `#N` means the N-th person from the left **in each sampled frame**. Ticking several activities makes a multi-label sample (`carrying+walking`). For intervals longer than 10 slider steps (the slider assumes 30 fps), the server checks for changes in person count or position jumps. If it finds any, it splits the interval into stable sub-ranges.
4. **Add & Train.** Samples every 3rd frame of each interval, after 1.2 s of tracker warm-up before it. Extracts 15 features per chosen person, appends them to the project's `dataset.json`, and fits `MultiOutputClassifier(RandomForest, 100 trees)` on the whole dataset. Inside a project, labels are saved per video and reload when you reopen that video. Each Add & Train sends all labels of the current video and replaces that video's earlier samples in the dataset, so re-training a clip never duplicates it.
5. **Dataset & Model panel.** Sample counts per activity and per source clip, removing a source clip, *Rebuild* (retrain on the stored dataset), switching or deleting versions, and *Ensemble* (averages all saved versions).

Features (details in [../docs/how-activity-recognition-works.md](../docs/how-activity-recognition-works.md)):

- **7 pose features from one frame:** `torso_angle`, `arm_spread`, `vertical_movement`, `hand_height`, `stride_length`, `body_speed`, `symmetry`. Despite its name, `body_speed` is 1 minus the mean visibility of the shoulder and hip keypoints.
- **8 motion features** over the last 1.2 s of the same tracked person ([backend/services/motion.py](backend/services/motion.py)): `body_velocity`, `wrist_motion`, `gait_rhythm`, `hands_ahead`, `wrist_near_head`, `head_down`, `wrist_rise`, `nearest_person`. Distances are in torso lengths. In Pose Preview and training, times are video seconds (the `video_time` form field), and a seek (backwards, or more than 1 s forwards) resets the tracker, so the first ~1 s after a seek has no motion history. Live Analysis sends no `video_time`, so it uses the server clock, and any gap over 1 s between processed frames resets the tracker too. Three of the eight (`wrist_near_head`, `head_down`, `nearest_person`) are computed from the current frame only.
- **Older 7-feature models still load:** each model gets the first `n_features_in_` values. Appending new samples drops old 7-feature rows from the dataset.

## Evaluation

Don't evaluate on a random split of frames. Neighbouring frames are near-duplicates, so test frames leak into training. Hold out whole clips instead (run from `Feature-RTMPose/`):

```bash
venv/bin/python scripts/eval_leave_one_clip_out.py data/projects/<project_id>/dataset.json --out result.json
venv/bin/python scripts/eval_leave_one_clip_out.py data/projects/<project_id>/dataset.json --features 7   # pose only
```

For each label it prints precision, recall and F1, and lists the clips where the label is untestable (a label that appears in only one clip can't be scored when that clip is held out). It ends with macro F1 over the testable labels. It reads any `dataset.json` that has `features`, `label_sets` and `metadata[].source`, including Feature-MediaPipe's.

## Person re-identification (optional)

OSNet-AIN turns each person crop into a 512-d embedding. Each frame is matched on its own against the active gallery: cosine similarity, best of up to 10 embeddings per person, threshold 0.65, no tracking. Background, and what cross-camera matching would need: [../docs/cross-camera-reid.md](../docs/cross-camera-reid.md).

**Export the model once**, in a throwaway venv (torch is not an app dependency):

```bash
python3 -m venv /tmp/reid-export && source /tmp/reid-export/bin/activate
pip install torch torchvision torchreid onnx onnxscript onnxruntime Pillow tensorboard \
            scipy opencv-python-headless matplotlib h5py yacs   # torchreid also imports the second line
python scripts/export_reid_model.py        # from Feature-RTMPose/
deactivate && rm -rf /tmp/reid-export
```

The export writes `data/models/osnet_ain_x1_0.onnx` and `osnet_ain_x1_0.onnx.data` (~9 MB; keep the two files together). Restart the server to load them. The weights are ImageNet weights (`pretrained=True`), not ReID-trained ones. ReID-trained OSNet weights exist, but check their dataset licenses before using them.

**Enroll people.** The easiest way is the Workers panel in the Training tab (`+`, `+ Photo`). Its buttons write to the *active* gallery: the collection linked to the active project, or the global one. The collection selector only changes which gallery is listed. Over HTTP:

```bash
curl -X POST localhost:8001/api/workers/collections -F "name=<collection name>"        # returns its id
curl -X POST localhost:8001/api/workers/collections/<collection_id>/enroll-from-frame \
     -F "file=@frame.jpg" -F "name=<person label>" -F "person_index=0"    # 0-based, left to right
curl -X POST localhost:8001/api/workers/collections/<collection_id>/enroll \
     -F "file=@crop.jpg" -F "name=<person label>"                         # pre-cropped person
curl -X PATCH localhost:8001/api/projects/<project_id> -H "Content-Type: application/json" \
     -d '{"collection_id": "<collection_id>"}'
curl -X POST localhost:8001/api/projects/<project_id>/activate
```

Enroll each person from every viewpoint you expect to see them in. Use neutral labels, not real names. Collections are stored in `data/workers/<collection_id>/` (`reid_gallery.json`, `reid_thumbs/`). The `/api/reid/*` endpoints act on the active gallery, which is the global one in `data/models/` when no project is active.

## API

The full list is at http://localhost:8001/docs. `run.sh` binds to `0.0.0.0`, so the API is reachable from the network; there is no authentication.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/predict/landmarks` | POST | Frame → all persons with landmarks, activities, `track_id`, identity. Form: `file`, optional `video_time` (s) |
| `/api/predict` | POST | Frame → activity of the primary (leftmost) person |
| `/ws/live`, `/ws/dashboard` | WS | Live frame stream; KPI push every 2 s |
| `/api/pose-model/switch` | POST | Form `mode`: `lightweight` / `balanced` / `performance` |
| `/api/demo/videos`, `/api/demo/upload` | GET / POST | List `data/demo/`; store an uploaded clip there |
| `/api/projects`, `/api/projects/{id}` | GET, POST / GET, PATCH, DELETE | Training projects; PATCH sets `name`, `collection_id` or `pose_mode` (`null` is ignored, so a link cannot be cleared this way) |
| `/api/projects/{id}/activate`, `/api/projects/deactivate` | POST | Load a project's model, dataset and collection |
| `/api/projects/{id}/labels/{video}` | GET / POST | Saved interval labels per video |
| `/api/train/from-labels` | POST | JSON `video_url`, `labels[{start_time, end_time, activities, person}]` → extract + train |
| `/api/train/analyze-label-stability` | POST | Auto-split check for one interval (person count, position jumps) |
| `/api/train/retrain`, `/api/train/status`, `/api/train/progress` | POST / GET / GET | Rebuild; model info; extraction progress |
| `/api/train/dataset`, `/api/train/dataset/source/{name}` | GET, DELETE / DELETE | Dataset breakdown or clear; drop one source clip |
| `/api/train/switch-version`, `/api/train/versions/{id}`, `/api/train/ensemble` | POST / DELETE / POST | Model versions and ensemble mode |
| `/api/workers/collections`, `/api/workers/collections/{id}` | GET, POST / GET, DELETE | ReID collections |
| `/api/workers/collections/{id}/enroll`, `…/enroll-from-frame`, `…/gallery/{name}` | POST / POST / DELETE | Enroll, remove (`GET …/gallery` lists) |
| `/api/reid/status`, `/api/reid/gallery`, `/api/reid/enroll`, `…/enroll-from-frame`, `…/gallery/{name}` | GET / GET / POST / POST / DELETE | Same as the collection endpoints, on the active gallery |
| `/api/health`, `/api/activities`, `/api/metrics/timeline` | GET | Health check; trained classes; KPI history |
| `/api/annotations/projects/…` | various | Annotate tab: frames, detect, save, `export` (COCO JSON), `export-frames` |
| `/api/events`, `/api/metrics` | GET | Event log, KPIs |

## Layout

```
backend/main.py        FastAPI app, all endpoints
backend/services/      classifier.py (pose, features, RandomForest), motion.py (tracker, motion features),
                       reid.py, workers.py (collections), projects.py, database.py, kpi_engine.py
frontend/              index.html, app.js, annotate.js, styles.css (vanilla JS)
scripts/               eval_leave_one_clip_out.py, export_reid_model.py
data/                  git-ignored: demo/, projects/<project_id>/, workers/<collection_id>/, models/, annotations/, warehouse.db
```

## Limitations

- The classifier needs several clips per activity, filmed from the views you deploy. With one clip per activity it does not generalize to new clips. Judge it only by leave-one-clip-out.
- `#N` labels are positional in each frame. If people swap left-to-right order inside an interval, samples go to the wrong person. Keep `#N` intervals short, or rely on the auto-split.
- ReID with ImageNet weights matches only within a similar viewpoint and confuses people in similar clothing. Frames are matched one at a time, so two people can get the same name. Treat names as hints.
- Live Analysis classifies only the leftmost person and shows no identities. Use Pose Preview to see every person.
- Objects (boxes, carts, scanners) are not modelled; see [../docs/objects-and-context.md](../docs/objects-and-context.md). Design rationale: [../docs/design-decisions.md](../docs/design-decisions.md).
