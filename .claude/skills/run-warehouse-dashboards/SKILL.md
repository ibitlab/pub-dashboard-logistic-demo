---
name: run-warehouse-dashboards
description: Run, drive, screenshot, benchmark or record GIFs of the Warehouse Activity Intelligence dashboards (Feature-MediaPipe on :8000, Feature-RTMPose on :8001). Use when asked to start the app, take screenshots of a tab, capture pose-detection frames, measure detection rate or latency, or switch pose models.
---

# Running the Warehouse Activity Intelligence dashboards

Two FastAPI + vanilla-JS dashboards that classify warehouse worker activity from
video. They ship the **same frontend shell** (same tabs, same element IDs), so one
driver handles both — `--app` picks the port and whether the pose stage is
multi-person.

| App | Port | Pose stage |
|---|---|---|
| `Feature-MediaPipe` | 8000 | MediaPipe, 33 landmarks, **single person only** |
| `Feature-RTMPose` | 8001 | YOLOX + RTMPose, 17 COCO keypoints, **all people**, plus OSNet ReID |

Agent path is **`driver.mjs`** (Playwright over system Chrome) for anything visual,
and **`bench.py`** (stdlib HTTP) for measurement. All paths below are relative to
the repo root.

Verified on macOS (Apple silicon), Python 3.14, Node 20. Not tried on Linux;
`run.sh` is bash and should port, but nothing here claims it.

## Prerequisites

```bash
python3 --version          # 3.10+; 3.14 works
node --version             # 20+
ffmpeg -version | head -1  # only for frame extraction / GIFs
ls -d "/Applications/Google Chrome.app"   # the driver uses system Chrome
```

Install the driver's one dependency (Playwright the library; the browser is your
installed Chrome):

```bash
npm install --prefix .claude/skills/run-warehouse-dashboards
```

## Start the servers

Each app builds its own venv on first run and then serves with `--reload`.

```bash
cd Feature-RTMPose   && bash run.sh 8001    # ~150 MB model download on first use
cd Feature-MediaPipe && bash run.sh 8000    # downloads missing pose .task models (~45 MiB) on first run
```

Run them in the background; first start takes minutes while pip installs. Wait for
health before driving anything:

```bash
curl -s http://localhost:8000/api/health && curl -s http://localhost:8001/api/health
```

**If `run.sh` dies with `pip: command not found` (exit 127), the venv is stale** —
see Gotchas.

## Run: driver.mjs (agent path)

Four subcommands. Everything writes PNGs into `--out`.

```bash
# Tab screenshots: Live Analysis (streaming), Pose Preview, Events Log
# --video: a file in data/demo/ (default: first one listed); --project: select a Training project by name
node .claude/skills/run-warehouse-dashboards/driver.mjs shots --app rtmpose --out /tmp/shots \
  --video <clip.mp4> --project "<project name>"

# Annotate tab: fresh project from the demo video + auto-detected keypoints
node .claude/skills/run-warehouse-dashboards/driver.mjs annotate --app mediapipe --out /tmp/shots

# Workers/ReID gallery panel, scroll box expanded so all identities fit (rtmpose only)
node .claude/skills/run-warehouse-dashboards/driver.mjs reid --out /tmp/shots --collection "<collection name>"

# Per-frame pose captures, cropped to the video content — the input for a GIF
node .claude/skills/run-warehouse-dashboards/driver.mjs preview --app rtmpose \
  --from 463 --to 610 --step 4 --out /tmp/frames
```

`preview` prints a line per frame and writes `meta.json` with the per-frame person
count and ReID names, so you can assert on detection quality:

```
  f=463 persons=3 3 PERSONS [Left,Center,Right]
  f=471 persons=3 3 PERSONS [Left,Right,Center]
```

`shots` ends with a JSON summary (`connection`, current activity, persons found).
A `Disconnected` connection or `persons: 0` means the run is not worth keeping.

### Build a GIF from `preview` frames

Verified exactly as written; the crop drops the empty ceiling/floor:

```bash
ffmpeg -y -loglevel error -framerate 10 -i "/tmp/frames/f%03d.png" \
  -vf "crop=iw:ih*0.74:0:ih*0.18,scale=280:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64:stats_mode=full[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  -loop 0 /tmp/out.gif
```

**Do not go below 48 colors.** At 32 the per-person skeleton colors — the entire
point of the RTMPose GIFs — collapse into grey. Cropping saves more bytes than
dropping colors does. ~37 frames at 280px lands around 950 KB.

## Run: bench.py (measurement, no browser)

```bash
python3 .claude/skills/run-warehouse-dashboards/bench.py --app rtmpose --every 5
python3 .claude/skills/run-warehouse-dashboards/bench.py --app mediapipe --all-models --out bench.json
```

Reports detection rate, persons/frame, server-side latency (median + p90) and the
predicted-activity histogram. `--video` defaults to the first clip in the app's
`data/demo/`. Each frame is posted with its `video_time`, so RTMPose's motion
features measure speed in video seconds, not in request round-trips. `--all-models` loops every model size, **verifies each
switch took effect**, and restores the original in a `finally`.

Rough expectations on an Apple-silicon CPU: MediaPipe lite ~10–20 ms/frame,
RTMPose balanced ~300–400 ms/frame including ReID and the classifier (it grows
with the number of people). Results for the demo clips are in the root README.

⚠️ **`mean_confidence` is not accuracy.** bench.py has no ground-truth labels. For
accuracy use `Feature-RTMPose/scripts/eval_leave_one_clip_out.py` on a project's
`dataset.json` (MediaPipe writes one to `data/training/dataset.json` too).

## Gotchas

**The model-switch endpoints take `Form(...)`, not JSON.** This is the one that
silently produces fake benchmarks. Posting JSON returns 422 and the server keeps
serving the *previous* model, so "three models measured" becomes one model measured
three times with suspiciously identical latency.

| Endpoint | App | Field |
|---|---|---|
| `/api/models/switch` | MediaPipe | `filename=pose_landmarker_lite.task` |
| `/api/pose-model/switch` | RTMPose | `mode=balanced` |
| `/api/train/switch-version` | RTMPose | `version_id=<YYYYMMDD_hhmmss>` |

```bash
curl -X POST http://localhost:8001/api/pose-model/switch -F "mode=lightweight"   # works
curl -X POST http://localhost:8001/api/pose-model/switch -d '{"mode":"x"}'       # 422, model unchanged
```

Always read the state back (`/api/train/status` → `pose_mode`, or `/api/models` →
`active`). `bench.py` does this and aborts on mismatch.

**Never quote a random frame split as accuracy.** Adjacent frames are
near-duplicates: on the demo dataset a random 80/20 split scores macro F1 ≈ 0.94
while holding out whole clips scores ≈ 0.2. Use
`Feature-RTMPose/scripts/eval_leave_one_clip_out.py <project>/dataset.json`.

**Motion features need consecutive frames.** RTMPose tracks people across calls
to `/api/predict/landmarks` using the `video_time` form field; a seek (time jump
backwards or > 1 s forwards) resets the tracker, so the first ~1 s after a seek
has no motion features. A `preview` capture with a large `--step` degrades the
same way.

**ReID export runs in a throwaway venv.** `torchreid` pulls in `scipy`, `cv2`,
`matplotlib`, `h5py`, `yacs`; the pip line in the RTMPose README lists them all.
Install them outside the app's venv. The export loads **ImageNet** weights (the
log says so), not ReID-trained ones.

**Stale venvs after moving the repo.** venvs are not relocatable: after a move,
`pip` inside them is a dangling symlink and `run.sh` exits 127. Fix:

```bash
mv Feature-RTMPose/venv Feature-RTMPose/venv.stale && bash Feature-RTMPose/run.sh 8001
```

Saved annotation projects store **absolute** `video_path`s, so a project created
before a move cannot load frames. That is why `driver.mjs annotate` always creates a
fresh project.

**Playwright's bundled Chromium may lack H.264**, and the demo clips are H.264. The
driver uses `channel: 'chrome'` against installed Google Chrome, which avoids both
the codec problem and browser-build mismatches. Do not "fix" this by running
`npx playwright install`.

**Screenshot the app in a *running* state.** An idle Live Analysis tab is all zeros
and a black video box. `driver.mjs shots` selects a demo video and waits ~18 s so
KPIs, scores and the timeline actually populate.

**Layout differs between the two apps.** MediaPipe puts Pose Preview in a full-width
panel *below the fold*; RTMPose has it in a side panel. The driver calls
`scrollIntoView` either way — without it the MediaPipe shot is cut off.

**Demo video is portrait (720×1280) in a wide dark box.** A naive element screenshot
is ~60% black. `driver.mjs` computes the letterboxed content rect from
`videoWidth/videoHeight` before clipping.

**Seek the frame slider with both events.** `input` fires on drag, `change` on
release; the app listens for both. After seeking, wait on the
`/api/predict/landmarks` response rather than sleeping — sleeping is flaky and ~3×
slower.

**ReID identities are unstable on identical uniforms.** Over 37 frames of the demo,
only 16 resolved to three distinct names and 18 assigned one name to two or more
people. Expect label churn in captures; it is the model, not the driver.

**zsh eats `$var[p]`** as an array subscript. If you parametrise the GIF filter
chain (`[p]` labels after a variable), put the ffmpeg command in a file rather than
inlining it.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `run.sh` exits 127, `pip: command not found` | Stale venv (repo was moved). `mv Feature-*/venv Feature-*/venv.stale` and re-run. |
| `browserType.launch: Executable doesn't exist ... chromium_headless_shell-1243` | Driver already avoids this via `channel: 'chrome'`. If you wrote your own script, do the same. |
| Benchmark shows identical latency across models | The switch never happened — JSON instead of form fields. See Gotchas. |
| All `/api/reid/thumb/...` return 500 | `main.py` imported a `THUMBS_DIR` that no longer exists in `services/reid.py`; use `classifier.reid.thumbs_dir`. Fixed, but the shape recurs when `reid.py` is refactored. |
| Annotate tab blank / frames won't load | Saved project holds a dead absolute `video_path`. Create a fresh project from a demo video. |
| Pose Preview shows `NO PERSON` on MediaPipe | Expected when only legs are visible — a Stage-1 limit no retraining fixes. Pick a frame with torso/face visible. |
| `sklearn InconsistentVersionWarning` at startup | A `.pkl` classifier was pickled under an older scikit-learn. Warning only; models load. Retrain to get rid of it. |
