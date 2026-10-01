#!/usr/bin/env python3
"""
Benchmark the pose stage by replaying video frames through /api/predict/landmarks.

No browser needed - this talks to the FastAPI server directly.

  python3 bench.py --app rtmpose
  python3 bench.py --app mediapipe --all-models --out bench-mp.json

Reports detection rate, persons/frame, server-side latency and the distribution
of predicted activities.

IMPORTANT, and the reason this script exists rather than a pile of curl:
  * the model-switch endpoints take multipart FORM fields, not JSON. Posting
    JSON returns 422 and the server keeps serving the PREVIOUS model, so a
    benchmark "across three models" silently measures one model three times.
    This script switches, then READS THE STATE BACK and aborts if it did not
    take effect.
  * it records the originally-active model and restores it in a finally block.

What the numbers are NOT: the demo clips have no ground-truth activity labels,
so nothing here is accuracy/precision/recall. `mean_confidence` is the model's
self-reported score. Check what the active classifier was trained on before
quoting it - a model trained on one class reports 100% confidence forever.
"""
import argparse, json, mimetypes, os, statistics, subprocess, sys, tempfile, urllib.error, urllib.request, uuid

APPS = {"mediapipe": 8000, "rtmpose": 8001}


# ---------- tiny HTTP helpers (stdlib only) ----------
def _multipart(fields=None, files=None):
    b = uuid.uuid4().hex
    body = b""
    for k, v in (fields or {}).items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, path in (files or {}).items():
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            data = fh.read()
        body += (
            f'--{b}\r\nContent-Disposition: form-data; name="{k}"; '
            f'filename="{os.path.basename(path)}"\r\nContent-Type: {ctype}\r\n\r\n'
        ).encode() + data + b"\r\n"
    body += f"--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


def post(url, fields=None, files=None):
    body, ctype = _multipart(fields, files)
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


# ---------- model state ----------
def active_model(base, app):
    if app == "mediapipe":
        return next(m["filename"] for m in get(f"{base}/api/models")["models"] if m["active"])
    return get(f"{base}/api/train/status")["pose_mode"]


def switch_model(base, app, target):
    """Switch, then verify - a 422 here is silent and leaves the old model live."""
    field = "filename" if app == "mediapipe" else "mode"
    try:
        post(f"{base}/api/{'models' if app == 'mediapipe' else 'pose-model'}/switch", fields={field: target})
    except urllib.error.HTTPError as e:
        sys.exit(f"switch to {target!r} rejected ({e.code}). These endpoints take FORM fields, not JSON.")
    now = active_model(base, app)
    if now != target:
        sys.exit(f"switch did not take effect: asked for {target!r}, server reports {now!r}")
    return now


# ---------- measurement ----------
def extract_frames(video, every, dest):
    os.makedirs(dest, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
         "-vf", f"select=not(mod(n\\,{every}))", "-vsync", "0", "-q:v", "3",
         os.path.join(dest, "fr%04d.jpg")],
        check=True,
    )
    return sorted(os.path.join(dest, f) for f in os.listdir(dest) if f.endswith(".jpg"))


def video_fps(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
         "-of", "csv=p=0", video], capture_output=True, text=True, check=True).stdout.strip()
    num, _, den = out.partition("/")
    return float(num) / float(den or 1)


def measure(base, frames, label, seconds_per_frame):
    detected, persons, confs, lats, acts = 0, [], [], [], {}
    for i, f in enumerate(frames):
        # video_time lets RTMPose's tracker measure motion in video seconds
        d = post(f"{base}/api/predict/landmarks", fields={"video_time": f"{i * seconds_per_frame:.3f}"},
                 files={"file": f})
        if d.get("pose_detected"):
            detected += 1
            confs.append(d.get("confidence", 0))
            acts[d.get("activity")] = acts.get(d.get("activity"), 0) + 1
        persons.append(d.get("person_count", 1 if d.get("pose_detected") else 0))
        lats.append(d.get("latency_ms", 0))
        if (i + 1) % 25 == 0:
            print(f"    ...{i+1}/{len(frames)}", file=sys.stderr)
    n = len(frames)
    seen = [p for p in persons if p > 0]
    return {
        "label": label,
        "frames": n,
        "frames_with_pose": detected,
        "detection_rate_pct": round(100 * detected / n, 1),
        "mean_persons_when_detected": round(statistics.mean(seen), 2) if seen else 0,
        "max_persons": max(persons) if persons else 0,
        "mean_confidence_pct": round(100 * statistics.mean(confs), 1) if confs else 0,
        "median_latency_ms": round(statistics.median(lats), 1),
        "p90_latency_ms": round(sorted(lats)[max(0, int(0.9 * n) - 1)], 1),
        "activities": acts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", choices=APPS, required=True)
    ap.add_argument("--video", help="defaults to the first clip in the app's data/demo/")
    ap.add_argument("--every", type=int, default=5, help="sample every Nth frame (default 5)")
    ap.add_argument("--all-models", action="store_true", help="loop every model size, restoring the original after")
    ap.add_argument("--out", help="write combined JSON here")
    a = ap.parse_args()

    base = f"http://localhost:{APPS[a.app]}"
    get(f"{base}/api/health")  # fail fast if the server is not up

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    sub = "Feature-MediaPipe" if a.app == "mediapipe" else "Feature-RTMPose"
    demo = os.path.join(root, sub, "data", "demo")
    video = a.video or os.path.join(demo, sorted(f for f in os.listdir(demo) if f.endswith(".mp4"))[0])
    seconds_per_frame = a.every / video_fps(video)

    targets = [None]
    if a.all_models:
        targets = (
            [m["filename"] for m in get(f"{base}/api/models")["models"]]
            if a.app == "mediapipe"
            else ["lightweight", "balanced", "performance"]
        )

    original = active_model(base, a.app)
    print(f"active model at start: {original}")

    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(video, a.every, tmp)
        print(f"{len(frames)} frames from {os.path.basename(video)} (every {a.every})")
        runs = []
        try:
            for t in targets:
                if t:
                    print(f"  switching -> {t}")
                    switch_model(base, a.app, t)
                runs.append(measure(base, frames, f"{a.app} {t or original}", seconds_per_frame))
                print(json.dumps(runs[-1], indent=2))
        finally:
            if a.all_models and active_model(base, a.app) != original:
                print(f"  restoring -> {original}")
                switch_model(base, a.app, original)

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"app": a.app, "video": os.path.basename(video), "runs": runs}, fh, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
