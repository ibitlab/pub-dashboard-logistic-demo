"""Per-person tracking and motion features over a short time window.

The pose features in classifier.py describe how a body looks in ONE frame.
The features here describe how it MOVES, which needs the same person across
consecutive frames — so a lightweight IoU tracker links detections first.

All distances are measured in torso lengths around the body center, so the
numbers do not depend on where the person stands or how far the camera is.
Times are in seconds of video, so the features do not depend on how often
frames are sampled.
"""

import itertools
from collections import deque

import numpy as np

WINDOW_SEC = 1.2          # how much history one feature vector looks at
TRACK_MAX_GAP_SEC = 1.0   # a track unseen for longer than this is dropped
MIN_VIS = 0.3             # keypoint confidence below this counts as not visible
MIN_IOU = 0.1

MOTION_FEATURE_NAMES = [
    "body_velocity",    # how fast the body center moves (torso lengths / s)
    "wrist_motion",     # how fast the hands move relative to the body
    "gait_rhythm",      # spread of the ankle-to-ankle offset: steps oscillate
    "hands_ahead",      # hands in front (+, pushing) or behind (-, pulling) along the walking direction
    "wrist_near_head",  # closest wrist-to-head distance (phone at the ear)
    "head_down",        # nose relative to shoulders (reading, scanning)
    "wrist_rise",       # net upward movement of the hands over the window (lifting)
    "nearest_person",   # distance to the closest other person (conversation)
]

NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH = 5, 6
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_AN, R_AN = 15, 16


def _mid(kps, scs, a, b):
    """Midpoint of two keypoints, or the visible one, or None."""
    va, vb = scs[a] > MIN_VIS, scs[b] > MIN_VIS
    if va and vb:
        return (kps[a] + kps[b]) / 2
    if va:
        return kps[a].copy()
    if vb:
        return kps[b].copy()
    return None


def bbox(kps, scs):
    visible = scs > MIN_VIS
    pts = kps[visible] if np.any(visible) else kps
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def body_frame(kps, scs):
    """Body center (pixels) and scale (one torso length in pixels).

    Hips are often hidden on overhead cameras, so fall back to shoulders for the
    center and to the box height for the scale.
    """
    hip = _mid(kps, scs, L_HIP, R_HIP)
    sh = _mid(kps, scs, L_SH, R_SH)
    center = hip if hip is not None else sh
    if center is None:
        x1, y1, x2, y2 = bbox(kps, scs)
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    if hip is not None and sh is not None and np.linalg.norm(sh - hip) > 1:
        scale = float(np.linalg.norm(sh - hip))
    else:
        x1, y1, x2, y2 = bbox(kps, scs)
        scale = max((y2 - y1) / 3, 1.0)
    return np.asarray(center, dtype=float), scale


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, track_id: int):
        self.id = track_id
        self.history: deque = deque()  # (t, kps, scs)
        self.bbox = None
        self.last_t = None

    def add(self, t, kps, scs):
        self.history.append((t, kps.copy(), scs.copy()))
        while self.history and t - self.history[0][0] > WINDOW_SEC:
            self.history.popleft()
        self.bbox = bbox(kps, scs)
        self.last_t = t


class PoseTracker:
    """Greedy IoU tracker. Good enough for a few people in a fixed camera view.

    A jump backwards in time or a gap longer than TRACK_MAX_GAP_SEC (a seek in
    the video) resets it, so a person's history never mixes two moments.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)
        self.last_t = None

    def update(self, t: float, persons: list[tuple]) -> list[Track]:
        """Assign each (kps, scs) to a track. Returns tracks aligned with `persons`."""
        if self.last_t is not None and (t < self.last_t or t - self.last_t > TRACK_MAX_GAP_SEC):
            self.reset()
        self.last_t = t
        self.tracks = {i: tr for i, tr in self.tracks.items() if t - tr.last_t <= TRACK_MAX_GAP_SEC}

        boxes = [bbox(k, s) for k, s in persons]
        pairs = sorted(
            ((_iou(tr.bbox, b), tid, pi) for tid, tr in self.tracks.items() for pi, b in enumerate(boxes)),
            reverse=True,
        )
        assigned: dict[int, Track] = {}
        used_tracks = set()
        for iou, tid, pi in pairs:
            if iou < MIN_IOU:
                break
            if tid in used_tracks or pi in assigned:
                continue
            assigned[pi] = self.tracks[tid]
            used_tracks.add(tid)

        result = []
        for pi, (kps, scs) in enumerate(persons):
            tr = assigned.get(pi)
            if tr is None:
                tr = Track(next(self._ids))
                self.tracks[tr.id] = tr
            tr.add(t, kps, scs)
            result.append(tr)
        return result


def _rel_wrists(kps, scs, center, scale):
    w = _mid(kps, scs, L_WR, R_WR)
    return None if w is None else (w - center) / scale


def motion_features(track: Track, others: list[tuple]) -> list[float]:
    """Motion feature vector for a track's latest frame, in MOTION_FEATURE_NAMES order.

    others: (center, scale) of every other person in the same frame.
    The five window features yield 0 with too little history or hidden keypoints;
    the single-frame cues default to 'far away' (wrist_near_head 3.0, nearest_person 10.0)
    or 0 (head_down).
    """
    hist = list(track.history)
    t_now, kps, scs = hist[-1]
    center, scale = body_frame(kps, scs)

    # Single-frame cues that only make sense in body-relative coordinates
    head_pts = [kps[i] for i in (NOSE, L_EAR, R_EAR) if scs[i] > MIN_VIS]
    wrist_pts = [kps[i] for i in (L_WR, R_WR) if scs[i] > MIN_VIS]
    if head_pts and wrist_pts:
        wrist_near_head = min(np.linalg.norm(w - h) for w in wrist_pts for h in head_pts) / scale
    else:
        wrist_near_head = 3.0
    sh = _mid(kps, scs, L_SH, R_SH)
    head_down = float((kps[NOSE][1] - sh[1]) / scale) if sh is not None and scs[NOSE] > MIN_VIS else 0.0
    if others:
        nearest_person = min(np.linalg.norm(center - c) / scale for c, _ in others)
    else:
        nearest_person = 10.0

    body_velocity = wrist_motion = gait_rhythm = hands_ahead = wrist_rise = 0.0
    dt = t_now - hist[0][0]
    if len(hist) >= 3 and dt >= 0.3:
        frames = [(t, *body_frame(k, s), k, s) for t, k, s in hist]
        c0, c1 = frames[0][1], frames[-1][1]
        displacement = (c1 - c0) / scale
        body_velocity = float(np.linalg.norm(displacement) / dt)

        rel = [(t, _rel_wrists(k, s, c, sc)) for t, c, sc, k, s in frames]
        rel = [(t, r) for t, r in rel if r is not None]
        if len(rel) >= 2:
            steps = [np.linalg.norm(b[1] - a[1]) / max(b[0] - a[0], 1e-3) for a, b in zip(rel, rel[1:])]
            wrist_motion = float(np.mean(steps))
            # image y points down, so a rise is a decrease in y
            wrist_rise = float(rel[0][1][1] - rel[-1][1][1])
            if body_velocity > 0.3:
                direction = displacement / np.linalg.norm(displacement)
                mean_rel = np.mean([r for _, r in rel], axis=0)
                hands_ahead = float(np.dot(mean_rel, direction))

        ankles = [(k[L_AN] - k[R_AN]) / sc for _, _, sc, k, s in frames
                  if s[L_AN] > MIN_VIS and s[R_AN] > MIN_VIS]
        if len(ankles) >= 3:
            gait_rhythm = float(np.std([np.linalg.norm(a) for a in ankles]))

    return [
        float(np.clip(body_velocity, 0, 5)),
        float(np.clip(wrist_motion, 0, 10)),
        float(np.clip(gait_rhythm, 0, 3)),
        float(np.clip(hands_ahead, -3, 3)),
        float(np.clip(wrist_near_head, 0, 3)),
        float(np.clip(head_down, -3, 3)),
        float(np.clip(wrist_rise, -3, 3)),
        float(np.clip(nearest_person, 0, 10)),
    ]
