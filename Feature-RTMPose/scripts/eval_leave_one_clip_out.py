"""Leave-one-clip-out evaluation of the activity classifier.

Adjacent frames are near-duplicates, so a random train/test split of frames
scores a model on frames it has effectively already seen. This script instead
holds out one whole source video at a time, trains on the rest (same model as
the Training tab: MultiOutput RandomForest), and scores the held-out clip.

A label can only be scored on a held-out clip if some OTHER clip also carries
it; otherwise the model never saw an example. Such cases are reported as
"untestable", not silently counted.

Usage (from Feature-RTMPose/):
    venv/bin/python scripts/eval_leave_one_clip_out.py data/projects/<id>/dataset.json [--out result.json]
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset.json written by the Training tab")
    ap.add_argument("--features", type=int, default=None,
                    help="use only the first N features (7 = pose only, default = all)")
    ap.add_argument("--out", help="write the full result as JSON")
    a = ap.parse_args()

    with open(a.dataset) as f:
        data = json.load(f)
    X = np.array(data["features"], dtype=float)
    if a.features:
        X = X[:, :a.features]
    label_sets = data["label_sets"]
    sources = [m.get("source", "unknown") for m in data["metadata"]]
    labels = sorted({l for ls in label_sets for l in ls})
    Y = np.array([[int(l in ls) for l in labels] for ls in label_sets])
    clips = sorted(set(sources))

    print(f"{len(X)} samples, {X.shape[1]} features, {len(labels)} labels, {len(clips)} clips\n")

    # tp/fp/fn per label, summed over folds where the label is testable
    counts = {l: {"tp": 0, "fp": 0, "fn": 0} for l in labels}
    untestable = defaultdict(list)
    folds = []

    for clip in clips:
        test = np.array([s == clip for s in sources])
        train = ~test
        if len(set(sources[i] for i in np.where(train)[0])) == 0:
            continue
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
        # A label column with a single class in training cannot be fit as binary;
        # RandomForest handles it, but predict_proba then returns one column.
        model.fit(X[train], Y[train])
        proba = model.predict_proba(X[test])

        fold = {"clip": clip, "samples": int(test.sum()), "labels": {}}
        for j, l in enumerate(labels):
            in_train = Y[train, j].any()
            in_test = Y[test, j].any()
            est = model.estimators_[j]
            p = proba[j]
            p_yes = p[:, list(est.classes_).index(1)] if 1 in est.classes_ else np.zeros(len(p))
            pred = p_yes >= 0.5
            truth = Y[test, j].astype(bool)
            tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
            if in_test and not in_train:
                untestable[l].append(clip)
                fold["labels"][l] = {"untestable": True, "true": int(truth.sum())}
                # false positives on other clips are still real errors
                counts[l]["fp"] += fp
                continue
            counts[l]["tp"] += tp
            counts[l]["fp"] += fp
            counts[l]["fn"] += fn
            if in_test or fp:
                fold["labels"][l] = {"true": int(truth.sum()), "tp": tp, "fp": fp, "fn": fn}
        folds.append(fold)

    print(f"{'label':<22}{'precision':>10}{'recall':>9}{'F1':>7}   untestable on")
    summary = {}
    f1s = []
    for l in labels:
        c = counts[l]
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if rec is not None else None)
        if f1 is not None:
            f1s.append(f1)
        fmt = lambda v: f"{v:.2f}" if v is not None else "—"
        print(f"{l:<22}{fmt(prec):>10}{fmt(rec):>9}{fmt(f1):>7}   {', '.join(untestable[l]) or '-'}")
        summary[l] = {**c, "precision": prec, "recall": rec, "f1": f1, "untestable_on": untestable[l]}

    macro = float(np.mean(f1s)) if f1s else None
    print(f"\nmacro F1 over testable labels: {macro:.2f}" if macro is not None else "\nno testable labels")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"samples": len(X), "features": X.shape[1], "clips": clips,
                       "labels": summary, "macro_f1": macro, "folds": folds}, f, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
