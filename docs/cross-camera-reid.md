# Cross-camera ReID

Person re-identification answers "is this the same person as in that other frame or camera?". Feature-RTMPose uses **OSNet-AIN** for it. This page covers how it works, why it fails on identical uniforms and viewpoint changes, and how to measure it.

Related: [how-activity-recognition-works.md](how-activity-recognition-works.md) · [objects-and-context.md](objects-and-context.md) · [design-decisions.md](design-decisions.md).

## Idea: an appearance fingerprint, not a face

ReID does **not** use faces. On warehouse cameras faces are small, turned away or hidden under a cap. The model looks at the whole figure instead: clothing colors and textures, headwear, silhouette, proportions. The output is a **512-d embedding**. Crops of the same person should land close together and crops of different people far apart. Comparing two people means taking the cosine of the angle between their vectors.

```
frame ─► RTMPose skeleton ─► person crop ─► OSNet-AIN ─► 512-d vector
                                                              │
                gallery of enrolled workers ◄── cosine similarity
```

## What the code does

| Step | Details | Code |
|---|---|---|
| Crop | box around keypoints with confidence > 0.3, plus 15% padding per side; skipped if < 4 visible points or smaller than 20×40 px | [reid.py](../Feature-RTMPose/backend/services/reid.py) `crop_person` |
| Preprocess | resize to 128×256, ImageNet mean/std | `preprocess` |
| Embed | OSNet-AIN (ONNX, CPU) → 512 values, L2-normalized | `extract_embedding` |
| Compare | cosine similarity against **every** stored embedding of every person; best wins | `match` |
| Decide | best ≥ **0.65** → name, else `unknown` | `DEFAULT_SIMILARITY_THRESHOLD` |

**Enrollment.** Add a person from a photo or from a video frame: Training → Workers, or `POST /api/workers/collections/{id}/enroll` (photo) and `.../enroll-from-frame` (frame plus `person_index`, counted left to right). Up to **10 embeddings per person**. Once full, a new embedding replaces the *most similar* stored one, so the gallery collects different views instead of ten copies of one.

**Why several embeddings, not an average.** Overhead and eye-level views of the same person give very different vectors. Their mean resembles neither, so matching against the nearest stored view works better.

**Collections.** There can be several galleries ("Workers Collections"), each linked to a project. One shared ONNX model serves all of them. Activating a project switches to the gallery its enroll endpoints write to (`activate_project` in [main.py](../Feature-RTMPose/backend/main.py)). A project without a collection falls back to the global gallery.

## Why it fails, and the fixes

| Cause | Effect | Fix |
|---|---|---|
| **Per-frame matching.** Each detection is matched on its own. The IoU tracker's `track_id` is not used by ReID, and nothing enforces one name per person in a frame | a person's name can flip between frames; two people in one frame can get the same name | average the embedding over a track; assign tracks to gallery identities with the Hungarian algorithm (each name at most once per frame); stronger tracking (ByteTrack / OC-SORT) |
| **ImageNet weights.** [export_reid_model.py](../Feature-RTMPose/scripts/export_reid_model.py) builds `osnet_ain_x1_0` with `pretrained=True`, "pretrained on ImageNet": generic image features, not trained to tell people apart | color and texture help, but clearly weaker than ReID weights | OSNet-AIN weights trained on a ReID dataset (torchreid model zoo). torchreid code is MIT, but Market-1501, MSMT17 and DukeMTMC restrict use, and DukeMTMC was withdrawn. Settle the weights' license **before** shipping them in a public repo or product |
| **Uniforms** | with identical workwear, only headwear, hair and build differ, a small part of the crop. Headwear works in a demo and is gone where everyone wears the same helmet | ReID weights; track-level averaging; cross-camera constraints below |
| **Viewpoint change** | from above you see the crown and shoulders; at eye level the face, chest and legs. To the model these are nearly different images, and overhead → eye level is the hardest case | enroll one embedding per camera / viewpoint |
| **Single threshold (0.65)** | higher → more `unknown` for enrolled people; lower → strangers named as someone in the gallery | tune on data with the threshold curve below |

## Cross-camera constraints

The appearance model knows nothing about cameras. Real systems add:

- **Time and space.** Nobody is on camera 1 and camera 3 at once, and moving between cameras takes a minimum transit time.
- **Topology.** Aisle A leads to the loading dock, not straight to the office.
- **Shift gallery.** Enroll people at shift start, or as they pass an entrance camera, and search only among them.

These constraints only make sense for recordings that are actually synchronized.

## How to measure

1. **Gallery:** enroll each worker from **one** clip or camera.
2. **Queries:** run other clips with the same people, including a different viewpoint.
3. **Strangers:** run clips with **other** people. The correct answer for them is `unknown`.
4. Per frame, record ground truth (manual labels) and the model's answer.

| Metric | Meaning |
|---|---|
| Rank-1 | share of queries whose nearest gallery match is the right person |
| Stranger false matches | how often a non-enrolled person gets a gallery name |
| In-frame duplicates | frames where one name is given to two people |
| Threshold curve | the two error rates above as the threshold moves over 0.5–0.8 |

Report the overhead ↔ eye-level pair separately; expect it to be the worst. Measured results on the demo clips: [README](../README.md#demo-results).

## Privacy

An appearance embedding lets a system follow a specific person across cameras. In many jurisdictions (GDPR in the EU, for example) that is processing of personal data, and in a real workplace it is also a labor-law question. For demos in this repo:

- neutral labels only, no invented personal names;
- no judgmental captions ("inefficient", "idle");
- raw clips are not committed; `docs/` holds only derived GIFs and screenshots.
