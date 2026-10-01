# Objects and context

The pipeline sees only people. Its detector is YOLOX with `*_humanart` weights from `rtmlib.Body`, trained on one class (person), and RTMPose places keypoints only inside those boxes. Boxes, carts and phones do not exist for it. `carrying` is inferred **from pose** (hands in front at chest height, symmetric). The box itself is never seen.

To tell "on the phone" from "scratching the head", or "pulling a cart" from "walking", you need one more signal: pose cues over time, an object detector, or the location in the frame.

Related: [how-activity-recognition-works.md](how-activity-recognition-works.md) · [cross-camera-reid.md](cross-camera-reid.md) · [design-decisions.md](design-decisions.md).

## 1. Pose-only cues (already in the feature set)

Motion features from [motion.py](../Feature-RTMPose/backend/services/motion.py) cover some of this. They are heuristics and confuse similar poses. The reliability column is an expectation, not a measurement; see the [README](../README.md#demo-results) for what was actually measured.

| Activity | Skeleton cue | Feature | Expected reliability |
|---|---|---|---|
| On the phone | wrist near the ear, elbow bent, held for seconds | `wrist_near_head` | medium: looks like adjusting a cap |
| Pulling a cart | walking, hands **behind** along the direction of motion | `hands_ahead` < 0 | medium; hands often hidden from above |
| Pushing a cart | walking, both hands **in front** at waist height | `hands_ahead` > 0 | medium; easily confused with carrying |
| Talking to someone | two people close, both standing, for seconds | `nearest_person` | high with tracking. Speech itself can't be detected without audio |
| Reading a tablet | head down, wrists close together in front of the chest | `head_down` | low without an object detector |

## 2. Object detector + human-object interaction

A second model finds object boxes. Geometry then links each object to a person:

- phone within ~0.3 torso lengths of ear and wrist → on the phone;
- cart box moving with the person, hands on the handle → pulling or pushing (who leads along the motion direction);
- box between the wrists → carrying (confirms the pose).

| Model | Knows | Code license | Notes |
|---|---|---|---|
| YOLOX on COCO | 80 classes incl. `cell phone`, `book`, `laptop`, `suitcase`; **no** cart, pallet jack, box, scanner | Apache-2.0 | cheapest step: `rtmlib`'s YOLOX supports `det_mode='multiclass'`; drop in a COCO ONNX model |
| RT-DETR (original repo) | same 80 COCO classes | Apache-2.0 | better on small objects, slower on CPU |
| Grounding DINO | open vocabulary: "pallet jack", "barcode scanner" as text | Apache-2.0 | no training, but seconds per frame on CPU; good for labeling, not real time |
| OWLv2 | open vocabulary | Apache-2.0 | same as above |
| YOLO-World | open vocabulary, fast | **GPL-3.0** | copyleft |
| Ultralytics YOLOv8 / YOLO11 | anything after fine-tuning | **AGPL-3.0** | rejected for this repo, see [design-decisions.md](design-decisions.md) |

**Check the license of the weights and of the training dataset, not only the code.** Weights can carry stricter terms than the repository. Verify the LICENSE before adding any model.

For classes COCO lacks (cart, pallet jack, box, scanner):

1. Label object boxes (CVAT, Label Studio; the dashboard's Annotate tab only does skeleton points).
2. Fine-tune an Apache-licensed detector (YOLOX, RT-DETR). Plan for hundreds of labeled examples per class, from several viewpoints.
3. Shortcut: use Grounding DINO / OWLv2 as an automatic labeler for step 1, then train the fast model on its output.

## 3. Zones

Often the cheapest and most reliable signal. Draw polygons once per fixed camera: packing table, loading dock, aisle, rack A.

- in the table zone + hands at table height + moving → packing;
- next to a rack + arm extended to a shelf → picking / scanning;
- in the loading zone + cart → loading.

## Which signal for which activity

Expected, not measured on the demo clips.

| Activity | Skeleton over time | Object detector | Zones | Recommendation |
|---|---|---|---|---|
| Walking / standing | yes | — | — | skeleton only |
| Carrying a box | yes | confirms | — | skeleton, detector as a bonus |
| Lifting | yes | — | — | skeleton |
| Pulling / pushing a cart | partial | yes, needs a cart class | partial | skeleton + detector |
| Phone | partial | yes, `cell phone` is in COCO | — | skeleton + COCO detector |
| Tablet / paperwork | partial | partial (`book`, `laptop`; tablets unstable) | — | skeleton + detector, flag as unreliable |
| Talking | yes, from pairs of tracks | — | — | skeleton + tracking |
| Scanning | partial | yes, needs a scanner class | yes, near racks | detector + zones |
| Packing | partial | partial | yes, table | zones + skeleton |

## Rollout order

1. **Tracking + motion features.** Done ([motion.py](../Feature-RTMPose/backend/services/motion.py)). Still missing: temporal smoothing, one label per track segment instead of per frame.
2. **COCO YOLOX in `multiclass` mode.** Gives phone, book and laptop almost for free.
3. **Zones per camera.** A small JSON of polygons.
4. **Custom cart / box / scanner detector.** Only once steps 1–3 work and labeled data exists.

Each step adds per-frame latency on top of an already expensive CPU pipeline (see [README](../README.md#demo-results)). Run the object detector every 5–10 frames, not on every frame.
