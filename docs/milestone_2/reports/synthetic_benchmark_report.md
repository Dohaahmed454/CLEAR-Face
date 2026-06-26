# Synthetic Benchmark Report

## Goal
Validate that each synthetic failure condition harms FER performance and that
its intended correction creates measurable improvement. This is the main
exit-gate test for Milestone 2.

## Method
- Degradations are generated **only from held-out test images** (`src/synthetic/`,
  via `bench_test_manifest.csv`) so the FER model has never trained on them.
- Each condition has mild / medium / severe levels with full provenance in
  `data/milestone_2/metadata/synthetic_degradation_manifest.csv`.
- "Degraded accuracy" = primary FER accuracy on the degraded images
  (`primary_synthetic_degraded_eval.csv`). "Clean accuracy" = accuracy on the
  same originals (`primary_internal_eval.csv`).
- "Recovery rate" = fraction of CLEAR-Face-R records where the post-correction
  prediction is correct while the initial (degraded) prediction was wrong
  (`synthetic_clear_face_r_records_validated.csv`).

Degradation methods: lighting = gamma darkening; blur/motion = increasing
Gaussian kernels (mild/medium) + horizontal motion blur (severe); pose =
perspective warp (**weak proxy — see limitation**); distance = shrink-and-pad
(smaller face + more background + resolution loss).

## Lighting
- Clean accuracy: `<fill>`
- Degraded accuracy: `<fill>` (group by `failure_condition=lighting`)
- Accuracy drop: `<fill>`
- Recovery rate: `<fill>`
- Decision: valid / revise / remove

## Blur/Motion
- Clean accuracy: `<fill>`
- Degraded accuracy: `<fill>`
- Accuracy drop: `<fill>`
- Recovery rate: `<fill>`
- Decision: valid / revise / remove

## Pose
- Clean accuracy: `<fill>`
- Degraded accuracy: `<fill>`
- Accuracy drop: `<fill>`
- Recovery rate: `<fill>`
- Decision: valid / revise / remove
- NOTE: realistic out-of-plane pose cannot be synthesized from a single 2D
  image; the perspective warp is a weak proxy. Consider limiting pose
  conclusions until real trials.

## Distance
- Clean accuracy: `<fill>`
- Degraded accuracy: `<fill>`
- Accuracy drop: `<fill>`
- Recovery rate: `<fill>`
- Decision: valid / revise / remove

## Accuracy-drop summary
> Fill by grouping `primary_synthetic_degraded_eval.csv` by condition × severity.

| Condition | Severity | Clean accuracy | Degraded accuracy | Accuracy drop |
|-----------|----------|----------------|-------------------|---------------|
| lighting | severe | `<fill>` | `<fill>` | `<fill>` |
| blur_motion | medium | `<fill>` | `<fill>` | `<fill>` |
| pose | severe | `<fill>` | `<fill>` | `<fill>` |
| distance | severe | `<fill>` | `<fill>` | `<fill>` |

## Final decision
| Condition | Intended action | Valid? | Reason |
|-----------|-----------------|--------|--------|
| Lighting | Improve lighting | `<Yes/No>` | `<fill>` |
| Blur/motion | Hold still / short sequence | `<Yes/No>` | `<fill>` |
| Pose | Face camera | `<Yes/No>` | `<fill>` (proxy weak) |
| Distance | Move closer | `<Yes/No>` | `<fill>` |
