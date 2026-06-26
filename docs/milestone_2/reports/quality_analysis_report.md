# Quality Analysis Report

The quality module (`src/quality/`) diagnoses *why* an observation may be
unreliable. It writes one row per image with the columns below
(`experiments/milestone_2/results/quality_clean_images.csv` and
`..._synthetic_degraded_images.csv`).

## Metric definitions

| Metric | What it measures | How it is computed |
|--------|------------------|--------------------|
| `brightness` | Too dark / too bright | Mean of the grayscale image (0–255) |
| `contrast` | Visual clarity of the face | Std-dev of the grayscale image |
| `blur_score` | Image / face-region blur | Variance of the Laplacian (higher = sharper) |
| `face_area_ratio` | How large the face is in frame | Detected face box area ÷ image area |
| `yaw`, `pitch`, `roll` | Head pose (frontal vs turned) | MediaPipe FaceMesh landmarks + `cv2.solvePnP` (Euler degrees) |
| `occlusion_score` | Whether key regions are blocked | Left/right facial symmetry difference proxy (0–1) |
| `motion_score` | Motion blur / frame instability | Undefined for a single still image (NaN); only meaningful for sequences |
| `detector_confidence` | Face-detector confidence | MediaPipe detection score (NaN under the Haar fallback) |

## Implementation details and thresholds
- Backend: MediaPipe (face detection + FaceMesh) when available; otherwise an
  OpenCV Haar-cascade fallback that still provides brightness/contrast/blur/face
  box, but reports `yaw/pitch/roll/occlusion/detector_confidence` as NaN.
- No hard thresholds are baked in: raw continuous values are recorded so that
  thresholds can be tuned later for CARE. Suggested starting points (to confirm
  against clean-image statistics): very dark `brightness < 40`, blurry
  `blur_score < 100`, small face `face_area_ratio < 0.05`, non-frontal
  `|yaw| > 25°`.

## Summary statistics on clean images
> Fill from `quality_clean_images.csv` (e.g. `df.describe()`).

| Metric | mean | std | p05 | p95 |
|--------|------|-----|-----|-----|
| brightness | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| contrast | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| blur_score | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| face_area_ratio | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| `|yaw|` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

Face detected in `<fill>` % of clean images.

## Examples of low-quality cases
> Reference image IDs from the CSV (kept local; do not push face images).
- Darkest: `<fill image_id>` (brightness `<fill>`)
- Blurriest: `<fill image_id>` (blur_score `<fill>`)
- Smallest face: `<fill image_id>` (face_area_ratio `<fill>`)
- Most non-frontal: `<fill image_id>` (`|yaw|` `<fill>`)

## Limitations
- **Pose** depends on MediaPipe landmark quality; on heavily degraded or
  non-frontal faces landmarks may fail, returning NaN.
- **Occlusion** is a coarse symmetry proxy, not a trained occlusion detector —
  treat as indicative only.
- **Motion** cannot be estimated from a single still image; it is NaN here and
  deferred to the real-trial / video stage.
- Under the Haar fallback, pose/occlusion/detector-confidence are unavailable.
