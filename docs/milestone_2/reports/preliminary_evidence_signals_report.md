# Preliminary Evidence Signals Report

## Goal
Identify which XAI-derived numerical signals are stable and feasible enough to
support CARE feature construction. Signals are extracted on the degraded
(initial) observations (`src/evidence_signals/`), one row per CLEAR-Face-R
record (`preliminary_evidence_signals.csv`), then aggregated and classified
(`evidence_signal_stability.csv`).

## Signal definitions
| Signal | What it checks | How it is computed |
|--------|----------------|--------------------|
| AU-prediction consistency | Facial-muscle cues vs FER prediction | Experimental: no AU detector wired in yet → emitted as NaN |
| Landmark reliability | Are landmarks detected reliably | Fraction of MediaPipe FaceMesh landmarks within frame |
| Attribution concentration | Is attribution on useful face regions | Grad-CAM mass inside the face box ÷ total CAM mass |
| Background-attribution ratio | Does the model over-focus on background | `1 − attribution_concentration` |
| Occlusion support | Does hiding key regions change the prediction | Predicted-class prob drop when the face centre is occluded |
| Pseudo-temporal stability | Is the prediction stable under tiny changes | Agreement of predicted class under flip / small rotation / brightness jitter |

## Signals tested
> Fill from `evidence_signal_stability.csv` (`computed_frac`, `std`,
> `classification`).

| Signal | Computed? | Stable? | Computationally feasible? | Use in CARE? |
|--------|-----------|---------|---------------------------|--------------|
| AU-prediction consistency | `<No>` | `<No>` | `<No>` | `<No>` |
| Landmark reliability | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` |
| Attribution concentration | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` |
| Background-attribution ratio | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` |
| Occlusion support | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` |
| Pseudo-temporal stability | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` |

## Final selected preliminary CARE evidence signals
> List the signals classified `usable`.
- Signal 1: `<fill>`
- Signal 2: `<fill>`
- Signal 3: `<fill>`

## Excluded signals
- Signal: AU-prediction consistency
- Reason: no facial-action-unit detector integrated yet; emitted as NaN and
  marked `exclude_from_CARE` (revisit when an AU model is added).
- Signal: `<fill any others classified experimental/exclude>`
- Reason: `<fill>`

## Limitations
"Stable" here means *reproducibly computable with a non-degenerate distribution*
across records. Full temporal/perturbation stability (re-extracting each signal
under repeated augmentation and measuring variance per image) is deferred; the
`pseudo_temporal_stability` signal is a first step toward that. XAI extraction is
compute-heavy, so signals are extracted on a capped subset (`--signals_limit`).
