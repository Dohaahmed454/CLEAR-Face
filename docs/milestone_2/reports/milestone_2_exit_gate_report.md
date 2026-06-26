# Milestone 2 Exit-Gate Report

> Fill the `Yes/No` / evidence cells from the result CSVs after running the
> pipeline. Proceed to Milestone 3 only if conditions 1 and 2 hold for the
> retained failure conditions and at least one evidence signal passes
> condition 3.

## Exit-gate condition 1
Each failure condition reliably harms FER performance.
(Evidence: `primary_synthetic_degraded_eval.csv` vs `primary_internal_eval.csv`.)

| Failure condition | Harm observed? | Evidence (accuracy drop) | Decision |
|-------------------|----------------|--------------------------|----------|
| Lighting | `<Yes/No>` | `<fill>` | `<Keep/Revise/Remove>` |
| Blur/motion | `<Yes/No>` | `<fill>` | `<Keep/Revise/Remove>` |
| Pose | `<Yes/No>` | `<fill>` (proxy weak) | `<Keep/Revise/Remove>` |
| Distance | `<Yes/No>` | `<fill>` | `<Keep/Revise/Remove>` |

## Exit-gate condition 2
Each intended corrective action creates measurable improvement.
(Evidence: recovery rate in `synthetic_clear_face_r_records_validated.csv`.)

| Failure condition | Intended action | Improvement observed? | Evidence (recovery rate) | Decision |
|-------------------|-----------------|-----------------------|--------------------------|----------|
| Lighting | Improve lighting | `<Yes/No>` | `<fill>` | `<fill>` |
| Blur/motion | Hold still / short sequence | `<Yes/No>` | `<fill>` | `<fill>` |
| Pose | Face camera | `<Yes/No>` | `<fill>` | `<fill>` |
| Distance | Move closer | `<Yes/No>` | `<fill>` | `<fill>` |

## Exit-gate condition 3
Preliminary evidence signals are suitable for CARE feature construction.
(Evidence: `evidence_signal_stability.csv`.)

| Signal | Stable? | Feasible? | Use in CARE? |
|--------|---------|-----------|--------------|
| AU-prediction consistency | `<No>` | `<No>` | `<No>` |
| Landmark reliability | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` |
| Attribution concentration | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` |
| Background-attribution ratio | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` |
| Occlusion support | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` |
| Pseudo-temporal stability | `<Yes/No>` | `<Yes/No>` | `<Yes/No>` |

## Final decision
Proceed to Milestone 3: `<Yes/No>`

## Notes
If any condition-action pair is weak, revise or remove it before CARE training.
The pose condition uses a weak single-image proxy; if it fails to harm
performance reliably, mark it limited and defer firm pose conclusions to real
trials rather than discarding the action outright.
