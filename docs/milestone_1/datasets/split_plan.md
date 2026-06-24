# Dataset Split Plan

## Principle

Splits must be participant-disjoint wherever participant IDs are available. The same participant must not appear in training, policy development, and final testing. This ensures unbiased evaluation and prevents identity leakage across model training and system evaluation.

---

## Split types

- Model split: used to train, validate, and test the FER recognition model.
- Policy-development split: used to design and tune CARE decision logic under controlled conditions.
- Final-test split: used for unbiased evaluation of the full CLEAR-Face system, including prediction, reacquisition, and explanation modules.

---

## Suggested split

### FER model split
- Training: 60%
- Validation: 15%
- Internal test: 25%

### CARE policy split
- Policy-development participants: 15% of total participants (non-overlapping with model split)

### Final evaluation split
- Final-test participants: 25% of total participants (strictly disjoint from all other splits)

---

## Leakage prevention

- No participant may appear in more than one of the following: training, policy development, or final evaluation.
- All samples from the same participant must remain in the same split.
- Synthetic degradations (blur, lighting, pose, distance) must inherit the split of their original clean image.
- Pre- and post-synthetic pairs must never be separated across different splits.
- Final-test participants must not be used for tuning thresholds, prompts, features, or CARE policy decisions.
- Any hyperparameter tuning must be restricted to the validation or policy-development splits only.