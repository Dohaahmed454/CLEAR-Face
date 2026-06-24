# CLEAR-Face-R Record Schema

Each record represents one reacquisition trial in the CLEAR-Face system, capturing the full decision cycle from initial unreliable observation to post-action outcome.

---

## Required fields

- record_id: Unique identifier for each reacquisition trial.
- participant_id: Anonymous participant or source identifier.
- source_dataset: Dataset origin or real-world trial source.
- split: model / policy_dev / final_test.

- observation_type: synthetic / real.
- initial_observation_id: ID or path of the original image or sequence that triggered failure.

- failure_condition: lighting / blur_motion / pose / distance.
- failure_severity: mild / medium / severe.

- available_actions: Set of valid CARE actions at decision time.
- selected_action: Action chosen by CARE policy or experimental protocol.

- post_action_observation_id: ID or path of the corrected image or new observation.

---

## Prediction and uncertainty fields

- initial_prediction: FER prediction before correction.
- initial_confidence: Model confidence before correction.
- initial_uncertainty: Uncertainty measures (entropy / margin / calibration score).

- initial_quality: Image quality metrics (brightness, blur score, face size, pose estimate).

- post_prediction: FER prediction after correction.
- post_confidence: Model confidence after correction.
- post_uncertainty: Uncertainty measures after correction.

- post_quality: Image quality metrics after correction.

---

## Ground truth and evaluation

- ground_truth_label: True annotation label from dataset or human annotation.
- recovery: Whether the CARE action improved or corrected the prediction outcome.

- risk_change: Difference in prediction risk before and after action.

---

## System-level metadata

- compliance: assumed / simulated / measured / unknown.
- burden: User effort score (e.g., number of prompts or perceived effort).
- latency: Time cost or simulated response delay.
- notes: Additional remarks, edge cases, or exclusion flags.