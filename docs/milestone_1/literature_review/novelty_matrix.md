# CLEAR-Face Novelty Comparison Matrix

---

## Comprehensive Novelty Comparison Matrix (All Papers)

| Paper | Year | Venue | Task | Method | Human-in-the-loop? | Reacquisition? | Uncertainty? | XAI? | Language Explanation? | CLEAR-Face Criterion | Gap for CLEAR-Face |
|-------|------|-------|------|--------|:------------------:|:--------------:|:------------:|:----:|:---------------------:|----------------------|--------------------|
| **FER Backbones & Robustness** |
| Wang et al. (2024) | 2024 | Computers, Materials & Continua | Facial Expression Recognition | Uncertainty + Active Learning | ❌ | ❌ | ✅ | ❌ | ❌ | Passive Training | No inference-time uncertainty use, no user interaction, no reacquisition |
| Radoi & Cioroiu (2024) | 2024 | IEEE Access | Multimodal Emotion Recognition | Uncertainty-based iterative learning | ❌ | ❌ | ✅ | ❌ | ❌ | Passive Training | Uncertainty only for training, not for deployment decisions or user feedback |
| Karamizadeh et al. (2025) | 2025 | IEEE Access | Facial Expression Analysis | Zero-shot + Deep Learning + LLM | ❌ | ❌ | ❌ | ✅ (partial) | ✅ | Black-box VLM | No uncertainty, no reacquisition, no interpretability at facial evidence level |
| Naseem et al. (2026) | 2026 | IEEE Access | Multimodal FER | CNN + IR fusion + Attention | ❌ | ❌ | ❌ | ✅ (partial) | ❌ | Passive Fusion | No uncertainty, no image quality control, no explanation or abstention |
| Rieger et al. (2025) - FMC-Net | 2025 | Applied Intelligence | FER | Multi-task + Fuzzy rules + AU-FE correlation | ❌ (passive risk flag) | ❌ | ✅ (disagreement score) | ✅ (fuzzy rules) | ❌ | Passive Risk Triage | No mechanism to automatically fix bad inputs; relies on external human |
| Wu et al. (2026) - Facial-R1 | 2026 | AAAI | Facial Emotion Analysis | VLM + Instruction fine-tuning + RL | ❌ | ❌ | ❌ | ✅ (VLM reasoning) | ✅ (structured text) | High-Latency VLM | High latency; not suitable for real-time control loops |
| Liu & Shi (2025) - AU-DFER | 2025 | arXiv | Dynamic FER | AU weight matrices + Dynamic AU loss | ❌ | ❌ | ❌ | ❌ | ❌ | Static Video Assumption | Assumes pristine video; no handling of real-world noise |
| **Explainable AI (XAI) & Action Unit Faithfulness** |
| Nahulanthran et al. (2025) | 2025 | AAMAS | FER Explanation | Multimodal symbolic XAI + AU + visual maps | ❌ | ❌ | ❌ | ✅ (multimodal) | ✅ (text+visual) | Post-hoc Only | Post-hoc only; cannot explain active reacquisition or abstention |
| Verlekar et al. (2025) - XFER-AU | 2025 | IEEE Access | FER | AU-spatial CNN masks | ❌ | ❌ | ❌ | ✅ (AU-spatial) | ❌ | Passive Spatial Restriction | Passive; cannot categorise type of environmental degradation |
| Mohana et al. (2025) - XAI-DSCSA | 2025 | SIVP | FER | Semi-supervised sparse autoencoder | ❌ | ❌ | ❌ | ✅ (saliency) | ❌ | Saliency-only | Heatmaps only; no multi-criteria quality flaws or human-readable terms |
| Khokhar et al. (2026) - RobustDRNet | 2026 | ESWA | Medical Image Classification | CNN+Transformer ensemble + Multi-method XAI | ❌ | ❌ | ❌ | ✅ (multi-method) | ❌ | Domain-specific XAI | No uncertainty-aware acquisition or human interaction loop |
| Giddaluru et al. (2025) | 2025 | ICOECA | Multimodal Emotion Recognition | CNN + LSTM + Transformer + XAI | ❌ | ❌ | ❌ | ✅ (multi-method) | ❌ | Passive Multimodal | No facial quality assessment or reacquisition logic |
| Keerthiga et al. (2025) | 2025 | Springer AISC | Multimodal Emotion Recognition | Multimodal fusion + XAI | ❌ | ❌ | ❌ | ✅ (multi-method) | ❌ | Passive Multimodal | No uncertainty, no reacquisition, no decision abstention |
| Keerthiga et al. (2026) | 2026 | Springer AISC | Multimodal Emotion Recognition | Fusion + XAI + explanation-guided learning | ❌ | ❌ | ❌ | ✅ (multi-method) | ❌ | Passive Multimodal | No image-level decision control or quality-aware inference |
| Macchiarulo et al. (2025) - XFERa | 2025 | ACM IUI Workshops | FER | CNN + Grad-CAM + AU + GPT-4 | ✅ (partial, user study) | ❌ | ❌ | ✅ (multi-modal) | ✅ (GPT-4) | LLM-based Explanation | No image quality assessment, no abstention, no uncertainty-driven decisions |
| **Active Perception & Reacquisition** |
| Al-dabbagh et al. (2025) - ALDAM | 2025 | Algorithms | Driver Emotion Recognition | Active Learning + Attention | ❌ | ❌ | ❌ | ✅ (partial) | ❌ | Passive Active Learning | No inference-time decision control, no reacquisition or user guidance |
| Active Perception Foundations | 2025 | — | Active Vision | Sequential decision-making for sensor control | ✅ | ✅ (robotic) | ❌ | ❌ | ❌ | Robotic Active Perception | Not mapped to facial analysis; lacks human-oriented instructions |
| **Uncertainty, Selective Prediction & Abstention** |
| Zhou et al. (2024) - UA-FER | 2024 | Neurocomputing | FER | CLIP + EDL + calibration | ❌ | ❌ | ✅ | ✅ (partial) | ❌ | Uncertainty-only | No reacquisition, no image quality awareness, no user interaction |
| Kim (2026) - Affective Sovereignty | 2026 | Discover AI | Emotion AI Governance | Policy gates + DRIFT + ASC + Handoff | ✅ | ✅ (partial) | ✅ | ❌ | ✅ | Policy-level Abstention | No vision module, no facial image quality assessment |
| Selective Prediction Theory | 2025 | — | Classification | Selection function g(x) ∈ {0,1} | ❌ | ❌ | ✅ (abstention) | ❌ | ❌ | Binary Abstention | Binary (predict/abstain); no third option to salvage the frame |
| **Natural-Language Explanation & Vision-Language Models** |
| AlDahoul et al. (2026) - FaceScanPaliGemma | 2026 | Scientific Reports | Facial Attribute Recognition | Multi-agent VLM (PaliGemma) | ❌ | ❌ | ❌ | ✅ (partial) | ✅ (VQA) | Multi-agent VLM | No image quality control, no uncertainty gating, no reacquisition mechanism |
| Zhao et al. (2026) - EmotionReasoner | 2026 | IEEE T-Affective Computing | Explainable MER | RL + explanation-aware reward | ❌ | ❌ | ✅ | ✅ (partial) | ✅ (structured) | RL-based Explanation | No visual grounding, no facial quality assessment, no reacquisition logic |
| **Confidence Calibration & Human-in-the-Loop** |
| Calibration & Human-in-the-loop | 2025 | — | Trust Calibration | Temperature Scaling + ECE + Utility balancing | ✅ | ❌ | ✅ (ECE) | ❌ | ❌ | Probability-only Calibration | Ignores physical stream parameters; only uses classification probabilities |
| **Domain Surveys** |
| Priego et al. (2026) - AU Survey | 2026 | Scientific Reports | FER Survey | Systematic review of AU-driven models | ❌ | ❌ | ❌ | ❌ | ❌ | Survey | Survey only; no active runtime fix for domain degradation |
| Georgiou et al. (2026) - FER Review | 2026 | arXiv | FER Survey | Multi-criteria systematic review | ❌ | ❌ | ❌ | ❌ | ❌ | Survey | Identifies active reacquisition as future direction, but provides no blueprint |

---

## CLEAR-Face Positioning

CLEAR-Face is positioned against **five groups of work**:

1. **Standard FER systems** that predict directly from the available image.
2. **Robust FER systems** that handle degradation but do not request a new physical observation.
3. **Selective prediction systems** that abstain but do not choose targeted corrective actions.
4. **XAI systems** that explain predictions but do not use explanations to support reacquisition decisions.
5. **Natural-language explanation systems** that generate text but are not grounded in multi-evidence visual records.

---

## Final Novelty Claim

**CLEAR-Face differs from prior work by:**

- Learning ***when*** a genuinely new visual observation is worth requesting,
- Selecting the ***most useful physical corrective action*** under risk, burden, and latency constraints,
- Producing ***evidence-grounded justification*** for prediction, request, ambiguity, or abstention.

| CLEAR-Face Component | What Prior Work Does | What CLEAR-Face Adds |
|----------------------|---------------------|---------------------|
| **CARE Controller** | Active perception limited to robotics; passive FER systems only | Active reacquisition in FER domain with human-oriented instructions |
| **Uncertainty Estimation** | Confidence/entropy only; ignores physical stream parameters | Blends model confidence with MediaPipe tracking quality metrics |
| **Selective Prediction** | Binary: predict or abstain | Adds third option: request active environmental reacquisition |
| **XAI Justification** | Post-hoc explanations for predictions only | Explains predictions, reacquisition choices, and abstention decisions |
| **Natural-Language Explanation** | LLM-generated text; high latency or hallucination risk | Rule-based template generation grounded in multi-evidence visual records |

---

## CLEAR-Face Criterion Legend

| Criterion | Description | CLEAR-Face Contribution |
|-----------|-------------|------------------------|
| **Passive Training** | Uncertainty/active learning only during training, not inference | CLEAR-Face uses uncertainty actively during inference |
| **Passive Risk Triage** | Flags uncertainty but stops; expects external human | CLEAR-Face automatically initiates reacquisition |
| **High-Latency VLM** | Uses large language models for explanation | CLEAR-Face uses fast rule-based template generation |
| **Static Video Assumption** | Assumes pristine capture conditions | CLEAR-Face detects and responds to degradation |
| **Post-hoc Only** | Explains after prediction only | CLEAR-Face explains active choices and state transitions |
| **Passive Spatial Restriction** | Focuses on AU regions but cannot diagnose degradation type | CLEAR-Face uses cross-method agreement to identify degradation causes |
| **Saliency-only** | Only pixel-level heatmaps | CLEAR-Face adds multi-criteria quality flaws and human-readable terms |
| **Domain-specific XAI** | XAI for medical images, not facial expressions | CLEAR-Face adapts XAI for FER with AU-based explanations |
| **Passive Multimodal** | Fuses modalities without quality assessment | CLEAR-Face assesses quality before fusion/decision |
| **LLM-based Explanation** | Uses GPT for text generation | CLEAR-Face uses grounded template generation avoiding hallucination |
| **Robotic Active Perception** | Active sensing for robotics/navigation | CLEAR-Face maps active perception to facial analysis |
| **Passive Active Learning** | Active learning only for training data selection | CLEAR-Face uses active reacquisition at inference |
| **Uncertainty-only** | Estimates uncertainty but takes no action | CLEAR-Face acts on uncertainty via reacquisition |
| **Policy-level Abstention** | Abstains based on policies, not vision quality | CLEAR-Face abstains based on both vision quality and uncertainty |
| **Binary Abstention** | Predict or abstain only | CLEAR-Face adds reacquisition as third option |
| **Multi-agent VLM** | Uses multiple VLMs for attribute recognition | CLEAR-Face uses specialized lightweight modules for speed |
| **RL-based Explanation** | Uses RL for explanation generation | CLEAR-Face uses deterministic rule-based generation |
| **Probability-only Calibration** | Calibrates only classification probabilities | CLEAR-Face adds physical stream parameters to calibration |
| **Survey** | Reviews literature but provides no solution | CLEAR-Face implements the solution |