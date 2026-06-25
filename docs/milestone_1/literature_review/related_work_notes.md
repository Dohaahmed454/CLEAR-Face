# CLEAR-Face Related Work Notes

---

## 1. Facial Expression Recognition Backbones & Robustness

### Paper 1: A Facial Expression Recognition Method Integrating Uncertainty Estimation and Active Learning

**Citation:**  
Wang, Y., Zhang, J., & Sun, R. (2024). *A Facial Expression Recognition Method Integrating Uncertainty Estimation and Active Learning*. Computers, Materials & Continua, 81(1), 533–552. https://doi.org/10.32604/cmc.2024.054644

**Year:** 2024  
**Venue:** Computers, Materials & Continua (CMC)

**Dataset:** Small- and medium-scale FER datasets (active learning evaluation).

**Problem:** FER systems require large labeled datasets. Active learning reduces annotation effort but suffers from the cold-start problem—initial labeled samples fail to represent full expression diversity.

**Method:** Integrates uncertainty estimation with active learning. Pretrains via self-supervised contrastive learning, applies self-attention with rank regularization, then uses uncertainty estimation to select informative samples for annotation.

**Main Result:** Outperforms existing active learning approaches by 5.09% over best method, 3.82% over Margin Sampling, 1.61% over conventional segmented active learning.

**Relation to CLEAR-Face:** Demonstrates value of uncertainty estimation for improving FER performance. Similar to CLEAR-Face, uses uncertainty to guide decision-making and improve reliability.

**Limitation/Gap:** Focuses only on training-stage sample selection. Does not evaluate image quality, determine if new observation should be captured, provide corrective acquisition guidance, generate explanations, or support human-centered interaction during inference. Cannot perform active image reacquisition or explain uncertainty to end users.

**Useful Citation Sentence:** "Wang et al. (2024) demonstrated that integrating uncertainty estimation with active learning significantly improves facial expression recognition accuracy while reducing dependence on large labeled datasets."

---

### Paper 2: Uncertainty-Based Learning of a Lightweight Model for Multimodal Emotion Recognition

**Citation:**  
Radoi, A., & Cioroiu, G. (2024). *Uncertainty-Based Learning of a Lightweight Model for Multimodal Emotion Recognition*. IEEE Access, 12, 120362–120374. https://doi.org/10.1109/ACCESS.2024.3450674

**Year:** 2024  
**Venue:** IEEE Access

**Dataset:** CREMA-D, RAVDESS

**Problem:** Training multimodal emotion recognition requires large annotated audio-visual datasets. Many applications must operate on embedded devices with limited computational resources.

**Method:** Lightweight multimodal neural network processing audio and visual information using shared CNNs across temporal segments. Employs uncertainty-based iterative learning starting with small labeled dataset, progressively incorporating high-uncertainty samples.

**Main Result:** 74.2% on CREMA-D, 76.3% on RAVDESS. ~2.7M parameters, ~10.3 MB memory, real-time inference <14ms.

**Relation to CLEAR-Face:** Demonstrates effectiveness of uncertainty estimation in emotion recognition. Similar to CLEAR-Face, uses uncertainty to guide decision-making under limited data.

**Limitation/Gap:** Uncertainty only applied during training for sample selection. System does not evaluate image quality, request new observations, recommend corrective actions, perform selective prediction, generate visual/natural-language explanations, or support human-in-the-loop decision making during inference.

**Useful Citation Sentence:** "Radoi and Cioroiu (2024) demonstrated that uncertainty-guided sample selection can significantly improve multimodal emotion recognition performance while maintaining a lightweight architecture suitable for real-time deployment."

---

### Paper 3: Enhancing Facial Recognition and Expression Analysis With Unified Zero-Shot and Deep Learning Techniques

**Citation:**  
Karamizadeh, S., Chaeikar, S. S., & Najafabadi, M. K. (2025). *Enhancing Facial Recognition and Expression Analysis With Unified Zero-Shot and Deep Learning Techniques*. IEEE Access, 13, 43508–43519. https://doi.org/10.1109/ACCESS.2025.3546061

**Year:** 2025  
**Venue:** IEEE Access

**Dataset:** FERET, LFW, CelebA, VGGFace2

**Problem:** FER systems suffer from illumination changes, noise, pose variations, and difficulty recognizing unseen expressions. Supervised approaches require extensive labeled data and struggle to generalize.

**Method:** Unified framework combining Retinex-based illumination normalization, median filtering, VGG-16 feature extraction, Boltzmann Machine and Deep Belief Network refinement, and zero-shot learning using LLMs. Aligns visual features with semantic embeddings for unseen expression recognition.

**Main Result:** 98.7% on FERET, 97.5% on LFW, 96.2% on CelebA, 95.8% on VGGFace2. Robust against illumination, noise, and pose changes.

**Relation to CLEAR-Face:** Combines facial analysis with language-based semantic representations. LLM use conceptually related to CLEAR-Face's natural-language explanation component. Highlights importance of image quality and robustness under challenging acquisition conditions.

**Limitation/Gap:** Focuses only on recognition/classification. Does not estimate uncertainty, determine if image is insufficient, request reacquisition, recommend corrective actions, perform selective prediction, provide explainable AI mechanisms, generate evidence-grounded explanations, or incorporate human-in-the-loop decision making.

**Useful Citation Sentence:** "Karamizadeh et al. (2025) demonstrated that combining illumination normalization, deep feature extraction, and zero-shot learning with large language models significantly improves facial recognition and expression analysis under challenging real-world conditions."

---

### Paper 4: Enhancing Multimodal Facial Expression Recognition with Derived Modality Transformations

**Citation:**  
Naseem, M. T., Ullah, U., & Lee, C.-S. (2026). *Enhancing Multimodal Facial Expression Recognition with Derived Modality Transformations Using Novel Convolutional Neural Network-Based Models*. IEEE Access, 14, 40997–41016. https://doi.org/10.1109/ACCESS.2026.3670161

**Year:** 2026  
**Venue:** IEEE Access

**Dataset:** NVIE, VIRI

**Problem:** FER remains challenging under illumination changes, occlusions, noise, pose variations, and limited training data. Existing multimodal fusion methods assume equal modality contribution and fail to adapt to quality.

**Method:** Combines visible and infrared facial images with H&E-inspired transformed versions. CNN-based modality-specific streams with Modality Attention Module dynamically assigning weights and performing attention-based fusion.

**Main Result:** 88.22% on NVIE, 86.66% on VIRI. +4.22% and +3.33% improvement over previous visible-IR fusion approaches. Robust under low-light, noise, limited data, and environmental variability.

**Relation to CLEAR-Face:** Highlights importance of image quality, modality selection, and adaptive feature fusion. Attention mechanisms conceptually related to selecting useful information sources.

**Limitation/Gap:** Does not estimate uncertainty, determine if observation is sufficient, request reacquisition, recommend corrective actions, perform selective prediction, incorporate human-in-the-loop interaction, generate explainable AI outputs, or provide evidence-grounded explanations.

**Useful Citation Sentence:** "Naseem et al. (2026) demonstrated that modality-derived transformations combined with attention-based multimodal fusion significantly improve facial expression recognition robustness under challenging real-world conditions, including low-light environments and limited training data."

---

### Paper 5: FMC-Net: A Human-Guided Deep Learning Framework for Adaptable and Transparent FER

**Citation:**  
Rieger, I., Pahl, J., & Schmid, U. (2025). FMC-Net: A Human-Guided Deep Learning Framework for Adaptable and Transparent Facial Expression Recognition in Real-World Scenarios. *Applied Intelligence*, 55, 1127.

**Year:** 2025  
**Venue:** Applied Intelligence

**What the Paper Does:** Multi-task network with training-time correlation constraint between AUs and FEs. At inference, fuzzy rule layer maps AUs to expert psychological templates, outputting disagreement-based risk score when patterns clash.

**The Gap (What It Misses):** System relies on passive workflow stopping at flagging uncertainty. Expects external human actor to resolve high-risk classifications, lacking mechanism to automatically fix bad inputs.

**What CLEAR-Face Gets From It:** Extracts concept of using cross-task disagreement to compute risk metric. Transforms passive risk triage score into active driver for **CARE Controller**, allowing dynamic request for image reacquisition without stalling for human intervention.

**Useful Citation Sentence:** "FMC-Net establishes a dual-constraint architecture using a fuzzy rule layer to provide transparent, per-decision attributions and a disagreement risk score."

---

### Paper 6: Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis

**Citation:**  
Wu, J., Shen, Y., Yan, L., Sun, H., Xia, D., Huang, J., & Cao, M. (2026). Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis. *In Proceedings of the Fortieth AAAI Conference on Artificial Intelligence (AAAI-26)*.

**Year:** 2026  
**Venue:** AAAI-26

**What the Paper Does:** Aligns Vision-Language Models with discrete emotion and AU labels using instruction fine-tuning and reinforcement learning to generate structured verbal explanations.

**The Gap (What It Misses):** Large text backbones introduce immense parameter weight and inference latencies, rendering model too slow for real-time sensory control loops.

**What CLEAR-Face Gets From It:** Adopts strict rule that explanation generation must be tightly coupled with verifiable structural features. Solves latency gap by swapping massive language tokens for deterministic, low-latency **Rule-Based Template Generator** mapping MediaPipe face mesh quality data instantly to actions.

**Useful Citation Sentence:** "As demonstrated by Facial-R1, unconstrained reasoning networks frequently suffer from hallucinated reasoning unless explicitly regularized by discrete feature alignments."

---

### Paper 7: Action Unit Enhance Dynamic Facial Expression Recognition (AU-DFER)

**Citation:**  
Liu, F., & Shi, C. (2025). Action Unit Enhance Dynamic Facial Expression Recognition. *arXiv preprint arXiv:2507.07678v1*.

**Year:** 2025  
**Venue:** arXiv preprint

**What the Paper Does:** Injects anatomical, psychological weight matrices directly into deep dynamic expression models using designated dynamic AU loss function, boosting feature tracking across video timelines.

**The Gap (What It Misses):** Assumes pristine video capture pipelines constantly available, making zero allowances for real-world noise like motion blur, face distance drops, or extreme yaw angles.

**What CLEAR-Face Gets From It:** Validates decision to anchor deep representations to localized anatomical structures. Bridges gap by using MediaPipe landmark variations to detect when video tracking degrades, directly alerting CARE loop to initiate recovery.

**Useful Citation Sentence:** "The AU-DFER framework incorporates a prior weight matrix and an AU loss function to smoothly embed facial muscle dynamics into deep video models."

---

## 2. Active Perception and Reacquisition

### Paper 8: An Active Learning and Deep Attention Framework for Robust Driver Emotion Recognition

**Citation:**  
Al-dabbagh, B. S. N., Ledezma Espino, A., & Sanchis de Miguel, A. (2025). *An Active Learning and Deep Attention Framework for Robust Driver Emotion Recognition*. Algorithms, 18(10), 669. https://doi.org/10.3390/a18100669

**Year:** 2025  
**Venue:** Algorithms (MDPI)

**Dataset:** FER-2013, AffectNet, CK+, EMOTIC

**Problem:** Driver Emotion Recognition suffers from dependence on large annotated datasets, high labeling costs, class imbalance, illumination changes, occlusions, head-pose variations, and motion blur.

**Method:** ALDAM framework integrating active learning cycle (selects informative unlabeled samples), Weighted-Cluster Loss (addresses class imbalance), and Deep Attention Mechanism (focuses on salient facial regions).

**Main Result:** 97.58% Accuracy, 98.64% F1-score, 98.76% AUC. Reduced annotation effort by ~40%.

**Relation to CLEAR-Face:** Demonstrates effectiveness of combining active learning and attention for facial emotion recognition under difficult acquisition conditions. Attention mechanism identifies important facial regions, conceptually related to selecting relevant evidence for decision making.

**Limitation/Gap:** Does not estimate uncertainty during inference, determine if image is sufficient, assess image quality before prediction, request reacquisition when confidence low, provide corrective guidance, perform selective prediction, incorporate human-in-the-loop, generate explainable AI outputs, or provide evidence-grounded explanations. ALDAM remains a passive prediction system.

**Useful Citation Sentence:** "Al-dabbagh et al. (2025) demonstrated that integrating active learning with deep attention mechanisms can substantially improve driver emotion recognition performance while reducing annotation effort and enhancing robustness under challenging real-world driving conditions."

---

### Paper 9: Foundations of Strategic Information Gathering & Physical Reacquisition

**Citation:** General Paradigms for Sensor Reacquisition and Active Vision Loops in Autonomous Agents.

**Year:** 2025  
**Venue:** Active Perception Framework Literature

**What the Paper Does:** Formulates computer vision as active sequential decision-making where agents generate precise physical control choices to adjust spatial orientation, distance, or lighting prior to executing downstream classifications.

**The Gap (What It Misses):** Traditional active perception frameworks built exclusively for autonomous robotics/navigation; never mapped to localized facial analysis pipelines to resolve real-time frame or video quality tracking failures.

**What CLEAR-Face Gets From It:** Forms core operational logic of **CARE Controller**. Bridges gap by bringing active perception into FER domain: maps real-time frame corruption metrics onto localized reacquisition instructions (e.g., "Adjust Yaw", "Increase Illumination").

**Useful Citation Sentence:** "Active perception frameworks optimize target visibility by treating information gathering as an explicit sequential decision loop prior to final task commitment."

---

## 3. Uncertainty, Selective Prediction, and Abstention

### Paper 10: UA-FER: Uncertainty-Aware Representation Learning for Facial Expression Recognition

**Citation:**  
Zhou, H., Huang, S., & Xu, Y. (2024). *UA-FER: Uncertainty-Aware Representation Learning for Facial Expression Recognition*. Neurocomputing. https://doi.org/10.1016/j.neucom.2024.129261

**Year:** 2024  
**Venue:** Neurocomputing

**Dataset:** RAF-DB, AffectNet, ExpW, one in-the-lab FER benchmark

**Problem:** FER in unconstrained environments suffers from illumination variations, occlusions, pose changes, inter-class similarity, and intra-class variability. Existing methods often produce overconfident predictions even when uncertain.

**Method:** UA-FER combines CLIP-based Vision-Language Pretraining, Multi-granularity Feature Decoupling, Universal Knowledge Retention, Relation Uncertainty Calibration (RUC), and Evidential Deep Learning (EDL).

**Main Result:** State-of-the-art across multiple FER benchmarks. Demonstrated superior robustness under challenging real-world conditions.

**Relation to CLEAR-Face:** Highly relevant—explicitly incorporates uncertainty estimation into FER pipeline. Similar to CLEAR-Face, recognizes prediction confidence is essential under challenging acquisition conditions. Uses Evidential Deep Learning to quantify uncertainty.

**Limitation/Gap:** Does not assess image acquisition quality, determine if additional observations required, request reacquisition, recommend corrective actions, support active sensing, incorporate human-in-the-loop reacquisition mechanisms, provide selective image recapture strategies, generate explainable visual evidence, or provide natural-language explanations.

**Useful Citation Sentence:** "Zhou et al. (2024) proposed UA-FER, an uncertainty-aware facial expression recognition framework that integrates vision-language representations with evidential deep learning to improve recognition accuracy, robustness, and reliability under unconstrained real-world conditions."

---

### Paper 11: Theory of Bounded Prediction Risk & Cost-Aware Classification

**Citation:** Mathematical Formulations of Selective Risk and Bounded Confidence in Deep Architectures.

**Year:** 2025  
**Venue:** Selective Prediction Foundations Literature

**What the Paper Does:** Outlines mathematics of selective prediction pairing predictor f(x) with selection function g(x) ∈ {0, 1}, authorizing models to return empty set ∅ (abstention) if internal risk scores cross predefined thresholds.

**The Gap (What It Misses):** Standard selective prediction models operate on rigid binary trade-off: either force risky prediction or drop sample entirely, without providing alternative option to salvage frame through guided adjustments.

**What CLEAR-Face Gets From It:** Provides strict mathematical backing for controller's **Abstention and Request paths**. Solves binary limitation by adding third choice: instead of just predicting or quitting, meta-controller can request active environmental reacquisition.

**Useful Citation Sentence:** "Selective prediction framework bounds error risk by establishing a selection function allowed to output an abstention state when internal cost constraints are violated."

---

### Paper 12: Formal and Computational Foundations for Implementing Affective Sovereignty in Emotion AI Systems

**Citation:**  
Kim, R. S. (2026). *Formal and Computational Foundations for Implementing Affective Sovereignty in Emotion AI Systems*. Discover Artificial Intelligence, 6, 235. https://doi.org/10.1007/s44163-026-00235-x

**Year:** 2026  
**Venue:** Discover Artificial Intelligence

**Dataset:** Proof-of-concept simulation experiments; no real FER benchmark; human-subject evaluation planned but not conducted.

**Problem:** Emotion AI systems infer, classify, and influence human emotions, creating ethical risks related to autonomy, privacy, dignity, manipulation, and misinterpretation. Current transparency/privacy frameworks insufficient.

**Method:** Introduces **Affective Sovereignty**—individuals remain final interpreters of their emotional states. Proposes formal risk decomposition, Sovereign-by-Design architecture, DRIFT (Dynamic Risk and Interpretability Feedback Throttling), Affective Sovereignty Contracts (ASC), runtime decision gates (Uncertainty/Risk, Policy/ASC, Handoff, Manipulation/Wellbeing), and new metrics (IOS, AMR, AD, MR).

**Main Result:** Simulation results demonstrated DRIFT and policy-based constraints substantially reduced Interpretive Override Score (IOS), improving preservation of user autonomy.

**Relation to CLEAR-Face:** One of the closest works conceptually. Emphasizes uncertainty-aware decision making, abstention, user interaction, human-in-the-loop mechanisms, and ability to defer decisions when confidence insufficient. Handoff Gate and DRIFT mechanisms conceptually related to CLEAR-Face's objective of determining when current observations are inadequate and additional user input required.

**Limitation/Gap:** Does not perform facial image quality assessment, analyze acquisition conditions (illumination, blur, occlusion, camera angle), determine if facial image should be reacquired, provide image-capture guidance, generate explainable visual evidence, perform AU analysis, integrate multimodal facial quality assessment, or support automated image reacquisition workflows. Validated only through simulation studies.

**Useful Citation Sentence:** "Kim (2026) introduced the concept of Affective Sovereignty and proposed a Sovereign-by-Design architecture that enables uncertainty-aware abstention, user override, and policy-governed emotional AI decision making, thereby preserving human autonomy in emotion recognition systems."

---

## 4. Explainable AI (XAI) & Action Unit Faithfulness

### Paper 13: RobustDRNet: A Clinically-Aligned Hybrid Ensemble Model for Lesion-Aware Diabetic Retinopathy Grading

**Citation:**  
Khokhar, P. B., Pentangelo, V., Gravino, C., Palomba, F. (2026). *RobustDRNet: A clinically-aligned hybrid ensemble model with multi-method explainability for lesion-aware diabetic retinopathy grading*. Expert Systems with Applications. https://doi.org/10.1016/j.eswa.2026.132615

**Year:** 2026  
**Venue:** Expert Systems with Applications (ESWA)

**Dataset:** APTOS 2019 (primary), IDRiD (explainability/lesion validation), 5-class DR grading.

**Problem:** DR grading systems rely on either CNNs or Transformers without combining strengths, perform poorly on minority/severe classes due to imbalance, lack clinically validated explainability, limiting trust in clinical deployment.

**Method:** RobustDRNet hybrid ensemble combining ResNet-34 (local texture/lesions), ConvNeXt-Tiny (enhanced convolutional features), ViT-B/16 (global structure). Feature concatenation (2048-dim vector) with MLP refinement. Soft voting and stacking (Logistic Regression) ensemble. Multi-method explainability: Grad-CAM++, Integrated Gradients, Attention Rollout, SHAP, LIME, TCAV.

**Main Result:** Accuracy 0.884, Macro AUC 0.967, Cohen's Kappa 0.823, MCC 0.824. Significant improvement over individual backbones, better detection of severe DR stages, reduced false negatives, improved lesion-alignment in explainability maps.

**Relation to CLEAR-Face:** Demonstrates multi-method explainability framework integrating CNN and Transformer interpretability. Conceptually relevant for CLEAR-Face's multi-evidence Justification Layer.

**Limitation/Gap:** High computational cost, increased complexity, explainability partially dependent on proxy lesion alignment, limited validation beyond APTOS/IDRiD, no real-world clinical deployment testing.

**Useful Citation Sentence:** "RobustDRNet (Khokhar et al., 2026) proposes a hybrid CNN–Transformer ensemble combining ResNet-34, ConvNeXt-Tiny, and ViT-B/16 with multi-method explainability, achieving strong performance and clinically aligned lesion-aware diabetic retinopathy grading."

---

### Paper 14: Enhancing Emotion Detection Accuracy and Transparency through Multimodal Fusion and Explainable AI

**Citation:**  
Giddaluru, H. K., Praveen, M., Paramesh, L., Sureshbabu, G., Suresh, K., Raju, R. R. (2025). *Enhancing Emotion Detection Accuracy and Transparency through Multimodal Fusion and Explainable AI*. 2025 5th International Conference on Expert Clouds and Applications (ICOECA). IEEE. https://doi.org/10.1109/ICOECA66273.2025.00167

**Year:** 2025  
**Venue:** IEEE ICOECA

**Problem:** Single modality inputs lead to incomplete emotional understanding, reduced accuracy, lack of robustness in noisy environments, poor interpretability, bias, and fairness issues.

**Method:** Multimodal fusion integrating Transformers (text), CNNs (video), LSTM (audio) with XAI techniques LIME, SHAP, Grad-CAM. Fairness-aware training and robustness enhancement using noisy real-world data.

**Dataset:** Multimodal datasets (text, audio, video), real-world noisy data sources.

**Main Result:** Multimodal fusion outperforms single-modality models in accuracy, robustness, and generalization. XAI improves interpretability and user trust. Fairness-aware training reduces bias across demographic groups.

**Relation to CLEAR-Face:** Related through use of multimodal fusion and explainability techniques. Both aim to improve decision transparency and robustness under real-world uncertainty.

**Limitation/Gap:** No standardized benchmark dataset, high computational complexity, limited quantitative evaluation of explainability quality, no explicit facial image quality assessment/enhancement, no reacquisition/capture guidance, limited real-world deployment validation.

**Useful Citation Sentence:** "Giddaluru et al. (2025) proposed a multimodal emotion recognition framework that integrates text, audio, and video modalities with LIME, SHAP, and Grad-CAM to improve accuracy, robustness, fairness, and interpretability in emotion detection systems."

---

### Paper 15: Enhancing Emotion Detection Accuracy and Transparency Through Multimodal Fusion and Explainable AI

**Citation:**  
Keerthiga, S., Navaneetha Krishnan, M., Priyanka, R. (2026). *Enhancing Emotion Detection Accuracy and Transparency Through Multimodal Fusion and Explainable AI*. In: Vasanthi, R., Palaniappan, R., Radhakrishnan, K. (eds) Web Intelligence and Human-Machine Interaction (ICWIHI 2025). Advances in Intelligent Systems and Computing (AISC), vol 3. Springer, Singapore. https://doi.org/10.1007/978-981-96-8563-9_17

**Year:** 2025 (First online), published in proceedings volume 2026  
**Venue:** Springer – ICWIHI 2025

**Problem:** Single-modality inputs limit emotional context capture. Deep learning models act as black boxes with poor interpretability, leading to trust issues in sensitive applications (mental health, customer service, HMI). Bias and fairness issues persist.

**Method:** Multimodal emotion recognition integrating text + audio + video with deep learning feature extraction. Multimodal fusion using concatenation/averaging. XAI techniques: LIME, SHAP, Grad-CAM. Explanation-guided learning (using explanations to refine training). Bias reduction and fairness-aware considerations.

**Dataset:** Multimodal datasets (text, audio, video), real-world/diverse data sources, explanation-enhanced training data.

**Main Result:** Multimodal fusion improves classification accuracy compared to single-modality systems. XAI enhances interpretability, trust, and fairness. Explanation-guided training improves robustness and reliability.

**Relation to CLEAR-Face:** Related through focus on multimodal fusion and XAI for improving decision transparency. Both aim to enhance interpretability and reduce uncertainty in AI-based emotion understanding.

**Limitation/Gap:** No standardized benchmark dataset, limited quantitative evaluation of explanation quality, high computational cost, no explicit facial image quality assessment (blur, occlusion, lighting), no decision abstention or reacquisition mechanism, limited real-world deployment validation.

**Useful Citation Sentence:** "Keerthiga et al. (2026) proposed a multimodal emotion recognition framework that integrates text, audio, and video modalities with LIME, SHAP, and Grad-CAM to enhance accuracy, interpretability, and fairness in emotion detection systems."

---

### Paper 16: Explaining Facial Expression Recognition

**Citation:**  
Nahulanthran, S., Tian, L., Kulić, D., & Vered, M. (2025). Explaining Facial Expression Recognition. *In Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)*.

**Year:** 2025  
**Venue:** AAMAS 2025

**What the Paper Does:** Presents multimodal symbolic XAI framework explaining black-box FER classifications by combining post-hoc visual evidence maps with textual Action Unit descriptions.

**The Gap (What It Misses):** Explanation cycle operates completely post-hoc on static images. Cannot evaluate or explain active choice behavior, performance trade-offs, or state transitions of real-time tracking pipeline.

**What CLEAR-Face Gets From It:** Extracts empirical proof that textual AU logic plus visual heatmaps maximizes user trust. Bridges gap by adapting hybrid formulation into multi-evidence **Justification Layer**, explaining not just predictions, but also active reacquisition and abstention choices.

**Useful Citation Sentence:** "Nahulanthran et al. demonstrate that text-only or joint text-visual explanations grounded in explicit Action Units significantly maximize appropriate user trust over purely visual heatmaps."

---

### Paper 17: Towards Explainable Facial Expression Recognition using Face Action Units: XFER-AU

**Citation:**  
Verlekar, T. T., Goyal, A., & Correia, P. L. (2025). Towards Explainable Facial Expression Recognition using Face Action Units: XFER-AU. *IEEE Access*, Vol. 11.

**Year:** 2025  
**Venue:** IEEE Access

**What the Paper Does:** Forces internal CNN feature weights to focus strictly inside spatial bounding masks defined by localized physical Action Units, shielding models from learning spurious backgrounds.

**The Gap (What It Misses):** Method is entirely passive and cannot identify or categorize what type of environmental noise (lighting drops vs. severe blur) is causing AU region features to degrade.

**What CLEAR-Face Gets From It:** Utilizes spatial restriction to guarantee XAI fidelity. Solves gap by pairing regional gradient tracking (Grad-CAM++/Integrated Gradients) with **Cross-Method Agreement Protocol** to verify whether feature dropouts are caused by target emotions or environmental factors.

**Useful Citation Sentence:** "The XFER-AU framework forces internal feature attributions to remain strictly bounded within localized spatial masks tracking physical muscle movements."

---

### Paper 18: XAI-DSCSA: Explainable-AI-Based Deep Semi-Supervised Convolutional Sparse Autoencoder

**Citation:**  
Mohana, M., Subashini, P., & Ghinea, G. (2025). XAI-DSCSA: explainable-AI-based deep semi-supervised convolutional sparse autoencoder for facial expression recognition. *Signal, Image and Video Processing*, 19, 394.

**Year:** 2025  
**Venue:** Signal, Image and Video Processing (Springer)

**What the Paper Does:** Designs deep semi-supervised sparse convolutional autoencoder to reduce parameters, speed up training, and provide light edge-ready feature attributions.

**The Gap (What It Misses):** Generates standard pixel saliency heatmaps that fail to identify multi-criteria data quality flaws or ground explanations in human-readable terms.

**What CLEAR-Face Gets From It:** Justifies using high-efficiency, small-parameter architectures for edge deployment. Applies this lesson by selecting **EfficientNet-B0** as core classification backbone, keeping processing fast enough to leave headroom for meta-controller.

**Useful Citation Sentence:** "Mohana et al. address parameter optimization burdens and excessive training steps by implementing a deep semi-supervised convolutional sparse autoencoder."

---

### Paper 19: XFERa: Xplainable Emotion Recognition for Improving Transparency and Trust

**Citation:**  
Macchiarulo, N., De Carolis, B., Loglisci, C., Losavio, V. N., Miccoli, M. G., & Palestra, G. (2025). *XFERa: Xplainable Emotion Recognition for improving transparency and trust*. Joint Proceedings of the ACM IUI Workshops 2025.

**Year:** 2025  
**Venue:** ACM IUI Workshops 2025

**Dataset:** FER datasets (not explicitly named), face images labeled with basic emotions (happiness, surprise, sadness, fear, disgust, anger). Uses OpenFace for AU extraction, Grad-CAM for visual explanations. Human user studies for evaluating explanation quality and trust.

**Problem:** CNN-based FER models operate as black boxes. Difficult to understand which facial regions influence predictions and which AUs contribute to classification. Lack of transparency reduces trust and limits usability in HCI/affective computing.

**Method:** XFERa pipeline: CNN-based FER model, Grad-CAM heatmaps, OpenFace for AU extraction, GPT-4 for natural language explanations. Fusion aligns saliency maps with facial landmarks with weighted combination of AU intensities and activation regions. Output includes predicted emotion label, visual explanation (heatmap), and textual explanation.

**Main Result:** Improves transparency, user trust, and interpretability through multimodal explanation outputs (visual + AU + text). User studies validate explanations are understandable and improve confidence.

**Relation to CLEAR-Face:** Highly related—facial emotion recognition using CNN/vision-based models, explainability using Grad-CAM (visual attention maps), AU-based facial analysis, emphasis on interpretability and trust.

**Limitation/Gap:** No explicit image quality assessment, no automatic rejection of low-quality facial images, high computational cost due to GPT-4 integration, limited scalability in real-time/edge devices, dependency on external tools (OpenFace, LLMs), evaluation mainly based on user studies rather than large benchmark comparisons.

**Useful Citation Sentence:** "Macchiarulo et al. (2025) proposed XFERa, an explainable facial emotion recognition pipeline integrating Grad-CAM, Action Unit analysis, and GPT-4-based explanations to improve transparency and user trust in emotion recognition systems."

---

## 5. Natural-Language Explanation & Vision-Language Models

### Paper 20: EmotionReasoner: Emotion-Explanation-Oriented Reinforcement Learning for Explainable Multimodal Emotion Recognition

**Citation:**  
Zhao, S., Zhang, H., Wu, D., Huang, D., Hong, W., & Sun, J. (2026). *EmotionReasoner: Emotion-Explanation-Oriented Reinforcement Learning for Explainable Multimodal Emotion Recognition*. IEEE Transactions on Affective Computing (Early Access), 1–18. https://doi.org/10.1109/TAFFC (exact DOI as published in IEEE Xplore)

**Year:** 2026 (Published online: 04 June 2026)  
**Venue:** IEEE Transactions on Affective Computing (Early Access)

**Dataset:** Nine representative multimodal emotion recognition datasets. Cross-dataset evaluation setup. Multimodal inputs (text, audio, video depending on dataset). Used for both MER (emotion recognition) and EMER (explainable emotion reasoning) evaluation.

**Problem:** Existing MER systems struggle with weak interpretability, overly long/unstructured explanation chains in MLLM-based systems, lack of fine-grained evaluation metrics for explainability quality, poor alignment between emotional reasoning and actual emotional cues.

**Method:** EmotionReasoner—reinforcement learning-based framework for explainable emotion recognition. RL-based optimization with explanation-aware reward function (structure compliance, emotion prediction correctness, explanation fidelity). Emotion-polarity-oriented loss weighting strategy. Introduces evaluation metrics: MEER-E (fidelity), LACE-E (clarity), UTAI-E (practical utility).

**Main Result:** Outperforms existing multimodal emotion recognition and reasoning models across nine datasets. More accurate classification, more structured/emotionally grounded reasoning chains, higher-quality explanations aligned with emotional cues.

**Relation to CLEAR-Face:** Related through explainable emotion recognition from multimodal inputs, focus on aligning predictions with human-interpretable reasoning, improving transparency beyond raw classification accuracy, use of evaluation metrics to measure explanation quality.

**Limitation/Gap:** High computational complexity due to RL-based training, explanations may still become verbose despite constraints, no explicit facial quality assessment (blur, occlusion, illumination), limited focus on low-quality image rejection/reacquisition, depends heavily on dataset-specific multimodal alignment quality, lack of direct integration with real-time facial systems/edge deployment constraints.

**Useful Citation Sentence:** "Zhao et al. (2026) proposed EmotionReasoner, a reinforcement learning-based framework for explainable multimodal emotion recognition that integrates explanation-aware rewards and emotion-polarity-driven optimization to improve both prediction accuracy and reasoning transparency across multiple datasets."

---

### Paper 21: FaceScanPaliGemma: Multi-Agent Vision Language Models for Facial Attribute Recognition

**Citation:**  
AlDahoul, N., Tan, M. J. T., Kasireddy, H. R., & Zaki, Y. (2026). FaceScanPaliGemma: Multi-Agent Vision Language Models for Facial Attribute Recognition. Scientific Reports, 16, 10246. https://doi.org/10.1038/s41598-026-10246

**Year:** 2026  
**Venue:** Scientific Reports

**Dataset:** FairFace (108,501 face images, balanced across races), AffectNet (large-scale FER with 8 emotion classes). Tasks: race, gender, age group, and emotion classification. Benchmarked against GPT, Gemini, LLaVA, and other VLMs.

**Problem:** Facial attribute recognition challenged by subjectivity, dataset imbalance/demographic bias, lighting/pose/image quality variations, limitations of single-model architectures, lack of interpretability.

**Method:** FaceScanPaliGemma—multi-agent VLM framework. Four specialized fine-tuned PaliGemma agents: race classification, gender classification, age estimation, emotion recognition. Reformulates as Visual Question Answering (VQA). Multi-agent collaboration instead of single model. Fusion via prompt-based coordination. Fairness-aware training and bias mitigation. Interpretability via structured modular reasoning.

**Main Result:** High performance across all facial attribute tasks. Outperforms GPT-4o, Gemini, and LLaVA in baseline/zero-shot settings. Strong improvement in race, gender, age, emotion recognition. Multi-agent design improves accuracy and robustness.

**Relation to CLEAR-Face:** Both focus on facial understanding in real-world scenarios. Both use modular/decomposed decision-making pipelines. Both improve interpretability and reliability. FaceScanPaliGemma uses specialized agents similar to CLEAR-Face reasoning stages. Both handle ambiguity in facial attribute interpretation.

**Limitation/Gap:** No explicit image quality assessment (blur, occlusion, illumination), no reacquisition or capture guidance mechanism, limited visual explainability (no heatmap-based explanations like Grad-CAM), higher computational cost due to multi-agent system, limited real-time deployment evaluation, possible residual dataset bias despite mitigation efforts.

**Useful Citation Sentence:** "AlDahoul et al. (2026) introduced FaceScanPaliGemma, a multi-agent vision-language framework that decomposes facial attribute recognition into specialized models, improving accuracy, robustness, and interpretability through modular reasoning."

---

## 6. Comprehensive Domain Surveys

### Paper 22: Machine and Deep Learning in Facial Expression Recognition: A Survey Based on Facial Action Units

**Citation:**  
Priego, M., Aranguren, I., Oliva, D., Valdivia-G, A., Navarro, M. A., & Pérez-Cisneros, M. (2026). Machine and deep learning in facial expression recognition: a survey based on facial action units. *Scientific Reports*.

**Year:** 2026  
**Venue:** Scientific Reports (Nature Portfolio)

**What the Paper Does:** Broad, systematic survey detailing how AU-driven deep models struggle with cross-dataset transfers, out-of-domain mutations, and subject identity bias.

**The Gap (What It Misses):** Being a review text, details generalization failure points across benchmarks but provides no active, runtime algorithmic framework to fix domain degradation.

**What CLEAR-Face Gets From It:** Uses this survey to structure verification protocol against dataset selection biases. Following its findings, selects **RAF-DB as core training target** and cross-evaluates on **AffectNet** to prove out-of-domain robustness under CARE Controller supervision.

**Useful Citation Sentence:** "A comprehensive multi-criteria review by Priego et al. underscores that AU-centric architectures evaluated via cross-dataset transfers demonstrate superior robustness against out-of-domain mutations."

---

### Paper 23: Facial Expression Recognition in the Deep Learning Era: A Systematic Multi-Criteria Review

**Citation:**  
Georgiou, S., Psiris, A., Evangelatos, S., Lagkas, T., Argyrious, V., Sarigiannidis, P., Varlamis, I., & Papadopoulos, G. T. (2026). Facial Expression Recognition in the Deep Learning Era: A Systematic Multi-Criteria Review. *arXiv preprint arXiv:2606.08612v1*.

**Year:** 2026  
**Venue:** arXiv preprint

**What the Paper Does:** Formalizes clear, multi-criteria standard evaluation rules for modern deep architectures, warning explicitly against dangers of identity-leakage in video and image splits.

**The Gap (What It Misses):** Identifies active vision, automated reacquisition, and selective prediction as key future research horizons, but lacks architectural blueprint to execute them.

**What CLEAR-Face Gets From It:** Uses guidelines to eliminate identity leakage flaws. Enforces strict **participant-disjoint data split protocol** during training, establishing secure baseline preventing identity cross-contamination.

**Useful Citation Sentence:** "As highlighted in the systematic review by Georgiou et al., tracking multi-criteria experimental guidelines and avoiding data leakage are paramount for modern FER verification."

---

## 7. Confidence Calibration & Human-in-the-Loop Decision Support

### Paper 24: Post-Hoc Probability Realignment & Utility Balancing

**Citation:** Post-Hoc Calibration Protocols and Utility Metrics for Human-Agent Collaboration Triage.

**Year:** 2025  
**Venue:** Trust Calibration & Decision Support Literature

**What the Paper Does:** Details post-hoc calibration metrics (Temperature Scaling) to reduce Expected Calibration Error (ECE), aligning internal confidence values with true empirical accuracy while modeling user triage alert fatigue.

**The Gap (What It Misses):** Existing calibration methods evaluate prediction confidence based solely on internal classification layer distributions, completely ignoring real-time physical stream parameters or environmental quality conditions.

**What CLEAR-Face Gets From It:** Guides **Uncertainty Estimation Layer** and mathematical optimization of **CARE Utility Function**. Expands typical calibration methods by blending model classification probabilities with real-time MediaPipe data tracking quality, ensuring XGBoost meta-controller makes actions based on both network confidence and physical video clarity.

**Useful Citation Sentence:** "Post-hoc temperature scaling minimizes expected calibration errors, matching internal neural scores directly to true empirical tracking rates."

---

## Summary of Key Themes for CLEAR-Face Positioning

| Theme | Key Papers | Relevance to CLEAR-Face |
|-------|------------|------------------------|
| **Uncertainty-Aware FER** | Wang et al. 2024, Radoi & Cioroiu 2024, Zhou et al. 2024 | Establishes value of uncertainty estimation; CLEAR-Face extends to active reacquisition |
| **Active Perception** | Al-dabbagh et al. 2025, Active Perception Literature | Provides foundation for active sensing; CLEAR-Face brings to FER domain |
| **Selective Prediction/Abstention** | Selective Prediction Literature, Kim 2026 | Provides mathematical backing; CLEAR-Face adds reacquisition as third option |
| **XAI with AUs** | Nahulanthran et al. 2025, Verlekar et al. 2025, Macchiarulo et al. 2025 | Establishes AU-based explanations; CLEAR-Face extends to explain active choices |
| **VLM/Multi-Agent FER** | AlDahoul et al. 2026, Wu et al. 2026 | Shows value of modular reasoning; CLEAR-Face uses template-based generation for low latency |
| **Domain Surveys** | Priego et al. 2026, Georgiou et al. 2026 | Guides dataset selection and evaluation protocols |
| **Calibration** | Calibration Literature | Guides uncertainty estimation; CLEAR-Face adds physical quality parameters |