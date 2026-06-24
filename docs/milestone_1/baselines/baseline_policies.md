# Baseline Policies 
 - Direct prediction: always predict from the initial 
observation. - Immediate abstention: always abstain. - Generic retry: always request another capture without 
diagnosing the failure. - Random valid action: randomly choose one valid action. - Fixed quality rules: use hand-written thresholds for 
lighting, blur, pose, and face size. - Uncertainty-only policy: request correction or abstain 
based only on uncertainty. - CARE without costs: CARE without burden and latency 
terms. - CARE without XAI signals: CARE using uncertainty and 
image quality but no XAI-derived signals. - Full CARE: proposed risk-burden-latency controller. - Oracle: retrospective upper bound using observed action 
outcomes.