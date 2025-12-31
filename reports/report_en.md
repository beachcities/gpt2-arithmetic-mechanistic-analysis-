# Arithmetic Without Algorithms: Mechanistic Analysis of Arithmetic Mismatch in GPT-2 Small
### — Causal Analysis of Induction Heads and SAE Features —

**Author:** Masayuki Yamada  
**Date:** January 1, 2026  
**Affiliation:** Applied Large Language Model Course, Matsuo-Iwasawa Lab, The University of Tokyo

---

## 1. Introduction

### 1.1 Background: Inductive Bias and Arithmetic Incompatibility
Large Language Models (LLMs) are optimized for the objective of "next-token prediction," making them adept at mimicking statistical patterns within a context. However, arithmetic operations involve strict structural dependencies—such as digit alignment and carry-overs—which are inherently incompatible with the **inductive bias of "context-dependent pattern matching"** possessed by LLMs.

Instead of dismissing arithmetic errors as mere lack of training, we employ **Mechanistic Interpretability** to causally explain what heuristic circuits the model uses internally to attempt (and fail at) these tasks.

**Why GPT-2 Small?** We utilize GPT-2 Small as a "model organism" for this study due to its manageable size, fully transparent architecture, and the extensive body of prior mechanistic interpretability research available for reference.

### 1.2 Objective: Identifying the Mechanism Behind "5 + 5 = 6"
This report investigates the phenomenon where GPT-2 Small consistently answers "6" to the prompt "5 + 5 =". We demonstrate that this is not a random error but a mechanistically explainable fallacy caused by the **inappropriate coupling of "Induction Heads" (context copying) and "Sequence Features" (sequence completion)**.

## 2. Experimental Setup and Reproducibility

To ensure reproducibility, the analysis was conducted under the following environment:

* **Model:** `gpt2-small` (12 layers, 12 heads, 117M params)
* **Compute:** NVIDIA A100-SXM4-40GB (Google Colab)
* **Libraries:** `transformer_lens` (v2.16.1), `SAELens` (v0.2.0), `PyTorch` (v2.9.0+cu126)
* **Tokenizer:** GPT-2 byte-level BPE
* **Prompt Template:** Few-shot format `1 + 1 = 2. 2 + 2 = 4. 5 + 5 =`
* **Data Availability:** Minimal working code is provided in Appendix A.

## 3. Experimental Protocols

To verify the causality of the error mechanism, we conducted ablation (intervention) experiments based on the following hypotheses. The results are summarized in Table 1.

**Table 1: Intervention Results for Hypothesis Verification (N=200 seeds)**

| Hypothesis | Intervention | Expected Outcome | Observed Result | Effect Size |
| :--- | :--- | :--- | :--- | :--- |
| **A: L8H6 copies context** | Head Ablation (Zeroing L8H6 output) | Context "4" is blocked, reducing "6" logit | **Error Prob: -8.6 pt**<br>(p < 0.001) | Large |
| **B: Feature #1076 completes sequence** | Feature Ablation (Zeroing SAE feature) | Prediction "Next is 6" is suppressed | **Error Prob: -11.9 pt**<br>(p < 0.0001) | Very Large |

![Figure 4: Effect of Causal Interventions](../figures/figure4_intervention_effect.png)
> **Figure 4: Effect of Causal Interventions (N=200)**
> Compared to the Baseline (23.1%), ablating L8H6 reduced the error rate to 14.5%, and ablating the SAE feature reduced it to 11.2%. This confirms both circuits are causal factors in generating the error.

## 4. Analysis Results

### 4.1 Knowledge State Scan and Layer-wise Trace
Analysis using **Logit Lens** revealed that while input tokens dominate layers 0-7, **the probability of "6" spikes sharply between Layers 8 and 9**.
In contrast, the probability of the correct answer "10" remains below 5.6% across all layers, suggesting the model is not performing arithmetic calculation.

![Figure 1: Probability Trace](../figures/figure1_probability_trace.png)
> **Figure 1: Probability Trace of Answer Candidates**
> The X-axis represents the Transformer layer (0–11), and the Y-axis represents the token probability estimated by Logit Lens. The probability of "6" spikes at Layers 8–9, reflecting the activation of specific prediction circuits (Induction Head → Sequence Feature) rather than arithmetic computation.

### 4.2 Source Identification: Induction Head Behavior of L8H6
Attention Pattern Analysis identified **Layer 8 Head 6 (L8H6)** as exhibiting distinct behavior. This head ignores the current input "5" (Score: 0.0008) and strongly attends to the answer of the preceding equation, "4" (Score: 0.6480). This is a typical behavior of an **Induction Head**.

![Figure 2: Attention Heatmap](../figures/figure2_attention_heatmap.png)
> **Figure 2: Attention Pattern of Layer 8 Head 6 (L8H6)**
> The heatmap shows attention scores (Vertical: Query, Horizontal: Key). L8H6 focuses intensely on the previous answer "4", providing direct evidence of context copying.

### 4.3 Mechanism Identification: Feature Analysis with SAE
Decomposing the Layer 9 MLP using a **Sparse Autoencoder (SAE)** identified **Feature #1076** firing at an abnormally high value (Activation: 14.00).
This feature was found to promote tokens like "6" or "546" (sequence continuation) upon receiving "4", while **actively suppressing the correct answer "10"**.

![Figure 3: SAE Analysis](../figures/figure3_sae_analysis.png)
> **Figure 3: Activation of SAE Feature #1076**
> The left chart shows the high activation of Feature #1076.

![Figure 9: Feature Contribution](../figures/figure9_feature_contribution.png)
> **Figure 9: Logit Contribution of Feature #1076**
> This feature strongly promotes the error "6" (Promotion) while providing a negative contribution to the correct answer "10" (Suppression). This indicates that the model is structurally inhibited from accessing the correct answer.

## 5. Robustness Checks

We performed multiple validations to confirm universality and rule out alternative hypotheses.

### 5.1 Tokenization Sensitivity
We verified whether the difference between " 5" (with space) and "5" (without space) affected the results. No significant difference was observed in Attention patterns or SAE activation.

![Figure 7: Tokenization Sensitivity](../figures/figure7_tokenization_sensitivity.png)
> **Figure 7: Robustness to Tokenization**
> Regardless of spacing, the Attention (top) and SAE Feature #1076 activation (bottom) remain consistent, proving this phenomenon is not a tokenizer artifact.

### 5.2 Excluding Frequency Bias
To address the hypothesis that "6" is output simply due to high corpus frequency, we compared it with a Zero-shot (no context) setting.

![Figure 8: Frequency Bias](../figures/figure8_frequency_bias.png)
> **Figure 8: Rejecting the Frequency Bias Hypothesis**
> While there is no significant difference in corpus frequency (A), the generation probability (B) spikes for "6" only in the Few-shot context. This proves context, not frequency, is the cause.

### 5.3 Structural Failure: Carry-over Tasks
In tasks requiring carry-over, such as "9 + 9 =", the model fails to calculate and relies on context matching, resulting in structural failure.

![Figure 6: Carry-over Analysis](../figures/figure6_carry_over_analysis.png)
> **Figure 6: Failure Analysis across Arithmetic Contexts**
> In tasks involving carry-over, the error rate reaches 88.0%, demonstrating the model's algorithmic bankruptcy.

### 5.4 Case Comparison (Control Experiment)
We compared the internal behavior of error cases versus control (no-error) cases.

![Figure 5: Case Comparison](../figures/figure5_case_comparison.png)
> **Figure 5: Comparison of Mechanisms in Wrong vs. Control Cases**
> In the error case (Left), the Induction Head and Feature #1076 are active. In the control case (Right), these circuits remain inactive, visually confirming the causal link.

## 6. Conclusion and Future Work

### 6.1 "Arithmetic Without Algorithms"
This investigation mechanistically explains the "5+5=6" phenomenon in GPT-2 Small through the **inappropriate interaction of two heuristics: "Context Copying by Induction Heads" and "Sequence Completion by SAE Features."** The model substitutes superficial pattern matching for arithmetic algorithms.

![Figure 10: Causal Circuit Diagram](../figures/figure10_causal_circuit.png)
> **Figure 10: Causal Circuit Diagram**
> The overall causal chain from input context to error output (Induction → SAE → Output).

### 6.2 Future Directions
These findings can be applied to **Circuit Steering** engineering—specifically, detecting and suppressing hallucination-inducing features (like Feature #1076) during inference to improve model reliability.

## Acknowledgments
This research was conducted as a final project for the **Applied Large Language Model Course** at the **Matsuo-Iwasawa Lab, The University of Tokyo**.
I would like to express my gratitude to the lecturers and teaching assistants for their valuable feedback and educational resources.
---

### Appendix A: Minimal Working Example

Please refer to the following notebook for reproduction code:
[notebooks/experiment.ipynb](../notebooks/experiment.ipynb)
