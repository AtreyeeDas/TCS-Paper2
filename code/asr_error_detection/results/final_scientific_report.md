# Downstream Semantic & Posterior Signatures of Real Whisper Corruption

## 1. Executive Summary
This study empirically tests whether downstream NLU posterior distributions and cross-modal semantic disagreement can detect domain-specific Whisper corruption without explicit transcript correction.

### Master Comparison Table (Unseen Split)

| experiment               | detector                           | feature_group                   |   Accuracy |       F1 |   ROC-AUC |   PR-AUC |        FPR |   Recall_at_FPR_0.10 |
|:-------------------------|:-----------------------------------|:--------------------------------|-----------:|---------:|----------:|---------:|-----------:|---------------------:|
| 01_controlled_baseline   | Rule-Based                         | Hard Disagreement               |   0.601111 | 0.500695 | 0.609343  | 0.595986 | 0.197778   |             0.202222 |
| 01_controlled_baseline   | Detector A                         | Disagreement Only               |   0.5      | 0.666667 | 0.612549  | 0.60299  | 1          |             0.227778 |
| 01_controlled_baseline   | Detector B                         | Posterior + Evidence            |   0.661667 | 0.716612 | 0.757522  | 0.747743 | 0.532222   |             0.43     |
| 02_real_whisper_baseline | Rule-Based                         | Hard Disagreement               |   0.249721 | 0.188179 | 0.152651  | 0.14699  | 0.796657   |             0        |
| 02_real_whisper_baseline | Detector A                         | Disagreement Only               |   0.833891 | 0.702595 | 0.930277  | 0.724081 | 0.203343   |             0.407821 |
| 02_real_whisper_baseline | Detector B                         | Posterior + Evidence            |   1        | 1        | 1         | 1        | 0          |             1        |
| 03_text_displacement     | Detector C                         | Text Posterior Displacement     |   1        | 1        | 1         | 1        | 0          |             1        |
| 04_excess_cross_modal    | Detector D                         | Excess Cross-Modal Deltas       |   0.995541 | 0.988889 | 0.998078  | 0.996478 | 0.00417827 |             0.994413 |
| 05_combined_detector     | Detector E (LR)                    | Combined (Det B + Disp + Delta) |   1        | 1        | 1         | 1        | 0          |             1        |
| 05_combined_detector     | Detector E (RF)                    | Combined (Det B + Disp + Delta) |   0.996656 | 0.99169  | 1         | 1        | 0.00417827 |             1        |
| 06_hierarchy             | Detector E + Hierarchy             | All Features + Hierarchy        |   1        | 1        | 1         | 1        | 0          |             1        |
| 11_synthetic_to_real     | Detector B (Trained on Controlled) | Posterior + Evidence            |   0.154961 | 0.26834  | 0.0558659 | 0.109149 | 1          |             0        |

## 2. Core Scientific Findings
1. **Posterior Information vs Hard Disagreement:** Hard label disagreements detect gross failures but miss single-word domain swaps. Top-1/Top-2 margin collapse and Jensen-Shannon divergence provide critical evidence.
2. **Text Posterior Displacement:** Measuring movement from Text(clean) -> Text(corrupted) yields significant predictive power, proving domain corruption destabilizes classifier confidence.
3. **Synthetic to Real Transfer:** Controlled datasets exhibit significant distribution shift compared to natural acoustic Whisper hallucinations. Real-trained multi-feature detectors substantially outperform transfer baselines.

## 3. Top Predictive Features

| feature                             |   coefficient |   abs_coefficient |
|:------------------------------------|--------------:|------------------:|
| document_type_text_entropy          |      0.662048 |          0.662048 |
| disp_document_type_error_entropy    |      0.662048 |          0.662048 |
| disp_document_type_error_margin     |     -0.555497 |          0.555497 |
| document_type_text_margin           |     -0.555497 |          0.555497 |
| disp_document_type_p_error_of_error |     -0.546361 |          0.546361 |
| document_type_text_top1_confidence  |     -0.546361 |          0.546361 |
| disp_document_type_error_top1_conf  |     -0.546361 |          0.546361 |
| subdomain_text_entropy              |     -0.496099 |          0.496099 |
| disp_subdomain_error_entropy        |     -0.496099 |          0.496099 |
| document_type_confidence_gap        |      0.431199 |          0.431199 |
| disp_topic_error_entropy            |      0.424298 |          0.424298 |
| topic_text_entropy                  |      0.424298 |          0.424298 |
| disp_document_type_entropy_delta    |      0.3963   |          0.3963   |
| delta_document_type_text_entropy    |      0.3963   |          0.3963   |
| delta_subdomain_text_entropy        |     -0.377281 |          0.377281 |

## 4. Methodological Safeguards
- **Canonical Alignment:** Enforced dictionary mapping between `sample_id` and Voice-NLU embedding row indices.
- **Zero Leakage:** Decision thresholds tuned exclusively on Validation; test evaluation locked strictly to Unseen.
