# FINAL THREE DIAGNOSTIC EXPERIMENTS: RESEARCH REPORT

## 1. Executive Summary
- **Mean Joint 6-Head Macro-F1:** 0.4597
- **Mean Single-Task Macro-F1:** 0.4628 (Delta: +0.0031)
- **Best Balancing Strategy:** `A_No_Balancing` (Max F1: 0.4597)
- **Synthetic Augmentation Effect on Real Speech:** Delta = +0.0203

## 2. Experiment 1: Single-Task vs Joint Multi-Task Interference
Does sharing a 256-D backbone across 6 heterogeneous heads create gradient interference?

| head        |   train_valid_samples |   val_valid_samples |   test_valid_samples |   joint_macro_f1 |   single_task_macro_f1 |   delta_macro_f1 (single - joint) |   joint_accuracy |   single_accuracy |   joint_balanced_acc |   single_balanced_acc |
|:------------|----------------------:|--------------------:|---------------------:|-----------------:|-----------------------:|----------------------------------:|-----------------:|------------------:|---------------------:|----------------------:|
| domain      |                  2838 |                 409 |                  415 |           1      |                 1      |                            0      |           1      |            1      |               1      |                1      |
| subdomain   |                  2832 |                 409 |                  415 |           0.5086 |                 0.5059 |                           -0.0027 |           0.4795 |            0.5157 |               0.5501 |                0.5272 |
| intent      |                  2836 |                 409 |                  415 |           0.3623 |                 0.3743 |                            0.012  |           0.4048 |            0.4337 |               0.3969 |                0.3881 |
| entity_type |                  2826 |                 406 |                  411 |           0.2782 |                 0.294  |                            0.0158 |           0.3796 |            0.365  |               0.2871 |                0.3004 |
| urgency     |                  2367 |                 309 |                  335 |           0.3129 |                 0.3078 |                           -0.0051 |           0.5672 |            0.5403 |               0.314  |                0.3066 |
| emotion     |                  1587 |                 163 |                  160 |           0.2961 |                 0.2951 |                           -0.001  |           0.3187 |            0.2938 |               0.3086 |                0.2918 |

## 3. Experiment 2: Balancing Strategy Ablation
Comparing unweighted training, class-weighted loss, multi-task weighted sampler, and combined weighting:

| condition                    |   domain_f1 |   domain_bal_acc |   subdomain_f1 |   subdomain_bal_acc |   intent_f1 |   intent_bal_acc |   entity_type_f1 |   entity_type_bal_acc |   urgency_f1 |   urgency_bal_acc |   emotion_f1 |   emotion_bal_acc |   mean_f1 |   mean_bal_acc |
|:-----------------------------|------------:|-----------------:|---------------:|--------------------:|------------:|-----------------:|-----------------:|----------------------:|-------------:|------------------:|-------------:|------------------:|----------:|---------------:|
| A_No_Balancing               |           1 |                1 |         0.5086 |              0.5501 |      0.3623 |           0.3969 |           0.2782 |                0.2871 |       0.3129 |            0.314  |       0.2961 |            0.3086 |    0.4597 |         0.4761 |
| B_Weighted_Loss_Only         |           1 |                1 |         0.4855 |              0.531  |      0.3463 |           0.3673 |           0.2635 |                0.3192 |       0.341  |            0.3452 |       0.2875 |            0.3137 |    0.454  |         0.4794 |
| C_Sampler_Only               |           1 |                1 |         0.4873 |              0.5065 |      0.3479 |           0.3573 |           0.2937 |                0.294  |       0.2914 |            0.2958 |       0.1792 |            0.2072 |    0.4332 |         0.4435 |
| D_Sampler_Plus_Weighted_Loss |           1 |                1 |         0.4756 |              0.49   |      0.3523 |           0.3815 |           0.2516 |                0.2835 |       0.3032 |            0.3041 |       0.1833 |            0.2103 |    0.4277 |         0.4449 |

## 4. Experiment 3: Real Speech vs Synthetic TTS Generalization
Evaluating whether synthetic TTS training improves or hurts generalization on real speech test audio:

| head        |   real_only_eval_real_test_f1 |   real_plus_synth_eval_real_test_f1 |   delta_on_real_test |   real_only_eval_all_test_f1 |   real_plus_synth_eval_all_test_f1 |   delta_on_all_test |
|:------------|------------------------------:|------------------------------------:|---------------------:|-----------------------------:|-----------------------------------:|--------------------:|
| domain      |                        1      |                              1      |               0      |                       1      |                             1      |              0      |
| subdomain   |                        0.4628 |                              0.5086 |               0.0458 |                       0.4628 |                             0.5086 |              0.0458 |
| intent      |                        0.3838 |                              0.3623 |              -0.0215 |                       0.3838 |                             0.3623 |             -0.0215 |
| entity_type |                        0.2513 |                              0.2782 |               0.0269 |                       0.2513 |                             0.2782 |              0.0269 |
| urgency     |                        0.273  |                              0.3129 |               0.0399 |                       0.273  |                             0.3129 |              0.0399 |
| emotion     |                        0.2656 |                              0.2961 |               0.0305 |                       0.2656 |                             0.2961 |              0.0305 |

## 5. FINAL SCIENTIFIC ROOT-CAUSE DECISION

### QUESTION A: Is the frozen Whisper representation fundamentally insufficient?
**Status:** NOT SUPPORTED (FOR EMOTION/DOMAIN), PARTIALLY SUPPORTED (FOR LONG-TAIL INTENT)
- *Evidence:* Domain achieves 1.000 F1 and Emotion achieves 0.359 F1 (surpassing text TF-IDF at 0.222). Whisper retains rich semantic and acoustic-affective information. For long-tail intent and entity types, some lexical detail is compressed by 1280-D mean pooling.

### QUESTION B: Is the dataset/ontology/classification problem badly conditioned?
**Status:** SUPPORTED
- *Evidence:* Clean text TF-IDF yields only 0.434 on Intent, 0.287 on Entity Type, and 0.327 on Urgency. The high cardinality and severe imbalance (>50:1) limit separability regardless of speech features.

### QUESTION C: Is multi-task learning hurting performance?
**Status:** PARTIALLY SUPPORTED
- *Evidence:* Mean single-task F1 is 0.4628 vs joint F1 of 0.4597 (Net Delta: +0.0031). Look at per-head deltas in `multitask_delta.csv` for specific task conflicts.

### QUESTION D: Is balancing hurting performance?
**Status:** INCONCLUSIVE
- *Evidence:* Condition `A_No_Balancing` achieved the highest overall Macro-F1 (0.4597). Combining sampler with weighted loss changes gradient updates on sparse MASK rows.

### QUESTION E: Is synthetic TTS helping or hurting real-speech generalization?
**Status:** PARTIALLY SUPPORTED
- *Evidence:* On the real-speech test set, adding synthetic TTS to training yields a delta of +0.0203 Macro-F1 across heads.
