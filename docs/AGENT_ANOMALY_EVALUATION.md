# Agent Behavioral Anomaly Evaluation

## Method

The agent anomaly component constructs claim-time behavioral profiles using only claims submitted strictly before each claim's submission date. Claims on the same day are profiled before any same-day history is added. Isolation Forest is fit on the early 70% chronological segment of eligible profiles; only later eligible claims receive a non-neutral anomaly score.

The model is unsupervised. `fraud_label` and synthetic scenario metadata are not loaded into profile construction or Isolation Forest fitting. Labels below are joined only after scoring to compare synthetic cohorts.

## Score interpretation

`agent_anomaly_score` ranges from 0 to 1. Higher values mean the observed agent behavior is less typical of the model's early historical profile distribution. It is a review signal only; it does not prove fraud or establish that an agent or claim is improper.

## Synthetic evaluation results

Run: fixed seed synthetic dataset, 50,000 claims, default anomaly configuration (`random_state=42`, 200 trees, 10% contamination, minimum three prior agent claims).

| Synthetic cohort | Eligible scored claims | Mean score | Median score | Share in top score decile |
| --- | ---: | ---: | ---: | ---: |
| Baseline-labelled claims | 13,346 | 0.3820 | 0.3462 | 8.88% |
| Injected suspicious-pattern claims | 1,442 | 0.4879 | 0.4711 | 20.46% |

The injected-pattern cohort has a higher mean and median anomaly score and is more represented in the top decile. This validates that the synthetic behavior injections produce detectable distributional differences; it is not a real-world fraud-performance claim.
