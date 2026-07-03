## What makes this project stand out

- **FM interaction layer** (Rendle 2010): explicit pairwise feature interaction in O(k·d) — the same core technique as DeepFM (Guo et al., IJCAI 2017).
- **PCGrad** (Yu et al., NeurIPS 2020): implemented gradient-surgery from the published paper, including a real bug fix against the plan's own reference pseudocode (see README).
- **A real leakage catch, not a hypothetical one**: found via a suspiciously perfect R²=0.999, root-caused to an algebraic identity between the target and two input features, fixed via a temporal train/target window split.
- **Calibration analysis that goes beyond "run Platt scaling"**: diagnosed *why* temperature scaling failed (bias vs. sharpness) before reaching for isotonic regression, and then documented isotonic's own limitation (ranking ties) rather than presenting a clean-looking result that wasn't fully clean.
- **A business layer with an actual quantified comparison**: budget-allocation simulation showing the joint churn+LTV ranking beats naive strategies (random selection, churn-only ranking) by 750-1300% in expected revenue saved — not just a formula, a measured claim.