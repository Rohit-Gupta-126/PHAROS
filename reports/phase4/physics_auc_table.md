# Stream A physics ROC/AUC — background vs A->4l

Model: `/mnt/e/AI_Projects/PHAROS/models/physics_vae_r20260711T173416Z` · n_bg=100000 n_sig=55969

| Score | AUC | bg median | sig median | bg mean | sig mean |
|-------|-----|-----------|------------|---------|----------|
| Σμ² (trigger) | **0.775** | 1.46e-06 | 5.12e-06 | 6.01e-05 | 0.195 |
| recon MSE (offline) | **0.889** | 0.231 | 2.3 | 0.895 | 267 |

_Medians quoted alongside means: both scores are heavy-tailed and non-negative, so recon-MSE means are outlier-dominated._
