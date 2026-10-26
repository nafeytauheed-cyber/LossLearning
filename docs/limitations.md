# Limitations and unverified work

- No demonstrated general advantage over CE; minority harm occurs even on clean data.
- Existing instance-adaptive robust losses and gradient influence are close prior art.
  Novelty is unestablished; detached GCE has weighted-CE-equivalent gradients.
- Binary residual cosine is essentially label agreement. The evidence can be a
  neighborhood consistency heuristic rather than distinctive gradient information.
- Teacher confidence and stability can confidently encode incorrect labels.
  Equal class quotas cannot restore a class the teacher never recognizes.
- Missing evidence conservatively falls back to CE; this often disables adaptation
  at high corruption and throughout the small image pilot.
- Exact factorization is final-layer only. The counterfactual is SGD at fixed
  features, while actual training uses full-network AdamW.
- Cached features may change neighbors and cause large individual score errors.
- Class counts are observed counts; corruption changes them. Quantile safeguards
  look promising in one ablation but have not been independently validated.
- Small synthetic datasets, three seeds, no uncertainty-adjusted significance
  testing, no tuning budget, and one short image seed limit inference.
- Co-teaching, DivideMix, ELR, ProMeCD and other strong published methods were not
  reproduced. Optional KL consistency is not a substitute labeled as ELR.
- No CIFAR training, full Fashion-MNIST/CIFAR protocol, CUDA validation, AMP,
  gradient accumulation, distributed support, class suppression budgets, or full
  sklearn estimator compliance certification. Image global deduplication is absent.
- Per-run CPU peak memory was not measured. Timing is shared-host descriptive.
- Frozen encoder mode freezes initial random features unless explicitly supplied
  pretrained weights through the lower-level model interface.
- Checkpoints use pickle; load trusted files only. Continuation assumes identical
  indexed datasets and deterministic provided loaders, not arbitrary augmentations.

Next useful experiments: preregister the quantile-threshold variant on fresh rare-
class data, compare scalar-weight-equivalent CE directly, add multiclass instance-
dependent noise, measure teacher error and bank coverage by class, and reproduce a
strong noisy-label baseline with matching budgets. A controller that helps only
when references are already correct may have little value over a simpler method.
