# LossLearning v0.1.0 implementation report

The repository is ready to inspect and publish as an experimental research project.
It is not evidence of a generally superior new classifier. No remote was created.

Implemented: MLP/CNN encoders; linear/potential heads; stable detached per-example
GCE; focal/fixed-GCE/CE baselines; EMA teacher; bounded class-stratified memory;
chunked exact/cached head-gradient evidence; entropy contradiction; stable-ID EMA
history; count/quantile safeguards; warmup/ramp; missing-evidence fallback;
optional consistency/regularization; sklearn-inspired estimator; indexed image
trainer; inference save/load; resumable epoch checkpoints; CLI/configuration;
logging, plots, tests, CI workflow, MIT license and citation metadata.

Mathematics and provenance: docs/mathematics.md derives the potential functions,
CE gradients, outer-product factorization, Taylor approximation, contradiction,
controller and stable GCE limit/gradient. It explicitly proves weighted-CE gradient
equivalence and binary cosine degeneracy. docs/literature_review.md separates
established components from the experimental combination; novelty is unestablished.

Verification: editable installation succeeded; 25 tests pass, including finite
loss derivatives, explicit autograd, zero-gradient behavior, stability, duplicate
exclusion, stable IDs, CPU CNN training, deterministic continuation, save/load,
input validation, leakage checks and a complete tiny benchmark. Ruff lint/format
pass locally. README quickstart ran at 96% accuracy. All documented command entry
points were exercised except the uncompleted CIFAR training command. GitHub CI is
configured but not remotely executed. CUDA remains unverified.

Executed: 504 symmetric-noise synthetic fits, 84 ablations, 36 class-dependent
fits, two quick demo fits, four reduced Fashion-MNIST fits, numerical influence
and stale-cache diagnostics. A machine audit recomputes metrics and verifies split
separation for all 626 synthetic runs. Saved JSON/CSV/NPZ and PNG artifacts are in
results/release and assets. No model-selection use of clean validation/test data.

Observed: full versus CE wins/ties/losses are 25/17/30 across 72 paired settings,
with mean balanced accuracy -0.31 points and median wall time about 2.34x. Default
adaptation harms several tasks, especially minority recall. Some conditions improve.
Quantile thresholds are a promising exploratory ablation, not a validated replacement
default. Potential features, history and gradient evidence do not consistently help.
The tiny Fashion-MNIST adaptive runs end at q=0, so they supply no image-adaptation
success evidence. See docs/experiments.md and docs/result_tables.md for all outcomes.

Remaining: CIFAR transfer was incomplete and stopped, so no CIFAR model was trained.
Full image protocols, strong published baselines, GPU checks, AMP/accumulation,
suppression budgets and comprehensive sklearn compliance were not completed.
The smaller supported release is explicitly distinguished from those future tasks.

Reproduce using the commands in README.md and docs/reproducibility.md. Review the
code and negative results before publishing. The project is useful as an honest,
tested experimental framework; a research breakthrough or SOTA claim is unsupported.
