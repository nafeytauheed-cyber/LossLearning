# Reproduction

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
ruff check .
ruff format --check .
losslearning-influence --output results/influence
losslearning-train --config configs/quick.yaml
losslearning-benchmark --config configs/synthetic.yaml
losslearning-ablate --config configs/ablation.yaml
losslearning-benchmark --config configs/class_dependent.yaml
```

Quick: 2 fits on noisy moons, figures and checkpoints. Synthetic: 504 fits,
6 geometries x 4 noise levels x 3 seeds x 7 methods. Pilot: smaller 189-fit subset.
Ablations: 84 fits, 2 geometries x 3 seeds x 14 variants. Class-dependent: 36 fits,
2 geometries x 2 levels x 3 seeds x 3 methods. Each synthetic run uses 1,000
samples and 40 epochs. Splits are 60% train / 15% clean validation / 25% clean
test. Corruption applies ONLY after splitting training data. Symmetric flips
always change labels. Class-dependent class c flips with probability rate*(c+1)/K
to the next class cyclically. Both nominal and realized corruption are recorded.

Reasonable defaults were fixed before evaluation; no validation/test tuning or
early stopping was used. Results are final-epoch scores. Same architecture seeds,
splits and corruption masks are reused across methods. Wider-MLP CE provides an
approximate parameter-count comparison for potentials, not an exactly matched
functional architecture. AdamW, weight decay and epochs match across losses.

Every run records configuration, seed, device, versions, optimizer, runtime,
parameter count, metrics, split sizes and Git revision (null if not yet committed).
Per-run JSON and CSV summaries preserve every run. NPZ diagnostics contain split
IDs, known clean labels and corruption masks for post-training analysis only;
fit receives only observed training labels. Zero score AUROC can mean no available
evidence, not a meaningful noise detector. Hard-clean means the top 20% of final
observed-label CE among truly clean training samples; it is a diagnostic proxy,
not a ground-truth boundary annotation.

```bash
losslearning-figures --results results/runs/YOUR_RUN_DIRECTORY
```

Regenerates summary plots. Detailed panels also regenerate if model.pt exists
(quick runs retain it). Large runs omit checkpoints by default. Release results
are copied into results/release without datasets, large checkpoints or caches.
PNG panels contain measured data. Standard deviation across 3 seeds is descriptive,
not a confidence interval or a significance test. CPU runtime is shared-host wall
time; no dedicated-hardware speed claim. Peak memory is null when unmeasured.

## Image pilots

```bash
pip install -e ".[images]"
losslearning-benchmark --config configs/fashion_mnist.yaml
losslearning-benchmark --config configs/cifar10.yaml
```

These configs intentionally use only 2,000 training, 1,000 validation and 1,000
test samples, 10 epochs, seed 42, 20% noise. They are pipeline pilots, not published
full-data protocols. Increase sample limits, epochs, seeds and noise conditions
for a meaningful image study. No image dataset is redistributed. The small CNN
uses no augmentation; CIFAR is consequently especially limited. Install an
appropriate CUDA build for your own driver; no RTX 3050 VRAM capacity is assumed.
CUDA execution, AMP and full-scale image accuracy are not verified by CPU tests.

A workspace reset during development discarded an earlier draft and its outputs.
The delivered artifacts are regenerated runs, not reconstructed benchmark claims.
The source revision and measured output files included here are the audit trail.
