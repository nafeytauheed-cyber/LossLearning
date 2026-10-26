# Architecture and API

LossLearning subclasses ClassifierMixin and BaseEstimator. Public operations:
fit, predict, predict_proba, score, get_params, set_params, save, and classmethod
load. Attributes: classes_, history_, contradiction_scores_, q_. Full sklearn
estimator-check compliance is not certified.

Defaults: MLP, linear head, 32-dimensional normalized embedding, hidden width 64,
40 epochs, batch size 128, AdamW lr=.003, seed 42. `head="potential"` adds 16
prototypes. `freeze_encoder=True` freezes initial features; without replacing the
encoder with pretrained weights those features are random, not pretrained.

```python
model = LossLearning(
    head="potential",
    trainer_options={
        "warmup": 8,
        "q_max": 0.8,
        "rho": 0.8,
        "bank_size": 256,
        "reference_mode": "exact",
        "neighbors": 12,
    },
)
model.fit(X_train, y_observed, X_val=X_val, y_val=y_val)
```

`TrainingConfig` in training.py is authoritative. Constructor parameters own
epochs/lr/seed/adaptive; trainer_options cannot override those again. Unknown
options raise errors. Tabular standardization fits ONLY training inputs. CNN
arrays are NCHW floats, conventionally [0,1], without per-feature scaling.
Validation logs metrics; it never selects references, thresholds or checkpoints.
Repeated fit initializes a fresh model. String labels are preserved.

## DataLoader interface

Instantiate LossLearningNet(input_shape, n_classes, ...) and
LossLearningTrainer(model, observed_labels, config, device). Training and reference
batches must be `(x, y_observed, dense_id, duplicate_group_id)`. IDs are integers
0..N-1; the supplied label vector is in this exact order. Preserve the mapping
across shuffles and continuation. Reference loaders contain only training samples,
with stable transforms. Optional validation batches are `(x,y,...)`.
`images.py` contains a working torchvision adapter. Source datasets may retain
compact image bytes, but float transformed images are loaded per item/batch.
Global image deduplication is not implemented. Tabular exact duplicates are grouped.

## State and continuation

`model.save(path)` stores inference weights, class mapping, scaler and diagnostic
summaries, not training images or optimizer state. `LossLearning.load(path)`
restores inference. `trainer.save_checkpoint(path)` stores model, teacher,
optimizer, reference bank, per-ID histories and epoch state. `load_checkpoint`
permits changing only total epochs. `fit(..., checkpoint_dir=...)` writes latest.pt
at each epoch. Same indexed data is required. Per-epoch Torch/DataLoader seeds
make CPU continuation deterministic for supplied loaders. Python/NumPy stochastic
custom augmentations and arbitrary workers are not covered by that guarantee.
Only load trusted files: pickle checkpoint deserialization can execute code.

Live per-ID histories are O(N); optional diagnostic snapshots are capped at
history_snapshots=40. No unbounded per-example history accumulates. CUDA is detected
but not tested here. AMP, gradient accumulation, suppression budgets and distributed
training are explicitly not implemented in this release.
