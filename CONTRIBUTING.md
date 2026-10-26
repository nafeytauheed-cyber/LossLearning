# Contributing

Install `pip install -e ".[dev]"`; run pytest -q, ruff check ., and ruff format
--check .. Preserve explicit-autograd and finite-step influence checks, stable
sample IDs, training-only reference construction, and save/load round trips.

Compare identical splits, corruption, seeds and budgets. Report failures and
minority recall. Never use test outcomes to choose hyperparameters. Name incomplete
published-baseline reproductions accurately. Add regression tests for bugs.
Keep datasets, checkpoints, caches, credentials and private data out of commits.
