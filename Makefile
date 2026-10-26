.PHONY: test lint quick benchmark ablate figures

test:
	pytest -q
lint:
	ruff check .
	ruff format --check .
quick:
	losslearning-train --config configs/quick.yaml
benchmark:
	losslearning-benchmark --config configs/synthetic.yaml
ablate:
	losslearning-ablate --config configs/ablation.yaml
figures:
	losslearning-figures --results $(RESULTS)
