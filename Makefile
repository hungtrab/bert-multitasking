.PHONY: setup data train eval test lint clean

PYTHON ?= python3
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

CONFIG ?= configs/round_robin.yaml
CHECKPOINT ?= runs/latest/best.pt
DATA_DIR ?= data
V1_DATA  := ../Multitask BERT/source_code/data

setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

data:
	@mkdir -p "$(DATA_DIR)"
	@find "$(DATA_DIR)" -maxdepth 1 -type l -delete
	@if [ -d "$(V1_DATA)" ]; then \
		echo "Linking CSVs from v1 data dir..."; \
		find "$$(realpath "$(V1_DATA)")" -maxdepth 1 -type f -name '*.csv' \
			-exec ln -sf {} "$(DATA_DIR)/" \; ; \
	else \
		echo "v1 data not found at $(V1_DATA). Place SST/Quora/STS CSVs in $(DATA_DIR)/ manually."; \
	fi
	@ls "$(DATA_DIR)"

train: setup
	$(PY) -m scripts.train --config $(CONFIG)

eval: setup
	$(PY) -m scripts.evaluate --checkpoint $(CHECKPOINT) --config $(CONFIG)

predict: setup
	$(PY) -m scripts.predict --checkpoint $(CHECKPOINT) --config $(CONFIG)

test: setup
	$(PY) -m pytest

lint: setup
	$(VENV)/bin/ruff check src tests

clean:
	rm -rf $(VENV) **/__pycache__ .pytest_cache .ruff_cache runs/
