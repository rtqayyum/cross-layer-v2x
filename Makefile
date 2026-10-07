PYTHON ?= .venv/bin/python

.PHONY: setup test verify analyze

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests -q

verify:
	./verify_artifact.sh

analyze:
	PYTHON=$(PYTHON) ./analyze_frozen_results.sh

