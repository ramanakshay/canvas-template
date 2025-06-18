{% if cookiecutter.environment_manager == 'venv' %}
PYTHON = ".venv/bin/python"

.venv:
	python -m venv .venv

.PHONY: install
install: .venv
	@echo "Installing Dependencies.."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Dependencies installed."

.PHONY: train
train:
	$(PYTHON) src/main.py
{% elif cookiecutter.environment_manager == 'uv' %}
.PHONY: install
install:
	uv sync

.PHONY: train
train:
	uv run src/main.py
{% else %}
PYTHON = python

.PHONY: install
install:
	python -m pip install -r requirements.txt

.PHONY: train
train:
	python src/main.py
{% endif %}
.PHONY: clean
clean:
	@echo "Cleaning up project artifacts.."
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
	rm -rf build dist src/*.egg-info
{% if cookiecutter.documentation == 'mkdocs' %}
.PHONY: docs-build
docs-build:
		mkdocs build

.PHONY: docs-serve
docs-serve:
		mkdocs serve
{% endif %}
