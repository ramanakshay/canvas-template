{% if cookiecutter.environment_manager == 'venv' -%}
PYTHON = .venv/bin/python

.venv:
	python -m venv .venv

.PHONY: install
install: .venv
	@echo "Installing Dependencies.."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --group dev {% if cookiecutter.documentation != "None" -%} --group docs {%- endif %}
	@echo "Dependencies installed."
{% elif cookiecutter.environment_manager == 'uv' -%}
PYTHON = uv run

.PHONY: install
install:
	uv sync --all-groups
{% else -%}
PYTHON = python

.PHONY: install
install:
	@echo "Installing Dependencies.."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --group dev {% if cookiecutter.documentation != "None" -%} --group docs {%- endif %}
	@echo "Dependencies installed."
{%- endif %}
.PHONY: train
train:
	$(PYTHON) src/main.py

.PHONY: clean
clean:
	@echo "Cleaning up project artifacts.."
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delet
	find . -name '__pycache__' -delete
	rm -rf build dist src/*.egg-info
	{% if cookiecutter.linting_and_formatting == "ruff" -%}
	$(PYTHON) -m ruff clean
	{%- endif %}
	{% if cookiecutter.documentation == 'mkdocs' -%}
	rm -rf site
	{% elif cookiecutter.documentation == 'sphinx' -%}
	rm -rf docs/_build
	{%- endif %}
{% if cookiecutter.documentation == 'mkdocs' -%}
.PHONY: docs
docs:
	$(PYTHON) -m mkdocs build

.PHONY: docs-serve
docs-serve:
	$(PYTHON) -m mkdocs serve
{% elif cookiecutter.documentation == 'sphinx' -%}
.PHONY: docs
docs:
	$(PYTHON) -m sphinx -b html docs/ docs/_build

.PHONY: docs-serve
docs-serve:
	$(PYTHON) -m http.server 8000 --bind 127.0.0.1 --directory docs/_build
{%- endif %}
{% if cookiecutter.linting_and_formatting == 'ruff' -%}
.PHONY: lint
lint:
	$(PYTHON) -m ruff check . --fix

.PHONY: format
format:
	$(PYTHON) -m ruff format .
{% elif cookiecutter.linting_and_formatting == 'flake8+isort+black' -%}
.PHONY: lint
lint:
	$(PYTHON) -m flake8 .
	$(PYTHON) -m isort --check-only .

.PHONY: format
format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort .
{%- endif %}
