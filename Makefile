
.PHONY: install
install:
	uv sync

.PHONY: build
build:
	uv build

.PHONY: publish
publish: dist
	uv publish

.PHONY: clean
clean:
	@echo "Cleaning up project artifacts.."
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
	rm -rf build dist src/*.egg-info
	rm -rf site
	uv run -m ruff clean

.PHONY: lint
lint:
	uv run -m ruff check . --fix

.PHONY: format
format:
	uv run -m ruff format .

.PHONY: docs-install
docs-install:
	uv sync --group docs

.PHONY: docs
docs:
	uv run -m mkdocs build

.PHONY: docs-serve
docs-serve:
	uv run -m mkdocs serve
