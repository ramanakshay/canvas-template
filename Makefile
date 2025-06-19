
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

.PHONY: docs-build
docs-build:
		mkdocs build

.PHONY: docs-serve
docs-serve:
		mkdocs serve
