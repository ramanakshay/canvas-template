# {{ cookiecutter.project_name }}

{{ cookiecutter.description}}

## Commands

* `make install` - Install dependencies.
* `make train` - Train model.
* `make clean` - Clean project artifacts.
{% if cookiecutter.documentation != "none" -%}
* `make docs` - Build documentation.
* `make docs-serve` - Serve documentation locally.
{%- endif %}
{% if cookiecutter.linting_and_formatting != "none" -%}
* `make lint` - Run linting checks.
* `make format` - Apply code formatting.
{%- endif %}
