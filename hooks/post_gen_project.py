from pathlib import Path

# Remove LICENSE if "No license file"
if "{{ cookiecutter.open_source_license }}" == "None":
    Path("LICENSE").unlink()