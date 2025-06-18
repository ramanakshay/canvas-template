import os
import shutil

# Remove LICENSE if "No license file"
if "{{ cookiecutter.license }}" == "None":
    os.remove("LICENSE")

# Remove documentation if "None"
if "{{ cookiecutter.documentation }}" == "None":
    os.remove("mkdocs.yml")
    shutil.rmtree("docs/")

# Choose lockfile based on environment manager
if "{{ cookiecutter.environment_manager }}" == "uv":
    os.remove("requirements.txt")
elif "{{ cookiecutter.environment_manager }}" == "venv":
    os.remove("uv.lock")
else:
    os.remove("requirements.txt")
    os.remove("uv.lock")
