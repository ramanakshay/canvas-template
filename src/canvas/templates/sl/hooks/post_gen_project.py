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
else:
    os.remove("uv.lock")
