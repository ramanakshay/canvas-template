import os

# Remove LICENSE if "No license file"
if "{{ cookiecutter.license }}" == "None":
    os.remove("LICENSE")

# Choose lockfile based on environment manager
if "{{ cookiecutter.environment_manager }}" == "uv":
    os.remove("requirements.txt")
else:
    os.remove("uv.lock")
