import subprocess
import os

if "{{ cookiecutter.documentation }}" == "mkdocs":
    command = ['mkdocs', 'new', '.', '--quiet']
    subprocess.run(command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
elif "{{ cookiecutter.documentation }}" == "sphinx":
    os.mkdir('docs/')
    command = [
        'sphinx-quickstart',
        '-q',
        '-p', "{{ cookiecutter.project_name }}",
        '-a', "{{ cookiecutter.author_name }}",
        '--no-makefile',
        '--no-batchfile',
        'docs/'
    ]
    subprocess.run(command,
        stdout=subprocess.DEVNULL,  # Suppress standard output
        stderr=subprocess.DEVNULL)
