# Canvas ☯︎

![PyPI - Version](https://img.shields.io/pypi/v/canvas-template)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/canvas-template)
[![Documentation](https://img.shields.io/badge/docs-reference-blue.svg)](https://akshayraman.com/canvas-template/)

> "Beauty is as important in computing as it is in painting or architecture." — Donald E. Knuth

A simple, flexible, and modular canvas for designing your deep learning projects. Powered by PyTorch + Hydra, Canvas aims to provide a unified template for all kinds of machine learning projects.

<p align="center">
<img src="https://raw.githubusercontent.com/ramanakshay/canvas/main/docs/images/architecture.svg"
alt="Canvas Architecture">
</p>

## Available Templates

Canvas offers unique templates to kickstart various machine learning tasks:

* **Supervised Learning (SL):** For tasks like classification and regression with labeled data. Example - ResNet for Image Classification.

* **Self-Supervised Learning (SSL):** For training models to learn representations from unlabeled data. Example - Next-word Prediction using GPT.

* **Reinforcement Learning (RL):**  For building agents that learn by interacting with an environment. Example - PPO on Gymnasium Environment.

## Installation

Canvas requires **Python 3.11+**. Since it's a command-line tool, we highly recommend using [uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/stable/) for installation.

```
# with uv (Recommended)
uv tool install canvas-template

# with pipx
pipx install canvas-template

# with pip
pip install canvas-template
```

## Usage

To create a new project, simply run the `canvas init` command and select your template. Canvas will automatically create the project directory for you.

```
canvas init [sl|ssl|rl]
```

**Example:** The following command will create a project named *my_project* using the self-supervised learning (SSL) template.

<p align="center">
<img src="https://raw.githubusercontent.com/ramanakshay/canvas/main/docs/images/canvas_demo.png"
alt="Canvas Demo">
</p>

## Running the Project

Once your project is created, you can use the included `Makefile` to run common tasks.

```
# Install dependencies
make install

# Start model training
make train
```

## Directory Structure

The structure of your new project will look something like this (depending on the settings that you choose):

```
example-project/
├── Makefile                # Convenient make commands
├── LICENSE                 # Project license
├── README.md               # Your project's main README file
├── mkdocs.yml              # Configuration for building documentation
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # uv lock file for consistent environments
├── docs/                   # Where your project's documentation lives
│   └── index.md
├── src/                    # All the source code for your deep learning project
│   ├── main.py             # The main entry point of your project
│   ├── config/             # Hydra configuration files for managing settings
│   │   └── config.yaml
│   ├── model/              # Core deep learning model definitions
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── network.py
│   │   └── loss.py
│   ├── data/               # Scripts for loading and preparing your data
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── preprocess.py
│   └── algorithm/          # Training and evaluation algorithms
│       ├── __init__.py
│       ├── trainer.py
│       └── evaluator.py
├── dataset/                # Place your raw or processed datasets here
│   └── ...
└── outputs/                # Where hydra logs, model checkpoints, and results are saved
    └── ...
```

## Contributing

If you have bug fixes, new features, or any improvements, your contributions are highly appreciated! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to get involved.

## License

Canvas is licensed under the MIT license.
