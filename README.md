# Canvas ☯︎

> "Beauty is as important in computing as it is in painting or architecture." — Donald E. Knuth

A simple, flexible, and modular PyTorch template for kickstarting your deep learning projects. Inspired by the agent-environment interface, Canvas aims to provide a unified template for all types of ML projects.

<p align="center">
<img src="https://raw.githubusercontent.com/ramanakshay/canvas/main/docs/images/architecture.svg"
alt="Canvas Architecture">
</p>

## Available Templates

Canvas offers unique templates to kickstart various machine learning tasks:

* **Supervised Learning (SL):** Perfect for tasks like classification and regression, where you have labeled data.

* **Reinforcement Learning (RL):**  Designed for building agents that learn by interacting with an environment.

* **Self-Supervised Learning (SSL):** Train models to learn representations from unlabeled data.

## Installation

Canvas requires **Python 3.11+**. Since it's a command-line tool, we highly recommend using [uv](https://docs.astral.sh/uv/) for installation.

```
# Recommended: with uv
uv tool install canvas-template

# Alternative: with pip
pip install canvas-template
```

## Usage

To create a new project, run the `canvas init` command and pick your template. No need to create a directory first; Canvas will do it for you.

```
canvas init [sl|ssl|rl]
```

**Example:** The following settings will create a project named *my_project* with the self-supervised learning (SSL) template.

<p align="center">
<img src="https://raw.githubusercontent.com/ramanakshay/canvas/main/docs/images/canvas_demo.png"
alt="Canvas Demo">
</p>


## Project Structure

The directory structure of your new project will look something like this (depending on the settings that you choose):

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
│   ├── model/              # Your deep learning model definitions
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── network.py
│   │   └── loss.py
│   ├── data/               # Scripts for loading and preparing your data
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── preprocess.py
│   └── algorithm/          # The core algorithm: training, evaluation
│       ├── __init__.py
│       ├── trainer.py
│       └── evaluator.py
├── dataset/                # Place your raw or processed datasets here
│   └── ...
└── outputs/                # Where experiment logs, model checkpoints, and results are saved
    └── ...
```

## Contributing

Any kind of enhancement or contribution is welcome. If you have bug fixes, new features, or any improvements, I'd love your help!

### TODOs

- [ ] Experiment Tracking (wandb, tensorboard)
- [ ] Testing Framework (pytest, unittest)
- [ ] Distributed Training (pytorch, accelerate)
