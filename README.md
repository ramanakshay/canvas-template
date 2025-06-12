# Canvas ☯︎

A simple, flexible, and modular pytorch template for your deep learning projects. There are multiple templates available for different kinds of machine learning tasks:

- Supervised Learning (SL)
- Reinforcement Learning (RL)
- Self-Supervised Learning (SSL)

<div align="center">

<img align="center" src="docs/assets/architecture.svg">

</div>

## Installation

```
# Clone Reposity
pip install canvas-template

# Run Commmand
canvas create {sl,ssl,rl}
```

**Core Requirements**
- [pytorch](https://pytorch.org/) (An open source deep learning platform)
- [hydra](https://hydra.cc/) (A framework for configuring complex applications)


## Folder Structure
```
├── model                - this folder contains all code (networks, layers, loss) of the model
│   ├── weights
│   ├── model.py
│   ├── network.py
│   └── loss.py
│
├── data                 - this folder contains code relevant to the data and datasets
│   ├── datasets
|   ├── data.py
│   └── prepare.py
│
├── algorithm            - this folder contains different algorithms of your project
│   ├── train.py
│   └── test.py
│
├── config
│   └── config.yaml      - YAML config file for project
│
└── main.py              - entry point of the project

```


## TODOs

Any kind of enhancement or contribution is welcomed.

- [ ] Support for loggers
- [ ] Distributed training integration
