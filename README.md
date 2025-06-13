# Canvas ☯︎

> "Beauty is as important in computing as it is in painting or architecture." — Donald E. Knuth

A simple, flexible, and modular pytorch template for your deep learning projects. There are multiple templates available for different kinds of machine learning tasks:

- Supervised Learning (SL)
- Reinforcement Learning (RL)
- Self-Supervised Learning (SSL)

<p align="center">
<img width="40%" src="https://raw.githubusercontent.com/ramanakshay/canvas/main/docs/assets/architecture.svg">
</p>

## Installation

Canvas is a command-line application that requires Python 3.9+. Currently, we recommend installing it from TestPyPi.

```
# With pip from TestPyPi
pip install -i https://test.pypi.org/simple/ canvas-template==0.1.4
```

## Commands

To start a new project, run the following command:

```
canvas create {sl/ssl/rl}
```

The following settings will create a project named *my_project* with the self-supervised learning (ssl) template. 

<p align="center">
<img src="https://raw.githubusercontent.com/ramanakshay/canvas/main/docs/assets/terminal_output.png">
</p>


## Directory Structure

The directory structure of your new project will look something like this (depending on the settings that you choose):

```
├── model                - this folder contains all code (networks, layers, loss) of the model
│   ├── weights
│   ├── model.py
│   ├── network.py
│   
│
├── data                 - this folder contains code relevant to the data and datasets
│   ├── datasets
|   ├── data.py
│   └── prepare.py
│
├── algorithm            - this folder contains different algorithms of your project
│   ├── loss.py  
│   ├── trainer.py
│   └── evaluator.py
│
├── config
│   └── config.yaml      - YAML config file for project
│
└── main.py              - entry point of the project

```


## Contributing

Any kind of enhancement or contribution is welcomed.

- [ ] Support for loggers
- [ ] Distributed training integration
