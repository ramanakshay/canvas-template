# Canvas ☯︎

A simple, flexible, and modular pytorch template for your deep learning projects. There are multiple templates available for different kinds of machine learning tasks. **Switch to the appropriate branch** from above and see the installation section to download the template:

- Supervised Learning (SL)
- Reinforcement Learning (RL)
- Self-Supervised Learning (SSL)

<div align="center">

<img align="center" src="assets/architecture.svg">

</div>

## Installation

1.  Clone the repository.
```
git clone https://github.com/ramanakshay/canvas --depth 1 --branch sl
```

2. Install dependencies from requirements file. Make sure to create a virtual/[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment before running this command.
```
pip install -r requirements.txt
```

3. Run the code.
```
# navigate to src folder
cd src

# run the main file
python main.py
```

**Main Requirements**
- [pytorch](https://pytorch.org/) (An open source deep learning platform)
- [hydra](https://hydra.cc/) (A framework for configuring complex applications)


## Folder Structure
```
├── model                - this folder contains all code (networks, layers) of the model
│   ├── weights
│   ├── model.py
│   ├── network.py
│   └── layers.py
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
- [ ] Distributed training sample code
