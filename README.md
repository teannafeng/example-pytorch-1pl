This repository contains a minimal working example that simulates binary response data and fits a 1-Parameter Logistic (1PL) Item Response Theory (IRT) model using PyTorch.

## Note
- This repository is intended as a proof-of-concept demo and can be further developed for pedagogical purposes (e.g., classroom labs on IRT and PyTorch).

## Requirements

- torch
- matplotlib
- tqdm

## Folder structure
```text
example-pytorch-1pl
├── RUN_1PL.py  # main script with model, estimation/training, evaluation, and plotting
│   ├── Classes:
│   │   └── IRT_1PL
│   ├── Functions:
│   │   ├── sample_true_params(...)
│   │   ├── simulate_binary_data(...)
│   │   ├── train(...)
│   │   ├── evaluate(...)
│   │   ├── plot_est_path(...)
│   │   └── wrapper(...)
├── README.md
├── .gitignore
└── .gitattributes
```

