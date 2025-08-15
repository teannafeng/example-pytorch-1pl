This is a minimal example that simulates binary response data and fits a 1-Parameter Logistic (1PL) Item Response Theory (IRT) model using PyTorch.

## Notes
The repo is provided as a minimal working example. The script has not been extensively tested and may omit error handling or edge cases.

## Requirements

- torch
- matplotlib
- tqdm

## Structure (subject to change)
```text
.
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

