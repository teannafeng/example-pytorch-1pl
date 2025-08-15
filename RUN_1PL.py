"""
A Minimal Example: 1PL IRT Model Estimation with PyTorch

Simulates binary IRT data and fits a 1PL model with a learned learning rate using PyTorch.

Provided as a minimal working example. Not extensively tested.
"""

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Optional

# %%
# Simulation functions
def sample_true_params(N: int, I: int, mean: float = 0.0, std: float = 1.0, seed: int = 1010):
    torch.manual_seed(seed)
    a = torch.ones(I)
    b = torch.normal(mean=mean, std=std, size=(I,))
    c = -a * b
    eta = torch.randn(N)
    return a, b, c, eta

def simulate_binary_data(a: torch.Tensor, c: torch.Tensor, eta: torch.Tensor):
    eta = eta.view(-1, 1)
    a = a.view(1, -1)
    c = c.view(1, -1)
    Z = eta * a + c
    P = torch.sigmoid(Z)
    Y = torch.bernoulli(P)
    return Y

# %%
# Model class
class IRT_1PL(nn.Module):
    def __init__(self, N: int, I: int):
        super().__init__()
        self.register_buffer("a", torch.ones(I))
        self.c = nn.Parameter(torch.zeros(I))
        self.eta = nn.Parameter(torch.randn(N))
        self._log_lr = nn.Parameter(torch.tensor(0.0)) # exp(0.0) = 1.0

    def forward(self):
        eta = self.eta.view(-1, 1)
        Z = eta * self.a + self.c
        P = torch.sigmoid(Z)
        return P 
    
    def get_lr(self):
        return torch.exp(self._log_lr)

# %%
# Training / estimation function
def train(Y: torch.Tensor, N: int, I: int, max_iter: int = 1000, 
          verbose: bool = False, track_c: bool = False):
    model = IRT_1PL(N, I)
    criterion = nn.BCELoss()

    mp = ['_log_lr']
    item_params = [p for n, p in model.named_parameters() if n not in mp]
    meta_params = [p for n, p in model.named_parameters() if n in mp]

    main_optimizer = optim.Adam(item_params, lr=1.0)
    meta_optimizer = optim.Adam(meta_params, lr=1.0)

    c_history = []

    for _ in tqdm(range(max_iter), disable=not verbose):
        model.train()
        P = model()
        loss = criterion(P, Y)

        main_optimizer.zero_grad()
        meta_optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mp:
                    continue
                if param.grad is not None:
                    param -= model.get_lr() * param.grad

        meta_optimizer.step()

        if track_c:
            c_history.append(model.c.detach().clone().cpu())

    return model, c_history

# %%
# Evaluation functions
def evaluate(model: torch.nn.Module, true_c: torch.Tensor, round_to: int = 2, 
             bias_upper: float = 0.05, bias_lower: float = 0.01):
    def coral_bold(text):
        return f"\033[1m\033[38;2;255;127;80m{text}\033[0m"
    
    def blue_bold(text):
        return f"\033[1m\033[38;2;70;130;180m{text}\033[0m"
    
    print()
    print(f"{'Item':<6} {'c (est)':>10} {'c (true)':>10} {'bias':>10}")
    print("-" * 40)

    for i in range(len(true_c)):
        est = model.c[i].item()
        true = true_c[i].item()
        bias = est - true

        bias_str = f"{bias:>{10}.{round_to+1}f}"
        if abs(bias) >= bias_upper:
            bias_str = coral_bold(bias_str)
        elif abs(bias) <= bias_lower:
            bias_str = blue_bold(bias_str)

        print(f"{i+1:<6} {est:>10.{round_to}f} {true:>10.{round_to}f} {bias_str}")

def plot_est_path(
        est_history: List[torch.Tensor], 
        true_c: torch.Tensor, 
        item_indices: Optional[List[int]] = None, 
        fig_size: tuple = (6, 5),
        show_plot: bool = True,
        save_plot: bool = False,
        save_path: str = "./figs/est_path.png"
    ):
    
    history_tensor = torch.stack(est_history)

    if item_indices is None:
        item_indices = list(range(history_tensor.shape[1]))  # Plot all items

    plt.figure(figsize=fig_size)
    for i in item_indices:
        est_path = history_tensor[:, i]
        true_val = true_c[i].item()
        
        plt.plot(est_path, label=f'Item {i+1}')
        plt.axhline(true_val, color='gray', linestyle='--', linewidth=0.5)

    plt.xlabel('Iteration')
    plt.ylabel('Est. C')
    plt.tight_layout(rect=[0.0, 0.1, 1.0, 1.0])
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=5,
        frameon=False
    )

    if save_plot:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"--> Est. path plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

# %%
# Wrapper function
def wrapper(seed: int = 1010, N: int = 60000, I: int = 36, max_iter: int = 1000, track_c: bool =False):
    torch.manual_seed(seed)
    true_a, true_b, true_c, eta_true = sample_true_params(N=N, I=I, seed=seed)
    Y = simulate_binary_data(a=true_a, c=true_c, eta=eta_true)
    model, c_history = train(Y=Y, N=N, I=I, max_iter=max_iter, verbose=True, track_c=track_c)
    return model, true_b, true_c, c_history

# %%
if __name__ == "__main__":
    
    torch.manual_seed(12345)

    model, true_b, true_c, c_history = wrapper(
        seed=1010, N=100_000, I=5, max_iter=1000, track_c=True
    )

    evaluate(model, true_c)

    plot_est_path(
        est_history=c_history, true_c=true_c, 
        show_plot=False, save_plot=True, save_path="./figs/est_path.png"
    )

# %%
