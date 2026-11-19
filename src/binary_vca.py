"""
Tier 3C: Binary Variational Classical Annealing (Binary VCA) using a
Multi-Layer Dilated Recurrent Neural Network without Weight Sharing
trained with the Adam optimizer (Equations 15-17).

Adapted directly from Hibat-Allah et al. (Nature Machine Intelligence, 2021;
github.com/VectorInstitute/VariationalNeuralAnnealing) natively onto
CarbonIQ's binary decision space z in {0, 1}^N.
"""

import math
import time
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from src.qubo_builders import QUBOInstance
from src.solvers import SolverResult


class DilatedBinaryRNN(nn.Module):
    """
    Site-specific (no weight sharing) multi-layer Dilated RNN modeling the
    normalized autoregressive dispatch distribution:
        pi_theta(z) = prod_{p=1}^N pi_theta(z_p | z_1, ..., z_{p-1})
    """

    def __init__(self, n_vars: int, hidden_dim: int = 24, n_layers: Optional[int] = None):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers if n_layers is not None else max(1, int(math.ceil(math.log2(max(2, n_vars)))))

        self.W_in = nn.ParameterList()
        self.W_rec = nn.ParameterList()
        self.b_cell = nn.ParameterList()
        for layer in range(self.n_layers):
            in_dim = 2 if layer == 0 else hidden_dim
            self.W_in.append(nn.Parameter(torch.empty(n_vars, in_dim, 2 * hidden_dim)))
            self.W_rec.append(nn.Parameter(torch.empty(n_vars, hidden_dim, 2 * hidden_dim)))
            self.b_cell.append(nn.Parameter(torch.zeros(n_vars, 2 * hidden_dim)))
            nn.init.xavier_uniform_(self.W_in[-1])
            nn.init.xavier_uniform_(self.W_rec[-1])

        self.W_out = nn.Parameter(torch.empty(n_vars, hidden_dim, 1))
        self.b_out = nn.Parameter(torch.zeros(n_vars, 1))
        nn.init.xavier_uniform_(self.W_out)

    def sample_and_log_prob(self, batch_size: int, device: torch.device):
        N = self.n_vars
        H = self.hidden_dim
        h_hist = [[None] * N for _ in range(self.n_layers)]
        z_samples = []
        log_probs = []

        prev_onehot = torch.zeros(batch_size, 2, device=device)

        for p in range(N):
            x_cur = prev_onehot
            for l in range(self.n_layers):
                dil = 1 << l
                prev_idx = p - dil
                if prev_idx >= 0:
                    h_prev = h_hist[l][prev_idx]
                else:
                    h_prev = torch.zeros(batch_size, H, device=device)

                pre = x_cur @ self.W_in[l][p] + h_prev @ self.W_rec[l][p] + self.b_cell[l][p]
                u_gate, cand = torch.chunk(pre, 2, dim=-1)
                u_gate = torch.sigmoid(u_gate)
                cand = torch.tanh(cand)
                h_new = (1.0 - u_gate) * h_prev + u_gate * cand
                h_hist[l][p] = h_new
                x_cur = h_new

            logit = (x_cur @ self.W_out[p] + self.b_out[p]).squeeze(-1)
            prob1 = torch.sigmoid(logit).clamp(1e-6, 1.0 - 1e-6)
            with torch.no_grad():
                z_p = torch.bernoulli(prob1)

            lp_p = z_p * torch.log(prob1) + (1.0 - z_p) * torch.log(1.0 - prob1)
            z_samples.append(z_p)
            log_probs.append(lp_p)
            prev_onehot = torch.stack([1.0 - z_p, z_p], dim=-1)

        Z = torch.stack(z_samples, dim=1)
        log_pi = torch.stack(log_probs, dim=1).sum(1)
        return Z, log_pi


def solve_binary_vca(
    inst: QUBOInstance,
    batch_size: int = 64,
    hidden_dim: int = 20,
    n_warmup: int = 40,
    n_anneal: int = 120,
    n_train: int = 2,
    chi0: Optional[float] = None,
    lr: float = 5e-3,
    seed: int = 42,
    device: str = "cpu",
) -> SolverResult:
    """Solve canonical QUBO via Binary VCA (Eqs. 15-17) with Adam optimizer.
    Set device='auto' to automatically accelerate on CUDA GPUs (e.g., Google Colab).
    """
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    device = torch_device

    N = inst.n_vars
    Q_np = inst.Q.astype(np.float32)
    Q_t = torch.tensor(Q_np, dtype=torch.float32, device=device)
    c0 = float(inst.c0)

    q_scale = float(np.max(np.abs(Q_np))) + 1e-6
    Q_norm = Q_t / q_scale
    if chi0 is None:
        chi0 = 1.5

    model = DilatedBinaryRNN(n_vars=N, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_e = np.inf
    best_z = np.zeros(N, dtype=int)

    def eval_norm_energy(Z_batch: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,ij,bj->b", Z_batch, Q_norm, Z_batch)

    def train_step(chi_s: float):
        nonlocal best_e, best_z
        optimizer.zero_grad()
        Z_b, log_pi = model.sample_and_log_prob(batch_size, device)
        with torch.no_grad():
            E_norm = eval_norm_energy(Z_b)
            E_real = (E_norm.cpu().numpy() * q_scale) + c0
            min_idx = int(np.argmin(E_real))
            if E_real[min_idx] < best_e:
                best_e = float(E_real[min_idx])
                best_z = Z_b[min_idx].cpu().numpy().astype(int).copy()

            ell_b = E_norm + chi_s * log_pi.detach()
            ell_bar = ell_b.mean()
            adv = ell_b - ell_bar

        loss = (log_pi * adv).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    for _ in range(n_warmup):
        train_step(chi0)

    for step in range(1, n_anneal + 1):
        s = step / max(1, n_anneal)
        chi_s = chi0 * (1.0 - s)
        for _ in range(n_train):
            train_step(chi_s)

    with torch.no_grad():
        Z_final, _ = model.sample_and_log_prob(batch_size * 2, device)
        E_real = inst.energy_batch(Z_final.cpu().numpy())
        min_idx = int(np.argmin(E_real))
        if E_real[min_idx] < best_e:
            best_e = float(E_real[min_idx])
            best_z = Z_final[min_idx].cpu().numpy().astype(int).copy()

    best_e = inst.energy(best_z)
    return SolverResult(
        solver_name="Binary_VCA_DilatedRNN",
        best_z=best_z,
        best_energy=best_e,
        runtime_s=time.perf_counter() - t0,
        extra={
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "n_layers": model.n_layers,
            "n_warmup": n_warmup,
            "n_anneal": n_anneal,
            "n_train": n_train,
        },
    )
