"""
Tier 3A (Exact Polynomial & Exact Reference) and Tier 3B (Simulated Annealing,
Subspace SA, Simulated Bifurcation with Spectral Relaxation Warm-Start) Solvers,
plus Domain Baselines for CarbonIQ + EcoCycle.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import heapq
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.qubo_builders import QUBOInstance, compute_street_weights


@dataclass
class SolverResult:
    solver_name: str
    best_z: np.ndarray
    best_energy: float
    runtime_s: float
    extra: Dict


def solve_dijkstra_p5(
    df_road: pd.DataFrame,
    origin: int = 0,
    dest: Optional[int] = None,
    use_emission_weights: bool = True,
) -> Dict:
    """
    Exact O(|E| + |V| log |V|) Dijkstra shortest path for P5 (Eq. 9).
    Set use_emission_weights=False for the Shortest-GPS physical distance baseline.
    """
    t0 = time.perf_counter()
    W = compute_street_weights(df_road)
    dists = df_road["distance_km"].to_numpy(dtype=np.float64)
    from_n = df_road["from_node"].to_numpy(dtype=int)
    to_n = df_road["to_node"].to_numpy(dtype=int)
    n_nodes = int(max(from_n.max(), to_n.max()) + 1)
    if dest is None:
        dest = n_nodes - 1

    adj: List[List[Tuple[int, float, int]]] = [[] for _ in range(n_nodes)]
    for e_idx in range(len(df_road)):
        cost = float(W[e_idx] if use_emission_weights else dists[e_idx])
        adj[from_n[e_idx]].append((to_n[e_idx], cost, e_idx))

    dist_arr = np.full(n_nodes, np.inf)
    dist_arr[origin] = 0.0
    prev_node = np.full(n_nodes, -1, dtype=int)
    prev_edge = np.full(n_nodes, -1, dtype=int)
    pq = [(0.0, origin)]

    while pq:
        d_cur, u = heapq.heappop(pq)
        if d_cur > dist_arr[u]:
            continue
        if u == dest:
            break
        for v, w_uv, e_idx in adj[u]:
            nd = d_cur + w_uv
            if nd < dist_arr[v]:
                dist_arr[v] = nd
                prev_node[v] = u
                prev_edge[v] = e_idx
                heapq.heappush(pq, (nd, v))

    edges_used = []
    nodes_path = []
    cur = dest
    if np.isfinite(dist_arr[dest]):
        while cur != -1:
            nodes_path.append(cur)
            if prev_edge[cur] != -1:
                edges_used.append(int(prev_edge[cur]))
            cur = int(prev_node[cur])
        nodes_path.reverse()
        edges_used.reverse()

    p_vec = np.zeros(len(df_road), dtype=int)
    for e_idx in edges_used:
        p_vec[e_idx] = 1

    elapsed = time.perf_counter() - t0
    return {
        "feasible": len(nodes_path) > 0 and nodes_path[0] == origin,
        "nodes_path": nodes_path,
        "edges_used": edges_used,
        "binary_vec": p_vec,
        "emission_cost": round(float(W @ p_vec), 4),
        "distance_km": round(float(dists @ p_vec), 4),
        "runtime_s": elapsed,
    }


def solve_hungarian_p5(
    df_road: pd.DataFrame,
    df_fleet: pd.DataFrame,
    df_requests: pd.DataFrame,
) -> Dict:
    """
    Exact O(n_bar^3) Hungarian 1-to-1 collector-to-household matching (Eq. 10),
    compared against a Greedy Nearest-Neighbor baseline.
    """
    t0 = time.perf_counter()
    collectors = df_fleet[df_fleet["vehicle_type"] == "aboboyaa"].reset_index(drop=True)
    n_match = min(len(collectors), len(df_requests))
    if n_match == 0:
        return {"n_pairs": 0, "hungarian_cost": 0.0, "greedy_cost": 0.0, "improvement_pct": 0.0, "runtime_s": 0.0}

    W_edges = compute_street_weights(df_road)
    from_n = df_road["from_node"].to_numpy(dtype=int)
    to_n = df_road["to_node"].to_numpy(dtype=int)
    n_nodes = int(max(from_n.max(), to_n.max()) + 1)
    adj = [[] for _ in range(n_nodes)]
    for e in range(len(df_road)):
        adj[from_n[e]].append((to_n[e], float(W_edges[e])))
        adj[to_n[e]].append((from_n[e], float(W_edges[e]) * 1.05))

    def shortest_all(src):
        d = np.full(n_nodes, np.inf)
        d[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            dc, u = heapq.heappop(pq)
            if dc > d[u]:
                continue
            for v, w in adj[u]:
                if dc + w < d[v]:
                    d[v] = dc + w
                    heapq.heappush(pq, (d[v], v))
        return d

    cost_mat = np.zeros((n_match, n_match), dtype=np.float64)
    for k in range(n_match):
        src = int((int(collectors.loc[k, "start_node"]) + k) % n_nodes)
        d_all = shortest_all(src)
        for h in range(n_match):
            dst = int(df_requests.loc[h, "node_id"])
            cost_mat[k, h] = d_all[dst]

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    hungarian_cost = float(cost_mat[row_ind, col_ind].sum())
    t_hung = time.perf_counter() - t0

    assigned_h = set()
    greedy_cost = 0.0
    for k in range(n_match):
        order = np.argsort(cost_mat[k])
        for h in order:
            if int(h) not in assigned_h:
                assigned_h.add(int(h))
                greedy_cost += float(cost_mat[k, h])
                break

    improvement_pct = 100.0 * (greedy_cost - hungarian_cost) / max(greedy_cost, 1e-9)
    return {
        "n_pairs": n_match,
        "hungarian_cost": round(hungarian_cost, 4),
        "greedy_cost": round(greedy_cost, 4),
        "improvement_pct": round(improvement_pct, 2),
        "runtime_s": t_hung,
    }


def solve_exact_qubo(inst: QUBOInstance, max_n: int = 18, chunk_size: int = 16384) -> SolverResult:
    """Exact bitstring enumeration over all 2^N binary states for N <= max_n."""
    N = inst.n_vars
    if N > max_n:
        raise ValueError(f"Exact 2^N enumeration disabled for N={N} > max_n={max_n}.")
    t0 = time.perf_counter()
    Q = inst.Q
    c0 = inst.c0
    total_states = 1 << N
    best_e = np.inf
    best_idx = 0

    bit_shifts = np.arange(N, dtype=np.uint32)
    for start in range(0, total_states, chunk_size):
        end = min(total_states, start + chunk_size)
        ints = np.arange(start, end, dtype=np.uint32)[:, None]
        Z = ((ints >> bit_shifts) & 1).astype(np.float64)
        energies = np.einsum("bi,ij,bj->b", Z, Q, Z) + c0
        local_min = int(np.argmin(energies))
        if energies[local_min] < best_e:
            best_e = float(energies[local_min])
            best_idx = start + local_min

    best_z = ((best_idx >> bit_shifts) & 1).astype(int)
    return SolverResult(
        solver_name="Exact_Enumeration",
        best_z=best_z,
        best_energy=best_e,
        runtime_s=time.perf_counter() - t0,
        extra={"states_evaluated": total_states},
    )


def solve_exact_vpp_subspace(inst: QUBOInstance) -> SolverResult:
    """Exact enumeration over the cardinality-preserving feasible manifold T^|A| for P2."""
    t0 = time.perf_counter()
    meta = inst.metadata
    n_app, n_slots = meta["n_app"], meta["n_slots"]
    P, kappa, P_target = meta["P"], meta["kappa"], meta["P_target"]
    lam_solar = meta["lam_solar"]

    total = n_slots ** n_app
    if total > 500_000:
        raise ValueError(f"Feasible VPP manifold T^|A|={total} exceeds exact enumeration cap.")

    idx = np.arange(total, dtype=np.int64)
    slots = np.zeros((total, n_app), dtype=np.int64)
    rem = idx.copy()
    for a in range(n_app - 1, -1, -1):
        slots[:, a] = rem % n_slots
        rem //= n_slots

    loads = np.zeros((total, n_slots), dtype=np.float64)
    for a in range(n_app):
        np.add.at(loads, (np.arange(total), slots[:, a]), P[a])

    carbon = loads @ kappa
    solar_pen = lam_solar * np.sum((loads - P_target[None, :]) ** 2, axis=1)
    energies = carbon + solar_pen
    best_i = int(np.argmin(energies))

    best_z = np.zeros(n_app * n_slots, dtype=int)
    for a in range(n_app):
        best_z[a * n_slots + int(slots[best_i, a])] = 1

    exact_e = inst.energy(best_z)
    return SolverResult(
        solver_name="Exact_VPP_Manifold",
        best_z=best_z,
        best_energy=exact_e,
        runtime_s=time.perf_counter() - t0,
        extra={"feasible_states_evaluated": int(total)},
    )


def solve_simulated_annealing(
    inst: QUBOInstance,
    n_sweeps: int = 800,
    n_restarts: int = 4,
    theta0: Optional[float] = None,
    theta_f: float = 0.05,
    seed: int = 42,
) -> SolverResult:
    """Fast incremental O(N)-per-flip Simulated Annealing with geometric cooling (Eq. 11)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    N = inst.n_vars
    Q = inst.Q
    Q_sym = Q + Q.T - np.diag(np.diag(Q))
    diag_Q = np.diag(Q).copy()

    if theta0 is None:
        scale_est = float(np.mean(np.abs(diag_Q)) + 0.5 * np.mean(np.sum(np.abs(Q_sym - np.diag(diag_Q)), axis=1)))
        theta0 = max(5.0, scale_est)

    temps = theta0 * ((theta_f / theta0) ** (np.arange(n_sweeps) / max(1, n_sweeps - 1)))
    best_z = np.zeros(N, dtype=int)
    best_e = np.inf

    for _ in range(n_restarts):
        z = rng.integers(0, 2, size=N).astype(np.int8)
        e = inst.energy(z)
        Q_off = Q_sym.copy()
        np.fill_diagonal(Q_off, 0.0)
        field = diag_Q + Q_off @ z

        if e < best_e:
            best_e = e
            best_z = z.astype(int).copy()

        for m in range(n_sweeps):
            inv_T = 1.0 / temps[m]
            order = rng.permutation(N)
            for i in order:
                sign = 1.0 - 2.0 * z[i]
                de = sign * field[i]
                if de <= 0.0 or rng.random() < np.exp(-de * inv_T):
                    z[i] = 1 - z[i]
                    e += de
                    field += sign * Q_off[:, i]
                    if e < best_e:
                        best_e = e
                        best_z = z.astype(int).copy()

    best_e = inst.energy(best_z)
    return SolverResult(
        solver_name="Simulated_Annealing",
        best_z=best_z,
        best_energy=best_e,
        runtime_s=time.perf_counter() - t0,
        extra={"n_sweeps": n_sweeps, "n_restarts": n_restarts},
    )


def solve_subspace_sa_vpp(
    inst: QUBOInstance,
    n_sweeps: int = 600,
    n_restarts: int = 4,
    seed: int = 42,
) -> SolverResult:
    """Cardinality-preserving Subspace SA over feasible slot coordinates t in T^|A| for P2."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    meta = inst.metadata
    n_app, n_slots = meta["n_app"], meta["n_slots"]
    P, kappa, P_target = meta["P"], meta["kappa"], meta["P_target"]
    lam_solar = meta["lam_solar"]

    def eval_slots(t_vec):
        loads = np.zeros(n_slots, dtype=np.float64)
        for a in range(n_app):
            loads[t_vec[a]] += P[a]
        return float(loads @ kappa + lam_solar * np.sum((loads - P_target) ** 2))

    theta0, theta_f = 50.0, 0.02
    temps = theta0 * ((theta_f / theta0) ** (np.arange(n_sweeps) / max(1, n_sweeps - 1)))
    best_t = np.zeros(n_app, dtype=int)
    best_e = np.inf

    for _ in range(n_restarts):
        t_vec = rng.integers(0, n_slots, size=n_app)
        e = eval_slots(t_vec)
        if e < best_e:
            best_e = e
            best_t = t_vec.copy()

        for m in range(n_sweeps):
            inv_T = 1.0 / temps[m]
            for a in range(n_app):
                old_slot = int(t_vec[a])
                new_slot = int(rng.integers(0, n_slots - 1))
                if new_slot >= old_slot:
                    new_slot += 1
                t_vec[a] = new_slot
                new_e = eval_slots(t_vec)
                de = new_e - e
                if de <= 0.0 or rng.random() < np.exp(-de * inv_T):
                    e = new_e
                    if e < best_e:
                        best_e = e
                        best_t = t_vec.copy()
                else:
                    t_vec[a] = old_slot

    best_z = np.zeros(n_app * n_slots, dtype=int)
    for a in range(n_app):
        best_z[a * n_slots + int(best_t[a])] = 1
    return SolverResult(
        solver_name="Subspace_SA_VPP",
        best_z=best_z,
        best_energy=inst.energy(best_z),
        runtime_s=time.perf_counter() - t0,
        extra={"n_sweeps": n_sweeps, "n_restarts": n_restarts},
    )


def solve_simulated_bifurcation(
    inst: QUBOInstance,
    n_steps: int = 1200,
    n_trials: int = 16,
    dt: float = 0.08,
    seed: int = 42,
) -> SolverResult:
    """
    Ballistic Simulated Bifurcation (Goto et al., 2019) integrated via
    semi-implicit Euler-Cromer updates (Eqs. 12-14) with Spectral Relaxation
    Warm-Start from the lowest eigenmode of the symmetrized Ising matrix.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    N = inst.n_vars
    Q = inst.Q

    Q_sym = 0.5 * (Q + Q.T)
    Q_off = Q_sym.copy()
    np.fill_diagonal(Q_off, 0.0)
    h_vec = 0.5 * np.diag(Q) + 0.5 * (Q_off @ np.ones(N))

    spec_bound = float(np.max(np.sum(np.abs(Q_off), axis=1))) + 1e-6
    M_shift = spec_bound * np.eye(N) - Q_off
    psi = rng.normal(size=N)
    psi /= np.linalg.norm(psi) + 1e-12
    for _ in range(35):
        psi = M_shift @ psi
        norm = np.linalg.norm(psi)
        if norm > 1e-12:
            psi /= norm
    psi_0 = psi / (np.max(np.abs(psi)) + 1e-12)

    frob = float(np.linalg.norm(Q_off))
    scale_denom = max(frob / np.sqrt(max(N, 1)), float(np.linalg.norm(h_vec)) / np.sqrt(max(N, 1)), 1e-6)
    rho = 0.7 / scale_denom

    Z_tilde = rng.uniform(-0.15, 0.15, size=(n_trials, N))
    Z_tilde[0] = 0.25 * psi_0
    Z_tilde[1] = -0.25 * psi_0
    V_mom = rng.uniform(-0.05, 0.05, size=(n_trials, N))

    best_z = np.zeros(N, dtype=int)
    best_e = np.inf

    for m in range(n_steps):
        vartheta = m / max(1, n_steps - 1)
        grad = 0.5 * (Z_tilde @ Q_off) + h_vec[None, :]
        V_mom -= dt * ((1.0 - vartheta) * Z_tilde + rho * grad)
        Z_tilde += dt * V_mom
        hit = np.abs(Z_tilde) >= 1.0
        Z_tilde = np.clip(Z_tilde, -1.0, 1.0)
        V_mom[hit] = 0.0

        if m >= int(0.75 * n_steps) and (m % 25 == 0 or m == n_steps - 1):
            Z_bin = (Z_tilde > 0.0).astype(np.float64)
            energies = inst.energy_batch(Z_bin)
            idx_min = int(np.argmin(energies))
            if energies[idx_min] < best_e:
                best_e = float(energies[idx_min])
                best_z = Z_bin[idx_min].astype(int).copy()

    for _ in range(2):
        improved = False
        for i in range(N):
            cand = best_z.copy()
            cand[i] = 1 - cand[i]
            ce = inst.energy(cand)
            if ce < best_e - 1e-12:
                best_e = ce
                best_z = cand
                improved = True
        if not improved:
            break

    return SolverResult(
        solver_name="Simulated_Bifurcation_SRWS",
        best_z=best_z,
        best_energy=best_e,
        runtime_s=time.perf_counter() - t0,
        extra={"n_steps": n_steps, "n_trials": n_trials, "spectral_norm_bound": round(spec_bound, 4)},
    )


def baseline_p2_uncoordinated(inst: QUBOInstance) -> SolverResult:
    """Default uncoordinated evening appliance operation baseline."""
    t0 = time.perf_counter()
    meta = inst.metadata
    n_app, n_slots, hours = meta["n_app"], meta["n_slots"], meta["hours"]
    z = np.zeros(n_app * n_slots, dtype=int)
    for a, dhr in enumerate(meta["default_hours"]):
        t_idx = int(np.argmin(np.abs(hours - dhr)))
        z[a * n_slots + t_idx] = 1
    return SolverResult(
        solver_name="Baseline_Uncoordinated",
        best_z=z,
        best_energy=inst.energy(z),
        runtime_s=time.perf_counter() - t0,
        extra={},
    )


def baseline_p4_nar_greedy(inst: QUBOInstance) -> SolverResult:
    """NAR Greedy heuristic (matching Q-Energic baseline)."""
    t0 = time.perf_counter()
    meta = inst.metadata
    C_raw, H_raw = meta["C_raw"], meta["H_raw"]
    budget = meta["budget_usd"]
    target_hubs = meta["target_hubs"]

    ratios = H_raw / np.maximum(C_raw, 1.0)
    order = np.argsort(-ratios)
    z = np.zeros(inst.n_vars, dtype=int)
    spent = 0.0
    count = 0
    for idx in order:
        if count >= target_hubs:
            break
        if spent + C_raw[idx] <= budget:
            z[idx] = 1
            spent += C_raw[idx]
            count += 1
    return SolverResult(
        solver_name="Baseline_NAR_Greedy",
        best_z=z,
        best_energy=inst.energy(z),
        runtime_s=time.perf_counter() - t0,
        extra={"spent_usd": spent, "selected": count},
    )
