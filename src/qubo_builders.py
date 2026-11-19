"""
Grounded QUBO Builders and MRV Ledger for CarbonIQ + EcoCycle.

Maps standardized CSV schemas directly into canonical upper-triangular QUBO
instances:
    J(z) = z^T Q z + c_0,    z in {0, 1}^N

Corresponds directly to Equations (1)-(10) in the manuscript:
  - Eq. (1): Deterministic Tier 1-2 MRV Carbon Credit Ledger
  - Eqs. (2)-(3): ST-GNN Dynamic Street Weights W_ij
  - Eq. (5): P1 Multi-Collector Roaming Waste Routing
  - Eq. (6): P2 Virtual Power Plant (VPP) Appliance & EV Scheduling
  - Eq. (7): P3 SmartTransyt EV Corridor Allocation (with Aayalolo QBS)
  - Eq. (8): P4 Budgeted BSF Hub & Charging Depot Siting
  - Eqs. (9)-(10): P5 Clean-Air Navigation & One-to-One Matching Inputs
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class QUBOInstance:
    problem_id: str
    name: str
    Q: np.ndarray              # Upper-triangular (N, N) float64 matrix
    c0: float                  # Constant offset
    var_names: List[str]
    metadata: Dict = field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        return int(self.Q.shape[0])

    def energy(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=np.float64)
        return float(z @ self.Q @ z + self.c0)

    def energy_batch(self, Z: np.ndarray) -> np.ndarray:
        """Vectorized evaluation for batch Z of shape (B, N)."""
        Z = np.asarray(Z, dtype=np.float64)
        return np.einsum("bi,ij,bj->b", Z, self.Q, Z) + self.c0


def add_linear(Q: np.ndarray, idx: int, coeff: float) -> None:
    Q[idx, idx] += coeff


def add_quadratic(Q: np.ndarray, i: int, j: int, coeff: float) -> None:
    if i == j:
        Q[i, i] += coeff
    elif i < j:
        Q[i, j] += coeff
    else:
        Q[j, i] += coeff


def add_squared_constraint(
    Q: np.ndarray,
    c0: float,
    terms: List[Tuple[int, float]],
    rhs: float,
    lam: float,
) -> float:
    """
    Add lam * (sum_k coeff_k * z_{idx_k} - rhs)^2 into upper-triangular Q and offset c0.
    Merges duplicate variable indices automatically before expanding.
    """
    merged: Dict[int, float] = {}
    for idx, coeff in terms:
        merged[idx] = merged.get(idx, 0.0) + float(coeff)

    items = [(i, c) for i, c in merged.items() if c != 0.0]
    c0 += lam * (rhs ** 2)
    for idx_a, (i, ci) in enumerate(items):
        Q[i, i] += lam * (ci * ci - 2.0 * rhs * ci)
        for j, cj in items[idx_a + 1:]:
            add_quadratic(Q, i, j, 2.0 * lam * ci * cj)
    return c0


def bounded_binary_weights(upper_bound: int) -> List[int]:
    """
    Exact bounded-integer binary encoding weights a_r such that
    sum_r a_r * xi_r spans every integer in [0, upper_bound] with minimal bits.
    """
    ub = int(max(0, upper_bound))
    if ub == 0:
        return []
    weights = []
    power = 1
    rem = ub
    while rem >= power:
        weights.append(power)
        rem -= power
        power *= 2
    if rem > 0:
        weights.append(rem)
    return weights


def compute_mrv_credits(
    transit_trips: List[Dict],
    waste_logs: List[Dict],
    appliance_logs: List[Dict],
    e_car: float = 0.192,
    eta_transit: float = 0.85,
) -> Dict[str, float]:
    """Deterministic O(N) MRV credit calculation (Equation 1)."""
    transit_kg = 0.0
    for tr in transit_trips:
        e_veh = float(tr["e_veh_kg_km"])
        n_pax = max(1, int(tr["occupancy"]))
        dist = float(tr["distance_km"])
        transit_kg += eta_transit * max(0.0, e_car - e_veh / n_pax) * dist

    waste_kg = 0.0
    for wl in waste_logs:
        gate_ok = 1 if wl.get("gate_confirmed", False) else 0
        hub_ok = 1 if wl.get("hub_weighbridge_verified", False) else 0
        waste_kg += float(wl["gamma_kg_co2_per_kg"]) * (gate_ok * hub_ok) * float(wl["mass_kg"])

    energy_kg = 0.0
    for al in appliance_logs:
        saved_kwh = max(0.0, float(al["base_kwh"]) - float(al["actual_kwh"]))
        energy_kg += float(al["kappa_grid_kg_kwh"]) * saved_kwh

    return {
        "transit_kg_co2e": round(transit_kg, 4),
        "waste_kg_co2e": round(waste_kg, 4),
        "energy_kg_co2e": round(energy_kg, 4),
        "total_cc_kg_co2e": round(transit_kg + waste_kg + energy_kg, 4),
    }


def compute_street_weights(
    df_road: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 0.08,
    phi_bc: float = 0.65,
    phi_nx: float = 0.35,
) -> np.ndarray:
    """Compute strictly positive composite edge weights W_ij from road CSV (Eqs. 2-3)."""
    d = df_road["distance_km"].to_numpy(dtype=np.float64)
    vmax = df_road["speed_limit_kmh"].to_numpy(dtype=np.float64)
    tau_hat = df_road["pred_travel_hr"].to_numpy(dtype=np.float64)
    iota = np.maximum(1.0, (vmax * tau_hat) / np.maximum(d, 1e-6))
    pm25 = df_road["pm25_ug_m3"].to_numpy(dtype=np.float64)
    nox = df_road["nox_ug_m3"].to_numpy(dtype=np.float64)
    return alpha * d + beta * iota * (phi_bc * pm25 + phi_nx * nox)


def build_p1_routing_qubo(
    df_road: pd.DataFrame,
    n_vehicles: int = 1,
    origin_node: int = 0,
    dest_node: Optional[int] = None,
    lam_flow: Optional[float] = None,
    lam_cong: float = 4.0,
    df_requests: Optional[pd.DataFrame] = None,
    payload_cap_kg: Optional[int] = None,
    lam_cap: float = 0.0,
) -> QUBOInstance:
    """Build Problem P1 QUBO (Eq. 5)."""
    W = compute_street_weights(df_road)
    from_nodes = df_road["from_node"].to_numpy(dtype=int)
    to_nodes = df_road["to_node"].to_numpy(dtype=int)
    n_edges = len(df_road)
    n_nodes = int(max(from_nodes.max(), to_nodes.max()) + 1)
    if dest_node is None:
        dest_node = n_nodes - 1

    node_weights = np.zeros(n_nodes, dtype=int)
    if df_requests is not None and lam_cap > 0.0 and payload_cap_kg is not None:
        for _, row in df_requests.iterrows():
            nid = int(row["node_id"])
            if 0 <= nid < n_nodes:
                node_weights[nid] += int(row["est_weight_kg"])

    slack_weights = (
        bounded_binary_weights(payload_cap_kg)
        if (lam_cap > 0.0 and payload_cap_kg is not None)
        else []
    )
    n_slack_per_veh = len(slack_weights)
    n_edge_vars = n_vehicles * n_edges
    n_total = n_edge_vars + n_vehicles * n_slack_per_veh

    if lam_flow is None:
        lam_flow = float(np.ceil(float(np.sum(W)) * 0.65))

    Q = np.zeros((n_total, n_total), dtype=np.float64)
    c0 = 0.0
    var_names = []
    for k in range(n_vehicles):
        for e in range(n_edges):
            var_names.append(f"x_{from_nodes[e]}_{to_nodes[e]}_k{k}")
    for k in range(n_vehicles):
        for r in range(n_slack_per_veh):
            var_names.append(f"xi_k{k}_r{r}")

    for k in range(n_vehicles):
        for e in range(n_edges):
            add_linear(Q, k * n_edges + e, W[e])

    for k in range(n_vehicles):
        for i in range(n_nodes):
            b_ik = 1.0 if i == origin_node else (-1.0 if i == dest_node else 0.0)
            terms = []
            for e in range(n_edges):
                if from_nodes[e] == i:
                    terms.append((k * n_edges + e, 1.0))
                if to_nodes[e] == i:
                    terms.append((k * n_edges + e, -1.0))
            c0 = add_squared_constraint(Q, c0, terms, b_ik, lam_flow)

    if n_vehicles > 1 and lam_cong > 0.0:
        for e in range(n_edges):
            for k1 in range(n_vehicles):
                for k2 in range(k1 + 1, n_vehicles):
                    add_quadratic(Q, k1 * n_edges + e, k2 * n_edges + e, lam_cong)

    if lam_cap > 0.0 and payload_cap_kg is not None and n_slack_per_veh > 0:
        for k in range(n_vehicles):
            terms = []
            for e in range(n_edges):
                wj = float(node_weights[to_nodes[e]])
                if wj > 0.0:
                    terms.append((k * n_edges + e, wj))
            base_s = n_edge_vars + k * n_slack_per_veh
            for r, ar in enumerate(slack_weights):
                terms.append((base_s + r, float(ar)))
            c0 = add_squared_constraint(Q, c0, terms, float(payload_cap_kg), lam_cap)

    return QUBOInstance(
        problem_id="P1",
        name=f"P1_routing_{n_nodes}n_{n_edges}e_{n_vehicles}v",
        Q=Q,
        c0=c0,
        var_names=var_names,
        metadata={
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_vehicles": n_vehicles,
            "origin": origin_node,
            "dest": dest_node,
            "from_nodes": from_nodes,
            "to_nodes": to_nodes,
            "W": W,
            "distances": df_road["distance_km"].to_numpy(dtype=np.float64),
            "lam_flow": lam_flow,
            "lam_cong": lam_cong,
            "lam_cap": lam_cap,
            "payload_cap_kg": payload_cap_kg,
            "node_weights": node_weights,
        },
    )


def evaluate_p1_solution(inst: QUBOInstance, z: np.ndarray) -> Dict:
    meta = inst.metadata
    n_edges = meta["n_edges"]
    n_nodes = meta["n_nodes"]
    n_veh = meta["n_vehicles"]
    from_n = meta["from_nodes"]
    to_n = meta["to_nodes"]
    W = meta["W"]
    dists = meta["distances"]

    flow_residual = 0.0
    emission_cost = 0.0
    total_dist = 0.0
    edge_counts = np.zeros(n_edges, dtype=int)

    for k in range(n_veh):
        xk = np.asarray(z[k * n_edges : (k + 1) * n_edges], dtype=int)
        edge_counts += xk
        emission_cost += float(W @ xk)
        total_dist += float(dists @ xk)
        for i in range(n_nodes):
            b_ik = 1 if i == meta["origin"] else (-1 if i == meta["dest"] else 0)
            net = int(np.sum(xk[from_n == i]) - np.sum(xk[to_n == i]))
            flow_residual += (net - b_ik) ** 2

    shared_edges = int(np.sum(edge_counts > 1))
    return {
        "feasible": flow_residual == 0,
        "flow_residual_sq": float(flow_residual),
        "emission_cost": round(emission_cost, 4),
        "distance_km": round(total_dist, 4),
        "shared_corridor_edges": shared_edges,
        "qubo_energy": round(inst.energy(z), 4),
    }


def build_p2_vpp_qubo(
    df_vpp: pd.DataFrame,
    df_grid: pd.DataFrame,
    lam_solar: float = 0.05,
    lam_onehot: Optional[float] = None,
) -> QUBOInstance:
    """Build Problem P2 QUBO (Eq. 6)."""
    P = df_vpp["rated_power_kw"].to_numpy(dtype=np.float64)
    kappa = df_grid["carbon_intensity_kg_kwh"].to_numpy(dtype=np.float64)
    P_target = df_grid["solar_target_kw"].to_numpy(dtype=np.float64)
    hours = df_grid["hour_gmt"].to_numpy(dtype=int)
    n_app = len(P)
    n_slots = len(kappa)
    n_vars = n_app * n_slots

    if lam_onehot is None:
        max_dev = float(np.max(P) ** 2 + 2.0 * np.max(P) * np.max(P_target))
        lam_onehot = float(np.ceil(max(250.0, 2.5 * lam_solar * max_dev)))

    Q = np.zeros((n_vars, n_vars), dtype=np.float64)
    c0 = 0.0
    var_names = [f"y_a{a}_t{t}" for a in range(n_app) for t in range(n_slots)]

    for a in range(n_app):
        for t in range(n_slots):
            add_linear(Q, a * n_slots + t, kappa[t] * P[a])

    for t in range(n_slots):
        terms = [(a * n_slots + t, P[a]) for a in range(n_app)]
        c0 = add_squared_constraint(Q, c0, terms, P_target[t], lam_solar)

    for a in range(n_app):
        terms = [(a * n_slots + t, 1.0) for t in range(n_slots)]
        c0 = add_squared_constraint(Q, c0, terms, 1.0, lam_onehot)

    return QUBOInstance(
        problem_id="P2",
        name=f"P2_vpp_{n_app}x{n_slots}",
        Q=Q,
        c0=c0,
        var_names=var_names,
        metadata={
            "n_app": n_app,
            "n_slots": n_slots,
            "P": P,
            "kappa": kappa,
            "P_target": P_target,
            "hours": hours,
            "default_hours": df_vpp["default_hour_gmt"].to_numpy(dtype=int),
            "lam_solar": lam_solar,
            "lam_onehot": lam_onehot,
        },
    )


def evaluate_p2_solution(inst: QUBOInstance, z: np.ndarray) -> Dict:
    meta = inst.metadata
    n_app, n_slots = meta["n_app"], meta["n_slots"]
    P, kappa, P_target, hours = meta["P"], meta["kappa"], meta["P_target"], meta["hours"]
    Y = np.asarray(z, dtype=int).reshape(n_app, n_slots)
    row_sums = Y.sum(axis=1)
    onehot_viol = int(np.sum(np.abs(row_sums - 1)))
    slot_loads = P @ Y
    carbon_kg = float(np.sum(slot_loads * kappa))
    solar_mse = float(np.mean((slot_loads - P_target) ** 2))

    eve_mask = hours >= 17
    eve_load_kw = float(np.sum(slot_loads[eve_mask]))

    base_loads = np.zeros(n_slots, dtype=np.float64)
    for a, dhr in enumerate(meta["default_hours"]):
        idx = int(np.argmin(np.abs(hours - dhr)))
        base_loads[idx] += P[a]
    base_carbon = float(np.sum(base_loads * kappa))
    base_eve_kw = float(np.sum(base_loads[eve_mask]))
    peak_shaving_pct = 100.0 * (base_eve_kw - eve_load_kw) / max(base_eve_kw, 1e-6)
    carbon_red_pct = 100.0 * (base_carbon - carbon_kg) / max(base_carbon, 1e-6)

    return {
        "feasible": onehot_viol == 0,
        "onehot_violations": onehot_viol,
        "carbon_kg_co2e": round(carbon_kg, 4),
        "baseline_carbon_kg_co2e": round(base_carbon, 4),
        "carbon_reduction_pct": round(carbon_red_pct, 2),
        "evening_load_kw": round(eve_load_kw, 2),
        "baseline_evening_kw": round(base_eve_kw, 2),
        "evening_peak_shaving_pct": round(peak_shaving_pct, 2),
        "solar_rmse_kw": round(float(np.sqrt(solar_mse)), 2),
        "qubo_energy": round(inst.energy(z), 4),
    }


def build_p3_ev_corridor_qubo(
    df_corr: pd.DataFrame,
    df_fleet: pd.DataFrame,
    omega_disp: float = 12.0,
    lam_single: Optional[float] = None,
    lam_range: Optional[float] = None,
    max_buses: Optional[int] = None,
    use_range_slack: bool = True,
) -> QUBOInstance:
    """
    Build Problem P3 QUBO coordinating SmartTransyt EVs with Aayalolo QBS
    residual demand D_ct = max(0, D_ct^0 - Q_ct^0) (Eq. 7).
    """
    ev_df = df_fleet[df_fleet["vehicle_type"] == "smarttransyt_ev"].reset_index(drop=True)
    if max_buses is not None:
        ev_df = ev_df.iloc[:max_buses].reset_index(drop=True)
    n_ev = len(ev_df)
    q_cap = float(ev_df["passenger_cap"].iloc[0]) * 2.0
    range_unit_km = 10
    R_units = [max(1, int(r // range_unit_km)) for r in ev_df["battery_or_fuel_range_km"]]

    corridors = list(df_corr["corridor_id"].unique())
    hours = sorted(list(df_corr["hour_gmt"].unique()))
    n_c = len(corridors)
    n_t = len(hours)
    c_map = {c: i for i, c in enumerate(corridors)}
    t_map = {h: i for i, h in enumerate(hours)}

    D_ct = np.zeros((n_c, n_t), dtype=np.float64)
    L_c = np.zeros(n_c, dtype=np.float64)
    e_c = np.zeros(n_c, dtype=np.float64)
    for _, row in df_corr.iterrows():
        ci = c_map[row["corridor_id"]]
        ti = t_map[int(row["hour_gmt"])]
        D_ct[ci, ti] = float(row["residual_demand_pax_hr"])
        L_c[ci] = float(row["round_trip_km"])
        e_c[ci] = float(row["diesel_trotro_kg_co2_km"])

    L_units = np.maximum(1, np.round(L_c / range_unit_km).astype(int))
    n_f = n_ev * n_c * n_t

    slack_lists = []
    if use_range_slack:
        max_poss = int(n_t * np.max(L_units))
        for n in range(n_ev):
            eff_r = min(R_units[n], max_poss)
            R_units[n] = eff_r
            slack_lists.append(bounded_binary_weights(eff_r))
    else:
        slack_lists = [[] for _ in range(n_ev)]

    n_slack = sum(len(s) for s in slack_lists)
    n_vars = n_f + n_slack
    Q = np.zeros((n_vars, n_vars), dtype=np.float64)
    c0 = 0.0

    def f_idx(n, ci, ti):
        return (n * n_c + ci) * n_t + ti

    var_names = [
        f"f_n{n}_c{ci}_t{ti}"
        for n in range(n_ev) for ci in range(n_c) for ti in range(n_t)
    ]
    slack_offsets = []
    cur = n_f
    for n, sl in enumerate(slack_lists):
        slack_offsets.append(cur)
        for r in range(len(sl)):
            var_names.append(f"zeta_n{n}_r{r}")
        cur += len(sl)

    for ti in range(n_t):
        for ci in range(n_c):
            terms = [(f_idx(n, ci, ti), q_cap) for n in range(n_ev)]
            c0 = add_squared_constraint(Q, c0, terms, D_ct[ci, ti], 1.0)

    for n in range(n_ev):
        for ci in range(n_c):
            for ti in range(n_t):
                add_linear(Q, f_idx(n, ci, ti), -omega_disp * L_c[ci] * e_c[ci])

    # Auto-calibrate hard physical penalties strictly above max squared demand gap
    max_dem_sq = float(np.max(D_ct) ** 2) + 1.0
    if lam_single is None:
        lam_single = float(np.ceil(2.5 * max_dem_sq))
    if lam_range is None:
        lam_range = float(np.ceil(0.8 * max_dem_sq))

    for ti in range(n_t):
        for n in range(n_ev):
            for c1 in range(n_c):
                for c2 in range(c1 + 1, n_c):
                    add_quadratic(Q, f_idx(n, c1, ti), f_idx(n, c2, ti), lam_single)

    if use_range_slack and lam_range > 0.0:
        for n in range(n_ev):
            terms = []
            for ti in range(n_t):
                for ci in range(n_c):
                    terms.append((f_idx(n, ci, ti), float(L_units[ci])))
            for r, gr in enumerate(slack_lists[n]):
                terms.append((slack_offsets[n] + r, float(gr)))
            c0 = add_squared_constraint(Q, c0, terms, float(R_units[n]), lam_range)

    return QUBOInstance(
        problem_id="P3",
        name=f"P3_ev_{n_ev}b_{n_c}c_{n_t}t",
        Q=Q,
        c0=c0,
        var_names=var_names,
        metadata={
            "n_ev": n_ev,
            "n_c": n_c,
            "n_t": n_t,
            "q_cap": q_cap,
            "D_ct": D_ct,
            "L_c": L_c,
            "L_units": L_units,
            "e_c": e_c,
            "R_units": R_units,
            "range_unit_km": range_unit_km,
            "corridors": corridors,
            "hours": hours,
            "lam_single": lam_single,
            "lam_range": lam_range,
        },
    )


def evaluate_p3_solution(inst: QUBOInstance, z: np.ndarray) -> Dict:
    meta = inst.metadata
    n_ev, n_c, n_t = meta["n_ev"], meta["n_c"], meta["n_t"]
    F = np.asarray(z[: n_ev * n_c * n_t], dtype=int).reshape(n_ev, n_c, n_t)

    multi_corr_viol = int(np.sum(np.maximum(0, F.sum(axis=1) - 1)))
    bus_dist_units = np.einsum("nct,c->n", F, meta["L_units"])
    range_viol = int(np.sum(np.maximum(0, bus_dist_units - np.asarray(meta["R_units"]))))
    served_ct = meta["q_cap"] * F.sum(axis=0)
    unmet_pax = float(np.sum(np.maximum(0.0, meta["D_ct"] - served_ct)))
    displaced_co2 = float(np.einsum("nct,c-> ", F, meta["L_c"] * meta["e_c"]))

    return {
        "feasible": (multi_corr_viol == 0 and range_viol == 0),
        "multi_corridor_violations": multi_corr_viol,
        "range_violations": range_viol,
        "unmet_residual_pax_hr": round(unmet_pax, 2),
        "displaced_diesel_co2_kg": round(displaced_co2, 2),
        "active_bus_hours": int(F.sum()),
        "qubo_energy": round(inst.energy(z), 4),
    }


def build_p4_hub_siting_qubo(
    df_sites: pd.DataFrame,
    budget_usd: Optional[float] = None,
    min_households: Optional[float] = None,
    target_hubs: Optional[int] = None,
    omega_cov: float = 0.35,
    omega_cap: float = 0.08,
    lam_budget: float = 1.8,
    lam_cov: float = 0.45,
    lam_hubs: float = 0.60,
    strict_budget_slack: bool = False,
) -> QUBOInstance:
    """Build Problem P4 QUBO (Eq. 8 soft-penalty or slack-augmented strict-budget)."""
    C_raw = df_sites["install_cost_usd"].to_numpy(dtype=np.float64)
    H_raw = df_sites["households_covered"].to_numpy(dtype=np.float64)
    G_raw = df_sites["daily_capacity_tonnes"].to_numpy(dtype=np.float64)
    L = len(df_sites)

    if target_hubs is None:
        target_hubs = max(3, min(10, int(round(0.28 * L))))
    if budget_usd is None:
        budget_usd = float(np.round(np.mean(C_raw) * target_hubs * 0.95 / 1000.0) * 1000.0)
    if min_households is None:
        min_households = float(np.round(np.mean(H_raw) * target_hubs * 0.80 / 100.0) * 100.0)

    c_scale, h_scale, g_scale = 1e5, 1e3, 10.0
    C, H, G = C_raw / c_scale, H_raw / h_scale, G_raw / g_scale
    B, M, K = budget_usd / c_scale, min_households / h_scale, float(target_hubs)

    slack_weights = []
    slack_unit = 0.25  # $25k resolution
    if strict_budget_slack:
        slack_weights = bounded_binary_weights(int(np.ceil(B / slack_unit)))

    n_vars = L + len(slack_weights)
    Q = np.zeros((n_vars, n_vars), dtype=np.float64)
    c0 = 0.0
    var_names = [f"s_{i}" for i in range(L)] + [f"sb_{r}" for r in range(len(slack_weights))]

    for i in range(L):
        add_linear(Q, i, C[i] - omega_cov * H[i] - omega_cap * G[i])

    eff_lam_budget = lam_budget * 3.5 if strict_budget_slack else lam_budget
    b_terms = [(i, C[i]) for i in range(L)]
    if strict_budget_slack:
        for r, wr in enumerate(slack_weights):
            b_terms.append((L + r, wr * slack_unit))
    c0 = add_squared_constraint(Q, c0, b_terms, B, eff_lam_budget)

    h_terms = [(i, H[i]) for i in range(L)]
    c0 = add_squared_constraint(Q, c0, h_terms, M, lam_cov)

    k_terms = [(i, 1.0) for i in range(L)]
    c0 = add_squared_constraint(Q, c0, k_terms, K, lam_hubs)

    return QUBOInstance(
        problem_id="P4",
        name=f"P4_siting_{L}sites" + ("_slack" if strict_budget_slack else "_soft"),
        Q=Q,
        c0=c0,
        var_names=var_names,
        metadata={
            "L": L,
            "C_raw": C_raw,
            "H_raw": H_raw,
            "G_raw": G_raw,
            "budget_usd": budget_usd,
            "min_households": min_households,
            "target_hubs": target_hubs,
            "strict_budget_slack": strict_budget_slack,
            "site_names": df_sites["neighborhood"].tolist(),
        },
    )


def evaluate_p4_solution(inst: QUBOInstance, z: np.ndarray) -> Dict:
    meta = inst.metadata
    L = meta["L"]
    s = np.asarray(z[:L], dtype=int)
    cost = float(meta["C_raw"] @ s)
    hh = float(meta["H_raw"] @ s)
    cap = float(meta["G_raw"] @ s)
    n_sel = int(s.sum())
    budget = meta["budget_usd"]
    min_hh = meta["min_households"]
    util_pct = 100.0 * cost / max(budget, 1e-6)
    strict_feasible = (cost <= budget) and (hh >= min_hh)
    near_feasible = (cost <= 1.025 * budget) and (hh >= min_hh)
    return {
        "strict_feasible": strict_feasible,
        "near_feasible_2_5pct": near_feasible,
        "selected_sites": n_sel,
        "total_cost_usd": round(cost, 2),
        "budget_utilization_pct": round(util_pct, 2),
        "households_covered": int(hh),
        "daily_capacity_tonnes": round(cap, 2),
        "qubo_energy": round(inst.energy(z), 4),
    }
