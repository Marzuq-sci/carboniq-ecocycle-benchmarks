#!/usr/bin/env python3
"""
Unified Multi-Solver Benchmark Runner for CarbonIQ + EcoCycle (Problems P1-P5).
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.qubo_builders import (
    compute_mrv_credits,
    build_p1_routing_qubo,
    evaluate_p1_solution,
    build_p2_vpp_qubo,
    evaluate_p2_solution,
    build_p3_ev_corridor_qubo,
    evaluate_p3_solution,
    build_p4_hub_siting_qubo,
    evaluate_p4_solution,
)
from src.solvers import (
    solve_dijkstra_p5,
    solve_hungarian_p5,
    solve_exact_qubo,
    solve_exact_vpp_subspace,
    solve_simulated_annealing,
    solve_subspace_sa_vpp,
    solve_simulated_bifurcation,
    baseline_p2_uncoordinated,
    baseline_p4_nar_greedy,
)
from src.binary_vca import solve_binary_vca


def run_benchmark(
    data_dir: str,
    scale: str,
    out_dir: str,
    seed: int = 42,
    run_vca: bool = True,
    vca_anneal_steps: int = 120,
):
    os.makedirs(out_dir, exist_ok=True)
    df_road = pd.read_csv(os.path.join(data_dir, "accra_road_network.csv"))
    df_req = pd.read_csv(os.path.join(data_dir, "consumer_waste_requests.csv"))
    df_fleet = pd.read_csv(os.path.join(data_dir, "fleet_and_collectors.csv"))
    df_vpp = pd.read_csv(os.path.join(data_dir, "vpp_appliances.csv"))
    df_grid = pd.read_csv(os.path.join(data_dir, "grid_hourly_profile.csv"))
    df_corr = pd.read_csv(os.path.join(data_dir, "corridor_demand_aayalolo.csv"))
    df_sites = pd.read_csv(os.path.join(data_dir, "candidate_hub_sites.csv"))

    summary_rows = []
    full_report = {"scale": scale, "data_dir": data_dir, "seed": seed, "problems": {}}

    def record(prob_id, variant, n_vars, solver_name, energy, e_gs, feasible, runtime_s, metrics_dict):
        res_per_var = ((energy - e_gs) / n_vars) if (e_gs is not None and n_vars > 0) else None
        summary_rows.append({
            "problem": prob_id,
            "variant": variant,
            "N_vars": n_vars,
            "solver": solver_name,
            "energy": round(energy, 4) if energy is not None else None,
            "exact_gs": round(e_gs, 4) if e_gs is not None else None,
            "residual_per_var": round(res_per_var, 6) if res_per_var is not None else None,
            "feasible": feasible,
            "runtime_s": round(runtime_s, 4),
            "key_metric": json.dumps(metrics_dict),
        })

    # Tier 1-2 Deterministic MRV Ledger Check (Eq. 1)
    mrv_demo = compute_mrv_credits(
        transit_trips=[
            {"mode": "trotro", "e_veh_kg_km": 0.85, "occupancy": 14, "distance_km": 12.0},
            {"mode": "aayalolo_qbs", "e_veh_kg_km": 1.15, "occupancy": 55, "distance_km": 18.0},
            {"mode": "smarttransyt_ev", "e_veh_kg_km": 0.00, "occupancy": 20, "distance_km": 10.0},
        ],
        waste_logs=[
            {"stream": "org_bsf", "gamma_kg_co2_per_kg": 1.85, "mass_kg": 12.0, "gate_confirmed": True, "hub_weighbridge_verified": True},
            {"stream": "PET_ecocycle", "gamma_kg_co2_per_kg": 1.50, "mass_kg": 6.0, "gate_confirmed": True, "hub_weighbridge_verified": True},
            {"stream": "unverified_fraud_attempt", "gamma_kg_co2_per_kg": 1.85, "mass_kg": 50.0, "gate_confirmed": True, "hub_weighbridge_verified": False},
        ],
        appliance_logs=[
            {"appliance": "smart_washer", "base_kwh": 1.8, "actual_kwh": 1.1, "kappa_grid_kg_kwh": 0.74},
        ],
    )
    full_report["mrv_ledger_sample"] = mrv_demo

    # Problem P5: Exact Polynomial Navigation & 1-to-1 Matching (Eqs. 9-10)
    gps_base = solve_dijkstra_p5(df_road, use_emission_weights=False)
    clean_air = solve_dijkstra_p5(df_road, use_emission_weights=True)
    em_red_pct = 100.0 * (gps_base["emission_cost"] - clean_air["emission_cost"]) / max(gps_base["emission_cost"], 1e-6)
    hung = solve_hungarian_p5(df_road, df_fleet, df_req)

    record("P5_nav", scale, len(df_road), "Baseline_Shortest_GPS", gps_base["emission_cost"], clean_air["emission_cost"], gps_base["feasible"], gps_base["runtime_s"], {"distance_km": gps_base["distance_km"]})
    record("P5_nav", scale, len(df_road), "Tier3A_Emission_Dijkstra", clean_air["emission_cost"], clean_air["emission_cost"], clean_air["feasible"], clean_air["runtime_s"], {"distance_km": clean_air["distance_km"], "emission_reduction_pct": round(em_red_pct, 2)})
    record("P5_match", scale, hung["n_pairs"] ** 2, "Baseline_Greedy_NN", hung["greedy_cost"], hung["hungarian_cost"], True, 0.0, {"n_pairs": hung["n_pairs"]})
    record("P5_match", scale, hung["n_pairs"] ** 2, "Tier3A_Hungarian", hung["hungarian_cost"], hung["hungarian_cost"], True, hung["runtime_s"], {"n_pairs": hung["n_pairs"], "improvement_pct": hung["improvement_pct"]})
    full_report["problems"]["P5"] = {"gps_baseline": gps_base["emission_cost"], "clean_air_dijkstra": clean_air["emission_cost"], "emission_reduction_pct": round(em_red_pct, 2), "hungarian": hung}

    # Problem P1: Multi-Collector Roaming Waste Routing (Eq. 5)
    n_veh_p1 = 2 if scale == "small" else (3 if scale == "medium" else 4)
    p1_inst = build_p1_routing_qubo(df_road, n_vehicles=n_veh_p1, lam_cong=6.0)
    p1_gs = None
    if p1_inst.n_vars <= 18:
        res_ex = solve_exact_qubo(p1_inst)
        p1_gs = res_ex.best_energy
        ev = evaluate_p1_solution(p1_inst, res_ex.best_z)
        record("P1", p1_inst.name, p1_inst.n_vars, res_ex.solver_name, res_ex.best_energy, p1_gs, ev["feasible"], res_ex.runtime_s, ev)

    for res in [
        solve_simulated_annealing(p1_inst, n_sweeps=800, n_restarts=4, seed=seed),
        solve_simulated_bifurcation(p1_inst, n_steps=1200, n_trials=24, seed=seed),
    ]:
        ev = evaluate_p1_solution(p1_inst, res.best_z)
        record("P1", p1_inst.name, p1_inst.n_vars, res.solver_name, res.best_energy, p1_gs, ev["feasible"], res.runtime_s, ev)

    if run_vca and p1_inst.n_vars <= 80:
        res_vca = solve_binary_vca(p1_inst, batch_size=64, hidden_dim=20, n_warmup=35, n_anneal=vca_anneal_steps, n_train=2, seed=seed)
        ev = evaluate_p1_solution(p1_inst, res_vca.best_z)
        record("P1", p1_inst.name, p1_inst.n_vars, res_vca.solver_name, res_vca.best_energy, p1_gs, ev["feasible"], res_vca.runtime_s, ev)

    lam1_crit = clean_air["emission_cost"] / 2.0
    full_report["problems"]["P1"] = {
        "name": p1_inst.name,
        "N_vars": p1_inst.n_vars,
        "analytical_single_veh_lambda1_crit": round(lam1_crit, 4),
        "exact_gs_energy": round(p1_gs, 4) if p1_gs is not None else None,
    }

    # Problem P2: VPP Appliance & EV Charging Scheduling (Eq. 6)
    p2_inst = build_p2_vpp_qubo(df_vpp, df_grid)
    p2_gs = None
    if p2_inst.metadata["n_slots"] ** p2_inst.metadata["n_app"] <= 100_000:
        res_ex = solve_exact_vpp_subspace(p2_inst)
        p2_gs = res_ex.best_energy
        ev = evaluate_p2_solution(p2_inst, res_ex.best_z)
        record("P2", p2_inst.name, p2_inst.n_vars, res_ex.solver_name, res_ex.best_energy, p2_gs, ev["feasible"], res_ex.runtime_s, ev)

    for res in [
        baseline_p2_uncoordinated(p2_inst),
        solve_simulated_annealing(p2_inst, n_sweeps=1000, n_restarts=4, seed=seed),
        solve_subspace_sa_vpp(p2_inst, n_sweeps=600, n_restarts=4, seed=seed),
        solve_simulated_bifurcation(p2_inst, n_steps=1200, n_trials=24, seed=seed),
    ]:
        ev = evaluate_p2_solution(p2_inst, res.best_z)
        record("P2", p2_inst.name, p2_inst.n_vars, res.solver_name, res.best_energy, p2_gs, ev["feasible"], res.runtime_s, ev)

    if run_vca and p2_inst.n_vars <= 80:
        res_vca = solve_binary_vca(p2_inst, batch_size=64, hidden_dim=20, n_warmup=35, n_anneal=vca_anneal_steps, n_train=2, seed=seed)
        ev = evaluate_p2_solution(p2_inst, res_vca.best_z)
        record("P2", p2_inst.name, p2_inst.n_vars, res_vca.solver_name, res_vca.best_energy, p2_gs, ev["feasible"], res_vca.runtime_s, ev)

    # Problem P3: SmartTransyt EV Corridor Allocation + Aayalolo QBS (Eq. 7)
    max_b = 2 if scale == "small" else (4 if scale == "medium" else 6)
    p3_inst = build_p3_ev_corridor_qubo(df_corr, df_fleet, max_buses=max_b, use_range_slack=True)
    p3_gs = None
    if p3_inst.n_vars <= 18:
        res_ex = solve_exact_qubo(p3_inst)
        p3_gs = res_ex.best_energy
        ev = evaluate_p3_solution(p3_inst, res_ex.best_z)
        record("P3", p3_inst.name, p3_inst.n_vars, res_ex.solver_name, res_ex.best_energy, p3_gs, ev["feasible"], res_ex.runtime_s, ev)

    for res in [
        solve_simulated_annealing(p3_inst, n_sweeps=900, n_restarts=4, seed=seed),
        solve_simulated_bifurcation(p3_inst, n_steps=1200, n_trials=24, seed=seed),
    ]:
        ev = evaluate_p3_solution(p3_inst, res.best_z)
        record("P3", p3_inst.name, p3_inst.n_vars, res.solver_name, res.best_energy, p3_gs, ev["feasible"], res.runtime_s, ev)

    if run_vca and p3_inst.n_vars <= 80:
        res_vca = solve_binary_vca(p3_inst, batch_size=64, hidden_dim=20, n_warmup=35, n_anneal=vca_anneal_steps, n_train=2, seed=seed)
        ev = evaluate_p3_solution(p3_inst, res_vca.best_z)
        record("P3", p3_inst.name, p3_inst.n_vars, res_vca.solver_name, res_vca.best_energy, p3_gs, ev["feasible"], res_vca.runtime_s, ev)

    # Problem P4: Budgeted BSF Hub Siting — Soft & Strict-Slack Variants (Eq. 8)
    for strict_slack in [False, True]:
        p4_inst = build_p4_hub_siting_qubo(df_sites, strict_budget_slack=strict_slack)
        p4_gs = None
        if p4_inst.n_vars <= 18:
            res_ex = solve_exact_qubo(p4_inst)
            p4_gs = res_ex.best_energy
            ev = evaluate_p4_solution(p4_inst, res_ex.best_z)
            record("P4", p4_inst.name, p4_inst.n_vars, res_ex.solver_name, res_ex.best_energy, p4_gs, ev["strict_feasible"], res_ex.runtime_s, ev)

        for res in [
            baseline_p4_nar_greedy(p4_inst),
            solve_simulated_annealing(p4_inst, n_sweeps=900, n_restarts=4, seed=seed),
            solve_simulated_bifurcation(p4_inst, n_steps=1200, n_trials=24, seed=seed),
        ]:
            ev = evaluate_p4_solution(p4_inst, res.best_z)
            record("P4", p4_inst.name, p4_inst.n_vars, res.solver_name, res.best_energy, p4_gs, ev["strict_feasible"], res.runtime_s, ev)

        if run_vca and (not strict_slack) and p4_inst.n_vars <= 60:
            res_vca = solve_binary_vca(p4_inst, batch_size=64, hidden_dim=20, n_warmup=35, n_anneal=vca_anneal_steps, n_train=2, seed=seed)
            ev = evaluate_p4_solution(p4_inst, res_vca.best_z)
            record("P4", p4_inst.name, p4_inst.n_vars, res_vca.solver_name, res_vca.best_energy, p4_gs, ev["strict_feasible"], res_vca.runtime_s, ev)

    df_out = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_dir, f"benchmark_summary_{scale}.csv")
    json_path = os.path.join(out_dir, f"benchmark_report_{scale}.json")
    df_out.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n=== CarbonIQ + EcoCycle Benchmark Summary (scale={scale}) ===")
    display_cols = ["problem", "variant", "N_vars", "solver", "energy", "exact_gs", "residual_per_var", "feasible", "runtime_s"]
    print(df_out[display_cols].to_string(index=False))
    print(f"\nSaved CSV to {csv_path} and JSON to {json_path}")
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Run unified multi-solver benchmarks across P1-P5.")
    parser.add_argument("--data-dir", type=str, default="data/starter_small")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-vca", action="store_true")
    parser.add_argument("--vca-anneal", type=int, default=120)
    args = parser.parse_args()

    run_benchmark(
        data_dir=args.data_dir,
        scale=args.scale,
        out_dir=args.out_dir,
        seed=args.seed,
        run_vca=not args.skip_vca,
        vca_anneal_steps=args.vca_anneal,
    )


if __name__ == "__main__":
    main()
