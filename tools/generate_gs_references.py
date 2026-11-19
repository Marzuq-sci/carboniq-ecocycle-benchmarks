#!/usr/bin/env python3
"""
Generate exact ground-state reference files (GS_energies_*.txt) across 25
deterministic seeds for Problems P1 through P4.

Directly inspired by VectorInstitute/VariationalNeuralAnnealing's
data/EA/GS_energies_10x10.txt convention:
    # seed  gs_energy  gs_energy_per_var  N  feasible  gs_bitstring
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.generate_synthetic_data import (
    generate_road_network,
    generate_vpp_and_grid,
    generate_corridor_demand,
    generate_fleet,
    generate_candidate_hub_sites,
)
from src.qubo_builders import (
    build_p1_routing_qubo,
    evaluate_p1_solution,
    build_p2_vpp_qubo,
    evaluate_p2_solution,
    build_p3_ev_corridor_qubo,
    evaluate_p3_solution,
    build_p4_hub_siting_qubo,
    evaluate_p4_solution,
)
from src.solvers import solve_exact_qubo, solve_exact_vpp_subspace


def write_gs_file(path: str, title: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {title}\n")
        f.write("# Columns: seed  gs_energy  gs_energy_per_var  N  feasible  gs_bitstring\n")
        for r in rows:
            f.write(
                f"{r['seed']:4d}  {r['gs_energy']:14.6f}  {r['gs_per_var']:12.6f}  "
                f"{r['N']:3d}  {int(r['feasible'])}  {r['bits']}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Generate exact 25-seed GS_energies_*.txt files.")
    parser.add_argument("--n-seeds", type=int, default=25)
    parser.add_argument("--base-seed", type=int, default=1)
    args = parser.parse_args()

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    p1_1v_rows, p1_2v_rows = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        df_road = generate_road_network("small", rng)

        inst1 = build_p1_routing_qubo(df_road, n_vehicles=1, lam_flow=50.0)
        res1 = solve_exact_qubo(inst1)
        ev1 = evaluate_p1_solution(inst1, res1.best_z)
        p1_1v_rows.append({
            "seed": s,
            "gs_energy": res1.best_energy,
            "gs_per_var": res1.best_energy / inst1.n_vars,
            "N": inst1.n_vars,
            "feasible": ev1["feasible"],
            "bits": "".join(map(str, res1.best_z.tolist())),
        })

        inst2 = build_p1_routing_qubo(df_road, n_vehicles=2, lam_flow=50.0, lam_cong=6.0)
        res2 = solve_exact_qubo(inst2)
        ev2 = evaluate_p1_solution(inst2, res2.best_z)
        p1_2v_rows.append({
            "seed": s,
            "gs_energy": res2.best_energy,
            "gs_per_var": res2.best_energy / inst2.n_vars,
            "N": inst2.n_vars,
            "feasible": ev2["feasible"],
            "bits": "".join(map(str, res2.best_z.tolist())),
        })

    write_gs_file(
        "data/P1_routing/GS_energies_p1_small_1v.txt",
        "Problem P1 Single-Vehicle Accra Routing QUBO (6 nodes, 8 edges, N=8, exact 2^8)",
        p1_1v_rows,
    )
    write_gs_file(
        "data/P1_routing/GS_energies_p1_small_2v.txt",
        "Problem P1 Multi-Collector Accra Routing QUBO (6 nodes, 8 edges, 2 vehicles, N=16, exact 2^16)",
        p1_2v_rows,
    )

    p2_rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        df_vpp, df_grid = generate_vpp_and_grid("small", rng)
        inst = build_p2_vpp_qubo(df_vpp, df_grid)
        res = solve_exact_vpp_subspace(inst)
        ev = evaluate_p2_solution(inst, res.best_z)
        p2_rows.append({
            "seed": s,
            "gs_energy": res.best_energy,
            "gs_per_var": res.best_energy / inst.n_vars,
            "N": inst.n_vars,
            "feasible": ev["feasible"],
            "bits": "".join(map(str, res.best_z.tolist())),
        })
    write_gs_file(
        "data/P2_vpp/GS_energies_p2_5x6.txt",
        "Problem P2 Accra VPP Scheduling QUBO (5 clusters x 6 diurnal slots, N=30, exact 6^5 manifold)",
        p2_rows,
    )

    p3_rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        df_road = generate_road_network("small", rng)
        n_nodes = int(max(df_road["from_node"].max(), df_road["to_node"].max()) + 1)
        df_fleet = generate_fleet("small", n_nodes, rng)
        df_corr = generate_corridor_demand("small", rng)
        inst = build_p3_ev_corridor_qubo(df_corr, df_fleet, max_buses=2, use_range_slack=True)
        res = solve_exact_qubo(inst, max_n=18)
        ev = evaluate_p3_solution(inst, res.best_z)
        p3_rows.append({
            "seed": s,
            "gs_energy": res.best_energy,
            "gs_per_var": res.best_energy / inst.n_vars,
            "N": inst.n_vars,
            "feasible": ev["feasible"],
            "bits": "".join(map(str, res.best_z.tolist())),
        })
    write_gs_file(
        "data/P3_ev_corridor/GS_energies_p3_small.txt",
        "Problem P3 SmartTransyt EV Corridor Allocation with Aayalolo QBS Supply (exact 2^N)",
        p3_rows,
    )

    p4_rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        df_sites = generate_candidate_hub_sites("small", rng)
        inst = build_p4_hub_siting_qubo(df_sites, strict_budget_slack=False)
        res = solve_exact_qubo(inst, max_n=18)
        ev = evaluate_p4_solution(inst, res.best_z)
        p4_rows.append({
            "seed": s,
            "gs_energy": res.best_energy,
            "gs_per_var": res.best_energy / inst.n_vars,
            "N": inst.n_vars,
            "feasible": ev["near_feasible_2_5pct"],
            "bits": "".join(map(str, res.best_z.tolist())),
        })
    write_gs_file(
        "data/P4_hub_siting/GS_energies_p4_14sites.txt",
        "Problem P4 Budgeted BSF Hub Siting Soft-Penalty QUBO (L=14 Accra sites, N=14, exact 2^14)",
        p4_rows,
    )

    print(
        f"Generated 5 GS reference files ({len(seeds)} seeds each): "
        f"P1_1v feas={sum(r['feasible'] for r in p1_1v_rows)}/{len(seeds)}, "
        f"P1_2v feas={sum(r['feasible'] for r in p1_2v_rows)}/{len(seeds)}, "
        f"P2 feas={sum(r['feasible'] for r in p2_rows)}/{len(seeds)}, "
        f"P3 feas={sum(r['feasible'] for r in p3_rows)}/{len(seeds)}, "
        f"P4 near-feas={sum(r['feasible'] for r in p4_rows)}/{len(seeds)}"
    )


if __name__ == "__main__":
    main()
