#!/usr/bin/env python3
"""
Verification & Physics/Operations Research Unit Tests for CarbonIQ + EcoCycle.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.generate_synthetic_data import generate_all
from src.qubo_builders import (
    compute_mrv_credits,
    build_p1_routing_qubo,
    evaluate_p1_solution,
    build_p2_vpp_qubo,
    evaluate_p2_solution,
    build_p3_ev_corridor_qubo,
    evaluate_p3_solution,
    build_p4_hub_siting_qubo,
)
from src.solvers import (
    solve_dijkstra_p5,
    solve_exact_qubo,
    solve_exact_vpp_subspace,
    solve_subspace_sa_vpp,
)


def test_mrv_antifraud_gate():
    res = compute_mrv_credits(
        transit_trips=[{"e_veh_kg_km": 0.0, "occupancy": 20, "distance_km": 10.0}],
        waste_logs=[
            {"gamma_kg_co2_per_kg": 1.85, "mass_kg": 100.0, "gate_confirmed": True, "hub_weighbridge_verified": False},
            {"gamma_kg_co2_per_kg": 1.85, "mass_kg": 10.0, "gate_confirmed": True, "hub_weighbridge_verified": True},
        ],
        appliance_logs=[],
    )
    assert abs(res["waste_kg_co2e"] - 18.5) < 1e-6, f"Anti-fraud gate failed: {res}"


def test_p1_analytical_phase_transition():
    files = generate_all("small", 7, "/tmp/_ciq_test_data")
    df_road = files["accra_road_network.csv"]

    clean = solve_dijkstra_p5(df_road, use_emission_weights=True)
    gps = solve_dijkstra_p5(df_road, use_emission_weights=False)
    assert clean["emission_cost"] <= gps["emission_cost"] + 1e-9

    lam_crit = clean["emission_cost"] / 2.0
    inst_below = build_p1_routing_qubo(df_road, n_vehicles=1, lam_flow=lam_crit * 0.85)
    res_below = solve_exact_qubo(inst_below)
    assert int(res_below.best_z.sum()) == 0, "Expected zero vector below lambda_1_crit"

    inst_above = build_p1_routing_qubo(df_road, n_vehicles=1, lam_flow=lam_crit * 1.25)
    res_above = solve_exact_qubo(inst_above)
    ev = evaluate_p1_solution(inst_above, res_above.best_z)
    assert ev["feasible"], "Expected exact flow conservation above lambda_1_crit"
    assert abs(ev["emission_cost"] - clean["emission_cost"]) < 1e-4


def test_p2_qubo_vs_closed_form_and_subspace():
    files = generate_all("small", 11, "/tmp/_ciq_test_data")
    inst = build_p2_vpp_qubo(files["vpp_appliances.csv"], files["grid_hourly_profile.csv"])
    meta = inst.metadata
    rng = np.random.default_rng(11)

    for _ in range(20):
        z = rng.integers(0, 2, size=inst.n_vars)
        Y = z.reshape(meta["n_app"], meta["n_slots"])
        loads = meta["P"] @ Y
        closed_form = (
            float(np.sum(loads * meta["kappa"]))
            + meta["lam_solar"] * float(np.sum((loads - meta["P_target"]) ** 2))
            + meta["lam_onehot"] * float(np.sum((Y.sum(axis=1) - 1.0) ** 2))
        )
        assert abs(inst.energy(z) - closed_form) < 1e-6

    ex = solve_exact_vpp_subspace(inst)
    sa = solve_subspace_sa_vpp(inst, n_sweeps=500, n_restarts=4, seed=11)
    ev = evaluate_p2_solution(inst, ex.best_z)
    assert ev["feasible"]
    assert ev["evening_peak_shaving_pct"] > 0.0
    assert sa.best_energy <= ex.best_energy + 1e-4


def test_p3_and_p4_qubo_dimensions_and_exactness():
    files = generate_all("small", 13, "/tmp/_ciq_test_data")
    p3 = build_p3_ev_corridor_qubo(files["corridor_demand_aayalolo.csv"], files["fleet_and_collectors.csv"], max_buses=2)
    assert p3.n_vars <= 18
    res3 = solve_exact_qubo(p3)
    ev3 = evaluate_p3_solution(p3, res3.best_z)
    assert ev3["feasible"], f"P3 exact GS must be feasible: {ev3}"

    p4 = build_p4_hub_siting_qubo(files["candidate_hub_sites.csv"], strict_budget_slack=False)
    assert p4.n_vars == 14
    res4 = solve_exact_qubo(p4)
    assert np.isfinite(res4.best_energy)


if __name__ == "__main__":
    test_mrv_antifraud_gate()
    test_p1_analytical_phase_transition()
    test_p2_qubo_vs_closed_form_and_subspace()
    test_p3_and_p4_qubo_dimensions_and_exactness()
    print("ALL 4 VERIFICATION TEST SUITES PASSED!")
