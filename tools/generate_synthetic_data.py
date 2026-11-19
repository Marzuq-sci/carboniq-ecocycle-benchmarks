#!/usr/bin/env python3
"""
CarbonIQ + EcoCycle Synthetic Urban Benchmark Generator.

Generates seeded, schema-validated CSV datasets grounded in Accra, Ghana's
urban transport (trotros, Aayalolo QBS trunk corridors, SmartTransyt EVs),
dual-stream circular waste (EcoCycle recyclables + Aboboyaa/BSF organics),
empirical GAMA PM2.5 air-quality distributions, and diurnal solar/grid curves.

Inspired by VectorInstitute/VariationalNeuralAnnealing's tools/ and data/
conventions for reproducible multi-seed benchmarking.

Usage:
    python tools/generate_synthetic_data.py --scale small --seed 42 --out data/starter_small
    python tools/generate_synthetic_data.py --scale medium --seed 42 --out data/starter_medium
    python tools/generate_synthetic_data.py --scale large --seed 42 --out data/starter_large
"""

import argparse
import os
import numpy as np
import pandas as pd


ACCRA_NEIGHBORHOODS = [
    "Madina", "Okponglo", "Legon", "Haatso", "Kwabenya", "Achimota",
    "Adentan", "Atomic", "East_Legon", "Spintex", "Osu", "Circle",
    "Kaneshie", "Nima", "Lapaz", "Ofankor", "Amasaman", "Kasoa",
    "Dansoman", "Tema_Station", "Tudu", "Agbogbloshie", "Ashaiman",
    "Airport_Res", "Dzorwulu", "Roman_Ridge", "Kanda", "Labadi",
    "Teshie", "Nungua", "Weija", "Pokuase", "Dome", "Taifa",
    "Abokobi", "Oyibi", "Dodowa", "Sakumono", "Lashibi", "Cantonments",
    "Oyarifa", "Frafraha", "Bortianor", "Mallam", "Odorkor", "Awoshie",
    "Darkuman", "Chorkor", "Jamestown", "Korle_Bu",
]

# Real Accra corridors anchored to Aayalolo QBS trunk routes & key feeders
ACCRA_CORRIDORS = [
    ("C01", "Amasaman_Accra_Central", True,  22, 0.89),
    ("C02", "Adentan_Accra_Central",  True,  18, 0.86),
    ("C03", "Kasoa_Accra_Central",    True,  24, 0.92),
    ("C04", "Ofankor_Accra_Central",  True,  16, 0.84),
    ("C05", "Madina_Circle_Feeder",   False, 12, 0.85),
    ("C06", "Spintex_Osu_Feeder",     False, 14, 0.82),
    ("C07", "Dansoman_Tudu_Feeder",   False, 11, 0.80),
    ("C08", "Lapaz_Kaneshie_Feeder",  False, 10, 0.88),
]

# Empirical GAMA PM2.5 medians (ug/m3) by road class (Alli et al., ERL 2021)
GAMA_PM25_MEDIANS = {
    "residential_bypass": 16.75,
    "commercial_mixed":   34.26,
    "trunk_roadside":     54.90,
}


def generate_road_network(scale: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate directed Accra street graph for P1 (Multi-Collector Routing)
    and P5 (Clean-Air Pedestrian Navigation).
    """
    if scale == "small":
        # 6-node, 8-edge canonical Accra subgraph (Madina -> Achimota BSF Hub)
        edges_spec = [
            (0, 1, "Madina",   "Okponglo", 2.4, 50.0, "trunk_roadside",     2.6),
            (1, 2, "Okponglo", "Legon",    2.2, 50.0, "trunk_roadside",     2.8),
            (2, 5, "Legon",    "Achimota", 3.6, 50.0, "trunk_roadside",     2.7),
            (0, 3, "Madina",   "Haatso",   2.8, 45.0, "residential_bypass", 1.15),
            (3, 4, "Haatso",   "Kwabenya", 3.1, 45.0, "residential_bypass", 1.10),
            (4, 5, "Kwabenya", "Achimota", 3.5, 45.0, "residential_bypass", 1.20),
            (1, 3, "Okponglo", "Haatso",   2.0, 40.0, "commercial_mixed",   1.75),
            (3, 2, "Haatso",   "Legon",    2.3, 40.0, "commercial_mixed",   1.85),
        ]
        rows = []
        for idx, (u, v, nu, nv, dist, vmax, rcls, base_iota) in enumerate(edges_spec):
            med_pm = GAMA_PM25_MEDIANS[rcls]
            pm25 = float(np.round(med_pm * rng.lognormal(0.0, 0.18), 2))
            nox = float(np.round(pm25 * rng.uniform(0.95, 1.35), 2))
            iota = float(np.round(max(1.0, base_iota * rng.uniform(0.90, 1.12)), 3))
            tau_free = dist / vmax
            pred_tau = float(np.round(tau_free * iota, 4))
            rows.append({
                "edge_id": f"E{idx:03d}",
                "from_node": u,
                "to_node": v,
                "from_name": nu,
                "to_name": nv,
                "road_class": rcls,
                "distance_km": dist,
                "speed_limit_kmh": vmax,
                "pred_travel_hr": pred_tau,
                "idling_index": iota,
                "pm25_ug_m3": pm25,
                "nox_ug_m3": nox,
            })
        return pd.DataFrame(rows)

    # Medium (15 nodes) and Large (30 nodes) layered Accra urban DAG
    # Direct arterials are geometrically shorter in km but suffer heavy diesel
    # soot & idling, while residential bypasses are slightly longer but clean.
    n_nodes = 15 if scale == "medium" else 30
    names = ACCRA_NEIGHBORHOODS[:n_nodes]
    rows = []
    edge_set = set()
    eid = 0

    def add_edge_dist(u, v, rcls, dist_lo, dist_hi):
        nonlocal eid
        if u == v or (u, v) in edge_set:
            return
        edge_set.add((u, v))
        dist = float(np.round(rng.uniform(dist_lo, dist_hi), 2))
        vmax = 50.0 if rcls == "trunk_roadside" else (40.0 if rcls == "commercial_mixed" else 35.0)
        base_iota = {"trunk_roadside": 2.65, "commercial_mixed": 1.75, "residential_bypass": 1.12}[rcls]
        iota = float(np.round(max(1.0, base_iota * rng.uniform(0.88, 1.18)), 3))
        pm25 = float(np.round(GAMA_PM25_MEDIANS[rcls] * rng.lognormal(0.0, 0.18), 2))
        nox = float(np.round(pm25 * rng.uniform(0.95, 1.35), 2))
        pred_tau = float(np.round((dist / vmax) * iota, 4))
        rows.append({
            "edge_id": f"E{eid:03d}",
            "from_node": u,
            "to_node": v,
            "from_name": names[u],
            "to_name": names[v],
            "road_class": rcls,
            "distance_km": dist,
            "speed_limit_kmh": vmax,
            "pred_travel_hr": pred_tau,
            "idling_index": iota,
            "pm25_ug_m3": pm25,
            "nox_ug_m3": nox,
        })
        eid += 1

    for u in range(n_nodes - 1):
        # Short, congested direct trunk jumps (fast on paper, polluted in reality)
        if u + 2 < n_nodes and u % 2 == 0:
            add_edge_dist(u, u + 2, "trunk_roadside", 2.0, 2.8)
        # Residential bypass edges (longer cumulative distance, clean air)
        add_edge_dist(u, u + 1, "residential_bypass", 1.8, 2.5)
        # Commercial mixed cross-connectors
        if u + 3 < n_nodes and rng.random() < 0.55:
            add_edge_dist(u, u + 3, "commercial_mixed", 3.6, 4.8)

    return pd.DataFrame(rows)


def generate_waste_requests(scale: str, n_nodes: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate EcoCycle inorganic recyclables (PET, metal, e-waste) and
    Aboboyaa organic food-waste gate pickup requests.
    """
    n_req = {"small": 4, "medium": 16, "large": 40}[scale]
    materials = [
        ("organic",   "food_waste", (0.0, 0.0),   (8, 35)),
        ("inorganic", "PET",        (2.0, 3.5),   (5, 30)),
        ("inorganic", "metal",      (4.0, 7.5),   (6, 40)),
        ("inorganic", "e_waste",    (8.0, 15.0),  (3, 18)),
    ]
    rows = []
    candidate_nodes = list(range(1, max(2, n_nodes - 1)))
    for i in range(n_req):
        stype, mat, (pmin, pmax), (wmin, wmax) = materials[int(rng.integers(0, len(materials)))]
        node_id = int(candidate_nodes[i % len(candidate_nodes)]) if i < len(candidate_nodes) else int(rng.choice(candidate_nodes))
        weight = int(rng.integers(wmin, wmax + 1))
        price = float(np.round(rng.uniform(pmin, pmax), 2)) if pmax > 0 else 0.0
        rows.append({
            "req_id": f"REQ{i:03d}",
            "node_id": node_id,
            "neighborhood": ACCRA_NEIGHBORHOODS[node_id % len(ACCRA_NEIGHBORHOODS)],
            "stream_type": stype,
            "material": mat,
            "est_weight_kg": weight,
            "est_price_ghs_kg": price,
            "window_start_hr": int(rng.choice([7, 8, 9, 10, 13, 14, 15])),
        })
    return pd.DataFrame(rows)


def generate_fleet(scale: str, n_nodes: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate collectors (Aboboyaa, scrap tricycles) and transit vehicles
    (conventional trotros, Aayalolo QBS buses, SmartTransyt EVs).
    """
    counts = {
        "small":  {"aboboyaa": 2, "trotro": 2, "aayalolo_qbs": 2, "smarttransyt_ev": 2},
        "medium": {"aboboyaa": 4, "trotro": 6, "aayalolo_qbs": 4, "smarttransyt_ev": 6},
        "large":  {"aboboyaa": 8, "trotro": 12, "aayalolo_qbs": 8, "smarttransyt_ev": 12},
    }[scale]
    specs = {
        "aboboyaa":        (220, 90,  0.210, 1),
        "trotro":          (15,  220, 0.850, 15),
        "aayalolo_qbs":    (65,  260, 1.150, 65),
        "smarttransyt_ev": (22,  160, 0.000, 22),
    }
    rows = []
    vid = 0
    for vtype, cnt in counts.items():
        cap, rng_km, ef, pax = specs[vtype]
        for _ in range(cnt):
            start_node = 0 if vtype == "aboboyaa" else int(rng.integers(0, max(1, n_nodes // 2)))
            rows.append({
                "vehicle_id": f"V{vid:03d}",
                "vehicle_type": vtype,
                "start_node": start_node,
                "dest_hub_node": n_nodes - 1,
                "max_payload_kg": cap if vtype == "aboboyaa" else 0,
                "passenger_cap": pax,
                "battery_or_fuel_range_km": int(rng_km * rng.uniform(0.90, 1.10)),
                "emission_kg_co2_km": ef,
            })
            vid += 1
    return pd.DataFrame(rows)


def generate_vpp_and_grid(scale: str, rng: np.random.Generator):
    """
    Generate Problem P2 datasets: shiftable appliance/EV clusters and
    Accra diurnal solar + grid carbon intensity curve.
    """
    if scale == "small":
        hours = [8, 11, 14, 17, 20, 23]
        carbon_base = [0.42, 0.25, 0.18, 0.55, 0.74, 0.62]
        solar_base = [22.0, 50.0, 65.0, 20.0, 5.0, 8.0]
        base_load = [180.0, 210.0, 230.0, 260.0, 320.0, 190.0]
        cluster_specs = [
            ("CL00", "smart_washer_cluster",  37, 20),
            ("CL01", "borehole_water_pumps",   5, 20),
            ("CL02", "cold_storage_defrost",  13, 20),
            ("CL03", "smarttransyt_ev_depot", 47, 20),
            ("CL04", "electric_water_heater", 20, 20),
        ]
        grid_rows = []
        for idx, (hr, kap, sol, bl) in enumerate(zip(hours, carbon_base, solar_base, base_load)):
            grid_rows.append({
                "slot_id": idx,
                "hour_gmt": hr,
                "carbon_intensity_kg_kwh": float(np.round(max(0.10, kap * rng.uniform(0.92, 1.08)), 3)),
                "solar_target_kw": float(np.round(max(2.0, sol * rng.uniform(0.88, 1.12)), 1)),
                "base_load_kw": float(np.round(bl * rng.uniform(0.95, 1.05), 1)),
            })
        app_rows = []
        for cid, cname, pkw, dhr in cluster_specs:
            app_rows.append({
                "cluster_id": cid,
                "cluster_name": cname,
                "rated_power_kw": int(max(4, round(pkw * rng.uniform(0.85, 1.15)))),
                "default_hour_gmt": dhr,
            })
        return pd.DataFrame(app_rows), pd.DataFrame(grid_rows)

    if scale == "medium":
        hours = [8, 11, 14, 17, 20, 23]
        carbon_base = [0.42, 0.25, 0.18, 0.55, 0.74, 0.62]
        solar_frac = [0.16, 0.30, 0.36, 0.10, 0.03, 0.05]
        n_clusters = 12
    else:
        hours = [6, 8, 10, 12, 14, 17, 20, 23]
        carbon_base = [0.58, 0.42, 0.28, 0.20, 0.18, 0.55, 0.74, 0.62]
        solar_frac = [0.04, 0.11, 0.20, 0.26, 0.25, 0.08, 0.03, 0.03]
        n_clusters = 25

    app_types = ["smart_washer", "ev_charger", "cold_storage", "water_pump", "water_heater", "bsf_dryer"]
    app_rows = []
    total_power = 0
    for i in range(n_clusters):
        pkw = int(rng.integers(6, 52))
        total_power += pkw
        app_rows.append({
            "cluster_id": f"CL{i:02d}",
            "cluster_name": f"{app_types[i % len(app_types)]}_{i:02d}",
            "rated_power_kw": pkw,
            "default_hour_gmt": int(rng.choice([17, 20, 20, 23])),
        })

    grid_rows = []
    for idx, (hr, kap, sf) in enumerate(zip(hours, carbon_base, solar_frac)):
        grid_rows.append({
            "slot_id": idx,
            "hour_gmt": hr,
            "carbon_intensity_kg_kwh": float(np.round(max(0.10, kap * rng.uniform(0.92, 1.08)), 3)),
            "solar_target_kw": float(np.round(total_power * sf * rng.uniform(0.90, 1.10), 1)),
            "base_load_kw": float(np.round((200 + 180 * kap) * rng.uniform(0.95, 1.05), 1)),
        })
    return pd.DataFrame(app_rows), pd.DataFrame(grid_rows)


def generate_corridor_demand(scale: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate Problem P3 dataset: total commuter demand D_ct^0 and scheduled
    Aayalolo QBS trunk supply Q_ct^0 yielding residual EV demand D_ct.
    """
    n_corr = {"small": 3, "medium": 5, "large": 8}[scale]
    hours = {"small": [7, 17], "medium": [7, 8, 17, 18], "large": [6, 7, 8, 12, 17, 18]}[scale]

    rows = []
    for cid, cname, has_qbs, loop_km, ef_diesel in ACCRA_CORRIDORS[:n_corr]:
        for hr in hours:
            is_peak = hr in (7, 8, 17, 18)
            d_tot = int(rng.integers(140, 260) if is_peak else rng.integers(60, 130))
            q_qbs = int(rng.integers(60, 130) if (has_qbs and is_peak) else (rng.integers(30, 65) if has_qbs else 0))
            d_res = max(0, d_tot - q_qbs)
            rows.append({
                "corridor_id": cid,
                "corridor_name": cname,
                "aayalolo_trunk": int(has_qbs),
                "hour_gmt": hr,
                "round_trip_km": loop_km,
                "diesel_trotro_kg_co2_km": ef_diesel,
                "total_demand_pax_hr": d_tot,
                "aayalolo_qbs_supply_pax_hr": q_qbs,
                "residual_demand_pax_hr": d_res,
            })
    return pd.DataFrame(rows)


def generate_candidate_hub_sites(scale: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate Problem P4 dataset: candidate BSF organic bioconversion hubs
    and charging depots across Accra neighborhoods (mirroring Q-Energic schema).
    """
    n_sites = {"small": 14, "medium": 50, "large": 100}[scale]
    rows = []
    for i in range(n_sites):
        base_name = ACCRA_NEIGHBORHOODS[i % len(ACCRA_NEIGHBORHOODS)]
        suffix = "" if i < len(ACCRA_NEIGHBORHOODS) else f"_Z{i // len(ACCRA_NEIGHBORHOODS) + 1}"
        tier = rng.uniform(0.65, 1.55)
        cost_usd = int(round(140_000 * tier * rng.uniform(0.85, 1.15) / 500) * 500)
        households = int(round(1_250 * tier * rng.uniform(0.75, 1.35) / 10) * 10)
        cap_tonnes = float(np.round(14.0 * tier * rng.uniform(0.80, 1.30), 1))
        rows.append({
            "site_id": f"S{i:03d}",
            "neighborhood": f"{base_name}{suffix}",
            "install_cost_usd": cost_usd,
            "households_covered": households,
            "daily_capacity_tonnes": cap_tonnes,
        })
    return pd.DataFrame(rows)


def generate_all(scale: str, seed: int, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    df_road = generate_road_network(scale, rng)
    n_nodes = int(max(df_road["from_node"].max(), df_road["to_node"].max()) + 1)
    df_req = generate_waste_requests(scale, n_nodes, rng)
    df_fleet = generate_fleet(scale, n_nodes, rng)
    df_vpp, df_grid = generate_vpp_and_grid(scale, rng)
    df_corr = generate_corridor_demand(scale, rng)
    df_sites = generate_candidate_hub_sites(scale, rng)

    files = {
        "accra_road_network.csv": df_road,
        "consumer_waste_requests.csv": df_req,
        "fleet_and_collectors.csv": df_fleet,
        "vpp_appliances.csv": df_vpp,
        "grid_hourly_profile.csv": df_grid,
        "corridor_demand_aayalolo.csv": df_corr,
        "candidate_hub_sites.csv": df_sites,
    }
    for fname, df in files.items():
        df.to_csv(os.path.join(out_dir, fname), index=False)
    return files


def main():
    parser = argparse.ArgumentParser(description="Generate CarbonIQ + EcoCycle synthetic benchmark CSVs.")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out or os.path.join("data", f"starter_{args.scale}")
    files = generate_all(args.scale, args.seed, out_dir)
    print(f"Generated {len(files)} CSV files in {out_dir} (scale={args.scale}, seed={args.seed}):")
    for fname, df in files.items():
        print(f"  - {fname:32s} ({len(df):3d} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
