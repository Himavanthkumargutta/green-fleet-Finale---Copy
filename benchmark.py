"""
benchmark.py
Step 4: Algorithmic Benchmarking Suite
Compares GPU QPSO against Classical Velocity-Based PSO and Monte Carlo Random Search.
Evaluates Multi-Vessel Multi-Route Fleet Optimization across high-dimensional state spaces.
Outputs benchmark_results.json
"""

import json
import time
import numpy as np
import torch
from optimizer_core import (
    GPU_QPSO_Optimizer,
    FUEL_TYPES,
    LHV_MULTIPLIERS,
    WTW_CO2_INTENSITIES,
    FUEL_COSTS_USD
)
import routing_core

def run_fleet_random_search(
    optimizer: GPU_QPSO_Optimizer,
    fleet_items: list,
    carbon_tax_per_ton: float = 100.0,
    swarm_size: int = 128,
    iterations: int = 80,
    seed: int = 42
):
    """Monte Carlo Random Search across multi-vessel fleet."""
    torch.manual_seed(seed)
    start_time = time.perf_counter()
    M = len(fleet_items)
    device = optimizer.device
    lb = optimizer.lb.view(1, 1, 3).repeat(swarm_size, M, 1)
    ub = optimizer.ub.view(1, 1, 3).repeat(swarm_size, M, 1)

    g_best_score = float("inf")
    convergence_history = []

    for it in range(iterations):
        rands = torch.rand(swarm_size, M, 3, device=device)
        particles = lb + rands * (ub - lb)

        scores, m_dict = optimizer.evaluate_multi_fleet_batch(particles, fleet_items, carbon_tax_per_ton)

        min_score, min_idx = torch.min(scores, dim=0)
        if min_score.item() < g_best_score:
            g_best_score = min_score.item()

        convergence_history.append(g_best_score)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "method": "Random Search",
        "best_fitness_usd": g_best_score,
        "elapsed_ms": elapsed_ms,
        "convergence_history": convergence_history,
    }

def run_fleet_classical_pso(
    optimizer: GPU_QPSO_Optimizer,
    fleet_items: list,
    carbon_tax_per_ton: float = 100.0,
    swarm_size: int = 128,
    iterations: int = 80,
    w: float = 0.729,
    c1: float = 1.494,
    c2: float = 1.494,
    seed: int = 42
):
    """Classical Velocity-Based Particle Swarm Optimization for multi-vessel fleet."""
    torch.manual_seed(seed)
    start_time = time.perf_counter()
    M = len(fleet_items)
    device = optimizer.device
    lb = optimizer.lb.view(1, 1, 3).repeat(swarm_size, M, 1)
    ub = optimizer.ub.view(1, 1, 3).repeat(swarm_size, M, 1)

    # Initialize positions and velocities
    rands = torch.rand(swarm_size, M, 3, device=device)
    particles = lb + rands * (ub - lb)
    velocities = (ub - lb) * (torch.rand(swarm_size, M, 3, device=device) * 0.2 - 0.1)

    p_best = particles.clone()
    scores, _ = optimizer.evaluate_multi_fleet_batch(particles, fleet_items, carbon_tax_per_ton)
    p_best_scores = scores.clone()

    g_best_idx = torch.argmin(p_best_scores)
    g_best_score = p_best_scores[g_best_idx].item()
    g_best = p_best[g_best_idx].clone()

    convergence_history = [g_best_score]

    for it in range(1, iterations):
        # Velocity Update: v(t+1) = w*v(t) + c1*r1*(p_best - x) + c2*r2*(g_best - x)
        r1 = torch.rand(swarm_size, M, 3, device=device)
        r2 = torch.rand(swarm_size, M, 3, device=device)

        cognitive = c1 * r1 * (p_best - particles)
        social = c2 * r2 * (g_best.unsqueeze(0) - particles)
        velocities = w * velocities + cognitive + social

        # Velocity clamping to prevent explosion
        v_max = (ub - lb) * 0.20
        velocities = torch.clamp(velocities, min=-v_max, max=v_max)

        # Position update
        particles = particles + velocities
        particles = torch.clamp(particles, min=optimizer.lb.view(1, 1, 3), max=optimizer.ub.view(1, 1, 3))

        scores, _ = optimizer.evaluate_multi_fleet_batch(particles, fleet_items, carbon_tax_per_ton)

        improved_mask = scores < p_best_scores
        p_best[improved_mask] = particles[improved_mask]
        p_best_scores[improved_mask] = scores[improved_mask]

        min_score, min_idx = torch.min(p_best_scores, dim=0)
        if min_score.item() < g_best_score:
            g_best_score = min_score.item()
            g_best = p_best[min_idx].clone()

        convergence_history.append(g_best_score)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "method": "Classical Velocity PSO",
        "best_fitness_usd": g_best_score,
        "elapsed_ms": elapsed_ms,
        "convergence_history": convergence_history,
    }

def run_fleet_qpso(
    optimizer: GPU_QPSO_Optimizer,
    fleet_items: list,
    carbon_tax_per_ton: float = 100.0,
    swarm_size: int = 128,
    iterations: int = 80,
    seed: int = 42
):
    """GPU Quantum-Inspired Particle Swarm Optimization for multi-vessel fleet."""
    res = optimizer.optimize_multi_fleet(
        fleet_items=fleet_items,
        swarm_size=swarm_size,
        iterations=iterations,
        carbon_tax_per_ton=carbon_tax_per_ton,
        seed=seed
    )
    return {
        "method": "GPU Quantum-Inspired PSO (Ours)",
        "best_fitness_usd": res["total_fleet_cost_usd"],
        "total_fleet_co2_tons": res["total_fleet_co2_tons"],
        "total_fleet_fuel_tons": res["total_fleet_fuel_tons"],
        "elapsed_ms": res["elapsed_ms"],
        "convergence_history": res["convergence_history"],
        "vessels": res["vessels"]
    }

def run_benchmarks(
    fleet_size: int = 12,
    carbon_tax_per_ton: float = 100.0,
    swarm_size: int = 128,
    iterations: int = 80,
    output_json: str = "benchmark_results.json"
):
    print("=" * 70)
    print(f"STEP 4: Multi-Vessel Fleet Benchmark Suite (Fleet Size = {fleet_size})")
    print("=" * 70)

    optimizer = GPU_QPSO_Optimizer()
    fleet = routing_core.generate_random_fleet(fleet_size)

    print(f"[*] Dispatching {len(fleet)} vessels across global shipping lanes (Americas, Australia, Asia, Europe)...")
    print(f"[*] Parameter Dimension Space: D = {fleet_size * 3} variables on GPU Tensor Batch")

    print("[1/3] Running Monte Carlo Random Search Baseline...")
    res_random = run_fleet_random_search(
        optimizer, fleet, carbon_tax_per_ton=carbon_tax_per_ton, swarm_size=swarm_size, iterations=iterations
    )
    print(f"  -> Best Fleet Cost: ${res_random['best_fitness_usd']:,.2f} | Time: {res_random['elapsed_ms']:.2f} ms")

    print("[2/3] Running Classical Velocity-Based PSO Baseline...")
    res_cpso = run_fleet_classical_pso(
        optimizer, fleet, carbon_tax_per_ton=carbon_tax_per_ton, swarm_size=swarm_size, iterations=iterations
    )
    print(f"  -> Best Fleet Cost: ${res_cpso['best_fitness_usd']:,.2f} | Time: {res_cpso['elapsed_ms']:.2f} ms")

    print("[3/3] Running GPU Quantum-Inspired PSO Engine...")
    res_qpso = run_fleet_qpso(
        optimizer, fleet, carbon_tax_per_ton=carbon_tax_per_ton, swarm_size=swarm_size, iterations=iterations
    )
    print(f"  -> Best Fleet Cost: ${res_qpso['best_fitness_usd']:,.2f} | Time: {res_qpso['elapsed_ms']:.2f} ms")

    results_payload = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fleet_size": fleet_size,
        "swarm_size": swarm_size,
        "iterations": iterations,
        "methods": {
            "random_search": res_random,
            "classical_pso": res_cpso,
            "gpu_qpso": res_qpso
        }
    }

    with open(output_json, "w") as f:
        json.dump(results_payload, f, indent=4)
    print(f"[+] Saved full benchmark results to '{output_json}'.")

    print("\n" + "=" * 80)
    print(f"{'Method':<32} | {'Fleet Cost ($)':<18} | {'Compute Time (ms)':<18} | {'Cost Advantage'}")
    print("-" * 80)
    c_rand = res_random['best_fitness_usd']
    c_cpso = res_cpso['best_fitness_usd']
    c_qpso = res_qpso['best_fitness_usd']
    adv_cpso = ((c_cpso - c_qpso) / c_cpso) * 100.0 if c_cpso > 0 else 0.0

    print(f"{res_random['method']:<32} | ${c_rand:>16,.2f} | {res_random['elapsed_ms']:>16.2f} ms | Baseline")
    print(f"{res_cpso['method']:<32} | ${c_cpso:>16,.2f} | {res_cpso['elapsed_ms']:>16.2f} ms | +{((c_rand - c_cpso)/c_rand)*100:.1f}% vs Random")
    print(f"{res_qpso['method']:<32} | ${c_qpso:>16,.2f} | {res_qpso['elapsed_ms']:>16.2f} ms | QPSO Savings: -${c_cpso - c_qpso:,.2f} ({adv_cpso:+.2f}%)")
    print("=" * 80)
    print("STEP 4 COMPLETE: Multi-Vessel Fleet Benchmark Finished Successfully.")
    print("=" * 80)

    return results_payload

if __name__ == "__main__":
    run_benchmarks(fleet_size=12)
