"""
optimizer_core.py
Step 3: GPU Quantum-Inspired Particle Swarm Optimizer (QPSO) Engine
Coupled to PyTorch Physics-Informed Neural Network (PINN).
Supports Single-Voyage and Multi-Vessel Multi-Route Fleet Optimization.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import joblib

# Fuel Specifications
FUEL_TYPES = ["HFO", "LNG", "Methanol", "Ammonia", "Hydrogen"]
LHV_MULTIPLIERS = [1.0, 0.82, 2.03, 2.17, 0.33]            # Fuel mass multiplier relative to HFO
WTW_CO2_INTENSITIES = [3.151, 2.750, 0.540, 0.150, 0.050]   # Tons CO2 per ton fuel
FUEL_COSTS_USD = [600.0, 850.0, 950.0, 1100.0, 2200.0]       # USD per metric ton

class PhysicsInformedFuelPredictor(nn.Module):
    """3 linear layers with ReLU activation matching train_pinn.py."""
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class GPU_QPSO_Optimizer:
    def __init__(
        self,
        model_path="maritime_pinn_predictor.pth",
        feature_scaler_path="feature_scaler.joblib",
        target_scaler_path="target_scaler.joblib",
        device=None
    ):
        if device is None:
            if torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1, device="cuda")
                    _ = test_tensor.item()
                    self.device = torch.device("cuda")
                except Exception:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load Scalers
        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)

        # Scaler parameters directly as GPU tensors for speed
        self.feat_mean = torch.tensor(self.feature_scaler.mean_, dtype=torch.float32, device=self.device)
        self.feat_scale = torch.tensor(self.feature_scaler.scale_, dtype=torch.float32, device=self.device)
        self.tgt_mean = float(self.target_scaler.mean_[0])
        self.tgt_scale = float(self.target_scaler.scale_[0])

        # Load PINN Model
        self.model = PhysicsInformedFuelPredictor().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # Multi-fuel CUDA tensors
        self.lhv_tensor = torch.tensor(LHV_MULTIPLIERS, dtype=torch.float32, device=self.device)
        self.wtw_tensor = torch.tensor(WTW_CO2_INTENSITIES, dtype=torch.float32, device=self.device)
        self.cost_tensor = torch.tensor(FUEL_COSTS_USD, dtype=torch.float32, device=self.device)

        # Bounds: [Speed (10-22), Fuel Type (0-4), Cargo Load (0.2-1.0)]
        self.lb = torch.tensor([10.0, 0.0, 0.20], dtype=torch.float32, device=self.device)
        self.ub = torch.tensor([22.0, 4.0, 1.00], dtype=torch.float32, device=self.device)

    def evaluate_fitness_batch(
        self,
        particles: torch.Tensor,
        distance_nm: float,
        max_eta_hours: float,
        wave_height_m: float,
        wind_knots: float,
        carbon_tax_per_ton: float = 100.0,
        fixed_fuel_idx: int = None,
        fixed_cargo_load: float = None,
    ):
        """
        Evaluate single-voyage objective function for particle swarm batch in pure GPU tensor operations.
        particles: (N, 3) tensor -> [speed, fuel_type_idx, cargo_load]
        """
        N = particles.shape[0]
        speed = particles[:, 0]
        
        if fixed_fuel_idx is not None:
            fuel_idx = torch.full((N,), fixed_fuel_idx, dtype=torch.long, device=self.device)
        else:
            fuel_idx = torch.clamp(torch.round(particles[:, 1]).long(), 0, 4)

        if fixed_cargo_load is not None:
            cargo_load = torch.full((N,), fixed_cargo_load, dtype=torch.float32, device=self.device)
        else:
            cargo_load = particles[:, 2]

        travel_time = distance_nm / speed

        # Prepare PINN input batch: [Speed, Cargo_Load, wave_height_m, wind_knots]
        wave_tensor = torch.full((N, 1), wave_height_m, dtype=torch.float32, device=self.device)
        wind_tensor = torch.full((N, 1), wind_knots, dtype=torch.float32, device=self.device)
        raw_inputs = torch.cat([
            speed.unsqueeze(1),
            cargo_load.unsqueeze(1),
            wave_tensor,
            wind_tensor
        ], dim=1)

        # Scale inputs on GPU
        inputs_scaled = (raw_inputs - self.feat_mean) / self.feat_scale

        with torch.no_grad():
            preds_scaled = self.model(inputs_scaled).squeeze(-1)

        # Unscale predictions to tons HFO / hour
        hfo_burn_rate = preds_scaled * self.tgt_scale + self.tgt_mean
        hfo_burn_rate = torch.clamp(hfo_burn_rate, min=0.01)

        # Fuel multipliers
        lhv = self.lhv_tensor[fuel_idx]
        wtw = self.wtw_tensor[fuel_idx]
        fuel_price = self.cost_tensor[fuel_idx]

        # Consumptions & Emissions
        fuel_consumed_tons = travel_time * hfo_burn_rate * lhv
        fuel_cost_usd = fuel_consumed_tons * fuel_price
        co2_emissions_tons = fuel_consumed_tons * wtw
        carbon_tax_cost = co2_emissions_tons * carbon_tax_per_ton

        # Hard penalty for ETA delay: torch.relu(travel_time - max_eta) * 15000.0
        eta_delay_hours = torch.relu(travel_time - max_eta_hours)
        eta_penalty = eta_delay_hours * 15000.0

        # Total Cost Objective ($)
        total_cost = fuel_cost_usd + carbon_tax_cost + eta_penalty

        return total_cost, {
            "speed": speed,
            "fuel_idx": fuel_idx,
            "cargo_load": cargo_load,
            "fuel_consumed_tons": fuel_consumed_tons,
            "co2_emissions_tons": co2_emissions_tons,
            "travel_time_hours": travel_time,
            "fuel_cost_usd": fuel_cost_usd,
            "carbon_tax_cost_usd": carbon_tax_cost,
            "eta_penalty": eta_penalty,
            "total_cost_usd": total_cost
        }

    def optimize(
        self,
        distance_nm: float = 1500.0,
        max_eta_hours: float = 100.0,
        wave_height_m: float = 2.5,
        wind_knots: float = 20.0,
        carbon_tax_per_ton: float = 100.0,
        swarm_size: int = 128,
        iterations: int = 60,
        seed: int = 42
    ):
        """Quantum Delta-Potential Swarm Optimization for single mission."""
        torch.manual_seed(seed)
        start_time = time.perf_counter()
        D = 3
        device = self.device
        lb = self.lb
        ub = self.ub

        # Initialize particles uniformly in bounds
        rands = torch.rand(swarm_size, D, device=device)
        particles = lb + rands * (ub - lb)
        p_best = particles.clone()

        scores, _ = self.evaluate_fitness_batch(
            particles, distance_nm, max_eta_hours, wave_height_m, wind_knots, carbon_tax_per_ton
        )
        p_best_scores = scores.clone()

        g_best_idx = torch.argmin(p_best_scores)
        g_best_score = p_best_scores[g_best_idx].item()
        g_best = p_best[g_best_idx].clone()

        convergence_history = [g_best_score]

        for it in range(1, iterations):
            # 1. Contraction-Expansion coefficient alpha decreases linearly
            alpha = 1.0 - 0.5 * (it / iterations)

            # 2. Compute Mean Best Position (m_best) across all personal bests
            m_best = torch.mean(p_best, dim=0)

            # 3. Compute Local Attractor p_i = phi * p_best + (1 - phi) * g_best
            phi = torch.rand(swarm_size, D, device=device)
            p_attractor = phi * p_best + (1.0 - phi) * g_best.unsqueeze(0)

            # 4. Quantum Delta Potential Wavefunction Position Update
            u = torch.rand(swarm_size, D, device=device)
            u = torch.clamp(u, min=1e-7, max=1.0 - 1e-7)
            ln_inv_u = torch.log(1.0 / u)

            sign = torch.where(torch.rand(swarm_size, D, device=device) > 0.5, 1.0, -1.0)
            step = alpha * torch.abs(m_best.unsqueeze(0) - particles) * ln_inv_u
            particles = p_attractor + sign * step

            # Clamp to bounds
            particles = torch.clamp(particles, min=lb, max=ub)

            # Evaluate Batch Fitness on GPU
            scores, _ = self.evaluate_fitness_batch(
                particles, distance_nm, max_eta_hours, wave_height_m, wind_knots, carbon_tax_per_ton
            )

            # Update Personal Best
            improved_mask = scores < p_best_scores
            p_best[improved_mask] = particles[improved_mask]
            p_best_scores[improved_mask] = scores[improved_mask]

            # Update Global Best
            min_score, min_idx = torch.min(p_best_scores, dim=0)
            if min_score.item() < g_best_score:
                g_best_score = min_score.item()
                g_best = p_best[min_idx].clone()

            convergence_history.append(g_best_score)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Evaluate final optimal metrics
        _, best_metrics = self.evaluate_fitness_batch(
            g_best.unsqueeze(0), distance_nm, max_eta_hours, wave_height_m, wind_knots, carbon_tax_per_ton
        )

        opt_speed = float(best_metrics["speed"][0].item())
        opt_fuel_idx = int(best_metrics["fuel_idx"][0].item())
        opt_cargo = float(best_metrics["cargo_load"][0].item())
        opt_co2 = float(best_metrics["co2_emissions_tons"][0].item())
        opt_fuel_mass = float(best_metrics["fuel_consumed_tons"][0].item())
        opt_time = float(best_metrics["travel_time_hours"][0].item())
        opt_cost = float(best_metrics["total_cost_usd"][0].item())
        opt_penalty = float(best_metrics["eta_penalty"][0].item())

        # Baseline HFO calculation
        _, hfo_base_metrics = self.evaluate_fitness_batch(
            g_best.unsqueeze(0), distance_nm, max_eta_hours, wave_height_m, wind_knots, carbon_tax_per_ton,
            fixed_fuel_idx=0, fixed_cargo_load=opt_cargo
        )
        base_hfo_co2 = float(hfo_base_metrics["co2_emissions_tons"][0].item())
        co2_reduction_pct = max(0.0, (base_hfo_co2 - opt_co2) / max(0.01, base_hfo_co2) * 100.0)

        return {
            "optimal_speed_knots": opt_speed,
            "optimal_fuel_idx": opt_fuel_idx,
            "optimal_fuel_name": FUEL_TYPES[opt_fuel_idx],
            "optimal_cargo_load": opt_cargo,
            "fuel_consumed_tons": opt_fuel_mass,
            "co2_emissions_tons": opt_co2,
            "baseline_hfo_co2_tons": base_hfo_co2,
            "co2_reduction_pct": co2_reduction_pct,
            "travel_time_hours": opt_time,
            "total_cost_usd": opt_cost,
            "eta_penalty_usd": opt_penalty,
            "elapsed_ms": elapsed_ms,
            "device": str(self.device),
            "convergence_history": convergence_history,
        }

    def evaluate_multi_fleet_batch(
        self,
        particles: torch.Tensor,
        fleet_items: list,
        carbon_tax_per_ton: float = 100.0,
        fleet_carbon_budget_tons: float = None
    ):
        """
        Evaluate objective function for a whole multi-vessel fleet in high-dimensional tensor space.
        particles: (N, M, 3) where N = swarm_size, M = number of ships
        """
        N, M, _ = particles.shape
        total_fleet_cost = torch.zeros(N, device=self.device)
        total_fleet_co2 = torch.zeros(N, device=self.device)
        total_fleet_fuel_cost = torch.zeros(N, device=self.device)
        total_fleet_delay_penalty = torch.zeros(N, device=self.device)

        ship_results = []

        for m_idx, ship in enumerate(fleet_items):
            ship_part = particles[:, m_idx, :] # (N, 3)
            dist = float(ship["distance_nm"])
            eta = float(ship["eta_window_hours"] if "eta_window_hours" in ship else (dist / 14.0 + 20.0))
            wave = float(ship.get("wave_height_m", 2.5))
            wind = float(ship.get("wind_knots", 20.0))

            c, m = self.evaluate_fitness_batch(
                ship_part, dist, eta, wave, wind, carbon_tax_per_ton
            )

            total_fleet_cost += c
            total_fleet_co2 += m["co2_emissions_tons"]
            total_fleet_fuel_cost += m["fuel_cost_usd"]
            total_fleet_delay_penalty += m["eta_penalty"]

        # Fleet-wide carbon budget penalty if applicable
        if fleet_carbon_budget_tons is not None:
            over_budget = torch.relu(total_fleet_co2 - fleet_carbon_budget_tons)
            budget_penalty = over_budget * 500.0
            total_fleet_cost += budget_penalty

        return total_fleet_cost, {
            "total_fleet_cost": total_fleet_cost,
            "total_fleet_co2": total_fleet_co2,
            "total_fleet_fuel_cost": total_fleet_fuel_cost,
            "total_fleet_delay_penalty": total_fleet_delay_penalty
        }

    def optimize_multi_fleet(
        self,
        fleet_items: list,
        swarm_size: int = 128,
        iterations: int = 80,
        carbon_tax_per_ton: float = 100.0,
        seed: int = 42
    ):
        """
        Runs GPU QPSO on a high-dimensional multi-vessel fleet (D = M * 3).
        """
        torch.manual_seed(seed)
        start_time = time.perf_counter()
        M = len(fleet_items)
        device = self.device
        lb = self.lb.view(1, 1, 3).repeat(swarm_size, M, 1)
        ub = self.ub.view(1, 1, 3).repeat(swarm_size, M, 1)

        # Initialize particles (N, M, 3)
        rands = torch.rand(swarm_size, M, 3, device=device)
        particles = lb + rands * (ub - lb)
        p_best = particles.clone()

        scores, _ = self.evaluate_multi_fleet_batch(particles, fleet_items, carbon_tax_per_ton)
        p_best_scores = scores.clone()

        g_best_idx = torch.argmin(p_best_scores)
        g_best_score = p_best_scores[g_best_idx].item()
        g_best = p_best[g_best_idx].clone() # (M, 3)

        convergence_history = [g_best_score]

        for it in range(1, iterations):
            alpha = 1.0 - 0.5 * (it / iterations)

            # Mean Best Position across swarm (M, 3)
            m_best = torch.mean(p_best, dim=0)

            # Local Attractor (N, M, 3)
            phi = torch.rand(swarm_size, M, 3, device=device)
            p_attractor = phi * p_best + (1.0 - phi) * g_best.unsqueeze(0)

            # Wavefunction collapse
            u = torch.rand(swarm_size, M, 3, device=device)
            u = torch.clamp(u, min=1e-7, max=1.0 - 1e-7)
            ln_inv_u = torch.log(1.0 / u)

            sign = torch.where(torch.rand(swarm_size, M, 3, device=device) > 0.5, 1.0, -1.0)
            step = alpha * torch.abs(m_best.unsqueeze(0) - particles) * ln_inv_u
            particles = p_attractor + sign * step
            particles = torch.clamp(particles, min=self.lb.view(1, 1, 3), max=self.ub.view(1, 1, 3))

            scores, _ = self.evaluate_multi_fleet_batch(particles, fleet_items, carbon_tax_per_ton)

            improved_mask = scores < p_best_scores
            p_best[improved_mask] = particles[improved_mask]
            p_best_scores[improved_mask] = scores[improved_mask]

            min_score, min_idx = torch.min(p_best_scores, dim=0)
            if min_score.item() < g_best_score:
                g_best_score = min_score.item()
                g_best = p_best[min_idx].clone()

            convergence_history.append(g_best_score)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Detailed vessel breakdown for g_best
        vessel_breakdowns = []
        for m_idx, ship in enumerate(fleet_items):
            v_vec = g_best[m_idx].unsqueeze(0) # (1, 3)
            dist = float(ship["distance_nm"])
            eta = float(ship["eta_window_hours"] if "eta_window_hours" in ship else (dist / 14.0 + 20.0))
            wave = float(ship.get("wave_height_m", 2.5))
            wind = float(ship.get("wind_knots", 20.0))

            c, m = self.evaluate_fitness_batch(v_vec, dist, eta, wave, wind, carbon_tax_per_ton)

            spd = float(m["speed"][0].item())
            f_idx = int(m["fuel_idx"][0].item())
            c_load = float(m["cargo_load"][0].item())
            fuel_mass = float(m["fuel_consumed_tons"][0].item())
            co2_mass = float(m["co2_emissions_tons"][0].item())
            t_hrs = float(m["travel_time_hours"][0].item())
            delay_pen = float(m["eta_penalty"][0].item())
            cost = float(c[0].item())

            vessel_breakdowns.append({
                "ship_id": ship.get("ship_id", f"SHIP-{m_idx+1:02d}"),
                "name": ship.get("name", f"Vessel_{m_idx+1}"),
                "vessel_class": ship.get("vessel_class", "Cargo Vessel"),
                "origin": ship.get("origin", "Port A"),
                "origin_country": ship.get("origin_country", ""),
                "dest": ship.get("dest", "Port B"),
                "dest_country": ship.get("dest_country", ""),
                "distance_nm": dist,
                "optimal_speed_knots": spd,
                "optimal_fuel_name": FUEL_TYPES[f_idx],
                "optimal_cargo_load": c_load,
                "fuel_consumed_tons": fuel_mass,
                "co2_emissions_tons": co2_mass,
                "travel_time_hours": t_hrs,
                "eta_penalty_usd": delay_pen,
                "voyage_cost_usd": cost,
            })

        total_co2 = sum(v["co2_emissions_tons"] for v in vessel_breakdowns)
        total_fuel = sum(v["fuel_consumed_tons"] for v in vessel_breakdowns)

        return {
            "fleet_size": M,
            "total_fleet_cost_usd": g_best_score,
            "total_fleet_co2_tons": total_co2,
            "total_fleet_fuel_tons": total_fuel,
            "elapsed_ms": elapsed_ms,
            "convergence_history": convergence_history,
            "vessels": vessel_breakdowns
        }
