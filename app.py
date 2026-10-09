"""
app.py
AURA-FLEET // Green Maritime Fleet Optimization & Decarbonization Platform
Enterprise Decision Support System powered by Physics-Informed Neural Networks and Quantum-Inspired Heuristics.
Supports Multi-Ship Custom Routes and Worldwide Fleet Dispatch (Americas, Australasia, Asia, Europe, Middle East).
"""

import os
import json
import time
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch

from optimizer_core import (
    GPU_QPSO_Optimizer,
    FUEL_TYPES,
    LHV_MULTIPLIERS,
    WTW_CO2_INTENSITIES,
    FUEL_COSTS_USD
)
import routing_core

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AURA-FLEET // Green Maritime Fleet Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 2. THEME CONTROLS & DYNAMIC CSS (DARK THEME DEFAULT)
# ------------------------------------------------------------------------------
with st.sidebar:
    dark_mode = st.toggle("Dark Mode", value=True, help="Toggle between Executive Dark Theme and Light Theme")

if dark_mode:
    theme_bg = "#070e17"
    theme_panel_bg = "#0c1726"
    theme_text = "#ffffff"
    theme_text_secondary = "#f1f5f9"
    theme_card_bg = "rgba(12, 23, 38, 0.95)"
    theme_card_border = "rgba(0, 240, 181, 0.35)"
    theme_card_hover_border = "rgba(0, 240, 181, 0.80)"
    theme_sidebar_bg = "#09121d"
    theme_metric_val = "#ffffff"
    plotly_template = "plotly_dark"
    plot_bg = "rgba(12, 23, 38, 0.85)"
    plot_paper_bg = "rgba(0, 0, 0, 0)"
    geo_land = "#13233a"
    geo_ocean = "#060d17"
    geo_country = "rgba(255, 255, 255, 0.20)"
    geo_coast = "rgba(0, 240, 181, 0.45)"
    grid_color = "rgba(255, 255, 255, 0.12)"
else:
    theme_bg = "#f4f6f9"
    theme_panel_bg = "#ffffff"
    theme_text = "#0f172a"
    theme_text_secondary = "#1e293b"
    theme_card_bg = "#ffffff"
    theme_card_border = "rgba(0, 160, 120, 0.35)"
    theme_card_hover_border = "rgba(0, 160, 120, 0.80)"
    theme_sidebar_bg = "#ffffff"
    theme_metric_val = "#0f172a"
    plotly_template = "plotly_white"
    plot_bg = "rgba(255, 255, 255, 0.95)"
    plot_paper_bg = "rgba(0, 0, 0, 0)"
    geo_land = "#d8e2ec"
    geo_ocean = "#f0f5fa"
    geo_country = "rgba(15, 23, 42, 0.25)"
    geo_coast = "rgba(0, 160, 120, 0.60)"
    grid_color = "rgba(0, 0, 0, 0.08)"

ENTERPRISE_THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: {theme_text};
    font-size: 16px;
    line-height: 1.6;
}}

.stApp {{
    background-color: {theme_bg};
}}

[data-testid="stSidebar"] {{
    background-color: {theme_sidebar_bg};
    border-right: 1px solid rgba(255, 255, 255, 0.10);
}}

p, span, label, div, [data-testid="stMarkdownContainer"] p {{
    color: {theme_text} !important;
    font-size: 1.02rem;
}}

.stCaption, [data-testid="stCaptionContainer"] {{
    color: {theme_text_secondary} !important;
    font-size: 0.92rem;
    font-weight: 500;
}}

.hero-title {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: {theme_text};
    letter-spacing: -0.5px;
    margin-bottom: 0.15rem;
    text-transform: uppercase;
}}

.hero-subtitle {{
    font-size: 1.05rem;
    color: {theme_text_secondary};
    font-weight: 500;
    margin-bottom: 1.2rem;
}}

.metric-card {{
    background: {theme_card_bg};
    border: 1px solid {theme_card_border};
    border-radius: 10px;
    padding: 1.15rem 1.30rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}
.metric-card:hover {{
    border-color: {theme_card_hover_border};
}}

.metric-label {{
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: {theme_text} !important;
    margin-bottom: 0.30rem;
}}

.metric-val {{
    font-size: 1.85rem;
    font-weight: 800;
    color: {theme_metric_val};
    line-height: 1.15;
    letter-spacing: -0.5px;
}}

.metric-sub {{
    font-size: 0.88rem;
    color: #00e5a3;
    margin-top: 0.35rem;
    font-weight: 600;
}}

.badge {{
    display: inline-block;
    padding: 0.30rem 0.75rem;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
.badge-cuda {{
    background: rgba(118, 185, 0, 0.18);
    color: #76b900;
    border: 1px solid rgba(118, 185, 0, 0.50);
}}
.badge-pinn {{
    background: rgba(0, 229, 163, 0.18);
    color: #00e5a3;
    border: 1px solid rgba(0, 229, 163, 0.50);
}}
.badge-quantum {{
    background: rgba(0, 180, 255, 0.18);
    color: #00b4ff;
    border: 1px solid rgba(0, 180, 255, 0.50);
}}

button[data-baseweb="tab"] {{
    font-size: 1.02rem !important;
    font-weight: 600 !important;
    color: {theme_text} !important;
}}
</style>
"""
st.markdown(ENTERPRISE_THEME_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. CACHED OPTIMIZER LOADER
# ------------------------------------------------------------------------------
@st.cache_resource
def load_optimizer():
    return GPU_QPSO_Optimizer()

try:
    optimizer = load_optimizer()
except Exception as e:
    st.error(f"Failed to initialize optimizer engine: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# 4. SIDEBAR: MISSION DISPATCH (PRESET OR MULTI-SHIP CUSTOM FLEET)
# ------------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("shipping_routes_banner.png"):
        st.image("shipping_routes_banner.png", use_container_width=True)
    st.markdown("### VOYAGE MISSION DISPATCH")
    st.caption("Multi-Objective Fleet Route Optimization across Navigable Sea Lanes")

    dispatch_mode = st.radio("Mission Dispatch Mode", ["Global Preset Corridor", "Custom Multi-Ship Fleet (1-6 Ships)"])

    mission_fleet = []

    if dispatch_mode == "Global Preset Corridor":
        route_options = [r["name"] for r in routing_core.PRESET_GLOBAL_ROUTES]
        selected_route_name = st.selectbox("Global Shipping Corridor", route_options, index=0)
        preset_info = next(r for r in routing_core.PRESET_GLOBAL_ROUTES if r["name"] == selected_route_name)

        p_orig = routing_core.get_port_by_name(preset_info["origin"])
        p_dest = routing_core.get_port_by_name(preset_info["dest"])

        lats_c, lons_c, dist_km = routing_core.compute_sea_route(
            p_orig["lat"], p_orig["lon"], p_dest["lat"], p_dest["lon"],
            origin_hub=p_orig.get("hub"), dest_hub=p_dest.get("hub")
        )
        lats_q, lons_q = routing_core.generate_quantum_sea_track(lats_c, lons_c, intensity=1.8, seed=42)
        calc_dist_nm = round(dist_km * routing_core.KM_TO_NM, 1)

        st.info(f"**Origin:** {p_orig['name']} ({p_orig['country']})\n\n**Destination:** {p_dest['name']} ({p_dest['country']})\n\n**Calculated Nautical Distance:** `{calc_dist_nm:,.1f} NM` ({dist_km:,.1f} km)")

        distance_nm = st.number_input("Confirmed Sea Distance (NM)", min_value=100.0, max_value=25000.0, value=float(calc_dist_nm), step=50.0)
        default_eta = max(24, int(distance_nm / 14.0 + 20))
        hard_eta_hours = st.slider("Required Schedule Window (Hours)", min_value=24, max_value=1200, value=default_eta, step=12)

        mission_fleet.append({
            "name": f"AURA {preset_info['vessel_class'].split()[0]} Flagship",
            "vessel_class": preset_info["vessel_class"],
            "origin": p_orig["name"],
            "origin_country": p_orig["country"],
            "dest": p_dest["name"],
            "dest_country": p_dest["country"],
            "distance_nm": distance_nm,
            "eta_window_hours": hard_eta_hours,
            "fuel_type": preset_info["primary_fuel"],
            "target_load": 0.85,
            "route_lats_classical": lats_c,
            "route_lons_classical": lons_c,
            "route_lats_quantum": lats_q,
            "route_lons_quantum": lons_q,
            "p_orig": p_orig,
            "p_dest": p_dest
        })

    else:
        st.markdown("#### Configure Custom Multi-Ship Fleet")
        num_custom_ships = st.slider("Number of Custom Vessels", min_value=1, max_value=6, value=2)
        port_names = [p["name"] for p in routing_core.GLOBAL_PORTS]
        vessel_classes = ["ULCS (24,000 TEU)", "VLCC Crude Carrier", "Panamax Container", "Capesize Bulk", "Aframax Tanker", "Post-Panamax Cargo"]

        for s_i in range(num_custom_ships):
            with st.expander(f"Vessel {s_i+1} Configuration", expanded=(s_i == 0)):
                c1_p, c2_p = st.columns(2)
                with c1_p:
                    orig_name = st.selectbox(f"V{s_i+1} Origin Port", port_names, index=(s_i * 3) % len(port_names), key=f"orig_{s_i}")
                    v_cls = st.selectbox(f"V{s_i+1} Vessel Class", vessel_classes, index=s_i % len(vessel_classes), key=f"vcls_{s_i}")
                with c2_p:
                    dest_name = st.selectbox(f"V{s_i+1} Destination Port", port_names, index=(s_i * 3 + 4) % len(port_names), key=f"dest_{s_i}")
                    f_pref = st.selectbox(f"V{s_i+1} Target Fuel", FUEL_TYPES, index=s_i % len(FUEL_TYPES), key=f"fuel_{s_i}")

                p_orig_i = routing_core.get_port_by_name(orig_name)
                p_dest_i = routing_core.get_port_by_name(dest_name)

                lats_c, lons_c, dist_km = routing_core.compute_sea_route(
                    p_orig_i["lat"], p_orig_i["lon"], p_dest_i["lat"], p_dest_i["lon"],
                    origin_hub=p_orig_i.get("hub"), dest_hub=p_dest_i.get("hub")
                )
                lats_q, lons_q = routing_core.generate_quantum_sea_track(lats_c, lons_c, intensity=1.8, seed=s_i*11+7)
                calc_dist_nm = round(dist_km * routing_core.KM_TO_NM, 1)

                st.caption(f"Calculated Sea Distance: **{calc_dist_nm:,.1f} NM** ({dist_km:,.1f} km)")
                eta_hrs = max(24, int(calc_dist_nm / 14.0 + 20))

                mission_fleet.append({
                    "name": f"AURA Custom {v_cls.split()[0]} {s_i+1:02d}",
                    "vessel_class": v_cls,
                    "origin": p_orig_i["name"],
                    "origin_country": p_orig_i["country"],
                    "dest": p_dest_i["name"],
                    "dest_country": p_dest_i["country"],
                    "distance_nm": calc_dist_nm,
                    "eta_window_hours": eta_hrs,
                    "fuel_type": f_pref,
                    "target_load": 0.80,
                    "route_lats_classical": lats_c,
                    "route_lons_classical": lons_c,
                    "route_lats_quantum": lats_q,
                    "route_lons_quantum": lons_q,
                    "p_orig": p_orig_i,
                    "p_dest": p_dest_i
                })

    st.markdown("---")
    st.markdown("### SEA STATE & METEOROLOGY")
    expanded_weather = routing_core.get_expanded_weather_types()
    weather_labels = [w["label"] for w in expanded_weather]
    weather_labels.append("Custom Meteorological Parameters")

    weather_choice = st.selectbox("Meteorological Preset", weather_labels, index=1)

    if weather_choice != "Custom Meteorological Parameters":
        selected_w = next(w for w in expanded_weather if w["label"] == weather_choice)
        wave_height = selected_w["wave"]
        wind_speed = selected_w["wind"]
    else:
        wave_height = st.slider("Significant Wave Height (m)", 0.5, 8.0, 2.5, 0.1)
        wind_speed = st.slider("Wind Velocity (knots)", 5.0, 60.0, 20.0, 1.0)

    st.markdown(f"**Operational Sea State:** Wave Height `{wave_height:.1f} m` | Wind Velocity `{wind_speed:.1f} kts`")

    # Inject environmental conditions into fleet
    for ship in mission_fleet:
        ship["wave_height_m"] = wave_height
        ship["wind_knots"] = wind_speed

    st.markdown("---")
    st.markdown("### OPTIMIZATION ENGINE CONFIGURATION")
    carbon_tax = st.slider("Regulatory Carbon Tax ($ / ton CO2)", min_value=0, max_value=300, value=100, step=10)
    swarm_size = st.select_slider("Optimizer Particle Swarm Size", options=[32, 64, 128, 256], value=128)
    iterations = st.slider("Optimization Convergence Iterations", min_value=20, max_value=120, value=60, step=10)

    st.markdown("---")
    run_btn = st.button("RUN FLEET OPTIMIZATION", type="primary", use_container_width=True)

# ------------------------------------------------------------------------------
# 5. EXECUTE MULTI-VESSEL / SINGLE-VESSEL OPTIMIZATION
# ------------------------------------------------------------------------------
dispatch_hash = f"{len(mission_fleet)}_{[s['origin']+s['dest'] for s in mission_fleet]}_{wave_height}_{wind_speed}_{carbon_tax}_{swarm_size}_{iterations}"

if "fleet_opt_results" not in st.session_state or run_btn or st.session_state.get("last_dispatch_hash") != dispatch_hash:
    st.session_state["last_dispatch_hash"] = dispatch_hash
    with st.spinner("Executing high-performance multi-vessel GPU optimization..."):
        if len(mission_fleet) == 1:
            ship0 = mission_fleet[0]
            opt_single = optimizer.optimize(
                distance_nm=float(ship0["distance_nm"]),
                max_eta_hours=float(ship0["eta_window_hours"]),
                wave_height_m=float(wave_height),
                wind_knots=float(wind_speed),
                carbon_tax_per_ton=float(carbon_tax),
                swarm_size=swarm_size,
                iterations=iterations
            )
            # Build single record
            st.session_state["fleet_opt_results"] = {
                "fleet_size": 1,
                "total_fleet_cost_usd": opt_single["total_cost_usd"],
                "total_fleet_co2_tons": opt_single["co2_emissions_tons"],
                "total_fleet_fuel_tons": opt_single["fuel_consumed_tons"],
                "elapsed_ms": opt_single["elapsed_ms"],
                "convergence_history": opt_single["convergence_history"],
                "vessels": [{
                    "ship_id": "IMO-98101",
                    "name": ship0["name"],
                    "vessel_class": ship0["vessel_class"],
                    "origin": ship0["origin"],
                    "origin_country": ship0["origin_country"],
                    "dest": ship0["dest"],
                    "dest_country": ship0["dest_country"],
                    "distance_nm": ship0["distance_nm"],
                    "optimal_speed_knots": opt_single["optimal_speed_knots"],
                    "optimal_fuel_name": opt_single["optimal_fuel_name"],
                    "optimal_cargo_load": opt_single["optimal_cargo_load"],
                    "fuel_consumed_tons": opt_single["fuel_consumed_tons"],
                    "co2_emissions_tons": opt_single["co2_emissions_tons"],
                    "travel_time_hours": opt_single["travel_time_hours"],
                    "eta_penalty_usd": opt_single["eta_penalty_usd"],
                    "voyage_cost_usd": opt_single["total_cost_usd"]
                }]
            }
        else:
            st.session_state["fleet_opt_results"] = optimizer.optimize_multi_fleet(
                fleet_items=mission_fleet,
                swarm_size=swarm_size,
                iterations=iterations,
                carbon_tax_per_ton=float(carbon_tax)
            )

        # Multi-fuel comparison for flagship / first vessel
        ship0 = mission_fleet[0]
        v0_opt = st.session_state["fleet_opt_results"]["vessels"][0]
        opt_s = v0_opt["optimal_speed_knots"]
        opt_c = v0_opt["optimal_cargo_load"]
        fuel_comparisons = []
        for f_idx, f_name in enumerate(FUEL_TYPES):
            part = torch.tensor([[opt_s, float(f_idx), opt_c]], dtype=torch.float32, device=optimizer.device)
            c, m = optimizer.evaluate_fitness_batch(
                part, float(ship0["distance_nm"]), float(ship0["eta_window_hours"]), float(wave_height), float(wind_speed),
                float(carbon_tax), fixed_fuel_idx=f_idx, fixed_cargo_load=opt_c
            )
            fuel_comparisons.append({
                "Fuel": f_name,
                "Fuel Cost ($)": float(m["fuel_cost_usd"][0].item()),
                "CO2 Tax ($)": float(m["co2_emissions_tons"][0].item() * carbon_tax),
                "Total Cost ($)": float(c[0].item()),
                "CO2 Emissions (tons)": float(m["co2_emissions_tons"][0].item()),
                "Fuel Mass (tons)": float(m["fuel_consumed_tons"][0].item()),
            })
        st.session_state["fuel_comp_df"] = pd.DataFrame(fuel_comparisons)
        st.session_state["mission_fleet_saved"] = mission_fleet

fleet_res = st.session_state["fleet_opt_results"]
comp_df = st.session_state["fuel_comp_df"]
active_mission_fleet = st.session_state["mission_fleet_saved"]

# ------------------------------------------------------------------------------
# 6. HEADER & PRIMARY KPI METRICS
# ------------------------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="hero-title">Green Maritime Fleet Optimization Platform</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero-subtitle">Physics-Informed Surrogate Modeling and High-Performance Multi-Objective Route Optimization for Maritime Decarbonization</div>',
        unsafe_allow_html=True
    )
with col_h2:
    st.markdown(
        f'<div style="text-align: right; padding-top: 0.3rem;">'
        f'<span class="badge badge-cuda">Hardware: NVIDIA {str(optimizer.device).upper()}</span><br>'
        f'<span class="badge badge-pinn" style="margin-top: 4px;">Physics Monotonic Loss: Active</span><br>'
        f'<span class="badge badge-quantum" style="margin-top: 4px;">Solver: Quantum-Inspired PSO</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# Primary KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    avg_speed = np.mean([v["optimal_speed_knots"] for v in fleet_res["vessels"]])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Cruising Speed</div>
        <div class="metric-val">{avg_speed:.2f} <span style="font-size: 1.05rem; font-weight: 600;">knots</span></div>
        <div class="metric-sub">Active Mission Vessels: {fleet_res['fleet_size']}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    rec_fuels = ", ".join(list(set([v["optimal_fuel_name"] for v in fleet_res["vessels"]])))
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Recommended Fuel Mix</div>
        <div class="metric-val" style="color: #00e5a3; font-size: 1.55rem;">{rec_fuels}</div>
        <div class="metric-sub">Fleet Fuel: {fleet_res['total_fleet_fuel_tons']:,.1f} metric tons</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    avg_load = np.mean([v["optimal_cargo_load"] for v in fleet_res["vessels"]]) * 100.0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fleet Capacity Utilization</div>
        <div class="metric-val">{avg_load:.1f} <span style="font-size: 1.05rem; font-weight: 600;">%</span></div>
        <div class="metric-sub">Displacement Scaled Efficiency</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Fleet CO2 Emissions</div>
        <div class="metric-val" style="color: #00b4ff;">{fleet_res['total_fleet_co2_tons']:,.1f} <span style="font-size: 1.05rem; font-weight: 600;">tons</span></div>
        <div class="metric-sub">Carbon Tax @ ${carbon_tax}/ton: ${fleet_res['total_fleet_co2_tons'] * carbon_tax:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Secondary Ribbon
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Total Fleet Financial Liability", f"${fleet_res['total_fleet_cost_usd']:,.2f}")
with s2:
    st.metric("Total Carbon Tax Liability", f"${fleet_res['total_fleet_co2_tons'] * carbon_tax:,.2f}")
with s3:
    tot_delays = sum(v["eta_penalty_usd"] for v in fleet_res["vessels"])
    st.metric("Total Schedule Delay Penalties", f"${tot_delays:,.2f}")
with s4:
    st.metric("Engine Execution Latency", f"{fleet_res['elapsed_ms']:.2f} ms")

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 7. DASHBOARD TABS
# ------------------------------------------------------------------------------
tab_map, tab_mission, tab_benchmarks, tab_pinn, tab_dataset = st.tabs([
    "Global Fleet Routing",
    "Mission Dispatch & Multi-Fuel Matrix",
    "Algorithmic Benchmark Evaluation",
    "Physics Monotonicity & Model Verification",
    "Vessel Telemetry Data Explorer"
])

# ==============================================================================
# TAB 0: GLOBAL REAL-TIME FLEET ROUTING
# ==============================================================================
with tab_map:
    st.markdown("### Global Real-Time Fleet Routing and Sea Navigation")
    st.caption("100% Sea-Navigable International Corridors spanning Americas, Australasia/Oceania, Asia, Europe, Middle East, and Africa.")

    c_map1, c_map2 = st.columns([1, 3])

    with c_map1:
        st.markdown("#### Worldwide Fleet Settings")
        fleet_size_ctrl = st.slider("Worldwide Active Fleet Size", 3, 20, 9)

        if st.button("Deploy Worldwide Fleet", type="primary", use_container_width=True):
            st.session_state["global_fleet"] = routing_core.generate_random_fleet(fleet_size_ctrl)

        if "global_fleet" not in st.session_state or len(st.session_state["global_fleet"]) != fleet_size_ctrl:
            st.session_state["global_fleet"] = routing_core.generate_random_fleet(fleet_size_ctrl)

        st.markdown("#### Global Trade Quadrants")
        st.markdown("- **Americas**: Transpacific, Transatlantic, Panama & South America\n- **Australasia**: Oceania-Americas, Oceania-Asia, Indo-Pacific\n- **Asia-Europe**: Suez Canal, Mediterranean & Cape Route\n- **Middle East**: Persian Gulf Energy Corridors")

    with c_map2:
        fig_map = go.Figure()

        for s_idx, ship in enumerate(st.session_state["global_fleet"]):
            lats_classical = ship["route_lats_classical"]
            lons_classical = ship["route_lons_classical"]
            lats_quantum = ship["route_lats_quantum"]
            lons_quantum = ship["route_lons_quantum"]

            # Standard Route
            fig_map.add_trace(go.Scattergeo(
                lon=lons_classical, lat=lats_classical, mode='lines',
                line=dict(width=1.8, color='#ff477e', dash='dot'),
                name="Standard Sea Lane",
                showlegend=(s_idx == 0),
                hoverinfo='skip'
            ))

            # Optimized Route
            fig_map.add_trace(go.Scattergeo(
                lon=lons_quantum, lat=lats_quantum, mode='lines+markers',
                line=dict(width=2.8, color='#00e5a3'),
                marker=dict(size=3.5, color='#00e5a3'),
                name="Hydrodynamically Optimized Track",
                showlegend=(s_idx == 0),
                hovertext=(
                    f"<b>{ship['name']}</b> ({ship['vessel_class']})<br>"
                    f"Route: {ship['origin']} to {ship['dest']}<br>"
                    f"Sea Distance: {ship['distance_nm']:,.1f} NM<br>"
                    f"Fuel: {ship['fuel_type']} | Load: {ship['target_load']*100:.0f}%"
                )
            ))

            # Ports
            fig_map.add_trace(go.Scattergeo(
                lon=[ship["start_lon"], ship["dest_lon"]],
                lat=[ship["start_lat"], ship["dest_lat"]],
                mode='markers+text',
                marker=dict(size=7, color='#00b4ff', symbol='diamond'),
                text=[ship["origin"].split()[0], ship["dest"].split()[0]],
                textposition="top center",
                textfont=dict(size=9, color=theme_metric_val),
                showlegend=False,
                hovertext=[f"Origin: {ship['origin']} ({ship['origin_country']})", f"Destination: {ship['dest']} ({ship['dest_country']})"]
            ))

        fig_map.update_layout(
            template=plotly_template,
            paper_bgcolor=plot_paper_bg,
            geo=dict(
                showland=True, landcolor=geo_land,
                showocean=True, oceancolor=geo_ocean,
                showcountries=True, countrycolor=geo_country,
                showcoastlines=True, coastlinecolor=geo_coast,
                projection_type="equirectangular",
                bgcolor=plot_paper_bg,
                center=dict(lat=15, lon=0)
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=10, b=0),
            height=530
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("#### Worldwide Active Fleet Deployment Roster")
    fleet_records = []
    for s in st.session_state["global_fleet"]:
        v_spd = 15.5
        v_hrs = s["distance_nm"] / v_spd
        fuel_tons = v_hrs * (1.2 + 0.0003 * s["distance_nm"])
        co2_tons = fuel_tons * 0.54 if s["fuel_type"] == "Methanol" else (fuel_tons * 0.15 if s["fuel_type"] == "Ammonia" else fuel_tons * 2.75)

        fleet_records.append({
            "Vessel ID": s["ship_id"],
            "Vessel Name": s["name"],
            "Vessel Class": s["vessel_class"],
            "Origin Port": f"{s['origin']} ({s['origin_country']})",
            "Destination Port": f"{s['dest']} ({s['dest_country']})",
            "Nautical Distance (NM)": s["distance_nm"],
            "Primary Fuel": s["fuel_type"],
            "Cargo Load": f"{s['target_load']*100:.0f}%",
            "Estimated Voyage (Hours)": f"{v_hrs:.1f} h",
            "Estimated Fuel (Tons)": f"{fuel_tons:.1f} t",
            "Estimated CO2 (Tons)": f"{co2_tons:.1f} t",
        })
    st.dataframe(pd.DataFrame(fleet_records), use_container_width=True)

# ==============================================================================
# TAB 1: MISSION DISPATCH & MULTI-FUEL MATRIX
# ==============================================================================
with tab_mission:
    st.markdown("### Mission Dispatch Optimization & Route Analysis")
    st.caption("Detailed breakdown for currently dispatched mission vessels and alternative fuel lifecycle trade-offs.")

    # Mission map showing all dispatched ships
    fig_mission_map = go.Figure()
    for s_idx, ship in enumerate(active_mission_fleet):
        lats_c = ship["route_lats_classical"]
        lons_c = ship["route_lons_classical"]
        lats_q = ship["route_lats_quantum"]
        lons_q = ship["route_lons_quantum"]

        fig_mission_map.add_trace(go.Scattergeo(
            lon=lons_c, lat=lats_c, mode='lines',
            line=dict(width=2.2, color='#ff477e', dash='dash'),
            name=f"{ship['name']} (Standard Fairway)",
            showlegend=(s_idx == 0)
        ))
        fig_mission_map.add_trace(go.Scattergeo(
            lon=lons_q, lat=lats_q, mode='lines+markers',
            line=dict(width=3.2, color='#00e5a3'),
            marker=dict(size=4, color='#00e5a3'),
            name=f"{ship['name']} (Optimized Track)",
            showlegend=(s_idx == 0),
            hovertext=f"Vessel: {ship['name']}<br>Route: {ship['origin']} to {ship['dest']}<br>Distance: {ship['distance_nm']:,.1f} NM"
        ))
        fig_mission_map.add_trace(go.Scattergeo(
            lon=[ship["p_orig"]["lon"], ship["p_dest"]["lon"]],
            lat=[ship["p_orig"]["lat"], ship["p_dest"]["lat"]],
            mode='markers+text',
            marker=dict(size=8, color='#00b4ff', symbol='diamond'),
            text=[ship["origin"].split()[0], ship["dest"].split()[0]],
            textposition="top center",
            textfont=dict(size=10, color=theme_metric_val),
            showlegend=False
        ))

    fig_mission_map.update_layout(
        template=plotly_template,
        paper_bgcolor=plot_paper_bg,
        geo=dict(
            showland=True, landcolor=geo_land,
            showocean=True, oceancolor=geo_ocean,
            showcountries=True, countrycolor=geo_country,
            showcoastlines=True, coastlinecolor=geo_coast,
            projection_type="equirectangular",
            bgcolor=plot_paper_bg
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=400
    )
    st.plotly_chart(fig_mission_map, use_container_width=True)

    # Dispatched Vessel Results Table
    st.markdown("#### Dispatched Fleet Optimization Summary")
    st.dataframe(pd.DataFrame(fleet_res["vessels"]).style.format({
        "distance_nm": "{:,.1f} NM",
        "optimal_speed_knots": "{:.2f} kts",
        "optimal_cargo_load": "{:.1%}",
        "fuel_consumed_tons": "{:,.2f} t",
        "co2_emissions_tons": "{:,.2f} t",
        "travel_time_hours": "{:.1f} h",
        "eta_penalty_usd": "${:,.2f}",
        "voyage_cost_usd": "${:,.2f}"
    }), use_container_width=True)

    row1_c1, row1_c2 = st.columns([1, 1])

    with row1_c1:
        st.markdown("#### Optimization Convergence Trajectory")
        st.caption("Objective function minimization ($) across optimization iterations")

        hist = fleet_res["convergence_history"]
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(range(len(hist))),
            y=hist,
            mode='lines+markers',
            name='Optimal Fitness',
            line=dict(color='#00e5a3', width=3),
            marker=dict(size=5, color='#00b4ff'),
            fill='tozeroy',
            fillcolor='rgba(0, 229, 163, 0.08)'
        ))
        fig_conv.update_layout(
            template=plotly_template,
            paper_bgcolor=plot_paper_bg,
            plot_bgcolor=plot_bg,
            xaxis=dict(title="Iteration Step", gridcolor=grid_color),
            yaxis=dict(title="Total Financial Liability ($)", gridcolor=grid_color, tickformat="$,.0f"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=320
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    with row1_c2:
        st.markdown("#### Alternative Fuel Lifecycle Comparison")
        st.caption("Evaluated at optimal speed & cargo load for flagship route")

        fig_fuel = go.Figure()
        fig_fuel.add_trace(go.Bar(
            name='Fuel Bunker Cost ($)',
            x=comp_df["Fuel"],
            y=comp_df["Fuel Cost ($)"],
            marker_color='#2d62ed'
        ))
        fig_fuel.add_trace(go.Bar(
            name='Carbon Tax Liability ($)',
            x=comp_df["Fuel"],
            y=comp_df["CO2 Tax ($)"],
            marker_color='#00e5a3'
        ))
        fig_fuel.update_layout(
            barmode='stack',
            template=plotly_template,
            paper_bgcolor=plot_paper_bg,
            plot_bgcolor=plot_bg,
            xaxis=dict(title="Marine Fuel Type", gridcolor=grid_color),
            yaxis=dict(title="Cost Breakdown ($)", gridcolor=grid_color, tickformat="$,.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=20, b=20),
            height=320
        )
        st.plotly_chart(fig_fuel, use_container_width=True)

# ==============================================================================
# TAB 2: BENCHMARK EVALUATION (MULTI-VESSEL HIGH-DIMENSIONAL FLEET)
# ==============================================================================
with tab_benchmarks:
    st.markdown("### Algorithmic Benchmark: Multi-Vessel Fleet Optimization")
    st.caption("Empirical performance comparison of GPU QPSO vs. Classical Velocity-Based PSO and Monte Carlo Random Search on high-dimensional multi-vessel fleet dispatch (D = 3M variables).")

    b_col1, b_col2 = st.columns([1, 3])
    with b_col1:
        st.markdown("#### Benchmark Configuration")
        bench_fleet_size = st.slider("Benchmark Fleet Size (Vessels)", min_value=4, max_value=24, value=12, step=4)
        bench_iters = st.slider("Benchmark Iterations", min_value=40, max_value=120, value=80, step=20)
        run_bench_btn = st.button("Run Multi-Fleet Benchmark", type="primary", use_container_width=True)

    with b_col2:
        benchmark_file = "benchmark_results.json"
        if run_bench_btn or not os.path.exists(benchmark_file):
            with st.spinner(f"Executing Multi-Fleet Benchmark on {bench_fleet_size} vessels across global routes..."):
                from benchmark import run_benchmarks
                bench_data = run_benchmarks(
                    fleet_size=bench_fleet_size,
                    carbon_tax_per_ton=float(carbon_tax),
                    swarm_size=128,
                    iterations=bench_iters
                )
        else:
            with open(benchmark_file, "r") as f:
                bench_data = json.load(f)

        methods = bench_data["methods"]
        m_names = [methods[k]["method"] for k in methods]
        m_costs = [methods[k]["best_fitness_usd"] for k in methods]
        m_times = [methods[k]["elapsed_ms"] for k in methods]

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("#### Total Fleet Voyage Cost ($)")
            fig_cost = px.bar(
                x=m_names, y=m_costs, color=m_names,
                color_discrete_sequence=['#ff6b6b', '#ffd166', '#00e5a3'],
                labels={"x": "Algorithm", "y": "Fleet Cost ($)"}
            )
            fig_cost.update_layout(
                template=plotly_template,
                paper_bgcolor=plot_paper_bg,
                plot_bgcolor=plot_bg,
                xaxis=dict(gridcolor=grid_color),
                yaxis=dict(gridcolor=grid_color, tickformat="$,.0f"),
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        with col_b2:
            st.markdown("#### Compute Execution Time (ms)")
            fig_time = px.bar(
                x=m_names, y=m_times, color=m_names,
                color_discrete_sequence=['#ff6b6b', '#ffd166', '#00e5a3'],
                labels={"x": "Algorithm", "y": "Execution Time (ms)"}
            )
            fig_time.update_layout(
                template=plotly_template,
                paper_bgcolor=plot_paper_bg,
                plot_bgcolor=plot_bg,
                xaxis=dict(gridcolor=grid_color),
                yaxis=dict(gridcolor=grid_color),
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("#### Multi-Algorithm Fleet Convergence Trajectory")
    fig_comp_conv = go.Figure()
    colors = {'random_search': '#ff6b6b', 'classical_pso': '#ffd166', 'gpu_qpso': '#00e5a3'}
    for k, v in methods.items():
        fig_comp_conv.add_trace(go.Scatter(
            x=list(range(len(v["convergence_history"]))),
            y=v["convergence_history"],
            mode='lines',
            name=v["method"],
            line=dict(color=colors.get(k, '#ffffff'), width=3.5 if k == 'gpu_qpso' else 2)
        ))
    fig_comp_conv.update_layout(
        template=plotly_template,
        paper_bgcolor=plot_paper_bg,
        plot_bgcolor=plot_bg,
        xaxis=dict(title="Iteration Step", gridcolor=grid_color),
        yaxis=dict(title="Best Discovered Fleet Cost ($)", gridcolor=grid_color, tickformat="$,.0f"),
        height=320,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_comp_conv, use_container_width=True)

# ==============================================================================
# TAB 3: MODEL MONOTONICITY & VERIFICATION
# ==============================================================================
with tab_pinn:
    st.markdown("### Neural Surrogate Benchmarking: Physics-Informed vs Classical")
    st.markdown(
        """
        The physics-informed surrogate strictly enforces hydrodynamic monotonicity ($\\frac{\\partial \\text{Fuel}}{\\partial v} \\ge 0$), eliminating unphysical predictions where higher vessel speeds incorrectly yield lower fuel consumption.
        $$\\mathcal{L}_{\\text{PINN}} = \\text{MSE}(y_{\\text{pred}}, y_{\\text{true}}) + 0.1 \\times \\text{Mean}\\left(\\text{ReLU}\\left(-\\frac{\\partial y_{\\text{pred}}}{\\partial v_{\\text{speed}}}\\right)\\right)$$
        """
    )

    col_p1, col_p2 = st.columns([1, 1])

    speeds_sweep = np.linspace(10.0, 22.0, 50)

    with col_p1:
        st.markdown("#### Hydrodynamic Monotonicity Verification")
        fig_pinn_curves = go.Figure()

        matrix = np.array([[s, 0.70, 2.5, 20.0] for s in speeds_sweep], dtype=np.float32)
        mat_scaled = (torch.tensor(matrix, device=optimizer.device) - optimizer.feat_mean) / optimizer.feat_scale

        from train_pinn import ClassicalFuelPredictor
        classical_model = ClassicalFuelPredictor(input_dim=4).to(optimizer.device)
        try:
            classical_model.load_state_dict(torch.load("classical_predictor.pth", map_location=optimizer.device, weights_only=True))
            classical_model.eval()
            has_classical = True
        except:
            has_classical = False

        with torch.no_grad():
            preds_pinn = optimizer.model(mat_scaled).squeeze(-1)
            burn_pinn = (preds_pinn * optimizer.tgt_scale + optimizer.tgt_mean).cpu().numpy()

            if has_classical:
                preds_class = classical_model(mat_scaled).squeeze(-1)
                burn_class = (preds_class * optimizer.tgt_scale + optimizer.tgt_mean).cpu().numpy()

        fig_pinn_curves.add_trace(go.Scatter(
            x=speeds_sweep, y=burn_pinn, mode='lines', name='PINN Surrogate (Physics-Constrained)',
            line=dict(color='#00e5a3', width=4)
        ))

        if has_classical:
            fig_pinn_curves.add_trace(go.Scatter(
                x=speeds_sweep, y=burn_class, mode='lines', name='Standard Neural Network (Unconstrained)',
                line=dict(color='#ff477e', width=2.5, dash='dash')
            ))

        fig_pinn_curves.update_layout(
            template=plotly_template,
            paper_bgcolor=plot_paper_bg,
            plot_bgcolor=plot_bg,
            xaxis=dict(title="Vessel Speed Over Ground (knots)", gridcolor=grid_color),
            yaxis=dict(title="HFO Fuel Burn Rate (tons/hr)", gridcolor=grid_color),
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pinn_curves, use_container_width=True)

    with col_p2:
        st.markdown("#### Interactive Surrogate Evaluation")
        test_speed = st.slider("Vessel Speed (knots)", 10.0, 22.0, 16.0, 0.5)
        test_cargo = st.slider("Cargo Capacity Load (%)", 20.0, 100.0, 75.0, 5.0) / 100.0
        test_wave = st.slider("Significant Wave Height (m)", 0.5, 5.0, 2.5, 0.5)
        test_wind = st.slider("Wind Velocity (knots)", 5.0, 45.0, 20.0, 5.0)

        sandbox_in = torch.tensor([[test_speed, test_cargo, test_wave, test_wind]], dtype=torch.float32, device=optimizer.device)
        sandbox_in_scaled = (sandbox_in - optimizer.feat_mean) / optimizer.feat_scale
        with torch.no_grad():
            pred_sandbox = optimizer.model(sandbox_in_scaled).item() * optimizer.tgt_scale + optimizer.tgt_mean

        st.markdown(f"""
        <div class="metric-card" style="margin-top: 15px;">
            <div class="metric-label">Predicted Instantaneous HFO Consumption</div>
            <div class="metric-val" style="color: #00e5a3;">{pred_sandbox:.4f} <span style="font-size: 1.05rem; color: {theme_text};">tons / hour</span></div>
            <div class="metric-sub">Daily Fuel Burn Rate: ~{pred_sandbox * 24.0:.2f} metric tons / day</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: TELEMETRY DATA EXPLORER
# ==============================================================================
with tab_dataset:
    st.markdown("### Vessel Telemetry and Fuel Consumption Dataset Explorer")
    st.caption("Inspection of hydrodynamic maritime telemetry records across operational vessel classes, speed regimes, and alternative fuels.")

    dataset_options = []
    if os.path.exists("green_maritime_fleet_dataset.csv"):
        dataset_options.append("Green Maritime Physics Dataset (green_maritime_fleet_dataset.csv)")
    if os.path.exists("processed_ship_telemetry.csv"):
        dataset_options.append("Preprocessed Telemetry (processed_ship_telemetry.csv)")

    if dataset_options:
        selected_ds_name = st.selectbox("Telemetry Dataset Source", dataset_options)
        ds_file = "green_maritime_fleet_dataset.csv" if "green_maritime_fleet_dataset.csv" in selected_ds_name else "processed_ship_telemetry.csv"
        telemetry_df = pd.read_csv(ds_file)

        st.markdown(f"**Active Records:** `{len(telemetry_df):,}` observations | `{len(telemetry_df.columns)}` feature dimensions")

        col_d1, col_d2 = st.columns([1, 1])

        with col_d1:
            st.markdown("#### Speed vs Fuel Consumption Distribution")
            speed_col = "Speed_Over_Ground_knots" if "Speed_Over_Ground_knots" in telemetry_df.columns else telemetry_df.columns[0]
            fuel_col = "hfo_consumption_ton_per_hr" if "hfo_consumption_ton_per_hr" in telemetry_df.columns else telemetry_df.columns[1]
            color_col = "Weather_Condition" if "Weather_Condition" in telemetry_df.columns else ("Primary_Fuel" if "Primary_Fuel" in telemetry_df.columns else None)

            fig_scatter = px.scatter(
                telemetry_df.sample(min(800, len(telemetry_df))),
                x=speed_col,
                y=fuel_col,
                color=color_col,
                color_discrete_map={"Calm": "#00b4ff", "Moderate": "#00e5a3", "Rough": "#ffd166", "Storm": "#ff477e"},
                labels={speed_col: "Speed (knots)", fuel_col: "Fuel Burn (tons/hr)"},
                template=plotly_template
            )
            fig_scatter.update_layout(
                paper_bgcolor=plot_paper_bg,
                plot_bgcolor=plot_bg,
                xaxis=dict(gridcolor=grid_color),
                yaxis=dict(gridcolor=grid_color),
                height=320
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_d2:
            st.markdown("#### Fleet Composition by Vessel Classification")
            type_col = "Vessel_Class" if "Vessel_Class" in telemetry_df.columns else ("Ship_Type" if "Ship_Type" in telemetry_df.columns else None)
            if type_col:
                ship_counts = telemetry_df[type_col].value_counts().reset_index()
                ship_counts.columns = [type_col, "Count"]
                fig_pie = px.pie(
                    ship_counts,
                    names=type_col,
                    values="Count",
                    color_discrete_sequence=px.colors.sequential.Tealgrn,
                    template=plotly_template,
                    hole=0.45
                )
                fig_pie.update_layout(
                    paper_bgcolor=plot_paper_bg,
                    height=320
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### Sample Telemetry Records")
        st.dataframe(telemetry_df.head(15), use_container_width=True)
    else:
        st.warning("No dataset files found. Run generate_green_maritime_dataset.py first.")
