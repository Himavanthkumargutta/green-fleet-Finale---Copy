"""
prepare_fleet_data.py
Generates a realistic transoceanic global green fleet across all world trade routes.
Saves fleet configuration to fleet_config.json.
"""

import json
import random
from routing_core import GLOBAL_PORTS, PRESET_GLOBAL_ROUTES, compute_sea_route, KM_TO_NM

VESSEL_CLASSES = [
    {"class": "ULCS (24,000 TEU)", "dwt_range": (200000, 240000), "beam": (58.0, 62.0), "draft": (15.5, 16.5)},
    {"class": "VLCC Crude Carrier", "dwt_range": (250000, 320000), "beam": (55.0, 60.0), "draft": (16.0, 18.0)},
    {"class": "Capesize Bulk", "dwt_range": (150000, 200000), "beam": (45.0, 50.0), "draft": (14.0, 16.0)},
    {"class": "Suezmax Tanker", "dwt_range": (120000, 160000), "beam": (42.0, 48.0), "draft": (13.0, 15.0)},
    {"class": "Aframax Tanker", "dwt_range": (80000, 120000), "beam": (38.0, 44.0), "draft": (11.5, 13.5)},
    {"class": "Panamax Container", "dwt_range": (55000, 85000), "beam": (32.2, 32.5), "draft": (11.0, 12.5)},
]

FUEL_TYPES = ["HFO", "LNG", "Methanol", "Ammonia", "Hydrogen"]

def generate_global_fleet(num_ships=20):
    fleet = []
    
    # 1. First add the flagship preset global routes
    for idx, preset in enumerate(PRESET_GLOBAL_ROUTES):
        p_orig = next(p for p in GLOBAL_PORTS if p["name"] == preset["origin"])
        p_dest = next(p for p in GLOBAL_PORTS if p["name"] == preset["dest"])
        
        _, _, dist_km = compute_sea_route(
            p_orig["lat"], p_orig["lon"], p_dest["lat"], p_dest["lon"],
            origin_hub=p_orig.get("hub"), dest_hub=p_dest.get("hub")
        )
        dist_nm = dist_km * KM_TO_NM
        
        ship = {
            "mmsi": 211000000 + idx * 243,
            "ship_id": f"IMO-98{idx+10:03d}",
            "name": f"AURA {preset['origin'].split()[0]}-{preset['dest'].split()[0]} EXPRESS",
            "vessel_class": preset["vessel_class"],
            "primary_fuel": preset["primary_fuel"],
            "cargo_load_tons": preset["cargo_tons"],
            "cargo_load_pct": 0.85,
            "origin_port": preset["origin"],
            "origin_country": p_orig["country"],
            "origin_lat": p_orig["lat"],
            "origin_lon": p_orig["lon"],
            "dest_port": preset["dest"],
            "dest_country": p_dest["country"],
            "dest_lat": p_dest["lat"],
            "dest_lon": p_dest["lon"],
            "distance_nm": round(dist_nm, 1),
            "distance_km": round(dist_km, 1),
            "eta_window_hours": int(dist_nm / 14.5 + 24), # nominal 14.5 kts speed + buffer
            "description": preset["description"]
        }
        fleet.append(ship)
        
    # 2. Add randomized global transoceanic pairings for the remainder
    remaining = num_ships - len(fleet)
    for i in range(remaining):
        p_orig = random.choice(GLOBAL_PORTS)
        p_dest = random.choice(GLOBAL_PORTS)
        while p_orig["name"] == p_dest["name"]:
            p_dest = random.choice(GLOBAL_PORTS)
            
        v_info = random.choice(VESSEL_CLASSES)
        dwt = random.uniform(*v_info["dwt_range"])
        load_pct = random.uniform(0.60, 0.95)
        
        _, _, dist_km = compute_sea_route(
            p_orig["lat"], p_orig["lon"], p_dest["lat"], p_dest["lon"],
            origin_hub=p_orig.get("hub"), dest_hub=p_dest.get("hub")
        )
        dist_nm = dist_km * KM_TO_NM
        
        idx = len(fleet)
        ship = {
            "mmsi": 211000000 + idx * 243,
            "ship_id": f"IMO-98{idx+10:03d}",
            "name": f"AURA {v_info['class'].split()[0]} {idx+1:02d}",
            "vessel_class": v_info["class"],
            "primary_fuel": random.choice(FUEL_TYPES),
            "cargo_load_tons": int(dwt * load_pct),
            "cargo_load_pct": round(load_pct, 2),
            "origin_port": p_orig["name"],
            "origin_country": p_orig["country"],
            "origin_lat": p_orig["lat"],
            "origin_lon": p_orig["lon"],
            "dest_port": p_dest["name"],
            "dest_country": p_dest["country"],
            "dest_lat": p_dest["lat"],
            "dest_lon": p_dest["lon"],
            "distance_nm": round(dist_nm, 1),
            "distance_km": round(dist_km, 1),
            "eta_window_hours": int(dist_nm / 14.0 + 30),
            "description": f"Transoceanic route connecting {p_orig['name']} ({p_orig['country']}) and {p_dest['name']} ({p_dest['country']})."
        }
        fleet.append(ship)
        
    return fleet

def main():
    fleet = generate_global_fleet(20)
    with open("fleet_config.json", "w", encoding="utf-8") as f:
        json.dump(fleet, f, indent=4)
    print(f"[+] Saved {len(fleet)} worldwide transoceanic vessels to fleet_config.json")

if __name__ == "__main__":
    main()
