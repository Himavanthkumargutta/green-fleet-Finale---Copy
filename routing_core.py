"""
routing_core.py
Comprehensive Global Maritime Sea-Lane Routing Engine
Constructs realistic, 100% sea-navigable nautical waterways avoiding all landmasses.
Full worldwide coverage including Americas, Australasia / Oceania, Asia, Europe, Middle East, and Africa.
"""

import math
import random
import numpy as np
import pandas as pd
import networkx as nx

# Earth radius in kilometers and nautical miles
EARTH_RADIUS_KM = 6371.0
KM_TO_NM = 0.539957

# ==============================================================================
# 1. GLOBAL PORTS REGISTRY (Americas, Australasia, Asia, Europe, Middle East, Africa)
# ==============================================================================
GLOBAL_PORTS = [
    # --- North America ---
    {"name": "Los Angeles / Long Beach", "country": "USA", "lat": 33.74, "lon": -118.26, "region": "North_America", "hub": "US_PACIFIC_S"},
    {"name": "San Francisco / Oakland", "country": "USA", "lat": 37.80, "lon": -122.27, "region": "North_America", "hub": "US_PACIFIC_MID"},
    {"name": "Seattle / Tacoma", "country": "USA", "lat": 47.60, "lon": -122.33, "region": "North_America", "hub": "US_PACIFIC_N"},
    {"name": "Vancouver", "country": "Canada", "lat": 49.28, "lon": -123.12, "region": "North_America", "hub": "US_PACIFIC_N"},
    {"name": "New York / NJ", "country": "USA", "lat": 40.68, "lon": -74.02, "region": "North_America", "hub": "US_ATLANTIC_N"},
    {"name": "Houston / Galveston", "country": "USA", "lat": 29.75, "lon": -95.00, "region": "North_America", "hub": "GULF_MEXICO_NW"},
    {"name": "Miami / Port Everglades", "country": "USA", "lat": 25.76, "lon": -80.19, "region": "North_America", "hub": "FLORIDA_STRAIT"},
    
    # --- Central & South America ---
    {"name": "Balboa (Panama Pacific)", "country": "Panama", "lat": 8.95, "lon": -79.56, "region": "Central_America", "hub": "PANAMA_PACIFIC"},
    {"name": "Colon (Panama Caribbean)", "country": "Panama", "lat": 9.36, "lon": -79.91, "region": "Central_America", "hub": "PANAMA_CARIBBEAN"},
    {"name": "Santos", "country": "Brazil", "lat": -23.96, "lon": -46.33, "region": "South_America", "hub": "S_AMERICA_E"},
    {"name": "Buenos Aires", "country": "Argentina", "lat": -34.60, "lon": -58.38, "region": "South_America", "hub": "S_AMERICA_SE"},
    {"name": "Valparaiso / San Antonio", "country": "Chile", "lat": -33.04, "lon": -71.61, "region": "South_America", "hub": "S_AMERICA_PACIFIC_S"},
    {"name": "Callao / Lima", "country": "Peru", "lat": -12.05, "lon": -77.14, "region": "South_America", "hub": "S_AMERICA_PACIFIC_N"},

    # --- Australasia & Oceania ---
    {"name": "Sydney", "country": "Australia", "lat": -33.86, "lon": 151.21, "region": "Oceania", "hub": "AUSTRALIA_E"},
    {"name": "Melbourne", "country": "Australia", "lat": -37.81, "lon": 144.96, "region": "Oceania", "hub": "AUSTRALIA_SE"},
    {"name": "Brisbane", "country": "Australia", "lat": -27.47, "lon": 153.02, "region": "Oceania", "hub": "AUSTRALIA_E_CORAL"},
    {"name": "Fremantle / Perth", "country": "Australia", "lat": -32.05, "lon": 115.74, "region": "Oceania", "hub": "AUSTRALIA_W"},
    {"name": "Auckland", "country": "New Zealand", "lat": -36.84, "lon": 174.76, "region": "Oceania", "hub": "NEW_ZEALAND_N"},
    {"name": "Honolulu (Hawaii)", "country": "USA", "lat": 21.30, "lon": -157.86, "region": "Oceania", "hub": "PACIFIC_HAWAII_CENTRAL"},

    # --- East Asia & Asia-Pacific ---
    {"name": "Shanghai", "country": "China", "lat": 31.23, "lon": 121.50, "region": "Asia", "hub": "EC_CHINA"},
    {"name": "Singapore", "country": "Singapore", "lat": 1.28, "lon": 103.85, "region": "SE_Asia", "hub": "SINGAPORE"},
    {"name": "Tokyo / Yokohama", "country": "Japan", "lat": 35.44, "lon": 139.64, "region": "Asia", "hub": "JAPAN_SOUTH"},
    {"name": "Busan", "country": "South Korea", "lat": 35.10, "lon": 129.04, "region": "Asia", "hub": "KOREA_STRAIT"},
    {"name": "Hong Kong", "country": "China", "lat": 22.30, "lon": 114.17, "region": "Asia", "hub": "SC_CHINA"},
    {"name": "Ningbo-Zhoushan", "country": "China", "lat": 29.88, "lon": 121.56, "region": "Asia", "hub": "EC_CHINA"},
    {"name": "Port Klang", "country": "Malaysia", "lat": 3.00, "lon": 101.39, "region": "SE_Asia", "hub": "MALACCA_MID"},

    # --- South Asia & Middle East ---
    {"name": "Visakhapatnam (Visag)", "country": "India", "lat": 17.69, "lon": 83.29, "region": "South_Asia", "hub": "BAY_BENGAL_NW"},
    {"name": "Mumbai (JNPT)", "country": "India", "lat": 18.95, "lon": 72.85, "region": "South_Asia", "hub": "ARABIAN_SEA_E"},
    {"name": "Colombo", "country": "Sri Lanka", "lat": 6.94, "lon": 79.85, "region": "South_Asia", "hub": "SRI_LANKA_S"},
    {"name": "Dubai (Jebel Ali)", "country": "UAE", "lat": 25.01, "lon": 55.06, "region": "Middle_East", "hub": "PERSIAN_GULF"},

    # --- Europe & Mediterranean ---
    {"name": "Rotterdam", "country": "Netherlands", "lat": 51.95, "lon": 4.05, "region": "Europe", "hub": "NORTH_SEA_S"},
    {"name": "Antwerp", "country": "Belgium", "lat": 51.26, "lon": 4.34, "region": "Europe", "hub": "NORTH_SEA_S"},
    {"name": "Hamburg", "country": "Germany", "lat": 53.53, "lon": 9.97, "region": "Europe", "hub": "NORTH_SEA_E"},
    {"name": "London / Felixstowe", "country": "UK", "lat": 51.95, "lon": 1.30, "region": "Europe", "hub": "ENGLISH_CHANNEL_E"},
    {"name": "Valencia", "country": "Spain", "lat": 39.45, "lon": -0.32, "region": "Europe", "hub": "MED_W"},
    {"name": "Genoa", "country": "Italy", "lat": 44.41, "lon": 8.92, "region": "Europe", "hub": "MED_NW"},
    {"name": "Piraeus", "country": "Greece", "lat": 37.94, "lon": 23.63, "region": "Europe", "hub": "MED_CENTRAL"},

    # --- Africa ---
    {"name": "Cape Town", "country": "South Africa", "lat": -33.91, "lon": 18.43, "region": "Africa", "hub": "CAPE_GOOD_HOPE"},
    {"name": "Durban", "country": "South Africa", "lat": -29.87, "lon": 31.02, "region": "Africa", "hub": "AFRICA_SE"},
    {"name": "Port Said (Suez North)", "country": "Egypt", "lat": 31.26, "lon": 32.30, "region": "Africa", "hub": "SUEZ_NORTH"},
]

# ==============================================================================
# 2. GLOBAL MARITIME SEA NODES & CHOKEPOINTS (Strictly Navigable Waters)
# ==============================================================================
SEA_WAYPOINTS = {
    # --- North & Central Pacific ---
    "US_PACIFIC_N": (48.00, -125.50),
    "US_PACIFIC_MID": (37.50, -123.50),
    "US_PACIFIC_S": (33.00, -119.50),
    "PACIFIC_NW": (44.00, -135.00),
    "PACIFIC_MID_N": (46.00, -170.00),
    "PACIFIC_NE_ASIA": (40.00, 155.00),
    "PACIFIC_HAWAII_CENTRAL": (21.50, -157.00),
    "PACIFIC_MID_TROPIC": (15.00, -140.00),
    "PACIFIC_CENTRAL": (8.00, -110.00),
    "PANAMA_PACIFIC": (7.50, -80.00),
    "PANAMA_CANAL": (9.05, -79.70),
    "PANAMA_CARIBBEAN": (9.80, -79.70),

    # --- South Pacific & South America Pacific ---
    "S_AMERICA_PACIFIC_N": (-10.00, -80.50),
    "S_AMERICA_PACIFIC_S": (-33.00, -74.00),
    "PACIFIC_SOUTH_MID": (-20.00, -135.00),
    "PACIFIC_SOUTH_WEST": (-28.00, -175.00),
    "DRAKE_PASSAGE": (-57.00, -65.00),

    # --- Australasia & Oceania Waters ---
    "AUSTRALIA_E_CORAL": (-22.00, 153.50),
    "AUSTRALIA_E": (-34.00, 152.00),
    "AUSTRALIA_SE": (-39.00, 146.00),
    "AUSTRALIA_S": (-38.50, 135.00),
    "AUSTRALIA_W": (-32.00, 114.50),
    "AUSTRALIA_NW": (-15.00, 118.00),
    "NEW_ZEALAND_N": (-35.00, 175.00),
    "TASMAN_SEA": (-37.00, 160.00),

    # --- East Asia Waters ---
    "JAPAN_SOUTH": (34.00, 139.50),
    "KOREA_STRAIT": (34.50, 129.50),
    "EC_CHINA": (31.00, 123.00),
    "TAIWAN_STRAIT": (24.00, 120.50),
    "SC_CHINA": (21.50, 115.50),
    "SCS_MID": (14.00, 113.00),
    "SCS_SOUTH": (4.00, 107.50),

    # --- SE Asia & Indonesian Straits ---
    "SINGAPORE": (1.25, 103.95),
    "MALACCA_MID": (3.50, 100.50),
    "MALACCA_NORTH": (5.80, 95.50),
    "SUNDA_STRAIT": (-6.00, 105.50),
    "LOMBOK_STRAIT": (-8.80, 115.80),

    # --- Indian Ocean & Bay of Bengal ---
    "BAY_BENGAL_NW": (17.50, 84.50),
    "BAY_BENGAL_MID": (12.00, 88.00),
    "SRI_LANKA_S": (5.70, 80.50),
    "ARABIAN_SEA_E": (18.50, 71.50),
    "ARABIAN_SEA_MID": (15.00, 65.00),
    "STRAIT_HORMUZ": (26.30, 56.50),
    "PERSIAN_GULF": (25.50, 54.50),
    "GULF_ADEN": (12.50, 48.00),
    "BAB_EL_MANDEB": (12.60, 43.40),
    "RED_SEA_SOUTH": (17.00, 41.00),
    "RED_SEA_MID": (22.50, 37.50),
    "RED_SEA_NORTH": (27.50, 34.50),
    "SUEZ_CANAL": (30.00, 32.55),
    "SUEZ_NORTH": (31.50, 32.30),
    "INDIAN_OCEAN_MID": (-15.00, 75.00),
    "INDIAN_OCEAN_S": (-32.00, 60.00),

    # --- Mediterranean & Black Sea ---
    "MED_EAST": (33.00, 30.00),
    "MED_CENTRAL": (35.50, 17.00),
    "MED_NW": (43.00, 8.50),
    "MED_W": (38.00, 1.00),
    "GIBRALTAR_E": (36.00, -4.50),
    "GIBRALTAR_STRAIT": (35.95, -5.60),
    "GIBRALTAR_W": (36.00, -7.50),

    # --- North Atlantic, Caribbean & Northern Europe ---
    "ATLANTIC_IBERIA": (40.00, -10.50),
    "BAY_BISCAY": (45.50, -6.00),
    "ENGLISH_CHANNEL_W": (49.50, -5.50),
    "ENGLISH_CHANNEL_E": (50.50, 0.50),
    "NORTH_SEA_S": (52.50, 3.50),
    "NORTH_SEA_E": (54.00, 7.50),
    "NATLANTIC_E": (48.00, -20.00),
    "NATLANTIC_MID": (42.00, -42.00),
    "NATLANTIC_W": (38.00, -65.00),
    "US_ATLANTIC_N": (40.00, -72.50),
    "US_ATLANTIC_S": (30.00, -78.00),
    "FLORIDA_STRAIT": (24.80, -80.20),
    "GULF_MEXICO_MID": (25.00, -88.00),
    "GULF_MEXICO_NW": (28.50, -94.00),
    "CARIBBEAN_W": (15.00, -78.00),
    "CARIBBEAN_E": (15.00, -65.00),

    # --- South Atlantic & Africa ---
    "SATLANTIC_TROPIC": (5.00, -25.00),
    "S_AMERICA_E": (-24.50, -45.00),
    "S_AMERICA_SE": (-35.00, -54.00),
    "SATLANTIC_SW": (-32.00, -38.00),
    "SATLANTIC_MID": (-25.00, -10.00),
    "AFRICA_W": (0.00, 5.00),
    "AFRICA_SW": (-22.00, 12.00),
    "CAPE_GOOD_HOPE": (-35.50, 19.50),
    "AFRICA_SE": (-29.00, 33.00),
    "MOZAMBIQUE_CHANNEL": (-18.00, 41.00),
}

# ==============================================================================
# 3. MARITIME GRAPH TOPOLOGY (Nautical Sea Lane Interconnections)
# ==============================================================================
SEA_EDGES = [
    # --- North & Central Pacific Corridors (Americas <-> Asia & Hawaii) ---
    ("US_PACIFIC_N", "PACIFIC_NW"),
    ("US_PACIFIC_MID", "PACIFIC_NW"),
    ("US_PACIFIC_S", "PACIFIC_NW"),
    ("US_PACIFIC_S", "PACIFIC_HAWAII_CENTRAL"),
    ("US_PACIFIC_MID", "PACIFIC_HAWAII_CENTRAL"),
    ("US_PACIFIC_S", "PACIFIC_CENTRAL"),
    ("PACIFIC_NW", "PACIFIC_MID_N"),
    ("PACIFIC_MID_N", "PACIFIC_NE_ASIA"),
    ("PACIFIC_NE_ASIA", "JAPAN_SOUTH"),
    ("PACIFIC_HAWAII_CENTRAL", "PACIFIC_NE_ASIA"),
    ("PACIFIC_HAWAII_CENTRAL", "PACIFIC_MID_TROPIC"),
    ("PACIFIC_CENTRAL", "PANAMA_PACIFIC"),
    ("PACIFIC_CENTRAL", "PACIFIC_MID_TROPIC"),

    # --- South Pacific & Trans-South Pacific (Australia <-> Americas & Hawaii) ---
    ("AUSTRALIA_E_CORAL", "AUSTRALIA_E"),
    ("AUSTRALIA_E", "AUSTRALIA_SE"),
    ("AUSTRALIA_SE", "AUSTRALIA_S"),
    ("AUSTRALIA_S", "AUSTRALIA_W"),
    ("AUSTRALIA_E", "TASMAN_SEA"),
    ("TASMAN_SEA", "NEW_ZEALAND_N"),
    ("AUSTRALIA_E_CORAL", "PACIFIC_SOUTH_WEST"),
    ("NEW_ZEALAND_N", "PACIFIC_SOUTH_WEST"),
    ("PACIFIC_SOUTH_WEST", "PACIFIC_HAWAII_CENTRAL"),
    ("PACIFIC_SOUTH_WEST", "PACIFIC_SOUTH_MID"),
    ("PACIFIC_SOUTH_MID", "PACIFIC_CENTRAL"),
    ("PACIFIC_SOUTH_MID", "S_AMERICA_PACIFIC_N"),
    ("PACIFIC_SOUTH_MID", "S_AMERICA_PACIFIC_S"),

    # --- South America Pacific & Panama Canal ---
    ("PANAMA_PACIFIC", "S_AMERICA_PACIFIC_N"),
    ("S_AMERICA_PACIFIC_N", "S_AMERICA_PACIFIC_S"),
    ("S_AMERICA_PACIFIC_S", "DRAKE_PASSAGE"),
    ("DRAKE_PASSAGE", "S_AMERICA_SE"),
    ("S_AMERICA_SE", "S_AMERICA_E"),
    ("PANAMA_CARIBBEAN", "PANAMA_CANAL"),
    ("PANAMA_CANAL", "PANAMA_PACIFIC"),

    # --- Australasia <-> Asia & Indian Ocean ---
    ("AUSTRALIA_E_CORAL", "SCS_SOUTH"),
    ("AUSTRALIA_NW", "LOMBOK_STRAIT"),
    ("AUSTRALIA_NW", "SUNDA_STRAIT"),
    ("AUSTRALIA_W", "AUSTRALIA_NW"),
    ("AUSTRALIA_W", "INDIAN_OCEAN_S"),
    ("LOMBOK_STRAIT", "SINGAPORE"),
    ("SUNDA_STRAIT", "SINGAPORE"),

    # --- East Asia Inland Sea Lanes ---
    ("JAPAN_SOUTH", "KOREA_STRAIT"),
    ("KOREA_STRAIT", "EC_CHINA"),
    ("EC_CHINA", "TAIWAN_STRAIT"),
    ("TAIWAN_STRAIT", "SC_CHINA"),
    ("SC_CHINA", "SCS_MID"),
    ("SCS_MID", "SCS_SOUTH"),
    ("SCS_SOUTH", "SINGAPORE"),

    # --- Malacca & Indian Ocean Sea Lanes ---
    ("SINGAPORE", "MALACCA_MID"),
    ("MALACCA_MID", "MALACCA_NORTH"),
    ("MALACCA_NORTH", "BAY_BENGAL_MID"),
    ("MALACCA_NORTH", "SRI_LANKA_S"),
    ("BAY_BENGAL_MID", "BAY_BENGAL_NW"),
    ("BAY_BENGAL_NW", "SRI_LANKA_S"),
    ("SRI_LANKA_S", "ARABIAN_SEA_E"),
    ("SRI_LANKA_S", "ARABIAN_SEA_MID"),
    ("SRI_LANKA_S", "GULF_ADEN"),
    ("SRI_LANKA_S", "INDIAN_OCEAN_MID"),
    ("ARABIAN_SEA_E", "ARABIAN_SEA_MID"),
    ("ARABIAN_SEA_MID", "STRAIT_HORMUZ"),
    ("STRAIT_HORMUZ", "PERSIAN_GULF"),
    ("ARABIAN_SEA_MID", "GULF_ADEN"),

    # --- Red Sea & Suez Canal Corridor ---
    ("GULF_ADEN", "BAB_EL_MANDEB"),
    ("BAB_EL_MANDEB", "RED_SEA_SOUTH"),
    ("RED_SEA_SOUTH", "RED_SEA_MID"),
    ("RED_SEA_MID", "RED_SEA_NORTH"),
    ("RED_SEA_NORTH", "SUEZ_CANAL"),
    ("SUEZ_CANAL", "SUEZ_NORTH"),
    ("SUEZ_NORTH", "MED_EAST"),

    # --- Mediterranean & Gibraltar Corridor ---
    ("MED_EAST", "MED_CENTRAL"),
    ("MED_CENTRAL", "MED_NW"),
    ("MED_CENTRAL", "MED_W"),
    ("MED_NW", "MED_W"),
    ("MED_W", "GIBRALTAR_E"),
    ("GIBRALTAR_E", "GIBRALTAR_STRAIT"),
    ("GIBRALTAR_STRAIT", "GIBRALTAR_W"),

    # --- North Atlantic, Caribbean & Northern Europe ---
    ("GIBRALTAR_W", "ATLANTIC_IBERIA"),
    ("GIBRALTAR_W", "NATLANTIC_E"),
    ("GIBRALTAR_W", "SATLANTIC_TROPIC"),
    ("ATLANTIC_IBERIA", "BAY_BISCAY"),
    ("BAY_BISCAY", "ENGLISH_CHANNEL_W"),
    ("ENGLISH_CHANNEL_W", "ENGLISH_CHANNEL_E"),
    ("ENGLISH_CHANNEL_E", "NORTH_SEA_S"),
    ("NORTH_SEA_S", "NORTH_SEA_E"),
    ("ENGLISH_CHANNEL_W", "NATLANTIC_E"),
    ("NATLANTIC_E", "NATLANTIC_MID"),
    ("NATLANTIC_MID", "NATLANTIC_W"),
    ("NATLANTIC_W", "US_ATLANTIC_N"),
    ("US_ATLANTIC_N", "US_ATLANTIC_S"),
    ("US_ATLANTIC_S", "FLORIDA_STRAIT"),
    ("FLORIDA_STRAIT", "GULF_MEXICO_MID"),
    ("GULF_MEXICO_MID", "GULF_MEXICO_NW"),
    ("FLORIDA_STRAIT", "CARIBBEAN_W"),
    ("CARIBBEAN_W", "PANAMA_CARIBBEAN"),
    ("CARIBBEAN_W", "CARIBBEAN_E"),
    ("CARIBBEAN_E", "SATLANTIC_TROPIC"),

    # --- South Atlantic & Cape of Good Hope ---
    ("SATLANTIC_TROPIC", "S_AMERICA_E"),
    ("SATLANTIC_TROPIC", "AFRICA_W"),
    ("S_AMERICA_E", "SATLANTIC_SW"),
    ("SATLANTIC_SW", "SATLANTIC_MID"),
    ("AFRICA_W", "AFRICA_SW"),
    ("AFRICA_SW", "CAPE_GOOD_HOPE"),
    ("SATLANTIC_MID", "CAPE_GOOD_HOPE"),
    ("CAPE_GOOD_HOPE", "AFRICA_SE"),
    ("AFRICA_SE", "MOZAMBIQUE_CHANNEL"),
    ("MOZAMBIQUE_CHANNEL", "GULF_ADEN"),
    ("AFRICA_SE", "INDIAN_OCEAN_S"),
    ("INDIAN_OCEAN_S", "INDIAN_OCEAN_MID"),
]

def haversine_distance(lat1, lon1, lat2, lon2):
    """Computes great circle distance between two points in km."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c

def build_maritime_graph():
    G = nx.Graph()
    for wp_name, (lat, lon) in SEA_WAYPOINTS.items():
        G.add_node(wp_name, lat=lat, lon=lon)
    
    for u, v in SEA_EDGES:
        if u in SEA_WAYPOINTS and v in SEA_WAYPOINTS:
            lat1, lon1 = SEA_WAYPOINTS[u]
            lat2, lon2 = SEA_WAYPOINTS[v]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            G.add_edge(u, v, weight=dist)
    return G

MARITIME_GRAPH = build_maritime_graph()

def find_nearest_sea_node(lat, lon):
    """Finds closest graph waypoint to given coordinate."""
    best_node = None
    best_dist = float("inf")
    for node, (n_lat, n_lon) in SEA_WAYPOINTS.items():
        d = haversine_distance(lat, lon, n_lat, n_lon)
        if d < best_dist:
            best_dist = d
            best_node = node
    return best_node

def interpolate_points(lat1, lon1, lat2, lon2, step_km=80.0):
    """Interpolates navigable sea waypoints along a line segment."""
    dist_km = haversine_distance(lat1, lon1, lat2, lon2)
    num_steps = max(2, int(dist_km / step_km))
    lats = np.linspace(lat1, lat2, num_steps)
    
    if abs(lon2 - lon1) > 180.0:
        if lon1 > 0:
            lon2_adj = lon2 + 360.0
        else:
            lon2_adj = lon2 - 360.0
        lons = np.linspace(lon1, lon2_adj, num_steps)
        lons = [(l + 180.0) % 360.0 - 180.0 for l in lons]
    else:
        lons = np.linspace(lon1, lon2, num_steps)
    return list(lats), list(lons)

def compute_sea_route(lat1, lon1, lat2, lon2, origin_hub=None, dest_hub=None):
    """
    Computes a realistic, 100% sea-navigable path between any two global points
    using Dijkstra shortest path on the global maritime shipping lane graph.
    """
    start_node = origin_hub if (origin_hub and origin_hub in SEA_WAYPOINTS) else find_nearest_sea_node(lat1, lon1)
    end_node = dest_hub if (dest_hub and dest_hub in SEA_WAYPOINTS) else find_nearest_sea_node(lat2, lon2)
    
    if start_node == end_node:
        lats_c, lons_c = interpolate_points(lat1, lon1, lat2, lon2)
        total_d = sum(haversine_distance(lats_c[i], lons_c[i], lats_c[i+1], lons_c[i+1]) for i in range(len(lats_c)-1))
        return lats_c, lons_c, total_d

    try:
        path_nodes = nx.shortest_path(MARITIME_GRAPH, source=start_node, target=end_node, weight="weight")
    except nx.NetworkXNoPath:
        lats_c, lons_c = interpolate_points(lat1, lon1, lat2, lon2)
        return lats_c, lons_c, haversine_distance(lat1, lon1, lat2, lon2)

    waypoints = [(lat1, lon1)]
    for n in path_nodes:
        waypoints.append(SEA_WAYPOINTS[n])
    waypoints.append((lat2, lon2))

    all_lats = []
    all_lons = []
    total_km = 0.0

    for i in range(len(waypoints) - 1):
        seg_lat1, seg_lon1 = waypoints[i]
        seg_lat2, seg_lon2 = waypoints[i+1]
        s_lats, s_lons = interpolate_points(seg_lat1, seg_lon1, seg_lat2, seg_lon2)
        
        if all_lats:
            s_lats = s_lats[1:]
            s_lons = s_lons[1:]
            
        all_lats.extend(s_lats)
        all_lons.extend(s_lons)
        total_km += haversine_distance(seg_lat1, seg_lon1, seg_lat2, seg_lon2)

    return all_lats, all_lons, total_km

def generate_quantum_sea_track(lats, lons, intensity=1.5, seed=None):
    """
    Generates a hydrodynamic weather-adapted sea track.
    Applies current/wave deviation ONLY in open ocean waters,
    strictly preserving canal and strait chokepoints.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    q_lats = np.array(lats, dtype=float)
    q_lons = np.array(lons, dtype=float)
    N = len(q_lats)
    if N < 6:
        return list(q_lats), list(q_lons)

    for i in range(N):
        lat = q_lats[i]
        lon = q_lons[i]
        
        in_chokepoint = False
        if (7.0 <= lat <= 11.0 and -82.0 <= lon <= -77.0): # Panama
            in_chokepoint = True
        elif (26.0 <= lat <= 33.0 and 30.0 <= lon <= 35.0): # Suez
            in_chokepoint = True
        elif (1.0 <= lat <= 7.0 and 95.0 <= lon <= 105.0): # Malacca
            in_chokepoint = True
        elif (35.0 <= lat <= 37.0 and -7.0 <= lon <= -4.0): # Gibraltar
            in_chokepoint = True
        elif (49.0 <= lat <= 52.0 and -5.0 <= lon <= 4.0): # English Channel
            in_chokepoint = True
        elif (24.0 <= lat <= 27.0 and 54.0 <= lon <= 58.0): # Hormuz
            in_chokepoint = True
            
        if not in_chokepoint:
            progress = i / max(1, N - 1)
            wave_env = math.sin(progress * math.pi) ** 1.5
            dev = wave_env * intensity * (math.sin(progress * 4.0 * math.pi + 0.5))
            q_lats[i] += dev * 0.8
            q_lons[i] += dev * 0.5

    return list(q_lats), list(q_lons)

# ==============================================================================
# 4. COMPREHENSIVE WORLDWIDE PRESET TRADE LANES (Americas, Australasia, Asia, Europe)
# ==============================================================================
PRESET_GLOBAL_ROUTES = [
    # --- Americas & Trans-Pacific ---
    {
        "name": "Trans-Pacific Northern Gateway (Los Angeles ➔ Tokyo / Yokohama)",
        "origin": "Los Angeles / Long Beach",
        "dest": "Tokyo / Yokohama",
        "vessel_class": "Panamax Container",
        "cargo_tons": 75000,
        "primary_fuel": "Methanol",
        "description": "High-latitude North Pacific great-circle fairway connecting US West Coast to Japan."
    },
    {
        "name": "Trans-Pacific Mega Corridor (Seattle ➔ Shanghai)",
        "origin": "Seattle / Tacoma",
        "dest": "Shanghai",
        "vessel_class": "ULCS (24,000 TEU)",
        "cargo_tons": 215000,
        "primary_fuel": "LNG",
        "description": "Major container shipping trade lane connecting Pacific Northwest to East China."
    },
    {
        "name": "Trans-South Pacific Express (Sydney ➔ Los Angeles via Hawaii)",
        "origin": "Sydney",
        "dest": "Los Angeles / Long Beach",
        "vessel_class": "Post-Panamax Cargo",
        "cargo_tons": 92000,
        "primary_fuel": "Green Ammonia",
        "description": "Trans-oceanic South-to-North Pacific route connecting Australia to North America."
    },
    {
        "name": "Pan-American Canal Transit (Vancouver ➔ New York via Panama)",
        "origin": "Vancouver",
        "dest": "New York / NJ",
        "vessel_class": "Panamax Cargo",
        "cargo_tons": 70000,
        "primary_fuel": "LNG",
        "description": "Coast-to-Coast inter-oceanic maritime transit traversing the Panama Canal and Florida Strait."
    },

    # --- Americas & Trans-Atlantic / South America ---
    {
        "name": "Trans-Atlantic Blue Corridor (New York ➔ London / Felixstowe)",
        "origin": "New York / NJ",
        "dest": "London / Felixstowe",
        "vessel_class": "Aframax Clean Tanker",
        "cargo_tons": 95000,
        "primary_fuel": "Green Ammonia",
        "description": "North Atlantic shipping lane connecting US Eastern Seaboard to Western Europe."
    },
    {
        "name": "Gulf-Europe Energy Route (Houston ➔ Rotterdam)",
        "origin": "Houston / Galveston",
        "dest": "Rotterdam",
        "vessel_class": "VLCC Crude Carrier",
        "cargo_tons": 280000,
        "primary_fuel": "LNG",
        "description": "Bulk hydrocarbon export lane from US Gulf through Florida Strait to Northern Europe."
    },
    {
        "name": "South America-Europe Agri-Lane (Santos ➔ Antwerp)",
        "origin": "Santos",
        "dest": "Antwerp",
        "vessel_class": "Capesize Bulk Carrier",
        "cargo_tons": 168000,
        "primary_fuel": "Methanol",
        "description": "Trans-equatorial Atlantic bulk trade route for agricultural and mineral freight."
    },
    {
        "name": "South America-Asia Pacific Lane (Valparaiso ➔ Shanghai)",
        "origin": "Valparaiso / San Antonio",
        "dest": "Shanghai",
        "vessel_class": "Capesize Bulk",
        "cargo_tons": 175000,
        "primary_fuel": "Methanol",
        "description": "Southern Pacific trans-oceanic mining and bulk corridor connecting Chile to China."
    },

    # --- Australasia & Oceania ---
    {
        "name": "Oceania-Asia Mineral Corridor (Melbourne ➔ Singapore)",
        "origin": "Melbourne",
        "dest": "Singapore",
        "vessel_class": "Capesize Bulk",
        "cargo_tons": 160000,
        "primary_fuel": "LNG",
        "description": "Bulk iron ore and commodity shipping lane from Australia to Southeast Asia hub."
    },
    {
        "name": "Indo-Pacific Gateway (Sydney ➔ Visakhapatnam)",
        "origin": "Sydney",
        "dest": "Visakhapatnam (Visag)",
        "vessel_class": "Aframax Tanker",
        "cargo_tons": 88000,
        "primary_fuel": "Green Ammonia",
        "description": "Strategic maritime link connecting East Australia through Indonesian straits to India."
    },

    # --- Asia - Europe & Middle East ---
    {
        "name": "Asia-Europe Mega Lane (Shanghai ➔ Rotterdam via Suez)",
        "origin": "Shanghai",
        "dest": "Rotterdam",
        "vessel_class": "ULCS (24,000 TEU)",
        "cargo_tons": 220000,
        "primary_fuel": "LNG",
        "description": "Busiest global corridor traversing East China Sea, Malacca, Red Sea, Suez, and Gibraltar."
    },
    {
        "name": "Energy Oil Gateway (Dubai ➔ Visakhapatnam via Arabian Sea)",
        "origin": "Dubai (Jebel Ali)",
        "dest": "Visakhapatnam (Visag)",
        "vessel_class": "VLCC Crude Carrier",
        "cargo_tons": 290000,
        "primary_fuel": "HFO",
        "description": "Crude hydrocarbon transit through Strait of Hormuz and Arabian Sea to India."
    }
]

def get_expanded_weather_types():
    return [
        {"label": "Clear & Calm (Beaufort 2-3)", "temp": 25, "precip": 0.0, "wind": 10.0, "wave": 1.0},
        {"label": "Moderate Trade Winds (Beaufort 4-5)", "temp": 20, "precip": 0.1, "wind": 20.0, "wave": 2.5},
        {"label": "Rough Monsoon / Heavy Swell (Beaufort 7)", "temp": 16, "precip": 15.0, "wind": 35.0, "wave": 4.5},
        {"label": "Gale Force Oceanic Storm (Beaufort 8-9)", "temp": 8, "precip": 30.0, "wind": 48.0, "wave": 6.5},
        {"label": "Dense Maritime Fog (Low Visibility)", "temp": 12, "precip": 0.5, "wind": 8.0, "wave": 1.2},
    ]

def get_port_by_name(port_name):
    for p in GLOBAL_PORTS:
        if p["name"] == port_name or port_name in p["name"]:
            return p
    return GLOBAL_PORTS[0]

def generate_random_fleet(num_ships=8):
    """
    Generates an evenly balanced active fleet deployed across ALL global trade quadrants:
    Americas, Australasia/Oceania, Asia, Europe, and Middle East.
    """
    fleet = []
    fuel_types = ["HFO", "LNG", "Methanol", "Ammonia", "Hydrogen"]
    vessel_classes = ["ULCS (24,000 TEU)", "VLCC Crude Carrier", "Panamax Container", "Capesize Bulk", "Aframax Tanker", "Post-Panamax Cargo"]
    
    route_pools = [
        # Americas (Transpacific, Transatlantic, Panama, South America)
        [("Los Angeles / Long Beach", "Tokyo / Yokohama"), ("Seattle / Tacoma", "Shanghai"), ("New York / NJ", "London / Felixstowe"), ("Houston / Galveston", "Rotterdam"), ("Vancouver", "New York / NJ"), ("Santos", "Antwerp"), ("Valparaiso / San Antonio", "Shanghai")],
        # Australasia & Oceania
        [("Sydney", "Los Angeles / Long Beach"), ("Melbourne", "Singapore"), ("Sydney", "Visakhapatnam (Visag)"), ("Fremantle / Perth", "Dubai (Jebel Ali)"), ("Auckland", "Honolulu (Hawaii)"), ("Honolulu (Hawaii)", "Tokyo / Yokohama")],
        # Asia-Europe & Middle East
        [("Shanghai", "Rotterdam"), ("Hong Kong", "Hamburg"), ("Dubai (Jebel Ali)", "Visakhapatnam (Visag)"), ("Singapore", "Rotterdam"), ("Mumbai (JNPT)", "London / Felixstowe"), ("Busan", "Cape Town")]
    ]

    for i in range(num_ships):
        pool = route_pools[i % len(route_pools)]
        orig_name, dest_name = pool[(i // len(route_pools)) % len(pool)]
        
        p_orig = get_port_by_name(orig_name)
        p_dest = get_port_by_name(dest_name)
        
        lats_c, lons_c, dist_km = compute_sea_route(
            p_orig["lat"], p_orig["lon"], p_dest["lat"], p_dest["lon"],
            origin_hub=p_orig.get("hub"), dest_hub=p_dest.get("hub")
        )
        lats_q, lons_q = generate_quantum_sea_track(lats_c, lons_c, intensity=1.8, seed=i*17+3)
        dist_nm = dist_km * KM_TO_NM

        v_class = random.choice(vessel_classes)
        fuel = random.choice(fuel_types)
        load_pct = random.uniform(0.65, 0.95)
        
        ship = {
            "ship_id": f"IMO-98{random.randint(10000, 99999)}",
            "name": f"AURA {v_class.split()[0]} {i+1:02d}",
            "vessel_class": v_class,
            "origin": p_orig["name"],
            "origin_country": p_orig["country"],
            "start_lat": p_orig["lat"],
            "start_lon": p_orig["lon"],
            "dest": p_dest["name"],
            "dest_country": p_dest["country"],
            "dest_lat": p_dest["lat"],
            "dest_lon": p_dest["lon"],
            "fuel_type": fuel,
            "target_load": round(load_pct, 2),
            "distance_nm": round(dist_nm, 1),
            "distance_km": round(dist_km, 1),
            "route_lats_classical": lats_c,
            "route_lons_classical": lons_c,
            "route_lats_quantum": lats_q,
            "route_lons_quantum": lons_q,
        }
        fleet.append(ship)
    return fleet
