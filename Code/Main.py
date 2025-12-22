from pathlib import Path

import pandas as pd

try:
    from pyomo.environ import (
        Binary,
        ConcreteModel,
        Constraint,
        Objective,
        Set,
        SolverFactory,
        Var,
        minimize,
        value,
    )
except ImportError:  # Optional, but required for optimization.
    ConcreteModel = None

try:
    import folium
except ImportError:  # Optional dependency for map output.
    folium = None


BASE_DIR = Path(__file__).resolve().parent


DESTINATION_MAP = {
    "Alabama": "Montgomery",
    "Alaska": "Juneau",
    "Arizona": "Phoenix",
    "Arkansas": "Little Rock",
    "California": "Sacramento",
    "Colorado": "Denver",
    "Connecticut": "Hartford",
    "Delaware": "Dover",
    "Florida": "Tallahassee",
    "Georgia": "Atlanta",
    "Hawaii": "Honolulu",
    "Idaho": "Boise",
    "Illinois": "Springfield",
    "Indiana": "Indianapolis",
    "Iowa": "Des Moines",
    "Kansas": "Topeka",
    "Kentucky": "Frankfort",
    "Louisiana": "Baton Rouge",
    "Maine": "Augusta",
    "Maryland": "Annapolis",
    "Massachusetts": "Boston",
    "Michigan": "Lansing",
    "Minnesota": "Saint Paul",
    "Mississippi": "Jackson",
    "Missouri": "Jefferson City",
    "Montana": "Helena",
    "Nebraska": "Lincoln",
    "Nevada": "Carson City",
    "New Hampshire": "Concord",
    "New Jersey": "Trenton",
    "New Mexico": "Santa Fe",
    "New York": "Albany",
    "North Carolina": "Raleigh",
    "North Dakota": "Bismarck",
    "Ohio": "Columbus",
    "Oklahoma": "Oklahoma City",
    "Oregon": "Salem",
    "Pennsylvania": "Harrisburg",
    "Rhode Island": "Providence",
    "South Carolina": "Columbia",
    "South Dakota": "Pierre",
    "Tennessee": "Nashville",
    "Texas": "Austin",
    "Utah": "Salt Lake City",
    "Vermont": "Montpelier",
    "Virginia": "Richmond",
    "Washington": "Olympia",
    "West Virginia": "Charleston",
    "Wisconsin": "Madison",
    "Wyoming": "Cheyenne",
}


CITY_COORDINATES = {
    "Chicago": (41.8781, -87.6298),
    "Montgomery": (32.3792, -86.3077),
    "Juneau": (58.3019, -134.4197),
    "Phoenix": (33.4484, -112.0740),
    "Little Rock": (34.7465, -92.2896),
    "Sacramento": (38.5758, -121.4789),
    "Denver": (39.7392, -104.9903),
    "Hartford": (41.7658, -72.6734),
    "Dover": (39.1582, -75.5244),
    "Tallahassee": (30.4383, -84.2807),
    "Atlanta": (33.7490, -84.3880),
    "Honolulu": (21.3070, -157.8584),
    "Boise": (43.6150, -116.2023),
    "Springfield": (39.7983, -89.6543),
    "Indianapolis": (39.7684, -86.1581),
    "Des Moines": (41.5868, -93.6250),
    "Topeka": (39.0489, -95.6780),
    "Frankfort": (38.2009, -84.8733),
    "Baton Rouge": (30.4515, -91.1871),
    "Augusta": (44.3106, -69.7795),
    "Annapolis": (38.9784, -76.4922),
    "Boston": (42.3601, -71.0589),
    "Lansing": (42.7325, -84.5555),
    "Saint Paul": (44.9537, -93.0900),
    "Jackson": (32.2988, -90.1848),
    "Jefferson City": (38.5767, -92.1735),
    "Helena": (46.5884, -112.0245),
    "Lincoln": (40.8136, -96.7026),
    "Carson City": (39.1638, -119.7674),
    "Concord": (43.2081, -71.5376),
    "Trenton": (40.2206, -74.7597),
    "Santa Fe": (35.6870, -105.9378),
    "Albany": (42.6526, -73.7562),
    "Raleigh": (35.7796, -78.6382),
    "Bismarck": (46.8083, -100.7837),
    "Columbus": (39.9612, -82.9988),
    "Oklahoma City": (35.4676, -97.5164),
    "Salem": (44.9429, -123.0351),
    "Harrisburg": (40.2732, -76.8867),
    "Providence": (41.8240, -71.4128),
    "Columbia": (34.0007, -81.0348),
    "Pierre": (44.3683, -100.3510),
    "Nashville": (36.1627, -86.7816),
    "Austin": (30.2672, -97.7431),
    "Salt Lake City": (40.7608, -111.8910),
    "Montpelier": (44.2601, -72.5754),
    "Richmond": (37.5407, -77.4360),
    "Olympia": (47.0379, -122.9007),
    "Charleston": (38.3498, -81.6326),
    "Madison": (43.0731, -89.4012),
    "Cheyenne": (41.1400, -104.8202),
}


def load_data():
    data = pd.read_csv(BASE_DIR / "Data.csv")
    data_salesman = pd.read_csv(BASE_DIR / "Data_salesman.csv")
    data_train = pd.read_csv(BASE_DIR / "Data_train.csv")
    data_flight = pd.read_csv(BASE_DIR / "Data_flight.csv")
    data_bus = pd.read_csv(BASE_DIR / "Data_bus.csv")
    return data, data_salesman, data_train, data_flight, data_bus


def build_transport_graph(data_bus, data_train, data_flight):
    edges = []
    edge_data = {}
    edge_id = 0
    nodes = set()
    restricted_cities = {"Honolulu", "Juneau"}

    for mode, df in (("Bus", data_bus), ("Train", data_train), ("Flight", data_flight)):
        for _, row in df.iterrows():
            origin = row["From"]
            destination = row["To"]
            price = float(row["Price (USD)"])
            duration = float(row["Total Duration (mins)"])
            stops_raw = row.get("Stops", "")
            stops = []
            if isinstance(stops_raw, str) and stops_raw.strip():
                stops = [stop.strip() for stop in stops_raw.split(",") if stop.strip()]
            if mode in {"Bus", "Train"}:
                if destination in restricted_cities or origin in restricted_cities:
                    continue
                if any(stop in restricted_cities for stop in stops):
                    continue
            path = [origin] + stops + [destination]
            if len(path) < 2:
                continue
            legs = len(path) - 1
            cost_per_leg = price / legs
            duration_per_leg = duration / legs
            for i in range(legs):
                u = path[i]
                v = path[i + 1]
                nodes.update([u, v])
                idx = (u, v, mode, edge_id)
                edge_id += 1
                edges.append(idx)
                edge_data[idx] = {
                    "cost": cost_per_leg,
                    "duration": duration_per_leg,
                    "mode": mode,
                }

    # Add a synthetic Phoenix -> Honolulu flight leg using best known Honolulu flight.
    honolulu_flights = data_flight[data_flight["To"] == "Honolulu"]
    if not honolulu_flights.empty:
        best = honolulu_flights.loc[
            honolulu_flights["Price (USD)"].idxmin()
        ]
        synthetic_cost = float(best["Price (USD)"])
        synthetic_duration = float(best["Total Duration (mins)"])
        if ("Phoenix", "Honolulu") not in {(e[0], e[1]) for e in edges}:
            nodes.update(["Phoenix", "Honolulu"])
            idx = ("Phoenix", "Honolulu", "Flight", edge_id)
            edge_id += 1
            edges.append(idx)
            edge_data[idx] = {
                "cost": synthetic_cost,
                "duration": synthetic_duration,
                "mode": "Flight",
            }

    return nodes, edges, edge_data


def allowed_modes_for_destination(destination):
    if destination in {"Honolulu", "Juneau"}:
        return {"Flight"}
    return {"Bus", "Train", "Flight"}


def nearest_city(destination, candidates, excluded=None):
    destination_coords = CITY_COORDINATES.get(destination)
    if not destination_coords:
        return None
    closest = None
    best_distance = float("inf")
    excluded = excluded or set()
    for city in candidates:
        if city in excluded:
            continue
        city_coords = CITY_COORDINATES.get(city)
        if not city_coords:
            continue
        distance = (destination_coords[0] - city_coords[0]) ** 2 + (
            destination_coords[1] - city_coords[1]
        ) ** 2
        if distance < best_distance:
            best_distance = distance
            closest = city
    return closest


def reachable_cities(origin, edges):
    adjacency = {}
    for u, v, _, _ in edges:
        adjacency.setdefault(u, set()).add(v)
    visited = {origin}
    queue = [origin]
    while queue:
        current = queue.pop(0)
        for nxt in adjacency.get(current, set()):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return visited


def solve_route_pyomo(origin, destination, allowed_modes, nodes, edges, edge_data, max_stops):
    if ConcreteModel is None:
        print("Pyomo not installed. Install it to use optimization.")
        return None

    filtered_edges = [e for e in edges if e[2] in allowed_modes]
    if not filtered_edges:
        return None

    reachable = reachable_cities(origin, filtered_edges)
    excluded = set()
    if destination not in {"Honolulu", "Juneau"}:
        excluded = {"Honolulu", "Juneau"}
    actual_destination = (
        destination if destination in reachable else nearest_city(destination, reachable, excluded)
    )
    if not actual_destination:
        return None

    model = ConcreteModel()
    model.EDGES = Set(initialize=filtered_edges, dimen=4)
    model.NODES = Set(initialize=sorted(nodes))
    model.x = Var(model.EDGES, domain=Binary)

    def flow_rule(model, node):
        incoming = sum(model.x[e] for e in model.EDGES if e[1] == node)
        outgoing = sum(model.x[e] for e in model.EDGES if e[0] == node)
        if node == origin:
            return outgoing - incoming == 1
        if node == actual_destination:
            return incoming - outgoing == 1
        return incoming - outgoing == 0

    model.flow = Constraint(model.NODES, rule=flow_rule)

    max_legs = max_stops + 1
    model.max_legs = Constraint(expr=sum(model.x[e] for e in model.EDGES) <= max_legs)

    def objective_rule(model):
        return sum(
            (edge_data[e]["cost"] + edge_data[e]["duration"]) * model.x[e]
            for e in model.EDGES
        )

    model.objective = Objective(rule=objective_rule, sense=minimize)

    solver = SolverFactory("glpk")
    if not solver.available():
        solver = SolverFactory("cbc")
    if not solver.available():
        print("No supported solver found (glpk or cbc). Install one to run Pyomo.")
        return None

    solver.solve(model, tee=False)

    selected_edges = [e for e in model.EDGES if value(model.x[e]) > 0.5]
    if not selected_edges:
        return None

    next_edge = {}
    for e in selected_edges:
        next_edge[e[0]] = e

    path = [origin]
    modes = []
    leg_durations = []
    current = origin
    while current in next_edge and len(path) <= max_legs + 1:
        e = next_edge[current]
        current = e[1]
        path.append(current)
        modes.append(edge_data[e]["mode"])
        leg_durations.append(edge_data[e]["duration"])
        if current == actual_destination:
            break

    total_cost = sum(edge_data[e]["cost"] for e in selected_edges)
    total_duration = sum(edge_data[e]["duration"] for e in selected_edges)

    return {
        "actual_destination": actual_destination,
        "path": path,
        "modes": modes,
        "leg_durations": leg_durations,
        "cost": total_cost,
        "duration": total_duration,
    }


def build_assignments_pyomo_routes(data, data_salesman, nodes, edges, edge_data, max_stops):
    assignments = []
    routes = []
    salary_map = {
        (row["Company"], row["SalesmanID"]): float(row["Salary (USD)"])
        for _, row in data_salesman.iterrows()
    }
    route_cache = {}

    for company in sorted(data["Company"].unique()):
        company_salesmen = (
            data_salesman[data_salesman["Company"] == company]
            .sort_values("Efficiency (%)", ascending=False)
            .reset_index(drop=True)
        )
        if company_salesmen.empty:
            continue

        salesman_ids = company_salesmen["SalesmanID"].tolist()
        salesman_idx = 0

        company_products = sorted(data[data["Company"] == company]["Product"].unique())
        for product in company_products:
            rows = data[(data["Company"] == company) & (data["Product"] == product)]
            if rows.empty:
                continue

            top_row = rows.loc[rows["Revenue"].idxmax()]
            state = top_row["State"]
            destination = DESTINATION_MAP.get(state)
            if not destination:
                continue

            allowed_modes = allowed_modes_for_destination(destination)
            cache_key = (destination, tuple(sorted(allowed_modes)))
            if cache_key not in route_cache:
                route_cache[cache_key] = solve_route_pyomo(
                    "Chicago",
                    destination,
                    allowed_modes,
                    nodes,
                    edges,
                    edge_data,
                    max_stops,
                )
            route = route_cache[cache_key]
            if not route:
                route = {
                    "actual_destination": destination,
                    "path": ["Chicago", destination],
                    "modes": ["Unknown"],
                    "leg_durations": [],
                    "cost": None,
                    "duration": None,
                }

            salesman = salesman_ids[salesman_idx % len(salesman_ids)]
            salesman_idx += 1
            salary = salary_map.get((company, salesman), 0.0)
            revenue = float(top_row["Revenue"])

            assignments.append(
                {
                    "Salesman": salesman,
                    "Company": company,
                    "Destination": route["actual_destination"],
                    "Revenue": revenue,
                    "Salary": salary,
                    "Mode": " -> ".join(route["modes"]) if route["modes"] else "Unknown",
                    "Cost": route["cost"],
                    "Duration": route["duration"],
                    "Departure Time": "",
                    "Arrival Time": "",
                    "Mode Note": "Mode can remain the same; changes are optional.",
                    "Requested Destination": destination,
                    "Allowed Modes": ", ".join(sorted(allowed_modes)),
                    "Cities Visited": " -> ".join(route["path"]),
                    "Leg Durations (mins)": ", ".join(f"{d:.1f}" for d in route["leg_durations"]),
                }
            )
            routes.append(
                {
                    "salesman": salesman,
                    "path": route["path"],
                    "modes": route["modes"],
                }
            )

    return assignments, routes


def visualize_routes(routes, output_path):
    if folium is None:
        print("folium not installed; skipping map output.")
        return

    us_map = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
    mode_colors = {"Bus": "green", "Train": "orange", "Flight": "red", "Unknown": "gray"}

    for route in routes:
        path = route.get("path", [])
        modes = route.get("modes", [])
        if len(path) < 2:
            continue
        coords = [CITY_COORDINATES.get(city) for city in path]
        if any(coord is None for coord in coords):
            continue
        folium.Marker(
            coords[0], popup=f"Start: {path[0]} (Salesman: {route['salesman']})"
        ).add_to(us_map)
        folium.Marker(coords[-1], popup=f"End: {path[-1]}").add_to(us_map)
        for idx in range(len(coords) - 1):
            mode = modes[idx] if idx < len(modes) else "Unknown"
            color = mode_colors.get(mode, "blue")
            folium.PolyLine(
                [coords[idx], coords[idx + 1]],
                color=color,
                weight=3,
                opacity=0.9,
                popup=f"Mode: {mode}",
            ).add_to(us_map)

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 9999;
                background: white; padding: 8px 12px; border: 1px solid #ccc;
                border-radius: 6px; font-size: 12px;">
      <div><span style="color:red;">&#9632;</span> Flight</div>
      <div><span style="color:orange;">&#9632;</span> Train</div>
      <div><span style="color:green;">&#9632;</span> Bus</div>
      <div><span style="color:gray;">&#9632;</span> Unknown</div>
    </div>
    """
    us_map.get_root().html.add_child(folium.Element(legend_html))
    us_map.save(output_path)
    print(f"Map of routes saved as '{output_path}'. Open this file in a web browser to view the map.")


def main():
    data, data_salesman, data_train, data_flight, data_bus = load_data()
    nodes, edges, edge_data = build_transport_graph(data_bus, data_train, data_flight)
    assignments, routes = build_assignments_pyomo_routes(
        data, data_salesman, nodes, edges, edge_data, max_stops=6
    )
    assignments_df = pd.DataFrame(assignments)

    # Sample assignments intentionally omitted from console output.
    total_revenue = sum(item["Revenue"] for item in assignments)
    total_salary = sum(item["Salary"] for item in assignments)
    total_transport = sum(item["Cost"] for item in assignments if item["Cost"] is not None)
    net_gain = total_revenue - total_salary - total_transport
    print(
        "Totals:",
        {
            "Total Revenue": round(total_revenue, 2),
            "Total Salary": round(total_salary, 2),
            "Total Transport Cost": round(total_transport, 2),
            "Net Gain": round(net_gain, 2),
        },
    )

    if not assignments_df.empty:
        assignments_df["Net Gain"] = (
            assignments_df["Revenue"]
            - assignments_df["Salary"]
            - assignments_df["Cost"].fillna(0)
        )
        combined_rows = []
        for company, company_df in assignments_df.groupby("Company", sort=True):
            company_df = company_df.copy()
            combined_rows.append(company_df)
            company_totals = {
                "Salesman": "TOTAL",
                "Company": company,
                "Destination": "",
                "Revenue": round(company_df["Revenue"].sum(), 2),
                "Salary": round(company_df["Salary"].sum(), 2),
                "Mode": "",
                "Cost": round(company_df["Cost"].fillna(0).sum(), 2),
                "Duration": "",
                "Departure Time": "",
                "Arrival Time": "",
                "Mode Note": "",
                "Requested Destination": "",
                "Allowed Modes": "",
                "Cities Visited": "",
                "Leg Durations (mins)": "",
                "Net Gain": round(company_df["Net Gain"].fillna(0).sum(), 2),
            }
            combined_rows.append(pd.DataFrame([company_totals]))
        combined_df = pd.concat(combined_rows, ignore_index=True)
        file_path = BASE_DIR / "all_companies_report.xlsx"
        combined_df.to_excel(file_path, index=False)
        print(f"Saved combined report to '{file_path}'.")

    output_path = str(BASE_DIR / "us_routes.html")
    visualize_routes(routes, output_path)


if __name__ == "__main__":
    main()
