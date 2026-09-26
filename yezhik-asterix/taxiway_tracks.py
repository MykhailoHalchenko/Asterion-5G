from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform as transform_geometry
from shapely.strtree import STRtree


@dataclass
class AerodromeMap:
    graph: nx.MultiDiGraph
    stands: Any
    runways: Any

    def __post_init__(self) -> None:
        self.node_ids = list(self.graph.nodes)
        self.node_points = [
            Point(self.graph.nodes[node]["x"], self.graph.nodes[node]["y"])
            for node in self.node_ids
        ]
        self.node_tree = STRtree(self.node_points)

    def nearest_node(self, point: Point) -> Any:
        return self.node_ids[int(self.node_tree.nearest(point))]


def _normalise_label(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def build_pair_profiles(
    movements: pd.DataFrame,
    reference_taxi_speed_mps: float = 4.0,
    max_taxi_seconds: float = 14_400.0,
) -> pd.DataFrame:
    """Calculate mean taxi duration and a comparable virtual distance per pair."""
    required = {"STAND_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt"}
    missing = required.difference(movements.columns)
    if missing:
        raise KeyError(f"Missing columns for taxi profiles: {sorted(missing)}")

    data = movements.copy()
    data["TAXITIME_SEC_mvt"] = pd.to_numeric(data["TAXITIME_SEC_mvt"], errors="coerce")
    data = data[
        data["STAND_mvt"].notna()
        & data["RUNWAY_mvt"].notna()
        & data["TAXITIME_SEC_mvt"].between(1, max_taxi_seconds)
    ]
    profiles = (
        data.groupby(["STAND_mvt", "RUNWAY_mvt"], as_index=False)["TAXITIME_SEC_mvt"]
        .mean()
        .rename(columns={"TAXITIME_SEC_mvt": "mean_taxi_seconds"})
    )
    profiles["virtual_distance_m"] = profiles["mean_taxi_seconds"] * reference_taxi_speed_mps
    return profiles


def load_aerodrome_map(place: str) -> AerodromeMap:
    """Load taxiway edges and OSM stand/runway features for one aerodrome."""
    graph = ox.graph_from_place(
        place,
        custom_filter='["aeroway"="taxiway"]',
        simplify=True,
        retain_all=True,
    )
    graph = ox.project_graph(graph)
    stands = ox.features_from_place(place, tags={"aeroway": "parking_position"})
    runways = ox.features_from_place(place, tags={"aeroway": "runway"})
    if graph.number_of_nodes() == 0 or stands.empty or runways.empty:
        raise ValueError(f"OSM did not return a usable taxiway/stand/runway map for {place!r}.")
    return AerodromeMap(graph=graph, stands=stands, runways=runways)


def _find_feature(features: Any, label: str, feature_type: str) -> Any:
    wanted = _normalise_label(label)
    if not wanted:
        raise ValueError(f"Empty {feature_type} label.")

    candidates = []
    for column in ("ref", "name", "local_ref", "stand", "description"):
        if column not in features.columns:
            continue
        values = features[column].fillna("").astype(str)
        exact = features[
            values.map(
                lambda value: any(
                    _normalise_label(part) == wanted for part in re.split(r"[/;]", value)
                )
            )
        ]
        if not exact.empty:
            candidates.extend(row for _, row in exact.iterrows())
    if not candidates:
        raise ValueError(f"OSM {feature_type} {label!r} was not found by ref/name.")
    return candidates[0]


def _feature_point(geometry: Any, target: Point | None = None) -> Point:
    if geometry.geom_type == "Point":
        return geometry
    if geometry.geom_type == "MultiPoint":
        points = list(geometry.geoms)
        return min(points, key=lambda point: point.distance(target)) if target else points[0]
    if geometry.geom_type in {"LineString", "MultiLineString"}:
        if geometry.geom_type == "MultiLineString":
            lines = list(geometry.geoms)
            line = max(lines, key=lambda item: item.length)
        else:
            line = geometry
        return line.interpolate(line.project(target)) if target else line.interpolate(0.5, normalized=True)
    return geometry.representative_point()


def _projector(graph: nx.MultiDiGraph) -> Transformer:
    return Transformer.from_crs("EPSG:4326", graph.graph["crs"], always_xy=True)


def _projected_route(
    aerodrome: AerodromeMap,
    stand: str,
    runway: str,
    phase: str,
    max_connector_m: float,
) -> LineString:
    stand_feature = _find_feature(aerodrome.stands, stand, "stand")
    runway_feature = _find_feature(aerodrome.runways, runway, "runway")
    stand_point = _feature_point(stand_feature.geometry)

    transformer = _projector(aerodrome.graph)
    stand_xy = Point(transformer.transform(stand_point.x, stand_point.y))
    runway_geometry = transform_geometry(transformer.transform, runway_feature.geometry)
    runway_node = aerodrome.nearest_node(runway_geometry)
    runway_node_point = Point(
        aerodrome.graph.nodes[runway_node]["x"], aerodrome.graph.nodes[runway_node]["y"]
    )
    start_point, end_point = (stand_xy, runway_node_point)
    if str(phase).upper() == "ARR":
        start_point, end_point = runway_node_point, stand_xy

    start_node = aerodrome.nearest_node(start_point)
    end_node = aerodrome.nearest_node(end_point)
    start_node_point = Point(
        aerodrome.graph.nodes[start_node]["x"], aerodrome.graph.nodes[start_node]["y"]
    )
    end_node_point = Point(
        aerodrome.graph.nodes[end_node]["x"], aerodrome.graph.nodes[end_node]["y"]
    )
    if start_point.distance(start_node_point) > max_connector_m:
        raise ValueError(f"Stand is more than {max_connector_m:.0f} m from taxiway graph.")
    runway_distance = runway_geometry.distance(runway_node_point)
    if runway_distance > max_connector_m:
        raise ValueError(f"Runway is more than {max_connector_m:.0f} m from taxiway graph.")

    try:
        nodes = nx.shortest_path(aerodrome.graph, start_node, end_node, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        raise ValueError("No connected taxiway route between stand and runway.") from exc

    points = []
    for first_node, second_node in zip(nodes, nodes[1:]):
        edge_variants = aerodrome.graph.get_edge_data(first_node, second_node)
        edge = min(edge_variants.values(), key=lambda data: data.get("length", math.inf))
        geometry = edge.get("geometry")
        if geometry is None:
            geometry = LineString(
                [
                    (aerodrome.graph.nodes[first_node]["x"], aerodrome.graph.nodes[first_node]["y"]),
                    (aerodrome.graph.nodes[second_node]["x"], aerodrome.graph.nodes[second_node]["y"]),
                ]
            )
        coordinates = list(geometry.coords)
        first_xy = (aerodrome.graph.nodes[first_node]["x"], aerodrome.graph.nodes[first_node]["y"])
        if math.dist(coordinates[0], first_xy) > math.dist(coordinates[-1], first_xy):
            coordinates.reverse()
        points.extend(Point(x, y) for x, y, *_ in coordinates)
    if not points:
        raise ValueError("Stand and runway snap to the same taxiway node.")
    deduplicated = []
    for point in points:
        if not deduplicated or point.distance(deduplicated[-1]) > 0.01:
            deduplicated.append(point)
    if len(deduplicated) < 2:
        raise ValueError("Taxiway route has no measurable length.")
    return LineString(deduplicated)


def _aircraft_time_factor(aircraft_type: Any) -> float:
    code = str(aircraft_type).upper()
    if code.startswith(("B74", "B76", "B77", "B78", "B79", "A33", "A34", "A35", "A38", "A39")):
        return 1.12
    if code.startswith(("E17", "E19", "CRJ", "AT7", "DH8")):
        return 1.04
    return 1.0


def sample_route_track(
    route: LineString,
    start_timestamp: float,
    duration_seconds: float,
    interval_seconds: float,
    aircraft_type: Any = None,
) -> list[dict[str, float]]:
    """Sample an OSM route at regular times; duration remains the observed taxi time."""
    if duration_seconds <= 0 or interval_seconds <= 0:
        raise ValueError("Duration and sampling interval must be positive.")
    if route.length <= 0:
        raise ValueError("Route length must be positive.")

    count = max(1, math.ceil(duration_seconds / interval_seconds))
    distances = [route.length * index / count for index in range(count + 1)]

    # Keep a mild type-dependent speed profile while scaling it to the observed duration.
    factor = _aircraft_time_factor(aircraft_type)
    weights = []
    for index in range(count):
        midpoint = (distances[index] + distances[index + 1]) / 2
        phase = midpoint / route.length
        terminal_slowdown = 1.0 + (0.35 * factor) * max(0.0, 1.0 - abs(phase - 0.5) * 2)
        weights.append((distances[index + 1] - distances[index]) * terminal_slowdown)
    total_weight = sum(weights)

    segment_durations = [duration_seconds * weight / total_weight for weight in weights]
    segment_velocities = []
    for index, segment_seconds in enumerate(segment_durations):
        first = route.interpolate(distances[index])
        second = route.interpolate(distances[index + 1])
        segment_velocities.append(
            ((second.x - first.x) / segment_seconds, (second.y - first.y) / segment_seconds)
        )

    segment_ends = []
    elapsed = 0.0
    for segment_seconds in segment_durations:
        elapsed += segment_seconds
        segment_ends.append(elapsed)

    result = []
    sample_times = [min(index * interval_seconds, duration_seconds) for index in range(count)]
    sample_times.append(duration_seconds)
    segment_index = 0
    for sample_time in sample_times:
        while segment_index < count - 1 and sample_time > segment_ends[segment_index]:
            segment_index += 1
        segment_start_time = segment_ends[segment_index] - segment_durations[segment_index]
        fraction = min(max((sample_time - segment_start_time) / segment_durations[segment_index], 0.0), 1.0)
        distance = distances[segment_index] + (
            distances[segment_index + 1] - distances[segment_index]
        ) * fraction
        point = route.interpolate(distance)
        vx, vy = segment_velocities[segment_index]
        result.append(
            {
                "timestamp": start_timestamp + sample_time,
                "x": point.x,
                "y": point.y,
                "vx": vx,
                "vy": vy,
            }
        )
    return result


def generate_ground_tracks(
    input_parquet_path: str,
    airport_icao: str,
    airport_place: str,
    output_csv_path: str,
    interval_seconds: float = 30.0,
    max_taxi_seconds: float = 14_400.0,
    max_connector_m: float = 250.0,
    aerodrome: AerodromeMap | None = None,
) -> pd.DataFrame:
    """Create OSM-constrained ground tracks for one selected aerodrome."""
    movements = pd.read_parquet(input_parquet_path)
    required = {
        "STAND_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt", "PHASE_mvt",
        "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt",
    }
    missing = required.difference(movements.columns)
    if missing:
        raise KeyError(f"Input Parquet is missing fields: {sorted(missing)}")

    airport = airport_icao.upper()
    local_airport = movements["ADEP_mvt"].where(
        movements["PHASE_mvt"].eq("DEP"), movements["ADES_mvt"]
    )
    movements = movements[local_airport.eq(airport)].copy()
    if movements.empty:
        raise ValueError(f"No movements found for airport {airport}.")

    profiles = build_pair_profiles(movements, max_taxi_seconds=max_taxi_seconds)
    duration_by_pair = {
        (str(row.STAND_mvt), str(row.RUNWAY_mvt)): float(row.mean_taxi_seconds)
        for row in profiles.itertuples(index=False)
    }
    aerodrome = aerodrome or load_aerodrome_map(airport_place)
    transformer = _projector(aerodrome.graph)
    tracks = []
    skipped = 0
    route_cache: dict[tuple[str, str, str], LineString] = {}
    route_failures: dict[tuple[str, str, str], str] = {}

    for _, movement in movements.iterrows():
        pair = (str(movement["STAND_mvt"]), str(movement["RUNWAY_mvt"]))
        raw_duration = pd.to_numeric(movement["TAXITIME_SEC_mvt"], errors="coerce")
        if pd.notna(raw_duration) and 1 <= float(raw_duration) <= max_taxi_seconds:
            duration = float(raw_duration)
        else:
            mean_duration = duration_by_pair.get(pair)
            duration = (
                mean_duration * _aircraft_time_factor(movement["AIRCRAFT_TYPE_mvt"])
                if mean_duration is not None
                else None
            )
        if duration is None:
            skipped += 1
            continue
        route_key = (pair[0], pair[1], str(movement["PHASE_mvt"]).upper())
        if route_key in route_failures:
            skipped += 1
            continue
        if route_key not in route_cache:
            try:
                route_cache[route_key] = _projected_route(
                    aerodrome,
                    pair[0],
                    pair[1],
                    movement["PHASE_mvt"],
                    max_connector_m,
                )
            except (ValueError, KeyError) as exc:
                route_failures[route_key] = str(exc)
                skipped += 1
                print(f"Пропущено {route_key}: {exc}")
                continue
        route = route_cache[route_key]

        anchor = pd.Timestamp(movement["MVT_TIME_UTC_mvt"])
        if anchor.tzinfo is None:
            anchor = anchor.tz_localize("UTC")
        else:
            anchor = anchor.tz_convert("UTC")
        anchor_seconds = anchor.timestamp()
        start_timestamp = anchor_seconds - duration if movement["PHASE_mvt"] == "DEP" else anchor_seconds
        points = sample_route_track(
            route, start_timestamp, duration, interval_seconds, movement["AIRCRAFT_TYPE_mvt"]
        )
        flight_id = movement.get("FLIGHT_ID_mvt", movement.get("MVT_ID_mvt", "unknown"))
        track_number = int(movement.get("MVT_ID_mvt", 0)) % 65_536
        for point in points:
            lon, lat = transformer.transform(point["x"], point["y"], direction="INVERSE")
            tracks.append(
                {
                    "icao24": str(flight_id),
                    "timestamp": point["timestamp"],
                    "lat": lat,
                    "lon": lon,
                    "vx": point["vx"],
                    "vy": point["vy"],
                    "velocity": math.hypot(point["vx"], point["vy"]),
                    "track_number": track_number,
                    "source": "osm_simulated",
                    "stand": pair[0],
                    "runway": pair[1],
                    "phase": movement["PHASE_mvt"],
                    "aircraft_type": movement["AIRCRAFT_TYPE_mvt"],
                    "taxi_duration_seconds": duration,
                    "route_distance_m": route.length,
                }
            )

    result = pd.DataFrame(tracks)
    if not result.empty:
        result = result.sort_values(["timestamp", "icao24"]).reset_index(drop=True)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Створено {len(result)} точок OSM-треків; пропущено рухів: {skipped}.")
    return result