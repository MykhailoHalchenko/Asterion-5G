import geopandas as gpd
import networkx as nx
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point

from asterix_wrapper import AsterixEncoder
from taxiway_tracks import (
    AerodromeMap,
    _projected_route,
    build_pair_profiles,
    sample_route_track,
)


def test_pair_profiles_average_valid_times_and_calculate_virtual_distance():
    movements = pd.DataFrame(
        [
            {"STAND_mvt": "D8", "RUNWAY_mvt": "18", "TAXITIME_SEC_mvt": 900},
            {"STAND_mvt": "D8", "RUNWAY_mvt": "18", "TAXITIME_SEC_mvt": 300},
            {"STAND_mvt": "D8", "RUNWAY_mvt": "18", "TAXITIME_SEC_mvt": -10},
            {"STAND_mvt": "A1", "RUNWAY_mvt": "25", "TAXITIME_SEC_mvt": 500},
        ]
    )

    profiles = build_pair_profiles(movements).set_index(["STAND_mvt", "RUNWAY_mvt"])

    assert profiles.loc[("D8", "18"), "mean_taxi_seconds"] == 600
    assert profiles.loc[("D8", "18"), "virtual_distance_m"] == 2400
    assert len(profiles) == 2


def test_route_sampling_preserves_duration_and_aircraft_profile():
    route = LineString([(0, 0), (100, 0), (100, 100), (200, 100)])
    standard = sample_route_track(route, 1000, 120, 30, "A320")
    heavy = sample_route_track(route, 1000, 120, 30, "B77L")

    assert [point["timestamp"] for point in standard] == [1000, 1030, 1060, 1090, 1120]
    assert standard[-1]["timestamp"] - standard[0]["timestamp"] == 120
    assert standard[-1]["x"] == 200
    assert standard[-1]["y"] == 100
    assert standard[1]["x"] != heavy[1]["x"]
    assert any(point["vx"] or point["vy"] for point in standard[1:])


def test_projected_route_uses_osm_edge_geometry():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    project = lambda coordinate: transformer.transform(*coordinate)
    first = (8.5, 50.0)
    middle = (8.5004, 50.0008)
    junction = (8.501, 50.001)
    runway_node = (8.502, 50.0)

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:32632"
    nodes = {1: first, 2: junction, 3: runway_node}
    for node, coordinate in nodes.items():
        x, y = project(coordinate)
        graph.add_node(node, x=x, y=y)
    graph.add_edge(1, 2, length=100, geometry=LineString([project(first), project(middle), project(junction)]))
    graph.add_edge(
        2,
        3,
        length=100,
        geometry=LineString([project(junction), project((8.5016, 50.0007)), project(runway_node)]),
    )

    stands = gpd.GeoDataFrame({"ref": ["A1"]}, geometry=[Point(first)], crs="EPSG:4326")
    runways = gpd.GeoDataFrame(
        {"ref": ["18"]},
        geometry=[LineString([runway_node, (8.503, 50.0)])],
        crs="EPSG:4326",
    )
    route = _projected_route(AerodromeMap(graph, stands, runways), "A1", "18", "DEP", 250)

    assert len(route.coords) == 5
    assert route.length > 100


def test_encoder_emits_cat_062_with_position_time_and_velocity():
    encoded = bytes.fromhex(
        AsterixEncoder().encode(
            lat=50.0333,
            lon=8.5706,
            vx=3.0,
            vy=4.0,
            timestamp=1_736_284_200,
            track_number=42,
        )
    )

    assert encoded[0] == 62
    assert len(encoded) > 20