"""Airport topology model using Sionna RT plus a Matplotlib overview."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .config import GNODEB_TOPOLOGY
    from .signal_postprocessing import TrackEstimate
except ImportError:
    from config import GNODEB_TOPOLOGY
    from signal_postprocessing import TrackEstimate


@dataclass(frozen=True)
class AirportMapConfig:
    hangar_center: tuple[float, float] = (30.0, 0.0)
    hangar_size: tuple[float, float] = (60.0, 40.0)
    gate_center: tuple[float, float] = (-25.0, 35.0)
    gate_size: tuple[float, float] = (30.0, 20.0)
    wall_height: float = 8.0


def _box(mi: object, rt: object, name: str, center: tuple[float, float, float],
         size: tuple[float, float, float], material: object) -> object:
    mesh = mi.load_dict({
        "type": "cube",
        "to_world": mi.ScalarTransform4f.translate(center).scale(size),
    })
    return rt.SceneObject(mi_mesh=mesh, name=name, radio_material=material)


def build_scene(config: AirportMapConfig | None = None) -> object:
    """Build a Sionna RT scene containing hangar and gate obstacles."""

    try:
        import mitsuba as mi
        import sionna.rt as rt
    except ImportError as exc:
        raise RuntimeError("Sionna RT and Mitsuba are required for the map") from exc

    cfg = config or AirportMapConfig()
    scene = rt.Scene()
    concrete = rt.RadioMaterial(
        name="airport_concrete",
        relative_permittivity=5.31,
        conductivity=0.014,
    )
    hx, hy = cfg.hangar_size
    gx, gy = cfg.gate_size
    objects = [
        _box(mi, rt, "hangar_north",
             (cfg.hangar_center[0], cfg.hangar_center[1] + hy / 2, cfg.wall_height / 2),
             (hx, 0.5, cfg.wall_height), concrete),
        _box(mi, rt, "hangar_south",
             (cfg.hangar_center[0], cfg.hangar_center[1] - hy / 2, cfg.wall_height / 2),
             (hx, 0.5, cfg.wall_height), concrete),
        _box(mi, rt, "hangar_east",
             (cfg.hangar_center[0] + hx / 2, cfg.hangar_center[1], cfg.wall_height / 2),
             (0.5, hy, cfg.wall_height), concrete),
        _box(mi, rt, "gate_terminal",
             (cfg.gate_center[0], cfg.gate_center[1], cfg.wall_height / 2),
             (gx, gy, cfg.wall_height), concrete),
    ]
    scene.edit(add=objects)
    return scene


def plot_map(
    tracks: Iterable[TrackEstimate] = (),
    config: AirportMapConfig | None = None,
    output_path: str | Path | None = None,
):
    """Visualize topology, gNodeBs, and reconstructed aircraft coordinates."""

    import matplotlib.pyplot as plt

    cfg = config or AirportMapConfig()
    figure, axis = plt.subplots(figsize=(10, 7))
    hx, hy = cfg.hangar_size
    gx, gy = cfg.gate_size
    axis.add_patch(plt.Rectangle(
        (cfg.hangar_center[0] - hx / 2, cfg.hangar_center[1] - hy / 2),
        hx, hy, facecolor="#cbd5e1", edgecolor="#334155", alpha=0.65, label="Hangar",
    ))
    axis.add_patch(plt.Rectangle(
        (cfg.gate_center[0] - gx / 2, cfg.gate_center[1] - gy / 2),
        gx, gy, facecolor="#fde68a", edgecolor="#92400e", alpha=0.75, label="Gate",
    ))
    for name, (x, y, _) in GNODEB_TOPOLOGY.items():
        axis.scatter(x, y, marker="^", s=130, color="tab:blue", label=name)
    points = list(tracks)
    if points:
        axis.plot([point.x for point in points], [point.y for point in points],
                  "o-", color="tab:red", linewidth=2, label="Aircraft track")
        for index, point in enumerate(points):
            axis.annotate(str(index), (point.x, point.y), xytext=(5, 5),
                          textcoords="offset points")
    axis.set_title("Asterion-5G airport sensing map")
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys())
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=150)
    return figure, axis