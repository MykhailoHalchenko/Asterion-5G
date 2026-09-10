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


def render_mitsuba_map(
    config: AirportMapConfig | None = None,
    output_path: str | Path = "airport_map_mitsuba.png",
    resolution: int = 512,
) -> Path:
    """Render the airport topology with Mitsuba's path tracer."""

    if resolution < 16:
        raise ValueError("resolution must be at least 16")
    try:
        import mitsuba as mi
    except ImportError as exc:
        raise RuntimeError("Mitsuba is required for map rendering") from exc

    mi.set_variant("llvm_ad_rgb")
    cfg = config or AirportMapConfig()
    hx, hy = cfg.hangar_size
    gx, gy = cfg.gate_size

    def shape(center: tuple[float, float, float],
              size: tuple[float, float, float],
              color: tuple[float, float, float]) -> dict:
        return {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate(center).scale(size),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": color},
            },
        }

    shapes: dict[str, dict] = {
        "hangar_north": shape(
            (cfg.hangar_center[0], cfg.hangar_center[1] + hy / 2, cfg.wall_height / 2),
            (hx, 0.5, cfg.wall_height), (0.35, 0.45, 0.60),
        ),
        "hangar_south": shape(
            (cfg.hangar_center[0], cfg.hangar_center[1] - hy / 2, cfg.wall_height / 2),
            (hx, 0.5, cfg.wall_height), (0.35, 0.45, 0.60),
        ),
        "hangar_east": shape(
            (cfg.hangar_center[0] + hx / 2, cfg.hangar_center[1], cfg.wall_height / 2),
            (0.5, hy, cfg.wall_height), (0.35, 0.45, 0.60),
        ),
        "gate_terminal": shape(
            (cfg.gate_center[0], cfg.gate_center[1], cfg.wall_height / 2),
            (gx, gy, cfg.wall_height), (0.80, 0.55, 0.20),
        ),
        "ground": shape((10.0, 5.0, -0.25), (180.0, 150.0, 0.5), (0.12, 0.15, 0.18)),
    }
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": {
            "type": "perspective",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(20.0, -105.0, 100.0),
                target=(15.0, 8.0, 0.0),
                up=(0.0, 0.0, 1.0),
            ),
            "fov": 55.0,
            "film": {
                "type": "hdrfilm",
                "width": resolution,
                "height": int(resolution * 0.7),
                "pixel_format": "rgb",
            },
            "sampler": {"type": "independent", "sample_count": 16},
        },
        **shapes,
    })
    image = mi.render(scene, spp=16)
    destination = Path(output_path).resolve()
    mi.util.write_bitmap(str(destination), image)
    return destination


def plot_map(
    tracks: Iterable[TrackEstimate] = (),
    config: AirportMapConfig | None = None,
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
    return figure, axis


def plot_signals(
    tracks: Iterable[TrackEstimate],
    figure=None,
    output_path: str | Path | None = None,
):
    """Display and optionally save reconstructed RF-sensing signals."""

    import matplotlib.pyplot as plt

    points = list(tracks)
    if figure is None:
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), num="Asterion-5G signals")
        axes = axes.ravel()
    else:
        axes = list(figure.axes)
        if len(axes) != 4:
            figure.clear()
            axes = figure.subplots(2, 2).ravel()

    for axis in axes:
        axis.clear()
        axis.grid(True, alpha=0.25)

    if points:
        samples = np.arange(len(points))
        axes[0].plot(samples, [point.range_m for point in points], "o-", color="tab:blue")
        axes[1].plot(samples, [point.speed for point in points], "o-", color="tab:orange")
        axes[2].plot([point.x for point in points], [point.y for point in points],
                     "o-", color="tab:red")
        axes[3].plot(samples, np.rad2deg([point.azimuth_rad for point in points]),
                     "o-", color="tab:green")

    axes[0].set(title="ToA / range", xlabel="Sample", ylabel="Range [m]")
    axes[1].set(title="Doppler / speed", xlabel="Sample", ylabel="Speed [m/s]")
    axes[2].set(title="Reconstructed trajectory", xlabel="X [m]", ylabel="Y [m]")
    axes[3].set(title="AoA / azimuth", xlabel="Sample", ylabel="Angle [deg]")
    axes[2].set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    figure.canvas.draw_idle()
    figure.canvas.flush_events()
    if output_path is not None:
        figure.savefig(output_path, format="png", dpi=150)
    return figure