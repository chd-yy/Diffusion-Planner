#!/usr/bin/env python3
"""Export synchronized, high-resolution videos from paired nuPlan simulation logs.

This deliberately renders the saved simulation histories rather than re-running the
planners. Both models of one scene use the same viewport, frame rate, and timeline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.simulation_log import SimulationLog


WIDTH, HEIGHT, FPS = 1920, 1080, 10
MAP_BOX = (32, 100, 1320, 1020)
MODEL_DIRS = {
    "B10": "B10-qualitative-hard",
    "PostTrain": "full306801-qualitative-hard",
}
TOKENS = (
    "33b448fab23a5d03",
    "00de3f6da9205a0c",
    "ebf09047e62a5fb8",
    "820bf3685bc955ef",
    "9e7b8fa4248d55de",
    "d3be3fa904135dd9",
    "63aa10eaed9f53c5",
)
COLORS = {
    "B10": (235, 158, 67),
    "PostTrain": (92, 170, 248),
    "expert": (117, 224, 137),
}


def cvtext(img: np.ndarray, text: str, xy: tuple[int, int], size: float = 0.75,
           color: tuple[int, int, int] = (235, 238, 245), thickness: int = 2) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def points(states) -> np.ndarray:
    return np.asarray([[s.center.x, s.center.y] for s in states], dtype=np.float64)


def load_pair(root: Path, token: str) -> dict[str, SimulationLog]:
    pair = {}
    for name, dirname in MODEL_DIRS.items():
        matches = list((root / dirname / "simulation_log").rglob(f"{token}.msgpack.xz"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {name} log for {token}, found {len(matches)}")
        pair[name] = SimulationLog.load_data(matches[0])
    return pair


def world_to_pixel(xy: np.ndarray, center: np.ndarray, span: np.ndarray,
                   box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    if box is None:
        box = MAP_BOX
    x0, y0, x1, y1 = box
    px = x0 + (xy[:, 0] - center[0]) / span[0] * (x1 - x0) + (x1 - x0) / 2
    py = y0 + (y1 - y0) / 2 - (xy[:, 1] - center[1]) / span[1] * (y1 - y0)
    return np.rint(np.stack((px, py), axis=1)).astype(np.int32)


def viewport(pair: dict[str, SimulationLog], expert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_xy = np.concatenate(
        [points(s.ego_state for s in log.simulation_history.data) for log in pair.values()] + [expert], axis=0
    )
    lo, hi = np.min(all_xy, axis=0), np.max(all_xy, axis=0)
    center = (lo + hi) / 2
    # The minimum field of view preserves readable road/agent context in stopped scenes.
    x0, y0, x1, y1 = MAP_BOX
    aspect = (x1 - x0) / (y1 - y0)
    span_y = max(70.0, float(hi[1] - lo[1]) + 32.0)
    span_x = max(98.0, float(hi[0] - lo[0]) + 32.0, span_y * aspect)
    span_y = span_x / aspect
    return center, np.array([span_x, span_y], dtype=np.float64)


def map_geometries(log: SimulationLog, center: np.ndarray, span: np.ndarray) -> dict:
    layers = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.STOP_LINE,
    ]
    radius = float(max(span) * 0.8)
    raw = log.scenario.map_api.get_proximal_map_objects(Point2D(*center), radius, layers)
    polys: dict[str, list[np.ndarray]] = {}
    lines: dict[str, list[np.ndarray]] = {}
    for layer, objects in raw.items():
        polys[layer.name] = []
        lines[layer.name] = []
        for obj in objects:
            geom = obj.polygon
            if geom.is_empty:
                continue
            coords = np.asarray(geom.exterior.coords, dtype=np.float64)
            polys[layer.name].append(world_to_pixel(coords, center, span))
            if layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
                path = np.asarray([(p.x, p.y) for p in obj.baseline_path.discrete_path], dtype=np.float64)
                if len(path) >= 2:
                    lines[layer.name].append(world_to_pixel(path, center, span))
    return {"polys": polys, "lines": lines}


def draw_poly(img: np.ndarray, polygons: list[np.ndarray], fill: tuple[int, int, int],
              edge: tuple[int, int, int] | None = None) -> None:
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        cv2.fillPoly(img, [polygon], fill, cv2.LINE_AA)
        if edge:
            cv2.polylines(img, [polygon], True, edge, 1, cv2.LINE_AA)


def static_map(geometries: dict, expert: np.ndarray, center: np.ndarray, span: np.ndarray,
               raw: bool = False) -> np.ndarray:
    img = np.full((HEIGHT, WIDTH, 3), (24, 27, 36), dtype=np.uint8)
    x0, y0, x1, y1 = MAP_BOX
    cv2.rectangle(img, (x0, y0), (x1, y1), (34, 39, 49), -1)
    map_img = img.copy()
    p = geometries["polys"]
    for name, fill, edge in [
        ("WALKWAYS", (59, 57, 58), None),
        ("INTERSECTION", (71, 75, 81), (99, 104, 111)),
        ("LANE", (77, 83, 91), (113, 120, 128)),
        ("LANE_CONNECTOR", (75, 81, 88), None),
        ("CROSSWALK", (126, 120, 111), (170, 162, 151)),
        ("STOP_LINE", (159, 168, 178), None),
    ]:
        draw_poly(map_img, p.get(name, []), fill, edge)
    for line in geometries["lines"].get("LANE", []) + geometries["lines"].get("LANE_CONNECTOR", []):
        cv2.polylines(map_img, [line], False, (109, 114, 117), 1, cv2.LINE_AA)
    if not raw and len(expert) >= 2:
        pix = world_to_pixel(expert, center, span)
        for i in range(0, len(pix) - 1, 4):
            cv2.line(map_img, tuple(pix[i]), tuple(pix[min(i + 2, len(pix) - 1)]), COLORS["expert"], 2, cv2.LINE_AA)
    img[y0:y1, x0:x1] = map_img[y0:y1, x0:x1]
    if not raw:
        cv2.rectangle(img, (x0, y0), (x1, y1), (92, 99, 110), 2)
    return img


def draw_actors(img: np.ndarray, sample, center: np.ndarray, span: np.ndarray) -> None:
    if not hasattr(sample.observation, "tracked_objects"):
        return
    for actor in sample.observation.tracked_objects.tracked_objects:
        c = actor.center
        if abs(c.x - center[0]) > span[0] * 0.58 or abs(c.y - center[1]) > span[1] * 0.58:
            continue
        poly = np.asarray(actor.box.geometry.exterior.coords, dtype=np.float64)
        pix = world_to_pixel(poly, center, span)
        kind = actor.tracked_object_type.name
        fill = (89, 181, 224) if kind == "PEDESTRIAN" else (190, 184, 157)
        cv2.fillPoly(img, [pix], fill, cv2.LINE_AA)
        cv2.polylines(img, [pix], True, (30, 35, 42), 1, cv2.LINE_AA)


def mini_paths(img: np.ndarray, pair_xy: dict[str, np.ndarray], expert: np.ndarray,
               frame: int, selected: str) -> None:
    box = (1400, 450, 1878, 850)
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), (35, 41, 51), -1)
    whole = np.concatenate([*pair_xy.values(), expert], axis=0)
    lo, hi = whole.min(axis=0), whole.max(axis=0)
    center = (lo + hi) / 2
    span_y = max(26.0, float(hi[1] - lo[1]) * 1.35)
    span_x = max(30.0, float(hi[0] - lo[0]) * 1.35, span_y * (x1-x0) / (y1-y0))
    span_y = span_x * (y1-y0) / (x1-x0)
    span = np.array((span_x, span_y))
    exp_pix = world_to_pixel(expert, center, span, box)
    cv2.polylines(img, [exp_pix], False, COLORS["expert"], 2, cv2.LINE_AA)
    for name, xy in pair_xy.items():
        pix = world_to_pixel(xy, center, span, box)
        cv2.polylines(img, [pix], False, COLORS[name], 3 if name == selected else 2, cv2.LINE_AA)
        at = pix[min(frame, len(pix)-1)]
        cv2.circle(img, tuple(at), 7 if name == selected else 5, COLORS[name], -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 98, 112), 2)


def render_frame(base: np.ndarray, log: SimulationLog, pair_xy: dict[str, np.ndarray],
                 expert: np.ndarray, model: str, frame: int, token: str,
                 center: np.ndarray, span: np.ndarray, scenario_type: str) -> np.ndarray:
    map_img = base.copy()
    sample = log.simulation_history.data[min(frame, len(log.simulation_history.data)-1)]
    draw_actors(map_img, sample, center, span)
    xy = pair_xy[model]
    cv2.polylines(map_img, [world_to_pixel(xy[:frame+1], center, span)], False, COLORS[model], 4, cv2.LINE_AA)
    planned = points(sample.trajectory.get_sampled_trajectory())
    if len(planned) >= 2:
        planpix = world_to_pixel(planned, center, span)
        for i in range(0, len(planpix)-1, 3):
            cv2.line(map_img, tuple(planpix[i]), tuple(planpix[min(i+2, len(planpix)-1)]), (126, 207, 244), 2, cv2.LINE_AA)
    ego = np.asarray(sample.ego_state.car_footprint.oriented_box.geometry.exterior.coords, dtype=np.float64)
    ego_pix = world_to_pixel(ego, center, span)
    cv2.fillPoly(map_img, [ego_pix], COLORS[model], cv2.LINE_AA)
    cv2.polylines(map_img, [ego_pix], True, (247, 249, 251), 2, cv2.LINE_AA)
    img = base.copy()
    x0, y0, x1, y1 = MAP_BOX
    img[y0:y1, x0:x1] = map_img[y0:y1, x0:x1]

    cvtext(img, "TEST14-HARD  /  PAIRED REPLAY", (37, 52), 0.9)
    cvtext(img, f"{model}  |  {token}", (38, 84), 0.74, COLORS[model])
    cvtext(img, "SCENARIO", (1400, 138), 0.62, (149, 160, 177))
    cvtext(img, scenario_type.replace("_", " ").upper(), (1400, 180), 0.60)
    cvtext(img, "MODEL", (1400, 245), 0.62, (149, 160, 177))
    cvtext(img, model.upper(), (1400, 296), 1.05, COLORS[model])
    cvtext(img, f"TIME   {frame / FPS:05.1f} s", (1400, 357), 0.73)
    cvtext(img, f"FRAME  {frame+1:03d}/{len(log.simulation_history.data):03d}", (1400, 395), 0.73)
    cvtext(img, "COMPLETE EGO PATHS", (1400, 430), 0.62, (149, 160, 177))
    mini_paths(img, pair_xy, expert, frame, model)
    for idx, (label, color) in enumerate([
        ("B10", COLORS["B10"]), ("Post-train", COLORS["PostTrain"]), ("Recorded expert", COLORS["expert"])
    ]):
        y = 899 + idx*38
        cv2.line(img, (1403, y), (1440, y), color, 3, cv2.LINE_AA)
        cvtext(img, label, (1454, y+7), 0.59)
    cvtext(img, "Map: nuPlan   |   Replay: 10 Hz", (1400, 1040), 0.48, (147, 156, 170), 1)
    return img


def render_raw_frame(base: np.ndarray, log: SimulationLog, frame: int,
                     center: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Only the simulated scene: map, tracked actors, and current ego footprint."""
    img = base.copy()
    sample = log.simulation_history.data[frame]
    draw_actors(img, sample, center, span)
    ego = np.asarray(sample.ego_state.car_footprint.oriented_box.geometry.exterior.coords, dtype=np.float64)
    ego_pix = world_to_pixel(ego, center, span)
    cv2.fillPoly(img, [ego_pix], COLORS["B10"], cv2.LINE_AA)
    cv2.polylines(img, [ego_pix], True, (247, 249, 251), 2, cv2.LINE_AA)
    return img


def export_video(log: SimulationLog, base: np.ndarray, pair_xy: dict[str, np.ndarray],
                 expert: np.ndarray, model: str, token: str, center: np.ndarray,
                 span: np.ndarray, scenario_type: str, output: Path, raw: bool = False) -> dict:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    temporary = output.with_name(output.stem + ".partial.mp4")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in range(len(log.simulation_history.data)):
            image = (
                render_raw_frame(base, log, frame, center, span)
                if raw else
                render_frame(base, log, pair_xy, expert, model, frame, token, center, span, scenario_type)
            )
            assert process.stdin is not None
            process.stdin.write(image.tobytes())
        process.stdin.close()
        assert process.stderr is not None
        error = process.stderr.read().decode("utf-8", "replace")
        status = process.wait()
        if status:
            raise RuntimeError(f"ffmpeg failed ({status}): {error}")
        os.replace(temporary, output)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return {"file": str(output), "frames": len(log.simulation_history.data),
            "fps": FPS, "duration_s": len(log.simulation_history.data) / FPS,
            "size_bytes": output.stat().st_size}


def main() -> None:
    global MAP_BOX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="closed_loop_nonreactive_agents directory containing both run names")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token", choices=TOKENS, action="append", help="Only export selected token(s)")
    parser.add_argument("--model", choices=tuple(MODEL_DIRS), action="append", help="Only export model(s)")
    parser.add_argument("--mode", choices=("annotated", "raw"), default="annotated")
    args = parser.parse_args()
    raw = args.mode == "raw"
    if raw:
        MAP_BOX = (0, 0, WIDTH, HEIGHT)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for token in (args.token or TOKENS):
        print(f"Loading pair {token}", flush=True)
        pair = load_pair(args.root, token)
        pair_xy = {
            name: points(s.ego_state for s in log.simulation_history.data)
            for name, log in pair.items()
        }
        expert = points(pair["B10"].scenario.get_expert_ego_trajectory())
        center, span = viewport(pair, expert)
        geometries = map_geometries(pair["B10"], center, span)
        base = static_map(geometries, expert, center, span, raw=raw)
        scenario_type = pair["B10"].scenario.scenario_type
        for model in (args.model or tuple(MODEL_DIRS)):
            suffix = "_raw_1080p.mp4" if raw else "_1080p.mp4"
            path = args.output / f"{token}_{model}{suffix}"
            print(f"Rendering {model}: {path}", flush=True)
            result = export_video(pair[model], base, pair_xy, expert, model, token,
                                  center, span, scenario_type, path, raw=raw)
            print(f"Saved {result['frames']} frames, {result['size_bytes']/1048576:.1f} MiB", flush=True)
            results.append({"token": token, "model": model, **result})
    manifest = args.output / ("video_manifest_raw.json" if raw else "video_manifest.json")
    existing = json.loads(manifest.read_text()) if manifest.exists() else []
    by_key = {(item["token"], item["model"]): item for item in existing + results}
    manifest.write_text(json.dumps(list(by_key.values()), indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
