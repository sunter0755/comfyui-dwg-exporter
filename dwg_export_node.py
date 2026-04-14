import datetime
import json
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import ezdxf
import numpy as np
from PIL import Image
import torch

try:
    import folder_paths
except Exception:
    folder_paths = None


def _tensor_to_uint8_image(image_tensor) -> np.ndarray:
    """
    ComfyUI IMAGE tensor is expected as [B, H, W, C] float32 in [0, 1].
    We export the first image in the batch.
    """
    array = image_tensor[0].cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).astype(np.uint8)
    return array


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize_filename_part(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch not in r'<>:"/\|?*').strip()
    return cleaned or "comfy_vector"


def _resolve_output_basename(
    output_dir: str,
    output_name: str,
    filename_template: str,
    counter_digits: int,
    sequence_start: int,
    conflict_policy: str,
) -> str:
    safe_name = _sanitize_filename_part(output_name)
    safe_template = (filename_template or "{name}_{counter}").strip()
    if "{name}" not in safe_template:
        safe_template = "{name}_{counter}"

    safe_policy = (conflict_policy or "next_sequence").strip().lower()
    if safe_policy not in {"next_sequence", "overwrite", "timestamp"}:
        safe_policy = "next_sequence"

    # First candidate for overwrite mode.
    if safe_policy == "overwrite":
        return safe_template.format(name=safe_name, counter=str(sequence_start).zfill(counter_digits))

    if safe_policy == "timestamp":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_name}_{timestamp}"

    # next_sequence mode
    index = max(int(sequence_start), 0)
    digits = max(int(counter_digits), 1)

    while True:
        counter = str(index).zfill(digits)
        candidate = safe_template.format(name=safe_name, counter=counter)
        dxf_path = os.path.join(output_dir, f"{candidate}.dxf")
        dwg_path = os.path.join(output_dir, f"{candidate}.dwg")
        png_path = os.path.join(output_dir, f"{candidate}.png")
        if not (os.path.exists(dxf_path) or os.path.exists(dwg_path) or os.path.exists(png_path)):
            return candidate
        index += 1


def _find_odafc_executable(custom_path: str) -> str:
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    env_path = os.environ.get("ODAFC_PATH", "")
    if env_path and os.path.isfile(env_path):
        return env_path

    candidates = [
        r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
        r"C:\Program Files\ODA\ODAFileConverter 25.12.0\ODAFileConverter.exe",
        r"C:\Program Files\ODA\ODAFileConverter 25.6.0\ODAFileConverter.exe",
    ]
    for exe in candidates:
        if os.path.isfile(exe):
            return exe

    return ""


def _cv_contours_from_image(
    image: np.ndarray,
    threshold: int,
    min_area_px: float,
    simplify_px: float,
):
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    simplified: List[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        approx = cv2.approxPolyDP(cnt, simplify_px, True)
        if len(approx) < 2:
            continue
        simplified.append(approx)

    return simplified


def _auto_canny_thresholds(gray: np.ndarray, sigma: float = 0.33) -> Tuple[int, int]:
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if lower >= upper:
        lower = max(0, upper - 1)
    return lower, upper


def _zhang_suen_thinning(binary: np.ndarray, max_iters: int = 100) -> np.ndarray:
    """
    Zhang-Suen thinning (skeletonization) for 0/255 binary images.
    Returns 0/255 skeleton image.
    """
    img = (binary > 0).astype(np.uint8)
    changed = True
    it = 0

    def neighbors(x, y):
        p2 = img[x - 1, y]
        p3 = img[x - 1, y + 1]
        p4 = img[x, y + 1]
        p5 = img[x + 1, y + 1]
        p6 = img[x + 1, y]
        p7 = img[x + 1, y - 1]
        p8 = img[x, y - 1]
        p9 = img[x - 1, y - 1]
        return p2, p3, p4, p5, p6, p7, p8, p9

    def transitions(p2, p3, p4, p5, p6, p7, p8, p9):
        seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
        return sum((seq[i] == 0 and seq[i + 1] == 1) for i in range(8))

    # pad to avoid bounds checks
    img = np.pad(img, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    while changed and it < max_iters:
        changed = False
        it += 1

        to_remove = []
        rows, cols = img.shape
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                p1 = img[x, y]
                if p1 != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(x, y)
                n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if n < 2 or n > 6:
                    continue
                s = transitions(p2, p3, p4, p5, p6, p7, p8, p9)
                if s != 1:
                    continue
                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue
                to_remove.append((x, y))

        if to_remove:
            for x, y in to_remove:
                img[x, y] = 0
            changed = True

        to_remove = []
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                p1 = img[x, y]
                if p1 != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(x, y)
                n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if n < 2 or n > 6:
                    continue
                s = transitions(p2, p3, p4, p5, p6, p7, p8, p9)
                if s != 1:
                    continue
                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue
                to_remove.append((x, y))

        if to_remove:
            for x, y in to_remove:
                img[x, y] = 0
            changed = True

    skel = (img[1:-1, 1:-1] * 255).astype(np.uint8)
    return skel


def _preprocess_edge_mode(
    image_rgb: np.ndarray,
    canny_sigma: float,
    canny_low: int,
    canny_high: int,
    blur_ksize: int,
    morph_ksize: int,
    morph_close_iters: int,
    morph_open_iters: int,
    min_component_area_px: int,
    thinning_iters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mask_255, skeleton_255) as uint8 images.
    """
    if image_rgb.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    if blur_ksize > 0:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if canny_low < 0 or canny_high < 0:
        low, high = _auto_canny_thresholds(gray, sigma=canny_sigma)
    else:
        low, high = int(canny_low), int(canny_high)
        if low >= high:
            low = max(0, high - 1)

    edges = cv2.Canny(gray, low, high)

    k = max(int(morph_ksize), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    mask = edges
    if morph_close_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(morph_close_iters))
    if morph_open_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(morph_open_iters))

    # Connected component filtering
    if min_component_area_px > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= int(min_component_area_px):
                keep[labels == i] = 255
        mask = keep
    else:
        mask = (mask > 0).astype(np.uint8) * 255

    skeleton = _zhang_suen_thinning(mask, max_iters=int(thinning_iters))
    return mask, skeleton


def _extract_paths_from_mask_contours(mask_255: np.ndarray, min_contour_points: int) -> List[List[Tuple[int, int]]]:
    """
    Extract ordered paths from binary mask contours.
    Path point format is (row, col) to match downstream writer.
    """
    contours, _ = cv2.findContours(mask_255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths: List[List[Tuple[int, int]]] = []
    min_pts = max(int(min_contour_points), 2)
    for cnt in contours:
        if len(cnt) < min_pts:
            continue
        path: List[Tuple[int, int]] = []
        for pt in cnt[:, 0, :]:
            col = int(pt[0])
            row = int(pt[1])
            path.append((row, col))
        if len(path) >= 2:
            paths.append(path)
    return paths


def _extract_paths_from_mask_pixels(mask_255: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Fallback extractor: each foreground pixel becomes a tiny 2-point path.
    Guarantees non-empty geometry when contours are too fragmented.
    """
    ys, xs = np.where(mask_255 > 0)
    paths: List[List[Tuple[int, int]]] = []
    for row, col in zip(ys.tolist(), xs.tolist()):
        paths.append([(int(row), int(col)), (int(row), int(col + 1))])
    return paths


def _skeleton_degrees(skel01: np.ndarray) -> np.ndarray:
    """
    For skel01 (0/1), compute 8-neighborhood degree at each pixel.
    """
    padded = np.pad(skel01, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    deg = np.zeros_like(skel01, dtype=np.uint8)
    # sum of 8 neighbors
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            deg += padded[1 + dx : 1 + dx + skel01.shape[0], 1 + dy : 1 + dy + skel01.shape[1]]
    return deg


def _trace_skeleton_paths(skeleton_255: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Trace skeleton pixels into ordered paths (in pixel coords). Returns list of paths.
    """
    skel01 = (skeleton_255 > 0).astype(np.uint8)
    if skel01.sum() == 0:
        return []

    deg = _skeleton_degrees(skel01)
    visited = np.zeros_like(skel01, dtype=np.uint8)

    h, w = skel01.shape

    def neighbors8(x, y):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and skel01[nx, ny]:
                    yield nx, ny

    def trace_from(start):
        path = [start]
        visited[start] = 1
        prev = None
        cur = start
        while True:
            nbrs = [p for p in neighbors8(cur[0], cur[1]) if not visited[p]]
            if not nbrs:
                break
            # prefer continuing direction if possible
            if prev is not None and len(nbrs) > 1:
                vx, vy = cur[0] - prev[0], cur[1] - prev[1]
                def score(p):
                    wx, wy = p[0] - cur[0], p[1] - cur[1]
                    return vx * wx + vy * wy
                nbrs.sort(key=score, reverse=True)
            nxt = nbrs[0]
            prev = cur
            cur = nxt
            path.append(cur)
            visited[cur] = 1
            # stop at junction or endpoint (but include it)
            d = int(deg[cur])
            if d != 2:
                break
        return path

    paths: List[List[Tuple[int, int]]] = []

    # First, start from endpoints and junctions to get open paths
    seeds = np.argwhere((skel01 == 1) & (deg != 2))
    for x, y in seeds:
        if visited[x, y]:
            continue
        p = trace_from((int(x), int(y)))
        if len(p) >= 2:
            paths.append(p)

    # Then, handle remaining loops (deg==2 everywhere) by tracing any unvisited pixel
    remaining = np.argwhere((skel01 == 1) & (visited == 0))
    for x, y in remaining:
        if visited[x, y]:
            continue
        loop = trace_from((int(x), int(y)))
        if len(loop) >= 3:
            paths.append(loop)

    return paths


def _rdp_simplify(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    if len(points) < 3 or epsilon <= 0:
        return points

    def perp_dist(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return float(np.hypot(px - ax, py - ay))
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t * dx, ay + t * dy
        return float(np.hypot(px - cx, py - cy))

    a = points[0]
    b = points[-1]
    max_d = -1.0
    idx = -1
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i], a, b)
        if d > max_d:
            max_d = d
            idx = i
    if max_d > epsilon:
        left = _rdp_simplify(points[: idx + 1], epsilon)
        right = _rdp_simplify(points[idx:], epsilon)
        return left[:-1] + right
    return [a, b]


def _px_to_mm_points(
    contour: np.ndarray,
    image_height_px: int,
    mm_per_pixel: float,
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for point in contour[:, 0, :]:
        x_px = float(point[0])
        y_px = float(point[1])
        # Flip Y axis so +Y goes up in CAD.
        x_mm = x_px * mm_per_pixel
        y_mm = (image_height_px - y_px) * mm_per_pixel
        pts.append((x_mm, y_mm))
    return pts


def _write_dxf_polylines(
    contours: Sequence[np.ndarray],
    dxf_path: str,
    image_height_px: int,
    mm_per_pixel: float,
    layer_name: str = "VECTORIZED",
) -> Tuple[int, Optional[Tuple[float, float, float, float]]]:
    doc = ezdxf.new("R2018")
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    if layer_name not in doc.layers:
        doc.layers.new(name=layer_name, dxfattribs={"color": 7})

    entity_count = 0
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for contour in contours:
        points_mm = _px_to_mm_points(contour, image_height_px, mm_per_pixel)
        if len(points_mm) < 2:
            continue

        for x, y in points_mm:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

        if len(points_mm) == 2:
            msp.add_line(
                points_mm[0],
                points_mm[1],
                dxfattribs={"layer": layer_name},
            )
            entity_count += 1
            continue

        msp.add_lwpolyline(
            points_mm,
            format="xy",
            close=True,
            dxfattribs={"layer": layer_name},
        )
        entity_count += 1

    doc.saveas(dxf_path)
    if entity_count == 0:
        return 0, None
    return entity_count, (min_x, min_y, max_x, max_y)


def _write_dxf_paths(
    paths_px: Sequence[Sequence[Tuple[int, int]]],
    dxf_path: str,
    image_height_px: int,
    mm_per_pixel: float,
    simplify_mm: float,
    min_path_length_mm: float,
    geometry_mode: str = "lines",
    layer_name: str = "VECTORIZED",
) -> Tuple[int, Optional[Tuple[float, float, float, float]]]:
    doc = ezdxf.new("R12")
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    if layer_name not in doc.layers:
        doc.layers.new(name=layer_name, dxfattribs={"color": 7})

    entity_count = 0
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    normalized_paths: List[List[Tuple[float, float]]] = []
    raw_paths_mm: List[List[Tuple[float, float]]] = []

    def _dedupe_consecutive(points: List[Tuple[float, float]], eps: float = 1e-6):
        if not points:
            return []
        out = [points[0]]
        for p in points[1:]:
            if abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
                out.append(p)
        return out

    for path in paths_px:
        pts_mm = []
        for x_px, y_px in path:
            # Note: path pixels are (row,col) = (y,x) style; our path is (x,y) as (row,col)
            # Convert to (x_px, y_px) in image coords: x=col, y=row
            col = float(y_px)
            row = float(x_px)
            x_mm = col * mm_per_pixel
            y_mm = (image_height_px - row) * mm_per_pixel
            pts_mm.append((x_mm, y_mm))
        if len(pts_mm) >= 2:
            raw_paths_mm.append(pts_mm)

        if len(pts_mm) < 2:
            continue

        # simplify in mm space
        simplified = _rdp_simplify(pts_mm, epsilon=float(simplify_mm))
        simplified = _dedupe_consecutive(simplified)
        if len(simplified) < 2:
            continue

        # length filter
        length = 0.0
        for i in range(1, len(simplified)):
            x0, y0 = simplified[i - 1]
            x1, y1 = simplified[i]
            length += float(np.hypot(x1 - x0, y1 - y0))
        if length < float(min_path_length_mm):
            continue

        for x, y in simplified:
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        normalized_paths.append(simplified)

    if not normalized_paths:
        # Fallback: write raw paths to help diagnose over-filtering.
        raw_layer = "VECTORIZED_RAW"
        if raw_layer not in doc.layers:
            doc.layers.new(name=raw_layer, dxfattribs={"color": 6})
        raw_count = 0
        mode = (geometry_mode or "lines").strip().lower()
        if mode not in {"lines", "polyline"}:
            mode = "lines"
        for pts in raw_paths_mm:
            if len(pts) < 2:
                continue
            if mode == "polyline":
                msp.add_polyline2d(pts, dxfattribs={"layer": raw_layer})
                raw_count += 1
            else:
                for i in range(1, len(pts)):
                    p0 = pts[i - 1]
                    p1 = pts[i]
                    if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-6:
                        continue
                    msp.add_line(p0, p1, dxfattribs={"layer": raw_layer})
                    raw_count += 1
        doc.saveas(dxf_path)
        if raw_count == 0:
            return 0, None
        # Unknown bounds for raw fallback in this branch; still return visible extents marker.
        return raw_count, (0.0, 0.0, 100.0, 100.0)

    # Shift everything near origin to avoid far-away geometry visibility issues.
    shift_x = min_x if np.isfinite(min_x) else 0.0
    shift_y = min_y if np.isfinite(min_y) else 0.0

    mode = (geometry_mode or "lines").strip().lower()
    if mode not in {"lines", "polyline"}:
        mode = "lines"

    for simplified in normalized_paths:
        shifted = [(x - shift_x, y - shift_y) for x, y in simplified]
        if mode == "polyline":
            msp.add_polyline2d(shifted, dxfattribs={"layer": layer_name})
            entity_count += 1
        else:
            for i in range(1, len(shifted)):
                p0 = shifted[i - 1]
                p1 = shifted[i]
                if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-6:
                    continue
                msp.add_line(p0, p1, dxfattribs={"layer": layer_name})
                entity_count += 1

    # Add a small reference frame on debug layer to guarantee visible extents.
    debug_layer = "VECTORIZED_DEBUG"
    if debug_layer not in doc.layers:
        doc.layers.new(name=debug_layer, dxfattribs={"color": 1})
    width = max_x - min_x
    height = max_y - min_y
    if np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0:
        frame = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height), (0.0, 0.0)]
        for i in range(1, len(frame)):
            msp.add_line(frame[i - 1], frame[i], dxfattribs={"layer": debug_layer})

    # Always add a fixed-size visible reference at origin for CAD import diagnostics.
    always_layer = "R2V_ALWAYS_VISIBLE"
    if always_layer not in doc.layers:
        doc.layers.new(name=always_layer, dxfattribs={"color": 3})
    ref = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)]
    for i in range(1, len(ref)):
        msp.add_line(ref[i - 1], ref[i], dxfattribs={"layer": always_layer})
    msp.add_line((0.0, 0.0), (100.0, 100.0), dxfattribs={"layer": always_layer})
    msp.add_line((0.0, 100.0), (100.0, 0.0), dxfattribs={"layer": always_layer})

    doc.saveas(dxf_path)
    if entity_count == 0:
        return 0, None
    return entity_count, (0.0, 0.0, max_x - min_x, max_y - min_y)


def _save_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cache_dir(base_output_dir: str) -> str:
    # Never write cache under output_dir; keep output directory clean.
    # Use a system temp folder instead (Windows: %TEMP%).
    path = os.path.join(tempfile.gettempdir(), "comfyui_r2v_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _write_vector_package_temp(output_dir: str, safe_name: str, payload: Dict[str, object]) -> str:
    cache = _cache_dir(output_dir)
    path = os.path.join(cache, f"{safe_name}_vector_package.json")
    _save_json(path, payload)
    return path


def _try_delete(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _write_probe_dxf(probe_path: str) -> None:
    """
    Write a minimal, guaranteed-visible DXF probe file.
    This is independent from vectorization results and helps diagnose CAD import/display issues.
    """
    doc = ezdxf.new("R12")
    msp = doc.modelspace()
    layer = "PROBE_VISIBLE"
    if layer not in doc.layers:
        doc.layers.new(name=layer, dxfattribs={"color": 2})
    msp.add_line((0.0, 0.0), (100.0, 0.0), dxfattribs={"layer": layer})
    msp.add_line((100.0, 0.0), (100.0, 100.0), dxfattribs={"layer": layer})
    msp.add_line((100.0, 100.0), (0.0, 100.0), dxfattribs={"layer": layer})
    msp.add_line((0.0, 100.0), (0.0, 0.0), dxfattribs={"layer": layer})
    msp.add_line((0.0, 0.0), (100.0, 100.0), dxfattribs={"layer": layer})
    msp.add_line((0.0, 100.0), (100.0, 0.0), dxfattribs={"layer": layer})
    doc.saveas(probe_path)


def _convert_dxf_to_dwg(odafc_exe: str, dxf_path: str, dwg_path: str) -> None:
    dxf_dir = os.path.dirname(dxf_path)
    input_name = os.path.basename(dxf_path)
    output_dir = os.path.dirname(dwg_path)
    output_name = os.path.basename(dwg_path)
    _ensure_dir(output_dir)

    # ODA converter converts all files in input dir to output dir.
    # Arguments:
    # ODAFileConverter <in_folder> <out_folder> <in_ver> <out_ver> <recurse> <audit> [input_filter]
    cmd = [
        odafc_exe,
        dxf_dir,
        output_dir,
        "ACAD2007",
        "ACAD2007",
        "0",
        "1",
        input_name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ODAFileConverter failed.\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    generated_dwg = os.path.join(output_dir, os.path.splitext(input_name)[0] + ".dwg")
    if not os.path.isfile(generated_dwg):
        raise RuntimeError(
            "DWG not generated by ODAFileConverter. "
            f"Expected: {generated_dwg}"
        )

    if os.path.normcase(generated_dwg) != os.path.normcase(dwg_path):
        shutil.move(generated_dwg, dwg_path)

    if not os.path.isfile(dwg_path):
        raise RuntimeError(f"DWG output file missing: {dwg_path}")

    # Try to remove temporary DXF if it still exists in output dir.
    out_dxf = os.path.join(output_dir, input_name)
    if os.path.isfile(out_dxf):
        try:
            os.remove(out_dxf)
        except OSError:
            pass


class ComfyImageToDWG:
    @classmethod
    def INPUT_TYPES(cls):
        default_output_dir = r"C:\ComfyUI\output"
        if folder_paths is not None:
            try:
                default_output_dir = folder_paths.get_output_directory()
            except Exception:
                pass
        return {
            "required": {
                "image": ("IMAGE",),
                "output_dir": ("STRING", {"default": default_output_dir}),
                "output_name": ("STRING", {"default": "comfy_vector"}),
                "filename_template": ("STRING", {"default": "{name}_{counter}"}),
                "counter_digits": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "sequence_start": ("INT", {"default": 1, "min": 0, "max": 999999999, "step": 1}),
                "conflict_policy": (["next_sequence", "overwrite", "timestamp"],),
                "vector_mode": (["edge_photo_v2"],),
                "save_debug_images": ("BOOLEAN", {"default": True}),
                "canny_low": ("INT", {"default": -1, "min": -1, "max": 255, "step": 1}),
                "canny_high": ("INT", {"default": -1, "min": -1, "max": 255, "step": 1}),
                "canny_sigma": ("FLOAT", {"default": 0.33, "min": 0.01, "max": 0.99, "step": 0.01}),
                "blur_ksize": ("INT", {"default": 3, "min": 0, "max": 31, "step": 1}),
                "morph_ksize": ("INT", {"default": 3, "min": 1, "max": 31, "step": 1}),
                "morph_close_iters": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1}),
                "morph_open_iters": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "min_component_area_px": ("INT", {"default": 40, "min": 0, "max": 1000000, "step": 1}),
                "thinning_iters": ("INT", {"default": 80, "min": 1, "max": 500, "step": 1}),
                "simplify_mm": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 50.0, "step": 0.1}),
                "min_path_length_mm": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10000.0, "step": 0.5}),
                "threshold": ("INT", {"default": 160, "min": 0, "max": 255, "step": 1}),
                "mm_per_pixel": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.001}),
                "min_area_px": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 1000000.0, "step": 1.0}),
                "simplify_px": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "save_preview_png": ("BOOLEAN", {"default": True}),
                "allow_dxf_fallback": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "odafc_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dwg_path", "status")
    FUNCTION = "export_dwg"
    OUTPUT_NODE = True
    CATEGORY = "CAD/Export"

    def export_dwg(
        self,
        image,
        output_dir,
        output_name,
        filename_template,
        counter_digits,
        sequence_start,
        conflict_policy,
        vector_mode,
        save_debug_images,
        canny_low,
        canny_high,
        canny_sigma,
        blur_ksize,
        morph_ksize,
        morph_close_iters,
        morph_open_iters,
        min_component_area_px,
        thinning_iters,
        simplify_mm,
        min_path_length_mm,
        threshold,
        mm_per_pixel,
        min_area_px,
        simplify_px,
        save_preview_png,
        allow_dxf_fallback,
        odafc_path="",
    ):
        _ensure_dir(output_dir)
        resolved_name = _resolve_output_basename(
            output_dir=output_dir,
            output_name=output_name,
            filename_template=filename_template,
            counter_digits=counter_digits,
            sequence_start=sequence_start,
            conflict_policy=conflict_policy,
        )

        image_uint8 = _tensor_to_uint8_image(image)
        h_px = image_uint8.shape[0]

        if save_preview_png:
            preview_path = os.path.join(output_dir, f"{resolved_name}.png")
            Image.fromarray(image_uint8).save(preview_path)

        report: Dict[str, object] = {
            "vector_mode": vector_mode,
            "mm_per_pixel": float(mm_per_pixel),
        }

        # V2 Edge pipeline for photo/render style images.
        mask, skeleton = _preprocess_edge_mode(
            image_rgb=image_uint8,
            canny_sigma=float(canny_sigma),
            canny_low=int(canny_low),
            canny_high=int(canny_high),
            blur_ksize=int(blur_ksize),
            morph_ksize=int(morph_ksize),
            morph_close_iters=int(morph_close_iters),
            morph_open_iters=int(morph_open_iters),
            min_component_area_px=int(min_component_area_px),
            thinning_iters=int(thinning_iters),
        )
        paths_px = _trace_skeleton_paths(skeleton)
        report["mask_nonzero_px"] = int((mask > 0).sum())
        report["skeleton_nonzero_px"] = int((skeleton > 0).sum())
        report["paths"] = int(len(paths_px))

        if save_debug_images:
            Image.fromarray(mask).save(os.path.join(output_dir, f"{resolved_name}_mask.png"))
            Image.fromarray(skeleton).save(os.path.join(output_dir, f"{resolved_name}_skeleton.png"))

        if not paths_px:
            raise RuntimeError(
                "No skeleton paths detected. For photo/render images try: "
                "increase morph_close_iters, lower min_component_area_px, or adjust canny thresholds."
            )

        dwg_path = os.path.join(output_dir, f"{resolved_name}.dwg")
        dxf_final_path = os.path.join(output_dir, f"{resolved_name}.dxf")

        # Write DXF first, then convert to DWG.
        with tempfile.TemporaryDirectory(prefix="comfy_dwg_") as tmpdir:
            dxf_path = os.path.join(tmpdir, f"{resolved_name}.dxf")
            entity_count, bounds = _write_dxf_paths(
                paths_px=paths_px,
                dxf_path=dxf_path,
                image_height_px=h_px,
                mm_per_pixel=float(mm_per_pixel),
                simplify_mm=float(simplify_mm),
                min_path_length_mm=float(min_path_length_mm),
            )
            if entity_count == 0:
                raise RuntimeError(
                    "Vectorization produced zero drawable entities. "
                    "Try lowering simplify_mm and min_path_length_mm."
                )
            shutil.copyfile(dxf_path, dxf_final_path)

            odafc_exe = _find_odafc_executable(odafc_path)
            if not odafc_exe:
                if allow_dxf_fallback:
                    status = (
                        "ODAFileConverter not found. Exported DXF fallback: "
                        f"{dxf_final_path} | paths: {len(paths_px)} | entities: {entity_count} "
                        f"| bounds_mm: {bounds} | mm_per_pixel: {mm_per_pixel}"
                    )
                    report_path = os.path.join(output_dir, f"{resolved_name}_report.json")
                    with open(report_path, "w", encoding="utf-8") as f:
                        json.dump({**report, "entities": entity_count, "bounds_mm": bounds}, f, ensure_ascii=False, indent=2)
                    return (dxf_final_path, status)
                raise RuntimeError(
                    "ODAFileConverter not found. Install it and set odafc_path "
                    "or environment variable ODAFC_PATH.\n"
                    "Download: https://www.opendesign.com/guestfiles/oda_file_converter"
                )

            _convert_dxf_to_dwg(odafc_exe=odafc_exe, dxf_path=dxf_path, dwg_path=dwg_path)

        status = (
            f"DWG exported: {dwg_path} | dxf: {dxf_final_path} | paths: {len(paths_px)} | entities: {entity_count} "
            f"| bounds_mm: {bounds} | mm_per_pixel: {mm_per_pixel}"
        )
        report_path = os.path.join(output_dir, f"{resolved_name}_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({**report, "entities": entity_count, "bounds_mm": bounds, "dwg_path": dwg_path, "dxf_path": dxf_final_path}, f, ensure_ascii=False, indent=2)
        print(f"[comfyui_dwg_exporter] {status}")
        return (dwg_path, status)


class R2VStyleVectorizeAuto:
    @classmethod
    def INPUT_TYPES(cls):
        default_output_dir = r"C:\ComfyUI\output"
        if folder_paths is not None:
            try:
                default_output_dir = folder_paths.get_output_directory()
            except Exception:
                pass
        return {
            "required": {
                "image": ("IMAGE",),
                "output_dir": ("STRING", {"default": default_output_dir}),
                "package_name": ("STRING", {"default": "r2v_vector"}),
                "save_debug_images": ("BOOLEAN", {"default": False}),
                "canny_low": ("INT", {"default": -1, "min": -1, "max": 255, "step": 1}),
                "canny_high": ("INT", {"default": -1, "min": -1, "max": 255, "step": 1}),
                "canny_sigma": ("FLOAT", {"default": 0.33, "min": 0.01, "max": 0.99, "step": 0.01}),
                "blur_ksize": ("INT", {"default": 3, "min": 0, "max": 31, "step": 1}),
                "morph_ksize": ("INT", {"default": 3, "min": 1, "max": 31, "step": 1}),
                "morph_close_iters": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
                "morph_open_iters": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "min_component_area_px": ("INT", {"default": 40, "min": 0, "max": 1000000, "step": 1}),
                "thinning_iters": ("INT", {"default": 80, "min": 1, "max": 500, "step": 1}),
                "extract_mode": (["contour_direct_v1", "skeleton_trace_v1"],),
                "min_contour_points": ("INT", {"default": 8, "min": 2, "max": 10000, "step": 1}),
                "min_paths_required": ("INT", {"default": 10, "min": 1, "max": 1000000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("vector_package_path", "status")
    FUNCTION = "vectorize"
    OUTPUT_NODE = True
    CATEGORY = "CAD/R2V"

    def vectorize(
        self,
        image,
        output_dir,
        package_name,
        save_debug_images,
        canny_low,
        canny_high,
        canny_sigma,
        blur_ksize,
        morph_ksize,
        morph_close_iters,
        morph_open_iters,
        min_component_area_px,
        thinning_iters,
        extract_mode,
        min_contour_points,
        min_paths_required,
    ):
        _ensure_dir(output_dir)
        safe_name = _sanitize_filename_part(package_name)

        image_uint8 = _tensor_to_uint8_image(image)
        h_px = int(image_uint8.shape[0])
        w_px = int(image_uint8.shape[1])

        mask, skeleton = _preprocess_edge_mode(
            image_rgb=image_uint8,
            canny_sigma=float(canny_sigma),
            canny_low=int(canny_low),
            canny_high=int(canny_high),
            blur_ksize=int(blur_ksize),
            morph_ksize=int(morph_ksize),
            morph_close_iters=int(morph_close_iters),
            morph_open_iters=int(morph_open_iters),
            min_component_area_px=int(min_component_area_px),
            thinning_iters=int(thinning_iters),
        )
        mode = (extract_mode or "contour_direct_v1").strip().lower()
        if mode == "skeleton_trace_v1":
            paths_px = _trace_skeleton_paths(skeleton)
        else:
            paths_px = _extract_paths_from_mask_contours(mask, min_contour_points=int(min_contour_points))
            if not paths_px:
                paths_px = _extract_paths_from_mask_pixels(mask)

        if save_debug_images:
            Image.fromarray(image_uint8).save(os.path.join(output_dir, f"{safe_name}_source.png"))
            Image.fromarray(mask).save(os.path.join(output_dir, f"{safe_name}_mask.png"))
            Image.fromarray(skeleton).save(os.path.join(output_dir, f"{safe_name}_skeleton.png"))

        if len(paths_px) < int(min_paths_required):
            raise RuntimeError(
                f"Vectorization quality gate failed: paths={len(paths_px)} < min_paths_required={min_paths_required}. "
                "Tune edge params before export."
            )

        payload: Dict[str, object] = {
            "version": "r2v_style_v1",
            "image_size_px": [w_px, h_px],
            "paths_px": [[[int(p[0]), int(p[1])] for p in path] for path in paths_px],
            "metrics": {
                "paths": int(len(paths_px)),
                "mask_nonzero_px": int((mask > 0).sum()),
                "skeleton_nonzero_px": int((skeleton > 0).sum()),
            },
            "params": {
                "canny_low": int(canny_low),
                "canny_high": int(canny_high),
                "canny_sigma": float(canny_sigma),
                "blur_ksize": int(blur_ksize),
                "morph_ksize": int(morph_ksize),
                "morph_close_iters": int(morph_close_iters),
                "morph_open_iters": int(morph_open_iters),
                "min_component_area_px": int(min_component_area_px),
                "thinning_iters": int(thinning_iters),
                "extract_mode": mode,
                "min_contour_points": int(min_contour_points),
            },
        }
        package_path = _write_vector_package_temp(output_dir, safe_name, payload)
        status = f"Vector package saved: {package_path} | mode: {mode} | paths: {len(paths_px)}"
        print(f"[comfyui_dwg_exporter] {status}")
        return (package_path, status)


class R2VStyleExportDWG:
    @classmethod
    def INPUT_TYPES(cls):
        default_output_dir = r"C:\ComfyUI\output"
        if folder_paths is not None:
            try:
                default_output_dir = folder_paths.get_output_directory()
            except Exception:
                pass
        return {
            "required": {
                "vector_package_path": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": default_output_dir}),
                "output_name": ("STRING", {"default": "r2v_export"}),
                "mm_per_pixel": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.001}),
                "simplify_mm": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 50.0, "step": 0.1}),
                "min_path_length_mm": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10000.0, "step": 0.5}),
                "geometry_mode": (["polyline", "lines"],),
                "allow_dxf_fallback": ("BOOLEAN", {"default": True}),
                "write_probe_dxf": ("BOOLEAN", {"default": False}),
                "delete_vector_package_after_export": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "odafc_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dwg_or_dxf_path", "status")
    FUNCTION = "export_from_package"
    OUTPUT_NODE = True
    CATEGORY = "CAD/R2V"

    def export_from_package(
        self,
        vector_package_path,
        output_dir,
        output_name,
        mm_per_pixel,
        simplify_mm,
        min_path_length_mm,
        geometry_mode,
        allow_dxf_fallback,
        write_probe_dxf,
        delete_vector_package_after_export,
        odafc_path="",
    ):
        if not vector_package_path or not os.path.isfile(vector_package_path):
            raise RuntimeError(f"Vector package not found: {vector_package_path}")
        _ensure_dir(output_dir)

        package = _load_json(vector_package_path)
        image_size = package.get("image_size_px", [0, 0])
        if not isinstance(image_size, list) or len(image_size) != 2:
            raise RuntimeError("Invalid vector package: image_size_px missing.")
        h_px = int(image_size[1])
        if h_px <= 0:
            raise RuntimeError("Invalid vector package: image height <= 0.")

        raw_paths = package.get("paths_px", [])
        if not isinstance(raw_paths, list):
            raise RuntimeError("Invalid vector package: paths_px missing.")
        paths_px: List[List[Tuple[int, int]]] = []
        for path in raw_paths:
            if not isinstance(path, list):
                continue
            casted: List[Tuple[int, int]] = []
            for p in path:
                if isinstance(p, list) and len(p) == 2:
                    casted.append((int(p[0]), int(p[1])))
            if len(casted) >= 2:
                paths_px.append(casted)

        if not paths_px:
            raise RuntimeError("Vector package contains zero valid paths.")

        safe_name = _sanitize_filename_part(output_name)
        dxf_path = os.path.join(output_dir, f"{safe_name}.dxf")
        probe_dxf_path = os.path.join(output_dir, f"{safe_name}_probe.dxf")
        dwg_path = os.path.join(output_dir, f"{safe_name}.dwg")

        entity_count, bounds = _write_dxf_paths(
            paths_px=paths_px,
            dxf_path=dxf_path,
            image_height_px=h_px,
            mm_per_pixel=float(mm_per_pixel),
            simplify_mm=float(simplify_mm),
            min_path_length_mm=float(min_path_length_mm),
            geometry_mode=str(geometry_mode),
        )
        if entity_count == 0:
            raise RuntimeError("DXF export produced zero entities after filters.")

        if write_probe_dxf:
            _write_probe_dxf(probe_dxf_path)

        odafc_exe = _find_odafc_executable(odafc_path)
        if not odafc_exe:
            if allow_dxf_fallback:
                status = (
                    f"DXF exported (DWG skipped, ODA missing): {dxf_path} | "
                    f"entities: {entity_count} | bounds_mm: {bounds}"
                )
                if write_probe_dxf:
                    status += f" | probe: {probe_dxf_path}"
                print(f"[comfyui_dwg_exporter] {status}")
                if delete_vector_package_after_export:
                    _try_delete(vector_package_path)
                return (dxf_path, status)
            raise RuntimeError("ODAFileConverter not found and fallback disabled.")

        with tempfile.TemporaryDirectory(prefix="comfy_dwg_") as tmpdir:
            tmp_dxf = os.path.join(tmpdir, os.path.basename(dxf_path))
            shutil.copyfile(dxf_path, tmp_dxf)
            _convert_dxf_to_dwg(odafc_exe=odafc_exe, dxf_path=tmp_dxf, dwg_path=dwg_path)

        status = (
            f"DWG exported: {dwg_path} | DXF: {dxf_path} | entities: {entity_count} | bounds_mm: {bounds}"
        )
        if write_probe_dxf:
            status += f" | probe: {probe_dxf_path}"
        print(f"[comfyui_dwg_exporter] {status}")
        if delete_vector_package_after_export:
            _try_delete(vector_package_path)
        return (dwg_path, status)


class R2VStyleVectorizePreset:
    @classmethod
    def INPUT_TYPES(cls):
        default_output_dir = r"C:\ComfyUI\output"
        if folder_paths is not None:
            try:
                default_output_dir = folder_paths.get_output_directory()
            except Exception:
                pass
        return {
            "required": {
                "image": ("IMAGE",),
                "output_dir": ("STRING", {"default": default_output_dir}),
                "package_name": ("STRING", {"default": "r2v_vector"}),
                "detail_level": (["简", "中", "高"],),
                "extract_mode": (["contour_direct_v1", "skeleton_trace_v1"],),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("vector_package_path", "mask_preview", "skeleton_preview", "status")
    FUNCTION = "vectorize_preset"
    OUTPUT_NODE = True
    CATEGORY = "CAD/R2V"

    def vectorize_preset(self, image, output_dir, package_name, detail_level, extract_mode):
        _ensure_dir(output_dir)
        level = (detail_level or "中").strip()
        safe_name = _sanitize_filename_part(package_name) + f"_{level}"

        if level == "简":
            canny_sigma = 0.40
            blur_ksize = 5
            morph_ksize = 3
            morph_close_iters = 2
            min_component_area_px = 60
        elif level == "高":
            canny_sigma = 0.20
            blur_ksize = 1
            morph_ksize = 2
            morph_close_iters = 1
            min_component_area_px = 5
        else:
            canny_sigma = 0.30
            blur_ksize = 3
            morph_ksize = 3
            morph_close_iters = 2
            min_component_area_px = 20

        image_uint8 = _tensor_to_uint8_image(image)
        h_px = int(image_uint8.shape[0])
        w_px = int(image_uint8.shape[1])

        mask, skeleton = _preprocess_edge_mode(
            image_rgb=image_uint8,
            canny_sigma=float(canny_sigma),
            canny_low=-1,
            canny_high=-1,
            blur_ksize=int(blur_ksize),
            morph_ksize=int(morph_ksize),
            morph_close_iters=int(morph_close_iters),
            morph_open_iters=0,
            min_component_area_px=int(min_component_area_px),
            thinning_iters=80,
        )

        mode = (extract_mode or "contour_direct_v1").strip().lower()
        if mode == "skeleton_trace_v1":
            paths_px = _trace_skeleton_paths(skeleton)
        else:
            paths_px = _extract_paths_from_mask_contours(mask, min_contour_points=2)
            if not paths_px:
                paths_px = _extract_paths_from_mask_pixels(mask)

        if not paths_px:
            raise RuntimeError("No paths extracted. Try using a higher detail preset.")

        payload: Dict[str, object] = {
            "version": "r2v_style_v1",
            "image_size_px": [w_px, h_px],
            "paths_px": [[[int(p[0]), int(p[1])] for p in path] for path in paths_px],
            "metrics": {
                "paths": int(len(paths_px)),
                "mask_nonzero_px": int((mask > 0).sum()),
                "skeleton_nonzero_px": int((skeleton > 0).sum()),
            },
            "params": {
                "detail_level": level,
                "extract_mode": mode,
                "canny_sigma": float(canny_sigma),
                "blur_ksize": int(blur_ksize),
                "morph_ksize": int(morph_ksize),
                "morph_close_iters": int(morph_close_iters),
                "min_component_area_px": int(min_component_area_px),
            },
        }

        package_path = _write_vector_package_temp(output_dir, safe_name, payload)
        status = f"Vector package cached: {package_path} | preset: {level} | mode: {mode} | paths: {len(paths_px)}"
        print(f"[comfyui_dwg_exporter] {status}")

        mask_rgb = np.stack([mask, mask, mask], axis=-1).astype(np.float32) / 255.0
        skel_rgb = np.stack([skeleton, skeleton, skeleton], axis=-1).astype(np.float32) / 255.0
        mask_t = torch.from_numpy(mask_rgb)[None, ...]
        skel_t = torch.from_numpy(skel_rgb)[None, ...]
        return (package_path, mask_t, skel_t, status)


NODE_CLASS_MAPPINGS = {
    "R2VStyleVectorizeAuto": R2VStyleVectorizeAuto,
    "R2VStyleVectorizePreset": R2VStyleVectorizePreset,
    "R2VStyleExportDWG": R2VStyleExportDWG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "R2VStyleVectorizeAuto": "R2V Style Vectorize (Auto)",
    "R2VStyleVectorizePreset": "R2V Style Vectorize (简/中/高)",
    "R2VStyleExportDWG": "R2V Style Export (DXF/DWG)",
}
