# ComfyUI DWG Export Node

This custom node adds an `Image to DWG (mm)` node to ComfyUI.
![60d88de002471ce3f1c0567b49b2d14a](https://github.com/user-attachments/assets/100e381c-8247-4f99-8eee-bd9c7452e722)

It vectorizes the generated image into CAD polylines and exports a **DWG** file in **millimeter units**.
这个是CAD中打开后的效果
![428d00bad14c957cb70d8b9225c08a09](https://github.com/user-attachments/assets/3dc23c31-f9a3-4ec1-868b-44ac3238f652)

## Features

- Auto-discovered by ComfyUI when placed in `ComfyUI/custom_nodes/comfyui_dwg_exporter`
- Auto dependency bootstrap on startup (Windows friendly)
- Input image from any ComfyUI image pipeline
- Engineering-grade **Edge/Photo V2** pipeline (edge → denoise → skeleton → path tracing → vector)
- Output units set to **mm**
- Export flow: `Image -> DXF -> DWG`
- Node UI button to open a local folder picker dialog and fill `output_dir`
- `Save As...` button that opens a native save dialog and fills directory + filename
- SaveImage-like naming controls (`filename_template`, sequence, conflict policy)
- Recent folder memory in UI (`Use Recent Folder`)
- Optional DXF fallback when ODA converter is unavailable
- Debug outputs: `*_mask.png`, `*_skeleton.png`, `*_report.json`, plus always-saved `*.dxf`
- New R2V-style dual-node workflow:
  - `R2V Style Vectorize (Auto)` for extraction + quality gate + package
  - `R2V Style Export (DXF/DWG)` for deterministic CAD export

## Recommended Workflow (R2V Style)

### Option A (Recommended): Preset UI

1. Add `R2V Style Vectorize (简/中/高)` and connect your ComfyUI image output.
2. Choose `detail_level` (简/中/高). The node outputs `mask_preview` and `skeleton_preview` for ComfyUI preview.
3. Connect `vector_package_path` into `R2V Style Export (DXF/DWG)`.
4. Export to DXF/DWG.

### Option B: Advanced UI

1. Add `R2V Style Vectorize (Auto)` and connect your ComfyUI image output.
2. Tune edge parameters; if you want PNGs written to disk, enable `save_debug_images`.
3. Connect `vector_package_path` into `R2V Style Export (DXF/DWG)`.
4. Export to DXF/DWG.

This separates "vectorization quality" from "CAD export", making failures easy to diagnose.

## About the JSON file

The vectorizer produces a `vector_package.json` as an internal handoff between nodes.
It is now stored under the system temp folder (e.g. `%TEMP%/comfyui_r2v_cache`) and (by default) deleted after export.

For photo/render images, use:
- `extract_mode = contour_direct_v1` (default, most robust)
- `min_contour_points = 8`

## Why DXF then DWG

Direct DWG writing from Python is not reliably supported in open-source libraries.
This node writes standard CAD vectors to DXF (`ezdxf`) and then converts to DWG using **ODA File Converter**.

## Install

1. Copy this folder into:

   `ComfyUI/custom_nodes/comfyui_dwg_exporter`

2. Install dependencies (recommended one-click on Windows):

   ```bat
   ComfyUI\custom_nodes\comfyui_dwg_exporter\install_windows.bat
   ```

   Or manual:

   ```bash
   pip install -r ComfyUI/custom_nodes/comfyui_dwg_exporter/requirements.txt
   ```

3. Install **ODA File Converter** (Windows):

   - [https://www.opendesign.com/guestfiles/oda_file_converter](https://www.opendesign.com/guestfiles/oda_file_converter)

4. Option A: set environment variable:

   - `ODAFC_PATH=C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe`

   Option B: fill `odafc_path` input field in the node.

5. Restart ComfyUI.

## Node Inputs

- `image`: ComfyUI IMAGE input
- `output_dir`: output folder
- `output_name`: output base name
- `filename_template`: naming template, supports `{name}` and `{counter}`
- `counter_digits`: zero-pad length for `{counter}`
- `sequence_start`: start index for auto sequence
- `conflict_policy`:
  - `next_sequence`: auto increment to avoid overwrite
  - `overwrite`: write fixed name
  - `timestamp`: append timestamp suffix
- `vector_mode`: currently `edge_photo_v2`
- `save_debug_images`: write `*_mask.png` and `*_skeleton.png`
- `canny_low` / `canny_high`: set both to `-1` for auto thresholds
- `canny_sigma`: auto threshold sensitivity (smaller = tighter)
- `blur_ksize`: pre-blur to reduce noise (0 disables)
- `morph_ksize`: morphology kernel size
- `morph_close_iters`: connect broken edges (photo mode usually needs 1–3)
- `morph_open_iters`: remove specks (0–1 typical)
- `min_component_area_px`: remove tiny components (raise to reduce noise)
- `thinning_iters`: skeletonization iterations (increase if skeleton looks thick)
- `simplify_mm`: polyline simplification in mm
- `min_path_length_mm`: drop tiny paths in mm
- `threshold`: binary threshold for contour extraction
- `mm_per_pixel`: scale factor, image pixel -> mm
- `min_area_px`: ignore tiny contour noise
- `simplify_px`: contour simplification epsilon
- `save_preview_png`: also save source png
- `allow_dxf_fallback`: save DXF if DWG converter is missing
- `odafc_path` (optional): explicit ODA converter path

## Outputs

- `dwg_path`: generated DWG full path
- `status`: export status text

## CAD Compatibility

Generated DWG can be opened by AutoCAD and most CAD tools supporting ACAD2018 DWG.

## Notes

- For photo/render images, use the V2 parameters and inspect `*_mask.png` and `*_skeleton.png` to tune.
- Folder picker requires ComfyUI backend running on a desktop session.
- Example template: `{name}_{counter}` -> `comfy_vector_0001.dwg`
