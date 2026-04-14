import importlib.util
import os
import subprocess
import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PLUGIN_DIR / "requirements.txt"
WEB_DIRECTORY = "./web"


def _is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _ensure_runtime_dependencies() -> None:
    needed = {
        "cv2": "opencv-python",
        "ezdxf": "ezdxf",
        "PIL": "Pillow",
    }
    missing = [pkg for mod, pkg in needed.items() if not _is_module_available(mod)]
    if not missing:
        return
    if not REQUIREMENTS_FILE.exists():
        raise RuntimeError(
            f"[comfyui_dwg_exporter] Missing dependencies {missing}, and requirements.txt not found."
        )

    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "[comfyui_dwg_exporter] Failed to auto-install dependencies.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def _register_folder_picker_route() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return

    routes = PromptServer.instance.routes

    @routes.get("/cad_dwg_exporter/select_folder")
    async def cad_dwg_exporter_select_folder(_request):
        selected = ""
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(
                title="Select output folder for DWG export"
            )
            root.destroy()
        except Exception as ex:
            return web.json_response(
                {"ok": False, "error": f"Folder picker failed: {ex}"},
                status=500,
            )

        return web.json_response({"ok": True, "path": selected or ""})

    @routes.get("/cad_dwg_exporter/select_save_path")
    async def cad_dwg_exporter_select_save_path(_request):
        selected = ""
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.asksaveasfilename(
                title="Save DWG as",
                defaultextension=".dwg",
                filetypes=[("DWG files", "*.dwg"), ("All files", "*.*")],
            )
            root.destroy()
        except Exception as ex:
            return web.json_response(
                {"ok": False, "error": f"Save dialog failed: {ex}"},
                status=500,
            )

        if not selected:
            return web.json_response({"ok": True, "path": "", "dir": "", "name": ""})

        selected_dir = os.path.dirname(selected)
        selected_name = os.path.splitext(os.path.basename(selected))[0]
        return web.json_response(
            {"ok": True, "path": selected, "dir": selected_dir, "name": selected_name}
        )


_ensure_runtime_dependencies()
_register_folder_picker_route()

from .dwg_export_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
