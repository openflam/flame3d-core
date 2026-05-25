"""
Serve the Objects Inventory Matrix visualization.

Usage:
    python serve.py --dataset ProjectLabStudio_inv_method
    python serve.py --dataset ProjectLabStudio_inv_method --port 8765 --cell-size 14

The server serves:
  /              -> index.html (the visualization)
  /api/data      -> processed inventory JSON
  /images/<name> -> frame images
"""

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import urllib.parse

from config_io import get_images_output_path, get_output_path

BASE_DIR = Path(__file__).parent


def normalize_object(obj: str) -> str:
    """Strip punctuation and lowercase an object label."""
    return obj.strip().rstrip(".,;:!?").lower()


def build_inventory_response(dataset: str) -> dict:
    json_path = get_output_path(dataset) / "objects_inventory" / "objects_to_frames.json"
    with open(json_path, "r") as f:
        raw = json.load(f)

    # raw is {object: [[frame, ...], [frame, ...], ...]}
    # Flatten and normalize: object -> set of frames
    obj_to_frames: dict[str, set[str]] = {}
    for obj, frame_groups in raw.items():
        norm_obj = normalize_object(obj)
        if norm_obj not in obj_to_frames:
            obj_to_frames[norm_obj] = set()
        for group in frame_groups:
            obj_to_frames[norm_obj].update(group)

    # Collect all frames
    all_frames: set[str] = set()
    for frames_set in obj_to_frames.values():
        all_frames.update(frames_set)
    frames = sorted(all_frames)

    # Invert to frame -> [objects]
    normalized: dict[str, list[str]] = {f: [] for f in frames}
    for obj, frames_set in obj_to_frames.items():
        for frame in frames_set:
            normalized[frame].append(obj)
    for frame in normalized:
        normalized[frame] = sorted(normalized[frame])

    # Sort objects by frequency descending (most common first), then alphabetically
    freq: dict[str, int] = {obj: len(fs) for obj, fs in obj_to_frames.items()}
    unique_objects = sorted(obj_to_frames.keys(), key=lambda o: (-freq[o], o))

    return {
        "dataset": dataset,
        "frames": frames,
        "objects": unique_objects,
        "freq": {obj: freq[obj] for obj in unique_objects},
        "data": normalized,
    }


class Handler(BaseHTTPRequestHandler):
    dataset: str = ""
    _cached_data: dict | None = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(BASE_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/api/data":
            self._serve_json()
        elif path.startswith("/images/"):
            fname = path[len("/images/") :]
            img_path = get_images_output_path(self.dataset) / fname
            self._serve_file(img_path, "image/jpeg")
        else:
            self.send_error(404, f"Not found: {path}")

    def _serve_file(self, fpath: Path, content_type: str):
        try:
            data = fpath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404, f"File not found: {fpath}")

    def _serve_json(self):
        if Handler._cached_data is None:
            Handler._cached_data = build_inventory_response(self.dataset)
        payload = json.dumps(Handler._cached_data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        # Only log errors
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(
        description="Serve the Objects Inventory Matrix visualization"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (folder under outputs/ and data/)",
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port to listen on (default: 8765)"
    )
    args = parser.parse_args()

    # Validate paths
    out_path = (
        get_output_path(args.dataset) / "objects_inventory" / "objects_to_frames.json"
    )
    if not out_path.exists():
        print(f"ERROR: Inventory file not found: {out_path}")
        return 1
    img_dir = get_images_output_path(args.dataset)
    if not img_dir.exists():
        print(f"WARNING: Images directory not found: {img_dir}")
    else:
        print(f"Images: {img_dir}")

    Handler.dataset = args.dataset
    Handler._cached_data = None

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://0.0.0.0:{args.port}"
    print(f"\nObjects Inventory Visualization")
    print(f"  Dataset : {args.dataset}")
    print(f"  URL     : {url}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
