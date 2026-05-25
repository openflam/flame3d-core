import json
import time
from datetime import datetime
from pathlib import Path

from config_io import get_output_path

def save_runtime_stats(dataset_name: str, step_name: str, step_stats: dict) -> None:
    """
    Save or append runtime statistics for a specific pipeline step.
    
    Args:
        dataset_name: Name of the dataset
        step_name: Name of the step (e.g. "0_identify_frames")
        step_stats: Dictionary containing 'duration_seconds', 'status', and optionally 'parameters'
    """
    outputs_dir = get_output_path(dataset_name=dataset_name)
    stats_path = outputs_dir / "runtime_stats.json"
    
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
    else:
        stats = {
            "dataset_name": dataset_name,
            "pipeline_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "steps": {},
            "total_duration_seconds": 0.0,
            "total_duration_minutes": 0.0,
            "pipeline_end_time": ""
        }
    
    stats["steps"][step_name] = step_stats
    
    # Recalculate total duration
    total_sec = sum(step_info.get("duration_seconds", 0) for step_info in stats["steps"].values())
    stats["total_duration_seconds"] = total_sec
    stats["total_duration_minutes"] = total_sec / 60.0
    stats["pipeline_end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
