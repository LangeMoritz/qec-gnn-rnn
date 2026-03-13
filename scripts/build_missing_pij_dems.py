"""Build p_ij DEMs for any patch that doesn't have one yet.

Uses r50 detection events for each patch (50k shots at 50 rounds).
Existing DEMs are skipped.  Run locally before submitting fine-tuning jobs.

Usage:
    python scripts/build_missing_pij_dems.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "p_ij_from_google_data"))

import stim
import numpy as np
from pathlib import Path
from build_dem_from_detection_events import build_dem_from_detection_events

DATA_DIR = (
    Path(__file__).parent.parent
    / "p_ij_from_google_data"
    / "2024_google_105Q_surface_code_d3_d5_d7"
)

patches = sorted(DATA_DIR.glob("d*_at_*"))
if not patches:
    raise FileNotFoundError(f"No patch directories found in {DATA_DIR}")

built = 0
skipped = 0
for patch_dir in patches:
    dem_path = patch_dir / "Z" / "r50" / "decoding_results" / "pij_model" / "error_model.dem"
    if dem_path.exists():
        print(f"  skip  {patch_dir.name}  (DEM already exists)")
        skipped += 1
        continue

    r50_dir = patch_dir / "Z" / "r50"
    circuit_path = r50_dir / "circuit_noisy_si1000.stim"
    det_path = r50_dir / "detection_events.b8"

    if not circuit_path.exists():
        print(f"  WARN  {patch_dir.name}  (no circuit_noisy_si1000.stim at r50, skipping)")
        continue
    if not det_path.exists():
        print(f"  WARN  {patch_dir.name}  (no detection_events.b8 at r50, skipping)")
        continue

    print(f"  build {patch_dir.name} ...", end="", flush=True)
    circuit = stim.Circuit.from_file(str(circuit_path))
    det_events = stim.read_shot_data_file(
        path=str(det_path),
        format="b8",
        bit_packed=False,
        num_measurements=circuit.num_detectors,
    )
    dem = build_dem_from_detection_events(circuit, np.asarray(det_events))
    dem_path.parent.mkdir(parents=True, exist_ok=True)
    dem.to_file(str(dem_path))
    print(f"  saved ({len(list(dem))} error instructions)")
    built += 1

print(f"\nDone: {built} built, {skipped} skipped.")
