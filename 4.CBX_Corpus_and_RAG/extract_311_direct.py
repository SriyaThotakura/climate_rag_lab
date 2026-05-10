"""
extract_311_direct.py — Direct 311 complaint extraction to GeoJSON.

Reads 311_complaints_real.csv and maps georeferenced complaints
into trauma_points_311.geojson with trauma categories derived from
complaint_type. No RAG or embedding involved.

With 200K+ complaints the full dataset is too large for browser
rendering, so spatial grid-sampling keeps ~5,000 representative
points while preserving geographic spread across the corridor.

Usage:
    python extract_311_direct.py
    python extract_311_direct.py --max-points 8000
    python extract_311_direct.py --input ./cbx_corpus/311_complaints_real.csv
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Map 311 complaint_type to trauma category
COMPLAINT_TO_CATEGORY = {
    "Noise - Street/Sidewalk": "noise",
    "Noise - Vehicle": "noise",
    "Noise - Residential": "noise",
    "Air Quality": "air_quality",
    "Sewer": "health",
    "Blocked Driveway": "displacement",
}

# Grid cell size in degrees (~100m)
GRID_STEP = 0.001


def _grid_key(lon: float, lat: float) -> tuple:
    return (round(lon / GRID_STEP), round(lat / GRID_STEP))


def extract(input_path: str, output_path: str, max_points: int = 5000):
    all_rows = []

    with open(input_path, encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lat_str = row.get("latitude", "")
            lon_str = row.get("longitude", "")
            if not lat_str or not lon_str:
                continue
            try:
                lat = float(lat_str)
                lon = float(lon_str)
            except ValueError:
                continue
            if lat == 0.0 or lon == 0.0:
                continue
            all_rows.append((lat, lon, row))

    print(f"Total georeferenced complaints: {len(all_rows):,}")

    # Spatial grid sampling: bucket by grid cell, keep up to N per cell
    grid = defaultdict(list)
    for lat, lon, row in all_rows:
        grid[_grid_key(lon, lat)].append((lat, lon, row))

    # Calculate per-cell budget to hit max_points
    n_cells = len(grid)
    per_cell = max(1, max_points // n_cells) if n_cells > 0 else 1

    sampled = []
    for cell_rows in grid.values():
        if len(cell_rows) <= per_cell:
            sampled.extend(cell_rows)
        else:
            sampled.extend(random.sample(cell_rows, per_cell))

    # If still over budget, random subsample
    if len(sampled) > max_points:
        sampled = random.sample(sampled, max_points)

    print(f"Grid cells: {n_cells:,}, per-cell budget: {per_cell}, sampled: {len(sampled):,}")

    features = []
    category_counts = {"noise": 0, "air_quality": 0, "health": 0, "displacement": 0}

    for lat, lon, row in sampled:
        complaint_type = row.get("complaint_type", "")
        category = COMPLAINT_TO_CATEGORY.get(complaint_type, "health")
        descriptor = row.get("descriptor", "")
        address = row.get("incident_address", "")
        zip_code = row.get("zip_code", "")
        date = row.get("created_date", "")[:10]

        text = f"{complaint_type}: {descriptor}. Address: {address}, {zip_code}. Filed: {date}"

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(lon, 6), round(lat, 6)],
            },
            "properties": {
                "trauma_category": category,
                "trauma_intensity": 0.65,
                "source_type": "311_complaint",
                "source_file": Path(input_path).name,
                "geo_confidence": "exact",
                "text_excerpt": text[:200],
                "query_used": "direct_311_extract",
                "zip_code": zip_code or None,
            },
        }
        features.append(feature)
        category_counts[category] += 1

    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_features": len(features),
            "total_complaints": len(all_rows),
            "source": "311_complaints_real.csv (direct extract, grid-sampled)",
            "trauma_categories": category_counts,
        },
        "features": features,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(geojson, fh, indent=2, ensure_ascii=False)

    print(f"DONE: {out.name}: {len(features)} features (from {len(all_rows):,} total)")
    cat_line = " | ".join(f"{k}: {v}" for k, v in category_counts.items())
    print(f"   {cat_line}")

    # Zip distribution
    from collections import Counter
    zips = Counter(f["properties"]["zip_code"] for f in features if f["properties"]["zip_code"])
    print(f"   Zip spread: {dict(sorted(zips.items(), key=lambda x: -x[1]))}")


def main():
    parser = argparse.ArgumentParser(description="Direct 311 to GeoJSON extraction")
    parser.add_argument("--input", default="./cbx_corpus/311_complaints_real.csv")
    parser.add_argument("--output", default="./outputs/trauma_points_311.geojson")
    parser.add_argument("--max-points", type=int, default=5000,
                        help="Max features in output (grid-sampled)")
    args = parser.parse_args()
    extract(args.input, args.output, args.max_points)


if __name__ == "__main__":
    main()
