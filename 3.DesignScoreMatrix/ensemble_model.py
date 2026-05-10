"""
ensemble_model.py — Ensemble ML model for CBX environmental justice vulnerability.

Methodology grounded in:
  Paper 1 (SAGEPUB 23998083221083677): Urban morphology + ML for PM2.5
  Paper 2 (Frontiers feart.2021.659296): Ensemble stacking for spatial susceptibility

Uses REAL pre-computed GIS vulnerability data from 4 streams:
  Stream 1: RAG trauma scores (trauma_points.geojson)
  Stream 2: GIS vulnerability indicators (3 GeoJSON files)
  Stream 3: Vulnerable facilities proximity
  Stream 4: Distance to CBE

Target: Composite vulnerability score from real GIS data.

Usage:
    python ensemble_model.py
    python ensemble_model.py --data-dir ./cbx_corpus
    python ensemble_model.py --output-dir ./outputs
    python ensemble_model.py --with-cv
"""

import argparse
import csv
import json
import math
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import Point, shape
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════

CBE_WAYPOINTS = [
    (-73.9280, 40.8448),
    (-73.9127, 40.8489),
    (-73.8998, 40.8465),
    (-73.8876, 40.8420),
    (-73.8756, 40.8398),
    (-73.8634, 40.8376),
]

# Bronx bounding box for filtering facilities with BORO=None
BRONX_BBOX = {
    "lat_min": 40.78,
    "lat_max": 40.92,
    "lon_min": -73.94,
    "lon_max": -73.75,
}

ASTHMA_MAX = 355.8
CVI_RANGE = (-8.05, 2.67)

# Target weights
W_ASTHMA = 0.50
W_CVI = 0.30
W_TRAUMA = 0.20

# ZONEDIST mapping
ZONE_MAP = {
    "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5,
    "R6": 6, "R7": 7, "R8": 8, "R9": 9,
}


def parse_zone_dist(zd: str) -> int:
    """Extract base residential zone number from ZONEDIST string."""
    if not zd:
        return 0
    m = re.match(r"R(\d)", str(zd))
    return int(m.group(1)) if m else 0


def point_to_segment_dist_m(pt: tuple, seg_start: tuple, seg_end: tuple) -> float:
    """Minimum geodesic distance from point to a line segment (lon,lat tuples)."""
    ax, ay = seg_start
    bx, by = seg_end
    px, py = pt

    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return geodesic((py, px), (ay, ax)).meters

    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj = (ax + t * dx, ay + t * dy)
    return geodesic((py, px), (proj[1], proj[0])).meters


def dist_to_cbe(lon: float, lat: float) -> float:
    """Minimum distance in metres from (lon, lat) to any CBE segment."""
    min_d = float("inf")
    for i in range(len(CBE_WAYPOINTS) - 1):
        d = point_to_segment_dist_m(
            (lon, lat), CBE_WAYPOINTS[i], CBE_WAYPOINTS[i + 1]
        )
        min_d = min(min_d, d)
    return min_d


def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════════
# STREAM 1: RAG trauma scores
# ════════════════════════════════════════════

def compute_trauma_features(zone_polys: list, zone_fids: list, trauma_path: str) -> pd.DataFrame:
    """Point-in-polygon trauma density per intervention zone."""
    trauma = load_geojson(trauma_path)
    trauma_pts = []
    for feat in trauma["features"]:
        coords = feat["geometry"]["coordinates"]
        props = feat["properties"]
        trauma_pts.append({
            "point": Point(coords),
            "category": props.get("trauma_category", ""),
            "intensity": float(props.get("trauma_intensity", 0)),
            "geo_confidence": props.get("geo_confidence", ""),
        })

    records = []
    for i, poly in enumerate(zone_polys):
        inside = [t for t in trauma_pts if poly.contains(t["point"])]
        records.append({
            "fid": zone_fids[i],
            "trauma_noise_density": sum(1 for t in inside if t["category"] == "noise"),
            "trauma_health_density": sum(1 for t in inside if t["category"] == "health"),
            "trauma_air_density": sum(1 for t in inside if t["category"] == "air_quality"),
            "trauma_displacement_density": sum(1 for t in inside if t["category"] == "displacement"),
            "trauma_max_intensity": max((t["intensity"] for t in inside), default=0.0),
            "trauma_exact_count": sum(1 for t in inside if t["geo_confidence"] == "exact"),
        })

    return pd.DataFrame(records)


# ════════════════════════════════════════════
# STREAM 2: GIS vulnerability indicators
# ════════════════════════════════════════════

def compute_gis_features(
    zone_polys: list,
    zone_fids: list,
    zone_props: list,
    trivariate_path: str,
    residential_path: str,
) -> pd.DataFrame:
    """Spatial joins to trivariate and residential layers."""
    tri_data = load_geojson(trivariate_path)
    res_data = load_geojson(residential_path)

    # Build trivariate polygons
    tri_polys = []
    for feat in tri_data["features"]:
        poly = shape(feat["geometry"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        tri_polys.append((poly, feat["properties"]))

    # Build residential polygons
    res_polys = []
    for feat in res_data["features"]:
        poly = shape(feat["geometry"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        res_polys.append((poly, feat["properties"]))

    # Median canopy and HDS from trivariate for fallback
    tri_canopies = [p.get("Canopy_Pct_majority") for _, p in tri_polys if p.get("Canopy_Pct_majority") is not None]
    tri_hds_vals = [p.get("HDS_mean_majority") for _, p in tri_polys if p.get("HDS_mean_majority") is not None]
    median_canopy = float(np.median(tri_canopies)) if tri_canopies else 0.0
    median_hds = float(np.median(tri_hds_vals)) if tri_hds_vals else 0.0

    records = []
    for i, zone_poly in enumerate(zone_polys):
        row = {
            "fid": zone_fids[i],
            "Asthma_I_R_mean": zone_props[i].get("Asthma_I_R_mean", 0.0),
            "CVI_mean": zone_props[i].get("CVI_mean"),
        }

        # Trivariate overlap
        in_tri = False
        tri_canopy_val = None
        tri_hds_val = None
        for tri_poly, tri_props in tri_polys:
            if zone_poly.intersects(tri_poly):
                in_tri = True
                tri_canopy_val = tri_props.get("Canopy_Pct_majority")
                tri_hds_val = tri_props.get("HDS_mean_majority")
                break

        row["in_trivariate_zone"] = 1 if in_tri else 0
        row["trivariate_canopy"] = tri_canopy_val if tri_canopy_val is not None else median_canopy
        row["trivariate_hds"] = tri_hds_val if tri_hds_val is not None else median_hds

        # Residential majority overlap
        best_overlap = 0
        best_res_props = None
        for res_poly, res_props in res_polys:
            if zone_poly.intersects(res_poly):
                overlap = zone_poly.intersection(res_poly).area
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_res_props = res_props

        if best_res_props:
            row["canopy_pct"] = best_res_props.get("Canopy_Pct_majority", 0.0)
            row["hds_score"] = best_res_props.get("HDS_mean_majority", 0.0)
            row["zone_dist_int"] = parse_zone_dist(best_res_props.get("ZONEDIST", ""))
        else:
            row["canopy_pct"] = 0.0
            row["hds_score"] = 0.0
            row["zone_dist_int"] = 0

        records.append(row)

    df = pd.DataFrame(records)

    # Fill null CVI with median
    cvi_median = df["CVI_mean"].median()
    null_cvi = df["CVI_mean"].isna().sum()
    df["CVI_mean"] = df["CVI_mean"].fillna(cvi_median)
    print(f"  CVI_mean: {null_cvi} null values filled with median {cvi_median:.2f}")

    return df


# ════════════════════════════════════════════
# STREAM 3: Facilities proximity
# ════════════════════════════════════════════

def compute_facility_features(
    zone_centroids: list, zone_fids: list, facilities_path: str
) -> pd.DataFrame:
    """Distance to nearest hospital/school, facility count within 500m."""
    fac_data = load_geojson(facilities_path)

    # Filter to Bronx: use BORO field, but for BORO=None check bbox
    hospitals = []
    schools = []
    all_facs = []

    for feat in fac_data["features"]:
        if feat.get("geometry") is None:
            continue
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        if coords is None:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        boro = str(props.get("BORO", ""))
        ftype = str(props.get("f_type", ""))

        is_bronx = boro == "BRONX"
        if not is_bronx and boro in ("None", ""):
            # Check if within Bronx bbox
            is_bronx = (
                BRONX_BBOX["lat_min"] <= lat <= BRONX_BBOX["lat_max"]
                and BRONX_BBOX["lon_min"] <= lon <= BRONX_BBOX["lon_max"]
            )

        if not is_bronx:
            continue

        fac = (lat, lon)
        all_facs.append(fac)

        if ftype == "Hospital":
            hospitals.append(fac)
        if "School" in ftype or "school" in ftype:
            schools.append(fac)

    print(f"  Bronx facilities (bbox-filtered): {len(all_facs)} "
          f"(hospitals: {len(hospitals)}, schools: {len(schools)})")

    records = []
    for i, (clat, clon) in enumerate(zone_centroids):
        # Nearest hospital
        if hospitals:
            nearest_hosp = min(geodesic((clat, clon), h).meters for h in hospitals)
        else:
            nearest_hosp = 0.0

        # Nearest school
        if schools:
            nearest_sch = min(geodesic((clat, clon), s).meters for s in schools)
        else:
            nearest_sch = 0.0

        # Facilities within 500m
        count_500 = sum(
            1 for f in all_facs if geodesic((clat, clon), f).meters <= 500
        )

        records.append({
            "fid": zone_fids[i],
            "nearest_hospital_m": round(nearest_hosp, 1),
            "nearest_school_m": round(nearest_sch, 1),
            "facilities_500m": count_500,
        })

    return pd.DataFrame(records)


# ════════════════════════════════════════════
# STREAM 4: Distance to CBE
# ════════════════════════════════════════════

def compute_cbe_distance(zone_centroids: list, zone_fids: list) -> pd.DataFrame:
    records = []
    for i, (clat, clon) in enumerate(zone_centroids):
        records.append({
            "fid": zone_fids[i],
            "dist_to_cbe_m": round(dist_to_cbe(clon, clat), 1),
        })
    return pd.DataFrame(records)


# ════════════════════════════════════════════
# CV MORPHOLOGY (conditional)
# ════════════════════════════════════════════

def compute_cv_features(
    zone_centroids: list, zone_fids: list, cv_path: str
) -> pd.DataFrame | None:
    """Load CV segmentation metrics and spatial-join to zone centroids."""
    if not os.path.exists(cv_path):
        return None

    cv_df = pd.read_csv(cv_path)

    # Normalise coordinate column names
    col_map = {}
    if "lat" in cv_df.columns and "latitude" not in cv_df.columns:
        col_map["lat"] = "latitude"
    if "lon" in cv_df.columns and "longitude" not in cv_df.columns:
        col_map["lon"] = "longitude"
    if col_map:
        cv_df = cv_df.rename(columns=col_map)
        print(f"  CV column remap: {col_map}")

    # Compute concrete_ratio as residual (road/sidewalk/other hard surface)
    # The notebook's final pipeline (cell 4) only exported sky, building, vegetation.
    # Earlier cells computed concrete from ADE20K classes 0,3,6 but that was dropped.
    cv_df["concrete_ratio"] = (
        1.0 - cv_df["sky_ratio"] - cv_df["building_ratio"] - cv_df["vegetation_ratio"]
    ).clip(0, 1)
    print(f"  CV concrete_ratio (residual): mean={cv_df['concrete_ratio'].mean():.4f}, "
          f"range=[{cv_df['concrete_ratio'].min():.4f}, {cv_df['concrete_ratio'].max():.4f}]")

    required = {"latitude", "longitude", "sky_ratio", "vegetation_ratio", "concrete_ratio"}
    if not required.issubset(set(cv_df.columns)):
        print(f"  CV CSV missing columns: {required - set(cv_df.columns)}")
        return None

    print(f"  CV frames: {len(cv_df)}, lat [{cv_df['latitude'].min():.4f}, {cv_df['latitude'].max():.4f}], "
          f"lon [{cv_df['longitude'].min():.4f}, {cv_df['longitude'].max():.4f}]")

    MATCH_RADIUS_M = 2000
    cv_points = list(zip(cv_df["latitude"], cv_df["longitude"], cv_df.index))

    matched = 0
    records = []
    for i, (clat, clon) in enumerate(zone_centroids):
        # Find nearest CV frame within radius
        best_dist = float("inf")
        best_idx = None
        for plat, plon, idx in cv_points:
            d = geodesic((clat, clon), (plat, plon)).meters
            if d < best_dist:
                best_dist = d
                best_idx = idx

        row = {"fid": zone_fids[i]}
        if best_idx is not None and best_dist <= MATCH_RADIUS_M:
            r = cv_df.iloc[best_idx]
            sky = float(r["sky_ratio"])
            veg = float(r["vegetation_ratio"])
            con = float(r["concrete_ratio"])
            row["sky_ratio"] = sky
            row["vegetation_ratio"] = veg
            row["concrete_ratio"] = con
            row["canyon_ar"] = con / max(sky, 0.01)
            matched += 1
        else:
            row["sky_ratio"] = None
            row["vegetation_ratio"] = None
            row["concrete_ratio"] = None
            row["canyon_ar"] = None

        records.append(row)

    median_fill = len(zone_centroids) - matched
    print(f"  CV spatial join (radius={MATCH_RADIUS_M}m): "
          f"{matched} zones with real CV data, {median_fill} zones median-filled")

    df = pd.DataFrame(records)
    # Fill missing with median
    for col in ["sky_ratio", "vegetation_ratio", "concrete_ratio", "canyon_ar"]:
        med = df[col].median()
        null_count = df[col].isna().sum()
        df[col] = df[col].fillna(med)
        if null_count > 0:
            print(f"    {col}: {null_count} nulls filled with median {med:.4f}")

    return df


# ════════════════════════════════════════════
# SKY RATIO AGGREGATE FEATURES (2000m radius)
# ════════════════════════════════════════════

def compute_sky_ratio_features(
    zone_centroids: list, zone_fids: list, cv_path: str
) -> pd.DataFrame | None:
    """Mean sky_ratio and trench_enclosure for all CV frames within 2000m of each zone centroid."""
    if not os.path.exists(cv_path):
        return None

    df_cv = pd.read_csv(cv_path)

    # Normalise coordinate column names
    if "lat" in df_cv.columns and "latitude" not in df_cv.columns:
        df_cv = df_cv.rename(columns={"lat": "latitude"})
    if "lon" in df_cv.columns and "longitude" not in df_cv.columns:
        df_cv = df_cv.rename(columns={"lon": "longitude"})

    if "sky_ratio" not in df_cv.columns or "latitude" not in df_cv.columns:
        return None

    df_cv["trench_enclosure"] = 1.0 - df_cv["sky_ratio"]

    RADIUS_M = 2000
    zone_enclosure = []
    for clat, clon in zone_centroids:
        nearby = df_cv[
            df_cv.apply(
                lambda r: geodesic((clat, clon), (r["latitude"], r["longitude"])).meters < RADIUS_M,
                axis=1,
            )
        ]
        if len(nearby) > 0:
            zone_enclosure.append({
                "mean_sky_ratio": nearby["sky_ratio"].mean(),
                "mean_enclosure": nearby["trench_enclosure"].mean(),
                "min_sky_ratio": nearby["sky_ratio"].min(),
                "frame_count": len(nearby),
            })
        else:
            zone_enclosure.append({
                "mean_sky_ratio": df_cv["sky_ratio"].median(),
                "mean_enclosure": df_cv["trench_enclosure"].median(),
                "min_sky_ratio": df_cv["sky_ratio"].median(),
                "frame_count": 0,
            })

    cv_zone_df = pd.DataFrame(zone_enclosure)
    cv_zone_df["fid"] = zone_fids

    real_count = (cv_zone_df["frame_count"] > 0).sum()
    median_count = len(zone_fids) - real_count
    print(f"  Sky-ratio aggregate (radius={RADIUS_M}m): "
          f"{real_count} zones with real CV data, {median_count} zones median-filled")

    return cv_zone_df


# ════════════════════════════════════════════
# TARGET VARIABLE
# ════════════════════════════════════════════

def compute_target(df: pd.DataFrame) -> pd.Series:
    """Composite vulnerability score from real GIS data."""
    asthma_norm = df["Asthma_I_R_mean"].clip(0) / ASTHMA_MAX

    cvi_min, cvi_max = CVI_RANGE
    cvi_norm = (df["CVI_mean"] - cvi_min) / (cvi_max - cvi_min)
    cvi_norm = cvi_norm.clip(0, 1)

    trauma_intensity = df["trauma_max_intensity"].clip(0, 1)

    score = (asthma_norm * W_ASTHMA) + (cvi_norm * W_CVI) + (trauma_intensity * W_TRAUMA)
    return score


# ════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble ML model for CBX environmental justice vulnerability"
    )
    parser.add_argument("--data-dir", default="./cbx_corpus")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--with-cv", action="store_true",
                        help="Load CV morphology features if available")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 56)
    print("CBX ENSEMBLE VULNERABILITY MODEL")
    print("=" * 56)

    # ── Load intervention zones (spatial units) ──────
    print("\n[1/5] Loading intervention zones...")
    iz_path = data_dir / "Intervention_Zones_3D.geojson"
    iz_data = load_geojson(str(iz_path))

    zone_polys = []
    zone_fids = []
    zone_props = []
    zone_centroids = []
    zone_geoms = []

    for feat in iz_data["features"]:
        poly = shape(feat["geometry"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        zone_polys.append(poly)
        zone_fids.append(feat["properties"].get("fid"))
        zone_props.append(feat["properties"])
        centroid = poly.centroid
        zone_centroids.append((centroid.y, centroid.x))
        zone_geoms.append(feat["geometry"])

    print(f"  {len(zone_polys)} intervention zone polygons loaded")

    # ── Stream 1: Trauma features ────────────────────
    print("\n[2/5] Computing trauma features (Point-in-Polygon)...")
    trauma_path = output_dir / "trauma_points.geojson"
    trauma_df = compute_trauma_features(zone_polys, zone_fids, str(trauma_path))
    zones_with_trauma = (trauma_df["trauma_max_intensity"] > 0).sum()
    print(f"  {zones_with_trauma} zones have >= 1 trauma point, "
          f"{len(zone_polys) - zones_with_trauma} zones have zero")

    # ── Stream 2: GIS vulnerability ──────────────────
    print("\n[3/5] Computing GIS vulnerability features...")
    gis_df = compute_gis_features(
        zone_polys, zone_fids, zone_props,
        str(data_dir / "Secondary_Intervention_Residential_TRIVARIATE.geojson"),
        str(data_dir / "Secondary_Intervention_Residential.geojson"),
    )

    # ── Stream 3: Facilities proximity ───────────────
    print("\n[4/5] Computing facility proximity features...")
    fac_df = compute_facility_features(
        zone_centroids, zone_fids,
        str(data_dir / "Vulnerability_facilities.geojson"),
    )

    # ── Stream 4: CBE distance ───────────────────────
    print("\n[5/5] Computing distance to CBE...")
    cbe_df = compute_cbe_distance(zone_centroids, zone_fids)

    # ── Merge all streams ────────────────────────────
    print("\nMerging feature streams...")
    df = gis_df.merge(trauma_df, on="fid")
    df = df.merge(fac_df, on="fid")
    df = df.merge(cbe_df, on="fid")

    # ── CV morphology (conditional) ──────────────────
    cv_used = False
    # Check both possible locations
    cv_path = output_dir / "segmentation_metrics.csv"
    cv_path_alt = data_dir / "segmentation_metrics.csv"
    if not os.path.exists(cv_path) and os.path.exists(cv_path_alt):
        cv_path = cv_path_alt
    if args.with_cv or os.path.exists(cv_path):
        cv_df = compute_cv_features(zone_centroids, zone_fids, str(cv_path))
        if cv_df is not None:
            df = df.merge(cv_df, on="fid")
            cv_used = True
            print("  CV morphology features added (canyon_ar, sky/veg/concrete ratios)")
        else:
            print("  CV metrics not available -- running without morphology features.")
            print("  Run mapillary_pipeline.py to add them.")
    else:
        print("\n  CV metrics not available -- running without morphology features.")
        print("  Run mapillary_pipeline.py to add them.")

    # ── Sky-ratio aggregate features (standalone) ─
    sky_used = False
    sky_cv_path = cv_path if os.path.exists(cv_path) else cv_path_alt
    if os.path.exists(sky_cv_path):
        print("\n  Computing sky-ratio aggregate features (2000m radius)...")
        sky_df = compute_sky_ratio_features(zone_centroids, zone_fids, str(sky_cv_path))
        if sky_df is not None:
            df = df.merge(sky_df, on="fid")
            sky_used = True
            print("  Sky-ratio aggregate features added (mean_sky_ratio, mean_enclosure, min_sky_ratio, frame_count)")
    else:
        print("\n  Sky-ratio aggregate features skipped -- no CV data available.")

    # ── Target variable ──────────────────────────────
    print("\n" + "=" * 56)
    print("TARGET VARIABLE: Composite Vulnerability Score")
    print("=" * 56)
    print(f"\n  vulnerability_score = ")
    print(f"    (Asthma_I_R_mean / {ASTHMA_MAX}) * {W_ASTHMA:.2f} +")
    print(f"    (CVI_mean_normalised)            * {W_CVI:.2f} +")
    print(f"    (trauma_max_intensity)            * {W_TRAUMA:.2f}")
    print()
    print("  Weight justification:")
    print("  Asthma inpatient rate (50%): direct health outcome, from DOHMH real data")
    print("  Climate vulnerability index (30%): composite environmental risk, from GIS analysis")
    print("  RAG trauma intensity (20%): community-reported evidence of harm, from cbx_trauma")

    df["vulnerability_score"] = compute_target(df)
    print(f"\n  Score range: [{df['vulnerability_score'].min():.4f}, "
          f"{df['vulnerability_score'].max():.4f}]")
    print(f"  Score mean:  {df['vulnerability_score'].mean():.4f}")
    print(f"  Score std:   {df['vulnerability_score'].std():.4f}")

    # ── Feature matrix ───────────────────────────────
    core_features = [
        "dist_to_cbe_m",
        "Asthma_I_R_mean",
        "CVI_mean",
        "nearest_hospital_m",
        "nearest_school_m",
        "facilities_500m",
        "trauma_noise_density",
        "trauma_health_density",
        "trauma_air_density",
        "trauma_displacement_density",
        "trauma_max_intensity",
        "trauma_exact_count",
        "in_trivariate_zone",
        "trivariate_canopy",
        "trivariate_hds",
        "canopy_pct",
        "hds_score",
        "zone_dist_int",
    ]

    cv_features = []
    if cv_used:
        cv_features = ["sky_ratio", "vegetation_ratio", "concrete_ratio", "canyon_ar"]

    sky_features = []
    if sky_used:
        sky_features = ["mean_sky_ratio", "mean_enclosure", "min_sky_ratio", "frame_count"]

    feature_cols = core_features + cv_features + sky_features
    X = df[feature_cols].copy()
    y = df["vulnerability_score"].copy()

    # Replace any remaining NaN with 0
    X = X.fillna(0)

    print(f"\n  Feature matrix: {X.shape[0]} zones x {X.shape[1]} features")

    # ── Train/test split ─────────────────────────────
    print("\n" + "=" * 56)
    print("ENSEMBLE MODELS")
    print("=" * 56)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    print(f"\n  Train: {len(X_train)} zones, Test: {len(X_test)} zones")

    # ── Model definitions ────────────────────────────
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=0
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        ),
    }

    cv_scores = {}
    test_preds = {}
    test_metrics = {}

    print(f"\n  {'Model':<22} {'CV R2 (mean +/- std)':>22} {'Test R2':>10} {'Test RMSE':>10}")
    print("  " + "-" * 66)

    for name, model in models.items():
        # 5-fold CV on training set
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        cv_scores[name] = scores

        # Fit on full training set
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        test_preds[name] = preds

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        test_metrics[name] = {"r2": r2, "rmse": rmse}

        print(f"  {name:<22} {scores.mean():>8.4f} +/- {scores.std():.4f}  "
              f"{r2:>10.4f} {rmse:>10.4f}")

    # ── Ensemble: weighted mean ──────────────────────
    cv_r2_vals = {name: max(cv_scores[name].mean(), 0.001) for name in models}
    total_w = sum(cv_r2_vals.values())
    weights = {name: cv_r2_vals[name] / total_w for name in models}

    ensemble_preds_test = np.zeros(len(y_test))
    for name in models:
        ensemble_preds_test += weights[name] * test_preds[name]

    ens_r2 = r2_score(y_test, ensemble_preds_test)
    ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds_test))

    print(f"\n  {'Weighted Ensemble':<22} {'':>22} {ens_r2:>10.4f} {ens_rmse:>10.4f}")
    print(f"\n  Ensemble weights: " +
          " | ".join(f"{n}: {w:.3f}" for n, w in weights.items()))

    # ── Full dataset predictions ─────────────────────
    # Retrain on all data for final predictions
    all_preds = {}
    for name, model in models.items():
        model.fit(X, y)
        all_preds[name] = model.predict(X)

    ensemble_full = np.zeros(len(y))
    for name in models:
        ensemble_full += weights[name] * all_preds[name]
    df["predicted_vulnerability"] = ensemble_full
    df["rf_prediction"] = all_preds["Random Forest"]
    df["xgb_prediction"] = all_preds["XGBoost"]
    df["gb_prediction"] = all_preds["Gradient Boosting"]

    # ── Intervention priority ranking ────────────────
    df["intervention_priority"] = df["vulnerability_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    # ── Feature importance ───────────────────────────
    print("\n" + "=" * 56)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("=" * 56)

    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({
        "feature_name": feature_cols,
        "rf_importance": importances,
    })
    imp_df = imp_df.sort_values("rf_importance", ascending=False).reset_index(drop=True)
    imp_df["rank"] = imp_df.index + 1

    # Feature groups
    def get_group(feat_name):
        if feat_name.startswith("trauma"):
            return "trauma"
        if feat_name in ("nearest_hospital_m", "nearest_school_m", "facilities_500m"):
            return "facilities"
        if feat_name == "dist_to_cbe_m":
            return "distance"
        if feat_name in ("sky_ratio", "vegetation_ratio", "concrete_ratio", "canyon_ar",
                         "mean_sky_ratio", "mean_enclosure", "min_sky_ratio", "frame_count"):
            return "cv"
        if feat_name == "zone_dist_int":
            return "zoning"
        return "gis"

    imp_df["feature_group"] = imp_df["feature_name"].apply(get_group)

    print(f"\n  {'Rank':<5} {'Feature':<35} {'Importance':>10} {'Group':<10}")
    print("  " + "-" * 62)
    for _, row in imp_df.iterrows():
        print(f"  {row['rank']:<5} {row['feature_name']:<35} "
              f"{row['rf_importance']:>10.4f} {row['feature_group']:<10}")

    # ── Novel finding check ──────────────────────────
    print("\n" + "=" * 56)
    print("NOVEL FINDING CHECK")
    print("=" * 56)

    trauma_feature_names = [
        "trauma_noise_density", "trauma_health_density",
        "trauma_air_density", "trauma_displacement_density",
        "trauma_max_intensity",
    ]
    trauma_total_imp = imp_df[
        imp_df["feature_name"].isin(trauma_feature_names)
    ]["rf_importance"].sum()

    print(f"\n  Trauma features total importance: {trauma_total_imp:.1%}")
    if trauma_total_imp > 0.20:
        print(f"\n  NOVEL FINDING: RAG trauma scores contribute {trauma_total_imp:.1%} of "
              f"model predictive power -- community-reported evidence adds significant "
              f"explanatory value beyond physical and health indicators alone.")
    else:
        print(f"\n  RAG trauma contribution: {trauma_total_imp:.1%} -- physical/health "
              f"indicators dominate as expected from literature. Trauma scores serve "
              f"as corroborating evidence.")

    # ── Top priority zones ───────────────────────────
    print("\n" + "=" * 56)
    print("TOP 5 PRIORITY INTERVENTION ZONES")
    print("=" * 56)

    top5 = df.nsmallest(5, "intervention_priority")
    for _, row in top5.iterrows():
        print(f"\n  Zone fid={row['fid']} (priority #{int(row['intervention_priority'])})")
        print(f"    Asthma rate:    {row['Asthma_I_R_mean']:.1f}")
        print(f"    CVI:            {row['CVI_mean']:.2f}")
        print(f"    Trauma max:     {row['trauma_max_intensity']:.2f}")
        print(f"    Vulnerability:  {row['vulnerability_score']:.4f}")
        print(f"    Predicted:      {row['predicted_vulnerability']:.4f}")

    # ════════════════════════════════════════════
    # OUTPUT FILES
    # ════════════════════════════════════════════

    # File 1: ensemble_predictions.geojson
    print("\n" + "=" * 56)
    print("WRITING OUTPUTS")
    print("=" * 56)

    out_features = []
    for i, row in df.iterrows():
        props = {col: (None if pd.isna(row[col]) else row[col]) for col in df.columns if col != "fid"}
        props["fid"] = row["fid"]
        props["data_source"] = "ensemble_model_v1_real_gis"
        props["cv_used"] = cv_used
        # Convert numpy types for JSON serialization
        for k, v in props.items():
            if isinstance(v, (np.integer,)):
                props[k] = int(v)
            elif isinstance(v, (np.floating,)):
                props[k] = round(float(v), 6)

        out_features.append({
            "type": "Feature",
            "geometry": zone_geoms[i],
            "properties": props,
        })

    geojson_out = {
        "type": "FeatureCollection",
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_zones": len(out_features),
            "model": "weighted_ensemble_rf_xgb_gb",
            "cv_used": cv_used,
            "target": "composite_vulnerability_score",
            "ensemble_test_r2": round(ens_r2, 4),
            "ensemble_test_rmse": round(ens_rmse, 4),
        },
        "features": out_features,
    }

    geojson_path = output_dir / "ensemble_predictions.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_out, f, indent=2, ensure_ascii=False)
    print(f"\n  [1] {geojson_path} ({len(out_features)} zones)")

    # File 2: feature_importance.csv
    imp_path = output_dir / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"  [2] {imp_path} ({len(imp_df)} features)")

    # File 3: model_summary.txt
    summary_path = output_dir / "model_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 56 + "\n")
        f.write("CBX ENSEMBLE VULNERABILITY MODEL SUMMARY\n")
        f.write("=" * 56 + "\n\n")

        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")

        f.write("DATASET\n")
        f.write(f"  Zones: {len(df)}\n")
        f.write(f"  Features: {len(feature_cols)}\n")
        f.write(f"  CV morphology: {'yes' if cv_used else 'no'}\n")
        f.write(f"  Train/test split: {len(X_train)}/{len(X_test)}\n\n")

        f.write("TARGET: Composite Vulnerability Score\n")
        f.write(f"  = (Asthma_I_R_mean / {ASTHMA_MAX}) * {W_ASTHMA} "
                f"+ CVI_norm * {W_CVI} + trauma_max * {W_TRAUMA}\n")
        f.write(f"  Range: [{df['vulnerability_score'].min():.4f}, "
                f"{df['vulnerability_score'].max():.4f}]\n\n")

        f.write("  Weight justification:\n")
        f.write("  Asthma inpatient rate (50%): direct health outcome, from DOHMH real data\n")
        f.write("  Climate vulnerability index (30%): composite environmental risk, from GIS analysis\n")
        f.write("  RAG trauma intensity (20%): community-reported evidence of harm, from cbx_trauma\n\n")

        f.write("MODEL PERFORMANCE\n")
        f.write(f"  {'Model':<22} {'CV R2 (mean +/- std)':>22} {'Test R2':>10} {'Test RMSE':>10}\n")
        f.write("  " + "-" * 66 + "\n")
        for name in models:
            sc = cv_scores[name]
            tm = test_metrics[name]
            f.write(f"  {name:<22} {sc.mean():>8.4f} +/- {sc.std():.4f}  "
                    f"{tm['r2']:>10.4f} {tm['rmse']:>10.4f}\n")
        f.write(f"  {'Weighted Ensemble':<22} {'':>22} {ens_r2:>10.4f} {ens_rmse:>10.4f}\n")
        f.write(f"\n  Ensemble weights: " +
                " | ".join(f"{n}: {w:.3f}" for n, w in weights.items()) + "\n\n")

        f.write("FEATURE IMPORTANCE (Top 10)\n")
        for _, row in imp_df.head(10).iterrows():
            f.write(f"  {row['rank']:<3} {row['feature_name']:<35} "
                    f"{row['rf_importance']:.4f}  ({row['feature_group']})\n")

        f.write(f"\nNOVEL FINDING CHECK\n")
        f.write(f"  Trauma features total importance: {trauma_total_imp:.1%}\n")
        if trauma_total_imp > 0.20:
            f.write(f"  NOVEL FINDING: RAG trauma scores contribute {trauma_total_imp:.1%} of "
                    f"model predictive power -- community-reported evidence adds significant "
                    f"explanatory value beyond physical and health indicators alone.\n")
        else:
            f.write(f"  RAG trauma contribution: {trauma_total_imp:.1%} -- physical/health "
                    f"indicators dominate as expected from literature. Trauma scores serve "
                    f"as corroborating evidence.\n")

        f.write(f"\nTOP 5 PRIORITY INTERVENTION ZONES\n")
        for _, row in top5.iterrows():
            f.write(f"\n  Zone fid={row['fid']} (priority #{int(row['intervention_priority'])})\n")
            f.write(f"    Asthma rate:    {row['Asthma_I_R_mean']:.1f}\n")
            f.write(f"    CVI:            {row['CVI_mean']:.2f}\n")
            f.write(f"    Trauma max:     {row['trauma_max_intensity']:.2f}\n")
            f.write(f"    Vulnerability:  {row['vulnerability_score']:.4f}\n")
            f.write(f"    Predicted:      {row['predicted_vulnerability']:.4f}\n")

    print(f"  [3] {summary_path}")

    # ── Matplotlib visualisations ────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.tree import plot_tree

    # Decision tree from first RF estimator
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        rf_model.estimators_[0],
        feature_names=feature_cols,
        filled=True,
        max_depth=4,
        ax=ax,
        fontsize=7,
        class_names=None,
    )
    plt.title("RF Decision Tree -- CBX Vulnerability Model")
    plt.tight_layout()
    tree_path = output_dir / "rf_decision_tree.png"
    plt.savefig(str(tree_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [4] {tree_path}")

    # Feature correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = pd.DataFrame(X_train.values, columns=feature_cols).corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        annot_kws={"size": 7},
        linewidths=0.3,
    )
    plt.title("Feature Correlation Matrix -- CBX Ensemble")
    plt.tight_layout()
    corr_path = output_dir / "feature_correlation_heatmap.png"
    plt.savefig(str(corr_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [5] {corr_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
