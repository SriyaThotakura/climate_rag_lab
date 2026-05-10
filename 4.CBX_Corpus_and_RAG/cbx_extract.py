"""
cbx_extract.py — Query cbx_trauma collection and produce trauma_points.geojson.

Usage:
    python cbx_extract.py
    python cbx_extract.py --collection cbx_trauma
    python cbx_extract.py --chroma_dir ./chroma_cbx
    python cbx_extract.py --output ./outputs/trauma_points.geojson
    python cbx_extract.py --min_intensity 0.1
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "climate_rag_lab"))
from rag import LocalEmbedder, ensure_collection

# ── defaults ────────────────────────��─────────────────────────────��─────────────
COLLECTION_NAME = "cbx_trauma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 20

TRAUMA_QUERIES = {
    "noise": [
        "noise from highway trucks cars engines",
        "can't sleep loud traffic rumbling constant",
    ],
    "air_quality": [
        "diesel fumes exhaust smell pollution highway",
        "air quality breathing difficult asthma fumes",
    ],
    "health": [
        "asthma respiratory hospital emergency sick",
        "children health breathing problems PM2.5",
    ],
    "displacement": [
        "moved away forced out housing demolished",
        "community destroyed neighbourhood changed expressway",
    ],
}

# ── geolocation anchors ───────────────────────────���────────────────────────────
CBX_ANCHORS = {
    "10451": (-73.9249, 40.8173),
    "10452": (-73.9249, 40.8351),
    "10453": (-73.9101, 40.8513),
    "10454": (-73.9068, 40.8076),
    "10455": (-73.9101, 40.8076),
}

BRONX_CENTER = (-73.9050, 40.8448)
CBE_CENTROID = (-73.9050, 40.8400)


def geolocate(meta, text):
    """Return (lon, lat, geo_confidence) using priority cascade."""
    # Priority 1: exact lat/lon from metadata
    lat = meta.get("lat", "")
    lon = meta.get("lon", "")
    if lat and lon:
        try:
            lat_f = float(lat)
            lon_f = float(lon)
            if lat_f != 0.0 and lon_f != 0.0:
                return lon_f, lat_f, "exact"
        except (ValueError, TypeError):
            pass

    # Priority 2: zip_code anchor
    zip_code = str(meta.get("zip_code", ""))
    if zip_code in CBX_ANCHORS:
        anchor_lon, anchor_lat = CBX_ANCHORS[zip_code]
        jitter_lon = np.random.uniform(-0.003, 0.003)
        jitter_lat = np.random.uniform(-0.003, 0.003)
        return anchor_lon + jitter_lon, anchor_lat + jitter_lat, "zip_anchor"

    # Priority 3: "bronx" in chunk text
    if "bronx" in text.lower():
        jitter_lon = np.random.uniform(-0.008, 0.008)
        jitter_lat = np.random.uniform(-0.008, 0.008)
        return BRONX_CENTER[0] + jitter_lon, BRONX_CENTER[1] + jitter_lat, "bronx_anchor"

    # Priority 4: default centroid
    jitter_lon = np.random.uniform(-0.012, 0.012)
    jitter_lat = np.random.uniform(-0.012, 0.012)
    return CBE_CENTROID[0] + jitter_lon, CBE_CENTROID[1] + jitter_lat, "centroid_fallback"


def run_extraction(chroma_dir: str, collection_name: str, output_path: str, min_intensity: float):
    embedder = LocalEmbedder(EMBED_MODEL)
    collection = ensure_collection(chroma_dir, collection_name)

    count = collection.count()
    print(f"Collection: {collection_name} ({count} chunks)")
    print(f"Output:     {output_path}")
    print(f"Min intensity: {min_intensity}")
    print()

    # Collect all features keyed by text_excerpt for dedup
    features_by_excerpt = {}  # text_excerpt -> feature dict
    category_counts = {cat: 0 for cat in TRAUMA_QUERIES}
    geo_counts = {"exact": 0, "zip_anchor": 0, "bronx_anchor": 0, "centroid_fallback": 0}

    for category, queries in TRAUMA_QUERIES.items():
        for query in queries:
            query_embedding = embedder.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K,
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, distance in zip(docs, metas, distances):
                similarity = 1.0 - float(distance)
                trauma_intensity = (similarity - 0.3) / 0.7
                trauma_intensity = max(0.0, min(1.0, trauma_intensity))

                if trauma_intensity < min_intensity:
                    continue

                text_excerpt = doc[:200]
                lon, lat, geo_confidence = geolocate(meta, doc)

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(lon, 6), round(lat, 6)],
                    },
                    "properties": {
                        "trauma_category": category,
                        "trauma_intensity": round(trauma_intensity, 2),
                        "source_type": meta.get("source_type", ""),
                        "source_file": meta.get("source_file", ""),
                        "geo_confidence": geo_confidence,
                        "text_excerpt": text_excerpt,
                        "query_used": query,
                        "zip_code": meta.get("zip_code", "") or None,
                    },
                }

                # Dedup: keep higher intensity
                if text_excerpt in features_by_excerpt:
                    existing = features_by_excerpt[text_excerpt]
                    if feature["properties"]["trauma_intensity"] > existing["properties"]["trauma_intensity"]:
                        # Remove old counts
                        old_cat = existing["properties"]["trauma_category"]
                        old_geo = existing["properties"]["geo_confidence"]
                        category_counts[old_cat] -= 1
                        geo_counts[old_geo] -= 1
                        # Replace
                        features_by_excerpt[text_excerpt] = feature
                        category_counts[category] += 1
                        geo_counts[geo_confidence] += 1
                else:
                    features_by_excerpt[text_excerpt] = feature
                    category_counts[category] += 1
                    geo_counts[geo_confidence] += 1

    features = list(features_by_excerpt.values())

    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_features": len(features),
            "collection": collection_name,
            "trauma_categories": dict(category_counts),
        },
        "features": features,
    }

    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(geojson, fh, indent=2, ensure_ascii=False)

    print(f"DONE: trauma_points.geojson: {len(features)} features")
    cat_line = " | ".join(f"{k}: {v}" for k, v in category_counts.items())
    print(f"   {cat_line}")
    print(f"   geo_confidence breakdown:")
    geo_line = " | ".join(f"{k}: {v}" for k, v in geo_counts.items())
    print(f"   {geo_line}")


def main():
    parser = argparse.ArgumentParser(description="Extract trauma points from cbx_trauma collection")
    parser.add_argument("--collection", type=str, default="cbx_trauma")
    parser.add_argument("--chroma_dir", type=str, default="./chroma_cbx")
    parser.add_argument("--output", type=str, default="./outputs/trauma_points.geojson")
    parser.add_argument("--min_intensity", type=float, default=0.1)
    args = parser.parse_args()

    run_extraction(args.chroma_dir, args.collection, args.output, args.min_intensity)


if __name__ == "__main__":
    main()
