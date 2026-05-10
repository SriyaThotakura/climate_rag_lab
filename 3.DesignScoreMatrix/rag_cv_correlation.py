import geopandas as gpd
import pandas as pd
import json
from geopy.distance import geodesic

# Load trauma points
with open('./outputs/trauma_points.geojson', encoding='utf-8') as f:
    trauma = json.load(f)

# Load CV frames
df_cv = pd.read_csv(
    './cbx_corpus/segmentation_metrics.csv')
df_cv['trench_enclosure'] = 1.0 - df_cv['sky_ratio']

# For each trauma point find nearest
# CV frame within 500m
results = []
for feat in trauma['features']:
    t_lon = feat['geometry']['coordinates'][0]
    t_lat = feat['geometry']['coordinates'][1]
    props = feat['properties']

    best_dist = float('inf')
    best_cv = None

    for _, cv_row in df_cv.iterrows():
        d = geodesic(
            (t_lat, t_lon),
            (cv_row['lat'], cv_row['lon'])
        ).meters
        if d < best_dist:
            best_dist = d
            best_cv = cv_row

    if best_dist < 2000 and best_cv is not None:
        results.append({
            'trauma_category': props['trauma_category'],
            'trauma_intensity': props['trauma_intensity'],
            'geo_confidence':   props['geo_confidence'],
            'nearest_cv_m':     round(best_dist, 1),
            'sky_ratio':        round(float(best_cv['sky_ratio']),4),
            'trench_enclosure': round(float(best_cv['trench_enclosure']),4),
            'canyon_ar':        round(float(best_cv['building_ratio'] /
                                max(best_cv['sky_ratio'], 0.01)),4)
        })

df_results = pd.DataFrame(results)

print("=== RAG + CV CORRELATION ===")
print(f"Trauma points with CV match (<2000m): "
      f"{len(results)}")

if len(results) == 0:
    print("\nNo spatial overlap found. CV frames and trauma points are too far apart.")
    import sys; sys.exit(0)

print(f"\nMean enclosure near trauma points:")
print(df_results.groupby('trauma_category')
      ['trench_enclosure'].mean().round(3))
print(f"\nMean enclosure near exact 311 points:")
exact = df_results[
    df_results['geo_confidence']=='exact']
print(f"  {exact['trench_enclosure'].mean():.3f}")
print(f"\nKey finding:")
high_trauma = df_results[
    df_results['trauma_intensity'] > 0.7]
low_sky = df_results[
    df_results['sky_ratio'] < 0.3]
print(f"  High-intensity trauma in enclosed "
      f"zones (<0.3 sky): {len(low_sky)}")
print(f"  High-intensity trauma total: "
      f"{len(high_trauma)}")

df_results.to_csv(
    './outputs/rag_cv_correlation.csv',
    index=False)
print("\nSaved: rag_cv_correlation.csv")
