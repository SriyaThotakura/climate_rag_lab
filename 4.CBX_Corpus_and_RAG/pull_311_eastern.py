import requests
import pandas as pd
import os
from pathlib import Path

os.makedirs('./cbx_corpus', exist_ok=True)

# Eastern CBE corridor zips — missing from current data
EASTERN_ZIPS = [
    '10454','10455','10456','10457',
    '10460','10461','10462','10473','10474'
]

COMPLAINT_TYPES = [
    'Noise - Vehicle',
    'Air Quality',
    'Noise - Street/Sidewalk',
    'Noise - Residential',
    'Sewer',
    'Blocked Driveway'
]

zip_filter  = " OR ".join(
    [f"incident_zip='{z}'" for z in EASTERN_ZIPS])
type_filter = " OR ".join(
    [f"complaint_type='{t}'" for t in COMPLAINT_TYPES])

url = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

all_frames = []
YEARS = ['2021','2022','2023','2024']

for year in YEARS:
    print(f"Pulling eastern corridor {year}...")
    params = {
        '$where': (
            f"borough='BRONX' AND "
            f"({zip_filter}) AND "
            f"({type_filter}) AND "
            f"created_date >= '{year}-01-01T00:00:00' AND "
            f"created_date <= '{year}-12-31T23:59:59'"
        ),
        '$limit': 50000,
        '$order': 'created_date ASC',
        '$select': (
            'unique_key,created_date,'
            'complaint_type,descriptor,'
            'incident_address,borough,'
            'incident_zip,latitude,longitude,status'
        )
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            print(f"  {year}: {len(df):,} rows")
            all_frames.append(df)
        else:
            print(f"  {year}: no data")
    except Exception as e:
        print(f"  {year}: failed -- {e}")

# Merge new eastern data
eastern = pd.concat(all_frames, ignore_index=True)
eastern = eastern.rename(columns={
    'unique_key':   'complaint_id',
    'incident_zip': 'zip_code'
})
eastern['latitude']  = pd.to_numeric(
    eastern['latitude'],  errors='coerce')
eastern['longitude'] = pd.to_numeric(
    eastern['longitude'], errors='coerce')
eastern = eastern.dropna(subset=['latitude','longitude'])

# Expanded bbox covering full CBE + Bruckner corridor
# West: Highbridge (-73.935)
# East: Soundview / Parkchester (-73.811)
# South: Hunts Point / Bruckner (40.800)
# North: Fordham / Tremont (40.865)
eastern = eastern[
    (eastern['latitude']  >= 40.800) &
    (eastern['latitude']  <= 40.865) &
    (eastern['longitude'] >= -73.935) &
    (eastern['longitude'] <= -73.811)
]
eastern['data_source'] = 'NYC_311_REAL_EASTERN_2021_2024'

# Load existing western data
existing_path = './cbx_corpus/311_complaints_real.csv'
existing = pd.read_csv(existing_path)

# Merge and deduplicate
combined = pd.concat([existing, eastern],
                      ignore_index=True)
combined = combined.drop_duplicates(
    subset=['complaint_id'])
combined = combined.sort_values('created_date')

combined.to_csv(existing_path, index=False)

print(f"\nCombined corpus: {len(combined):,} complaints")
print(f"   Western (original): {len(existing):,}")
print(f"   Eastern (new):      {len(eastern):,}")
print(f"\nZip distribution:")
print(combined['zip_code'].value_counts()
      .head(15).to_string())
print(f"\nLon range: {combined['longitude'].min():.4f}"
      f" to {combined['longitude'].max():.4f}")
print(f"Lat range: {combined['latitude'].min():.4f}"
      f" to {combined['latitude'].max():.4f}")
