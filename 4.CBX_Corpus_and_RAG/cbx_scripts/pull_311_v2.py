import requests
import pandas as pd
import os

os.makedirs('./cbx_corpus', exist_ok=True)

CBX_ZIPS = ['10451','10452','10453','10454','10455']

COMPLAINT_TYPES = [
    'Noise - Vehicle',
    'Air Quality',
    'Noise - Street/Sidewalk',
    'Noise - Residential',
    'Sewer',
    'Blocked Driveway'
]

zip_filter  = " OR ".join([f"incident_zip='{z}'"
                            for z in CBX_ZIPS])
type_filter = " OR ".join([f"complaint_type='{t}'"
                            for t in COMPLAINT_TYPES])

url = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

# Pull year by year so no single call hits the limit
YEARS = ['2021','2022','2023','2024']
all_frames = []

for year in YEARS:
    print(f"Pulling {year}...")
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
            'unique_key,'
            'created_date,'
            'complaint_type,'
            'descriptor,'
            'incident_address,'
            'borough,'
            'incident_zip,'
            'latitude,'
            'longitude,'
            'status'
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
            print(f"  {year}: no data or error — {data}")
    except Exception as e:
        print(f"  {year}: request failed — {e}")

# Merge all years
combined = pd.concat(all_frames, ignore_index=True)

combined = combined.rename(columns={
    'unique_key':   'complaint_id',
    'incident_zip': 'zip_code'
})

combined['latitude']  = pd.to_numeric(
    combined['latitude'],  errors='coerce')
combined['longitude'] = pd.to_numeric(
    combined['longitude'], errors='coerce')
combined = combined.dropna(subset=['latitude','longitude'])

# Strict CBE corridor bbox
combined = combined[
    (combined['latitude']  >= 40.828) &
    (combined['latitude']  <= 40.862) &
    (combined['longitude'] >= -73.935) &
    (combined['longitude'] <= -73.872)
]

# Remove duplicates across year pulls
combined = combined.drop_duplicates(subset=['complaint_id'])

combined['data_source'] = 'NYC_311_REAL_2021_2024'
combined = combined.sort_values('created_date')

out = './cbx_corpus/311_complaints_real.csv'
combined.to_csv(out, index=False)

print(f"\n✅ {len(combined):,} complaints → {out}")
print(f"\nDate range:")
print(f"  {combined['created_date'].min()[:10]} → "
      f"{combined['created_date'].max()[:10]}")
print(f"\nComplaint types:")
print(combined['complaint_type'].value_counts().to_string())
print(f"\nZip distribution:")
print(combined['zip_code'].value_counts().to_string())