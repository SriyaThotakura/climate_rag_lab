import pandas as pd

df = pd.read_csv('./cbx_corpus/311_complaints_real.csv')

# Keep all Air Quality — only 214, every one matters
aq = df[df['complaint_type'] == 'Air Quality']

# Keep all Sewer — only 901
sewer = df[df['complaint_type'] == 'Sewer']

# Sample Noise - Vehicle proportionally by year
# 1,000 per year = 4,000 total, spatially diverse
noise_vehicle = (df[df['complaint_type'] == 'Noise - Vehicle']
    .groupby(df['created_date'].str[:4])
    .apply(lambda x: x.sample(
        min(len(x), 1000), random_state=42))
    .reset_index(drop=True))

# Sample Noise - Street proportionally
# 500 per year = 2,000 total
noise_street = (df[df['complaint_type'] 
                   == 'Noise - Street/Sidewalk']
    .groupby(df['created_date'].str[:4])
    .apply(lambda x: x.sample(
        min(len(x), 500), random_state=42))
    .reset_index(drop=True))

# Skip Noise - Residential and Blocked Driveway
# Not directly relevant to highway exposure argument

focused = pd.concat([aq, sewer, 
                     noise_vehicle, noise_street],
                    ignore_index=True)
focused = focused.drop_duplicates(subset=['complaint_id'])
focused = focused.sort_values('created_date')

out = './cbx_corpus/311_complaints_real.csv'
focused.to_csv(out, index=False)

print(f"Focused corpus: {len(focused):,} complaints")
print(f"\nComplaint types:")
print(focused['complaint_type'].value_counts().to_string())
print(f"\nYear distribution:")
print(focused['created_date'].str[:4]
      .value_counts().sort_index().to_string())