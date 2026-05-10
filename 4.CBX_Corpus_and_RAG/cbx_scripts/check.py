import pandas as pd
df = pd.read_csv('./cbx_corpus/311_complaints_real.csv')
print(f"Total rows: {len(df):,}")
print(f"Date range: {df['created_date'].min()} → {df['created_date'].max()}")
print(f"\nComplaint types:")
print(df['complaint_type'].value_counts())
print(f"\nZip distribution:")
print(df['zip_code'].value_counts())
print(f"\nRows with coordinates: {df[['latitude','longitude']].dropna().shape[0]:,}")