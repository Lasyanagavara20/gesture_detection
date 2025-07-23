import os
import pandas as pd

DATA_DIR = 'landmark_data'
merged_file = 'landmark_data.csv'

all_data = []

for file in os.listdir(DATA_DIR):
    if file.endswith('.csv'):
        filepath = os.path.join(DATA_DIR, file)
        df = pd.read_csv(filepath)
        all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(merged_file, index=False)
    print(f"Merged {len(all_data)} files into {merged_file}")
else:
    print("No CSV files found to merge.")
