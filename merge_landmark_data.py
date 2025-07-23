import os
import csv

# Directory where individual gesture CSVs are stored
directory = './landmark_data'
output_file = 'landmark_data.csv'

all_rows = []

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        label = filename.replace('.csv', '')
        path = os.path.join(directory, filename)
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Remove empty strings and commas
                row = [val.strip() for val in row if val.strip() != '']
                # Remove trailing commas (can cause extra columns)
                row = [v for v in row if v not in [',', ',\n']]
                # Ensure all except last value are numeric
                numeric = []
                for v in row:
                    try:
                        float(v)
                        numeric.append(v)
                    except:
                        pass
                # Only use row if it has the right number of floats
                if len(numeric) >= 63:
                    numeric.append(label)
                    all_rows.append(numeric)

# Write clean merged file
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_rows)

print(f"Merged {len(all_rows)} rows into {output_file}")
