import json
import csv

input_json = r"C:\Users\shahr\OneDrive\Desktop\Self Study\Sign-Language\data\raw\WLASL_v0.3.json"
output_csv = r"C:\Users\shahr\OneDrive\Desktop\Self Study\Sign-Language\data\labels.csv"

with open(input_json, 'r') as f:
    data = json.load(f)

rows = []

for entry in data:
    gloss = entry['gloss']
    for inst in entry['instances']:
        video_id = inst['video_id']
        rows.append((video_id, gloss))

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['video_id', 'gloss'])
    writer.writerows(rows)

print(f"Saved {len(rows)} video label mappings to {output_csv}")
