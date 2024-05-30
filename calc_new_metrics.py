import os
import csv
import json
import glob
import time
from collections import Counter
from tqdm import tqdm

import routes as G

exclude_pattern = "num_tracks-{250,300,350,400}.csv"
file_pattern = os.path.join(G.num_tracks_path, "*.csv")
all_files = set(glob.glob(file_pattern)) - set(glob.glob(os.path.join(G.num_tracks_path, exclude_pattern)))

ctr = Counter()

for cur_path in tqdm(all_files, desc="Processing files"):
    with open(cur_path, 'r') as file:
        csvr = csv.DictReader(file)
        i = 0
        for row in csvr:
            plfile = row['file']
            plidx = int(row['idx'])
            plpath = os.path.join(G.data_dir, plfile)
            try:
                with open(plpath, 'r') as f:
                    cur_json = json.load(f)
                    cur_pl = cur_json['playlists'][plidx]
                    trk_ids = [trk['track_uri'] for trk in cur_pl['tracks']]
                    ctr.update(trk_ids)
            except json.JSONDecodeive:
                print(f"Error decoding JSON from {plpath}")
            except FileNotFoundError:
                print(f"File not found: {plpath}")
            except Exception as e:
                print(f"Unexpected error: {e}, file: {plpath}")
            
            # Logging
            i+=1
            if i % 100 == 0:
                print(i/100)
                print(f"time:  {time.time()}")
                # break

# Write the popularity counts to file
output_file = os.path.join(G.num_tracks_path, 'popularity.csv')
with open(output_file, 'w', newline='') as f:
    csvw = csv.writer(f)
    csvw.writerow(['id', 'count'])
    csvw.writerows(ctr.items())
