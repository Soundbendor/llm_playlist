import os
import csv
import json
import glob
from collections import Counter
from tqdm import tqdm

import routes as G

exclude_pattern = "num_splits/num_tracks-{250,300,350,400}.csv"
file_pattern = os.path.join(G.num_tracks_path, "*.csv")
all_files = set(glob.glob(file_pattern)) - set(glob.glob(os.path.join(G.num_tracks_path, exclude_pattern)))

ctr = Counter()

# Load JSON data in advance if possible
playlist_data = {}
def load_playlist(plpath):
    if plpath not in playlist_data:
        try:
            with open(plpath, 'r') as f:
                playlist_data[plpath] = json.load(f)
        except Exception as e:
            print(f"Error loading {plpath}: {e}")
            playlist_data[plpath] = None
    return playlist_data[plpath]

for cur_path in tqdm(all_files, desc="Processing files"):
    with open(cur_path, 'r') as file:
        csv_reader = csv.DictReader(file)

        if 'file' not in csv_reader.fieldnames or 'idx' not in csv_reader.fieldnames:
            print(f"Missing required columns in {cur_path}")
            continue
        
        i = 0
        for row in csv_reader:
            plfile = row['file']
            plidx = int(row['idx'])
            plpath = os.path.join(G.data_dir, plfile)
            cur_json = load_playlist(plpath)
            if cur_json is None:
                continue

            cur_pl = cur_json['playlists'][plidx]
            ctr.update(trk['track_uri'] for trk in cur_pl['tracks'])
            i+=1
            if i % 1000 == 0:
                print(i)

# Write the popularity counts to file
output_file = os.path.join(G.num_tracks_path, 'stats/popularity.csv')
with open(output_file, 'w', newline='') as f:
    csvw = csv.writer(f)
    csvw.writerow(['uri', 'count'])
    csvw.writerows(ctr.items())

# ==========================================
#               GPT-3.5
# ==========================================
# Average R-Precision: 0.005194369973190343
# Average DCG: 0.3028171112828822
# Average IDCG: 1.5419510201942437
# Average NDCG: 0.10703044983499345
# Average Clicks: 33.34048257372654

# ==========================================
#               GPT-4
# ==========================================
# Average R-Precision: 0.046715817694369915
# Average DCG: 2.5058006113749993
# Average IDCG: 5.0890395306519265
# Average NDCG: 0.41323149562533745
# Average Clicks: 3.6380697050938338