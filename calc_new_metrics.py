import os
import csv
import json
import glob
from collections import Counter
from tqdm import tqdm

import routes as G

exclude_pattern = "num_tracks-{250,300,350,400}.csv"
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
        csvr = csv.DictReader(file)
        i = 0
        for row in csvr:
            plfile = row['file']
            plidx = int(row['idx'])
            plpath = os.path.join(G.data_dir, plfile)
            cur_json = load_playlist(plpath)
            if cur_json is None:
                continue

            cur_pl = cur_json['playlists'][plidx]
            trk_ids = [trk['track_uri'] for trk in cur_pl['tracks']]
            ctr.update(trk_ids)
            i+=1
            if i % 1000 == 0:
                print(i)

# Write the popularity counts to file
output_file = os.path.join(G.num_tracks_path, 'popularity.csv')
with open(output_file, 'w', newline='') as f:
    csvw = csv.writer(f)
    csvw.writerow(['id', 'count'])
    csvw.writerows(ctr.items())
