import os
import csv
import json
import glob
from collections import Counter
from tqdm import tqdm

import routes as G

data_path = "data/combined.csv"

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


with open(data_path, 'r') as file:
    csv_reader = csv.DictReader(file)
    
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
