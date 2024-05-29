import numpy as np
import os,csv,json
import glob as G
from collections import Counter
exclude = ['num_tracks-250.csv', 'num_tracks-300.csv', 'num_tracks-350.csv','num_tracks-400.csv'] 

ctr = Counter()
for _i,_f in enumerate(os.listdir(G.num_tracks_path)):
    if _f not in exclude:
        cur_path = os.path.join(G.num_tracks_path,_f)
        with open(cur_path, 'r') as _f2:
            csvr = csv.DictReader(_f2)
            for _j,row in enumerate(csvr):
                plfile = row['file']
                plidx = int(row['idx'])
                plpath = os.path.join(G.data_dir, plfile)
                with open(plpath, 'r') as f:
                    cur_json = json.load(f)
                    cur_pl = cur_json['playlists'][plidx]
                    trk_ids = [trk['track_uri'].strip().split(':')[-1] for trk in cur_pl['tracks']]
                    ctr.update(trk_ids)


with open(os.path.join(G.num_tracks_path, 'popularity.csv'), 'w') as f:
    csvw = csv.writer(f, delimiter=',')
    csvw.writerow(['id','count'])
    for (x,y) in ctr.items():
        csvw.writerow([x,y])
    
