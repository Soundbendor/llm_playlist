from collections import Counter
import matplotlib.pyplot as plt
import os, json,csv
import numpy as np
datadir = "/media/dxk/TOSHIBA EXT/ds/spotify_mpd/data"
data_path = os.path.join(os.getcwd(), 'data')

track_bins = {x: [] for x in range(50,450,50)}

if os.path.exists(data_path) == False:
    os.mkdir(data_path)

outkeys = ['name', 'num_tracks', 'idx', 'file', 'pid', 'modified_at', 'collaborative', 'num_albums', 'num_followers']

def get_playlist_info(cur_file, playlist, pl_idx):
    cur_name = playlist['name']
    cur_collab = playlist['collaborative']
    cur_pid = playlist['pid']
    cur_tracks = playlist['num_tracks']
    cur_albums = playlist['num_albums']
    cur_follow = playlist['num_followers']
    cur_mod = playlist['modified_at']
    return {'name': cur_name, 'idx': pl_idx, 'file': cur_file, 'pid': cur_pid, 'num_tracks': cur_tracks,
            'modified_at': cur_mod, 'collaborative': cur_collab, 'num_albums': cur_albums,
            'num_followers': cur_follow}

def get_bin(playlist):
    cur_tracks = playlist['num_tracks']
    cur_bin = int(cur_tracks//50)*50
    return cur_bin

for i,mpdslice in enumerate(os.listdir(datadir)):
    cpath = os.path.join(datadir, mpdslice)
    nums = mpdslice.split('.')[-2]
    local_track = Counter()
    local_album = Counter()
    local_follow = Counter()
    with open(cpath, 'r') as f:
        j = json.load(f)
        #print(j.keys())
        for pl_idx, playlist in enumerate(j['playlists']):
            #print(playlist['num_tracks'])
            cur_bin = get_bin(playlist)
            cur_info = get_playlist_info(mpdslice,playlist, pl_idx)
            if cur_bin not in track_bins.keys():
                track_bins[cur_bin] = []
            track_bins[cur_bin].append((playlist['num_tracks'], playlist['pid'], cur_info))

for cur_bin in track_bins.keys():
    cur_f = f"num_tracks-{cur_bin}.csv"
    track_bins[cur_bin].sort()
    with open(os.path.join(data_path, cur_f), 'w') as f:
        csvw = csv.DictWriter(f, fieldnames=outkeys)
        csvw.writeheader()
        for (_,_,cur_info) in track_bins[cur_bin]:
            csvw.writerow(cur_info)



        
