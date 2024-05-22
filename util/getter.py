from collections import Counter
import matplotlib.pyplot as plt
import os, json,csv
import numpy as np


datadir = "/media/dxk/TOSHIBA EXT/ds/spotify_mpd/data"
data_path = os.path.join(datadir, 'data')

def get_playlist(file, idx):
    cur_path = os.path.join(datadir, file)
    res = None
    with open(cur_path, 'r') as f:
        j = json.load(f)

        res = j['playlists'][idx]

    return res

if __name__ == "__main__":
    res = get_playlist('mpd.slice.549000-549999.json', 333)
    print(res)
    res2 = get_playlist('mpd.slice.549000-549999.json', 793)
    print(res2)




