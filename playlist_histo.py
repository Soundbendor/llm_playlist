from collections import Counter
import matplotlib.pyplot as plt
import os, json,csv
import numpy as np
datadir = "/media/dxk/TOSHIBA EXT/ds/spotify_mpd/data"
res_path = os.path.join(os.getcwd(), 'res')

def generate_chart(cur_ctr, title="Number of Playlists by Track Length", x_axis = "Track Length", y_axis="Number", out_basename="track_counts"):
    cur_x = np.array(list(cur_ctr.keys()), dtype=int)
    sort_idx = np.argsort(cur_x)
    x_sort = cur_x[sort_idx]
    cur_y = np.array(list(cur_ctr.values()), dtype=int)
    y_sort = cur_y[sort_idx]
    subp = plt.subplots()
    plt.bar(x_sort,y_sort)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(res_path, f"{out_basename}.png"))
    plt.clf()
    plt.close()
    with open(os.path.join(res_path, f"{out_basename}.csv"), 'w') as f:
        csvw = csv.writer(f)
        csvw.writerow(['length','count'])
        for l,c in zip(x_sort,y_sort):
            csvw.writerow([l,c])


if os.path.exists(res_path) == False:
    os.mkdir(res_path)

track_ctr = Counter()
album_ctr = Counter()
follow_ctr = Counter()
for i,mpdslice in enumerate(os.listdir(datadir)):
    cpath = os.path.join(datadir, mpdslice)
    nums = mpdslice.split('.')[-2]
    local_track = Counter()
    local_album = Counter()
    local_follow = Counter()
    with open(cpath, 'r') as f:
        j = json.load(f)
        #print(j.keys())
        for playlist in j['playlists']:
            #print(playlist['num_tracks'])
            cur_tracks = playlist['num_tracks']
            cur_follow = playlist['num_followers']
            cur_albums = playlist['num_albums']
            track_ctr.update([cur_tracks])
            album_ctr.update([cur_albums])
            follow_ctr.update([cur_follow])
            local_track.update([cur_tracks])
            local_album.update([cur_albums])
            local_follow.update([cur_follow])

    generate_chart(local_track, title=f"Number of Playlists by Track Length\nfor {nums}", x_axis = "Track Length", y_axis="Number", out_basename=f"track_counts_{nums}")
    generate_chart(local_album, title=f"Number of Playlists by Number of Albums\nfor {nums}", x_axis = "Number of Albums", y_axis="Number", out_basename=f"album_counts_{nums}")
    generate_chart(local_follow, title="fNumber of Playlists by Number of Followers\nfor {nums}", x_axis = "Number of Followers", y_axis="Number", out_basename=f"follow_counts_{nums}")


generate_chart(track_ctr, title="Number of Total Playlists by Track Length", x_axis = "Track Length", y_axis="Number", out_basename="track_counts_total")
generate_chart(album_ctr, title="Number of Total Playlists by Number of Albums", x_axis = "Number of Albums", y_axis="Number", out_basename="album_counts_total")
generate_chart(follow_ctr, title="Number of Total Playlists by Number of Followers", x_axis = "Number of Followers", y_axis="Number", out_basename="follow_counts_total")
