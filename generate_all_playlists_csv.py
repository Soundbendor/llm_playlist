import os,csv
import getter as UG
import routes as G

x = os.listdir(G.data_dir)
header = ['file', 'idx', 'pid']
with open(os.path.join(G.data_dir2, 'all_playlists.csv'), 'w') as f:
    csvw = csv.DictWriter(f, fieldnames=header)
    csvw.writeheader()
    for _i,_f in enumerate(x):
        plj = UG.get_playlist_json(_f)
        for _j,playlist in enumerate(plj['playlists']):

            cfile = _f
            cidx = _j
            cur_pid = playlist['pid']
            cdict = {'file': cfile, 'idx': cidx, 'pid': cur_pid}
            csvw.writerow(cdict)            

