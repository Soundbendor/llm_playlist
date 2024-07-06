import os,csv
import routes as G
import pandas as pd
search_range = [-2,5]
d = {}
with open('diffres2.txt', 'r') as f:
    csvr = csv.DictReader(f)
    for r in csvr:
        if r['file'] not in d.keys():
            d[r['file']] = []
        d[r['file']].append((r['uri'],int(r['line'])))


dlist = []
for _f, farr in d.items():
    with open(os.path.join(G.data_dir, _f), 'r') as f:
        cur_lines = list(f.readlines())
        for uri,line in farr:
            found_artist = False
            artist = None
            for l in cur_lines[line-2:line]:
                if 'artist_name' in l:
                    found_artist=True
                    cur = l.strip().split(":")
                    #print(cur)
                    artist = cur[1].strip()[1:-2]
                    #print(artist)
                    break
            found_track = False
            track = None
            for l in cur_lines[line: line+6]:
                if 'track_name' in l:
                    found_track = True
                    cur = l.strip().split(":")
                    track = cur[1].strip()[1:-2]
            if found_artist == True and found_track == True:
                cur_id = uri.strip().split(':')[-1]
                cur_dict = {'artist_name': artist, 'track_name': track, 'id': cur_id, 'uri': uri}
                dlist.append(cur_dict)

df_new = pd.DataFrame.from_dict(dlist)
df_new = df_new.drop_duplicates()
df_old = pd.read_csv(G.joined_csv2_path, index_col=[0])
df_old2 = df_old[['artist_name', 'track_name', 'id', 'uri']]
df_out = pd.concat((df_new, df_old2))
df_out.to_csv('allsongs.csv')
#df_new.to_csv('diffsongs.csv')

