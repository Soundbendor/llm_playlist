import os,csv,sys
import routes as G
import getter as UG
import pandas as pd

# using ';' as a sep (delimiter arg in csv module) and '|' as a quotechar due to issues with track names such as
#I Don't Want to Miss a Thing - From the Touchstone film, "Armageddon"
cols = ['artist_name', 'track_name', 'track_uri']
df = pd.DataFrame(columns=cols)
runidx = 0
seen_uri = set()

out_file = os.path.join(G.db_dir, 'song_db.csv')
for ij,j in enumerate(os.listdir(G.data_dir)):
    """
    if ij >= 1:
        break
    """
    print(f"processing {j}")
    cur_j = UG.get_playlist_json(j)
    cur_pls = cur_j['playlists']
    for ipl, pl in enumerate(cur_pls):
        cur_trx = pl['tracks']
        for cur_trk in cur_trx:
            cur_uri = cur_trk['track_uri']
            if cur_uri not in seen_uri:
                to_insert = [cur_trk[x] for x in cols]
                df.loc[runidx] = to_insert
                seen_uri.add(cur_uri)
                runidx += 1

df.to_csv(out_file, sep=';', index=False, quotechar='|')
