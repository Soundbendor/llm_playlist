from collections import Counter
import matplotlib.pyplot as plt
import os, json,csv,sqlite3
import numpy as np
import glob as G

nct_feat = ["track_name", "artist_name", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "type", "id", "uri", "track_href", "analysis_url", "duration_ms", "time_signature"]

# new combined table tup to dictionary when using select *
def nct_tup_to_dict(nct_tup):
    return {x:y for (x,y) in zip(nct_feat, nct_tup)}


def get_playlist(file, idx):
    cur_path = os.path.join(G.data_dir, file)
    res = None
    with open(cur_path, 'r') as f:
        j = json.load(f)

        res = j['playlists'][idx]

    return res

# connect to new combined table
def connect_to_nct():
    cnx = sqlite3.connect(G.joined_db_path)
    cur = cnx.cursor()
    return cnx, cur

def result_fetcher(res, get_all=False):
    ret = {}
    if get_all == False:
        restup = res.fetchone()
        if (restup is None) == False:
            ret = nct_tup_to_dict(restup)
    else:
        restups = res.fetchall()
        ret = [nct_tup_to_dict(x) for x in restups]
    return ret

            

def get_features_by_id(cur,_id):
    res = cur.execute(f'SELECT * FROM new_combined_table WHERE id="{_id}"')
    return result_fetcher(res)

def get_features_by_artist_and_trackname(cur,_artist, _track):
    res = cur.execute(f'SELECT * FROM new_combined_table WHERE artist_name="{_artist}" AND track_name="{_track}"')
    return result_fetcher(res)

def get_features_by_artist(cur,_artist):
    res = cur.execute(f'SELECT * FROM new_combined_table WHERE artist_name="{_artist}"')
    return result_fetcher(res,get_all=True)


def get_song_ids_from_playlist(playlist):
    return [x['track_uri'] for x in playlist['tracks']]


if __name__ == "__main__":
    res = get_playlist('mpd.slice.549000-549999.json', 333)
    #print(res)
    res2 = get_playlist('mpd.slice.549000-549999.json', 793)
    r2tracks = get_song_ids_from_playlist(res2)
    print(r2tracks)
    #print(res2)
    cnx, cur = connect_to_nct()
    resdict = get_features_by_id(cur, "6JHrzpRYiDx53iTgTbI76X")
    resdict2 = get_features_by_id(cur, "2Viqjkxmiu8hGIhjwtqYvI")
    resarr = get_features_by_artist(cur, "Radiohead")
    #print(resarr)
    #print(resdict)
    #print(resdict2)





