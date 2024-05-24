from collections import Counter
import matplotlib.pyplot as plt
import os, json,csv,sqlite3
import numpy as np
import glob as G

# joined.db new_combined_table has duplicates so need to use select distinct in queries
# for example: select * from new_combined_table where (artist_name="Prince") and track_name="Little Red Corvette";
# returns two entries

nct_feat = ["track_name", "artist_name", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "type", "id", "uri", "track_href", "analysis_url", "duration_ms", "time_signature"]
"""
example track
#{'track_name': 'Videotape', 'artist_name': 'Radiohead',
'danceability': 0.581, 'energy': 0.384, 'key': 9, 'loudness': -11.195,
'mode': 1, 'speechiness': 0.0336, 'acousticness': 0.697, 'instrumentalness': 0.813,
'liveness': 0.0889, 'valence': 0.0466, 'tempo': 77.412,
'type': 'audio_features', 'id': '3uxhyRdWVXp7GQvERQl6fA',
'uri': 'spotify:track:3uxhyRdWVXp7GQvERQl6fA',
'track_href': 'https://api.spotify.com/v1/tracks/3uxhyRdWVXp7GQvERQl6fA',
'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3uxhyRdWVXp7GQvERQl6fA',
'duration_ms': 279634, 'time_signature': 4}
"""

# features to compare by
comp_feat = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
comp_str = ','.join(comp_feat)
comp_query = f'select distinct {comp_str} from new_combined_table'

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

# get all = get all results, to_dict= return in dict form
def result_fetcher(res, get_all=False,to_dict=True):
    ret = {}
    if get_all == False:
        restup = res.fetchone()
        if (restup is None) == False:
            if to_dict == True:
                ret = nct_tup_to_dict(restup)
            else:
                ret = restup
    else:
        restups = res.fetchall()
        if to_dict == True:
            ret = [nct_tup_to_dict(x) for x in restups]
        else:
            ret = restups
    return ret

            

def get_features_by_id(cur,_id):
    res = cur.execute(f'SELECT DISTINCT * FROM new_combined_table WHERE id="{_id}"')
    return result_fetcher(res)

def get_features_by_artist_and_trackname(cur,_artist, _track):
    res = cur.execute(f'SELECT DISTINCT * FROM new_combined_table WHERE artist_name="{_artist}" AND track_name="{_track}"')
    return result_fetcher(res)

def get_features_by_artist(cur,_artist):
    res = cur.execute(f'SELECT DISTINCT * FROM new_combined_table WHERE artist_name="{_artist}"')
    return result_fetcher(res,get_all=True)


def get_track_uri_from_playlist(playlist):
    return [x['track_uri'] for x in playlist['tracks']]

def get_comp_feat_all_songs(cur):
    res = cur.execute(comp_query)
    return result_fetcher(res,get_all=True, to_dict = False)

def get_comp_feat_playlist(cur, playlist):
    songids = get_track_uri_from_playlist(playlist)
    songstr = '","'.join(songids)
    cur_q = f'select {comp_str} from new_combined_table where uri in ("{songstr}")'
    res = cur.execute(cur_q)
    return result_fetcher(res,get_all=True, to_dict = False)

if __name__ == "__main__":
    res = get_playlist('mpd.slice.549000-549999.json', 333)
    #print(res)
    res2 = get_playlist('mpd.slice.549000-549999.json', 793)
    r2tracks = get_track_uri_from_playlist(res2)
    #print(r2tracks)
    #print(res2)
    cnx, cur = connect_to_nct()
    res2f = get_comp_feat_playlist(cur,res2)
    #print(res2f)
    #resdict = get_features_by_id(cur, "6JHrzpRYiDx53iTgTbI76X")
    #resdict2 = get_features_by_id(cur, "2Viqjkxmiu8hGIhjwtqYvI")
    resdict3 = get_features_by_id(cur, "3uxhyRdWVXp7GQvERQl6fA")
    #print(resdict3)
    resarr = get_features_by_artist(cur, "Radiohead")
    #print(resarr)
    #print(resdict)
    #print(resdict2)





