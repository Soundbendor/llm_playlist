from collections import Counter
import matplotlib.pyplot as plt
import os, json,csv,sqlite3
import numpy as np
import pandas as pd
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
#comp_feat = ['danceability', 'energy', 'key', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
#comp_feat = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# predefining so don't have to run every time
non_comp_feat = [x for x in nct_feat if x not in comp_feat]

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

def get_features_by_id(cnx,_id):
    return pd.read_sql(f'SELECT DISTINCT * FROM new_combined_table WHERE id="{_id}"', cnx)


def get_features_by_artist_and_trackname(cnx,_artist, _track):
    return pd.read_sql(f'SELECT DISTINCT * FROM new_combined_table WHERE artist_name="{_artist}" AND track_name="{_track}"', cnx)

def get_features_by_artist(cnx,_artist, group = True):
    if group == True:
        return pd.read_sql(f'SELECT DISTINCT * FROM new_combined_table WHERE artist_name="{_artist}" GROUP BY artist_name,track_name ORDER BY duration_ms', cnx)
    else:
        return pd.read_sql(f'SELECT DISTINCT * FROM new_combined_table WHERE artist_name="{_artist}"', cnx)


def get_track_uri_from_playlist(playlist):
    return [x['track_uri'] for x in playlist['tracks']]

def get_feat_all_songs(cnx):
    return pd.read_sql('select distinct * from new_combined_table', cnx)

def get_feat_playlist(cnx, playlist):
    songids = get_track_uri_from_playlist(playlist)
    songstr = '","'.join(songids)
    cur_q = f'select distinct * from new_combined_table where uri in ("{songstr}")'
    return pd.read_sql(cur_q, cnx)

if __name__ == "__main__":
    res = get_playlist('mpd.slice.549000-549999.json', 333)
    #print(res)
    res2 = get_playlist('mpd.slice.549000-549999.json', 793)
    r2tracks = get_track_uri_from_playlist(res2)
    #print(r2tracks)
    #print(res2)
    cnx, cur = connect_to_nct()
    res2f = get_feat_playlist(cnx,res2)
    #print(res2f)
    #resdict = get_features_by_id(cur, "6JHrzpRYiDx53iTgTbI76X")
    #resdict2 = get_features_by_id(cur, "2Viqjkxmiu8hGIhjwtqYvI")
    resdict3 = get_features_by_id(cnx, "3uxhyRdWVXp7GQvERQl6fA")
    #print(resdict3)
    resarr = get_features_by_artist(cnx, "Radiohead")
    #print(resarr)
    #print(resdict)
    #print(resdict2)





