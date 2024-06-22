from collections import Counter,defaultdict
import matplotlib.pyplot as plt
import os, json,csv,sqlite3
import numpy as np
import pandas as pd
import routes as G
import sklearn.preprocessing as SKP
import sklearn.decomposition as SKD
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
#comp_feat = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
comp_feat = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
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

def get_features_by_ids(cnx, _ids):
    idstr = "(" + f"{_ids}"[1:-1] + ")"
    q =f"select distinct * from new_combined_table where id in {idstr}"
    return pd.read_sql(q, cnx)

def get_features_by_artist_and_trackname(cnx,_artist, _track):
    return pd.read_sql(f'SELECT DISTINCT * FROM new_combined_table WHERE artist_name="{_artist}" AND track_name="{_track}"', cnx)

def get_features_by_artist(cnx,_artist, group = True):
    if group == True:
        return pd.read_sql(f'SELECT DISTINCT * FROM new_combined_table WHERE artist_name="{_artist}" GROUP BY artist_name,track_name ORDER BY duration_ms ASC', cnx)
    else:
        return pd.read_sql(f'SELECT DISTINCT * FROM new_combined_table WHERE artist_name="{_artist}"', cnx)


def get_features_by_trackname(cnx,_track, group = True):
    if group == True:
        return pd.read_sql(f'SELECT DISTINCT * FROM new_combined_table WHERE track_name="{_track}" GROUP BY artist_name,track_name ORDER BY duration_ms ASC', cnx)
    else:
        return pd.read_sql(f'SELECT DISTINCT * FROM new_combined_table WHERE track_name="{_track}"', cnx)

def get_feat_from_uris(cnx, uris):
    ids = [x.split(":")[-1] for x in uris]
    idstr = '","'.join(ids)
    cur_q = f'select distinct * from new_combined_table where id in ("{idstr}")'
    return pd.read_sql(cur_q, cnx)

def get_track_uri_from_playlist(playlist):
    return [x['track_uri'] for x in playlist['tracks']]

def get_feat_all_songs(cnx):
    return pd.read_sql('select distinct * from new_combined_table', cnx)

def get_feat_playlist(cnx, playlist):
    songids = get_track_uri_from_playlist(playlist)
    songstr = '","'.join(songids)
    cur_q = f'select distinct * from new_combined_table where uri in ("{songstr}")'
    return pd.read_sql(cur_q, cnx)



# input is a dataframe with all features, (numpy) features and scaler
def all_songs_tx(df, normalize=True, pca = 3, seed=5):
    mmscl = None
    np_all_feat = None
    if normalize == True:
        mmscl = SKP.MinMaxScaler()
        np_all_feat = mmscl.fit_transform(df[comp_feat].to_numpy())
    else:
        np_all_feat = df[comp_feat].to_numpy()
    pcaer = None
    if pca > 0:
        pcaer = SKD.PCA(n_components=pca, whiten=True, random_state=seed)
        np_all_feat = pcaer.fit_transform(np_all_feat)
    txdict = defaultdict(lambda: None)
    txdict['scaler'] = mmscl
    txdict['pca'] = pcaer
    return np_all_feat, txdict

# pl_file is json file 
def get_playlist_json(pl_file):
    cpath = os.path.join(G.data_dir, pl_file)
    ret = None
    with open(cpath, 'r') as f:
        ret = json.load(f)
    return ret
    
# pl_csv is a csv listing playlists
# csv_path is the path to csvs
def playlist_csv_generator(pl_csv, csv_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data')):
    with open(os.path.join(csv_path, pl_csv), 'r') as f:
        csvr = csv.DictReader(f)
        for row in csvr:
            yield row

# input: all songs feature df
# returns all songs feature df with 'count' column
# default_count: default to put in case of missing count
def add_pop_to_feat(df, pop_path=os.path.join(G.num_tracks_path, 'stats', 'popularity_trimmed.csv'), default_count = 1):
    pop_df = pd.read_csv(pop_path)
    df = pd.merge(df, pop_df, how='left')
    df.loc[df['count'].isna(), 'count'] = default_count
    return df

if __name__ == "__main__":
    res = get_playlist('mpd.slice.549000-549999.json', 333)
    #print(res)
    res2 = get_playlist('mpd.slice.549000-549999.json', 793)
    r2tracks = get_track_uri_from_playlist(res2)
    #print(r2tracks)
    #print(res2)
    #cnx, cur = connect_to_nct()
    #_df = get_feat_all_songs(cnx)
    #_df = add_pop_to_feat(_df)
    #print(_df)
    #res2f = get_feat_playlist(cnx,res2)
    #print(res2f)
    #resdict = get_features_by_id(cur, "6JHrzpRYiDx53iTgTbI76X")
    #resdict2 = get_features_by_id(cur, "2Viqjkxmiu8hGIhjwtqYvI")
    #resdict3 = get_features_by_id(cnx, "3uxhyRdWVXp7GQvERQl6fA")
    #print(resdict3)
    #resarr = get_features_by_artist(cnx, "Radiohead")
    #print(resarr)
    #print(resdict)
    #print(resdict2)





