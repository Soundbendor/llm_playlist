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

chall_file = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data', 'valid_challenges.csv')

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

chall_header = ['challenge','file', 'file_idx', 'start_idx','end_idx','has_title','num_cond','random']
chall_bool = ['has_title', 'random']
chall_str = ['file']
def get_challenges():
    ret = []
    with open(chall_file, 'r') as f:
        csvr = csv.DictReader(f)
        for row in csvr:
            cur_d = {}
            for k,v in row.items():
                if k in chall_str:
                    cur_d[k] = f'{v}.csv'
                elif k in chall_bool:
                    cur_d[k] = int(v) > 0.5
                else:
                    cur_d[k] = int(v)
            ret.append(cur_d)
    return ret

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

def df_get_artists(df):
    return df.drop_duplicates(subset=['artist_name'])['artist_name'].values

def df_get_tracks(df):
    return df.drop_duplicates(subset=['track_name'])['track_name'].values


def df_filter_by_uris(df, uris):
    return df.loc[df['uri'].isin(uris)].reset_index()

def df_filter_by_uri_file(df, urifile, uri_dir = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data')):
    trk = []
    with open(os.path.join(uri_dir, urifile), 'r') as f:
        trk = set([x.strip() for x in f.readlines()])
    return df_filter_by_uris(df, trk)



# input is a dataframe with all features, (numpy) features and scaler
# filter_by_train means only return training songs
def all_songs_tx(df, normalize=True, train_uri_file = '', train_uri_dir =  os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data'), filter_by_train = False, pca = 3, seed=5):
    mmscl = None
    np_all_feat = None
    train_df = None
    train_filt = False
    if len(train_uri_file) > 0:
        train_filt = True
        train_df = df_filter_by_uri_file(df, train_uri_file, uri_dir = train_uri_dir) 
    if normalize == True:
        mmscl = SKP.MinMaxScaler()
        if train_filt == False:
            np_all_feat = mmscl.fit_transform(df[comp_feat].to_numpy())
        else:
            mmscl.fit(train_df[comp_feat].to_numpy())
            if filter_by_train == False:
                np_all_feat = mmscl.transform(df[comp_feat].to_numpy())
            else:
                np_all_feat = mmscl.transform(train_df[comp_feat].to_numpy())
            
    else:
        np_all_feat = df[comp_feat].to_numpy()
    pcaer = None
    if pca > 0:
        pcaer = SKD.PCA(n_components=pca, whiten=True, random_state=seed)
        np_all_feat = pcaer.fit_transform(np_all_feat)
    txdict = defaultdict(lambda: None)
    txdict['scaler'] = mmscl
    txdict['pca'] = pcaer
    ret_df = None
    if filter_by_train == True:
        ret_df = train_df
    else:
        ret_df = df
    return ret_df, np_all_feat, txdict


def get_joined_songs():
    res = []
    with open(os.path.join(G.data_dir2, 'joinedsongs.txt'), 'r') as f:
        res = np.array([x.strip() for x in f.readlines()])
    return res
 

def get_all_songs():
    res = []
    with open(os.path.join(G.data_dir2, 'allsongs.txt'), 'r') as f:
        res = np.array([x.strip() for x in f.readlines()])
    return res
   
def get_random_songs(songlist, rng, num=100):
    return songlist[rng.choice(np.arange(songlist.shape[0]), size=num, replace=False)]

# pl_file is json file 
def get_playlist_json(pl_file):
    cpath = os.path.join(G.data_dir, pl_file)
    ret = None
    with open(cpath, 'r') as f:
        ret = json.load(f)
    return ret
    
# pl_csv is a csv listing playlists
# csv_path is the path to csvs
def playlist_csv_generator(pl_csv, csv_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data'), rows=np.inf):
    with open(os.path.join(csv_path, pl_csv), 'r') as f:
        csvr = csv.DictReader(f)
        for row_idx,row in enumerate(csvr):
            if row_idx < rows:
                yield row
            if row_idx >= rows-1:
                break

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





