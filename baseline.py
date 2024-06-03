import os,csv,json
import metrics as UM
import getter as UG
import numpy as np
import routes as G
import sklearn.metrics as SKM
import sklearn.cluster as SKC
import gensim.corpora as GC
import gensim.models as GM
# input: playlist, output: candidate song ids
from collections import defaultdict

cur_seed = 5

# condition on 10, generate 100, should be at least 20 songs length
# 500 samples
csv_dir = os.path.join(__file__.split(os.sep)[0], 'data')
res_dir = os.path.join(__file__.split(os.sep)[0], 'res')
#playlist_csvs = list(os.listdir(csv_dir))
playlist_csvs = ['num_splits/num_tracks-50.csv']
num_csvs = len(playlist_csvs)
rec_cols = ['artist_name', 'track_name', 'id']
rec_cols2 = rec_cols + ['dist']



#print("all songs scaling", mmscl_all.scale_)
# options for metric: cityblock, cosine, euclidean, l1, l2, manhattan
# returns top songs as pandas dataframe, and distances

# input is the database connection, the "raw" playlist to compare things do, numpy data of all the songs, and the all song dataframe
def get_closest_songs_to_playlist(_cnx, playlist, all_song_feat, all_song_df, metric='euclidean', mask=10, k=500,
        weights = None,tx = defaultdict(lambda: None)):
    playlist_feat = UG.get_feat_playlist(_cnx, playlist)
    # results will be i,j for ith playlist song and jth song (from everything)
    has_weights = weights != None
    np_pl = None 
    pwdist = None   
    has_scaler = tx['scaler'] != None
    has_pca = tx['pca'] != None
    if has_scaler == True:
        np_pl = tx['scaler'].transform(playlist_feat[UG.comp_feat].to_numpy()[:mask])
    else:
        np_pl = playlist_feat[UG.comp_feat].to_numpy()[:mask]
    if has_pca == True:
        np_pl = tx['pca'].transform(np_pl)
    if has_weights == True:
        np_weights = None
        if 'dict' in type(weights).__name__:
            np_weights = np.array([[weights[i]] for i in UG.comp_feat])
        else:
            np_weights = np.array([[i] for i in weights])
        pwdist = SKM.pairwise_distances(np.multiply(np_pl, np_weights.T), np.multiply(all_song_feat, np_weights.T))
    else:
        pwdist = SKM.pairwise_distances(np_pl, all_song_feat)
    # average over all playlist songs to get average distances to each song
    all_song_dist = np.mean(pwdist,axis=0)
    sorted_idx = np.argsort(all_song_dist)
    if metric == 'cosine':
        sorted_idx = sorted_idx[::-1]
    top_dist = all_song_dist[sorted_idx[:k]]
    top_songs = all_song_df.iloc[sorted_idx[:k]]
    return playlist_feat, top_songs, top_dist



rng = np.random.default_rng(cur_seed)
def sample_playlists(_rng, min_length = 20, sample_num=500, max_tries=10):
    cur_num = 0
    pl_pid = set()
    playlists = []
    csv_path = os.path.join(csv_dir, playlist_csvs[0])
    with open(csv_path, 'r') as f:
        csvr = csv.DictReader(f)
        cur_playlists = [x for x in csvr]
        num_playlists = len(cur_playlists)
        while cur_num < sample_num:
            pl_len = 0
            cur_tries = 0
            while pl_len < min_length and cur_tries < max_tries:
                pl_idx = rng.integers(num_playlists)
                cur_pl = cur_playlists[pl_idx]
                cur_len = int(cur_pl['num_splits/num_tracks'])
                cur_id = int(cur_pl['pid'])
                if cur_len < min_length or cur_id in pl_pid:
                    max_tries += 1
                else:
                    pl_pid.add(cur_id)
                    cur_dict = {'file': playlist_csvs[0], 'idx': pl_idx, 'num_splits/num_tracks': cur_len, 'pid': cur_id}
                    playlists.append(cur_dict)
                    cur_num += 1
                    pl_len = cur_len
    return playlists

def get_playlists(sample_num=500):
    cur_playlists = None
    csv_path = os.path.join(csv_dir, playlist_csvs[0])
    with open(csv_path, 'r') as f:
        csvr = csv.DictReader(f)
        cur_playlists = [x for x in csvr]
    
    return cur_playlists[:sample_num]

if __name__ == "__main__":
    weights = {'danceability':1.0,
            'energy':1.0,
            'key':1.0,
            'loudness':1.0,
            'mode':1.0,
            'speechiness':1.0,
            'acousticness':1.0,
            'instrumentalness':1.0,
            'liveness':1.0,
            'valence':1.0,
            'tempo': 1.0}
    weights = {'danceability': 0.75,
            'energy':1.0,
            'key':0.0,
            'loudness':0.0,
            'mode':0.5,
            'speechiness':1.0,
            'acousticness':1.5,
            'instrumentalness':1.5,
            'liveness':0.25,
            'valence':1.0,
            'tempo': 2.5}
    #weights = None
    cond_num = 10
    gen_num = 100
    sample_num = 1000
    res_dir2 = os.path.join(__file__.split(os.sep)[0], 'res', f'prellm_test0-{cond_num}_{gen_num}_{sample_num}')
    pl = get_playlists(sample_num = sample_num)
    num_runs = 1000
    mheader = ['expr_idx', 'pl_idx', 'r_prec', 'dcg', 'idcg', 'ndcg', 'clicks']
    r2_path = os.path.join(res_dir2, f'metrics-{cond_num}_{gen_num}_{sample_num}.csv')
    w_path = os.path.join(res_dir2, f'weights-{cond_num}_{gen_num}_{sample_num}.csv')

    #weights = [1.0, 0.8,0.6,0.4,0.2]
    if os.path.exists(res_dir2) == False:
        os.mkdir(res_dir2)
    if weights != None: 
        if 'dict' in type(weights).__name__:
            with open(w_path, 'w') as f:
                csvw = csv.DictWriter(f, fieldnames = UG.comp_feat)
                csvw.writeheader()
                csvw.writerow({x:y for (x,y) in weights.items() if x in UG.comp_feat})
        else:
            wlen = len(weights)
            fieldnames = [f'w{i}' for i in range(1, wlen+1)]
            wzip = {y:x for (x,y) in zip(weights,fieldnames)}
            with open(w_path, 'w') as f:
                csvw = csv.DictWriter(f, fieldnames = fieldnames)
                csvw.writeheader()
                csvw.writerow(wzip)



    runs = []


    cnx, cursor = UG.connect_to_nct()

    all_song_df, all_song_feat, txs = PL.load_all_songs(cnx, normalize=True, pca=0,seed=cur_seed)

    for pl_i, pl_dict in enumerate(pl):
        if pl_i < num_runs:
        #if True:
            print(pl_i)
            print('-----')
            pl_c = UG.get_playlist(pl_dict['file'], int(pl_dict['idx']))
            pl_songs, res_songs, res_cos_sim = PL.get_closest_songs_to_playlist(cnx, pl_c, all_song_feat, all_song_df, metric='l1', mask=cond_num, k=gen_num, weights = weights, tx=txs)
            truth_ids = pl_songs['id'].to_numpy()[cond_num:]
            retr_ids = res_songs['id'].to_numpy()
            r_prec = UM.r_precision(truth_ids, retr_ids)
            dcg = UM.dcg(truth_ids, retr_ids)
            idcg = UM.idcg(truth_ids, retr_ids)
            ndcg = UM.ndcg(truth_ids, retr_ids)
            clicks = UM.rec_songs_clicks(truth_ids, retr_ids, max_clicks=99998)
            g_path = os.path.join(res_dir2, f'ground_truth-{cond_num}_{gen_num}_{sample_num}-{pl_i}.csv')
            r_path = os.path.join(res_dir2, f'retrieved-{cond_num}_{gen_num}_{sample_num}-{pl_i}.csv')
            pl_songs[cond_num:][rec_cols].to_csv(g_path)
            res_songs = res_songs.assign(dist = res_cos_sim)
            res_songs[rec_cols2].to_csv(r_path)
            mdict = {'expr_idx': pl_i, 'pl_idx': int(pl_dict['idx']), 'r_prec': r_prec, 'dcg': dcg, 'idcg': idcg, 'ndcg': ndcg, 'clicks': clicks}
            runs.append(mdict)
            print('r_prec', r_prec)
            print('dcg', dcg)
            print('idcg', idcg)
            print('ndcg', ndcg)
            print('clicks', clicks)
    with open(r2_path, 'w') as f:
        csvw = csv.DictWriter(f, fieldnames = mheader)
        csvw.writeheader()
        for run in runs:
            csvw.writerow(run)


    #get_similar_playlists_to_uris(test_uris, bm25, sim_idx, gdict, pl_info, k =5)
    #cnx, cursor = UG.connect_to_nct()
    #all_song_df = UG.get_feat_all_songs(cnx)
    #np_all_song, txs = UG.all_songs_tx(all_song_df, normalize=True, pca=3)
    #playlist = UG.get_playlist('mpd.slice.549000-549999.json', 793)
    #pl_songs, top_songs, top_dist = get_closest_songs_to_playlist(cnx, playlist, np_all_song, all_song_df, k=10, tx=txs)
    #print(top_songs)
    #print(top_dist)






