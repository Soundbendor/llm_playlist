import os,csv,json
import glob as G
import numpy as np
import getter as UG
import metrics as UM
import pre_llm as PL

cur_seed = 5

# condition on 10, generate 100, should be at least 20 songs length
# 500 samples
csv_dir = os.path.join(__file__.split(os.sep)[0], 'data')
res_dir = os.path.join(__file__.split(os.sep)[0], 'res')
#playlist_csvs = list(os.listdir(csv_dir))
playlist_csvs = ['num_tracks-50.csv']
num_csvs = len(playlist_csvs)
rec_cols = ['artist_name', 'track_name', 'id']
rec_cols2 = rec_cols + ['dist']


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
                cur_len = int(cur_pl['num_tracks'])
                cur_id = int(cur_pl['pid'])
                if cur_len < min_length or cur_id in pl_pid:
                    max_tries += 1
                else:
                    pl_pid.add(cur_id)
                    cur_dict = {'file': playlist_csvs[0], 'idx': pl_idx, 'num_tracks': cur_len, 'pid': cur_id}
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
    """
    cur_header = ['file', 'idx', 'num_tracks', 'pid']
    res = sample_playlists(rng, sample_num=500)
    with open(os.path.join(res_dir, 'pre_llm_test_playlists.csv'), 'w') as f:
        csvw = csv.DictWriter(f, fieldnames=cur_header)
        csvw.writeheader()
        for playlist in res:
            csvw.writerow(playlist)
    print(res)
    """
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
    """
    weights = {'danceability':0.75,
            'energy':0.75,
            'key':0.0,
            'loudness':0.125,
            'mode':0.5,
            'speechiness':0.75,
            'acousticness':0.5,
            'instrumentalness':0.75,
            'liveness':0.25,
            'valence':1.0,
            'tempo': 1.0}
    """
    cond_num = 10
    gen_num = 100

    res_dir2 = os.path.join(__file__.split(os.sep)[0], 'res', f'prellm_test2-{cond_num}_{gen_num}')
    pl = get_playlists(sample_num = 500)
    num_runs = 1000
    mheader = ['expr_idx', 'pl_idx', 'r_prec', 'dcg', 'idcg', 'ndcg', 'clicks']
    r2_path = os.path.join(res_dir2, f'metrics-{cond_num}_{gen_num}.csv')
    w_path = os.path.join(res_dir2, f'weights-{cond_num}_{gen_num}.csv')

    if os.path.exists(res_dir2) == False:
        os.mkdir(res_dir2)
    with open(w_path, 'w') as f:
        csvw = csv.DictWriter(f, fieldnames = UG.comp_feat)
        csvw.writeheader()
        csvw.writerow(weights)

    runs = []
    for pl_i, pl_dict in enumerate(pl):
        if pl_i < num_runs:
        #if True:
            print(pl_i)
            print('-----')
            pl_c = UG.get_playlist(pl_dict['file'], int(pl_dict['idx']))
            pl_songs, res_songs, res_cos_sim = PL.get_closest_songs_to_playlist(PL.cnx, pl_c, metric='manhattan', mask=cond_num, k=gen_num, weights = weights)
            truth_ids = pl_songs['id'].to_numpy()[cond_num:]
            retr_ids = res_songs['id'].to_numpy()
            r_prec = UM.r_precision(truth_ids, retr_ids)
            dcg = UM.dcg(truth_ids, retr_ids)
            idcg = UM.idcg(truth_ids, retr_ids)
            ndcg = UM.ndcg(truth_ids, retr_ids)
            clicks = UM.rec_songs_clicks(truth_ids, retr_ids, max_clicks=99998)
            g_path = os.path.join(res_dir2, f'ground_truth-{cond_num}_{gen_num}-{pl_i}.csv')
            r_path = os.path.join(res_dir2, f'retrieved-{cond_num}_{gen_num}-{pl_i}.csv')
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



    






