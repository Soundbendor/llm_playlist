import os,csv,json,time
import metrics as UM
import getter as UG
import numpy as np
import routes as G
import sklearn.metrics as SKM
import sklearn.cluster as SKC
import gensim.corpora as GC
import gensim.models as GM
import pandas as pd
cur_seed = 5

# condition on 10, generate 100, should be at least 20 songs length
# 500 samples
data_dir = os.path.join(__file__.split(os.sep)[0], 'data')
eval_csv = os.path.join(data_dir, 'validation_set.csv')
pop_dir = os.path.join(data_dir, 'stats')
#res_dir = os.path.join(__file__.split(os.sep)[0], 'res')
res_dir = '/media/dxk/TOSHIBA EXT/llm_playlist_res'
#playlist_csvs = list(os.listdir(csv_dir))
#num_csvs = len(playlist_csvs)

gen_num = 100
cond_num = 10
test_num = 100

# popularity csv format: uri, count
def get_popularity_uris(pop_file = 'popularity.csv', csv_dir = pop_dir):
    uris = []
    # uri, count
    with open(os.path.join(csv_dir, pop_file), 'r') as f:
        csvr = csv.DictReader(f)
        uris = [x['uri'] for x in csvr]
    return np.array(uris)

def get_guess(candidate_songs, _rng, guess_num = 100, expr_type = 'random'):
    guess = None
    num_uris = candidate_songs.shape[0]
    if expr_type == 'random':
        guess = _rng.choice(candidate_songs, size=num_uris, replace=False)
    return guess
    

res_header = ['R_Precision', 'DCG', 'IDCG', 'NDCG', 'Recommended_Songs_Clicks']
# validation_set.csv format, name,num_tracks,idx,file,pid,modified_at,collaborative,num_albums,num_followers
all_uris = get_popularity_uris()
#exprs = ['random']
test_num = 0
exprs = ['cos_sim']
for expr in exprs:
    rng = np.random.default_rng(seed=cur_seed)
    r_precs = []
    dcgs = []
    idcgs = []
    ndcgs = []
    rscs = []
    times = []
    val_plgen = UG.playlist_csv_generator('validation_set.csv', csv_path = data_dir)
    song_feat = None
    track_ids = None
    track_idxloc = None
    if expr in ['cos_sim']:
        cnx, cur = UG.connect_to_nct()
        track_ids = [x.split(':')[-1] for x in all_uris]
        print('got track_ids')
        num_tracks = len(track_ids)
        for ididx, _id in enumerate(track_ids):
            print(f'getting {ididx+1}/{num_tracks}')
            if ididx == 0:
                song_feat = UG.get_features_by_id(cnx, _id)
                #print(song_feat)
            elif ididx < 10:
                cur_feat = UG.get_features_by_id(cnx, _id)
                #print(cur_feat)
                song_feat = pd.concat([song_feat, cur_feat]).reset_index(drop=True)
                #print(song_feat)
            else:
                break
        #song_feat = [UG.get_features_by_id(cnx, _id) for _id in track_ids]
        song_feat.to_csv(G.joined_csv2_path)
        print('got features')
        #song_feat = pd.concat([song_feat])
        #print('concat')
        #track_idxloc =  [song_feat.loc[song_feat['id'] == y].index[0] for y in track_ids]
        #print('got locations')
        #song_feat = song_feat.iloc[track_idxloc].reset_index(drop=True)
        print(track_ids)
        print(song_feat)
    val_idx = 0
    res_path = os.path.join(res_dir, f'baseline_{expr}')
    for val_pl in val_plgen:
        if val_idx < test_num:
            cfile = val_pl['file']
            cidx = int(val_pl['idx'])
            pl_json = UG.get_playlist_json(cfile)
            query_pl = pl_json['playlists'][cidx]
            cur_uris = [x['track_uri'] for x in query_pl['tracks']]
            ground_truth = np.array(cur_uris[cond_num:])
            ground_truth_len = ground_truth.shape[0]
            if ground_truth_len > 0:
                print(f'running experiment {expr} {val_idx+1}/{test_num}')
                guess = get_guess(all_uris, rng, expr_type = expr, guess_num = gen_num)
                r_prec = UM.r_precision(ground_truth, guess)
                dcg = UM.dcg(ground_truth, guess)
                idcg = UM.idcg(ground_truth, guess)
                ndcg = UM.ndcg(ground_truth, guess)
                rsc = UM.rec_songs_clicks(ground_truth, guess, max_clicks = gen_num)
                r_precs.append(r_prec)
                dcgs.append(dcg)
                idcgs.append(idcg)
                ndcgs.append(ndcg)
                rscs.append(rsc)
                guess_path = os.path.join(res_path, f'guess_{val_idx}.txt')
                with open(guess_path, 'w') as f:
                    for guess2 in guess:
                        f.write(guess2)
                        f.write('\n')

                val_idx += 1
    r_precs = np.array(r_precs)
    dcgs = np.array(dcgs)
    idcgs = np.array(idcgs)
    ndcgs = np.array(ndcgs)
    rscs = np.array(rscs)
    avg_r_prec = np.mean(r_precs)
    avg_dcg = np.mean(dcgs)
    avg_idcg = np.mean(idcgs)
    avg_ndcg = np.mean(ndcgs)
    avg_rsc = np.mean(rscs)

    if os.path.exists(res_path) == False:
        os.mkdir(res_path)
    res_per_expr_path = os.path.join(res_path, 'metrics_by_expr.csv')
    with open(res_per_expr_path, 'w') as f:
        csvr = csv.writer(f, delimiter=',')
        csvr.writerow(res_header)
        for (_rprec,_dcg,_idcg,_ndcg,_rsc) in zip(r_precs,dcgs,idcgs,ndcgs,rscs):
            csvr.writerow([_rprec,_dcg, _idcg, _ndcg, _rsc])
    res_avg_path = os.path.join(res_path, 'metrics_avg.csv')
    with open(res_avg_path, 'w') as f:
        csvr = csv.writer(f, delimiter=',')
        csvr.writerow(res_header)
        csvr.writerow([avg_r_prec, avg_dcg, avg_idcg, avg_ndcg, avg_rsc])
        
