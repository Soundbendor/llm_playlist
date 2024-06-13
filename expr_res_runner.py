import os,csv,json,time
import metrics as UM
import getter as UG
import numpy as np
import routes as G
import sklearn.metrics as SKM
import sklearn.cluster as SKC
import gensim.corpora as GC
import gensim.models as GM
import gensim.similarities as GS
import gensim.test.utils as GT
import pandas as pd
import pre_llm as PL
from collections import defaultdict


res_header = ['R_Precision', 'DCG', 'IDCG', 'NDCG', 'Recommended_Songs_Clicks']

to_rec = True
res_folder = 'baseline_bm25_filt_500_real'
#expr_idx = 20
num_pl = 100

res_base = os.path.join(__file__.split(os.sep)[0], 'res')
res_dir = os.path.join(res_base, res_folder)
res_path = os.path.join(res_dir, 'baseline_bm25_500_to_100_real')
data_dir = os.path.join(__file__.split(os.sep)[0], 'data')
valid_csv = os.path.join(data_dir, 'filtered_validation_set.csv')
#song_df = pd.read_csv(G.joined_csv2_path, index_col=[0])
#song_feat, song_tx = UG.all_songs_tx(song_df, normalize=True, pca = 0, seed=cur_seed)

if os.path.exists(res_path) == False:
        os.mkdir(res_path)

rec_feat =  ['artist_name', 'track_name'] + UG.comp_feat
rec_feat2 =  ['artist_name', 'track_name', 'in_gt'] + UG.comp_feat

cs_weights = {'danceability': 1.0,
            'energy':1.0,
            'key':0.0,
            'loudness':0.0,
            'mode':1.0,
            'speechiness':1.0,
            'acousticness':1.0,
            'instrumentalness':1.0,
            'liveness':1.0,
            'valence':1.0,
            'tempo': 1.0}


cond_num = 10

want_plinfo = None
r_precs = []
dcgs = []
idcgs = []
ndcgs = []
rscs = []
times = []

pl_rows = None
guess_len = 100
gen_num = 500
with open(valid_csv, 'r') as f:
    csvr = csv.DictReader(f)
    pl_rows = [(ic, row) for ic, row in enumerate(csvr)]

for expr_idx in range(num_pl):
    want_plinfo = pl_rows[expr_idx]
    want_pl = UG.get_playlist(want_plinfo[1]['file'], int(want_plinfo[1]['idx']))
    pl_uris = [x['track_uri'] for x in want_pl['tracks']]
    gt_uris = np.array(pl_uris[cond_num:])
    #print(gt_uris)
    res_guessfile = os.path.join(res_dir, f'guess_{expr_idx}.txt')
    guess_uris = None
    with open(res_guessfile, 'r') as f:
        guess_uris = list(f.readlines())
    first_guess_uris = np.array([x.strip() for x in guess_uris[:guess_len]])
    r_prec = UM.r_precision(gt_uris, first_guess_uris)
    dcg = UM.dcg(gt_uris, first_guess_uris)
    idcg = UM.idcg(gt_uris, first_guess_uris)
    ndcg = UM.ndcg(gt_uris, first_guess_uris)
    rsc = UM.rec_songs_clicks(gt_uris, first_guess_uris, max_clicks = guess_len)
    print(f'r_prec: {r_prec}, dcg: {dcg}, idcg: {idcg}, rsc: {rsc}')
    r_precs.append(r_prec)
    dcgs.append(dcg)
    idcgs.append(idcg)
    ndcgs.append(ndcg)
    rscs.append(rsc)

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
 
