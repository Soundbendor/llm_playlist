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
pl_sampnum = 500
model_path = os.path.join(G.model_dir, 'bm25.model')
dict_path = os.path.join(G.model_dir, 'bm25.dict' )
idx_path = os.path.join(G.model_dir, 'bm25.index')
pl_path = os.path.join(G.model_dir, 'bm25.playlist')


def get_closest_songs_to_playlist(_cnx,playlist_ids,all_song_df, all_song_feat, metric='euclidean', k=99999,
    weights = None,tx = defaultdict(lambda: None)):
    playlist_feat = UG.get_features_by_ids(_cnx, playlist_ids)
    # results will be i,j for ith playlist song and jth song (from everything)
    has_weights = weights != None
    np_pl = None 
    pwdist = None   
    has_scaler = tx['scaler'] != None
    has_pca = tx['pca'] != None
    if has_scaler == True:
        np_pl = tx['scaler'].transform(playlist_feat[UG.comp_feat].to_numpy())
    else:
        np_pl = playlist_feat[UG.comp_feat].to_numpy()
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
    top_songs = all_song_df.iloc[sorted_idx[:k]].reset_index(drop=True)
    return sorted_idx, top_songs, top_dist



# popularity csv format: uri, count
def get_popularity_uris(pop_file = 'popularity.csv', csv_dir = pop_dir):
    uris = []
    # uri, count
    with open(os.path.join(csv_dir, pop_file), 'r') as f:
        csvr = csv.DictReader(f)
        uris = [x['uri'] for x in csvr]
    return np.array(uris)


def sample_from_playlists(_rng, _idxs, _sims, _plinfo, sample_num = 100, playlist_num = 10):
    songs = {}
    for plidx, playlist in enumerate(_plinfo[:playlist_num]):
        cur_sim = _sims[plidx]
        cfile = playlist['file']
        cidx = int(playlist['idx'])
        cur_pid = playlist['pid']
        pl_json = UG.get_playlist_json(cfile)

        cur_pl = pl_json['playlists'][cidx]
        add_track = False
        for track in cur_pl['tracks']:
            cur_uri = track['track_uri']
            if cur_uri not in songs.keys():
                add_track = True
            else:
                comp_score = songs[cur_uri]
                if comp_score < cur_sim:
                    add_track = True
        if add_track == True:
            song_info = cur_sim
            songs[cur_uri] = song_info
    uris = []
    scores = []
    for _uri, _score in songs.items():
        uris.append(_uri)
        scores.append(_score)

    scores = np.array(scores)/np.sum(scores)
    print(scores.shape[0])
    unranked_guess = _rng.choice(uris, size= sample_num, replace=False, p=scores)
    return unranked_guess
    

def get_guess(candidate_songs, playlist_uris, _rng, guess_num = 100, expr_type = 'random', mdict = None, playlist = None):
    guess = None
    num_uris = candidate_songs.shape[0]
    if expr_type == 'random':
        guess = _rng.choice(candidate_songs, size=gen_num, replace=False)
    elif expr_type == 'bm25':
        print('ranking playlists')
        top_idx, top_sim, top_pl = PL.rank_train_playlists_by_playlist(playlist,mdict['model'], mdict['dict'], mdict['sim'], mdict['plinfo'], mask = mdict['mask'])

        #print(top_idx, top_sim, top_pl)
        print('sampling from playlists')
        unranked = sample_from_playlists(_rng, top_idx, top_sim, top_pl, sample_num = gen_num, playlist_num = pl_sampnum)
        #print(unranked)
        unranked_ids = [x.split(':')[-1].strip() for x in unranked]
        playlist_ids = [x.split(':')[-1].strip() for x in playlist_uris[:mdict['mask']]]
        num_unranked = len(unranked_ids)
        lost_songs = []
        found_songs = []
        for x in unranked_ids:
            if x not in mdict['ididx']:
                lost_songs.append(f'spotify:track:{x}')
            else:
                found_songs.append(x)
        unranked_loc = [mdict['ididx'].get_loc(x) for x in found_songs]
        print('ranking songs')
        top_idx, top_songs, top_dist = get_closest_songs_to_playlist(bstuff['cnx'], playlist_ids,mdict['song_df'].iloc[unranked_loc], mdict['song_feat'][unranked_loc], metric='euclidean', weights = None,tx = mdict['txs'])
        guess = np.array([f'spotify:track:{_id}' for _id in top_songs['id'].values] + lost_songs)
        #print(_guess)
    elif expr_type == 'cos-sim':
        playlist_ids = [x.split(':')[-1].strip() for x in playlist_uris[:mdict['mask']]]
        use_locs = [mdict['ididx'].get_loc(x) for x in mdict['ididx'] if x not in playlist_ids]
        top_idx, top_songs, top_dist = get_closest_songs_to_playlist(bstuff['cnx'], playlist_ids,mdict['song_df'].iloc[use_locs], mdict['song_feat'][use_locs], metric='euclidean', weights = None,tx = mdict['txs'], k = gen_num)
        guess = np.array([f'spotify:track:{_id}' for _id in top_songs['id'].values])
    return guess
    

res_header = ['R_Precision', 'DCG', 'IDCG', 'NDCG', 'Recommended_Songs_Clicks']
# validation_set.csv format, name,num_tracks,idx,file,pid,modified_at,collaborative,num_albums,num_followers
all_uris = get_popularity_uris()
#exprs = ['random']
#test_num = 1
exprs = ['bm25', 'random', 'cos-sim']
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
    song_df = None
    txs = None
    bstuff = {}
    if expr in ['bm25','cos-sim']:
        bstuff['cnx'], _ = UG.connect_to_nct()
        bstuff['song_df'] = pd.read_csv(G.joined_csv2_path, index_col=[0])
        bstuff['song_feat'], bstuff['txs'] = UG.all_songs_tx(bstuff['song_df'], normalize=True, pca = 0, seed=cur_seed)
        bstuff['ididx'] = pd.Index(bstuff['song_df']['id'])
        #bstuff['song_df'] = bstuff['song_df'].set_index('id')
        bstuff['mask'] = cond_num
    #print('got here')
    if expr in ['bm25']:
        bstuff['model'] = GM.OkapiBM25Model.load(model_path)
        bstuff['dict'] = GC.Dictionary.load(dict_path)
        bstuff['tmp'] = GT.get_tmpfile(idx_path)
        #bsim = GS.Similarity.
        bstuff['sim'] = GS.Similarity.load(idx_path)
        bstuff['sim'].output_prefix = idx_path
        bstuff['sim'].check_moved()

        with open(pl_path, 'r') as f:
            csvr = csv.DictReader(f)
            bstuff['plinfo'] = np.array([row for row in csvr])

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
                guess = get_guess(all_uris, cur_uris, rng, expr_type = expr, guess_num = gen_num, mdict = bstuff, playlist = val_pl)
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
        
