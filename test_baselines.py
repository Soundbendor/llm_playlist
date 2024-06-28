import os,csv,json,time,sys
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
valid_dir = os.path.join(__file__.split(os.sep)[0], 'valid_retrain')
#valid_dir = os.path.join(__file__.split(os.sep)[0], 'valid')
pop_dir = os.path.join(data_dir, 'stats')
res_dir = os.path.join(__file__.split(os.sep)[0], 'res')
#res_dir = '/media/dxk/TOSHIBA EXT/llm_playlist_res'
#playlist_csvs = list(os.listdir(csv_dir))
#num_csvs = len(playlist_csvs)

#gen_num = 100
#pl_sampnum = 25
test_num = 1000
gen_num = 500
pl_sampnum = 100

model_path = os.path.join(G.model_dir, 'retrain_bm25.model')
dict_path = os.path.join(G.model_dir, 'retrain_bm25.dict' )
idx_path = os.path.join(G.model_dir, 'retrain_bm25.index')
pl_path = os.path.join(G.model_dir, 'retrain_bm25.playlist')



#model_path = os.path.join(G.model_dir, 'bm25.model')
#dict_path = os.path.join(G.model_dir, 'bm25.dict' )
#idx_path = os.path.join(G.model_dir, 'bm25.index')
#pl_path = os.path.join(G.model_dir, 'bm25.playlist')

chall_todo = []
if len(sys.argv) > 1:
    chall_todo = set([int(x) for x in sys.argv[1:]])
"""
cs_weights = {'danceability': 0.75,
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
"""

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


# sampling given bm25 results
# _rng: numpy rng, _idxs: indices returned from playlist ranking (probably not important, unused here),
# _sims: bm25 similarities (playlist level)
# _plinfo: ranked playlist info (file, idx, pid), ipt_uris, tracks in the input playlist
# sample_num = number of songs to sample
# playlist_num: top k playlists to sample from
def sample_from_playlists(_rng, _idxs, _sims, _plinfo, ipt_uris, sample_num = 100, playlist_num = 10, mask=99999):
    songs = {}
    ignore_uris = set(ipt_uris[:mask])
    for plidx, playlist in enumerate(_plinfo[:playlist_num]):
        cur_sim = _sims[plidx]
        cfile = playlist['file']
        cidx = int(playlist['idx'])
        cur_pid = playlist['pid']
        pl_json = UG.get_playlist_json(cfile)

        cur_pl = pl_json['playlists'][cidx]
        for track in cur_pl['tracks']:
            score_to_add = cur_sim
            cur_uri = track['track_uri']
            if cur_uri not in ignore_uris:
                if cur_uri in songs.keys():
                    old_score = songs[cur_uri]
                    score_to_add = cur_sim + old_score
                songs[cur_uri] = score_to_add
    uris = []
    scores = []
    for _uri, _score in songs.items():
        uris.append(_uri)
        scores.append(_score)

    #scores = np.array(scores)/np.sum(scores)
    scores = np.array(scores)

    top_idxs = np.argsort(scores)[::-1][:sample_num]
    top_uris = np.array(uris)[top_idxs]
    top_scores = scores[top_idxs]
    print(scores.shape[0], top_scores[:5])
    #unranked_guess = _rng.choice(uris, size= sample_num, replace=False, p=scores)
    #return unranked_guess
    return top_idxs, top_scores, top_uris
    

def get_guess(candidate_songs, playlist_uris, _rng, guess_num = 100, expr_type = 'random', mdict = None, playlist = None):
    guess = None
    num_uris = candidate_songs.shape[0]
    if expr_type == 'random':
        playlist_ids = [x.split(':')[-1].strip() for x in playlist_uris[:mdict['mask']]]
        use_songs = [x for x in mdict['ididx'] if x.strip() not in playlist_ids]
        guess_ids = _rng.choice(use_songs, size=guess_num, replace=False)
        guess = np.array([f'spotify:track:{_id}' for _id in guess_ids])
    elif expr_type == 'bm25':
        print('ranking playlists')
        top_idx, top_sim, top_pl = PL.rank_train_playlists_by_playlist(playlist,mdict['model'], mdict['dict'], mdict['sim'], mdict['plinfo'], mask = mdict['mask'])

        #print(top_idx, top_sim, top_pl)
        print('sampling from playlists')
        #unranked = sample_from_playlists(_rng, top_idx, top_sim, top_pl, playlist_uris, sample_num = guess_num, playlist_num = pl_sampnum)
        top_idxs2, top_scores2, guess = sample_from_playlists(_rng, top_idx, top_sim, top_pl, playlist_uris, sample_num = guess_num, playlist_num = pl_sampnum, mask = mdict['mask'])
        #print(unranked)
        """
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
        """
        #print(_guess)
    elif expr_type == 'euclid':
        playlist_ids = [x.split(':')[-1].strip() for x in playlist_uris[:mdict['mask']]]
        use_locs = [mdict['ididx'].get_loc(x) for x in mdict['ididx'] if x.strip() not in playlist_ids]
        top_idx, top_songs, top_dist = get_closest_songs_to_playlist(bstuff['cnx'], playlist_ids,mdict['song_df'].iloc[use_locs], mdict['song_feat'][use_locs], metric='euclidean', weights = cs_weights,tx = mdict['txs'], k = guess_num)
        guess = np.array([f'spotify:track:{_id}' for _id in top_songs['id'].values])
    return guess
    

res_header = ['R_Precision', 'DCG', 'IDCG', 'NDCG', 'Recommended_Songs_Clicks']
# validation_set.csv format, name,num_tracks,idx,file,pid,modified_at,collaborative,num_albums,num_followers
all_uris = get_popularity_uris()
#exprs = ['euclid', 'random']
#exprs = ['bm25','euclid','random']
exprs = ['bm25']
challenges = UG.get_challenges()
for expr in exprs:
    rng = np.random.default_rng(seed=cur_seed)
    r_precs = []
    dcgs = []
    idcgs = []
    ndcgs = []
    rscs = []
    times = []
    song_feat = None
    song_df = None
    txs = None
    bstuff = {}
    bstuff['cnx'], _ = UG.connect_to_nct()
    bstuff['song_df'] = pd.read_csv(G.joined_csv2_path, index_col=[0])
    bstuff['song_feat'], bstuff['txs'] = UG.all_songs_tx(bstuff['song_df'], normalize=True, pca = 0, seed=cur_seed)
    bstuff['ididx'] = pd.Index(bstuff['song_df']['id'])
    #bstuff['song_df'] = bstuff['song_df'].set_index('id')
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
    res_path = os.path.join(res_dir, f'bline-chall_{expr}_filt_{gen_num}_real')

    chall_avgarr = []
    for chall in challenges:
        chall_num = chall['challenge']
        if chall_num <= 1:
            # non baseline challenge
            continue
        if len(chall_todo) > 0:
            # specified challenges to do
            if chall_num not in chall_todo:
                # not in specified challenges
                continue
        cond_num = chall['num_cond']
        bstuff['mask'] = cond_num
        file_idx = chall['file_idx']
        chall_file = chall['file']
        val_plgen = UG.playlist_csv_generator(chall_file, csv_path = valid_dir)
        chall_res = []
        guess_arr = []
        for val_pl in val_plgen:
            if val_idx >= test_num:
                continue
            cfile = val_pl['file']
            cidx = int(val_pl['idx'])
            pl_json = UG.get_playlist_json(cfile)
            query_pl = pl_json['playlists'][cidx]
            cur_uris = [x['track_uri'].strip() for x in query_pl['tracks']]
            ground_truth = np.array(cur_uris[cond_num:])
            #print(ground_truth)
            ground_truth_len = ground_truth.shape[0]
            if ground_truth_len <= 0:
                print("skip")
                continue
            print(f'running experiment {expr} {val_idx+1}/{test_num}')
            print('---------')
            guess = get_guess(all_uris, cur_uris, rng, expr_type = expr, guess_num = gen_num, mdict = bstuff, playlist = val_pl)
            cur_m = UM.calc_metrics(ground_truth, guess, max_clicks=gen_num)
            chall_res.append(cur_m)
            guess_arr.append(guess)
            UM.metrics_printer(cur_m)
            val_idx += 1
        chall_avg = UM.get_mean_metrics(chall_res)
        chall_avgarr.append(chall_avg)
        cur_fname = f'chall-bin_{file_idx}-res.csv'
        cur_fname_avg = f'chall-bin_{file_idx}-resavg.csv'
        guess_fname = f'chall-bin_{file_idx}-guess.json'
        UM.metrics_writer(chall_res, fname=cur_fname, fpath=res_path)
        UM.metrics_writer([chall_avg], fname=cur_fname_avg, fpath=res_path)
        UM.guess_writer(guess_arr, fname=guess_fname, fpath=res_path)
    overall_avg = UM.get_mean_metrics(chall_avgarr)
    cur_fname2 = f'overall-res.csv'
    cur_fname_avg2 = f'overall-resavg.csv'
           
    UM.metrics_writer(chall_avgarr, fname=cur_fname2, fpath=res_path)
    UM.metrics_writer([overall_avg], fname=cur_fname_avg2, fpath=res_path)
