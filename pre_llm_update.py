import getter as UG
import os, csv
import numpy as np
import routes as G
import sklearn.metrics as SKM
import sklearn.cluster as SKC
import gensim.corpora as GC
import gensim.models as GM
import gensim.similarities as GS
import gensim.test.utils as GT
# input: playlist, output: candidate song ids
from collections import defaultdict


# playlists should be containing at least ['file'] and ['idx'] keys
# returns bm25 model, bow dictionary, similarity index, and playlist info
def get_bm25(playlists, idx_path, lazy = True):
    slices = {}
    prev_slice = None
    prev_cfile = ""
    gdict = GC.Dictionary()
    corpus = []
    pl = [] # playlist info to recorrespond back with corpus
    num_playlists = len(playlists)
    for i,playlist in enumerate(playlists):
        print(f"processing playlist {i+1} out of {num_playlists}")
        cfile = playlist['file']
        cidx = int(playlist['idx'])
        cur_pid = playlist['pid']
        pl.append({'file': cfile, 'idx': cidx, 'pid': cur_pid})
        cur_pl = None
        if lazy == False:
            cur_slice = cfile.split('.')[-2].strip()
            if cur_slice not in slices.keys():
                slices[cur_slice] = UG.get_playlist_json(cfile)
            cur_pl = slices[cur_slice]['playlists'][cidx]
        else:
            _slice = None
            if prev_cfile != cfile:
                _slice = UG.get_playlist_json(cfile)
                prev_slice = _slice
            else:
                _slice = prev_slice
            cur_pl = _slice['playlists'][cidx]
        cur_uris = [x['track_uri'] for x in cur_pl['tracks']]
        gdict.add_documents([cur_uris])
        bow = gdict.doc2bow(cur_uris)
        corpus.append(bow)
        prev_cfile = cfile
    cur_m = GM.OkapiBM25Model(corpus)
    tmp_file = GT.get_tmpfile("bm25_tmp")
    cur_sim = GS.Similarity(tmp_file, cur_m[corpus], len(gdict))
    return cur_m, gdict, cur_sim, pl

def get_similarities_from_corpus(playlists, gdict, idx_path, pl_path):
    slices = {}
    corpus = []
    num_playlists = len(playlists)
    pl = []
    for i,playlist in enumerate(playlists):
        print(f"processing playlist {i+1} out of {num_playlists}")
        cfile = playlist['file']
        cidx = int(playlist['idx'])
        cur_pid = playlist['pid']
        pl.append({'file': cfile, 'idx': cidx, 'pid': cur_pid})
        cur_slice = cfile.split('.')[-2].strip()
        if cur_slice not in slices.keys():
            slices[cur_slice] = UG.get_playlist_json(cfile)
        cur_pl = slices[cur_slice]['playlists'][cidx]
        cur_uris = [x['track_uri'] for x in cur_pl['tracks']]

        bow = gdict.doc2bow(cur_uris)
        corpus.append(bow)
    btmp = GT.get_tmpfile(idx_path)
    cur_sim = GS.Similarity(tmp_file, cur_m[corpus], len(gdict))
    cur_sim.save(idx_path)
    with open(pl_path, 'w') as f:
        csvw = csv.DictWriter(f, fieldnames=['file', 'idx', 'pid'])
        csvw.writeheader()
        for pl in pl_info:
            csvw.writerow(pl)



def rank_uri_by_playlist(_uri,_bm25, _dict, _idx, _plinfo):
    #cfile = playlist['file']
    #cidx = int(playlist['idx'])
    #cur_pid = playlist['pid']
    #pl_json = UG.get_playlist_json(cfile)

    #cur_pl = pl_json['playlists'][cidx]
    #cur_uris = [x['track_uri'] for x in cur_pl['tracks']][:mask]
    bow = _dict.doc2bow([_uri])
    
    cur_rep = _bm25[bow]
    cur_sim = _idx[cur_rep]
    top_idx = np.argsort(cur_sim)[::-1]
    top_sim = cur_sim[top_idx]
    top_pl = _plinfo[top_idx]
    return top_idx, top_sim, top_pl




def rank_train_playlists_by_playlist(playlist,_bm25, _dict, _idx, _plinfo, mask = 99999):
    cfile = playlist['file']
    cidx = int(playlist['idx'])
    cur_pid = playlist['pid']
    pl_json = UG.get_playlist_json(cfile)

    cur_pl = pl_json['playlists'][cidx]
    cur_uris = [x['track_uri'] for x in cur_pl['tracks']][:mask]
    bow = _dict.doc2bow(cur_uris)
    
    cur_rep = _bm25[bow]
    cur_sim = _idx[cur_rep]
    top_idx = np.argsort(cur_sim)[::-1]
    top_sim = cur_sim[top_idx]
    top_pl = _plinfo[top_idx]
    return top_idx, top_sim, top_pl


if __name__ == "__main__":
    if os.path.exists(G.model_dir) == False:
        os.mkdir(G.model_dir)
    
    
    pgen_train = UG.playlist_csv_generator('train_set.csv')
    #pgen_train = UG.playlist_csv_generator('train_pids_retrain.csv')
    pgen_valid = UG.playlist_csv_generator('validation_set.csv')
    model_path = os.path.join(G.model_dir, 'retrain_bm25.model')
    dict_path = os.path.join(G.model_dir, 'retrain_bm25.dict' )
    idx_path = os.path.join(G.model_dir, 'retrain_bm25.index')
    pl_path = os.path.join(G.model_dir, 'retrain_bm25.playlist')
    if os.path.exists(model_path) == False:
    #if True:
        print('training models')
        playlists = [x for x in pgen_train]
         
        #bdict = GC.Dictionary.load(dict_path)
        #get_similarities_from_corpus(playlists, bdict, idx_path, pl_path)
        bm25, gdict, sim_idx, pl_info = get_bm25(playlists, idx_path)
        #model_dir = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'models')
        bm25.save(model_path)
        gdict.save(dict_path)
        sim_idx.save(idx_path)
        with open(pl_path, 'w') as f:
            csvw = csv.DictWriter(f, fieldnames=['file', 'idx', 'pid'])
            csvw.writeheader()
            for pl in pl_info:
                csvw.writerow(pl)
    else:
        print('loading models')
        bmodel = GM.OkapiBM25Model.load(model_path)
        bdict = GC.Dictionary.load(dict_path)
        btmp = GT.get_tmpfile(idx_path)
        #bsim = GS.Similarity.
        #print(bdict)
        bsim = GS.Similarity.load(idx_path)
        bsim.output_prefix = idx_path
        bsim.check_moved()
        pl_info = []
        cur_uri = "spotify:track:6fndSocvpEUYd5OFDGNYzj"
        testing_playlists = True
        with open(pl_path, 'r') as f:
            csvr = csv.DictReader(f)
            pl_info = np.array([row for row in csvr])
        if testing_playlists == True:
            for i, pl in enumerate(pgen_train):
                if i < 5:

                    top_idx, top_sim, top_pl = rank_train_playlists_by_playlist(pl, bmodel, bdict, bsim, pl_info, mask=999999)
                    print(top_idx, top_sim, top_pl)

                else:
                    break
        else:
            top_idx, top_sim, top_pl = rank_uri_by_playlist(cur_uri, bmodel, bdict, bsim, pl_info)
            print(top_idx, top_sim, top_pl)
