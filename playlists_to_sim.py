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

#pls = UG.playlist_csv_generator('test.csv', csv_path = os.sep.join(__file__.split(os.sep)[:-1]))
pls = UG.playlist_csv_generator('all_playlists.csv', csv_path= G.data_dir2)
# things that should stay the same
model_path = os.path.join(G.model_dir, 'retrain2_bm25.model')
dict_path = os.path.join(G.model_dir, 'retrain2_bm25.dict' )

# things you want to change
idx_path = os.path.join(G.model_dir, 'retrain2_bm25_full2.index')
pl_path = os.path.join(G.model_dir, 'retrain2_bm25_full2.playlist')


def get_similarities_from_corpus(playlists, gdict, gmodel, idx_path, pl_path, lazy = True):
    prev_slice = None
    prev_cfile = ""
    slices = {}
    corpus = []
    pl_info = []
    for i,playlist in enumerate(playlists):
        print(f"processing playlist {i+1}")
        cfile = playlist['file']
        cidx = int(playlist['idx'])
        cur_pid = playlist['pid']
        pl_info.append({'file': cfile, 'idx': cidx, 'pid': cur_pid})
        cur_slice = cfile.split('.')[-2].strip()
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

        bow = gdict.doc2bow(cur_uris)
        corpus.append(bow)
    btmp = GT.get_tmpfile(idx_path)
    cur_sim = GS.Similarity(btmp, gmodel[corpus], len(gdict))
    cur_sim.save(idx_path)
    with open(pl_path, 'w') as f:
        csvw = csv.DictWriter(f, fieldnames=['file', 'idx', 'pid'])
        csvw.writeheader()
        for pl in pl_info:
            csvw.writerow(pl)


bmodel = GM.OkapiBM25Model.load(model_path)
bdict = GC.Dictionary.load(dict_path)


get_similarities_from_corpus(pls, bdict, bmodel, idx_path, pl_path, lazy = True)
