import getter as UG
import numpy as np
import routes as G
import sklearn.metrics as SKM
import sklearn.cluster as SKC
import gensim.corpora as GC
import gensim.models as GM
# input: playlist, output: candidate song ids
from collections import defaultdict

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

# playlists should be containing at least ['file'] and ['idx'] keys
# returns bm25 model and bow dictionary
def get_bm25(playlists):
    slices = {}
    gdict = GC.Dictionary()
    corpus = []
    for playlist in playlists:
        cfile = playlist['file']
        cidx = int(playlist['idx'])
        cur_pid = playlist['pid']
        cur_slice = cfile.split('.')[-2].strip()
        if cur_slice not in slices.keys():
            slices[cur_slice] = UG.get_playlist_json(cfile)
        cur_pl = slices[cur_slice]['playlists'][cidx]
        cur_uris = [x['track_uri'] for x in cur_pl['tracks']]
        gdict.add_documents([cur_uris])
        bow = gdict.doc2bow(cur_uris)
        corpus.append(bow)
    cur_m = GM.OkapiBM25Model(corpus)
    return cur_m, gdict



if __name__ == "__main__":
    pgen = UG.playlist_csv_generator('train_set.csv')
    playlists = []
    for i,pldict in enumerate(pgen):
        if i < 3:
            playlists.append(pldict)
        else:
            break
    bm25, tdict = get_bm25(playlists)

    #cnx, cursor = UG.connect_to_nct()
    #all_song_df = UG.get_feat_all_songs(cnx)
    #np_all_song, txs = UG.all_songs_tx(all_song_df, normalize=True, pca=3)
    #playlist = UG.get_playlist('mpd.slice.549000-549999.json', 793)
    #pl_songs, top_songs, top_dist = get_closest_songs_to_playlist(cnx, playlist, np_all_song, all_song_df, k=10, tx=txs)
    #print(top_songs)
    #print(top_dist)
    
    


