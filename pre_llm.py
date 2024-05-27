import getter as UG
import numpy as np
import sklearn.metrics as SKM
import sklearn.preprocessing as SKP
import sklearn.cluster as SKC
import sklearn.decomposition as SKD
# input: playlist, output: candidate song ids
from collections import defaultdict


# input is the database connection, returns full dataframe, (numpy) features and scaler
def load_all_songs(cnx, normalize=True, pca = 3, seed=5):
    all_feat = UG.get_feat_all_songs(cnx)
    mmscl = None
    np_all_feat = None
    if normalize == True:
        mmscl = SKP.MinMaxScaler()
        np_all_feat = mmscl.fit_transform(all_feat[UG.comp_feat].to_numpy())
    else:
        np_all_feat = all_feat[UG.comp_feat].to_numpy()
    pcaer = None
    if pca > 0:
        pcaer = SKD.PCA(n_components=pca, whiten=True, random_state=seed)
        np_all_feat = pcaer.fit_transform(np_all_feat)
    txdict = defaultdict(lambda: None)
    txdict['scaler'] = mmscl
    txdict['pca'] = pcaer
    return all_feat, np_all_feat, txdict

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


if __name__ == "__main__":
    cnx, cursor = UG.connect_to_nct()
    all_song_df, all_song_feat, txs = load_all_songs(cnx, normalize=True, pca=3)
    playlist = UG.get_playlist('mpd.slice.549000-549999.json', 793)
    pl_songs, top_songs, top_dist = get_closest_songs_to_playlist(cnx, playlist, all_song_feat, all_song_df, k=10, tx=txs)
    print(top_songs)
    print(top_dist)
    
    


