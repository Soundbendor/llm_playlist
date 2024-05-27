import getter as UG
import numpy as np
import sklearn.metrics as SKM
import sklearn.preprocessing as SKP

# input: playlist, output: candidate song ids



# input is the database connection, returns full dataframe, (numpy) features and scaler
def load_all_songs(cnx, normalize=True):
    all_feat = UG.get_feat_all_songs(cnx)
    mmscl = None
    np_all_feat = None
    if normalize == True:
        mmscl = SKP.MinMaxScaler()
        np_all_feat = mmscl.fit_transform(all_feat[UG.comp_feat].to_numpy())
    else:
        mmscl = SKP.MinMaxScaler()
        np_all_feat = all_feat[UG.comp_feat].to_numpy()
    return all_feat, np_all_feat, mmscl

    #print("all songs scaling", mmscl_all.scale_)
# options for metric: cityblock, cosine, euclidean, l1, l2, manhattan
# returns top songs as pandas dataframe, and distances

# input is the database connection, the "raw" playlist to compare things do, numpy data of all the songs, and the all song dataframe
def get_closest_songs_to_playlist(_cnx, playlist, all_song_feat, all_song_df, metric='euclidean', mask=10, k=500,
        weights = {x:1. for x in UG.comp_feat},scaler = None):
    playlist_feat = UG.get_feat_playlist(_cnx, playlist)
    # results will be i,j for ith playlist song and jth song (from everything)
    np_weights = np.array([[weights[i]] for i in UG.comp_feat])
    np_pl = None 
    if scaler != None:
        np_pl = scaler.transform(playlist_feat[UG.comp_feat].to_numpy()[:mask])
    else:
        np_pl = playlist_feat[UG.comp_feat].to_numpy()[:mask]
    pwdist = SKM.pairwise_distances(np.multiply(np_pl, np_weights.T), np.multiply(all_song_feat, np_weights.T))
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
    all_song_df, all_song_feat, scaler = load_all_songs(cnx, normalize=True)
    playlist = UG.get_playlist('mpd.slice.549000-549999.json', 793)
    pl_songs, top_songs, top_dist = get_closest_songs_to_playlist(cnx, playlist, all_song_feat, all_song_df, k=10, scaler=scaler)
    print(top_songs)
    print(top_dist)
    
    


