import getter as UG
import numpy as np
import sklearn.metrics as SKM

# input: playlist, output: candidate song ids

cnx, cursor = UG.connect_to_nct()
all_feat = UG.get_feat_all_songs(cnx)

# options for metric: cityblock, cosine, euclidean, l1, l2, manhattan
# returns top songs as pandas dataframe, and distances
def get_closest_songs_to_playlist(_cnx, playlist,metric='euclidean', k=500):
    playlist_feat = UG.get_feat_playlist(_cnx, playlist)
    # results will be i,j for ith playlist song and jth song (from everything)
    pwdist = SKM.pairwise_distances(playlist_feat[UG.comp_feat].to_numpy(), all_feat[UG.comp_feat].to_numpy())
    # average over all playlist songs to get average distances to each song
    all_song_dist = np.mean(pwdist,axis=0)
    sorted_idx = np.argsort(all_song_dist)
    if metric == 'cosine':
        sorted_idx = sorted_idx[::-1]
    top_dist = all_song_dist[sorted_idx[:k]]
    top_songs = all_feat.iloc[sorted_idx[:k]]
    return top_songs, top_dist

if __name__ == "__main__":
    playlist = UG.get_playlist('mpd.slice.549000-549999.json', 793)
    top_songs, top_dist = get_closest_songs_to_playlist(cnx, playlist, k=10)
    print(top_songs)
    print(top_dist)
    
    


