import getter as UG
import glob as G
import numpy as np
import nltk
import gensim.similarities.fastss as GSF
import pandas as pd
pd.options.mode.chained_assignment = None 
_artists = np.load(G.artists_path, allow_pickle=True)

# get_closest_tracks_by_artists_songs(df,artist,track, k=5, artist_wt = 1., track_wt = 1.) gets the closest tracks by artists and songs, see def below
def get_closest_artists(artist, k=5):
    comp = artist.lower().strip()
    edit_dists = np.array([GSF.editdist(comp, x.lower().strip()) for x in _artists])
    top_idxs = np.argsort(edit_dists)
    top_artists = _artists[top_idxs][:k]
    top_dists = edit_dists[top_idxs][:k]
    return top_artists, top_dists

def get_closest_tracks_by_artists(df,artists, track,k=5):
     filt_artists = df[df['artist_name'].isin(artists)].reset_index(drop=True)
     filt_tracks = filt_artists['track_name'].astype(str).unique()
     comp = track.lower().strip()
     edit_dists = np.array([GSF.editdist(comp, x.lower().strip()) for x in filt_tracks])
     top_idxs = np.argsort(edit_dists)
     top_tracks = filt_tracks[top_idxs][:k]
     top_dists = edit_dists[top_idxs][:k]
     return top_tracks, top_dists, filt_artists

# df is the track_features dataframe (like the results from 'select * from new_combined_table' from joined.db into a pandas df)
# calculates  and filters top k artists by smallest edit dist, calculates top k track names by this filtered result
# returns top k results as smallest total_dist (artist_wt * artist_editdist) + (track_wt * track_editdist) 
def get_closest_tracks_by_artists_songs(df,artist,track, k=5, artist_wt = 1., track_wt = 1.):
    top_artists, top_artist_dists = get_closest_artists(artist, k=k)
    top_tracks, top_track_dists, filt_artists = get_closest_tracks_by_artists(df, top_artists, track, k=k)
    filt_artists.assign(artist_dist= np.inf)
    for _artist,dist in zip(top_artists, top_artist_dists):
        filt_artists.loc[filt_artists['artist_name'] == _artist, 'artist_dist'] = dist * artist_wt
    filt_artists.assign(track_dist=np.inf)
    for _track,dist in zip(top_tracks, top_track_dists):
        filt_artists.loc[filt_artists['track_name'] == _track, 'track_dist'] = dist * track_wt
    filt_artists = filt_artists.assign(total_dist = filt_artists['artist_dist'] + filt_artists['track_dist']).reset_index(drop=True)
    print(filt_artists)
    top_idxs = np.argsort(filt_artists['total_dist'].to_numpy())[:k]
    ret = filt_artists.iloc[top_idxs].reset_index(drop=True)
    return ret




# returns dataframe
def get_closest_songs_by_artist(cnx, artist, song, k=1):
    artist_songs = UG.get_features_by_artist(cnx,artist.strip())
    edit_dists = np.array([nltk.distance.edit_distance(song.strip(), x ) for x in artist_songs['track_name']])
    sorted_dists = np.argsort(edit_dists)
    smallest_dists = edit_dists[sorted_dists]
    top_songs = artist_songs.iloc[sorted_dists]
    top_songs = top_songs.assign(dist = smallest_dists)
    return top_songs[:k]

if __name__ == "__main__":
    #cnx, cursor = UG.connect_to_nct()
    #ret_songs = get_closest_songs_by_artist(cnx, "Prince", "Little Red Beret", k=5)
    #print(ret_songs[['track_name', 'dist']])
    #top_artists = get_closest_artists('jemmy hindrickss')
    _df = pd.read_csv(G.joined_csv_path)
    ret = get_closest_tracks_by_artists_songs(_df,'jemmy hindrix','teh bird crys barry', k=5, artist_wt = 1., track_wt = 1.)
    print(ret)
