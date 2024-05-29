import getter as UG
import glob as G
import numpy as np
import nltk
import gensim.similarities.fastss as GSF
import pandas as pd

_artists = np.load(G.artists_path, allow_pickle=True)

def get_closest_artists(artist, k=5):
    comp = artist.lower().strip()
    edit_dists = np.array([GSF.editdist(comp, x.lower().strip()) for x in _artists])
    top_idxs = np.argsort(edit_dists)
    top_artists = _artists[top_idxs][:k]
    top_dists = edit_dists[top_idxs][:k]
    return top_artists, top_dists

def get_closest_tracks_by_artists(df,artists, track,k=5):
     filt_artists = df[df['artist_name'].isin(artists)]
     filt_tracks = filt_artists['track_name'].astype(str).unique()
     comp = track.lower().strip()
     edit_dists = np.array([GSF.editdist(comp, x.lower().strip()) for x in filt_tracks])
     top_idxs = np.argsort(edit_dists)
     top_tracks = filt_tracks[top_idxs]





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
    top_artists = get_closest_artists('jemmy hindrickss')
    #_df = pd.read_csv(G.joined_csv_path)
    print(top_artists)
