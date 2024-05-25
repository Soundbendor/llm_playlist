import getter as UG
import numpy as np
import nltk


# returns dataframe
def get_closest_songs_by_artist(cnx, artist, song, k=1):
    artist_songs = UG.get_features_by_artist(cnx,artist.strip())
    edit_dists = np.array([nltk.distance.edit_distance(song, x ) for x in artist_songs['track_name']])
    sorted_dists = np.argsort(edit_dists)
    smallest_dists = edit_dists[sorted_dists]
    top_songs = artist_songs.iloc[sorted_dists]
    top_songs = top_songs.assign(dist = smallest_dists)
    return top_songs[:k]

if __name__ == "__main__":
    cnx, cursor = UG.connect_to_nct()
    ret_songs = get_closest_songs_by_artist(cnx, "Prince", "Little Red Beret", k=5)
    print(ret_songs[['track_name', 'dist']])
