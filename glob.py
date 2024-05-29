import os

db_dir = "/media/dxk/TOSHIBA EXT/ds/spotify_track_features"
data_dir = "/media/dxk/TOSHIBA EXT/ds/spotify_mpd/data"
data_path = os.path.join(data_dir, 'data')
joined_db_path = os.path.join(db_dir, "joined.db")
joined_csv_path = os.path.join(db_dir, 'joined.csv')
artists_path = os.path.join(db_dir, 'artists.npy')
tracks_path = os.path.join(db_dir, 'tracks.npy')
num_tracks_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data')
