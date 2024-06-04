import pandas as pd
import os,csv,json,datetime

import getter as UG
import routes as G

cond_num = 10
gen_num = 100
num_samples = 100
context_num = 500

train_path = "data/filtered_validation_set.csv"
results_dir = "res/"
model = "gpt-4o"
songs_pth = G.songs_path
context_dir = "res/baseline_bm25_filt_500_real"

system_message = f"You are an AI playlist completer. \n\
Given a list of {context_num} tracks, pick the best {gen_num} tracks to complete the the playlist.\n\
Follow this format:\n\
1. track name - artist\n\
2. track name - artist\n\
... \n\
100. track name - artist\n\
"

songs_df = pd.read_csv(songs_pth, index_col=None, header=0)
songs_df.set_index(["uri"], inplace=True)

# Initialize a list to store the lists of URLs
candidates = []

# Loop through each .txt file in the context directory
for filename in os.listdir(context_dir):
    if filename.endswith('.txt'):
        print(filename)
        file_path = os.path.join(context_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            pl_cands = []
            for line in file:
                uri = line[:-1]
                track = songs_df.loc[uri]
                track_name = track["track_name"]
                artist_name = track["artist_name"]
                str_track_artist = f"{track_name} - {artist_name}"
                pl_cands.append(str_track_artist)
            print('\n'.join(pl_cands))
            candidates.append('\n'.join(pl_cands))

def get_playlists(csv_path, sample_num=500):
    df = pd.read_csv(csv_path)
    cur_playlists = df.to_dict('records')
    
    return cur_playlists[:sample_num]

playlists = get_playlists(train_path, sample_num = num_samples)

requests = []

for idx, playlist in enumerate(playlists):
    print(playlist)
    name = playlist['name']
    file = playlist['file']
    playlist_idx = int(playlist['idx'])
    tracks = UG.get_playlist(file, playlist_idx)['tracks'][:cond_num]
    # print(name)
    # Loop through each track in the playlist
    seed_tracks = []
    for track in tracks:
        # print(track)
        artist_name = track['artist_name']
        track_name = track['track_name']
        seed_tracks.append(track_name + " - " + artist_name)
        # print(f"Artist: {artist_name}, Track: {track_name}")
    print(seed_tracks)
    seed_str = ', '.join(seed_tracks)
    input_message = f"Pick from these songs: {candidates[idx]} to complete this playlist:\nPlaylist name: {name} Begining of playlist:{seed_str}\n List 100 songs by relivence to begining of playlist."
    print(input_message)
    request = {
        "custom_id": f"{file}_{playlist_idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_message}
            ]
        }
    }
    requests.append(request)

# Write the JSON objects to a file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'gpt_requests/{model}_{timestamp}.jsonl'

with open(filename, 'w') as f:
    for request in requests:
        json.dump(request, f)
        f.write('\n')

print(f"JSONL file created successfully {filename}.")