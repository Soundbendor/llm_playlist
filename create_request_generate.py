import pandas as pd
import os,csv,json,datetime

import getter as UG

cond_num = 10
gen_num = 100
num_samples = 100

system_message = f"You are an AI playlist completer.\n\
Given a Spotify playlist name and list of tracks and artists, you will generate {gen_num} unique recommendations to complete the playlist.\n\
Only recomend songs released before 2018.\n\
Follow this format:\n\
1. track name - artist\n\
2. track name - artist\n\
"

train_path = "data/filtered_validation_set.csv"
results_dir = "res/"
model = "gpt-3.5-turbo"

def get_playlists(csv_path, sample_num=500):
    df = pd.read_csv(csv_path)
    cur_playlists = df.to_dict('records')
    
    return cur_playlists[:sample_num]

playlists = get_playlists(train_path, sample_num = num_samples)

requests = []

for playlist in playlists:
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
    # print(seed_tracks)
    input_message = f"Playlist name: {name}\n\Input songs and artists: {', '.join(seed_tracks)}\nUSER: Please recommend {gen_num} songs."
    
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