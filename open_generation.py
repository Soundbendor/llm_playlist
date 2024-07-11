import os,csv,json,datetime

import torch
import transformers
import pandas as pd

import getter as UG
import routes as G

cond_num = 10
gen_num = 100
num_samples = 100
context_num = 500

system_message = f"You are an AI playlist completer.\n\
Given a Spotify playlist name and list of tracks and artists, you will generate {gen_num} unique recommendations to complete the playlist.\n\
Only recomend songs released before 2018.\n\
Follow this format:\n\
1. track name - artist\n\
2. track name - artist\n\
"

val_path = "data/filtered_validation_set.csv"
results_dir = "res/"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = "lmala-3-8B"
hf_token = "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"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    token= hf_token,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def get_playlists(csv_path, sample_num=500):
    df = pd.read_csv(csv_path)
    cur_playlists = df.to_dict('records')
    
    return cur_playlists[:sample_num]

playlists = get_playlists(val_path, sample_num = num_samples)

res = []

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
    
    message = system_message + "\n" + input_message
    print(message)
    exit(0)