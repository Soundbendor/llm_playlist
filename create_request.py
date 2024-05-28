import pandas as pd
import os,csv,json,datetime

import getter as UG

df = pd.read_csv('./data/num_tracks-50.csv')

cond_num = 10
gen_num = 100
num_runs = 500

system_message = """SYSTEM: You are an AI music recommender and Playlist completer. 
Given a playlist name and list of song names and artists, you will generate 100 unique recommendations to complete the playlist. 
Ensure the recommendations span various genres and artists to provide a diverse musical experience. Avoid repeating any songs within a playlist.

"""

# print(system_message)

# print(UG.get_playlist(df_10['file'][0], df_10['idx'][0])['name'])
# print(UG.get_playlist(df_10['file'][0], df_10['idx'][0])['tracks'])

csv_dir = os.path.join(__file__.split(os.sep)[0], 'data')
res_dir = os.path.join(__file__.split(os.sep)[0], 'res')
playlist_csvs = ['num_tracks-50.csv']

def get_playlists(sample_num=500):
    cur_playlists = None
    csv_path = os.path.join(csv_dir, playlist_csvs[0])
    with open(csv_path, 'r') as f:
        csvr = csv.DictReader(f)
        cur_playlists = [x for x in csvr]
    
    return cur_playlists[:sample_num]

playlists = get_playlists(sample_num = 500)

requests = []

for idx, playlist in enumerate(playlists[:num_runs]):
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
    input_message = f"Playlist name: {name}\nInput songs and artists: {', '.join(seed_tracks)}\nUSER: Please recommend {gen_num} songs."
    
    request = {
        "custom_id": f"{file}_{playlist_idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_message}
            ]
        }
    }
    requests.append(request)

# Write the JSON objects to a file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'gpt_requests/requests_{timestamp}.jsonl'

with open(filename, 'w') as f:
    for request in requests:
        json.dump(request, f)
        f.write('\n')

print("JSONL file created successfully.")