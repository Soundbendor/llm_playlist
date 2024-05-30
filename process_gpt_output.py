import json, pprint, re, time

import numpy as np
import pandas as pd

import metrics as UM
import getter as UG
import fuzzy_search as FZ
import post_llm as PL

output_file = ""

def clean_track_name(text):
    cleaned_text = re.sub(r'^\d+\.\s+', '', text)
    return cleaned_text


def extract_tracks_from_response(response_content):
    """
    Extracts track names and artist names from the assistant's response content.
    """
    tracks_and_artists = []
    # response_content = response_content[response_content.find('1.'):]
    
    for line in response_content.split('\n'):
        if len(line) > 0 and line[0].isdigit():
            try:
                parts = re.split(r' - | – ', line, maxsplit=1)
                if len(parts) == 2:
                    track, artist = parts
                else:
                    parts = re.split(r'-|–', line, maxsplit=1)
                    if len(parts) == 2:
                        track, artist = parts
                    else:
                        track, artist = str(parts), ""
                tracks_and_artists.append((clean_track_name(track), artist.strip()))

            except ValueError:
                print(f"Could not parse line: {line}")
    return tracks_and_artists

# File path to the JSONL file
jsonl_file_path = 'gpt_ouput/batch_qFsLkwWLlpH7F53pMorfrlzA_output.jsonl'

# List to hold all extracted track and artist names for each seed playlist
preds = []

# Read and process the JSONL file
with open(jsonl_file_path, 'r') as file:
    for line in file:
        # Load the JSON object from the line
        json_object = json.loads(line)

        file, idx = json_object['custom_id'].split('_')

        # print(file)
        # print(idx)

        # pprint.pprint(json_object, compact=False)
        
        # Extract the response content from the assistant
        response_content = json_object['response']['body']['choices'][0]['message']['content']

        # print(response_content)
        
        # Extract track names and artist names
        tracks = extract_tracks_from_response(response_content)
        preds.append({"file": file, "idx": idx, "tracks": tracks})

songs_path = "/nfs/guille/eecs_research/soundbendor/datasets/playlist_completion/spotify_track_features/songs.csv"
song_df = pd.read_csv(songs_path)
starting_threshold = 80

print(song_df.info())

start_time = time.time()
for i, pred in enumerate(preds):
    print(i)
    file, idx, tracks = pred['file'], pred['idx'], pred['tracks']
    # print(f"file: {file}")
    # print(f"idx: {idx}")
    songs = []

    for track_name, artist_name in tracks:
        # print("track:", track_name)
        # print("artist:", artist_name)
        try:
            matches = FZ.iter_fuzzy_search_song(track_name, artist_name, song_df, threshold=starting_threshold)
            top_match = matches[0]
            songs.append(top_match)
            # for match in matches:
            #     print(f"Track ID: {match['uri']}, Track Name: {match['track_name']}, Artist Name: {match['artist_name']}, Track Score: {match['track_score']}, Artist Score: {match['artist_score']}, Average Score: {match['average_score']}")
        except Exception as e:
            print(f"Error parsing line: {track_name}{artist_name}\nException: {e}")
            songs.append(song_df[0])
    preds[i]['songs'] = songs

end_time = time.time()

with open('gpt-preds.json', 'w') as json_file:
    json.dump(preds, json_file)
print(f"Execution time: {end_time - start_time} seconds")