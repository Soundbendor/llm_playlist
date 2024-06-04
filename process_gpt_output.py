import nltk, re, json

from tqdm import tqdm

import gensim.similarities.fastss as GSF
import numpy as np
import pandas as pd

import getter as UG
import routes as G

jsonl_file_path = 'gpt_ouput/bm25_500_gpt4o_corrected.jsonl'
res_path = 'res/gpt_preds/bm25_500_gpt4o_corrected.json'

songs_path = G.songs_path

print(jsonl_file_path)
print(res_path)

pd.options.mode.chained_assignment = None 
_artists = np.load(G.artists_path, allow_pickle=True)

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
                # Remove anything after 'ft.' in the artist string
                artist = re.split(r' ft\.| Ft\.', artist, maxsplit=1)[0]
                # Remove leading numbers and dots
                track = re.sub(r'^\d+\.\s*', '', track)
                # Remove surrounding quotation marks if they are there
                track = re.sub(r'^"(.*)"$', r'\1', track)

                tracks_and_artists.append((clean_track_name(track), artist.strip()))

            except ValueError:
                print(f"Could not parse line: {line}")

            # print(f"LINE: {line}")
            # print(f"TRACK: {track} - ARTIST: {artist}")
            # print()
    # exit(0)

    return tracks_and_artists

def process_gpt_jsonl(jsonl_file_path):
    # List to hold all extracted track and artist names for each seed playlist
    preds = []

    # Read and process the JSONL file
    with open(jsonl_file_path, 'r') as file:
        for i, line in enumerate(file):
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
        
    return preds

# get_closest_tracks_by_artists_songs(df,artist,track, k=5, artist_wt = 1., track_wt = 1.) gets the closest tracks by artists and songs, see def below
def get_closest_artists(artist, k=5):
    comp = artist.lower().strip()
    edit_dists = np.array([GSF.editdist(comp, x.lower().strip()) for x in _artists])
    top_idxs = np.argsort(edit_dists)
    top_artists = _artists[top_idxs][:k]
    top_dists = edit_dists[top_idxs][:k]
    return top_artists, top_dists

def clean_track(track):
    return re.split(r' ft\.| Ft\.| feat\. |Feat\.', track, maxsplit=1)[0]

def get_closest_tracks_by_artists(df,artists, track,k=5):
     filt_artists = df[df['artist_name'].isin(artists)].reset_index(drop=True)
     filt_tracks = filt_artists['track_name'].astype(str).apply(lambda x: clean_track(x)).unique()
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
    # print(filt_artists)
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
    _df = pd.read_csv(songs_path)
    # ret = get_closest_tracks_by_artists_songs(_df,'jemmy hindrix','teh bird crys barry', k=5, artist_wt = 1., track_wt = 1.)
    # print(ret)

    playlist_preds = process_gpt_jsonl(jsonl_file_path)
    # print(preds[0]['tracks'])

    res = []

    for i, playlist_pred in tqdm(enumerate(playlist_preds), total=len(playlist_preds), desc="Processing predictions"):
        file, idx, tracks = playlist_pred['file'], playlist_pred['idx'], playlist_pred['tracks']
    
        songs = []

        for track_name, artist_name in tracks:
            # try:
            matches = get_closest_tracks_by_artists_songs(_df,artist_name,track_name, k=10)
            
            print(f"TRACK: {track_name} - ARTIST: {artist_name}")
            print(matches)
            print()
            
            top_match = matches.iloc[0]  # NOTE: change this to pick the most popular match (Search subset where track distance is min and the same)
            # NOTE: I think we should weigh track title more than artist name. While debugging I found that track title was more accurate
            songs.append(top_match['uri'])
            # except Exception as e:
            #     print(f"Error parsing line: {track_name} - {artist_name}\nException: {e}")
        res.append(songs)

    with open(res_path, 'w') as json_file:
        json.dump(res, json_file)