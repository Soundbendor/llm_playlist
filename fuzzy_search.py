import pandas as pd
from fuzzywuzzy import fuzz
from multiprocessing import Pool, cpu_count
import re

import getter as UG

# print(song_df.info())

def fuzzy_match(row, track_name, artist_name, threshold):
    track_score = fuzz.token_sort_ratio(track_name, row['track_name'])
    artist_score = fuzz.token_sort_ratio(artist_name, row['artist_name'])
    
    if track_score >= threshold or artist_score >= threshold:
        return {
            'uri': row['uri'],
            'track_name': row['track_name'],
            'artist_name': row['artist_name'],
            'track_score': track_score,
            'artist_score': artist_score,
            'average_score': (track_score + artist_score) / 2
        }
    return None

def fuzzy_search_song(track_name, artist_name, song_df, threshold=70):
    # Initial filtering to narrow down candidates
    ts = re.split(r' ', track_name)
    arts = re.split(r' ', track_name)

    initial_candidates = song_df[
        song_df['track_name'].str.contains(ts[0], case=False, na=False) |
        song_df['artist_name'].str.contains(arts[0], case=False, na=False)
    ]
    if len(initial_candidates) == 0:
        initial_candidates = song_df

    # Prepare arguments for parallel processing
    args = [(row, track_name, artist_name, threshold) for _, row in initial_candidates.iterrows()]

    # Use multiprocessing to speed up fuzzy matching
    with Pool(cpu_count()) as pool:
        results = pool.starmap(fuzzy_match, args)

    # Filter out None results
    results = [result for result in results if result]

    # Sort results by the average score in descending order
    results = sorted(results, key=lambda x: x['average_score'], reverse=True)
    
    return results

def iter_fuzzy_search_song(search_track_name, search_artist_name, song_df, threshold=70):
    matches = []

    while len(matches) == 0 and threshold > 0:
        matches = fuzzy_search_song(search_track_name, search_artist_name, song_df, threshold=threshold)
        threshold -= 30

    if len(matches) == 0:
        print("ERROR: iter_fuzzy_search_song failed. No matches found")

    return matches

if __name__ == "__main__":
    # Example usage
    songs_path = "/nfs/guille/eecs_research/soundbendor/datasets/playlist_completion/spotify_track_features/songs.csv"

    # # Sample song database
    # test_data = {
    #     'uri': ['1', '2', '3', '4', '5'],
    #     'track_name': ['All I Want for Xmas', 'All I Want for Christmas Is You', 'Last Christmas', 'Happy Xmas (War Is Over)', 'Santa Tell Me'],
    #     'artist_name': ['Mariah', 'Mariah Carey', 'Wham!', 'John Lennon & Yoko Ono', 'Ariana Grande']
    # }

    song_df = pd.read_csv(songs_path)

    search_track_name = 'All I Want for Christmas'
    search_artist_name = 'Justin Bieber and Mariah Carey'
    starting_threshold = 70
    matches = iter_fuzzy_search_song(search_track_name, search_artist_name, song_df, threshold=starting_threshold)

    for match in matches:
        print(f"Track ID: {match['uri']}, Track Name: {match['track_name']}, Artist Name: {match['artist_name']}, Track Score: {match['track_score']}, Artist Score: {match['artist_score']}, Average Score: {match['average_score']}")