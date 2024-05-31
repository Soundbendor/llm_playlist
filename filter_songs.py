# filter_songs.py
import pandas as pd
import routes as G

# Load the CSV files into DataFrames
songs_df = pd.read_csv(G.songs_path)
pop_df = pd.read_csv("data/stats/popularity.csv")
# train_pop_df = pd.read_csv('data/stats/train_popularity.csv')

# remove songs that do not appear in the ENTIRE dataset
merged_df = pd.merge(songs_df, pop_df, on='uri', how='inner')
filtered_df = merged_df[merged_df['count'] > 0]

print(filtered_df.head())

# filtered_df.to_csv(G.fsongs_path, index=False)

filtered_df['track_artist'] = filtered_df['track_name'] + ' - ' + filtered_df['artist_name']
filtered_df = filtered_df.sort_values(by=['track_artist', 'count'], ascending=[True, False])
final_df = filtered_df.drop_duplicates(subset='track_artist', keep='first')
final_df = final_df.drop(columns=['track_artist'])

# Save the final DataFrame to a new CSV file
final_df.to_csv(G.fsongs_path, index=False)