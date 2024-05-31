# filter_songs.py
import pandas as pd
import routes as G

# Load the CSV files into DataFrames
songs_df = pd.read_csv(G.songs_path)
pop_df = pd.read_csv("data/stats/popularity.csv")
train_pop_df = pd.read_csv('data/stats/train_popularity.csv')

# remove songs that do not appear in the ENTIRE dataset
merged_df = pd.merge(songs_df, train_pop_df, on='uri', how='inner')
filtered_df = merged_df[merged_df['count'] > 0]

print(filtered_df.head())

# filtered_df.to_csv(G.fsongs_path, index=False)