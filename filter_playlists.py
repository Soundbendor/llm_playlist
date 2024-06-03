# filter_songs.py
import pandas as pd
import routes as G

# Load the CSV files into DataFrames
val_df = pd.read_csv("data/validation_set.csv")
print(val_df)
filtered_df = val_df[val_df['num_tracks'] > 30]
print(filtered_df)
# Save the final DataFrame to a new CSV file
filtered_df.to_csv("data/filtered_validation_set.csv", index=False)