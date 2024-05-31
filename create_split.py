# create_split.py
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

data_dir = 'data/num_splits/'
all_files = glob.glob(data_dir + "/*.csv")
print("Files found:", all_files)

# concatinate playlists into df
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    if df.empty:
        print(f"Skipping empty file: {filename}")  # Notify about an empty file
    else:
        li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

df.to_csv('data/combined.csv', index=False)

train, validate = train_test_split(df, test_size=0.10, random_state=42)

# Step 4: Save the training and validation DataFrames to CSV files
train.to_csv('data/train_set.csv', index=False)
validate.to_csv('data/validation_set.csv', index=False)