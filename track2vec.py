# track2vec.py
import time, csv, os
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

import getter as UG

train_path = 'data/train_set.csv'
val_path = 'data/filtered_validation_set.csv'

model_path = "models/track2vec.model"

train_uri_csv = "data/train_set_uris.csv"
val_uri_csv = "data/filtered_validation_set_uris.csv"

_cond_num = 10
_gen_num = 100

# Function to get the vector representation of a document by averaging word vectors
def get_mask_vector(mask, model):
    """ Get vector representation for a document by averaging word vectors. """
    doc_vectors = [model.wv[song] for song in mask if song in model.wv]
    if len(doc_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(doc_vectors, axis=0)

# ==================================
#             DATASETS
# ==================================
print("dfs loading...")
train_gen = UG.playlist_csv_generator("train_set.csv")
val_gen = UG.playlist_csv_generator("validation_set.csv")
print("dfs loaded")
# train_df = pd.read_csv(train_path)
# val_df = pd.read_csv(val_path)

train_uris = []
val_uris = []

# Train
if os.path.exists(train_uri_csv):
    with open(train_uri_csv, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            train_uris.append(row)
else:
    for playlist in train_gen:
        file = playlist['file']
        playlist_idx = int(playlist['idx'])
        # print("file:", file)
        # print("idx:", playlist_idx)
        try:
            tracks = UG.get_playlist(file, playlist_idx)['tracks']
            uris = [track['track_uri'] for track in tracks]
            train_uris.append(uris)
        except Exception as e:
            # Print the file and playlist index if an error occurs
            print(f"An error occurred with file: {file} and playlist_idx: {playlist_idx}")
            print(f"Error details: {e}")
        # print(uris)
        if len(train_uris) % 1000 == 0:
            print(f"time: {time.time()} completed: {len(train_uris)}")
        

# Validation
if os.path.exists(val_uri_csv):
    with open(val_uri_csv, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            val_uris.append(row)
else:
    for playlist in val_gen:
        # print(playlist)
        file = playlist['file']
        playlist_idx = int(playlist['idx'])
        # print("file:", file)
        # print("idx:", playlist_idx)
        try:
            tracks = UG.get_playlist(file, playlist_idx)['tracks']
            uris = [track['track_uri'] for track in tracks]
            val_uris.append(uris)
        except Exception as e:
            # Print the file and playlist index if an error occurs
            print(f"An error occurred with file: {file} and playlist_idx: {playlist_idx}")
            print(f"Error details: {e}")
        # print(uris)
        if len(val_uris) % 100 == 0:
            print(f"time: {time.time()} completed: {len(val_uris)}")

print("tracks loaded.")
with open(train_uri_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(train_uris)

with open(val_uri_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(val_uris)



# ==================================
#             Model
# ==================================

# subset = train_uris[:10]
# print(subset)
# if os.path.exists(model_path):
#     track2vec_model = Word2Vec.load(model_path)
# else:
#     start_time = time.time()
#     track2vec_model = Word2Vec()
#     track2vec_model = Word2Vec(vector_size=512, window=5, min_count=1, workers=1, epochs=10, seed=73)
#     track2vec_model.build_vocab(subset)
#     track2vec_model.train(subset, total_examples=len(subset), epochs=track2vec_model.epochs)
#     track2vec_model.save(model_path)
#     training_time = time.time() - start_time
#     print(f"Training time: {training_time} seconds")

# track_vectors = [ track2vec_model.wv[x] for x in train_uris ]

# # Test queries
# masks = [[
#     'spotify:track:2fl0B0OaXjWbjHCQFx2O8W', 
#     'spotify:track:2a1o6ZejUi8U3wzzOtCOYw', 
#     'spotify:track:3w1D8eAOBDZdb8RP5wbV65', 
#     'spotify:track:4Tid4MwqgR1CfKCun3tFon', 
#     'spotify:track:09WvikcE8OuI7Si8Gj1lV0', 
#     'spotify:track:2NBgfMo1xvE7D3o18zirjB', 
#     'spotify:track:0ytCr0BRergKfjMYayYycg', 
#     'spotify:track:1qxfdtshzyOo9JwdPbvMYL', 
#     'spotify:track:63daATqOdx3odypMZP2eUt'
# ]]

# # Retrieve top 3 documents for each query based on cosine similarity
# print("\\begin{itemize}[left=0pt, label={}, itemsep=2pt]")
# for mask in masks:
#     cond_num = len(mask)
#     similarities = track2vec_model.wv.most_similar(positive = mask, topn = cond_num + _gen_num)

#     print(f"\t============== Query: {mask} ==============")
#     for uri, simularity in similarities:
#         print(f"URI: {uri}")
#         print(f"Similarity Score: {simularity}")