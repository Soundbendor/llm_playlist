import os,csv,json,time
import metrics as UM
import getter as UG
import numpy as np
import routes as G
import sklearn.metrics as SKM
import sklearn.cluster as SKC
import gensim.corpora as GC
import gensim.models as GM
import pandas as pd

# condition on 10, generate 100, should be at least 20 songs length
# 500 samples
data_dir = os.path.join(__file__.split(os.sep)[0], 'data')
eval_csv = os.path.join(data_dir, 'validation_set.csv')
pop_dir = os.path.join(data_dir, 'stats')
#res_dir = os.path.join(__file__.split(os.sep)[0], 'res')
res_dir = '/media/dxk/TOSHIBA EXT/llm_playlist_res'
#playlist_csvs = list(os.listdir(csv_dir))
#num_csvs = len(playlist_csvs)



# popularity csv format: uri, count
def get_popularity_uris(pop_file = 'popularity.csv', csv_dir = pop_dir):
    uris = []
    # uri, count
    with open(os.path.join(csv_dir, pop_file), 'r') as f:
        csvr = csv.DictReader(f)
        uris = [x['uri'] for x in csvr]
    return np.array(uris)

all_uris = get_popularity_uris()
track_ids = [x.split(':')[-1].strip() for x in all_uris]
print('got track_ids')
song_feat = pd.read_csv(G.joined_csv_path, index_col=[0])
song_feat = song_feat.set_index('id')
groups = song_feat.groupby(level=0)
song_feat = groups.first()
#not_in = [x for x in track_ids if x not in song_feat.index]
#print(len(not_in))
track_ids = [x for x in track_ids if x in song_feat.index]
song_feat = song_feat.loc[track_ids].reset_index()
song_feat.to_csv(G.joined_csv2_path)
#song_feat = [UG.get_features_by_id(cnx, _id) for _id in track_ids]
#song_feat.to_csv(G.joined_csv2_path)
#print('got features')
#song_feat = pd.concat([song_feat])
#print('concat')
#track_idxloc =  [song_feat.loc[song_feat['id'] == y].index[0] for y in track_ids]
#print('got locations')
#song_feat = song_feat.iloc[track_idxloc].reset_index(drop=True)
#print(track_ids)
#print(song_feat)
#print(song_feat['id'])

