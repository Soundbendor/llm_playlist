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
cnx, cur = UG.connect_to_nct()
track_ids = [x.split(':')[-1] for x in all_uris]
print('got track_ids')
num_tracks = len(track_ids)
for ididx, _id in enumerate(track_ids):
    print(f'getting {ididx+1}/{num_tracks}')
    if ididx == 0:
        song_feat = UG.get_features_by_id(cnx, _id)
        #print(song_feat)
    else:
        cur_feat = UG.get_features_by_id(cnx, _id)
        #print(cur_feat)
        song_feat = pd.concat([song_feat, cur_feat]).reset_index(drop=True)
        #print(song_feat)
#song_feat = [UG.get_features_by_id(cnx, _id) for _id in track_ids]
song_feat.to_csv(G.joined_csv2_path)
print('got features')

