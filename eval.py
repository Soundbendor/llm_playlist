import os,csv,json

import pandas as pd

import routes as G
import numpy as np
import getter as UG
import metrics as UM
import pre_llm as PL

pl_csv_path = 'data/num_splits/num_tracks-250.csv'
preds_path = 'res/gpt_preds/gpt4-preds.json'
res_path = 'res/gpt_results/gpt4_preds_res.csv'

rec_cols = ['artist_name', 'track_name', 'id']
mheader = ['expr_idx', 'pl_idx', 'r_prec', 'dcg', 'idcg', 'ndcg', 'clicks']

cond_num = 10
gen_num = 250
sample_num = 500

def get_playlists(csv_path, sample_num=500):
    df = pd.read_csv(csv_path)
    cur_playlists = df.to_dict('records')
    
    return cur_playlists[:sample_num]

# debug
# print(get_playlists(pl_csv_path, sample_num=10))
pl = get_playlists(pl_csv_path, sample_num=sample_num)

with open(preds_path, 'r') as pred_file:
    preds = json.load(pred_file)

# for pl in preds:
#     print()
#     for track in pl:
#         print(track['uri'])

runs = []
cnx, cursor = UG.connect_to_nct()

for pl_i, playlist in enumerate(pl):
    print(pl_i)
    print('-----')
    file = playlist['file']
    playlist_idx = int(playlist['idx'])
    gt = UG.get_playlist(file, playlist_idx)
    gt_tracks = gt['tracks'][cond_num:]


    gt_names = [ track['track_name'] for track in gt_tracks ]
    retr_names = [ track['track_name'] for track in preds[pl_i] ]
    # print(gt_names)
    # print(retr_names)

    gt_ids = [ track['track_uri'] for track in gt_tracks ]
    retr_ids = [ track['uri'] for track in preds[pl_i] ]

    gt_ids = np.array(gt_ids)
    retr_ids = np.array(retr_ids)

    r_prec = UM.r_precision(gt_ids, retr_ids)
    dcg = UM.dcg(gt_ids, retr_ids)
    idcg = UM.idcg(gt_ids, retr_ids)
    ndcg = UM.ndcg(gt_ids, retr_ids)
    clicks = UM.rec_songs_clicks(gt_ids, retr_ids, max_clicks=501)
    
    mdict = {'expr_idx': pl_i, 'pl_idx': playlist_idx, 'r_prec': r_prec, 'dcg': dcg, 'idcg': idcg, 'ndcg': ndcg, 'clicks': clicks}
    runs.append(mdict)
    print('r_prec', r_prec)
    print('dcg', dcg)
    print('idcg', idcg)
    print('ndcg', ndcg)
    print('clicks', clicks)

with open(res_path, 'w') as f:
    csvw = csv.DictWriter(f, fieldnames = mheader)
    csvw.writeheader()
    for run in runs:
        csvw.writerow(run)


# Initialize sums
total_r_prec = 0
total_dcg = 0
total_idcg = 0
total_ndcg = 0
total_clicks = 0

# Sum up each metric
for run in runs:
    total_r_prec += run['r_prec']
    total_dcg += run['dcg']
    total_idcg += run['idcg']
    total_ndcg += run['ndcg']
    total_clicks += run['clicks']

# Compute averages
average_r_prec = total_r_prec / len(runs)
average_dcg = total_dcg / len(runs)
average_idcg = total_idcg / len(runs)
average_ndcg = total_ndcg / len(runs)
average_clicks = total_clicks / len(runs)

# Print the averages
print("Average R-Precision:", average_r_prec)
print("Average DCG:", average_dcg)
print("Average IDCG:", average_idcg)
print("Average NDCG:", average_ndcg)
print("Average Clicks:", average_clicks)