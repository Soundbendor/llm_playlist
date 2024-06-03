import os,csv,json

import pandas as pd

import routes as G
import numpy as np
import getter as UG
import metrics as UM
import pre_llm as PL

expr_name = "gpt4o_val_100"

# pl_csv_path = 'data/num_splits/num_tracks-250.csv'
# playlists_pth = "data/train_set.csv"
playlists_pth = "data/validation_set.csv"

preds_path = f'res/gpt_preds/{expr_name}.json'
res_path = f'res/gpt_results/{expr_name}_res.csv'
avg_res_path = f'res/gpt_results/{expr_name}_avg_res.csv'
res_track_names_path = f'res/gpt_results/{expr_name}_track_names_res.csv'

songs_pth = G.fsongs_path

rec_cols = ['artist_name', 'track_name', 'id']
mheader = ['expr_idx', 'pl_idx', 'r_prec', 'dcg', 'idcg', 'ndcg', 'clicks']

cond_num = 10
gen_num = 100

# results saved to         Metric   Average
# 0  R-Precision  0.042122
# 1          DCG  1.028230
# 2         IDCG  2.378920
# 3         NDCG  0.289734
# 4       Clicks  1.063830
# Average R-Precision: 0.04212188186435864
# Average DCG: 1.0282300211773296
# Average IDCG: 2.3789199893065223
# Average NDCG: 0.2897342069702966
# Average Clicks: 1.0638297872340425

songs_df = pd.read_csv(songs_pth, index_col=None, header=0)
with open(preds_path, 'r') as pred_file:
    preds = json.load(pred_file)
sample_num = len(preds)
pls = pd.read_csv(playlists_pth, nrows=sample_num).to_dict('records')

runs = []
res_track_names = []
cnx, cursor = UG.connect_to_nct()

for pl_i, playlist in enumerate(pls):
    print(pl_i)
    print('-----')
    file = playlist['file']
    playlist_idx = int(playlist['idx'])
    gt = UG.get_playlist(file, playlist_idx)
    gt_tracks = gt['tracks'][cond_num:]

    gt_ids = [ track['track_uri'] for track in gt_tracks ]
    if len(gt_ids) == 0:
        continue
    retr_ids = preds[pl_i][:gen_num]
    # print(gt_ids)
    # print(retr_ids)

    # gt_names = [ track['track_name'] for track in gt_tracks ]
    # retr_names = [ songs_df.loc[songs_df['uri'] == uri, 'track_name'].values[0] for uri in preds[pl_i] ]
    # print(gt_names)
    # print(retr_names)
    # for gt_name, retr_name in zip(gt_names, retr_names):
    #     res_track_names.append({'Playlist Index': pl_i, 'GT Name': gt_name, 'Retrieved Name': retr_name})

    gt_ids = np.array(gt_ids)
    retr_ids = np.array(retr_ids)

    r_prec = UM.r_precision(gt_ids, retr_ids)
    dcg = UM.dcg(gt_ids, retr_ids)
    idcg = UM.idcg(gt_ids, retr_ids)
    ndcg = UM.ndcg(gt_ids, retr_ids)
    clicks = UM.rec_songs_clicks(gt_ids, retr_ids, max_clicks=(gen_num//10)+1)

    mdict = {'expr_idx': pl_i, 'pl_idx': playlist_idx, 'r_prec': r_prec, 'dcg': dcg, 'idcg': idcg, 'ndcg': ndcg, 'clicks': clicks}
    runs.append(mdict)
    print('gt_len', len(gt_ids))
    print('retr_len', len(retr_ids))
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

# res_track_names_df = pd.DataFrame(res_track_names)
# res_track_names_df.to_csv(res_track_names_path, index=False)

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

data = {
    'Metric': ['R-Precision', 'DCG', 'IDCG', 'NDCG', 'Clicks'],
    'Average': [average_r_prec, average_dcg, average_idcg, average_ndcg, average_clicks]
}
avg_res_df = pd.DataFrame(data)

avg_res_df.to_csv(avg_res_path, index=False)
print(f"results saved to {avg_res_df}")

# Print the averages
print("Average R-Precision:", average_r_prec)
print("Average DCG:", average_dcg)
print("Average IDCG:", average_idcg)
print("Average NDCG:", average_ndcg)
print("Average Clicks:", average_clicks)

# ==========================================
#               GPT-3.5
# ==========================================
# Average R-Precision: 0.005194369973190343
# Average DCG: 0.3028171112828822
# Average IDCG: 1.5419510201942437
# Average NDCG: 0.10703044983499345
# Average Clicks: 33.34048257372654

# ==========================================
#               GPT-4
# ==========================================
# Average R-Precision: 0.046715817694369915
# Average DCG: 2.5058006113749993
# Average IDCG: 5.0890395306519265
# Average NDCG: 0.41323149562533745
# Average Clicks: 3.6380697050938338
