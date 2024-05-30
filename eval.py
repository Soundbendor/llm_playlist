import os,csv,json

import pandas as pd

import glob as G
import numpy as np
import getter as UG
import metrics as UM
import pre_llm as PL

pl_csv_path = 'data/num_tracks-50.csv'
preds_path = 'res/gpt-preds/c10_d50_g100.json'
res_dir = 'res/gpt-test-1'

rec_cols = ['artist_name', 'track_name', 'id']
mheader = ['expr_idx', 'pl_idx', 'r_prec', 'dcg', 'idcg', 'ndcg', 'clicks']

cond_num = 10
gen_num = 100
sample_num = 500

def print_json_structure(data, indent=2):
    """Recursively prints the structure of the JSON data."""
    for key, value in data.items():
        print(' ' * indent + str(key))
        if isinstance(value, dict):
            print_json_structure(value, indent + 4)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            print(' ' * (indent + 4) + "List of:")
            print_json_structure(value[0], indent + 8)

def get_playlists(csv_path, sample_num=500):
    df = pd.read_csv(csv_path)
    cur_playlists = df.to_dict('records')
    
    return cur_playlists[:sample_num]

# debug
# print(get_playlists(pl_csv_path, sample_num=10))
pl = get_playlists(pl_csv_path, sample_num=sample_num)

with open('file.json', 'r') as file:
    preds = json.load(preds_path)

print_json_structure(preds)

runs = []
cnx, cursor = UG.connect_to_nct()

for pl_i, pl_dict in enumerate(pl):
    print(pl_i)
    print('-----')
    truth_ids = pl_dict['tracks']['track_uri']
    retr_ids = preds[pl_i]['track_uri'].to_numpy()

    r_prec = UM.r_precision(truth_ids, retr_ids)
    dcg = UM.dcg(truth_ids, retr_ids)
    idcg = UM.idcg(truth_ids, retr_ids)
    ndcg = UM.ndcg(truth_ids, retr_ids)
    clicks = UM.rec_songs_clicks(truth_ids, retr_ids, max_clicks=99998)
    
    g_path = os.path.join(res_dir, f'ground_truth-{cond_num}_{gen_num}_{sample_num}-{pl_i}.csv')
    r_path = os.path.join(res_dir, f'retrieved-{cond_num}_{gen_num}_{sample_num}-{pl_i}.csv')
    
    pl_dict['tracks'][rec_cols].to_csv(g_path)
    preds[pl_i].to_csv(r_path)
    mdict = {'expr_idx': pl_i, 'pl_idx': int(pl_dict['idx']), 'r_prec': r_prec, 'dcg': dcg, 'idcg': idcg, 'ndcg': ndcg, 'clicks': clicks}
    runs.append(mdict)
    print('r_prec', r_prec)
    print('dcg', dcg)
    print('idcg', idcg)
    print('ndcg', ndcg)
    print('clicks', clicks)

    with open(res_dir, 'w') as f:
        csvw = csv.DictWriter(f, fieldnames = mheader)
        csvw.writeheader()
        for run in runs:
            csvw.writerow(run)