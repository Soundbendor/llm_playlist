import routes as G
import getter as UG
import os,sys, csv

to_rec = True
res_folder = 'baseline_cossim_filt_100_real'
expr_idx = 20
res_mask_out = f'{res_folder}_mask_{expr_idx}.csv'
res_gt_out = f'{res_folder}_gt_{expr_idx}.csv'
res_guess_out = f'{res_folder}_guess_{expr_idx}.csv'
stats_mask_out = f'{res_folder}_mask_{expr_idx}_stats.csv'
stats_gt_out = f'{res_folder}_gt_{expr_idx}_stats.csv'
stats_guess_out = f'{res_folder}_guess_{expr_idx}_stats.csv'


res_base = os.path.join(__file__.split(os.sep)[0], 'res')
res_dir = os.path.join(res_base, res_folder)
res_guessfile = os.path.join(res_dir, f'guess_{expr_idx}.txt')
res_metricfile = os.path.join(res_dir, 'metrics_by_expr.csv')
data_dir = os.path.join(__file__.split(os.sep)[0], 'data')
valid_csv = os.path.join(data_dir, 'filtered_validation_set.csv')
#song_df = pd.read_csv(G.joined_csv2_path, index_col=[0])
#song_feat, song_tx = UG.all_songs_tx(song_df, normalize=True, pca = 0, seed=cur_seed)

rec_feat =  ['artist_name', 'track_name'] + UG.comp_feat
rec_feat2 =  ['artist_name', 'track_name', 'in_gt'] + UG.comp_feat

cs_weights = {'danceability': 1.0,
            'energy':1.0,
            'key':0.0,
            'loudness':0.0,
            'mode':1.0,
            'speechiness':1.0,
            'acousticness':1.0,
            'instrumentalness':1.0,
            'liveness':1.0,
            'valence':1.0,
            'tempo': 1.0}



want_plinfo = None
with open(valid_csv, 'r') as f:
    csvr = csv.DictReader(f)
    rows = [(ic, row) for ic, row in enumerate(csvr)]
    want_plinfo = rows[expr_idx]

want_res = None
with open(res_metricfile, 'r') as f:
    csvr = csv.DictReader(f)
    rows = [(ic, row) for ic, row in enumerate(csvr)]
    want_res = rows[expr_idx]

print(want_plinfo)
print(want_res)

cond_num = 10
guess_uris = None
with open(res_guessfile, 'r') as f:
    guess_uris = list(f.readlines())
guess_ids = [x.split(':')[-1].strip() for x in guess_uris]
want_pl = UG.get_playlist(want_plinfo[1]['file'], int(want_plinfo[1]['idx']))
pl_uris = [x['track_uri'] for x in want_pl['tracks']]
pl_ids = [x.split(':')[-1].strip() for x in pl_uris]
gt_ids = set(pl_ids[cond_num:])
in_gt = [x in gt_ids for x in guess_ids]

#print(pl_ids)


cnx, cur = UG.connect_to_nct()
guess_feat = UG.get_feat_from_uris(cnx, guess_ids)
guess_feat = guess_feat.assign(in_gt=in_gt)
pl_feat = UG.get_feat_from_uris(cnx, pl_ids)
print(pl_feat[['track_name','artist_name']][:cond_num])
print(guess_feat[['track_name','artist_name', 'in_gt']][:cond_num])
#print(guess_feat)
pl_feat2 = pl_feat[UG.comp_feat][:cond_num].copy()
pl_feat3 = pl_feat[UG.comp_feat][cond_num:].copy()
pl_feat2d = pl_feat2.describe()
pl_feat3d = pl_feat3.describe()
guess_feat2 = guess_feat[UG.comp_feat].copy()
guess_feat2d = guess_feat2.describe()
pl_feat2d.loc['median'] = pl_feat2.median()
pl_feat3d.loc['median'] = pl_feat3.median()
guess_feat2d.loc['median'] = guess_feat2.median()
if to_rec == True:
    pl_feat[rec_feat][:cond_num].to_csv(res_mask_out)
    pl_feat[rec_feat][cond_num:].to_csv(res_gt_out)
    guess_feat[rec_feat2].to_csv(res_guess_out)
    pl_feat2d.to_csv(stats_mask_out)
    pl_feat3d.to_csv(stats_gt_out)
    guess_feat2d.to_csv(stats_guess_out)

#print(pl_feat[:10])
