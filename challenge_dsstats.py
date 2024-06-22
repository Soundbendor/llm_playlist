import csv,json,os
import routes as G
import getter as UG
import numpy as np
cj = None

# max lengths conditioning on...
# 0: 50, 1: 78,

# conditioning on length
# < 0.5: 199 (0.49749), >= 0.5: 250 (0.90000)

# conditioning on ratio
#< 0.5: 199 (0.49749), >= 0.5: 100 (0.95000)



out_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data')
out_file = os.path.join(out_path, 'challenge_set_stats.csv')
with open(G.challenge_file, 'r') as f:
    cj = json.load(f)

header = ['name', 'pid', 'num_tracks', 'num_holdouts', 'holdout_ratio']
max_len_cond_1 = -1
max_len_cond_0 = -1
max_len_cond_lt50 = -1
max_r_lt50 = -1
max_r_gt50 = -1
max_len_cond_gt50 = -1
with open(out_file, 'w') as f:
    csvw = csv.DictWriter(f, fieldnames=header)
    csvw.writeheader()
    for _pli, pl in enumerate(cj['playlists']):
        name = ''
        nt = pl['num_tracks']
        nh = pl['num_holdouts']
        nc = nt - nh
        r = float(nh)/float(nt)
        if nc == 0:
            max_len_cond_0 = max(max_len_cond_0, nt)
        elif nc == 1:
            max_len_cond_1 = max(max_len_cond_1, nt)
        elif r < 0.5:
            if r > max_r_lt50:
                max_len_cond_lt50 = nt
                max_r_lt50 = r
        else:
            if r > max_r_gt50:
                max_len_cond_gt50 = nt
                max_r_gt50 = r
 

        if 'name' in pl.keys():
            name = pl['name']
        d = {'name': name, 'pid': pl['pid'],
             'num_tracks': nt, 
             'num_holdouts': nh,
             'holdout_ratio': r
             }
        csvw.writerow(d)

print('max lengths conditioning on...')
print(f'0: {max_len_cond_0}, 1: {max_len_cond_1}, < 0.5: {max_len_cond_lt50} ({max_r_lt50:.5f}), >= 0.5: {max_len_cond_gt50} ({max_r_gt50:.5f})')








