import csv,json,os
import matplotlib.pyplot as plt
import routes as G
from collections import Counter
import getter as UG
import numpy as np
cj = None

# max lengths conditioning on...
# 0: 50, 1: 78,

# conditioning on length
# < 0.5: 199 (0.49749), >= 0.5: 250 (0.90000)

# conditioning on ratio
#< 0.5: 199 (0.49749), >= 0.5: 100 (0.95000)

res_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data', 'stats')
print(res_path)

def generate_chart(cur_ctr, title="Number of Playlists by Track Length", x_axis = "Track Length", y_axis="Number", out_basename="track_counts"):
    cur_x = np.array(list(cur_ctr.keys()), dtype=int)
    sort_idx = np.argsort(cur_x)
    x_sort = cur_x[sort_idx]
    cur_y = np.array(list(cur_ctr.values()), dtype=int)
    print(title, "\n", np.sum(list(cur_ctr.values())), "tracks")
    y_sort = cur_y[sort_idx]
    subp = plt.subplots()
    plt.bar(x_sort,y_sort)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(res_path, f"{out_basename}.png"))
    plt.clf()
    plt.close()
    with open(os.path.join(res_path, f"{out_basename}.csv"), 'w') as f:
        csvw = csv.writer(f)
        csvw.writerow(['length','count'])
        for l,c in zip(x_sort,y_sort):
            csvw.writerow([l,c])

out_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data')
out_file = os.path.join(out_path, 'challenge_set_stats.csv')
with open(G.challenge_file, 'r') as f:
    cj = json.load(f)

max_len_cond_1 = -1
max_len_cond_0 = -1
max_len_cond_lt50 = -1
max_r_lt50 = -1
max_r_gt50 = -1
max_len_cond_gt50 = -1
cur_bin = -1
cur_ctr = Counter()
bin_str = ""
for _pli, pl in enumerate(cj['playlists']):
    pl_bin = int(_pli/1000)
    if pl_bin != cur_bin:
        print("profiling", pl_bin)
        if cur_bin >= 0:
            #cur_lengths = sorted(cur_ctr.keys())
            cur_out_basename = f"chall-bin_{cur_bin}"
            cur_title = f"Number of Playlists by Track Length\n for {bin_str}"
            generate_chart(cur_ctr, title=cur_title, out_basename=cur_out_basename)
        del cur_ctr
        cur_ctr = Counter()
        cur_bin_val = pl_bin * 1000
        next_bin_val = cur_bin_val + 999
        bin_str = f"{cur_bin_val}-{next_bin_val}"
        cur_bin = pl_bin                
    nt = int(pl['num_tracks'])
    #nh = pl['num_holdouts']
    cur_ctr.update([nt])

   
cur_out_basename = f"chall-bin_{cur_bin}"
cur_title = f"Number of Playlists by Track Length\n for {bin_str}"
generate_chart(cur_ctr, title=cur_title, out_basename=cur_out_basename)

