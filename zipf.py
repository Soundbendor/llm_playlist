import os,csv,json,time
import metrics as UM
import getter as UG
import numpy as np
import routes as G
import sklearn.metrics as SKM
import sklearn.cluster as SKC
import gensim.corpora as GC
import gensim.models as GM
import gensim.similarities as GS
import gensim.test.utils as GT
import pandas as pd
import pre_llm as PL
import matplotlib.pyplot as plt
from scipy.stats import linregress


data_dir = os.path.join(__file__.split(os.sep)[0], 'data')
pop_dir = os.path.join(data_dir, 'stats')
pop_path = os.path.join(pop_dir, 'popularity.csv')
res_dir = os.path.join(__file__.split(os.sep)[0], 'res')

def parse_counts(counts):
    cnum = counts.shape[0]
    counts2 = np.array([np.log10(x) for x in counts])
    x_arr = np.arange(1,cnum+1)
    x_arr2 = np.log10(x_arr)
    res = linregress(x_arr2, counts2)
    return res, x_arr, x_arr2, counts2

def get_line_pts(cur_x, cur_m, cur_b):
    return np.power(10, cur_x * cur_m + cur_b)


uris = []
counts = []
with open(pop_path, 'r') as f:
    csvr = csv.DictReader(f)
    for row in csvr:
        uris.append(row['uri'])
        counts.append(int(row['count']))

uris = np.array(uris)
counts = np.array(counts)
idx_sort = np.argsort(counts)[::-1]
top_counts = counts[idx_sort]
top_uris = uris[idx_sort]
num_uris = uris.shape[0]

cur_res, cur_x, cur_xlog, cur_countslog = parse_counts(top_counts)
cur_m = cur_res.slope
cur_b = cur_res.intercept
cur_m_err = cur_res.stderr
cur_b_err = cur_res.intercept_stderr
title ="Track URI frequency by Rank" 
title_add = f"\nm: {cur_m:.7f} (err:{cur_m_err:.7f}), b: 10^{cur_b:.7f} (err:{cur_b_err:.7f})"
title += title_add
line_y = get_line_pts(cur_xlog, cur_m, cur_b)
subp = plt.subplots()
subp[1].set_xscale('log')
subp[1].set_yscale('log')
plt.plot(cur_x, top_counts, color='blue', label='counts')
plt.plot(cur_x, line_y, color='red', label='fitted line')
plt.legend(loc="upper right")
plt.xlabel("Rank (log)")
plt.ylabel("Counts (log)")
plt.suptitle(title)
plt.tight_layout()
plt.savefig(os.path.join(res_dir, "zipf.png"))
plt.clf()
plt.close()

