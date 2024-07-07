import csv, os
import metrics as UM
expr = "bm25"
gen_num = 500




res_dir = os.path.join(__file__.split(os.sep)[0], 'res')
res_path = os.path.join(res_dir, f'bline-chall_{expr}_{gen_num}_retrain2_full4_joinedstitched')
header = ["r_precision","ndcg","clicks","rr","recall"]

chall_avgarr = []
for file_idx in range(1,10):
    cur_fname_avg = f'chall-bin_{file_idx}-resavg.csv'
    fname_avg_path = os.path.join(res_path, cur_fname_avg)
    with open(fname_avg_path, 'r') as f:
        csvr = csv.DictReader(f)
        for row in csvr:
            chall_avgarr.append({x:float(y) for (x,y) in row.items()})


overall_avg = UM.get_mean_metrics(chall_avgarr)
cur_fname2 = f'overall-res.csv'
cur_fname_avg2 = f'overall-resavg.csv'
       
UM.metrics_writer(chall_avgarr, fname=cur_fname2, fpath=res_path)
UM.metrics_writer([overall_avg], fname=cur_fname_avg2, fpath=res_path)
