import os,csv

res_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data', 'stats')
#pool_file = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data', 'filtered_validation_set.csv')
num_tracks_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data', 'num_splits')
#out_path = "/media/dxk/TOSHIBA EXT/llm_playlist_res/valid"
out_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'valid_retrain2')

if os.path.exists(out_path) == False:
    os.mkdir(out_path)


header = ['pid', 'idx', 'file', 'num_tracks']

used_pid = set()
with open(os.path.join(out_path, 'valid_pids.txt'), 'r') as f:
    used_pid = set(list([int(x.strip()) for x in f.readlines()]))

with open(os.path.join(out_path, 'train_pids.csv'), 'w') as f:
    csvw = csv.DictWriter(f, fieldnames=header)
    csvw.writeheader()
    for cur_f in os.listdir(num_tracks_path):
        pool_file = os.path.join(num_tracks_path, cur_f)
        #print(pool_file)
        with open(pool_file, 'r') as f:
            csvr = csv.DictReader(f)
            for row in csvr:
                cur_tracks = int(row['num_tracks'])
                cur_pid = int(row['pid'])
                cur_file = row['file']
                cur_idx = int(row['idx'])
                cur_dict = {'pid': cur_pid, 'file': cur_file, 'num_tracks': cur_tracks, 'idx': cur_idx}
                if cur_pid not in used_pid:
                    csvw.writerow(cur_dict)
                
