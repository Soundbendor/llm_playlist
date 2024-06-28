import os,csv
import getter as G

res_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'valid_retrain')


used_pids = set()
dup = False
for i in range(10):
    curf = f'chall-bin_{i}-pids.csv'
    x = G.playlist_csv_generator(curf, csv_path = res_path)
    for _i,pl in enumerate(x):
        cur_pid = int(pl['pid'])
        if cur_pid not in used_pids:
            used_pids.add(cur_pid)
        else:
            dup = True
            print('dup')
            break

if dup == False:
    print('no dup')

"""
x = G.playlist_csv_generator('train_pids.csv', csv_path = res_path)
for _i, pl in enumerate(x):
    cur_pid = int(pl['pid'])
    if 
"""
