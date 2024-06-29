import os,csv
import getter as G

res_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'valid_retrain2')


used_pids = set()
dup = False
for i in range(10):
    if i <= 0:
        continue
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
    print('no dup in valid')

dup2 = False
dup3 = False
used_pids2 = set()
x = G.playlist_csv_generator('train_pids.csv', csv_path = res_path)
for _i, pl in enumerate(x):
    cur_pid = int(pl['pid'])
    if cur_pid in used_pids:
        dup2 = True
        break
    if cur_pid not in used_pids2:
        used_pids2.add(cur_pid)
    else:
        dup3 = True
        break

if dup2 == False:
    print('no train/valid overlap')

if dup3 == False:
    print('no dup in train')

