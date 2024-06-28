import os, json, csv
import getter as UG

model_path = os.path.join(__file__.split(os.sep)[0], 'models')
data_dir = os.path.join(__file__.split(os.sep)[0], 'data')

x = UG.playlist_csv_generator('bm25.playlist', csv_path = model_path)
y = UG.playlist_csv_generator('filtered_validation_set.csv', csv_path = data_dir)

train_pids = set()
for _x in x:
    cur_pid = int(_x['pid'])
    train_pids.add(cur_pid)

val_pids = set()
for _y in y:
    cur_pid = int(_y['pid'])
    val_pids.add(cur_pid)

z = train_pids.intersection(val_pids)

print('intersection of bm25 playlists and filtered validation playlists')
print(z)


