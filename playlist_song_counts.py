import os, json, csv
import getter as UG

model_path = os.path.join(__file__.split(os.sep)[0], 'models')
data_dir = os.path.join(__file__.split(os.sep)[0], 'sorted_csv')
res_dir = os.path.join(__file__.split(os.sep)[0], 'sorted_csv')

x = UG.playlist_csv_generator('filtered_validation_set_sort.csv', csv_path = data_dir)
y = UG.playlist_csv_generator('train_set_sort.csv', csv_path = data_dir)

train_uris = set()
val_uris = set()

prev_pid = -1
prev_pl = None

for _i,_x in enumerate(y):
    _y = None
    cur_pid = int(_x['pid'])
    print(f'processing train {_i}')
    if prev_pid != cur_pid:
        _y = UG.get_playlist(_x['file'], int(_x['idx']))
        prev_pl = _y
    else:
        _y = prev_pl
    prev_pid = cur_pid
    _u = [_t['track_uri'].strip() for _t in _y['tracks']]
    train_uris.update(set(_u))
with open(os.path.join(res_dir, 'train.uris'), 'w') as f:
    for uri in train_uris:
        f.write(uri)
        f.write('\n')
print('train song count:', len(train_uris))


prev_pid = -1
prev_pl = None

for _i,_x in enumerate(x):
    _y = None
    cur_pid = int(_x['pid'])
    print(f'processing valid {_i}')
    if prev_pid != cur_pid:
        _y = UG.get_playlist(_x['file'], int(_x['idx']))
        prev_pl = _y
    else:
        _y = prev_pl
    prev_pid = cur_pid

    _u = [_t['track_uri'].strip() for _t in _y['tracks']]
    val_uris.update(set(_u))
with open(os.path.join(res_dir, 'filt_val.uris'), 'w') as f:
    for uri in val_uris:
        f.write(uri)
        f.write('\n')

print('filtered_validation song count:', len(val_uris))

