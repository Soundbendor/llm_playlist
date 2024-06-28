import os, json, csv
import getter as UG

model_path = os.path.join(__file__.split(os.sep)[0], 'models')
data_dir = os.path.join(__file__.split(os.sep)[0], 'valid_retrain')
res_dir = os.path.join(__file__.split(os.sep)[0], 'valid_retrain')
#res_dir = '/media/dxk/TOSHIBA EXT/llm_playlist_res'

y = UG.playlist_csv_generator('train_pids.csv', csv_path = data_dir)

train_uris = set()

prev_pid = -1
prev_pl = None

for _i,_x in enumerate(y):
    print(f'processing {_i}')
    _y = None
    cur_pid = int(_x['pid'])
    if prev_pid != cur_pid:
        _y = UG.get_playlist(_x['file'], int(_x['idx']))
        prev_pl = _y
    else:
        _y = prev_pl
    prev_pid = cur_pid

    for _t in _y['tracks']:
        _u  = _t['track_uri'].strip()
        train_uris.add(_u)

with open(os.path.join(res_dir, 'train.uris'), 'w') as f:
    for uri in train_uris:
        f.write(uri)
        f.write('\n')
print('train song count:', len(train_uris))


