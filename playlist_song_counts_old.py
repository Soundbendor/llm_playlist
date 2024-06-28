import os, json, csv
import getter as UG

model_path = os.path.join(__file__.split(os.sep)[0], 'models')
data_dir = os.path.join(__file__.split(os.sep)[0], 'data')
res_dir = '/media/dxk/TOSHIBA EXT/llm_playlist_res'

x = UG.playlist_csv_generator('filtered_validation_set.csv', csv_path = data_dir)
y = UG.playlist_csv_generator('train_set.csv', csv_path = data_dir)

train_uris = set()
val_uris = set()
for _x in x:
    _y = UG.get_playlist(_x['file'], int(_x['idx']))
    for _t in _y['tracks']:
        _u  = _t['track_uri'].strip()
        val_uris.add(_u)

for _x in y:
    _y = UG.get_playlist(_x['file'], int(_x['idx']))
    for _t in _y['tracks']:
        _u  = _t['track_uri'].strip()
        train_uris.add(_u)

with open(os.path.join(res_dir, 'filt_val.uris'), 'w') as f:
    for uri in val_uris:
        f.write(uri)
        f.write('\n')

with open(os.path.join(res_dir, 'train.uris'), 'w') as f:
    for uri in train_uris:
        f.write(uri)
        f.write('\n')
print('filtered_validation song count:', len(val_uris))
print('train song count:', len(train_uris))


