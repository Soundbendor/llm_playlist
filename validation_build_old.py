import os,csv

res_path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data', 'stats')
pool_file = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'data', 'filtered_validation_set.csv')

out_path = "/media/dxk/TOSHIBA EXT/llm_playlist_res/valid"

if os.path.exists(out_path) == False:
    os.mkdir(out_path)



used_pid = set()
bin_size = 50
max_sz = 400
pbins = {k: {} for k in range(0,max_sz + bin_size, bin_size)}
pused = {k: {} for k in range(0, max_sz + bin_size, bin_size)}

#lengths that don't exist
min_num = float('inf')
max_num = 250
no_lens = set(list(range(max_sz+1)))
bin_getter = lambda x: int(x/bin_size) * bin_size
with open(pool_file, 'r') as f:
    csvr = csv.DictReader(f)
    for row in csvr:
        cur_tracks = int(row['num_tracks'])
        cur_pid = int(row['pid'])
        cur_file = row['file']
        cur_idx = int(row['idx'])
        cur_dict = {'pid': cur_pid, 'file': cur_file, 'num_tracks': cur_tracks, 'idx': cur_idx}
        cur_bin = bin_getter(cur_tracks)
        if cur_tracks not in pbins[cur_bin].keys():
            pbins[cur_bin][cur_tracks] = []
            pused[cur_bin][cur_tracks] = set()
        pbins[cur_bin][cur_tracks].append(cur_dict)
        min_num = min(cur_tracks, min_num)
        if cur_tracks in no_lens:
            no_lens.remove(cur_tracks)


#print(no_lens)
#print(pbins)
num_chall = 10
max_tol = 21

header = ['pid', 'idx', 'file', 'num_tracks', 'tol']

def playlist_getter(want_len, start_tol = 0):
    cur_tol = start_tol
    ret_pl = None
    found = False
    while cur_tol <= max_tol and found == False:
        #print(f'loop for {want_len}: {cur_tol}')
        lens = []
        if cur_tol > 0:
            lo_len = want_len - cur_tol
            hi_len = want_len + cur_tol
            if lo_len not in no_lens and lo_len >= min_num:
                lens.append(lo_len)
            if hi_len not in no_lens and hi_len <= max_num:
                lens.append(hi_len)
        elif want_len not in no_lens:
            lens.append(want_len)
        for _len in lens:
            _bin = bin_getter(_len)
            _pbin = pbins[_bin]
            #print(_bin, _len, _pbin.keys())
            _cbin = _pbin[_len]
            cur_idx = len(pused[_bin][_len])
            #print(len(_cbin), cur_idx)i
            if cur_idx < len(_cbin):
                cur_pl = _cbin[cur_idx]
                cur_pl['tol'] = cur_tol
                cur_pid = cur_pl['pid']
                if cur_pid not in pused[_bin][_len]:
                    pused[_bin][_len].add(cur_pid)
                    used_pid.add(cur_pid)
                    ret_pl = cur_pl
                    found = True
                    #print('found')
                    break
                    #print('found')
            else:
                #print('adding')
                no_lens.add(_len)
        cur_tol += 1
    return ret_pl







any_failed = False
for chall_idx in range(num_chall):
    cur_csv = f"chall-bin_{chall_idx}.csv"
    csv_path = os.path.join(res_path, cur_csv)
    all_pl = []
    failed = False
    #if chall_idx > 0:
    #    break
    with open(csv_path, 'r') as f:
        csvr = csv.DictReader(f)
        for row in csvr:
            want_len = int(row['length'])
            want_count = int(row['count'])
            cur_count = 0
            while cur_count < want_count:
                get_pl = playlist_getter(want_len)
                if get_pl != None:
                    all_pl.append(get_pl)
                    cur_count += 1
                else:
                    print(f"FAILED at {cur_csv}, len: {want_len}, count: {cur_count}")
                    failed =True
                    any_faled = True
                    break
            if failed == True:
                break
    if failed == False:
        print(f"WRITING {cur_csv}")
        cur_out = f"chall-bin_{chall_idx}-pids.csv"
        with open(os.path.join(out_path, cur_out), 'w') as f:
            csvw = csv.DictWriter(f, fieldnames=header)
            csvw.writeheader()
            for pl in all_pl:
                csvw.writerow(pl)

if any_failed == False:
    with open(os.path.join(out_path, 'valid_pids.txt'), 'w') as f:
        for pid in used_pid:
            f.write(f'{pid}\n')


                 
