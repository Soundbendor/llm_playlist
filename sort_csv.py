import os, json, csv
import getter as UG

model_path = os.path.join(__file__.split(os.sep)[0], 'models')
data_dir = os.path.join(__file__.split(os.sep)[0], 'data')
res_dir = os.path.join(__file__.split(os.sep)[0], 'sorted_csv')

header = ["name","num_tracks","idx","file","pid","modified_at","collaborative","num_albums","num_followers"]

x = UG.playlist_csv_generator('filtered_validation_set.csv', csv_path = data_dir)
y = UG.playlist_csv_generator('train_set.csv', csv_path = data_dir)


def sort_csver(plgen, outfile):
    pl = {}
    for _x in plgen:
        cur_pid = int(_x['pid'])
        cur_file = _x['file']
        if cur_file not in pl.keys():
            pl[cur_file] = []
        pl[cur_file].append(_x)


    with open(os.path.join(res_dir, outfile), 'w') as f:
        csvw = csv.DictWriter(f, fieldnames=header)
        csvw.writeheader()
        for plfile, plarr in pl.items():
            for _pl in plarr:
                csvw.writerow(_pl)

sort_csver(y, 'train_set_sort.csv')
sort_csver(x, 'filtered_validation_set_sort.csv')

