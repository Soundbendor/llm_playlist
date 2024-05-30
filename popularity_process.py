# removing spotify:track: from popularity.csv and changing field to id

import routes as G
import pandas as pd
import csv,os
pd.options.mode.chained_assignment = None 

pop_path=os.path.join(G.num_tracks_path, 'stats', 'popularity.csv')
pop_path2=os.path.join(G.num_tracks_path, 'stats', 'popularity_trimmed.csv')

with open(pop_path, 'r') as f:
    csvr = csv.DictReader(f)
    with open(pop_path2, 'w') as f2:
        csvw = csv.DictWriter(f2, fieldnames=['id','count'])
        csvw.writeheader()
        for row in csvr:
            cur_id = row['uri']
            cur_count = row['count']
            new_id = cur_id.split(':')[-1]
            csvw.writerow({'id': new_id, 'count': cur_count})





