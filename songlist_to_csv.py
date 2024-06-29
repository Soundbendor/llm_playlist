import os, csv
import pandas as pd
import routes as G
import getter as UG
song_file = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'res', 'baseline_random_filt_500_real', 'guess_0.txt')


trk = []
with open(song_file, 'r') as f:
    trk = set([x.strip() for x in f.readlines()])

#print(trk)
print(len(trk))    
df = pd.read_csv(G.joined_csv2_path, index_col=[0])

df2 = UG.get_uris_from_df(df, trk)
print(df2)
