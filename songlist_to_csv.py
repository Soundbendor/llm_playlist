import os, csv
import pandas as pd
import routes as G
import getter as UG
import numpy as np

out_dir = os.path.join( os.sep.join(__file__.split(os.sep)[:-1]), 'res') 
out_file = os.path.join(out_dir, 'df_train.csv')
out_np = os.path.join(out_dir, 'artists_train.npy')
song_dir = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'res', 'baseline_random_filt_500_real')

df = pd.read_csv(G.joined_csv2_path, index_col=[0])
df2 = UG.df_filter_by_uri_file(df, 'guess_0.txt', uri_dir = song_dir)

dfa = UG.df_get_artists(df2)
#df2.reset_index().to_csv(out_file)
#np.save(out_np, allow_pickle = True)
print(df2)
print(dfa)
