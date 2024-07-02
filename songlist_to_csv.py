import os, csv
import pandas as pd
import routes as G
import getter as UG
import numpy as np

out_dir = os.path.join( os.sep.join(__file__.split(os.sep)[:-1]), 'valid_retrain2') 
out_file = os.path.join(out_dir, 'df_train2.csv')
out_np = os.path.join(out_dir, 'artists_train.npy')
#song_dir = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), 'res', 'baseline_random_filt_500_real')

df = pd.read_csv(G.joined_csv2_path, index_col=[0])
df2 = UG.df_filter_by_uri_file(df, 'train.uris', uri_dir = out_dir)

dfa = UG.df_get_artists(df2)
df2.reset_index().to_csv(out_file)
np.save(out_np, dfa, allow_pickle = True)
print(df2)
print(dfa)
