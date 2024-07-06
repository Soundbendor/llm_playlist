import numpy as np
import os
import pandas as pd
import getter as UG
import routes as G
uri_dir = os.path.join( os.sep.join(__file__.split(os.sep)[:-1]), 'valid_retrain2') 
trk = []
urifile = 'train.uris'
with open(os.path.join(uri_dir, urifile), 'r') as f:
    trk = set([x.strip() for x in f.readlines()])

all_uris = np.load('uris.npy')
z = set([x for x in all_uris])
print(trk.difference(z))
print(len(trk.intersection(z)))
print(len(trk))

df = pd.read_csv(G.joined_csv2_path, index_col=[0])
