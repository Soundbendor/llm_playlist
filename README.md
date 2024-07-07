# llm_playlist

## notes
there are 101 tracks that are in popularity.csv that are not recorded with features in the feature database. these can be found in diffuris.txt

## res
- bline-chall... is over the (emulated) challenges (bline is baseline)
- retrain2 refers to results over validation 9k (setting aside 1000 for challenges 2-10 and training on rest)
- retrain refers to results over validation 10k (setting aside 1000 for challenges 1-10 and training on rest)
- full4 refers to results doing similarity search in bm25 over all playlists (while train refers to doing similarity search over just training playlists)
- joined refers to doing random (for challenges where you condition on first or random tracks) over songs in the joined.csv (feature database songs) while all refers to doing random over all songs
