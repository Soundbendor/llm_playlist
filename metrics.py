# from https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#dataset
# numpy intersection doesn't count duplicates but i count each duplicate in dcg
# note that G = [1,2,3,4,5], R = [1,6,7] has a NDCG of 1
# G = [1,2,3,4,5], R = [1,2,3,4,5,6,7] has a NDCG of 1
# G = [1,2,3,4,5], R = [1] also has a NDCG of 1
# G = [1,2,3,4,5, R = [0,1] has a NDCG < 1

# note that all methods expect numpy arrays (which work with strings), see example below
import numpy as np

def r_precision(g_arr,r_arr):
    """
    g_arr = ground truth numpy 1d array(s)
    r_arr = retrieved numpy 1d array(s)

    Note: counts repeated entries only once
    """
    num_g = g_arr.shape[0]

    numer = np.intersect1d(g_arr, r_arr[:num_g])
    num_numer = numer.shape[0]

    return num_numer/num_g

def dcg(g_arr, r_arr):
    """
    g_arr = ground truth numpy 1d array(s)
    r_arr = retrieved numpy 1d array(s)

    get "true" relevance labels as n-i from r_arr

    """
    num_g = g_arr.shape[0]
    #labels = np.arange(num_g-1,-1,-1)
    #labels = np.arange(num_g, 0, -1)


    #print(num_g, labels)
    # maybe this doesn't work with repeats
    #label_dict = {entry: label for (entry,label) in zip(g_arr, labels)}
    #label_dict = {entry: 1 for (entry,label) in zip(g_arr, labels)}

    #entries = label_dict.keys() # get all songs that have relevance labels

    # entries without relevance labels get 0
    labeled_r = np.array([1 if x in g_arr else 0 for x in r_arr])

    num_r = r_arr.shape[0]
    denom = np.hstack(([1.], np.log2(np.arange(2, num_r+1))))
    terms = np.divide(labeled_r,denom)
    #print(terms)
    cur_dcg = np.sum(terms)

    return cur_dcg


def idcg(g_arr, r_arr):
    """
    g_arr = ground truth numpy 1d array(s)
    r_arr = retrieved numpy 1d array(s)

    """
    cur_isect = np.intersect1d(g_arr, r_arr)
    num_isect = cur_isect.shape[0]

    denom_terms  = np.log2(np.arange(2, num_isect+1))
    sum_terms = np.sum(1./denom_terms) # without leading 1

    return 1. + sum_terms

def ndcg(g_arr, r_arr):
    cur_dcg = dcg(g_arr, r_arr)
    cur_idcg = idcg(g_arr, r_arr)
    return cur_dcg/cur_idcg


def rec_songs_clicks(g_arr, r_arr, max_clicks = 50):
    """
    g_arr = ground truth numpy 1d array(s)
    r_arr = retrieved numpy 1d array(s)

    """


    # check membership r_arr in g_arr
    r_in_g = np.isin(r_arr, g_arr)
    # get the true indices
    where_in = np.nonzero(r_in_g)[0]
    if where_in.shape[0] > 0:
        # original formulation has -1, but we are already 0-indexed
        first_idx = where_in[0]
    else:
        first_idx = max_clicks # already 0-indexed
    ret = int(first_idx/10.)

    return ret

def reciprocal_rank(g_arr, r_arr):
    ret = 0
    r_in_g = np.isin(r_arr, g_arr)
    where_in = np.nonzero(r_in_g)[0]
    if where_in.shape[0] > 0:
        ret = 1./(where_in[0]+1.)

    return ret

# wait, isn't this just r-precision
# but not cutting off r
# (or r-precision is just recall-ish)
def recall(g_arr, r_arr):
    g_in_r = np.isin(g_arr, r_arr)
    return np.mean(g_in_r)

def calc_metrics(g_arr, r_arr, max_clicks=50):
    _rprec = r_precision(g_arr, r_arr)
    _ndcg = ndcg(g_arr, r_arr)
    _clicks = rec_songs_clicks(g_arr, r_arr, max_clicks=max_clicks)
    _rr = reciprocal_rank(g_arr, r_arr)
    _recall = recall(g_arr, r_arr)
    ret = {'r_precision': _rprec, 'ndcg': _ndcg,
           'clicks': _clicks, 'rr': _rr,
           'recall': _recall}
    return ret

# expects array of results dicts (see above)
def get_mean_metrics(res_arr):
    h = ['r_precision', 'ndcg', 'clicks', 'rr', 'recall']
    h_arr = [[x[y] for y in h] for x in res_arr]
    h_means = np.mean(h_arr, axis=0)
    ret = {x:y for (x,y) in zip(h,h_means)}
    return ret

if __name__ == "__main__":
    p1 = np.array(['spotify:track:4vv1KjUzPwrtDbozizSfQc', 'spotify:track:0Ws8D3EWUDgY962Xftb0h5', 'spotify:track:0lMbuWUpfTWhEmOKxppEau', 'spotify:track:4KaIJ1FWXUoAAnOts1YWjD', 'spotify:track:60APt5N2NRaKWf5xzJdzyC', 'spotify:track:4c1BAfuPGZSun6aAvmmoHs', 'spotify:track:25oOaleife6E2MIKmFkPvg', 'spotify:track:1UAmQe8EwpxQ80OfYVD13z', 'spotify:track:2e3OgIbfZw5deCjLMGatSS', 'spotify:track:5de7ci7TFqbQ1PFgKAD7MR', 'spotify:track:29BXCsh4lGLrndprkgYL6O', 'spotify:track:1jQsKN68yE94tMYml0wHMd', 'spotify:track:59J5nzL1KniFHnU120dQzt', 'spotify:track:1e1JKLEDKP7hEQzJfNAgPl', 'spotify:track:5CtI0qwDJkDQGwXD1H1cLb', 'spotify:track:152lZdxL1OR0ZMW6KquMif', 'spotify:track:3DXncPQOG4VBw3QHh3S817', 'spotify:track:2of5xn0GU0TdFneR1saRLH', 'spotify:track:2INqEk4ko5AsGVLBsiKiQe', 'spotify:track:3LRddJIw2ymm1CHIO9xlkC', 'spotify:track:4uTTsXhygWzSjUxXLHZ4HW', 'spotify:track:1jNyxG5S2P9gztbfAnrq85', 'spotify:track:4HW5kSQ8M2IQWZhSxERvla', 'spotify:track:3nVDOYBJpdCkRR6r1DbZum', 'spotify:track:1rsAFUCa6BVMgRQ3FCQQsi', 'spotify:track:6kig1UFggPUyZBCvXD3Wod'])
    p2 = p1.copy()
    p2 = np.concatenate((p1,p1))
    p1len = p1.shape[0]
    #p1idx = np.arange(0,p1len)
    p2[0:5] = 'spotify:track:4vv1KjUzPwrtDbozizSfQd'
    res = calc_metrics(p1, p2, max_clicks=p1len)
    for x,r in res.items():
        print(x,':', r)
    get_mean_metrics([res, res, res])
    #_dcg = ndcg(p1,p1[::-2])
    #_rr = reciprocal_rank(p1, p2)
    #_pr = recall(p1, p2)
    #print(p1)
    #print(_rr)
    #print(_pr)
    #print(p1[::-2])
    #print(_dcg)







