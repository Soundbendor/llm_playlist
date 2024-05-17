# from https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#dataset

import numpy as np

def r_precision(g_arr,r_arr):
    """
    g_arr = ground truth numpy 1d array(s)
    r_arr = retrieved numpy 1d array(s)

    Note: counts repeated entries only once
    """
    numer = np.intersect1d(g_arr, r_arr)
    num_numer = numer.shape[0]
    num_g = g_arr.shape[0]
    return num_numer/num_g

def dgc(g_arr, r_arr):
    """
    g_arr = ground truth numpy 1d array(s)
    r_arr = retrieved numpy 1d array(s)
    
    get "true" relevance labels as n-i from r_arr

    """
    num_g = g_arr.shape[0]
    labels = np.arange(num_g,0,-1)

    # maybe this doesn't work with repeats
    label_dict = {entry: label for (entry,label) in zip(num_g, labels)}

    entries = label_dict.keys() # get all songs that have relevance labels
    
    # entries without relevance labels get 0
    labeled_r = np.array([label_dict[x] if x in entries else 0 for x in r_arr])

    num_r = r.arr.shape[0]
    denom = np.hstack(([1.], np.log2(np.arange(2, num_r))))
    terms = np.divide(labeled_r,denom)
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
    ret = max_clicks + 1 # the default if no r in g
    if where_in.shape[0] > 0:
        # original formulation has -1, but we are already 0-indexed
        first_idx = where_in[0]
        ret = int(first_idx/10.)

    return ret



    




