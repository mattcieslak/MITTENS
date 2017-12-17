#!/usr/bin/env python
import numpy as np

def aitchison_distance(null_probs,obs_probs,
                       numerator_indices=None, denominator_indices=None):
    """

    """
    if None in (numerator_indices, denominator_indices):
        out = np.zeros(obs_probs.shape[0], dtype=np.float)
        idx_pairs = []
        D = len(null_probs)
        for i in range(D):
            for j in range(i):
                idx_pairs.append(np.array([j,i]))
        numerator, denominator = np.row_stack(idx_pairs).T
    else:
        numerator, denominator = numerator_indices, denominator_indices
    
    with np.errstate(divide='ignore', invalid='ignore'):
        aitchison_minus = np.log(null_probs[numerator] / null_probs[denominator])
        for n, obs in enumerate(obs_probs):
            out[n] = np.sqrt(1./D * np.nansum( 
                (np.log(obs[numerator]/obs[denominator]) - aitchison_minus   )**2 ))
    
    return np.nan_to_num(out)

def kl_distance(null_probs, obs_probs):
    out = np.zeros(obs_probs.shape[0], dtype=np.float)
    with np.errstate(divide='ignore', invalid='ignore'):
        for n,obs in enumerate(obs_probs):
            out[n]= np.nansum(null_probs * np.log(null_probs / obs) )
    return np.nan_to_num(out)

def aitchison_asymmetry(half1, half2):
    out = np.zeros(half1.shape[0], dtype=np.float)
    idx_pairs = []
    D = half1.shape[1]
    for i in range(D):
        for j in range(i):
            idx_pairs.append(np.array([j,i]))
    numerator, denominator = np.row_stack(idx_pairs).T
    
    with np.errstate(divide='ignore', invalid='ignore'):
        for n, (x,y) in enumerate(zip(half1, half2)):
            out[n] = np.sqrt(1./D * np.nansum( 
                (np.log(x[numerator]/x[denominator]) - \
                 np.log(y[numerator]/y[denominator]))**2 ))
    
    return np.nan_to_num(out)

def kl_asymmetry(half1, half2):
    out1 = np.zeros(half1.shape[0], dtype=np.float)
    out2 = np.zeros(half1.shape[0], dtype=np.float)
    for n,(x,y) in enumerate(zip(half1,half2)):
        out1[n]= np.sum( x * np.log(x / y) )
        out2[n]= np.sum( y * np.log(y / x) )
    return out1, out2

