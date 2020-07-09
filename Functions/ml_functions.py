# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:13:26 2020

@author: GSCA
"""


import numpy as np


def decoupe_dataframe(df,look_back):
    dataX,dataY = [],[]
    for i in range(len(df) - look_back - 1):
        a = df[i:(i + look_back)]
        dataY=dataY+[df[i+look_back]]
        dataX.append(a)
    return (np.asarray(dataX),np.asarray(dataY).flatten())


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

# From this post : http://stackoverflow.com/a/14314054/3293881 by @Jaime
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def mad_numpy(a, W):
    a2D = strided_app(a, W, 1)
    return np.absolute(a2D - moving_average(a, W)[:, None]).mean(1)
