from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.familiarisation import plot_image
import matplotlib.pyplot as plt

import scipy
import numpy as np

def colxfm_2d(X,C):
    return colxfm(colxfm(X, C).T, C).T

def recover(Y,C):
    return colxfm(colxfm(Y.T, C.T).T, C.T)

def dctbpp(Yr, N):
    l = 256//N
    Yr_bits = 0
    for i in range(N):
        for j in range(N):
            Ys = Yr[i*l:(i+1)*l,j*l:(j+1)*l]
            H_sub = bpp(Ys)
            Ys_bits = H_sub * Ys.size
            Yr_bits += Ys_bits
    return Yr_bits


def dct(X,N,step_size,rise_ratio):
    C = dct_ii(N)
    Y = colxfm_2d(X,C)
    rise = rise_ratio * step_size
    Y_q = quantise(Y,step_size, rise)
    Y_r = regroup(Y_q, N)/N
    dct_encodebits = dctbpp(Y_r,N)
    Z = recover(Y_q,C)
    return Z,dct_encodebits

def plot_Z(X,Z):
    fig, axs = plt.subplots(1, 2,figsize=(13,8))
    plot_image(X, ax=axs[0])
    plot_image(Z, ax=axs[1])

    axs[0].set(title='original image')
    axs[1].set(title='DCT quantised image')

def plot_dct(X,N,step_size,start,end,inc):
    H_X = bpp(X)
    X_encodebits = H_X * X.size
    rise = []
    rms = []
    bit = []
    for i in range(start,end,inc):
        rise_ratio = i/10
        Z,dct_bits = dct(X,N,step_size,rise_ratio)
        rise.append(rise_ratio)
        bit.append(dct_bits)
        rms.append(np.std(X-Z))
    gain = X_encodebits/bit    
    fig, ax = plt.subplots(1,3,figsize=(10,3))
    ax[0].plot(rise, gain)
    plt.suptitle("DCT")
    ax[0].set_xlabel("rise ratio")
    ax[0].set_ylabel("gain")
    ax[1].plot(rise, rms)
    ax[1].set_xlabel("rise ratio")
    ax[1].set_ylabel("rms")
    ax[2].plot(gain, rms)
    ax[2].set_xlabel("gain")
    ax[2].set_ylabel("rms")
    
    
def suppress_dct(X,N,step_size,rise_ratio):
    C = dct_ii(N)
    l = 256// N
    Y = colxfm_2d(X,C)
    Y_r = regroup(Y,N)/N
    Y_rq = np.zeros_like(X)
    rise = rise_ratio * step_size
    for i in range(N):
        for j in range(N):
            subimage = Y_r[i*l:(i+1)*l,j*l:(j+1)*l]
            Y_rq[i*l:(i+1)*l,j*l:(j+1)*l] = quantise(subimage,step_size,rise[i][j])

    Y_q = regroup(Y_rq, 256//N)*N
    dct_encodebits = dctbpp(Y_q,N)
    Z = recover(Y_q,C)
    return Z,dct_encodebits