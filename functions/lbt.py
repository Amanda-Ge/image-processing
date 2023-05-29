from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.lbt import pot_ii
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


def lbt(X,Pf,N):
    t = np.s_[N//2:-N//2]
    Xp = X.copy()
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    C = dct_ii(N)
    Y = colxfm_2d(Xp,C)
    return Y
    
def ilbt(Y,Pr,N):
    t = np.s_[N//2:-N//2]
    C = dct_ii(N)
    Z = recover(Y,C)
    Zp = Z.copy()
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)
    return Zp

def LBT(X,N,s,step_size,rise_ratio):
    Pf, Pr = pot_ii(N,s)
    rise = rise_ratio*step_size
    Y = lbt(X,Pf,N)
    Yq = quantise(Y,step_size,rise)
    Zp = ilbt(Yq,Pr,N)
    Yr = regroup(Yq, N)
    lbt_encodebits = dctbpp(Yr,N)
    return Zp,lbt_encodebits

def plot_Zp(X,Zp):
    fig, axs = plt.subplots(1, 2,figsize=(13,8))
    plot_image(X, ax=axs[0])
    plot_image(Zp, ax=axs[1])

    axs[0].set(title='original image')
    axs[1].set(title='LBT quantised image')
    
def plot_lbt(X,N,s,step_size,start,end,inc):
    H_X = bpp(X)
    X_encodebits = H_X * X.size
    rise = []
    rms = []
    bit = []
    for i in range(start,end,inc):
        rise_ratio = i/10
        Z,lbt_bits = LBT(X,N,s,step_size,rise_ratio)
        rise.append(rise_ratio)
        bit.append(lbt_bits)
        rms.append(np.std(X-Z))
        
    gain = X_encodebits/bit
    fig, ax = plt.subplots(1,3,figsize=(10,3))
    ax[0].plot(rise, gain)
    plt.suptitle("LBT")
    ax[0].set_xlabel("rise ratio")
    ax[0].set_ylabel("gain")
    ax[1].plot(rise, rms)
    ax[1].set_xlabel("rise ratio")
    ax[1].set_ylabel("rms")
    ax[2].plot(gain, rms)
    ax[2].set_xlabel("gain")
    ax[2].set_ylabel("rms")