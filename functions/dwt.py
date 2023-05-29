
import warnings
import inspect
import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
import numpy as np
from typing import Tuple
from cued_sf2_lab.dwt import dwt
from cued_sf2_lab.dwt import idwt
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.laplacian_pyramid import bpp

def nlevdwt(X, n):
    m=256
    Y=dwt(X)
    for i in range(n-1):
        m = m//2
        Y[:m,:m] = dwt(Y[:m,:m])
    return Y

def nlevidwt(Y, n):
    m=256//(2**(n-1))
    for i in range(n-1):
        Y[:m,:m] = idwt(Y[:m,:m])
        m*=2
    Z = idwt(Y)
    return Z
    
    
def quantdwt(Y: np.ndarray, dwtstep: np.ndarray,rise_ratio) -> Tuple[np.ndarray, np.ndarray]:
    m = 256
    n = len(dwtstep[0]) - 1
    Yq = np.zeros_like(Y)
    dwtent = np.zeros((3,n+1))
    rise = dwtstep * rise_ratio
    for i in range(n):
        m //= 2
        Yq[m:2*m,:m] = quantise(Y[m:2*m,:m],dwtstep[1][i],rise[1][i])
        Yq[:m,m:2*m] = quantise(Y[:m,m:2*m],dwtstep[0][i],rise[0][i])
        Yq[m:2*m,m:2*m] = quantise(Y[m:2*m,m:2*m],dwtstep[2][i],rise[2][i])
        
        dwtent[1][i] = bpp(Yq[m:2*m,:m])
        dwtent[0][i] = bpp(Yq[:m,m:2*m])
        dwtent[2][i] = bpp(Yq[m:2*m,m:2*m])
    Yq[:m,:m] = quantise(Y[:m,:m],dwtstep[0][n],rise[0][n])
    dwtent[0][n] = bpp(Yq[:m,:m])
    return Yq, dwtent
    
def encode_bits(dwtent,n):
    m =256
    res = 0
    for i in range(n):
        m//=2
        res+=dwtent[0][i]*m*m
        res+=dwtent[1][i]*m*m
        res+=dwtent[2][i]*m*m
    res+=dwtent[0][n]*m*m
    return res
    
def stepsize_ratio(Y,n):
    m = 256
    stepsize = np.zeros((3,n+1))
    energy = np.zeros((3,n+1))
    for i in range(n):
        m//=2
        
        Y_Im1 = np.zeros_like(Y)
        Y_Im1[m+m//2][m//2] = 100
        
        Y_Im2 = np.zeros_like(Y)
        Y_Im2[m//2][m+m//2] = 100
        
        Y_Im3 = np.zeros_like(Y)
        Y_Im3[m+m//2][m+m//2] = 100
        
        Z_1 = nlevidwt(Y_Im1,i+1)
        Z_2 = nlevidwt(Y_Im2,i+1)
        Z_3 = nlevidwt(Y_Im3,i+1)
        
        energy[0][i] = np.sum(Z_1**2.0)
        energy[1][i] = np.sum(Z_2**2.0)
        energy[2][i] = np.sum(Z_3**2.0)
        
    Y_Im = np.zeros_like(Y)
    Y_Im[m//2][m//2] = 100
    Z_0 = nlevidwt(Y_Im,n)
    energy[0][n] = np.sum(Z_0**2.0)
    stepsize = np.sqrt(1/energy)*np.sqrt(energy[0][n])
    return stepsize
    
def DWT(X,n,step_size,rise_ratio):
    Y = nlevdwt(X,n)
    step_ratio = stepsize_ratio(Y,n)
    step_ratio[1:,n]=0
    dwtstep = step_ratio*step_size
    Yq, dwtent = quantdwt(Y,dwtstep,rise_ratio)
    Z = nlevidwt(Yq,n)
    dwt_bits = encode_bits(dwtent,n)
    return Z,dwt_bits
    
def plot_Z_dwt(X,Z):
    fig, axs = plt.subplots(1, 2,figsize=(13,8))
    plot_image(X, ax=axs[0])
    plot_image(Z, ax=axs[1])

    axs[0].set(title='original image')
    axs[1].set(title='DWT quantised image')
    
def plot_dwt(X,n,step_size,start,end,inc):
    H_X = bpp(X)
    X_encodebits = H_X * X.size
    rise = []
    rms = []
    bit = []
    for i in range(start,end,inc):
        rise_ratio = i/10
        Z,dwt_bits = DWT(X,n,step_size,rise_ratio)
        rise.append(rise_ratio)
        bit.append(dwt_bits)
        rms.append(np.std(X-Z))
    
    gain = X_encodebits/bit
    fig, ax = plt.subplots(1,3,figsize=(10,3))
    ax[0].plot(rise, gain)
    plt.suptitle("DWT")
    ax[0].set_xlabel("rise ratio")
    ax[0].set_ylabel("gain")
    ax[1].plot(rise, rms)
    ax[1].set_xlabel("rise ratio")
    ax[1].set_ylabel("rms")
    ax[2].plot(gain, rms)
    ax[2].set_xlabel("gain")
    ax[2].set_ylabel("rms")
    