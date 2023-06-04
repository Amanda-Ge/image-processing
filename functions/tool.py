import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise, quant1, quant2
from cued_sf2_lab.jpeg import *
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.dct import regroup
import numpy as np
import scipy
from .dct import *
from .dwt import *
from .lbt import *

'''
Xlp, _ = load_mat_img(img='../lighthouse.mat', img_info='X')
Xbp, _ = load_mat_img(img='../bridge.mat', img_info='X')
Xfp, _ = load_mat_img(img='../flamingo.mat', img_info='X')
Xl = Xlp - 128.0
Xb = Xbp - 128.0
Xf = Xfp - 128.0

X1p, _ = load_mat_img(img='images/2018.mat', img_info='X')
X2p, _ = load_mat_img(img='images/2019.mat', img_info='X')
X3p, _ = load_mat_img(img='images/2020.mat', img_info='X')
X4p, _ = load_mat_img(img='images/2021.mat', img_info='X')
X5p, _ = load_mat_img(img='../2022.mat', img_info='X')
X1 = X1p - 128.0
X2 = X2p - 128.0
X3 = X3p - 128.0
X4 = X4p - 128.0
X5 = X5p - 128.0

X_list = [Xl, Xb, Xf, X1, X2, X3, X4, X5]
original_size = [58.8, 71.7, 60.5, 57.4, 57.7, 60.7, 43.1, 52.6]
for i in range(len(original_size)):
    original_size[i] *= 8196
print(original_size)
'''

def myjpegenc(Y: np.ndarray, qstep: float, N: int = 8, M: int = 8,
        opthuff: bool = False, dcbits: int = 8, log: bool = True
        ):
    '''
    Encodes the image in X to generate a variable length bit stream.

    Parameters:
        Y: the input transformed image
        qstep: the quantisation step to use in encoding
        N: the width of the DCT |block (defaults to 8)
        M: the width of each block to be coded (defaults to N). Must be an
            integer multiple of N - if it is larger, individual blocks are
            regrouped.
        opthuff: if true, the Huffman table is optimised based on the data in X
        dcbits: the number of bits to use to encode the DC coefficients
            of the DCT.

    Returns:
        vlc: variable length output codes, where ``vlc[:,0]`` are the codes and
            ``vlc[:,1]`` the number of corresponding valid bits, so that
            ``sum(vlc[:,1])`` gives the total number of bits in the image
        hufftab: optional outputs containing the Huffman encoding
            used in compression when `opthuff` is ``True``.
    '''

    if M % N != 0:
        raise ValueError('M must be an integer multiple of N!')

    Yq = Y.astype("int") #quant1(Y, qstep, qstep).astype('int')

    # Generate zig-zag scan of AC coefs.
    scan = diagscan(M)

    # On the first pass use default huffman tables.

        #print('Generating huffcode and ehuf using default tables')
    dhufftab = huffdflt(1)  # Default tables.
    huffcode, ehuf = huffgen(dhufftab)


    # Generate run/ampl values and code them into vlc(:,1:2).
    # Also generate a histogram of code symbols.

    sy = Yq.shape
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yq = Yq[r:r+M,c:c+M]
            # Possibly regroup
            if M > N:
                yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            # Encode DC coefficient first
            dccoef = yqflat[0] + 2 ** (dcbits-1)
            if dccoef not in range(2**dcbits):
                raise ValueError(
                    'DC coefficients too large for desired number of bits')
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            ra1 = runampl(yqflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    # Return here if the default tables are sufficient, otherwise repeat the
    # encoding process using the custom designed huffman tables.
    if not opthuff:
        if log:
            pass
            #print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        return vlc, dhufftab

    # Design custom huffman tables.
    if log:
        print('Generating huffcode and ehuf using custom tables')
    dhufftab = huffdes(huffhist)
    huffcode, ehuf = huffgen(dhufftab)

    # Generate run/ampl values and code them into vlc(:,1:2).
    # Also generate a histogram of code symbols.
    if log:
        print('Coding rows (second pass)')
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yq = Yq[r:r+M, c:c+M]
            # Possibly regroup
            if M > N:
                yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            #print(yqflat)
            # Encode DC coefficient first
            dccoef = yqflat[0] + 2 ** (dcbits-1)
            #print(dccoef)
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            ra1 = runampl(yqflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    if log:
        print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        print('Bits for huffman table = {}'.format(
            (16 + max(dhufftab.huffval.shape))*8))

    return vlc, dhufftab

def myjpegdec(vlc: np.ndarray, qstep: float, N: int = 8, M: int = 8,
        hufftab = None,
        dcbits: int = 8, W: int = 256, H: int = 256, log: bool = True
        ) -> np.ndarray:
    '''
    Decodes a (simplified) JPEG bit stream to an image

    Parameters:

        vlc: variable length output code from jpegenc
        qstep: quantisation step to use in decoding
        N: width of the DCT block (defaults to 8)
        M: width of each block to be coded (defaults to N). Must be an
            integer multiple of N - if it is larger, individual blocks are
            regrouped.
        hufftab: if supplied, these will be used in Huffman decoding
            of the data, otherwise default tables are used
        dcbits: the number of bits to use to decode the DC coefficients
            of the DCT
        W, H: the size of the image (defaults to 256 x 256)

    Returns:

        Z: the output greyscale image
    '''

    opthuff = (hufftab is not None)
    if M % N != 0:
        raise ValueError('M must be an integer multiple of N!')

    # Set up standard scan sequence
    scan = diagscan(M)

    if opthuff:
        if len(hufftab.bits.shape) != 1:
            raise ValueError('bits.shape must be (len(bits),)')

    else:
 
        hufftab = huffdflt(1)
    # Define starting addresses of each new code length in huffcode.
    # 0-based indexing instead of 1
    huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
    # Set up huffman coding arrays.
    huffcode, ehuf = huffgen(hufftab)

    # Define array of powers of 2 from 1 to 2^16.
    k = 2 ** np.arange(17)

    # For each block in the image:

    # Decode the dc coef (a fixed-length word)
    # Look for any 15/0 code words.
    # Choose alternate code words to be decoded (excluding 15/0 ones).
    # and mark these with vector t until the next 0/0 EOB code is found.
    # Decode all the t huffman codes, and the t+1 amplitude codes.

    eob = ehuf[0]
    run16 = ehuf[15 * 16]
    i = 0
    Zq = np.zeros((H, W))

    for r in range(0, H, M):
        for c in range(0, W, M):
            yq = np.zeros(M**2)

            # Decode DC coef - assume no of bits is correctly given in vlc table.
            cf = 0
            if vlc[i, 1] != dcbits:
                raise ValueError(
                    'The bits for the DC coefficient does not agree with vlc table')
            yq[cf] = vlc[i, 0] - 2 ** (dcbits-1)
            i += 1

            # Loop for each non-zero AC coef.
            while np.any(vlc[i] != eob):
                run = 0

                # Decode any runs of 16 zeros first.
                while np.all(vlc[i] == run16):
                    run += 16
                    i += 1

                # Decode run and size (in bits) of AC coef.
                start = huffstart[vlc[i, 1] - 1]
                res = hufftab.huffval[start + vlc[i, 0] - huffcode[start]]
                run += res // 16
                cf += run + 1
                si = res % 16
                i += 1

                # Decode amplitude of AC coef.
                if vlc[i, 1] != si:
                    raise ValueError(
                        'Problem with decoding .. you might be using the wrong hufftab table')
                ampl = vlc[i, 0]

                # Adjust ampl for negative coef (i.e. MSB = 0).
                thr = k[si - 1]
                yq[scan[cf-1]] = ampl - (ampl < thr) * (2 * thr - 1)

                i += 1

            # End-of-block detected, save block.
            i += 1

            yq = yq.reshape((M, M)).T

            # Possibly regroup yq
            if M > N:
                yq = regroup(yq, M//N)
            Zq[r:r+M, c:c+M] = yq

    return Zq

from scipy import signal, ndimage
def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image 
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity 
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on 
    Signals, Systems and Computers, Nov. 2003 
    
    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(im1, im2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))

ltable = np.array(
[[16, 11, 10, 16, 124, 140, 151, 161],
[12, 12, 14, 19, 126, 158, 160, 155],
[14, 13, 16, 24, 140, 157, 169, 156],
[14, 17, 22, 29, 151, 187, 180, 162],
[18, 22, 37, 56, 168, 109, 103, 177],
[24, 35, 55, 64, 181, 104, 113, 192],
[49, 64, 78, 87, 103, 121, 120, 101],
[72, 92, 95, 98, 112, 100, 103, 199]])

ctable = np.array(
[[17, 18, 24, 47, 99, 99, 99, 99],
[18, 21, 26, 66, 99, 99, 99, 99],
[24, 26, 56, 99, 99, 99, 99, 99],
[47, 66, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99]])

def super_scheme(X):
    vlcs = []
    headers = []
    rms_value = []

    for t in range(2):
        d_ = 8
        while True:
            if d_ > 12: 
                print("wrong!")
                vlc = None
                header = [[100, 100, 100, 100, 100, 100, 100], None]
                break
            try:
                vlc, header = enc_two_lbt(X, N=4, M=4, d=d_, table = 0)
            except ValueError:
                d_ += 1
            else:
                break
        vlcs.append(vlc)
        headers.append(header)
        rms_value.append(header[0][4])

    min_rms = min(rms_value)
    idx = rms_value.index(min_rms)
    vlc = vlcs[idx]
    header = headers[idx]
    print(header[0])
    return vlc, header


def enc_two_lbt(X, N, M, d, table):
    max_bits = 39000

    def fun(x):
        stepsize = x
        Pf, Pr = pot_ii(N, s=1.2)
        Y = lbt(X, Pf, N)
        Y = regroup(Y, N)
        Y[:256//N, :256//N] = lbt(Y[:256//N, :256//N], Pf, N)

        for i in range(N):
            for j in range(N):
                if table == 0:
                    Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], stepsize*k**(i+j), stepsize*k**(i+j))
                if table == 1:
                    Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ltable[i][j]*stepsize*k**(i+j), ltable[i][j]*stepsize*k**(i+j))
                if table == 2:
                    Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ctable[i][j]*stepsize*k**(i+j), ctable[i][j]*stepsize*k**(i+j))
        Y = regroup(Y, 256//N)

        vlc, dhufftab = myjpegenc(Y, stepsize, N, M, dcbits = d, opthuff=True, log=False)
        bits = sum(vlc[:, 1])
        return abs(bits-max_bits)

    data = [N, M, d, table, 99, 0, 0]
    bestvlc = None
    besthufftab = None
    s = 1.2
    for k in range(100, 111):
        k/=100
        stepsize = scipy.optimize.minimize_scalar(fun,bounds=(1,100)).x
        Pf, Pr = pot_ii(N, s)
        Y = lbt(X, Pf, N)
        Y = regroup(Y, N)
        Y[:256//N, :256//N] = lbt(Y[:256//N, :256//N], Pf, N)

        for i in range(N):
            for j in range(N):
                if table == 0:
                    Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], stepsize*k**(i+j), stepsize*k**(i+j))
                if table == 1:
                    Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ltable[i][j]*stepsize*k**(i+j), ltable[i][j]*stepsize*k**(i+j))
                if table == 2:
                    Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ctable[i][j]*stepsize*k**(i+j), ctable[i][j]*stepsize*k**(i+j))

        Y = regroup(Y, 256//N)
        vlc, dhufftab = myjpegenc(Y, stepsize, N, M, dcbits = d, opthuff=True,log=False)

        Zi = myjpegdec(vlc, stepsize, N, M, dcbits = d, hufftab=dhufftab,log=False)
        #Zi = quant2(Zi, stepsize, stepsize)
        Zi = regroup(Zi, N)
        for i in range(N):
            for j in range(N):
                if table == 0:
                    Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], stepsize*k**(i+j), stepsize*k**(i+j))
                if table == 1:
                    Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ltable[i][j]*stepsize*k**(i+j), ltable[i][j]*stepsize*k**(i+j))
                if table == 2:
                    Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ctable[i][j]*stepsize*k**(i+j), ctable[i][j]*stepsize*k**(i+j))

        Zi[:256//N, :256//N] = ilbt(Zi[:256//N, :256//N], Pr, N)
        Zi = regroup(Zi, 256//N)
        Z = ilbt(Zi, Pr, N)
        if np.std(X-Z) < data[4]:
            data[4] = np.std(X-Z)
            data[5] = k
            data[6] = stepsize
            bestvlc = vlc
            besthufftab = dhufftab

    header = [data, besthufftab]
    return bestvlc, header

def dec_two_lbt(vlc, hufftab, N, M, stepsize, k, d, table, mylog=False):
    Pf, Pr = pot_ii(N, s=1.2)
    Zi = myjpegdec(vlc, stepsize, N, M, dcbits = d, hufftab=hufftab, log=mylog)
    #Zi = quant2(Zi, stepsize, stepsize)
    Zi = regroup(Zi, N)
    for i in range(N):
        for j in range(N):
            if table == 0:
                Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], stepsize*k**(i+j), stepsize*k**(i+j))
            if table == 1:
                Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ltable[i][j]*stepsize*k**(i+j), ltable[i][j]*stepsize*k**(i+j))
            if table == 2:
                Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ctable[i][j]*stepsize*k**(i+j), ctable[i][j]*stepsize*k**(i+j))

    Zi[:256//N, :256//N] = ilbt(Zi[:256//N, :256//N], Pr, N)
    Zi = regroup(Zi, 256//N)
    Z = ilbt(Zi, Pr, N)

    return Z



def enc_three_lbt(X, N, M, d, table):
    max_bits = 39000

    def fun(stepsize1):
        Pf, Pr = pot_ii(N, s=1.2)
        Y = lbt(X, Pf, N)
        Y = regroup(Y, N)
        Y[:256//N, :256//N] = lbt(Y[:256//N, :256//N], Pf, N)
        Y[:256//N, :256//N] = regroup(Y[:256//N, :256//N], N)
        for i in range(N):
            for j in range(N):
                if table == 0:
                    Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant1(Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], stepsize1*k**(i+j), stepsize1*k**(i+j))
                if table == 1:
                    Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant1(Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], ltable[i][j]*stepsize1*k**(i+j), ltable[i][j]*stepsize1*k**(i+j))                
                if table == 2:
                    Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant1(Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], ctable[i][j]*stepsize1*k**(i+j), ctable[i][j]*stepsize1*k**(i+j))
        
        Y[:256//N//N, :256//N//N] = lbt(Y[:256//N//N, :256//N//N], Pf, N)
        for i in range(N):
            for j in range(N):
                if i==0 and j==0:
                    pass
                else:
                    if table == 0:
                        Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], stepsize*k**(i+j), stepsize*k**(i+j))
                    if table == 1:
                        Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ltable[i][j]*stepsize*k**(i+j), ltable[i][j]*stepsize*k**(i+j))
                    if table == 2:
                        Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ctable[i][j]*stepsize*k**(i+j), ctable[i][j]*stepsize*k**(i+j))
                        
        Y[:256//N, :256//N] = regroup(Y[:256//N, :256//N], 256//N//N)
        Y = regroup(Y, 256//N)
        vlc, dhufftab = myjpegenc(Y, stepsize, N, M, dcbits = d, opthuff=True,log=False)
        bits = sum(vlc[:, 1])
        return abs(bits-max_bits)
    
    data = [N, M, d, table, 99, 0, 0]
    bestvlc = None
    besthufftab = None
    s = 1.2
    for k in range(91, 101):
        for stepsize in range(40,90,10):
            k/=100
            stepsize1 = scipy.optimize.minimize_scalar(fun,bounds=(1,300)).x
            Pf, Pr = pot_ii(N, s=1.2)
            Y = lbt(X, Pf, N)
            Y = regroup(Y, N)
            Y[:256//N, :256//N] = lbt(Y[:256//N, :256//N], Pf, N)
            Y[:256//N, :256//N] = regroup(Y[:256//N, :256//N], N)
            for i in range(N):
                for j in range(N):
                    if table == 0:
                        Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant1(Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], stepsize1*k**(i+j), stepsize1*k**(i+j))
                    if table == 1:
                        Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant1(Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], ltable[i][j]*stepsize1*k**(i+j), ltable[i][j]*stepsize1*k**(i+j))                
                    if table == 2:
                        Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant1(Y[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], ctable[i][j]*stepsize1*k**(i+j), ctable[i][j]*stepsize1*k**(i+j))

            Y[:256//N//N, :256//N//N] = lbt(Y[:256//N//N, :256//N//N], Pf, N)
            for i in range(N):
                for j in range(N):
                    if i==0 and j==0:
                        pass
                    else:
                        if table == 0:
                            Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], stepsize*k**(i+j), stepsize*k**(i+j))
                        if table == 1:
                            Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ltable[i][j]*stepsize*k**(i+j), ltable[i][j]*stepsize*k**(i+j))
                        if table == 2:
                            Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant1(Y[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ctable[i][j]*stepsize*k**(i+j), ctable[i][j]*stepsize*k**(i+j))

            Y[:256//N, :256//N] = regroup(Y[:256//N, :256//N], 256//N//N)
            Y = regroup(Y, 256//N)
            vlc, dhufftab = myjpegenc(Y, stepsize, N, M, dcbits = d, opthuff=True,log=False)

            Zi = myjpegdec(vlc, stepsize, N, M, dcbits = d, hufftab=dhufftab,log=False)
            Zi = regroup(Zi, N)
            Zi[:256//N, :256//N] = regroup(Zi[:256//N, :256//N], N)
            for i in range(N):
                for j in range(N):
                    if i==0 and j==0:
                        pass
                    else:
                        if table == 0:
                            Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], stepsize*k**(i+j), stepsize*k**(i+j))
                        if table == 1:
                            Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ltable[i][j]*stepsize*k**(i+j), ltable[i][j]*stepsize*k**(i+j))
                        if table == 2:
                            Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ctable[i][j]*stepsize*k**(i+j), ctable[i][j]*stepsize*k**(i+j))

            Zi[:256//N//N, :256//N//N] = ilbt(Zi[:256//N//N, :256//N//N], Pr, N)
            for i in range(N):
                for j in range(N):
                    if table == 0:
                        Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant2(Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], stepsize1*k**(i+j), stepsize1*k**(i+j))
                    if table == 1:
                        Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant2(Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], ltable[i][j]*stepsize1*k**(i+j), ltable[i][j]*stepsize1*k**(i+j))
                    if table == 2:
                        Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant2(Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], ctable[i][j]*stepsize1*k**(i+j), ctable[i][j]*stepsize1*k**(i+j))

            Zi[:256//N, :256//N] = regroup(Zi[:256//N, :256//N], 256//N//N)
            Zi[:256//N, :256//N] = ilbt(Zi[:256//N, :256//N], Pr, N)
            Zi = regroup(Zi, 256//N)
            Z = ilbt(Zi, Pr, N)
            if np.std(X-Z) < data[4]:
                data[4] = np.std(X-Z)
                data[5] = k
                data[6] = stepsize
                bestvlc = vlc
                besthufftab = dhufftab

    header = [data, besthufftab]
    return bestvlc, header
  
  
  
def dec_three_lbt(vlc, hufftab, N, M, stepsize, k, d, table, mylog=False):
    Pf, Pr = pot_ii(N, s=1.2)
    Zi = myjpegdec(vlc, stepsize, N, M, dcbits = d, hufftab=dhufftab,log=False)
    Zi = regroup(Zi, N)
    Zi[:256//N, :256//N] = regroup(Zi[:256//N, :256//N], N)
    for i in range(N):
        for j in range(N):
            if i==0 and j==0:
                pass
            else:
                if table == 0:
                    Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], stepsize*k**(i+j), stepsize*k**(i+j))
                if table == 1:
                    Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ltable[i][j]*stepsize*k**(i+j), ltable[i][j]*stepsize*k**(i+j))
                if table == 2:
                    Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N] = quant2(Zi[i*256//N:i*256//N+256//N, j*256//N:j*256//N+256//N], ctable[i][j]*stepsize*k**(i+j), ctable[i][j]*stepsize*k**(i+j))

    Zi[:256//N//N, :256//N//N] = ilbt(Zi[:256//N//N, :256//N//N], Pr, N)
    for i in range(N):
        for j in range(N):
            if table == 0:
                Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant2(Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], stepsize1*k**(i+j), stepsize1*k**(i+j))
            if table == 1:
                Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant2(Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], ltable[i][j]*stepsize1*k**(i+j), ltable[i][j]*stepsize1*k**(i+j))
            if table == 2:
                Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N] = quant2(Zi[i*256//N//N:i*256//N//N+256//N//N, j*256//N//N:j*256//N//N+256//N//N], ctable[i][j]*stepsize1*k**(i+j), ctable[i][j]*stepsize1*k**(i+j))

    Zi[:256//N, :256//N] = regroup(Zi[:256//N, :256//N], 256//N//N)
    Zi[:256//N, :256//N] = ilbt(Zi[:256//N, :256//N], Pr, N)
    Zi = regroup(Zi, 256//N)
    Z = ilbt(Zi, Pr, N)

    return Z
  
  
 



