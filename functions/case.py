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
    for k in range(91, 101):
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
