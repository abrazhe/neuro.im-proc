# Scent: spatial complexity-entropy via shearhlets
from __future__ import division # will remove this after switch to Py3

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


from FFST import (scalesShearsAndSpectra,
                  inverseShearletTransformSpect,
                  shearletTransformSpect)
from FFST._fft import ifftnc  # centered nD inverse FFT

from imfun.filt import dctsplines
from imfun import ui


def jmax(M,logfn=np.log2):
    "Max Jensen-Shannon divergence in a system of M states"
    return -0.5*((M+1)*logfn(M+1)/M -2*logfn(2*M) + logfn(M))


def parity_crop(f):
    sh = f.shape
    parity = [s%2==0 for s in sh]
    if parity[0] != parity[1]:
        k = np.argmax(sh)
        if k == 0:
            f = f[1:,:]
        else:
            f = f[:,k:]
    return f


def scale_slices(N):
    out = [slice(0,1)]
    n = 1
    k = 2
    while n < N:
        out.append(slice(n,n+2**k))
        n = n+2**k
        k += 1
    return out

def regroup_coefs(coefs):
    sh = coefs.shape
    ncoefs = sh[-1]
    coefs = np.squeeze(split(coefs, ncoefs, -1))
    slices = scale_slices(ncoefs)
    return  [coefs[s] for s in slices]

def shannon(v,logfn=np.log2):
    return -np.sum(v[v>0]*logfn(v[v>0]))
 

def cecp(probs,logfn=np.log2):
    "cecp: calculate complexity-entropy causality pair"
    M = len(probs)
    pe = np.ones(M)/M
    #Jmax = -0.5*((M+1)*log2(M+1)/M - 2*log2(2*M) + log2(M)) # Ribeiro
    #Jmax = 1.0 # because I use log base 2
    Sp = shannon(probs,logfn)
    J = shannon(0.5*(probs+pe)) - 0.5*Sp - 0.5*shannon(pe)
    Hr = Sp/logfn(M)
    Cr = (J/jmax(M,logfn))*Hr
    return (Hr,Cr)

def global_cecp_shearlets(ST,band_start=1,logfn=np.log2, central_slice=None,do_rescale=True):
    "cecp-shearlets: complexity-entropy causality pair for an image"
    nr,nc,fullncomps = ST.shape
    slx = scale_slices(fullncomps)
    #ST = abs(ST)
    #ST = np.exp(ST**2) # interpret |<x,\phi_i>|^2 as log probability
    ST = ST**2
    
    scale_starts = [sl.start for sl in slx]
    kstart = scale_starts[band_start]
    #ST = ST[...,kstart:]
    if central_slice is None:
        central_slice = (slice(None),slice(None))
    #E = ST.sum(0).sum(0) # sum over all locations, analogous to Ej in Rosso et al 2001
    E = ST[central_slice].sum(0).sum(0) # sum over all locations, analogous to Ej in Rosso et al 2001

    if do_rescale:
        ax = 1/(4**np.arange(len(slx)+2))
        for i, sl in enumerate(slx):
            E[sl] *= ax[i]**(6/4)
        E[slx[-1]] *= 2.5 # don't know why, but with this the energy is more uniform
        
    E = E[kstart:]
    prob = E/np.sum(E)
    return cecp(prob,logfn)


def cecp_from_shearlets(img,band_start=2,pmask=None,rho=3,verbose=False,logfn=np.log2,
                        Psi=None,
                        do_rescale=True,
                        numOfScales=None):
    img = parity_crop(img)
    img_range = np.abs(img.max()-img.min())
    ST,_ = shearletTransformSpect(img,Psi=Psi,numOfScales=numOfScales)
    
    _,_,ncomps = ST.shape
    slx = scale_slices(ncomps)
    
    #slice_sizes = [1] + [2**k for k in range(2, len(slx)+1)]
    #scale_starts = #np.cumsum(slice_sizes)
    scale_starts = [sl.start for sl in slx]
    kstart = scale_starts[band_start]
    if verbose:
        print('rho:', rho)
        print('Kstart:', kstart)

    details = ST[...,slx[-1]]
    sdmap = np.std(details,-1)
    details_range = abs(details.max()-details.min())
    
    ST = ST[...,kstart:]
    ST = ST**2
    
    nscales = len(slx[band_start:])

    if do_rescale:
        ax = 1/(4**np.arange(len(slx)+2))
        for i, sl in enumerate(slx[band_start:]):
            sl2 = slice(sl.start-kstart,sl.stop-kstart)
            ST[...,sl2] *= ax[i]**(6/4)
        ST[...,slx[-1]] *= 2.5 # don'# TODO:  know why, but with this the energy is more uniform
        

    
    if rho > 0:
        for i,sl in enumerate(slx[band_start:]):
            sl2 = slice(sl.start-kstart,sl.stop-kstart)
            rhox = rho*2**(nscales-i-1)
            #print('rhox', rhox)
            ST[...,sl2] = ndimage.gaussian_filter(ST[...,sl2],sigma=(rhox,rhox,0))
    
    probs = ST/ST.sum(-1)[:,:,None] 
    #probs = np.exp(ST**2)              # if <x,\phi>^2 is ~ to log probability
    #probs = probs/probs.sum(-1)[:,:,None] # -- normalization
   
    
    S = -np.sum(probs*logfn(1e-8+probs),-1)
    M = probs.shape[-1]
    #pe =ones(M)/M
    PPr = 0.5*(probs+1/M)
    J = -np.sum((PPr)*logfn(1e-8+PPr),-1) - 0.5*(S + logfn(M))
    Hr = S/logfn(M)
    Cr = J*Hr/jmax(M,logfn)
    
    if pmask is "nonsmooth":
        pmask = sdmap>0.01*details_range
        pmask = ndimage.binary_fill_holes(pmask)
    if pmask is not None and not isinstance(pmask,str):
        Hr *= pmask
        Cr *= pmask
    return Hr, Cr


def calc_cecp(img,startscale=4,
              rho=3,pad_to=1024,
              add_noise_amp=0.0,
              verbose=False,
              pmask=None,
              logfn = np.log2,
              with_plots=True,
              Psi=None,
              do_rescale=True,
              numOfScales = None):
    if verbose:
        print('img.shape before', img.shape)
    img,npad = pad_img(img, pad_to)
    nr,nc = img.shape
    crop = (slice(npad,-npad),)*2
    if verbose:
        print('crop:',crop)
        print('img.shape after', img.shape)
    img = img + add_noise_amp*np.random.randn(*img.shape)
    H,C = cecp_from_shearlets(img,startscale,pmask=pmask,rho=rho,Psi=Psi,numOfScales=numOfScales,do_rescale=do_rescale)
    if npad > 0:
        img,H,C = (a[crop] for a in (img, H,C))
    #H = DKL_from_shearlets(img,startscale)
    if verbose:
        print(np.amin(H),np.amax(H),np.amin(C),np.amax(C))
    #H = ndimage.gaussian_filter(H,rho)
    #C = ndimage.gaussian_filter(C,rho)
    if with_plots:
        to_show = (img, H,C )
        ui.group_maps(to_show, 
                      titles = ('orig','H','C','mask'),
                      samerange=0,individual_colorbars=1,figscale=5);
        f = plt.gcf()
        f.axes[2].images[0].set_cmap('magma')
        f.axes[4].images[0].set_cmap('inferno')
        plt.tight_layout()
        #f.axes[2].images[0].set_clim((0,1))
        
        f.set_size_inches(f.get_size_inches()+np.array([2.5,0]))
    return H,C


def pad_img(img,targ=1024):
    npad = max(0,int(np.ceil(np.max(targ-np.array(img.shape))*0.5)))
    #print 'pad width:', npad
    return np.pad(img, npad, 'constant'),npad
        
