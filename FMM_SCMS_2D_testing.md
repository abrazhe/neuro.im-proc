---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Testing FMM-based and SCMS-inspired approaches to segment astrocyte images in 2D

```{code-cell} ipython3
import os
import sys
```

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
import cv2
```

```{code-cell} ipython3
from functools import reduce
import operator as op
```

```{code-cell} ipython3
from importlib import reload
```

```{code-cell} ipython3
import scipy
from scipy import ndimage as ndi
import numpy as np
import networkx as nx

from pathlib import Path
```

```{code-cell} ipython3
import napari
```

```{code-cell} ipython3
import scipy as sp
```

```{code-cell} ipython3
#import xarray
```

```{code-cell} ipython3
from tqdm.auto import tqdm
```

```{code-cell} ipython3
import ccdb
import astromorpho as astro
```

```{code-cell} ipython3

```

```{code-cell} ipython3
from networx2napari import draw_edges, draw_nodes

import graph_utils as gu  
import visualization as vis
```

```{code-cell} ipython3
def eu_dist(p1, p2):
    return np.sqrt(np.sum([(x - y)**2 for x, y in zip(p1, p2)]))
```

```{code-cell} ipython3
def get_shell_mask(mask, do_skeletonize=False, as_points=False):
    out = ndi.binary_erosion(mask)^mask
    if do_skeletonize:
        out = skeletonize(out)
    if as_points:
        out = astro.morpho.mask2points(out)
    return out 
```

```{code-cell} ipython3
from skimage.filters import threshold_li, threshold_minimum, threshold_triangle
from skimage.morphology import remove_small_objects
```

```{code-cell} ipython3
def largest_region(mask):
    labels, nlab = ndi.label(mask)
    if nlab > 0:
        objs = ndi.find_objects(labels)
        sizes = [np.sum(labels[o]==k+1) for k,o in enumerate(objs)]
        k = np.argmax(sizes)
        return labels==k+1
    else:
        return mask
        
def crop_image(img, mask=None, margin=0, min_obj_size=0):
    if mask is None:
        mask = img > 0
    if min_obj_size > 0:
        mask = remove_small_objects(mask, min_obj_size)
    if margin > 0:
        mask = ndi.binary_dilation(mask, iterations=margin)
    objs = ndi.find_objects(mask)
    min_bnds = np.min([[sl.start for sl in o] for o in objs],0)
    max_bnds = np.max([[sl.stop for sl in o] for o in objs],0)
    crop = tuple(slice(mn,mx) for mn,mx in zip(min_bnds, max_bnds))
    return img[crop]
```

```{code-cell} ipython3

```

```{code-cell} ipython3
def percentile_rescale(arr, plow=1, phigh=99):
    vmin,vmax = np.percentile(arr, (plow, phigh))
    if vmin == vmax:
        return np.zeros_like(arr)
    else:
        return np.clip((arr-vmin)/(vmax-vmin),0,1)
```

```{code-cell} ipython3
def flat_indices(shape):
    idx = np.indices(shape)
    return np.hstack([np.ravel(x_)[:,None] for x_ in idx])
```

```{code-cell} ipython3

```

```{code-cell} ipython3
def show_field2d(vfield, background=None, crop=None, weights=None, scale=25, ax=None, figsize=(9,9)):
    
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize); 
    else:
        fig = ax.figure
    
    if crop is None:
        crop=(slice(None,),slice(None))
    
    if background is not None:
        ax.imshow(background[crop], cmap='gray')
    
    V = -vfield[crop][...,0]# ROW (Y) directions; negative sign due to 'origin="upper" by default in imshow'
    U = vfield[crop][...,1]
        
    ax.quiver(U,V, weights[crop], scale=scale, cmap='inferno')
    return fig
```

```{code-cell} ipython3
from skimage import feature as skf
import itertools as itt

_symmetric_image = (skf.corner._hessian_matrix_image if '_hessian_matrix_image' in dir(skf.corner)
                    else skf.corner._symmetric_image)


def hessian_by_dog(img, sigma, rel_scale=None, return_gradient=False,):
    ndim = np.ndim(img)
    if rel_scale is None:
        rel_scale = np.ones(ndim)
    ax_pairs = itt.combinations_with_replacement(range(ndim),2)
    sigma = sigma/np.sqrt(2)
    trunc = 6 # default
    if np.any(sigma*trunc < 3):
        trunc = 3/np.min(sigma)
    def dog(m,k):
        o = np.zeros(ndim, int)
        o[k] = 1
        g = ndi.gaussian_filter(m, sigma, order=o, truncate=trunc)
        return g#/rel_scale[k]**2
    
    double_dog = lambda axp: dog(dog(img, axp[0]),axp[1])
    out = [double_dog(axp) for axp in ax_pairs]
    if return_gradient:
        g = [dog(img, ax)/rel_scale[ax]**2 for ax in range(ndim)]
        return out, g
    else:
        return out



def hessian_eigen_decomp(H):
    #Hmat = skf.corner._hessian_matrix_image(H)
    # note we should ensure that eigenvalues in *descending* order
    # that is, in a bright filamentous structure the first eigenvalue should have small 
    # absolute value, while last eivenvalues should be negative and have large absolute 
    # value
    
    Hmat = _symmetric_image(H)
    w,v = np.linalg.eigh(Hmat)
    return w[...,::-1],v[...,::-1]


def barebone_sato(eigenvalues, gamma12=0.5, gamma23=0.5, alpha=0.25):

    lams = eigenvalues    
    
    lam1,lam2,lam3 = [lams[...,i] for i in range(3)]
    ratio1 = np.where(lam3!=0, lam2/(1e-6 + lam3),0)
    ratio2 = lam1/(1e-6 + np.abs(lam2))
    
    out = np.where(lam1 < 0, 
                   np.abs(lam3)*np.abs(ratio1)**gamma23*np.abs(1 + ratio2)**gamma12,
                   np.where((lam2 < 0) & (lam1 < np.abs(lam2)/alpha), 
                            np.abs(lam3)*np.abs(ratio1)**gamma23*np.abs(1 - alpha*ratio2)**gamma12,0))
    return out.astype(np.float32)
```

```{code-cell} ipython3
def get_multi_vecs_interp(locs, field, order=1):
    locs = np.asarray(locs)
    dims = np.arange(field.shape[-1])
    return np.array([ndi.map_coordinates(field[...,i], locs.T, order=order) for i in dims]).T
```

```{code-cell} ipython3
def project_points(points, stopped=None, img=None, ax=None,
                   alpha=0.25,
                   markersize=2,
                   xaxis=1,
                   yaxis=0,
                   figsize=(16,16)):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.axis('off')
        plt.tight_layout()
   
    if not len(ax.images) and img is not None:
        ax.imshow(img, cmap='gray')
    
    if not len(ax.lines):
        ax.plot(points[:,xaxis],points[:,yaxis], 'm.',markersize=markersize, 
                alpha=alpha)
        if stopped is not None:
            ax.plot(stopped[:,xaxis],stopped[:,yaxis], 'g.',markersize=1.5*markersize, 
                    alpha=alpha)
    else:
        lh = ax.lines[0]
        lh.set_data(points[:,xaxis],points[:,yaxis])
        if stopped is not None:
            if len(ax.lines) < 2:
                ax.plot(stopped[:,xaxis],stopped[:,yaxis], 'g.',
                        markersize=1.5*markersize, alpha=alpha)
            else:
                lh = ax.lines[1]
                lh.set_data(stopped[:,xaxis],stopped[:,yaxis])

    ax.figure.canvas.draw()
    return ax
```

**TODO:**
 - [ ] check if limiting the total travelled path for a particle (to ~1...2σ) would stop the particles from fragmentation? Will probably have to prune all particles that failed to reach a skeleton afterwards (possibly, via svd or something like that. Or another simulation stage)
 - [ ]

```{code-cell} ipython3
def calc_trajectory_basic(field, pts0, n_iter=10, tol=1e-3,
                          h=0.25,
                          max_dist =10000,
                          with_plot=False,
                          plot_save_pattern=None,
                          save_interval=10,
                          background=None,
                          knn=6,
                          agg_alpha=0.1,
                          gamma = 0.99):
    pts_prev = pts0
    travel_dist = np.zeros(len(pts0))

    if background is None:
        img_proj = np.zeros(field.shape[:-1])
    else:
        img_proj = background
    
    frozen = np.zeros(len(pts0), bool)
    stationary_counter = np.zeros(len(pts0), np.uint16)
    vec_prev = None
    
    ax = None

    # establish nearest-neighbors at the beginning
    # number of neighbors could be a parameter
    if knn > 1:
        kdt0 = sp.spatial.KDTree(pts_prev)
        nn_dists, nn_inds = kdt0.query(pts_prev, knn)    
        
    mult = 1.0
    for i in tqdm(range(n_iter)):

        if knn > 1:
            agg_force = (pts_prev[:,None,:] - pts_prev[nn_inds]).sum(1) 
        else:
            agg_force = 0

        

        # todo compare cos between previous vector orientations and new vector orientations
        # if angle > 90, freeze theze points. 
        vec = get_multi_vecs_interp(pts_prev, field)

        vec = vec - agg_alpha*agg_force
        
        vec[frozen] = 0 # don't move points that are already considered converged
        
        #pts = pts_prev + mult*h*vec

        # simple adams-bashforth scheme
        if vec_prev is None:
            pts = pts_prev + h*mult*vec      
        else:
            pts = pts_prev + h*mult*(1.5*vec - 0.5*vec_prev)        
        
        delta = np.linalg.norm(pts-pts_prev, axis=1)
        travel_dist += delta
        stop_cond = (delta < tol) | (travel_dist > max_dist) 
        stationary_counter[stop_cond] += 1
        frozen[stationary_counter > 5] = True

        if with_plot and not i%save_interval:
            ax = project_points(pts[~frozen], pts[frozen], img=img_proj, ax=ax)
            if plot_save_pattern is not None:
                ax.figure.savefig(plot_save_pattern.format(i=i))
        # switch  states
        pts_prev = pts
        vec_prev = vec
        mult *= gamma    
        if np.all(frozen):
            break        
    return pts
```

```{code-cell} ipython3
import skfmm
from skimage.morphology import dilation, skeletonize, flood
from skimage import measure
from astromorpho import morpho
```

## Load stupid test image

```{code-cell} ipython3
import tifffile
```

```{code-cell} ipython3
#%matplotlib qt
```

```{code-cell} ipython3
img = plt.imread('branches-blurred.png').sum(-1)
#img = plt.imread('branches-lessblurred.png').sum(-1)
#img = ndi.gaussian_filter(img, 1.5)
img += 100 +  0.1*np.random.randn(*img.shape)
img.min(), img.max()
```

```{code-cell} ipython3
# I wonder if it's downsampled. Maybe should work on original instead
#img = tifffile.imread('astro_3wk-both1-grn-raw.pic-maxproj.tif')
img = tifffile.imread('astro_4wk-ly24-raw.pic-ds_1-maxproj.tif')
```

```{code-cell} ipython3
img.shape
```

```{code-cell} ipython3
import ucats as uc
```

```{code-cell} ipython3
plt.close('all')
```

```{code-cell} ipython3
plt.figure()
plt.imshow(img, cmap='gray'); plt.colorbar()
#plt.gcf()
```

```{code-cell} ipython3
import morphsnakes
```

```{code-cell} ipython3
macwe = morphsnakes.MorphACWE(img, lambda2=2, smoothing=1,levelset=img > 0)
macwe.run(500)
```

```{code-cell} ipython3
#full_mask = ndi.binary_opening(ndi.binary_closing(macwe.levelset))
full_mask = ndi.binary_opening(ndi.binary_closing(img > 1))
full_mask = ndi.binary_fill_holes(full_mask)
full_mask = ndi.binary_dilation(uc.masks.largest_region(full_mask))
```

```{code-cell} ipython3
plt.imshow(img, cmap='gray')
plt.contour(full_mask, [0.5], colors=['r'])
```

```{code-cell} ipython3
astro.morpho.eigh = np.linalg.eigh
```

```{code-cell} ipython3
%matplotlib inline
```

## Set image center and initial levelset

```{code-cell} ipython3
smooth = ndi.gaussian_filter(img, 10)
#plt.imshow(smooth)
```

```{code-cell} ipython3
X1a = flat_indices(img.shape)
weights_s = percentile_rescale(np.ravel(smooth)**2,plow=99.5,phigh=99.99)
center = tuple(map(int, np.sum(X1a*weights_s[:,None],axis=0)/np.sum(weights_s)))
center_2 =tuple(measure.centroid(smooth**2).astype(int))
center, center_2
```

```{code-cell} ipython3
plt.hist(np.ravel(img),100);
th = threshold_minimum(img)
plt.axvline(th, color='r', ls='--')
```

```{code-cell} ipython3
tol = (smooth.max() - smooth[img>th/2].min())/50
soma_mask = flood(smooth, center, tolerance=tol)
soma_mask = morpho.expand_mask(soma_mask, smooth, iterations = 5)
```

```{code-cell} ipython3
plt.imshow(percentile_rescale(img), cmap='gray')
plt.contour(soma_mask,levels=[0.5], colors=['r'])
plt.plot(center[1], center[0], 'g+')
#plt.plot(center_2[1], center_2[0], 'b+')
```

```{code-cell} ipython3

```

## Quick test at a single sigma

```{code-cell} ipython3
sigma0 = 6
```

```{code-cell} ipython3
%time H,g  = hessian_by_dog(img, sigma=sigma0, rel_scale=None, \
                            return_gradient=True) # RC order by default

g = sigma0*np.stack(g,axis=2)
```

```{code-cell} ipython3
g.shape 
```

```{code-cell} ipython3
%time lams,Vf = hessian_eigen_decomp(H)
```

```{code-cell} ipython3
#np.mean(sato[~full_mask])
```

```{code-cell} ipython3
#%time sato = barebone_sato(lams)
sato0 = astro.morpho.sato2d(img, sigma0)
threshold = threshold_li(sato0[sato0>0])#*sigma0**0.5/
#threshold = threshold_li(sato[sato0>0])*sigma0**0.5
#mask = remove_small_objects(sato0 > threshold/2, min_size=int(sigma0*32))
mask = remove_small_objects(full_mask*(sato0 > np.mean(sato0[~full_mask])), 
                            min_size=int(sigma0*32))
#mask = img > 10
```

```{code-cell} ipython3
plt.imshow((sato0>0)*full_mask)
plt.contour(mask, colors=['r'])
```

```{code-cell} ipython3
plt.close('all')
%matplotlib inline
plt.rc('figure', dpi=150)
```

```{code-cell} ipython3
plt.imshow(sato0)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
def get_VVg(img, sigma):
    H,g  = hessian_by_dog(img, sigma=sigma, rel_scale=None, \
                            return_gradient=True) # RC order by default

    g = sigma*np.stack(g,axis=2)
    lams,Vf = hessian_eigen_decomp(H)
    
    VV = np.einsum('...ij,...jk', Vf[...,1:], np.einsum('...ji', Vf[...,1:]))
    
    # gradient projections onto Hessian-based direction
    VVg = np.einsum('...ij,...j->...i', VV, g)
    
    VVg_mag = np.linalg.norm(VVg,axis=-1)    

    pth = np.percentile(VVg_mag, 99)
    VVg_clipped = VVg/pth
    VVg_mag2 = np.linalg.norm(VVg_clipped,axis=-1)
    return VVg, VVg_clipped, VVg_mag2
```

```{code-cell} ipython3
# %time VV = np.einsum('...ij,...jk', Vf[...,1:], np.einsum('...ji', Vf[...,1:]))
# %time VVg = np.einsum('...ij,...j->...i', VV, g)
```

```{code-cell} ipython3
# VVg_mag = np.linalg.norm(VVg,axis=-1)
# VVg_sign = np.sign(np.sum(VVg,axis=-1))
# VVg_mag_max = np.max(VVg_mag)
```

```{code-cell} ipython3
# pth = np.percentile(VVg_mag, 99)
# VVg_clipped = VVg/pth
# #cond = np.linalg.norm(VVg_clipped,axis=-1)>1
# VVg_clipped[VVg_mag>pth] /= VVg_mag[VVg_mag>pth][:,None]/pth
```

```{code-cell} ipython3
#VVg_mag2 = np.linalg.norm(VVg_clipped,axis=-1)
```

### Define speed as just brightness

```{code-cell} ipython3
from imfun.bwmorph import neighbours
```

```{code-cell} ipython3
def stupid_2d_gd(field, p0, step=1, nsteps=100, max_drop=1):
    p0 = tuple(map(int, p0))
    traj = [p0]
    
    for i in range(nsteps):
        p = tuple(traj[-1])
        u = field[p]
        for n in neighbours(p, field.shape):
            n = tuple(map(int, n))
            v = field[n]
            if v < u and (u-v < max_drop):
                u = v
                p = n
        if p == traj[-1]:
            break
        traj.append(p)
    return np.array(traj)
```

```{code-cell} ipython3
speed0 = percentile_rescale(img, 0.1, 99.9)
tt_bright = skfmm.travel_time(1.-soma_mask, speed=speed0)
```

```{code-cell} ipython3
plt.imshow(img, cmap='gray')
plt.imshow(tt_bright, cmap='Spectral', vmin=0, alpha=0.5)
```

```{code-cell} ipython3
#locs = [(51,152),(79,26)] # for 3wk-both1-grn-raw.pic
locs = [(180,110), (800,500)]

paths_b = [stupid_2d_gd(tt_bright, loc, nsteps=10000, max_drop=30000) 
           for loc in locs]
len(paths_b[0])
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(uc.clip_outliers(img),cmap='gray')
plt.contour(full_mask, levels=[0.5], colors='g')
for p in paths_b:
    plt.plot(p[:,1], p[:,0],'tomato',lw=1,alpha=0.5)
```

It's already not so bad, but we shall go further

+++

### Define speed as ratio between sato contrast and inverse gradient

```{code-cell} ipython3
VVg, VVg_capped, VVg_mag2 = get_VVg(img, sigma0)
```

```{code-cell} ipython3
inv_grad = 1/(0.1 + VVg_mag2)
```

**NB**, how to exploit the "void" areas around the centerline?

```{code-cell} ipython3
plt.figure(figsize=(8,8))
plt.imshow(full_mask*inv_grad)
```

```{code-cell} ipython3
plt.figure(figsize=(8,8))
plt.imshow(VVg_mag2)
#plt.gcf()
```

```{code-cell} ipython3
from imfun import ui
ui.group_maps([percentile_rescale(m) 
               for m in (sato0, VVg_mag2, 
                         sato0/(0.1 + VVg_mag2), 
                         sato0*np.exp(-VVg_mag2*2))], 
              figscale = 5,
              colorbar=False)
```

```{code-cell} ipython3
plt.imshow(uc.clip_outliers(sato0/(0.1 + VVg_mag2)), cmap='plasma')
```

```{code-cell} ipython3
plt.imshow(sato0)
plt.contour(full_mask,colors=['r'])
```

```{code-cell} ipython3
##plt.imshow(sato>0)
```

```{code-cell} ipython3
speed0= full_mask*(sato0/(0.1 + VVg_mag2))
#speed = mask*(1/(0.1 + VVg_mag2))
#speed = sato
print(speed0.max())
speed0 = percentile_rescale(speed0, 0.001,99.9)
print(speed0.max())
```

```{code-cell} ipython3
plt.imshow(speed0, cmap='plasma')
```

```{code-cell} ipython3
tt_sato = skfmm.travel_time(1.-soma_mask, speed=speed0)
```

```{code-cell} ipython3
bounds = morpho.boundary_pixels(ndi.binary_erosion(full_mask,iterations=2))
bounds[:4]
```

```{code-cell} ipython3
plt.imshow(uc.clip_outliers(img),cmap='gray')
plt.imshow(tt_sato,cmap='Spectral', alpha=0.85,vmin=0); plt.colorbar()
loc0 = (16, 95)
plt.contour(soma_mask,levels=[0.5], colors=['k'])
# plt.axhline(loc0[0])
# plt.axvline(loc0[1])
# plt.axis((50,100, 50,0))
```

```{code-cell} ipython3
speed_low = np.percentile(speed0[mask], 10)
```

```{code-cell} ipython3
u,v = np.gradient(tt_sato)
plt.hist(np.ravel(np.abs(u[mask*(speed0 <= speed_low)])),50,log=False, histtype='step');
plt.hist(np.ravel(np.abs(v[mask*(speed0 <= speed_low)])),50, log=False, histtype='step');
```

```{code-cell} ipython3
(threshold_triangle(abs(u[mask*(speed0 <= speed_low)])),
 threshold_triangle(abs(v[mask*(speed0 <= speed_low)])))
```

```{code-cell} ipython3
#import plotly.express as px
```

```{code-cell} ipython3
# fig = px.imshow(img,binary_string=True)
# fig.update_layout(width=700,height=700)
# fig
```

```{code-cell} ipython3
paths = [stupid_2d_gd(tt_sato, loc, nsteps=10000, max_drop=300) 
           for loc in locs]
list(map(len, paths))
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(uc.clip_outliers(img),cmap='gray')
plt.contour(full_mask, levels=[0.5], colors='b')
plt.contour(mask, levels=[0.5], colors='g')

for p in paths_b:
    plt.plot(p[:,1], p[:,0],'lime',lw=1,alpha=0.5)

for p in paths:
    plt.plot(p[:,1], p[:,0],'m',lw=1,alpha=0.5)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
sparse_locs = np.random.permutation(bounds)[:1000]
sparse_locs[:4]
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(uc.clip_outliers(img),cmap='gray')
plt.plot(sparse_locs[:,1],sparse_locs[:,0], 'r.')
```

```{code-cell} ipython3
#np.max(tt_sato)
```

```{code-cell} ipython3
len(bounds)
```

```{code-cell} ipython3
paths = [stupid_2d_gd(tt_sato, tuple(loc), nsteps=1000, max_drop=1000) 
         for loc in tqdm(bounds) if tt_sato[tuple(loc)]]
paths = [p for p in paths if len(p) > 50 and soma_mask[tuple(p[-1])]]
#paths = [p for p in paths if len(p) > 150]
```

```{code-cell} ipython3
len(paths)
```

```{code-cell} ipython3
#soma_mask[tuple(paths[10][-1])]
```

```{code-cell} ipython3

```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(uc.clip_outliers(img),cmap='gray')
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(uc.clip_outliers(img),cmap='gray')
for path in paths:
    color = np.random.rand(3)
    plt.plot(path[:,1], path[:,0],lw=1,alpha=0.5,color=color)

#plt.axis([100,150, 150,100])
plt.title(f'$\\sigma$ ={sigma0:1.1f}')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
from numba import jit
```

## Now let's try to combine several scales

```{code-cell} ipython3
sigmas = [1.5, 2, 3, 4, 6, 8, 12, 16]
#sigmas = [1]
```

```{code-cell} ipython3
sato_coll ={sigma:astro.morpho.sato2d(img, sigma) for sigma in tqdm(sigmas)}
vvg_coll = {sigma:get_VVg(img, sigma)[2] for sigma in tqdm(sigmas)}

# shall I multiply by sigma^2 here?
speed_coll = {sigma:full_mask*sigma**2*sato_coll[sigma]/(0.1 + vvg_coll[sigma]) 
              for sigma in sigmas}
```

```{code-cell} ipython3

```

```{code-cell} ipython3
sato_best_sigma = np.argmax([v*sigma**2 for sigma,v in sato_coll.items()],0) + 1
sato_best_sigma[~full_mask] = 0
sato_best_sigma = np.ma.masked_where(~full_mask, sato_best_sigma)
```

```{code-cell} ipython3
ui.group_maps([uc.clip_outliers(sp) for sp in speed_coll.values()], samerange=False, figscale=3)
```

```{code-cell} ipython3
ui.group_maps([sato_best_sigma==i+1 for i in range(len(sigmas))], titles=sigmas, 
              figscale=5, colorbar=False)
plt.tight_layout()
```

```{code-cell} ipython3
id2sigma = {i+1:sigma for i, sigma in enumerate(sigmas)} # shift by one, so that zero doesn't correspond to a cell
sigma2id = {sigma:i+1 for i, sigma in enumerate(sigmas)}
```

```{code-cell} ipython3

```

```{code-cell} ipython3
masks = {}
for sigma in tqdm(sigmas):
    sato = sato_coll[sigma]*sigma**2
    threshold = threshold_li(sato[sato>0])#*sigma**0.5
    #print(sigma, threshold, np.sum(sato > threshold), sigma*32)
    masks[sigma] = remove_small_objects(sato > threshold, min_size=int(sigma*64))
```

```{code-cell} ipython3
plt.imshow(masks[sigmas[6]])
```

```{code-cell} ipython3
ui.group_maps(list(masks.values()), colorbar=False)
```

```{code-cell} ipython3
masks[sigmas[-1]] = uc.masks.select_overlapping(masks[sigmas[-1]], soma_mask)
```

```{code-cell} ipython3
mask_sum = np.zeros(img.shape,bool)
masks_exclusive = {}

for k in range(len(sigmas)-1,-1,-1):
# for k in range(len(sigmas)):
    sigma = sigmas[k]
    mask = masks[sigma]
    if k < len(sigmas)-1:
        mask = mask & (mask ^ mask_sum)
    mask_sum += mask.astype(bool)
    masks_exclusive[sigma] = mask
    
```

```{code-cell} ipython3
ui.group_maps(list(masks_exclusive.values()), colorbar=False)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
speed_ms = np.zeros(speed_coll[sigmas[0]].shape)

for i,sigma in enumerate(sigmas,start=1):
    cond = sato_best_sigma==i
    speed_ms[sato_best_sigma==i] = speed_coll[sigma][cond]

speed_ms = percentile_rescale(speed_ms, 0.01, 99.9)

plt.figure()
plt.imshow(speed_ms, cmap='plasma')
```

```{code-cell} ipython3
speed_ms2 = np.max([v for v in speed_coll.values()],0)
plt.figure()
plt.imshow(speed_ms2, cmap='plasma')
```

```{code-cell} ipython3

```

Note that areas with dominant large sigma can be actually **around** some thinner branches. This has to be corrected somehow.

```{code-cell} ipython3
tt_sato_ms = skfmm.travel_time(1.-soma_mask, speed=speed_ms2)
```

```{code-cell} ipython3
paths_ms = [stupid_2d_gd(tt_sato_ms, tuple(loc), nsteps=1000, max_drop=1000) 
         for loc in tqdm(bounds) if tt_sato[tuple(loc)]]
paths_ms = [p for p in paths_ms if len(p) > 50 and soma_mask[tuple(p[-1])]]
#paths = [p for p in paths if len(p) > 150]
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(uc.clip_outliers(img),cmap='gray')
for path in paths_ms:
    color = np.random.rand(3)
    plt.plot(path[:,1], path[:,0],lw=1,alpha=0.5,color=color)

plt.title('Multiscale sigma, take 1')
```

The problems include straight paths at thin processes and not following branches. Also, misplaced branchpoints.

+++

This is probably because of over-smoothing at large sigmas, and I'm not interested in "shoulders", only in centerlines. 

```{code-cell} ipython3

```

---

## Let's add centerlines, inspired by subspace-constrained mean-shift 
(although it's not really a mean-shift

```{code-cell} ipython3
#plt.imshow(sato)
```

```{code-cell} ipython3
show_field2d(VVg[...,:]/VVg_mag2.max(), background=uc.clip_outliers(img)**0.5, 
             scale=1000, weights=sato);
plt.axis([200,280,280,200])
```

```{code-cell} ipython3
show_field2d(VVg[...,:]/VVg_mag2.max(), background=uc.clip_outliers(img)**0.5, 
             figsize=(12,12),
             scale=500, weights=sato);
plt.axis([75,150,200,50])
plt.title(f'$\\sigma$={sigma}')
#plt.axis([350,450,450,350])
```

```{code-cell} ipython3
show_field2d(g[...,:]/np.linalg.norm(g,axis=-1).max(), 
             background=uc.clip_outliers(img)**0.5, 
             scale=25, weights=sato);
#plt.axis([200,300,300,200])
plt.axis([200,280,280,200])
```

```{code-cell} ipython3
mask.shape
```

```{code-cell} ipython3
plt.figure()
plt.imshow(mask)
```

```{code-cell} ipython3
pts = np.array(np.where(sato>0)).T
pts_all = np.array(np.where(full_mask)).T
```

```{code-cell} ipython3
pts = pts + np.random.randn(*pts.shape)*0.01
```

```{code-cell} ipython3
project_points(pts, img=img)
ax = plt.gca()
#ax.figure
```

```{code-cell} ipython3
rm figures/2dfilaments-*.png
```

```{code-cell} ipython3
#sigma0=3
```

```{code-cell} ipython3
#mkdir figures
```

```{code-cell} ipython3
# %%time 
# agg_alpha=0.1
# endpoints0 = calc_trajectory_basic(VVg_clipped,
#                                    pts, 
#                                    n_iter=1001, 
#                                    tol=0.001, 
#                                    max_dist=300*sigma,
#                                    with_plot=True,
#                                    save_interval=100,
#                                    gamma=1.,
#                                    knn=0,
#                                    agg_alpha=agg_alpha,
#                                    background=img,
#                                    plot_save_pattern=f'figures/2dfilaments-knn_alpha-{agg_alpha:0.2f}'+'-{i:04d}.png')
```

```{code-cell} ipython3
%%time 
agg_alpha=0.1
endpoints0 = calc_trajectory_basic(VVg_capped,
                                   pts, 
                                   n_iter=100, 
                                   tol=0.001, 
                                   max_dist=2*sigma0,
                                   with_plot=True,
                                   save_interval=99,
                                   gamma=1.,
                                   knn=6,
                                   agg_alpha=agg_alpha,
                                   background=img,
                                   plot_save_pattern=f'figures/2dfilaments-knn_alpha-{agg_alpha:0.2f}'+'-{i:04d}.png')
plt.title(f'$\\sigma={sigma0:1.1f}$')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
plt.close('all')
```

```{code-cell} ipython3
#sigma0
```

```{code-cell} ipython3
# %%time 
# endpoints1 = calc_trajectory_basic(VVg_clipped,
#                                   pts, 
#                                   n_iter=1502, 
#                                   tol=0.001, 
#                                   max_dist=2*sigma,
#                                   with_plot=True,
#                                   save_interval=1500,
#                                   gamma=0.999,
#                                   background=img,
#                                   plot_save_pattern='figures/2dfilaments-{i:04d}.png')
# plt.title(f'$\\sigma={sigma0:1.1f}$')
```

```{code-cell} ipython3
#plt.gcf()
```

```{code-cell} ipython3
plt.close('all')
```

```{code-cell} ipython3
#plt.figure(figsize=(6,6))
#plt.tight_layout()
#project_points(endpoints1,img=img**0.5)
#project_points(endpoints0, img=None)
#plt.gcf()
```

```{code-cell} ipython3
project_points(endpoints0,img=img**0.5)
#project_points(endpoints0, img=None)
#plt.gcf()
plt.title(f'$\\sigma={sigma0:1.1f}$')
```

```{code-cell} ipython3
# project_points(endpoints0,img=percentile_rescale(tt_sato))
# plt.title(f'$\\sigma={sigma:1.1f}$')
```

```{code-cell} ipython3
#_=1
```

```{code-cell} ipython3
sato_best_sigma.shape
```

```{code-cell} ipython3
def density_filter(points, radius=1, min_neighbors=2, with_hist_plot=False):
    kdt = sp.spatial.KDTree(points)
    nn = kdt.query_ball_point(points, radius, return_length=True)
    if with_hist_plot:
        fig = plt.figure()
        plt.hist(nn, 200);
    return points[nn >= min_neighbors]
```

```{code-cell} ipython3
pts1 = density_filter(endpoints0,radius=1, min_neighbors=3, with_hist_plot=True)
```

```{code-cell} ipython3
project_points(endpoints0,stopped=pts1, img=img**0.5,figsize=(9,9))
```

```{code-cell} ipython3
sigma_dict = {j:sigma for j,sigma in enumerate(sigmas,start=1)}
sigma_dict_back = {sigma:j for j,sigma in enumerate(sigmas,start=1)}
```

```{code-cell} ipython3

project_points(pts1, img=(sato_best_sigma==sigma_dict_back[3]),figsize=(9,9))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
pts_acc = dict()
for sigma in tqdm(sigmas):
    _,VVg_capped_, _ = get_VVg(img, sigma)
    sato  = sato_coll[sigma]
    #th = np.percentile(sato[~full_mask],95)
    cond = (~full_mask)
    #th = np.mean(sato[cond]) + 3*np.std(sato[cond])
    th=0
    pts0 = np.array(np.where(full_mask*(sato>=th))).T
    pts0 = pts0 + np.random.randn(*pts0.shape)*0.01
    
    endpoints0 = calc_trajectory_basic(VVg_capped_,
                                       pts0, 
                                       n_iter=501, 
                                       tol=0.001, 
                                       max_dist=3*sigma,
                                       with_plot=False,
                                       gamma=1.,
                                       knn=6,
                                       agg_alpha=agg_alpha,
                                       background=img,)
    pts1 = density_filter(endpoints0, radius=1, 
                          min_neighbors=3*sigma, with_hist_plot=False)
    #project_points(pts1, img=(sato_best_sigma==sigma_dict_back[sigma]),figsize=(6,6))
    project_points(pts1, img=img,figsize=(6,6))
    plt.title(f'$\\sigma$={sigma}')
    plt.contour(sato >= th, colors=['lime'])
    plt.tight_layout()
    pts_acc[sigma] = pts1    
```

```{code-cell} ipython3
#plt.imshow(sato)
```

```{code-cell} ipython3

plt.figure()

plt.hist(sato[sato>0], 100, histtype='step',label='all');
sato_ridge = np.array([sato[tuple(p)] for p in np.round(pts1).astype(int)])
plt.hist(sato_ridge, 100, histtype='step',label='centerlines');
plt.legend()
```

```{code-cell} ipython3
#plt.imshow(sato)
plt.imshow(img, cmap='gray')
plt.contour(sato,[0.2,0.4], colors=['b','r'])
#plt.contour((sato_best_sigma==7), colors='g')
```

```{code-cell} ipython3
plt.imshow(sato_best_sigma==7)
```

```{code-cell} ipython3
plt.imshow((sato > np.percentile(sato[sato>0],75))*(vvg_coll[8] < np.percentile(vvg_coll[8][sato>0],25)))
plt.contour(full_mask, colors=['lime'])
```

```{code-cell} ipython3
def test_mask(sigma):
    sx = sato_coll[sigma]
    vvg = vvg_coll[sigma]
    mask = (sx > np.percentile(sx[sx>0],75))\
           *(vvg < np.percentile(vvg[sx>0],25))
    return mask
```

```{code-cell} ipython3
# mask6 =\
# (sx:= sato_coll[6],
#  vvg:=vvg_coll[6],
#  mask:=
 
```

```{code-cell} ipython3
plt.imshow(test_mask(1.5))
plt.contour(full_mask, colors=['lime'])
```

```{code-cell} ipython3
sigmas
```

```{code-cell} ipython3
masks = [test_mask(sigma) for sigma in sigmas]
```

```{code-cell} ipython3
plt.figure(figsize=(8,8))

plt.imshow(masks[-1], cmap='Spectral_r')
plt.plot(pts_acc[8][:,1],pts_acc[8][:,0],'.',color='g',ms=1,alpha=0.2)
plt.contour(full_mask, colors=['k'])
```

```{code-cell} ipython3
def count_points(pts, img_shape,mask=None):
    counts = np.zeros(img_shape)    
    if mask is None:
        mask = np.ones(img_shape, bool)
    pts_coarse = np.round(pts).astype(int)
    for p in map(tuple, pts_coarse):
        if mask[p]:
            counts[p] +=1
    return counts
```

```{code-cell} ipython3
counts = sum(count_points(pts, img.shape) for pts in pts_acc.values())
```

```{code-cell} ipython3
plt.imshow(np.log10(1 + counts), 
           cmap='plasma'); 
plt.colorbar()
plt.contour(full_mask, colors=['k'])
```

```{code-cell} ipython3
plt.imshow(percentile_rescale(img) + np.log10(1+counts))
```

```{code-cell} ipython3
log_counts = np.log10(1+counts/len(sigmas))
```

```{code-cell} ipython3
plt.imshow(log_counts>0)
```

```{code-cell} ipython3
speed = percentile_rescale(img) + log_counts
plt.imshow(speed,cmap='plasma')
```

```{code-cell} ipython3
plt.imshow(img,cmap='gray')
plt.imshow(np.ma.masked_where(~full_mask, speed), alpha=0.5, cmap='plasma')
```

```{code-cell} ipython3
targets = np.array(np.where(log_counts <=0)).T
len(targets)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
tt_new = skfmm.travel_time(1.-soma_mask, speed)
plt.imshow(tt_new,cmap='Spectral')
```

```{code-cell} ipython3
paths = [stupid_2d_gd(tt_new, loc, nsteps=10000, max_drop=3000) 
           for loc in locs]
len(paths[0])
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(uc.clip_outliers(img),cmap='gray')
plt.contour(full_mask, levels=[0.5], colors='b')

for p in paths:
    plt.plot(p[:,1], p[:,0],'m',lw=1,alpha=0.5)
```

```{code-cell} ipython3
sparse_targets = np.random.permutation(targets)

paths = [stupid_2d_gd(tt_new, tuple(loc), nsteps=1000, max_drop=1000) 
         for loc in tqdm(sparse_targets) if tt_new[tuple(loc)]]
paths = [p for p in paths if len(p) > 50 and soma_mask[tuple(p[-1])]]
```

```{code-cell} ipython3
len(paths)
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(uc.clip_outliers(img)**0.5,cmap='gray')
for path in paths:
    color = np.random.rand(3)
    if np.random.rand() < 0.01:
        plt.plot(path[:,1], path[:,0],lw=1,alpha=0.1,color=color)

#plt.axis([100,150, 150,100])
plt.title(f'Log centerline counts')
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
plt.figure(figsize=(8,8))
masks = [test_mask(sigma) for sigma in sigmas]
plt.imshow(255*np.dstack(masks[-3:] + [np.sum(masks[-3:],0)]))
plt.contour(full_mask, colors=['k'])
```

```{code-cell} ipython3
plt.figure(figsize=(8,8))
masks = [test_mask(sigma) for sigma in sigmas]
plt.imshow(255*np.dstack(masks[:3] + [np.sum(masks[:3],0)]))
plt.contour(full_mask, colors=['k'])
```

```{code-cell} ipython3

```

```{code-cell} ipython3
indicator=sato_ridge>0.6
```

```{code-cell} ipython3
indicator = np.array([sato_best_sigma[tuple(p)]==sigma_dict_back[8] for p in np.round(pts1).astype(int)])
```

```{code-cell} ipython3
#indicator.shape
```

```{code-cell} ipython3
project_points(pts1[~indicator], pts1[indicator], img=img,figsize=(6,6))
```

```{code-cell} ipython3
#plt.imshow(uc.masks.select_overlapping(sato_best_sigma==sigma_dict_back[8], soma_mask))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
#sigmas
```

```{code-cell} ipython3
#get_multi_vecs_interp(pts1,)
```

```{code-cell} ipython3
project_points(pts_acc[8], img=(sato_best_sigma==sigma_dict_back[8]),figsize=(6,6))
```

```{code-cell} ipython3

```

----

```{code-cell} ipython3
pts0
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
%matplotlib qt
```

```{code-cell} ipython3
from semlabca import visuals 
```

```{code-cell} ipython3
out = astro.morpho.peaks_in_unfoldings(ndi.gaussian_filter(img, 9), np.ones(img.shape))
```

```{code-cell} ipython3
plt.figure(figsize=(5,5))
plt.imshow(img**0.5, cmap='gray')
plt.imshow(np.ma.masked_less_equal(out,0), interpolation='nearest', cmap='Reds');
plt.tight_layout()
plt.gcf()
```

```{code-cell} ipython3
plt.figure()
ky = 92
plt.imshow(img, cmap='gray')
plt.axhline(ky, color='m')
#plt.gcf()
plt.grid(True)
```

```{code-cell} ipython3
img.shape
```

```{code-cell} ipython3
#sl = slice(ky-2,ky+3)
sl = slice(ky-1,ky+1)
y = img[sl].mean(0)
yg = VVg_mag2[sl].mean(0)
ys = sato[sl].mean(0)

plt.figure(figsize=(12,5))
plt.plot(y)

twax = plt.twinx(plt.gca())
#twax.plot(yg,color='m')
twax.plot(ys,color='m')
#plt.xlim(150,300)


# plt.gcf()
```

```{code-cell} ipython3
y.shape
```

```{code-cell} ipython3
plt.close('all')
```

```{code-cell} ipython3
sigmas = [1, 1.5, 2, 3, 4, 6, 8]
#sigmas = [1]
```

```{code-cell} ipython3
sato_coll ={sigma:astro.morpho.sato2d(img, sigma) for sigma in sigmas}

vvg_coll = {sigma:get_VVg(img, sigma)[1] for sigma in sigmas}
```

```{code-cell} ipython3

```

```{code-cell} ipython3
sato_best_sigma = np.argmax([v*sigma**2 for sigma,v in sato_coll.items()],0) + 1
sato_best_sigma[~full_mask] = 0
sato_best_sigma = np.ma.masked_where(~full_mask, sato_best_sigma)
```

```{code-cell} ipython3
plt.imshow(uc.clip_outliers(img), cmap='gray')

plt.imshow(sato_best_sigma==7, cmap=plt.cm.Spectral_r, alpha=0.5)
```

```{code-cell} ipython3
project_points(endpoints0,img=sato_best_sigma==7)
```

```{code-cell} ipython3
ui.group_maps([sato_best_sigma==i+1 for i in range(len(sigmas))], titles=sigmas, 
              figscale=5, colorbar=False)
plt.tight_layout()
```

```{code-cell} ipython3
plt.figure(figsize=(8,8))
plt.imshow(img, cmap='gray')
plt.imshow(sato_best_sigma == 1, cmap='jet', alpha=0.5)
plt.tight_layout()
```

```{code-cell} ipython3

```

```{code-cell} ipython3
sato_merge = np.max([v*sigma**2 for sigma,v in sato_coll.items()],0)
VVg_merge1 = np.sum([v for v in vvg_coll.values()],0)
```

```{code-cell} ipython3
speed_merge = sato_merge / (0.01 + VVg_merge1)
```

```{code-cell} ipython3
plt.imshow(uc.clip_outliers(img), cmap='gray')
```

```{code-cell} ipython3
plt.imshow(uc.clip_outliers(speed_merge*full_mask, phigh=95))
```

```{code-cell} ipython3
sato_projs = {sigma:sato[sl].mean(0)*sigma**2 for sigma,sato in sato_coll.items()}
vvgmag_projs = {sigma:vvgmag[sl].mean(0) for sigma,vvgmag in vvg_coll.items()}
```

```{code-cell} ipython3
plt.figure(figsize=(12,6))
plt.plot(y, color='gray')
twax = plt.twinx(plt.gca())

#twax.plot(speed_merge[sl].mean(0))
twax.plot((sato_best_sigma==1)[sl].mean(0))

#plt.xlim(150,300)
plt.xlim(90,150)
twax.legend()
```

```{code-cell} ipython3
plt.figure(figsize=(12,6))
plt.plot(y, color='gray')
twax = plt.twinx(plt.gca())

for ys,sigma in zip(sato_projs.values(), sigmas):
    twax.plot(ys, label=sigma)



#plt.xlim(150,300)
plt.xlim(90,150)
twax.legend()


#plt.gcf()
```

```{code-cell} ipython3
sl
```

```{code-cell} ipython3
plt.figure(figsize=(12,6))
ax = plt.gca()
twax = plt.twinx(ax)

for ys,sigma in zip(sato_projs.values(), sigmas):
    ax.plot(ys, label=sigma)

twax.plot((sato_best_sigma==6)[sl].mean(0),color='c',lw=2)

twax.set_ylim(0,1)
#plt.xlim(150,300)
plt.xlim(90,150)
ax.legend()
```

```{code-cell} ipython3
plt.figure(figsize=(12,6))
plt.plot(y, color='k')
twax = plt.twinx(plt.gca())

for ys in vvgmag_projs.values():
    twax.plot(ys,)

#plt.xlim(150,300)
plt.xlim(90,150)

twax.axhline(0, color='gray', lw=0.5,ls='--')

#plt.gcf()
```

```{code-cell} ipython3
plt.figure(figsize=(12,6))
plt.plot(y, color='k')
twax = plt.twinx(plt.gca())

#twax.plot(sato_projs[16]*10, color='deepskyblue')
twax.plot(vvgmag_projs[16], color='deepskyblue',label=16)
twax.plot(vvgmag_projs[2], color='m',label=2)
#plt.xlim(150,300)

twax.axhline(0, color='gray', lw=0.5,ls='--')



#plt.gcf()
```

```{code-cell} ipython3
plt.figure(figsize=(12,6))
plt.plot(y, color='k')
twax = plt.twinx(plt.gca())

#twax.plot(sato_projs[16]*10, color='deepskyblue')
twax.plot(sato_projs[16], color='deepskyblue',label=16)
twax.plot(sato_projs[2], color='m',label=2)

twax.legend()

#plt.xlim(150,300)




#plt.gcf()
```

```{code-cell} ipython3
plt.close('all')
```

```{code-cell} ipython3

```
