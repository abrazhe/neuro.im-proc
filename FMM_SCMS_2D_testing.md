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
img = tifffile.imread('astro_3wk-both1-grn-raw.pic-maxproj.tif')
#img = tifffile.imread('astro_4wk-ly24-raw.pic-ds_1-maxproj.tif')
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
smooth = ndi.gaussian_filter(img, 12)
plt.imshow(smooth)
```

```{code-cell} ipython3
#%matplotlib qt
```

```{code-cell} ipython3
#from imfun import ui, fseq
#ui.Picker(fseq.from_array(np.array([img]*10))).start()
```

```{code-cell} ipython3
%matplotlib inline
```

#macwe = morphsnakes.MorphACWE(smooth,lambda2=0.1,levelset= smooth >= 0.9*np.max(smooth))
#macwe.run(500)

```{code-cell} ipython3
#plt.imshow(macwe.levelset)
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

```

```{code-cell} ipython3
tol = (smooth.max() - smooth[img>th/2].min())/25
soma_mask = flood(smooth, center, tolerance=tol)
#soma_mask = morpho.expand_mask(soma_mask, smooth, iterations = 50)
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
sigma0 = 4
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

def economic_gd(field, p0, nsteps=10000, max_drop=100000, visited=None):
    if visited is None:
        visited=set()
    p0 = tuple(map(int, p0))
    traj = [p0]
    visited.add(p0)
    
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
        if p in visited:
            break
        visited.add(p)
    return traj
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
locs = [(51,152),(79,26)] # for 3wk-both1-grn-raw.pic
#locs = [(180,110), (800,500)]

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
                         sato0*np.exp(-5*VVg_mag2))], 
              figscale = 5,
              colorbar=False)
```

```{code-cell} ipython3
plt.imshow(uc.clip_outliers(sato0/(0.05 + percentile_rescale(VVg_mag2))), cmap='plasma')
```

```{code-cell} ipython3
plt.imshow(sato0)
plt.contour(full_mask,colors=['r'])
```

```{code-cell} ipython3
##plt.imshow(sato>0)
```

```{code-cell} ipython3
plt.hist(VVg_mag2[sato0>0].ravel(),50);
```

```{code-cell} ipython3
#plt.imshow(uc.utils.rescale(VVg_mag2))
```

```{code-cell} ipython3
#speed0= full_mask*(sato0/(0.1 + VVg_mag2))
speed0= full_mask*(sato0/(0.05 + uc.utils.rescale(VVg_mag2)))


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

Note that it's not so bad already

```{code-cell} ipython3
from numba import jit
```

## Now let's try to combine several scales

```{code-cell} ipython3
reload(astro.morpho)
astro.morpho.eigh = np.linalg.eigh
```

```{code-cell} ipython3
#sigmas = [1, 1.5, 2, 3, 4, 6, 8, 12]
sigmas = 2**np.arange(0,3.1,0.25)

id2sigma = {i+1:sigma for i, sigma in enumerate(sigmas)} # shift by one, so that zero doesn't correspond to a cell
sigma2id = {sigma:i+1 for i, sigma in enumerate(sigmas)}
```

```{code-cell} ipython3
#(0.05 + uc.utils.rescale(VVg_mag2)
```

```{code-cell} ipython3
sato_coll ={sigma:astro.morpho.sato2d(img, sigma)*sigma**2 for sigma in tqdm(sigmas)}
vvg_coll = {sigma:get_VVg(img, sigma)[2] for sigma in tqdm(sigmas)}

```

```{code-cell} ipython3
jerman_coll = {sigma:astro.morpho.jerman2d(img, sigma, tau=0.5) for sigma in tqdm(sigmas)}
```

```{code-cell} ipython3
plt.imshow(jerman_coll[sigmas[-2]])
```

```{code-cell} ipython3
masks = {}
for sigma in tqdm(sigmas):
    sato = sato_coll[sigma]
    # multiplication by square root of sigma here is pure heuristics
    threshold = threshold_li(sato[sato>0])#*sigma**0.5
    #print(sigma, threshold, np.sum(sato > threshold), sigma*32)
    masks[sigma] = remove_small_objects(full_mask*(sato > threshold), min_size=int(sigma**2))
masks[sigmas[-1]] = uc.masks.select_overlapping(masks[sigmas[-1]], soma_mask)
```

```{code-cell} ipython3
masksj = {}
for sigma in tqdm(sigmas):
    jerman = jerman_coll[sigma]
    # multiplication by square root of sigma here is pure heuristics
    threshold = threshold_li(jerman[jerman>0])#*sigma**0.5
    #print(sigma, threshold, np.sum(sato > threshold), sigma*32)
    masksj[sigma] = remove_small_objects(full_mask*(jerman > threshold), min_size=int(sigma**2))
masksj[sigmas[-1]] = uc.masks.select_overlapping(masksj[sigmas[-1]], soma_mask)
```

```{code-cell} ipython3
#plt.imshow(masksj[sigmas[-1]])
```

```{code-cell} ipython3
#plt.imshow(masks[sigmas[-1]])
```

```{code-cell} ipython3
speed_coll = {sigma:masks[sigma]*sato_coll[sigma]/(0.05 + uc.utils.rescale(vvg_coll[sigma])) 
              for sigma in sigmas}
```

```{code-cell} ipython3
speed_collj = {sigma:full_mask*jerman_coll[sigma]/(0.01 + uc.utils.rescale(vvg_coll[sigma])) 
              for sigma in sigmas}
```

```{code-cell} ipython3
dynamic_ranges = {sigma:(np.max(v)-np.min([v>0])) for sigma,v in sato_coll.items()}
```

```{code-cell} ipython3
plt.plot(sigmas, [np.max(v) for v in sato_coll.values()], 'sk-')
plt.plot(sigmas, dynamic_ranges.values(), 'o--')
plt.plot(sigmas, [np.percentile(v[v>0],99) for v in sato_coll.values()],'o-' )
```

```{code-cell} ipython3
sigma_cutoff = id2sigma[np.argmax(list(dynamic_ranges.values()))+2]
sigma_cutoff
```

```{code-cell} ipython3

```

```{code-cell} ipython3
ui.group_maps([uc.clip_outliers(sp) 
               for sigma,sp in sato_coll.items()], 
              titles=[f"σ={sigma:1.1f}" for sigma in sigmas],
              imkw=dict(cmap='plasma'),
              colorbar=False, figscale=3)
```

```{code-cell} ipython3
ui.group_maps([sp
               for sigma,sp in jerman_coll.items()], 
              titles=[f"σ={sigma:1.1f}" for sigma in sigmas],
              imkw=dict(cmap='plasma'),
              colorbar=False, figscale=3)
```

```{code-cell} ipython3
ui.group_maps([uc.clip_outliers(sp) for sp in speed_coll.values()], samerange=True, figscale=3)
```

```{code-cell} ipython3
ui.group_maps([uc.clip_outliers(sp) for sp in speed_collj.values()], samerange=True, figscale=3)
```

```{code-cell} ipython3
speed_multiscale = sum((ndi.gaussian_filter(v,sigmas[0]) for v in speed_coll.values()))
speed_multiscalej = sum((ndi.gaussian_filter(v,sigmas[0]) for v in speed_collj.values()))
```

```{code-cell} ipython3
plt.figure(figsize=(8,8))
plt.imshow(speed_multiscale,cmap='plasma')
plt.tight_layout()
```

```{code-cell} ipython3
plt.figure(figsize=(8,8))
plt.imshow(speed_multiscalej,cmap='plasma')
plt.tight_layout()
```

```{code-cell} ipython3
# plt.figure(figsize=(8,8))
# plt.imshow(speed_multiscalej,cmap='plasma')
# plt.tight_layout()
```

```{code-cell} ipython3
#tt_ms = skfmm.travel_time(1.-soma_mask, speed_multiscale)
tt_ms = skfmm.travel_time(1.-soma_mask, speed_multiscalej)
plt.imshow(tt_ms,cmap='Spectral',vmin=0,vmax=np.percentile(tt_ms[~tt_ms.mask], 95))
```

```{code-cell} ipython3
plt.imshow(sum(masks.values())>0)
```

```{code-cell} ipython3
def follow_to_root(g, tip, max_nodes=1000000):
    visited = {tip}
    acc = [tip]
    for i in range(max_nodes):
        parents = list(g.predecessors(tip))
        parents = [p for p in parents if not p in visited]
        if not len(parents):
            break
        tip = parents[0]
        visited.add(tip)
        acc.append(tip)
    if i >= max_nodes-1:
        print('limit reached')
    return acc
```

```{code-cell} ipython3
#targets = np.array(np.where(sum(masks.values())==0)).T
#targets = np.array(np.where(masks[sigmas[0]])).T
targets = np.array(np.where(full_mask)).T
```

```{code-cell} ipython3
visited=set()

paths_eco = [economic_gd(tt_ms, loc, nsteps=10000, max_drop=30000, visited=visited) 
             for loc in tqdm(np.random.permutation(targets)) if tt_ms[tuple(loc)]]
```

```{code-cell} ipython3
Gtt = nx.DiGraph()
for path in tqdm(paths_eco):
    Gtt.add_edges_from(list(itt.pairwise(map(tuple, path[::-1]))))
```

```{code-cell} ipython3
#paths_x = vis.graph_to_paths(Gtt)
```

```{code-cell} ipython3
# tips = gu.get_tips(Gtt)
# plot_paths = [np.array(follow_to_root(Gtt, t)) for t in tqdm(tips)]
```

```{code-cell} ipython3
#plt.imshow(speed_multiscale>threshold_li(speed_multiscale[speed_multiscale>0]))
```

```{code-cell} ipython3
def count_occurences(G, shape):
    counts =  np.zeros(shape)
    for tip in tqdm(gu.get_tips(G)):
        for p in follow_to_root(G,tip):
            n = G.nodes[p]
            if 'count' in n:
               n['count'] += 1
            else:
               n['count'] = 1
            counts[p] += 1
    return counts
```

```{code-cell} ipython3
counts = count_occurences(Gtt, full_mask.shape)
```

```{code-cell} ipython3

#Gtt.nodes[pkeys[1]]
```

```{code-cell} ipython3
np.max(counts)
```

```{code-cell} ipython3
logcounts = np.log10(1 + counts)
```

```{code-cell} ipython3
#np.log10(5)
```

```{code-cell} ipython3
plt.figure(figsize=(8,8))
plt.imshow(img, cmap='gray')
plt.imshow(np.ma.masked_less_equal(logcounts,np.log10(5)),interpolation='nearest',cmap='Reds')
plt.tight_layout()
```

```{code-cell} ipython3
def filter_fn_(G, n, min_occ=1):
    ni = G.nodes[n]
    #is_high = ni['occurence'] > max(0, occ_threshs[ni['sigma_mask']])
    is_high = ni['count'] >= min_occ # very permissive, but some branches are valid and only occur once
    not_tip = len(list(G.successors(n)))
    return is_high and not_tip
```

```{code-cell} ipython3
# Gtt_filt = Gtt

# for i in tqdm(range(1)):
#     good_nodes = (node for node in Gtt_filt if filter_fn_(Gtt_filt, node, 2))
#     Gtt_filt = Gtt_filt.subgraph(good_nodes)
```

```{code-cell} ipython3
Gtt_filt = gu.filter_graph(Gtt, lambda n: n['count'] > 10)

# # prune tips
# for i in range(10):
#     Gtt_filt = gu.filter_graph(Gtt_filt, lambda n: len(list(Gtt_filt.successors(n))))


plot_paths2 = [np.array(follow_to_root(Gtt_filt, t)) for t in tqdm(gu.get_tips(Gtt_filt))]
plot_paths2 = [p for p in plot_paths2 if len(p)>100]
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(uc.clip_outliers(img)**0.5,cmap='gray')
for path in plot_paths2:
    color = np.random.rand(3)*0.75
    if np.random.rand() < 1:
        plt.plot(path[:,1], path[:,0],lw=1,alpha=0.75,color=color)

#plt.axis([100,150, 150,100])
plt.title(f'sum of speeds')
#plt.axis([300,500,600,400])
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(-uc.clip_outliers(speed_multiscalej)**0.5,cmap='gray')
for path in plot_paths2:
    color = np.random.rand(3)*0.75
    if np.random.rand() < 1:
        plt.plot(path[:,1], path[:,0],lw=1,alpha=0.75,color=color)
```

---

```{code-cell} ipython3

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

```

```{code-cell} ipython3
ui.group_maps(list(masks.values()), colorbar=False)
```

```{code-cell} ipython3
ui.group_maps(list(masksj.values()), colorbar=False)
```

```{code-cell} ipython3
sato_best_sigma = np.argmax([v*masks[sigma] for sigma,v in sato_coll.items()],0) + 1
sato_best_sigma[~full_mask] = 0
sato_best_sigma = np.ma.masked_where(~full_mask, sato_best_sigma)
```

```{code-cell} ipython3
ui.group_maps([sato_best_sigma==i+1 for i in range(len(sigmas))],
              5,
              titles=sigmas, 
              figscale=5, colorbar=False)
plt.tight_layout()
```

```{code-cell} ipython3
jerman_best_sigma = np.argmax([v*masks[sigma] for sigma,v in jerman_coll.items()],0) + 1
jerman_best_sigma[~full_mask] = 0
jerman_best_sigma = np.ma.masked_where(~full_mask, jerman_best_sigma)
```

```{code-cell} ipython3
ui.group_maps([jerman_best_sigma==i+1 for i in range(len(sigmas))], 
              5,
              titles=sigmas, 
              figscale=5, colorbar=False)
plt.tight_layout()
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
# mask_sum = np.zeros(img.shape,bool)
# masks_exclusive = {}

# for k in range(len(sigmas)-1,-1,-1):
# # for k in range(len(sigmas)):
#     sigma = sigmas[k]
#     mask = masks[sigma]
#     if k < len(sigmas)-1:
#         mask = mask & (mask ^ mask_sum)
#     mask_sum += mask.astype(bool)
#     masks_exclusive[sigma] = mask
    
```

```{code-cell} ipython3
#ui.group_maps(list(masks_exclusive.values()), colorbar=False)
```

```{code-cell} ipython3
sigmas
```

```{code-cell} ipython3
plt.figure(figsize=(6,6))
sigma_i = 8
vvg_i = vvg_coll[sigma_i]
sato_i = sato_coll[sigma_i]
img_s = ndi.gaussian_filter(img, sigma_i)

th0 = np.percentile(sato_i[(~full_mask)*(sato_i>0)],95)
th1 = threshold_li(sato_i[sato_i>0])
print('Th0: ', th0, 'Th Li:', th1)
plt.imshow(img**0.5, cmap='gray')
#plt.contour(sato_i, levels=sorted([th0,  250]), colors=['m', 'r'])
plt.contour(sato_i, levels=sorted([th1, 250]), colors=['m', 'r'])


mask_centerline = (sato_i > np.percentile(sato_i[sato_i > th1], 50))\
                   *(vvg_i < np.percentile(vvg_i[sato_i>th1],25))
mask_centerline = skeletonize(mask_centerline)
mask_skel = skeletonize(sato_i > th1)

plt.imshow(np.dstack([mask_centerline*1.0, 
                      mask_skel, 
                      np.zeros_like(mask_skel),
                     (mask_centerline | mask_skel)]))

#th_sato
cond = full_mask*(sato_i>0)
#plt.hist(img[],50);
# plt.figure()
# plt.plot(img[cond], sato_i[cond], ',', alpha=0.1)


# plt.axhline(th0, color='darkgray', ls='--')
# plt.axhline(th1, color='gray', ls='--')
plt.xlabel('img'), plt.ylabel('sato')
```

```{code-cell} ipython3
plt.figure(figsize=(8,8))
fig,axs = plt.subplots(1,2,figsize=(12,6))
axs[0].imshow(img_s, cmap='gray')
axs[1].imshow(img, cmap='gray')
axs[1].imshow(np.dstack([mask_centerline, 
                         (mask_skel | mask_centerline)^(sato_i > th1), 
                         mask_skel, 
                         0.5*(sato_i > th1)]))
```

```{code-cell} ipython3
plt.imshow(sato_i, cmap='plasma')
```

```{code-cell} ipython3
plt.imshow(vvg_i)
```

```{code-cell} ipython3
visited=set()
pts = np.array(np.where(sato_i>th1)).T
pts_sparse = np.random.permutation(pts)[:1000]
paths_gvv = [economic_gd(vvg_i, p, visited=visited) for p in tqdm(pts)]

Gtt = nx.DiGraph()
for path in tqdm(paths_gvv):
    Gtt.add_edges_from(list(itt.pairwise(map(tuple, path[::-1]))))
paths_x = vis.graph_to_paths(Gtt)
len(paths_x)
```

```{code-cell} ipython3
endpoints = np.array([p for p in paths_x.keys() if sato_i[tuple(p)]>th1])
```

```{code-cell} ipython3
len(endpoints)
```

```{code-cell} ipython3
mask_centerline2 = np.zeros_like(full_mask)
for p in endpoints:
    mask_centerline2[tuple(p)] = True
```

```{code-cell} ipython3

plt.figure()
plt.imshow(img, cmap='gray')
plt.plot(endpoints[:,1],endpoints[:,0],'r,')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
th_x = threshold_li(img[full_mask*(mask_centerline2)])
th_x
```

```{code-cell} ipython3
plt.imshow((full_mask)*mask_centerline2)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
plt.imshow((full_mask^soma_mask)*mask_skel)
```

```{code-cell} ipython3
plt.figure()

mask_opt = sato_best_sigma  >= sigma2id[sigma_i]

plt.hist(img[full_mask*mask_centerline], 50, density=True, histtype='step');
plt.hist(img[(full_mask^soma_mask)*(sato_i < th1)], 50, density=True, histtype='step');
#plt.hist(img[(full_mask^soma_mask)*mask_opt], 50, density=True, histtype='step');

#th_y = np.percentile(img[full_mask*(sato_i<th1)],95)
#plt.axvline(th_y, color='m')
```

```{code-cell} ipython3
plt.hexbin(np.ravel(img[full_mask]),np.ravel(sato_i[full_mask]),bins='log')
```

```{code-cell} ipython3
#th_x, th_y
```

```{code-cell} ipython3
th_z = 0.9*np.mean(img[soma_mask])
th_z
```

```{code-cell} ipython3
# plt.figure(figsize=(8,8))
# plt.imshow((img > th_y)*(sato_i  > th1))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
plt.figure(figsize=(8,8))
plt.imshow(img, cmap='gray')

mask_x = (mask_opt*(img > th_x)*(sato_i > th1))#, sigma_i**2)
#mask_x = (mask_opt*(sato_i > th1))#, sigma_i**2)
#mask_x = (mask_opt*(img > th_y)*(sato_i > th1))
plt.contour(mask_x, colors=['lime'])
plt.contour(mask_centerline2, colors='r')
plt.tight_layout()
```

```{code-cell} ipython3
dd = skfmm.travel_time(1.0 - (mask_centerline2), mask_x)
```

```{code-cell} ipython3
#mask_travel = uc.masks.threshold_object_size(dd < sigma_i, sigma_i**2)
mask_travel = np.array((dd < 1.5*sigma_i))*(~dd.mask)
#mask_travel = ndi.binary_opening(ndi.binary_closing(mask_travel))
mask_travel = uc.masks.threshold_object_size(mask_travel, (sigma_i)**2)

plt.figure(figsize=(8,8))
plt.imshow(img, cmap='gray')
plt.contour(mask_travel, colors='r')
plt.contour(soma_mask, colors='g')
```

```{code-cell} ipython3
from skimage.filters import threshold_otsu
```

```{code-cell} ipython3
#img[:10,:10]
```

```{code-cell} ipython3
def iterative_exclusive_masks(img, sigmas):
    mask_sum = np.zeros(img.shape, bool)
    masks_exclusive = {}
    bg = np.percentile(img[(~full_mask)],99)
    print('Bg:', bg)
    for σ in sorted(sigmas, reverse=True):
        vvg_i = vvg_coll[σ]
        sato_i = sato_coll[σ]
        th_lis = threshold_li(sato_i[sato_i>0])
        #th_ois = threshold_triangle(sato_i[sato_i>0])
        
        #mask_centerline = (sato_i > np.percentile(sato_i[sato_i > th_lis], 75))\
        #                  *(vvg_i < np.percentile(vvg_i[sato_i>th_lis],25))
        mask_skel = skeletonize(sato_i > th_lis)

        
        cond_img = (img > bg)*(full_mask ^ mask_sum) & mask_skel
        #cond_img = (img > bg) & (full_mask) & mask_skel
        cond_img_neg = (img > bg) & (full_mask ^ mask_sum) & (sato_i < th_lis)
        #th_lii = threshold_minimum(img[cond_img])
        th_lii = np.percentile(img[cond_img_neg], 95)
        #th_lii = uc.utils.estimate_mode(img[cond_img], top_cut=95, kind='max')
        print(th_lis, th_lii)
        
        fig, axs = plt.subplots(1,4,sharex='col', figsize=(12,4))
        #axs[0].imshow(cond_img)
        axs[0].hist(img[cond_img],50,density=True,histtype='step');
        axs[0].hist(img[cond_img_neg],50,density=True,histtype='step');
        axs[0].axvline(th_lii, color='m', ls='--')
        #axs[0].axvline(np.percentile(img[cond_img],95), color='gray', ls='--')
        axs[1].imshow(cond_img*1.0+(sato_i > th_lis), 
                      interpolation='nearest')
        
        # don't know how to have this built iteratively...
        mask_opt = sato_best_sigma  >= sigma2id[σ]
        #mask_x = mask_opt*(img > th_lii)*(sato_i > th_lis)
        mask_x = mask_opt*(sato_i > th_lis)
        #mask_x = (img > th_lii)*(sato_i > th_lis)


        
        mask_final = mask_x
        ridge_distance = skfmm.travel_time(1.0 - (mask_skel), mask_x)
        mask_final = np.array((ridge_distance < σ))*(~ridge_distance.mask)
        #overlap_target = soma
        mask_final = uc.masks.select_overlapping(mask_final, mask_sum | soma_mask)
        #mask_final = ndi.binary_opening(ndi.binary_closing(mask_final))
        #mask_final = uc.masks.threshold_object_size(mask_final, (σ)**2)

        
        
        if σ < np.max(sigmas):
            mask_final = mask_final & (mask_final ^ mask_sum)

        axs[2].imshow(mask_x, interpolation='nearest')
        mask_sum += mask_final.astype(bool)
        axs[3].imshow(mask_sum, interpolation='nearest')
        masks_exclusive[σ] = mask_final
    return masks_exclusive
```

```{code-cell} ipython3
masks_exclusive2 = iterative_exclusive_masks(img, sigmas)
```

```{code-cell} ipython3
plt.imshow(full_mask*(img < 50))
```

```{code-cell} ipython3
#plt.imshow(img > threshold_li[])
```

```{code-cell} ipython3
ui.group_maps(list(masks_exclusive2.values())[::-1], colorbar=False)
```

```{code-cell} ipython3
plt.imshow(sum(m for m in masks_exclusive2.values()))
```

```{code-cell} ipython3
#plt.imshow(full_mask ^ soma_mask)
```

```{code-cell} ipython3
cond2 = full_mask*(sato_i/vvg_i > 1000)*(sato_best_sigma==sigma2id[sigma_i])
```

```{code-cell} ipython3
plt.imshow(cond2)
```

```{code-cell} ipython3
plt.hist(img[cond2], 50);
```

```{code-cell} ipython3
plt.figure()
plt.plot(np.exp(-vvg_i[cond]), sato_i[cond], ',', alpha=0.15)

plt.axhline(th0, color='darkgray', ls='--')
plt.axhline(th1, color='gray', ls='--')
plt.xlabel('vvg'), plt.ylabel('sato')
```

```{code-cell} ipython3
bg = np.mean(img[~full_mask])
bgplt.figure()
plt.plot(img[cond], sato_i[cond], ',', alpha=0.1)

plt.axhline(th0, color='darkgray', ls='--')
plt.axhline(th1, color='gray', ls='--')
plt.xlabel('img'), plt.ylabel('sato')
```

```{code-cell} ipython3
plt.hist(img[(img>bg)*full_mask*(sato_i > 0)*(sato_i < th1)], 100, histtype='step');
plt.hist(img[(img>bg)*full_mask*(sato_i > th1)], 100, histtype='step');
```

```{code-cell} ipython3

```

```{code-cell} ipython3
plt.figure()
plt.imshow(img**0.5, cmap='gray')
mask_i = full_mask*(sato_i > th1)*(sato_best_sigma==sigma2id[sigma_i])
#mask_i = uc.masks.threshold_object_size(mask_i * (sato_best_sigma==sigma2id[sigma_i]), sigma_i*4)
plt.contour(mask_i, colors='r')
```

```{code-cell} ipython3
plt.plot()
```

```{code-cell} ipython3
plt.figure()

plt.hist(sato[sato>0], 100, histtype='step',label='all');
sato_ridge = np.array([sato[tuple(p)] for p in np.round(pts1).astype(int)])
plt.hist(sato_ridge, 100, histtype='step',label='centerlines');
plt.legend()
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
speed_ms2 = speed_multiscale
```

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
sigma0
```

```{code-cell} ipython3
pts = np.array(np.where(full_mask*(sato0>0))).T
pts_all = np.array(np.where(full_mask)).T
len(pts)
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
                                   n_iter=2000, 
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
project_points(pts1, img=(sato_best_sigma==sigma_dict_back[sigma0]),figsize=(9,9))
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
                                       n_iter=50001, 
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
import seaborn as sns
```

```{code-cell} ipython3
plt.figure()
```

```{code-cell} ipython3
plt.imshow(full_mask*np.exp(-VVg_mag2*10))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
plt.figure()
th0 = np.max(sato0[~full_mask])
plt.imshow(img**0.5, cmap='gray')
plt.contour(sato0, levels=sorted([th0, threshold_li(sato0[sato0>0]), 250]), colors=['g', 'm', 'r'])
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
sigmas
```

```{code-cell} ipython3
counts = sum(count_points(pts, img.shape) for sigma,pts in pts_acc.items() if sigma >= 8)
#counts = sum(ndi.gaussian_filter(count_points(pts, img.shape),sigma/4) for sigma,pts in pts_acc.items())
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
#plt.axis([300,500,600,400])
```

```{code-cell} ipython3
plt.imshow(img,cmap='gray')
plt.imshow(np.ma.masked_where(~full_mask, speed), alpha=0.5, cmap='plasma')
#plt.axis([300,500,600,400])
```

```{code-cell} ipython3
targets = np.array(np.where(log_counts <=0)).T
len(targets)
```

```{code-cell} ipython3
np.e**2
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
plt.axis([300,500,600,400])
```

```{code-cell} ipython3

```

```{code-cell} ipython3
visited=set()

paths_eco = [economic_gd(tt_new, loc, nsteps=10000, max_drop=30000, visited=visited) 
             for loc in tqdm(np.random.permutation(sparse_targets)) if tt_new[tuple(loc)]]
```

```{code-cell} ipython3
len(set((tuple(p) for p in sparse_targets if tt_new[tuple(p)])))
```

```{code-cell} ipython3
len(paths_eco)
```

```{code-cell} ipython3
len(visited), len(sparse_targets), len(paths)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
Gtt = nx.DiGraph()
for path in tqdm(paths_eco):
    Gtt.add_edges_from(list(itt.pairwise(map(tuple, path[::-1]))))
```

```{code-cell} ipython3
#%time Gtt = Gtt.reverse()
```

```{code-cell} ipython3
paths_x = vis.graph_to_paths(Gtt)
```

```{code-cell} ipython3
len(paths_x)
```

```{code-cell} ipython3
def follow_to_root(g, tip, max_nodes=1000000):
    visited = {tip}
    acc = [tip]
    for i in range(max_nodes):
        parents = list(g.predecessors(tip))
        parents = [p for p in parents if not p in visited]
        if not len(parents):
            break
        tip = parents[0]
        visited.add(tip)
        acc.append(tip)
    if i >= max_nodes-1:
        print('limit reached')
    return acc
```

```{code-cell} ipython3
tips = gu.get_tips(Gtt)
len(tips)
```

```{code-cell} ipython3
plot_paths = [np.array(follow_to_root(Gtt, t)) for t in tqdm(tips)]
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(uc.clip_outliers(img)**0.5,cmap='gray')
for path in plot_paths:
    color = np.random.rand(3)*0.75
    if np.random.rand() < 0.01:
        plt.plot(path[:,1], path[:,0],lw=1,alpha=0.5,color=color)

#plt.axis([100,150, 150,100])
plt.title(f'Log centerline counts')
plt.axis([300,500,600,400])
```

```{code-cell} ipython3

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
