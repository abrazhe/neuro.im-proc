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


# def driven_rw(p0, attractor, nsteps=10000):
#     #todo: add persistence
#     av = attractor
#     traj = p0
#     for i in range(nsteps):
#         p = tuple(traj[-1])
#         new_direction = np.random.rand
```

```{code-cell} ipython3
def follow_to_root_nx(g, tip, max_nodes=1000000):
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

def count_occurences_nx(G, shape):
    counts =  np.zeros(shape)
    for tip in tqdm(gu.get_tips(G)):
        for p in follow_to_root_nx(G,tip):
            n = G.nodes[p]
            if 'count' in n:
               n['count'] += 1
            else:
               n['count'] = 1
            counts[p] += 1
    return counts
```

## Intermezzo with RW

```{code-cell} ipython3
Npx = 512
X, Y = np.mgrid[:512,:512]
X.shape
```

```{code-cell} ipython3
field = np.zeros((Npx,Npx))
field[255,255] = 1000
#field = ndi.gaussian_filter(field, 128)
#field = - uc.utils.rescale(field)**2

#field = ndi.distance_transform_edt(field==0)/350

#field += ndi.gaussian_filter(np.random.randn(*field.shape),1)*0.01

plt.imshow(field)
plt.colorbar()
```

```{code-cell} ipython3
%timeit np.array(neighbours((10,10),(100,100))).astype(int)
```

```{code-cell} ipython3
def self_avoiding_2d_gd(field, p0, terminate_mask=None,  nsteps=100, travelled=None, pjitter=0.15):
    p0 = tuple(map(int, p0))
    if travelled is None:
        travelled = set()
    if terminate_mask is None:
        terminate_mask = np.zeros(field.shape, bool)
    traj = [p0]

    visited = set(p0)
    
    for i in range(nsteps):
        p = tuple(traj[-1])
        u = field[p]
        nns = (tuple(n) for n in np.array(neighbours(p, field.shape)).astype(int))
        nns = [n for n in nns if not n in traj]
        #print(nns)
        if not len(nns):
            break
        nn_fields = [field[n] for n in nns]
        ksort = np.argsort(nn_fields)
        if np.random.rand() < 1-pjitter:
            best = ksort[0]
        else:
            best = ksort[1 if len(nn_fields)>1 else 0]
        pnext = nns[best]        
        traj.append(pnext)
        if terminate_mask[p] or p in travelled or pnext==p:
            break
    travelled.update(set(traj))
    return np.array(traj)
```

```{code-cell} ipython3
%time neighbours((10,10), field.shape)
```

```{code-cell} ipython3
from imfun.bwmorph import neighbours_2
```

```{code-cell} ipython3
%time neighbours_2((10,10), field.shape)
```

```{code-cell} ipython3
#np.random.choice?
```

```{code-cell} ipython3
x = np.array([100,200,4,])
(1/x)/np.sum(1/x)
```

```{code-cell} ipython3
class PathNode:
    def __init__(self, loc, parent=None):
        self.v = np.array(loc)
        self.children = []
        self.parent = parent
        if parent is not None:
            parent.link(self)
    def link(self, child):
        child.parent = self
        if not child in self.children:
            self.children.append(child)
        

def merging_rw_gd(field, p0, terminate_mask=None,  nsteps=100, tree=None, pjitter=0.15):

    # tree is a hasmap, keys are locations, values are PathNodes
    if tree is None:
        tree = dict()
    
    if terminate_mask is None:
        terminate_mask = np.zeros(field.shape, bool)

    p0 = tuple(map(int, p0))
    path = [PathNode(p0)]
    traj = [p0]

    visited = set(p0)

    path_success = False
    
    for i in range(nsteps):
        prevnode = path[-1]
        p = tuple(prevnode.v)
        u = field[p]

        # look for nearest neighbors, not already visited 
        # when building the current path
        nns = (tuple(n) for n in np.array(neighbours_2(p, field.shape)).astype(int))
        nns = [n for n in nns if not n in visited]
        
        if not len(nns):
            break
        nns = np.random.permutation(nns)
        nn_fields = np.array([field[tuple(n)] for n in nns])
        ksort = np.argsort(nn_fields)
        linked_nns = [tuple(n) for n in nns if tuple(n) in tree]
        linked_fields = [field[n] for n in linked_nns]

        # preferential attachment:
        if len(linked_nns):
            best = np.argmin(linked_fields)
            pnext = linked_nns[best]
        else:
            # # choose second-best with some probability
            # if len(ksort)>1:
            #     first, second = ksort[:2]
            #     # ties?
            #     if nn_fields[first] == nn_fields[second]:
            #         best = first if np.random.rand() < 1 else second
            #     else:
            #         best = first if np.random.rand() < 1-pjitter else second
            # else:
            #     best = ksort[0]
            #best = np.random.choice(np.arange(len(nn_fields)),size=1,
            #                        p = (1/nn_fields)/np.sum(1/nn_fields))
            best = np.argmin(nn_fields)
            pnext = tuple(nns[best])
           
            if np.random.rand() < pjitter:
                # todo: approximately follow direction when choosing neighbor
                nns2 = (tuple(n) for n in 
                        np.array(neighbours(pnext, field.shape)).astype(int))
                nns = [pnext] + [n for n in nns2 if n in nns]
                pnext = nns[np.random.randint(len(nns))]
        
        # only create new node if this location hasn't been visited by other paths
        if pnext in tree:
            path_success = True
            node = tree[pnext]
            node.link(path[-1])
            break
        else:
            newnode = PathNode(pnext)
            visited.add(pnext)
            # now newnode is parent of prevnode
            newnode.link(prevnode)
            path.append(newnode)               
        
        if terminate_mask[pnext]:
            path_success = True
            break
    if path_success:
        for p in path:
            tree[tuple(p.v)] = p
    return path, path_success
```

```{code-cell} ipython3
from astromorpho import morpho
```

```{code-cell} ipython3
speed = 10*uc.utils.rescale(
    sum(s**2*morpho.sato2d(np.random.randn(*field.shape), s)
        for s in (1.5, 3, 6, 12))
    #1.5**2*morpho.sato2d(np.random.randn(*field.shape), 1.5)\
    #+ 3**2*morpho.sato2d(np.random.randn(*field.shape), 3)\
    #+ 1.5**2*ndi.gaussian_filter(np.random.randn(*field.shape),1.5)\
    #+ 3**2*ndi.gaussian_filter(np.random.randn(*field.shape),3)\
    #+ 6**2*ndi.gaussian_filter(np.random.randn(*field.shape),6)\
    #+ 12**2*ndi.gaussian_filter(np.random.randn(*field.shape),12)
)
plt.imshow(speed)
plt.colorbar()
```

```{code-cell} ipython3
# seeds =  np.array([(255,255), 
#                    (64,64), 
#                    (300, 64),
#                    (300, 440),
#                    (100,450), (450,100), (450,450)])

seeds = np.random.randint(10,500, size=(50,2))
seeds = [(255,255)] +  [s for s in seeds 
                        if eu_dist(s,(255,255)) > 250]
seeds = np.array(seeds)
#seeds = np.concatenate(([(255,255)],seeds))
#seeds += np.random.randint(-16,16, size=(len(seeds),2))
plt.imshow(speed)
plt.plot(seeds[:,1], seeds[:,0], 'r.')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
x = np.indices((2,3))
```

```{code-cell} ipython3
x.reshape((2,-1)).T
```

```{code-cell} ipython3

```

```{code-cell} ipython3
kdt = sp.spatial.KDTree(seeds)

px_locs = (np.indices(speed.shape)
           .reshape((2,-1))
           .T)

labels = kdt.query(px_locs)[1] + 1
labels = labels.reshape(speed.shape)
plt.imshow(labels)
```

```{code-cell} ipython3
phi0 = np.zeros(field.shape)
phi0[255,255] = 1
#phi0[250,150] = phi0[255,430] = 1
phi0 = ndi.binary_dilation(phi0,iterations=2)
plt.imshow(phi0, interpolation='nearest')

phi0 = ~phi0
```

```{code-cell} ipython3
ttx = skfmm.travel_time(phi0, speed=speed)
#ttx = ttx*(ttx>0)
#ttx[ttx.mask] = np.max(ttx)
#ttx = np.array(ttx)
ttx = np.ma.filled(ttx,np.max(ttx))

#boundary_mask = labels == 1
boundary_mask = ttx < np.percentile(ttx,33)

plt.figure()
plt.imshow(uc.clip_outliers(ttx),cmap='BuPu'); plt.colorbar()
plt.contour(ttx, levels=[np.percentile(ttx, 25)], colors='c')
plt.contour(boundary_mask, levels=[0.5], colors='r')
```

```{code-cell} ipython3
#np.array(ttx)
```

```{code-cell} ipython3
#plt.imshow(boundary_mask)
```

```{code-cell} ipython3
tm_mask = ndi.binary_dilation(ttx<np.percentile(ttx,0.1),iterations=1)
plt.imshow(tm_mask)
```

```{code-cell} ipython3
def get_tips(tree):
    return [n for n in tree.values() if not len(n.children)]

def get_roots(tree):
    return [n for n in tree.values() if not n.parent]

def follow_to_root(tip, max_nodes=1000000):
    acc = [tip]
    for i in range(max_nodes):
        parent = tip.parent
        if parent is None:
            break
        tip = parent
        acc.append(tip)
        if i >= max_nodes-1:
            print('limit reached')
            break
    return acc

def follow_to_root_rec(tip):
    if not tip.parent:
        return [tip]
    else:
        return [tip] + follow_to_root(tip.parent)

def follow_to_root_rec(tip):
        return [tip] + ([] if not tip.parent else follow_to_root(tip.parent))

def apath_to_root(tip):
    return np.array([n.v for n in follow_to_root(tip)])
    
def count_occurences(tree, shape):
    counts =  np.zeros(shape)
    for tip in tqdm(get_tips(tree)):
        for n in follow_to_root(tip):
            if hasattr(n, 'count'):
               n.count += 1
            else:
               n.count = 1
            counts[tuple(n.v)] += 1
    return counts
    

def plot_tree(tree, ax=None, random_colors=True, linecolor='m', lw=1, max_lw=10):
    
    if ax is None:
        fig, ax = plt.subplots(1,1)

    color = np.random.rand(3) if random_colors else linecolor
    for loc,n in tree.items():
        if n.parent is None:
            ax.plot(n.v[1], n.v[0], 'r.')
        
        for ch in n.children:
            vx = np.vstack([n.v, ch.v])
            if hasattr(ch,'diam'):
                lw = min(max_lw, ch.diam)
            else:
                lw = lw
                
            ax.plot(vx[:,1], vx[:,0], '-', lw=lw, alpha=0.95, color=color)
    ax.axis('equal')
```

```{code-cell} ipython3
def gauss2d(xmu=0, ymu=0, xsigma=10, ysigma=10):
    xsigma, ysigma = list(map(float, [xsigma, ysigma]))
    return lambda x,y: np.exp(-(x-xmu)**2/(2*xsigma**2) - (y-ymu)**2/(2*ysigma**2))

def gauss_blob(loc, sigma, shape):
    xx,yy = np.mgrid[:shape[0],:shape[1]]
    fn = gauss2d(xmu=loc[0],ymu=loc[1], xsigma=sigma,ysigma=sigma)
    return fn(xx,yy)
```

```{code-cell} ipython3
# gauss_locs = np.random.randint(10,500, size=(100,2))
# plt.imshow(uc.clip_outliers(ttx), cmap='gray')
# plt.plot(gauss_locs[:,0], gauss_locs[:,1], '.')
```

```{code-cell} ipython3
#bumps = sum(ttx[tuple(loc[::-1])]*0.1*gauss_blob(loc, 10, ttx.shape) for loc in tqdm(gauss_locs))
#plt.imshow(bumps); plt.colorbar()
```

```{code-cell} ipython3
#ttx_bumps = ttx + bumps
```

```{code-cell} ipython3
tree = dict()

init_pts = np.random.randint(50,450, size=(10,2))

#newtree = merging_rw_gd(ttx, (10,50), terminate_mask=tm_mask, nsteps=1000, tree=tree)
fails = []
for p in tqdm(init_pts):
    path,success = merging_rw_gd(ttx, p, 
                                 terminate_mask=tm_mask, 
                                 pjitter=0.0, 
                                 nsteps=10000, tree=tree)
    if not success:
        fails.append(path)

explored = set()
paths_prev = [self_avoiding_2d_gd(ttx, p, terminate_mask=tm_mask, 
                                  pjitter=0,
                                  nsteps=10000, travelled=explored)
         for p in tqdm(init_pts)]
```

```{code-cell} ipython3
len(tree)
```

```{code-cell} ipython3
plt.imshow(uc.clip_outliers(ttx_bumps), cmap='gray')
plot_tree(tree, ax=plt.gca(), random_colors=False, lw=0.75)
#plt.plot(gauss_locs[:,0], gauss_locs[:,1], '.')
for path in fails:
    path = np.array([n.v for n in path])
    plt.plot(*path.T[::-1], color='y',lw=0.75)
#for path in paths_prev:
#    plt.plot(*path.T[::-1], color='c',lw=0.75)
```

```{code-cell} ipython3
#tm_mask[tuple(fails[-1][-1].v)]
```

```{code-cell} ipython3
#paths[0]
```

```{code-cell} ipython3
#%time path1 = np.array([n.v for n in follow_to_root(tree[tuple(init_pts[-1])])])
path1 = apath_to_root(tree[tuple(init_pts[1])])
plt.imshow(uc.clip_outliers(ttx_bumps), cmap='gray')
plt.plot(path1[:,1], path1[:,0],'r')
```

```{code-cell} ipython3
plt.figure()
plt.plot(np.diff(ttx[path1[:,0],path1[:,1]]))
plt.plot(np.diff(ttx_bumps[path1[:,0],path1[:,1]]))
```

```{code-cell} ipython3
%time path2 = np.array([n.v for n in follow_to_root_rec(tree[tuple(init_pts[-1])])])
```

```{code-cell} ipython3
plt.imshow(uc.clip_outliers(ttx), cmap='gray')
plt.plot(*path1.T[::-1], 'r')
plt.plot(*path2.T[::-1], 'b--')
```

```{code-cell} ipython3
for tip in get_tips(tree):
    p = apath_to_root(tip)
    speed[tuple(p[:,i] for i in (0,1))] += 5
plt.imshow(speed)
```

```{code-cell} ipython3
ttx = skfmm.travel_time(phi0, speed=speed)
plt.imshow(uc.clip_outliers(ttx))
```

```{code-cell} ipython3
# #explored = set()

# init_pts = np.random.randint(50,450, size=(10,2))

# #path = self_avoiding_2d_gd(ttx, (10,50), terminate_mask=tm_mask, nsteps=1000, travelled=explored)

# paths = [self_avoiding_2d_gd(ttx, p, terminate_mask=tm_mask, nsteps=1000, travelled=explored)
#          for p in tqdm(init_pts)]
```

```{code-cell} ipython3
# plt.imshow(ttx, cmap='gray')
# for path in paths:
#     plt.plot(path[:,1], path[:,0], '-', lw=0.75)
```

```{code-cell} ipython3
from imfun.core.coords import eu_dist
```

```{code-cell} ipython3
init_pts = np.array([(i,j) for i in range(50,450) for j in range(50,450) 
                     if boundary_mask[(i,j)]])
                     #if eu_dist((i,j),(255,255)) < 200])

Npts = 5000
init_pts_dense = np.random.permutation(init_pts)
init_pts = np.random.permutation(init_pts)[:Npts]

init_pts = sorted(init_pts, key=lambda p: eu_dist(p, (255,255)), reverse=True)
#init_pts_dense = sorted(init_pts_dense, key=lambda p: eu_dist(p, (255,255)), reverse=True)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
#speed = 25*uc.utils.rescale(ndi.gaussian_filter(np.random.randn(*field.shape),3))

# speed0 = 10*uc.utils.rescale(
#       1.5**2*ndi.gaussian_filter(np.random.randn(*field.shape),1.5)\
#     + 3**2*ndi.gaussian_filter(np.random.randn(*field.shape),3)\
#     + 6**2*ndi.gaussian_filter(np.random.randn(*field.shape),6)\
#     + 12**2*ndi.gaussian_filter(np.random.randn(*field.shape),12))

speed0 = 10*uc.utils.rescale(
    sum(s**2*morpho.sato2d(np.random.randn(*field.shape), s)
        for s in (1.5, 3, 6, 12)))

speed = speed0.copy()
speed_gamma = 1

ttx0 = skfmm.travel_time(phi0, speed=speed0**speed_gamma)

# bumps = sum(ttx0[tuple(loc[::-1])]*0.1*gauss_blob(loc, 10, ttx.shape) 
#             for loc in tqdm(gauss_locs))

#ttx0 += bumps

ttx0 = np.ma.filled(ttx0, np.max(ttx0))

ttx = ttx0.copy()
tree = dict()

plt.imshow(speed**speed_gamma, cmap='plasma'); plt.colorbar()


ttx_acc = [ttx0]
speed_acc = [speed]

speed_update = np.zeros(field.shape)
speed_corr = np.zeros(field.shape)

Nseeds =  2500

fails = []
alpha = 0.999
j = 0
for p0 in tqdm(np.random.permutation(init_pts)[:Nseeds]):
#for p0 in tqdm(init_pts[:500]):
#for p0 in tqdm(init_pts[::-1][:500]):
    p0 = tuple(p0)
    if ttx[p0] == np.max(ttx):
        #print('skipping point: ', p0, 'because', ttx[p0])
        continue
    
    try_path, finished = merging_rw_gd(ttx, p0, 
                                       terminate_mask=tm_mask, 
                                       pjitter=0.0, 
                                       nsteps=10000, 
                                       tree=tree)
    #print(path[-1])
    #end = tuple(path[-1])
    #finished = tm_mask[end] or (end in explored)
    if finished:
        # slow part...
        speed_update = np.zeros(field.shape)
        #speed_update[tuple(path[:,i] for i in (0,1))] += 1
        #for i,p in enumerate(paths):
        #    speed_update[tuple(p[:-1,i] for i in (0,1))] += 1
        #speed_corr = ndi.gaussian_filter(speed_corr,1) + speed_update
        apath = apath_to_root(tree[p0])
        speed_update[tuple(apath[:-1,i] for i in (0,1))] += 1
        speed_corr = speed_corr + speed_update
        # [Q:] Do I need to Gauss-blur previous speed update?
        #      Any reason for this at all?
        #speed_corr = ndi.gaussian_filter(speed_corr,1) + speed_update
        speed += (speed_corr*alpha**j)**(1/speed_gamma)
        ttx = ttx + skfmm.travel_time(phi0, speed=speed)
        ttx = np.ma.filled(ttx, np.max(ttx))
        j += 1
        #ttx_acc.append(ttx)
        #speed_acc.append(speed)
    else:
        fails.append(np.array([n.v for n in try_path]))
        print('not finished for loc', p0)
```

```{code-cell} ipython3
len(tree), len(fails)
```

```{code-cell} ipython3
#plt.imshow(tm_mask)
```

```{code-cell} ipython3
#%matplotlib qt
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
plt.imshow(uc.clip_outliers(ttx), cmap='gray')
plot_tree(tree, ax=plt.gca(), random_colors=False, lw=0.5)
#for path in paths:
#    plt.plot(path[:,1], path[:,0], '-', lw=0.75, color='r')
```

```{code-cell} ipython3
plt.imshow(speed); plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(speed**2); plt.colorbar()
```

```{code-cell} ipython3
fig,axs = plt.subplots(1,2, figsize=(12,5))
axs[0].imshow(uc.clip_outliers(ttx0),cmap='Spectral')
axs[1].imshow(uc.clip_outliers(ttx),cmap='Spectral')
for ax in axs:
    ax.axis(False)
plt.tight_layout()
```

```{code-cell} ipython3
Gx = nx.DiGraph()

for tip in get_tips(tree):
    ap = apath_to_root(tip)
    Gx.add_edges_from(list(itt.pairwise(map(tuple, ap[::-1]))))
```

```{code-cell} ipython3
#speed.shape
```

```{code-cell} ipython3
counts1 = count_occurences(tree,speed.shape)
```

```{code-cell} ipython3
counts = count_occurences_nx(Gx, speed.shape)
```

```{code-cell} ipython3
plt.imshow(np.log10(1+counts), cmap='plasma')
plt.colorbar()
```

```{code-cell} ipython3
# plt.imshow(ndi.gaussian_filter(np.log10(10+counts),1.5), cmap='plasma')
# plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(counts > 20, interpolation='nearest', cmap='plasma')
plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(speed); plt.colorbar()
```

```{code-cell} ipython3
ttx2 = skfmm.travel_time(phi0, speed=speed0/5 + np.log10(1+counts))
ttx2 = np.ma.filled(ttx2, np.max(ttx2))
```

```{code-cell} ipython3
0.9**100
```

```{code-cell} ipython3
plt.imshow(uc.clip_outliers(ttx2))
plt.colorbar()
```

```{code-cell} ipython3
# explored = set()
# paths2 = []
# for j,p0 in enumerate(tqdm(init_pts_dense[:500])):
#     path = self_avoiding_2d_gd(ttx2, tuple(p0), terminate_mask=tm_mask, nsteps=1000, pjitter=0.0, travelled=explored.copy())
#     end = tuple(path[-1])
#     finished = tm_mask[end] or (end in explored)
#     if finished:
#         paths2.append(path)
#         explored.update([tuple(p) for p in path])
```

```{code-cell} ipython3
tree2 = dict()
Npts_dense=-1
for p0 in tqdm(init_pts_dense[:Npts_dense]):
    if ttx2[tuple(p0)] == np.max(ttx2):
        continue
    try_path, finished = merging_rw_gd(ttx2, tuple(p0), 
                                       terminate_mask=tm_mask, 
                                       pjitter=0.0, 
                                       nsteps=1000, 
                                       tree=tree2)
```

```{code-cell} ipython3
Gx2 = nx.DiGraph()
for tip in get_tips(tree2):
    ap = apath_to_root(tip)
    Gx2.add_edges_from(list(itt.pairwise(map(tuple, ap[::-1]))))
#for path in tqdm(paths2):
#    Gx2.add_edges_from(list(itt.pairwise(map(tuple, path[::-1]))))
```

```{code-cell} ipython3
counts2 = count_occurences_nx(Gx2,speed.shape)
```

```{code-cell} ipython3
fig,axs = plt.subplots(1,2,figsize=(12,5))
axs[0].imshow(np.log10(2+counts), cmap='plasma')
axs[1].imshow(np.log10(10+counts2), cmap='plasma')

plt.tight_layout()
```

```{code-cell} ipython3
def assign_diameters(tree, min_diam=0.01, max_diam=6, gamma=1.0):
    for loc,n in tree.items():
        n.diam = 0
        
    for tip in tqdm(get_tips(tree)):
        for n in follow_to_root(tip):
            if not hasattr(n, 'diam'):
                n.diam = 0
            n.diam += min_diam**gamma
    for loc,n in tree.items():
        n.diam = min(max_diam, n.diam**(1/gamma))


def assign_diameters_nx(G, min_diam=0.01, max_diam=6, gamma=1.0):
    for n in G:
        G.nodes[n]['diam'] = 0
        
    for tip in tqdm(gu.get_tips(G)):
        for p in follow_to_root_nx(G,tip):
            n = G.nodes[p]
            n['diam'] += min_diam**gamma
    for n in G:
        G.nodes[n]['diam'] = min(max_diam, G.nodes[n]['diam']**(1/gamma))
```

```{code-cell} ipython3
assign_diameters(tree, min_diam=0.1, gamma=1.5, max_diam=9)
assign_diameters(tree2, min_diam=0.01, gamma=1.1, max_diam=9)
```

```{code-cell} ipython3
plot_tree(tree, max_lw=4, random_colors=False,)
```

```{code-cell} ipython3
assign_diameters_nx(Gx, min_diam=0.01, gamma=1, max_diam=9)
assign_diameters_nx(Gx2, min_diam=0.01, gamma=1.5, max_diam=9)
```

```{code-cell} ipython3
#[Gx2.nodes[n]['diam'] for n in gu.get_roots(Gx2)]
```

```{code-cell} ipython3
def tanh_pulse(x0, width, sharpness=10):
    hw = width/2
    k = sharpness
    def _pulse(x):
        a = (1 + np.tanh((x-x0 + hw)*k))
        b = (1 + np.tanh(-(x-x0 - hw)*k))
        return a*b/4
    return _pulse
            

def tanh_kern2d(x0=0,y0=0, xw=10, yw=10, sharpness=5):
    kx = sharpness
    def _pulse(x,y):
        #a = (1 + np.tanh((x-x0 + xw/2)*kx + (y-y0 + yw/2)*kx))
        a = (1 + np.tanh((x-x0 + xw/2)*kx))*(1 + np.tanh((y-y0 + yw/2)*kx))
        #b = (1 + np.tanh(-(x-x0 - xw/2)*kx -(y-y0 - yw/2)*kx))
        return a#*b/4
    return _pulse
        
    

def tanh_blob(loc, width, fullshape, sharpness=5):
    xx,yy = np.mgrid[:fullshape[0],:fullshape[1]]
    fn = tanh_kern2d(loc[1], loc[0], xw=width, yw=width, sharpness=sharpness)
    return fn(xx,yy) 
```

```{code-cell} ipython3
# x = tanh_blob((255,255), 6, counts.shape)
# plt.imshow(x); plt.colorbar()
```

```{code-cell} ipython3
# x = np.linspace(-2,2,500)

# plt.plot(x, tanh_pulse(-0.5, 1.5, 20)(x))
```

```{code-cell} ipython3
# x = gauss_blob((255,255), 16, counts.shape)
# plt.imshow(x**0.5); plt.colorbar()
# np.max(x)
```

```{code-cell} ipython3
#plt.plot(x[255,:])
```

```{code-cell} ipython3
def make_portrait_nx(G, shape, min_diam_show=0, do_threshold=False):
    portrait = np.zeros(counts.shape)

    for n in tqdm(G):
        diam = G.nodes[n]['diam']
        if diam >= min_diam_show:
            amp = np.log10(0.1+G.nodes[n]['count'])
            #amp = diam
            #portrait += amp*gauss_blob(n, diam/2, portrait.shape)
            blob = gauss_blob(n, diam/2, portrait.shape)
            if do_threshold:
                blob = 1.0*(blob>0.5*np.max(blob))
            portrait = np.maximum(portrait, amp*blob)
    #portrait[tm_mask] = np.percentile(portrait[tm_mask],99)
    #portrait = np.maximum(portrait, np.max(portrait)*gauss_blob((255,255), 10, counts.shape))
    return portrait
    
```

```{code-cell} ipython3
# portrait = np.zeros(counts.shape)

# for n in tqdm(Gx):
#     diam = Gx.nodes[n]['diam']
#     if diam > 0.01:
#         amp = np.log10(0.1+Gx.nodes[n]['count'])
#         #amp = diam
#         #portrait += amp*gauss_blob(n, diam/2, portrait.shape)
#         portrait = np.maximum(portrait, amp*gauss_blob(n, diam/2, portrait.shape))


#         #portrait = np.maximum(portrait, np.max(portrait)*gauss_blob((255,255), 10, counts.shape))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
portrait = make_portrait_nx(Gx, speed.shape)

plt.imshow(portrait, cmap='plasma'); plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(portrait,cmap='gray'); plt.colorbar()
```

```{code-cell} ipython3
portrait2 = make_portrait_nx(Gx2, speed.shape, min_diam_show=0.05, do_threshold=True)
```

```{code-cell} ipython3
plt.imshow(portrait2, cmap='plasma'); plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(portrait2, cmap='gray_r')
plt.axis('off')
```

```{code-cell} ipython3
plt.imshow(portrait2, cmap='gray')
plt.axis('off')
```

```{code-cell} ipython3
# portrait2 = np.zeros(counts.shape)
# #portrait2 = 3*gauss_blob((255,255), 10, counts.shape)
# for n in tqdm(Gx2):
#     diam = Gx2.nodes[n]['diam']
#     if diam > 0.01:
#         amp = np.log10(0.1 + Gx2.nodes[n]['count'])
#         #amp = diam
#         blob = amp*gauss_blob(n, diam/2, portrait2.shape)
#         portrait2 = np.maximum(portrait2, blob)
# portrait2 = np.maximum(portrait2, np.log10(np.max(counts2))*gauss_blob((255,255), 10, counts.shape))
```

```{code-cell} ipython3
plt.imshow(portrait2, cmap='plasma'); plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(portrait2, cmap='plasma'); plt.colorbar()
```

```{code-cell} ipython3
# started from the very tips
plt.imshow(portrait2, cmap='plasma'); plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(portrait2); plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(portrait2); plt.colorbar()
```

```{code-cell} ipython3

```

```{code-cell} ipython3
from imfun import ui
```

```{code-cell} ipython3
ui.group_maps(ttx_acc[::10])
```

```{code-cell} ipython3
#p = paths[0]
```

```{code-cell} ipython3
#%timeit speed[tuple(p[:,i] for i in (0,1))]
```

```{code-cell} ipython3
#%timeit ndi.map_coordinates(speed, p.T, order=1)
```
