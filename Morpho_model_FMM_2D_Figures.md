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

```{code-cell} ipython3
import ucats as uc
```

```{code-cell} ipython3
import morphsnakes
```

```{code-cell} ipython3
# macwe = morphsnakes.MorphACWE(img, lambda2=2, smoothing=1,levelset=img > 0)
# macwe.run(500)
```

```{code-cell} ipython3
astro.morpho.eigh = np.linalg.eigh
```

```{code-cell} ipython3
%matplotlib inline
```

### Define speed as just brightness

```{code-cell} ipython3
#from imfun.bwmorph import neighbours
```

```{code-cell} ipython3
# def follow_to_root_nx(g, tip, max_nodes=1000000):
#     visited = {tip}
#     acc = [tip]
#     for i in range(max_nodes):
#         parents = list(g.predecessors(tip))
#         parents = [p for p in parents if not p in visited]
#         if not len(parents):
#             break
#         tip = parents[0]
#         visited.add(tip)
#         acc.append(tip)
#     if i >= max_nodes-1:
#         print('limit reached')
#     return acc

# def count_occurences_nx(G, shape):
#     counts =  np.zeros(shape)
#     for tip in tqdm(gu.get_tips(G)):
#         for p in follow_to_root_nx(G,tip):
#             n = G.nodes[p]
#             if 'count' in n:
#                n['count'] += 1
#             else:
#                n['count'] = 1
#             counts[p] += 1
#     return counts
```

## merging GD paths in stochastic FMM-based travel-time maps

```{code-cell} ipython3
Npx = 512
X, Y = np.mgrid[:512,:512]
X.shape
```

```{code-cell} ipython3
field_shape = (Npx,Npx)
```

```{code-cell} ipython3
speed = uc.utils.rescale(
    sum(s**2*morpho.sato2d(np.random.randn(*field_shape), s)
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
import seaborn as sns
```

## Set up initial points and target point

```{code-cell} ipython3
#px_locs = 
```

```{code-cell} ipython3
phi0 = np.zeros(field_shape)
phi0[255,255] = 1
phi0 = ndi.binary_dilation(phi0, iterations=3)
plt.imshow(phi0)
plt.axis([250,260,260,250])
phi0 = ~phi0
```

```{code-cell} ipython3
def show_tt_map(ttm, ax=None, with_cbar=False, with_boundary=False, 
                boundary_percentile=50):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    ttx = np.ma.filled(ttm,np.max(ttm))
    #ttx = ttm
    vmax = np.percentile(ttx[ttx<np.max(ttx)],99)
    ih = ax.imshow(ttx, vmin=0, vmax=vmax, cmap='BuPu')
    if with_cbar:
        plt.colorbar(ih, ax=ax)
    if with_boundary:
        boundary_mask = ttx < np.percentile(ttx,boundary_percentile)
        boundary_mask = ndi.binary_fill_holes(boundary_mask)
        plt.contour(boundary_mask, levels=[0.5], colors='r',linewidths=0.75)
    ax.axis('off')
```

```{code-cell} ipython3
ttx = skfmm.travel_time(phi0, speed=speed)
show_tt_map(ttx, with_boundary=True)
```

```{code-cell} ipython3
def cartesian2polar(x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x,y)
    return r,theta

def polar2cartesian(r,theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y
```

```{code-cell} ipython3
def gen_random_polar_points(Npoints, rmin=0.1, rmax=1):
    radii = np.random.uniform(rmin,rmax, size=Npoints)
    theta = np.random.uniform(0, 2*np.pi, size=Npoints)
    return np.array([radii, theta]).T
```

```{code-cell} ipython3
init_pts_polar = gen_random_polar_points(20, 50, 225)
init_pts = np.array([polar2cartesian(*p) for p in init_pts_polar]) + (255,255)
```

```{code-cell} ipython3
show_tt_map(ttx, with_boundary=False)
plt.plot(init_pts[:,0], init_pts[:,1], 'r.')
```

```{code-cell} ipython3
def plot_tree(tree, ax=None, random_colors=True, 
              mfc='r',
              linecolor='m', lw=1, max_lw=10):
    
    if ax is None:
        fig, ax = plt.subplots(1,1)

    color = np.random.rand(3) if random_colors else linecolor
    for loc,n in tree.items():
        if n.parent is None:
            ax.plot(n.v[1], n.v[0], '.', color=mfc, zorder=1e5)
        
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
import ifmm_branching as iffm
```

```{code-cell} ipython3
reload(iffm)
```

```{code-cell} ipython3
tree,fails = iffm.build_tree(ttx, init_pts, tm_mask =~phi0)
```

```{code-cell} ipython3
len(tree),len(fails)
```

```{code-cell} ipython3
plot_tree(tree)
```

## Figure 1: effect of speed field

```{code-cell} ipython3
filaments_ms = uc.utils.rescale(sum(s**2*morpho.sato2d(np.random.randn(*field_shape), s)
                                    for s in (1.5, 3, 6, 12)))

filaments_hf = uc.utils.rescale(morpho.sato2d(np.random.randn(*field_shape), 1.5))
filaments_lf = uc.utils.rescale(morpho.sato2d(np.random.randn(*field_shape), 6))
```

```{code-cell} ipython3
gauss_ms = uc.utils.rescale(sum(s**2*ndi.gaussian_filter(np.random.randn(*field_shape), s)
                                    for s in (1.5, 3, 6, 12)))
gauss_hf = uc.utils.rescale(ndi.gaussian_filter(np.random.randn(*field_shape), 1.5))
gauss_lf = uc.utils.rescale(ndi.gaussian_filter(np.random.randn(*field_shape), 6))
```

```{code-cell} ipython3
uniform_field = np.ones(field_shape)*0.85
uniform_field[0,0] = 0
```

```{code-cell} ipython3
fields = [
    uniform_field,
    gauss_hf,
    #gauss_lf,
    gauss_ms,
    filaments_hf*1.5,
    #filaments_lf,
    filaments_ms*1.5
]
```

```{code-cell} ipython3
#show_tt_map(skfmm.travel_time(phi0, speed=fields[5]))
```

```{code-cell} ipython3
ttms = [skfmm.travel_time(phi0, speed=m) for m in fields]
```

```{code-cell} ipython3
plt.rc('figure', dpi=150)
```

```{code-cell} ipython3
#uc.utils.percentile_rescale(uniform_field)
```

```{code-cell} ipython3
reload(vis)
```

```{code-cell} ipython3
fig,axs = plt.subplots(3,len(fields), figsize=(9,5))
for ax,m in zip(axs[0],fields):
    ax.imshow(m, vmin=0,vmax=1, cmap='viridis')
    ax.axis('off')
for ax,tt in zip(axs[1],ttms):
    show_tt_map(tt,ax)

for ax,tt in zip(axs[2],ttms):
    tree, fails = iffm.build_tree(tt, init_pts, tm_mask =~phi0)
    if len(tree):
        plot_tree(tree,ax,random_colors=False,lw=0.75,linecolor='k')
        ax.axis([0,512,512,0])
        ax.axis('off')
plt.tight_layout()
vis.multi_savefig(fig, 'figures/speed-field-effect')
```

## Fig. 2 Effect of speed field update

```{code-cell} ipython3
reload(iffm)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
ttx = skfmm.travel_time(phi0, speed=filaments_ms)
show_tt_map(ttx, with_boundary=True)
bmask = ttx < np.percentile(ttx, 50)
plt.imshow(bmask, alpha=0.25)
```

```{code-cell} ipython3
seeds = np.array(np.where(bmask)).T
#seeds = np.random.permutation(seeds)[:50]
seeds = np.random.permutation(seeds)[:500]
```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, phi0, 
                                             seeds, 
                                             update_amp=1,
                                             tm_mask=~phi0,
                                             batch_size=5)
```

```{code-cell} ipython3
plot_tree(tree)
```

```{code-cell} ipython3
plt.imshow(uc.clip_outliers(np.log2(1+speed)), interpolation='nearest'); plt.colorbar()
```

```{code-cell} ipython3
%%time 

fig, axs = plt.subplots(3,4,  figsize=(8,5))

col = 0

uamps = [0, 1, 0.5, 2]
algs = ['linear', 'log', 'linear', 'power']

for col, (uam, alg) in enumerate(zip(uamps,algs)): 
    tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, phi0, seeds[:250], 
                                                 update_amp=uam,
                                                 scaling=alg,
                                                 tm_mask=~phi0, batch_size=2)
    plot_tree(tree, axs[0,col], random_colors=False, mfc='k', linecolor='k', lw=0.75)
    axs[1,col].imshow(uc.clip_outliers(np.log2(1+speed)), cmap='viridis')
    show_tt_map(ttx, ax=axs[2,col])
    axs[0,col].axis([0,512, 512,0])


for ax in np.ravel(axs):
    ax.axis('off')

plt.tight_layout()
vis.multi_savefig(fig, 'figures/ifmm-updates')
```

## Fig 3. Sampling strategies

```{code-cell} ipython3
#import visualization as vis
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
bmask_filt =  uc.masks.largest_region(ndi.binary_fill_holes(np.ma.filled(bmask)))
```

```{code-cell} ipython3
reload(vis)
plasma_x = vis.make_seethrough_colormap()
jet_x = vis.make_seethrough_colormap('jet')
reds_x = vis.make_seethrough_colormap('Reds')
blues_x = vis.make_seethrough_colormap('Blues')
spectralr_x = vis.make_seethrough_colormap('Spectral_r')
```

```{code-cell} ipython3
central_prob = gauss_blob((255,255), 75, field_shape)*(bmask_filt)
central_prob = ndi.gaussian_filter(central_prob, 5)
central_prob = central_prob/np.sum(central_prob)
plt.imshow(central_prob)
```

```{code-cell} ipython3
periph_prob = gauss_blob((255,255), 200, field_shape) - gauss_blob((255,255), 175, field_shape)
periph_prob = periph_prob**2
periph_prob *= bmask_filt
periph_prob = ndi.gaussian_filter(periph_prob, 5)
periph_prob = periph_prob/np.sum(periph_prob)

plt.imshow(periph_prob)
```

```{code-cell} ipython3
uniform_prob = ndi.gaussian_filter(np.ones(field_shape)*bmask_filt,5)

uniform_prob = uniform_prob/np.sum(uniform_prob)
plt.imshow(uniform_prob)
```

```{code-cell} ipython3
def sample_points(prob_map, size=500):
    locs = np.array(np.where(prob_map>0)).T
    idx = np.random.choice(len(locs), size=size, replace=False, p=prob_map[prob_map>0])
    return locs[idx]
```

```{code-cell} ipython3
uniform_locs = sample_points(uniform_prob) 
periph_locs = sample_points(periph_prob)
central_locs = sample_points(central_prob)

uniform_locs_s   = sorted(uniform_locs, key=lambda p: eu_dist(p, (255,255)))

# periph_locs_s = sorted(periph_locs, 
#                        key=lambda p: -eu_dist(p,(255,255)))

# periph_locs_s2 = sorted(periph_locs, key=lambda p: eu_dist(p,(255,0)))

# central_locs_s = sorted(central_locs,key=lambda p: eu_dist(p,(255,255)))
```

```{code-cell} ipython3
periph_locs_dense = sample_points(periph_prob,1000)
```

```{code-cell} ipython3
# sorted_central = sorted(seeds[:500], key = lambda p: eu_dist(p, (255,255)))
# sorted_peripheral = sorted_central[::-1]
```

```{code-cell} ipython3
reload(iffm)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_s, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)
plt.figure()
plt.imshow(np.ma.masked_less(uniform_prob,1e-10), cmap='Wistia', alpha=0.25, )
ax = plt.gca()
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
plt.tight_layout(); ax.axis('off')
plt.figure(); plt.imshow(np.log2(1+speed), cmap='plasma')
```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_s[::-1], 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)
plt.figure()
plt.imshow(np.ma.masked_less(uniform_prob,1e-10), cmap='Wistia', alpha=0.25, )
ax = plt.gca()
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
plt.tight_layout(); ax.axis('off')
plt.figure(); plt.imshow(np.log2(1+speed), cmap='plasma')
```

```{code-cell} ipython3
# tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
#                                              phi0, 
#                                              uniform_locs_s[::-1], 
#                                              update_amp=1,
#                                              scaling='log',
#                                              tm_mask=~phi0, 
#                                              batch_size=2)
# plt.figure()
# plt.imshow(np.ma.masked_less(uniform_prob,1e-10), cmap='Wistia', alpha=0.25, )
# ax = plt.gca()
# plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
# plt.tight_layout(); ax.axis('off')
```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             periph_locs, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)
plt.figure()
plt.imshow(np.ma.masked_less(periph_prob,1e-10), cmap='Wistia', alpha=0.25, )
ax = plt.gca()
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
ax.axis('off')
plt.figure(); plt.imshow(np.log2(1+speed), cmap='plasma')
```

```{code-cell} ipython3
# tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
#                                              phi0, 
#                                              periph_locs_s, 
#                                              update_amp=1,
#                                              scaling='power',
#                                              tm_mask=~phi0,
#                                              batch_size=2)
# plt.figure()
# plt.imshow(np.ma.masked_less(periph_prob,1e-10), cmap='Wistia', alpha=0.25, )
# ax = plt.gca()
# plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
# ax.axis('off')
```

```{code-cell} ipython3
phi0_asym = np.zeros(field_shape,bool)
phi0_asym[255, 400] = True
phi0_asym = ~ndi.binary_dilation(phi0_asym,iterations=3)
```

```{code-cell} ipython3
#phi0_asym.shape
```

```{code-cell} ipython3
plt.imshow(filaments_ms + 1e4*periph_prob)
```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms+ 1e4*periph_prob, 
                                             phi0_asym, 
                                             periph_locs, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0_asym, 
                                             batch_size=2)
plt.figure()
plt.imshow(np.ma.masked_less(periph_prob,1e-10), cmap='Wistia', alpha=0.25, )
ax = plt.gca()
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
plt.tight_layout(); ax.axis('off')
plt.figure(); plt.imshow(np.log2(1+speed), cmap='plasma')
```

```{code-cell} ipython3
plt.imshow(filaments_ms+ 1e9*periph_prob**2)
```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms+ 1e9*periph_prob**2, 
                                             phi0_asym.T, 
                                             periph_locs_dense, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0_asym.T,
                                             batch_size=2)
plt.figure()
plt.imshow(np.ma.masked_less(periph_prob,1e-10), cmap='cool', alpha=0.25, )
ax = plt.gca()
plot_tree(tree, random_colors=False, mfc='k', linecolor='forestgreen', lw=1, ax=ax)
plt.tight_layout(); ax.axis('off')
plt.figure(); plt.imshow(np.log2(1+speed), cmap='plasma')
```

```{code-cell} ipython3
reload(iffm)
```

```{code-cell} ipython3
iffm.assign_diameters(tree, min_diam=0.5, max_diam=5, gamma=1.5)
```

```{code-cell} ipython3
#plt.plot()
```

```{code-cell} ipython3
plt.figure()
plt.imshow(np.ma.masked_less(periph_prob,1e-10), cmap='cool', alpha=0.25, )
ax = plt.gca()
plot_tree(tree, random_colors=False, mfc='r', linecolor='forestgreen', lw=1, max_lw=1, ax=ax)

for p in periph_locs_dense:
    if np.random.rand() < 0.15:
        color = np.clip(1 - np.random.rand(3)**2 + (0.3,0.25,0.75),0,1)
        angle = np.random.uniform(-60,60)
        size = np.random.uniform(2,16)
        zorder=np.random.uniform(0,20)
        ax.plot(p[1],p[0], color=color, marker=(6, 2, angle), alpha=0.95,
                ms=size,zorder=zorder)
    
    elif np.random.rand() < 0.025:
        ax.plot(p[1],p[0], color='r', marker='o', ms=5, alpha=0.5)
        
plt.tight_layout(); ax.axis('off')
plt.savefig('/tmp/ny-ring.png', )
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
np.max([n.diam for loc,n in tree.items()])
```

```{code-cell} ipython3
#cartesian2polar(*uniform_locs[4])
```

```{code-cell} ipython3
# uniform_locs_s2 = sorted(uniform_locs, 
#                          key=lambda x: cartesian2polar(x[0]-255,x[1]-255)[1])
uniform_locs_s2 = sorted(uniform_locs, 
                         key=lambda x: -eu_dist(x, (0,255)))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_s2, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)
plt.figure()
plt.imshow(np.ma.masked_less(uniform_prob,1e-10), cmap='Wistia', alpha=0.25, )
ax = plt.gca()
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
plt.tight_layout(); ax.axis('off')
plt.figure(); plt.imshow(np.log2(1+speed), cmap='plasma')
```

```{code-cell} ipython3
plt.imshow(filaments_ms + 5e3*central_prob)
```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             central_locs, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0, batch_size=2)
plt.figure()
plt.imshow(np.ma.masked_less(central_prob,1e-10), cmap='GnBu', alpha=0.25, )
ax = plt.gca()
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75,ax=ax)
plt.tight_layout(); ax.axis('off')
plt.figure(); plt.imshow(np.log2(1+speed), cmap='plasma')
```

```{code-cell} ipython3
# plt.figure()
# plt.imshow(np.ma.masked_less(central_prob,1e-10), cmap='GnBu', alpha=0.25, )
# ax = plt.gca()
# plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75,ax=ax)
# plt.tight_layout(); ax.axis('off')
```

```{code-cell} ipython3
prob_vmax = np.max([np.max(uniform_prob), np.max(central_prob), np.max(periph_prob)])
```

```{code-cell} ipython3
fig, axs = plt.subplots(2,3, figsize=(9,6), gridspec_kw=dict(hspace=0.05, wspace=0.05))

cmap='GnBu'

# -- uniform prob, ordered sampling (center)
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_s, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)

ax = axs[1,0]
ax.imshow(np.ma.masked_less(uniform_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25, )
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)

# -- uniform prob, ordered sampling (periphery)
ax = axs[0,0]
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_s[::-1], 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)

ax.imshow(np.ma.masked_less(uniform_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25)
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)


# -- non-uniform prob (periphery), uniform sampling
ax = axs[0,1]
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             periph_locs, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)
ax.imshow(np.ma.masked_less(periph_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25, )
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)


# -- non-uniform prob (center), uniform sampling
ax = axs[1,1]
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             central_locs, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0, batch_size=2)
ax.imshow(np.ma.masked_less(central_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25, )
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75,ax=ax)

# -- fundus
ax = axs[0,2]
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms+ 1e4*periph_prob, 
                                             phi0_asym, 
                                             periph_locs, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0_asym, 
                                             batch_size=2)
ax.imshow(np.ma.masked_less(periph_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25, )
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)


# -- taxis
ax = axs[1,2]
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_s2, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)

ax.imshow(np.ma.masked_less(uniform_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25, )
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)

for ax in np.ravel(axs):
   ax.axis('off')
fig.tight_layout()
vis.multi_savefig(fig, 'figures/sampling-strategies')
```

```{code-cell} ipython3
# fig.tight_layout()
# vis.multi_savefig(fig, 'figures/sampling-strategies')
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
# plt.figure()
# ax = plt.gca()
# ax.imshow(np.ma.masked_less(uniform_prob,1e-10), vmax=prob_vmax,
#           cmap=cmap, alpha=0.25, )
# plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
```

```{code-cell} ipython3
# tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, phi0, 
#                                              central_locs_s, 
#                                              update_amp=1,
#                                              scaling='linear',
#                                              tm_mask=~phi0, batch_size=2)
# plt.figure()
# plt.imshow(np.ma.masked_less(central_prob,1e-10), cmap='Wistia', alpha=0.25, )
# ax = plt.gca()
# plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75,ax=ax)
# ax.axis('off')
```

```{code-cell} ipython3
# tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, phi0, 
#                                              central_locs_s[::-1], 
#                                              update_amp=1,
#                                              scaling='linear',
#                                              tm_mask=~phi0, batch_size=2)
# plt.figure()
# plt.imshow(np.ma.masked_less(central_prob,1e-10), cmap='Wistia', alpha=0.25, )
# ax = plt.gca()
# plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75,ax=ax)
# plt.tight_layout(); ax.axis('off')
```

```{code-cell} ipython3
def iterative_rbf_sampler(mask, seed=None,  prob_map=None, sigma=12, iterations=500):
    locs = np.array(np.where(mask)).T
    if prob_map is None:
        prob_map = 1.*mask
        
    loc_idx = np.arange(len(locs))
    if seed is None:
        seed = np.random.permutation(locs)[0]
    field = np.zeros(mask.shape)
    lastloc = seed
    acc = [seed]
    for i in tqdm(range(iterations)):
        field = gauss_blob(lastloc, sigma, mask.shape)
        proba = field*prob_map*mask
        proba = proba[mask]/np.sum(proba[mask])
        new_id = np.random.choice(loc_idx,size=1,p=proba)
        newloc = tuple(locs[new_id][0])
        lastloc = newloc
        acc.append(newloc)
    return acc,field
```

```{code-cell} ipython3
bmask_filt[(150,390)]
```

```{code-cell} ipython3
locs_x,field_x = iterative_rbf_sampler(bmask_filt, (150,390),
                                       iterations=4000,
                                       prob_map=periph_prob, sigma=12)
```

```{code-cell} ipython3
plt.imshow(field_x)
```

```{code-cell} ipython3
locs_x[1]
```

```{code-cell} ipython3
locs_x = np.array(locs_x)
```

```{code-cell} ipython3
plt.imshow(field_x)
plt.scatter(locs_x[:,1],locs_x[:,0], c=np.arange(len(locs_x)), s=1, cmap='hot_r')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
len(locs_x)
```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             locs_x[::4], 
                                             update_amp=1,
                                             scaling='power',
                                             speed_gamma=0.1,
                                             tm_mask=~phi0, 
                                             batch_size=3)
plt.figure()
plt.imshow(np.ma.masked_less(uniform_prob,1e-10), cmap='Wistia', alpha=0.25, )
ax = plt.gca()
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75,ax=ax)
plt.tight_layout(); ax.axis('off')
plt.figure(); plt.imshow(np.log2(1+speed), cmap='plasma')
```

```{code-cell} ipython3
p_locs = np.array(np.where(bmask_filt)).T
p_locs = np.random.permutation(p_locs)[:20000]
p_locs = sorted(p_locs, key=lambda x: eu_dist(x,(255,255)))[::-1]
```

```{code-cell} ipython3
np.sum(bmask_filt)
```

```{code-cell} ipython3
reload(iffm)
```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             p_locs[::40], 
                                             update_amp=1,
                                             scaling='power',
                                             speed_gamma=0.01,
                                             tm_mask=~phi0, batch_size=5)
plt.figure()
plt.imshow(np.ma.masked_less(uniform_prob,1e-10), cmap='Wistia', alpha=0.25, )
ax = plt.gca()
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75,ax=ax)
plt.tight_layout(); ax.axis('off')
plt.figure(); plt.imshow(np.log2(1+speed), cmap='plasma')
```

```{code-cell} ipython3
reload(iffm)
```

```{code-cell} ipython3
10**0.001
```

```{code-cell} ipython3
# def iterative_build_tree_sampling(speed0, phi0, seed,
#                                   iterations=100, mask=None, tm_mask=None,
#                                   batch_size=1):
#     speed = speed0.copy()
#     if mask is None:
#         mask = speed > 0
#     ttx0 = skfmm.travel_time(phi0, speed=speed0) 
#     ttx0 = np.ma.filled(ttx0, np.max(ttx0))
#     ttx = ttx0.copy()
#     tree = dict()
#     speed_upd = np.zeros(speed.shape)    
#     fails = []
#     loc = tuple(seed)
#     update_fn = lambda m:m
    
#     all_locs = np.array(np.where(mask)).T
#     loc_idx = np.arange(len(all_locs))
    
#     for j in tqdm(range(iterations)):
#         if ttx[loc] == np.max(ttx):
#             continue
            
#         try_path, finished = iffm.merging_rw_gd(ttx, loc, 
#                                            terminate_mask=tm_mask, 
#                                            pjitter=0.0, 
#                                            nsteps=10000, 
#                                            tree=tree)
#         if finished:
#             apath = iffm.apath_to_root(tree[loc])
#             speed_upd[tuple(apath[:-1,i] for i in (0,1))] += 1
                                                    
#             speed = speed0 + update_fn(speed_upd)
    
#             if (not j%batch_size):    
#                 ttx = skfmm.travel_time(phi0, speed=speed)
#                 ttx = np.ma.filled(ttx, np.max(ttx))
#         else:
#             fails.append(np.array([n.v for n in try_path]))
#             print('not finished for loc', loc)

#         proba = mask*(ttx.max()/(1+ttx))
#         proba = proba[mask]/np.sum(proba[mask])
#         loc = np.random.choice(loc_idx, size=1, replace=False,p=proba)
#         loc = tuple(all_locs[loc[0]])
#     return tree, speed, ttx
```

```{code-cell} ipython3
#np.random.choice(np.arange(10), size=1)[0]
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# tree, speed, ttx = iterative_build_tree_sampling(filaments_ms, 
#                                              phi0, 
#                                              (150,390), 
#                                              mask = bmask_filt,
#                                              iterations=500,
#                                              tm_mask=~phi0, batch_size=2)
# plt.figure()
# plt.imshow(np.ma.masked_less(uniform_prob,1e-10), cmap='Wistia', alpha=0.25, )
# ax = plt.gca()
# plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75,ax=ax)
# plt.tight_layout(); ax.axis('off')
# plt.figure(); plt.imshow(np.log2(0.1+speed), cmap='plasma')
```

```{code-cell} ipython3

```

## Dense seeds and branch diameters

```{code-cell} ipython3
print(np.sum(bmask_filt))
uniform_locs_dense = sample_points(uniform_prob, np.sum(bmask_filt)) 
```

```{code-cell} ipython3
len(uniform_locs_dense)
```

```{code-cell} ipython3
reload(iffm)
```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_dense, 
                                             scaling='log',
                                             tm_mask=~phi0, 
                                             batch_size=1,
                                             batch_size_alpha=1.1)
#plt.figure()
#plt.imshow(np.ma.masked_less(uniform_prob,1e-10), cmap='Wistia', alpha=0.25, )
#ax = plt.gca()
#plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
#plt.tight_layout(); ax.axis('off')
plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='plasma')
```

```{code-cell} ipython3
#
```

```{code-cell} ipython3
# total_count = 0
# acc = []
# for tip in tips:
#     for p in iffm.follow_to_root(tip):
#         loc = tuple(p.v)
#         if not loc in tree:
#             print('point not in tree, but in path:', p)
            
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
# plt.figure()
# plt.imshow(np.ma.masked_less(uniform_prob,1e-10), cmap='Wistia', alpha=0.25, )
# ax = plt.gca()
# plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
# # plt.tight_layout(); ax.axis('off')
# plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='plasma')
```

```{code-cell} ipython3
plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='BuPu')
```

```{code-cell} ipython3
reload(iffm)
```

```{code-cell} ipython3
iffm.assign_diameters(tree, max_diam=12)
```

```{code-cell} ipython3
counts = iffm.count_occurences(tree, speed.shape)
```

```{code-cell} ipython3
# Gx = nx.DiGraph()
# for tip in tqdm(iffm.get_tips(tree)):
#     ap = iffm.apath_to_root(tip)
#     Gx.add_edges_from(list(itt.pairwise(map(tuple, ap[::-1]))))
```

```{code-cell} ipython3
# counts_nx = iffm.count_occurences_nx(Gx, speed.shape)
```

```{code-cell} ipython3
plt.imshow(np.log2(1 + counts),cmap='BuPu')
```

```{code-cell} ipython3
#plt.imshow(np.log(1 + counts_nx),cmap='BuPu')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='BuPu')
```

```{code-cell} ipython3
#plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='BuPu')
```

```{code-cell} ipython3
def make_portrait(tree, shape, min_diam_show=0, fill_soma=False, soma_mask=None):
    if soma_mask is None:
        soma_mask = np.zeros(shape, bool)
    portrait = np.zeros(shape)
    px_locs = np.indices(shape).reshape((2,-1)).T
    ktree = sp.spatial.KDTree(px_locs)
    for loc, n in tqdm(tree.items()):
        diam = n.diam
        if diam >= min_diam_show:
            amp = np.log10(0.1+n.count)
            #amp = diam
            #portrait += amp*gauss_blob(n, diam/2, portrait.shape)
            knns = ktree.query_ball_point(loc, diam/2)
            locs = px_locs[knns]
            for loc_ in locs:
                l = tuple(loc_)
                portrait[l] = np.maximum(portrait[l],amp)
    if fill_soma:
        portrait[soma_mask] = np.percentile(portrait[ndi.binary_dilation(soma_mask)],99)
    #portrait = np.maximum(portrait, np.max(portrait)*gauss_blob((255,255), 10, counts.shape))
    return portrait
    
```

```{code-cell} ipython3
# def make_portrait_nx2(G, shape, min_diam_show=0, fill_soma=False, soma_mask=None):
#     if soma_mask is None:
#         soma_mask = np.zeros(shape, bool)
#     portrait = np.zeros(shape)
#     px_locs = np.indices(shape).reshape((2,-1)).T
#     ktree = sp.spatial.KDTree(px_locs)
#     for n in tqdm(G):
#         diam = G.nodes[n]['diam']
#         if diam >= min_diam_show:
#             amp = np.log10(0.1+G.nodes[n]['count'])
#             #amp = diam
#             #portrait += amp*gauss_blob(n, diam/2, portrait.shape)
#             knns = ktree.query_ball_point(n, diam/2)
#             locs = px_locs[knns]
#             for loc in locs:
#                 l = tuple(loc)
#                 portrait[l] = np.maximum(portrait[l],amp)
#     if fill_soma:
#         portrait[soma_mask] = np.percentile(portrait[ndi.binary_dilation(soma_mask)],99)
#     #portrait = np.maximum(portrait, np.max(portrait)*gauss_blob((255,255), 10, counts.shape))
#     return portrait
    
```

```{code-cell} ipython3
iffm.assign_diameters(tree, max_diam=12, gamma=1.25)
portrait = make_portrait(tree, speed.shape, min_diam_show=0.02,
                         fill_soma=True,soma_mask=~phi0)
```

```{code-cell} ipython3
plt.imshow(portrait, cmap='gray_r')
```

```{code-cell} ipython3
reload(iffm)
```

```{code-cell} ipython3
print(len(tree), len(iffm.get_tips(tree)))
```

```{code-cell} ipython3
twigs = []
for tip in iffm.get_tips(tree):
    twig = iffm.prune_twig(tip, min_length=10, max_count_diff=5)
    if len(twig):
        twigs.append(twig)
```

```{code-cell} ipython3
len(twigs), len(iffm.get_tips(tree)), len(tree)
```

```{code-cell} ipython3
iffm.assign_diameters(tree, max_diam=12, gamma=1.25)
portrait2 = make_portrait(tree, speed.shape, min_diam_show=0.02)
plt.imshow(portrait2, cmap='gray_r')
```

```{code-cell} ipython3
twig_tree = dict()
for twig in twigs:
    for p in twig:
        twig_tree[tuple(p.v)] = p
```

```{code-cell} ipython3
clean_tree = dict()
for loc, p in tree.items():
    if len(iffm.follow_to_root(p))>10:
        if not loc in twig_tree:
            clean_tree[loc] = p
```

```{code-cell} ipython3
len(twig_tree), len(clean_tree)
```

```{code-cell} ipython3
portrait3 = make_portrait(twig_tree, speed.shape, min_diam_show=0.01)
plt.imshow(portrait3, cmap='gray_r')
```

```{code-cell} ipython3
plt.imshow(iffm.count_occurences(twig_tree, speed.shape))
```

```{code-cell} ipython3
iffm.assign_diameters(clean_tree, max_diam=12, gamma=1.25)
portrait_clean = make_portrait(clean_tree, speed.shape,
                               fill_soma=True,
                               soma_mask = ~phi0,
                               min_diam_show=0.01)
plt.imshow(portrait_clean, cmap='gray_r')
```

```{code-cell} ipython3
plt.imshow(np.dstack([portrait, portrait_clean,portrait_clean]))
```

```{code-cell} ipython3
plt.imshow(iffm.count_occurences(clean_tree, speed.shape)**0.25)
```

```{code-cell} ipython3
# iffm.count_occurences_nx(Gx, speed.shape)
# iffm.assign_diameters_nx(Gx, min_diam=0.01, gamma=1, max_diam=6)
```

```{code-cell} ipython3
# portrait2 = make_portrait_nx2(Gx, speed.shape, min_diam_show=0.02, 
#                               fill_soma=False, soma_mask=~phi0)
```

```{code-cell} ipython3
# plt.imshow(portrait2,cmap='gray_r')
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

## Multi-cellular network

```{code-cell} ipython3
# x.reshape((2,-1)).T # alternative ;)

def grid2points(X,Y):
    #return np.array(list(zip(np.ravel(X),np.ravel(Y))))
    return np.array([np.ravel(X),np.ravel(Y)]).T

def extract_edge_lengths(locs, tri):
    return np.array([np.sum((locs[edge[0]] - locs[edge[1]])**2)**0.5 for edge in tri.edges])
```

```{code-cell} ipython3
#np.arange(5)%2
```

```{code-cell} ipython3
def tri_grid(nx, ny):
    xx,yy = np.mgrid[:nx,:ny]
    print(xx.shape)
    xx1 = xx + (0.5 * (np.arange(ny)%2))[None,:]
    yy1 = yy*np.cos(np.pi/6)
    return xx1, yy1
             
    
```

```{code-cell} ipython3
tri_locs = grid2points(*tri_grid(4,5))
```

```{code-cell} ipython3
np.min(tri_locs,0),np.max(tri_locs,0)
```

```{code-cell} ipython3
plt.plot(tri_locs[:,0], tri_locs[:,1], '.')
plt.axis('equal')
```

```{code-cell} ipython3
tri_locsr = np.maximum(tri_locs + np.clip(np.random.randn(*tri_locs.shape)*0.25, -0.25,0.25),0)
```

```{code-cell} ipython3
tri_locsr.min(0)
```

```{code-cell} ipython3
plt.plot(tri_locsr[:,0], tri_locsr[:,1], '.')
plt.axis('equal')
```

```{code-cell} ipython3
# tri = plt.matplotlib.tri.Triangulation(*tri_locs.T)
# edge_lengths = extract_edge_lengths(tri_locs, tri)

# #plt.hist(edge_lengths, 25, color='gray',ec='k');
# sns.histplot(edge_lengths, bins=25, color='gray', kde=True)
```

```{code-cell} ipython3
#len(edge_lengths)
```

```{code-cell} ipython3
#np.unique(np.round(edge_lengths,3))
```

```{code-cell} ipython3
# seeds =  np.array([(255,255), 
#                    (64,64), 
#                    (300, 64),
#                    (300, 440),
#                    (100,450), (450,100), (450,450)])

#seeds = np.random.randint(10,500, size=(50,2))
#seeds = [(255,255)] +  [s for s in seeds 
#                        if eu_dist(s,(255,255)) > 250]


seeds = tri_locsr[:,::-1]*150
seeds = seeds[(seeds[:,0]>10)*(seeds[:,1]>10)*(seeds[:,0]<500)*(seeds[:,1]<500)]

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
valid_inits = kdt.query_ball_point(px_locs, 100,return_length=True)
```

```{code-cell} ipython3
somata_mask = kdt.query_ball_point(px_locs, 5,return_length=True).reshape(speed.shape)>0
```

```{code-cell} ipython3
len(valid_inits), len(px_locs)
```

```{code-cell} ipython3
plt.imshow(somata_mask)
```

```{code-cell} ipython3
#phi0 = np.zeros(field.shape)
#phi0[255,255] = 1
#phi0[250,150] = phi0[255,430] = 1
#phi0 = ndi.binary_dilation(phi0,iterations=2)
phi0 = somata_mask
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
boundary_mask = ttx < np.percentile(ttx,50)

plt.figure()
plt.imshow(ttx,cmap='BuPu', vmax=np.percentile(ttx[ttx<np.max(ttx)],99)); plt.colorbar()
#plt.contour(ttx, levels=[np.percentile(ttx, 25)], colors='c')
#plt.contour(boundary_mask, levels=[0.5], colors='r',linewidths=0.75)
```

```{code-cell} ipython3
#np.array(ttx)
```

```{code-cell} ipython3
#plt.imshow(boundary_mask)
```

```{code-cell} ipython3
tm_mask = ndi.binary_dilation(ttx<=np.percentile(ttx,0.5),iterations=1)
plt.imshow(tm_mask)
```

---

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
plt.imshow(uc.clip_outliers(ttx), cmap='gray')
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
plt.imshow(uc.clip_outliers(ttx), cmap='gray')
plt.plot(path1[:,1], path1[:,0],'r')
```

```{code-cell} ipython3
# plt.figure()
# plt.plot(np.diff(ttx[path1[:,0],path1[:,1]]))
# plt.plot(np.diff(ttxbumps[path1[:,0],path1[:,1]]))
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
plt.imshow(boundary_mask)
```

```{code-cell} ipython3
#init_pts = np.array([(i,j) for i in range(50,450) for j in range(50,450) 
#                    if boundary_mask[(i,j)]])
#                    #if eu_dist((i,j),(255,255)) < 200])

init_pts = uc.masks.mask2points(boundary_mask)

dists,_ = kdt.query(init_pts, k=1)

#Npts = 5000
init_pts_dense = np.random.permutation(init_pts)

init_pts_sorted = np.random.permutation(init_pts)
dists,_ = kdt.query(init_pts, k=1)


#init_pts = sorted(init_pts, key=lambda p: eu_dist(p, (255,255)), reverse=True)
init_pts = init_pts[np.argsort(dists)[::-1]]
#init_pts_dense = sorted(init_pts_dense, key=lambda p: eu_dist(p, (255,255)), reverse=True)
```

```{code-cell} ipython3
#dists
```

```{code-cell} ipython3

```

```{code-cell} ipython3
alpha**j
```

```{code-cell} ipython3
len(tree), len(fails)
```

```{code-cell} ipython3
plt.imshow(speed); plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(speed_corr, interpolation='nearest'); plt.colorbar()
```

```{code-cell} ipython3
#plt.imshow(ttx, interpolation='nearest', vmax=np.percentile(ttx[ttx<np.max(ttx)],99)); plt.colorbar()
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
plt.imshow(speed_corr**0.5); plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(np.log2(1 + speed_corr)); plt.colorbar()
```

```{code-cell} ipython3
#plt.imshow(speed**2); plt.colorbar()
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
plt.imshow(np.log(1 + speed)); plt.colorbar()
```

```{code-cell} ipython3
#ttx2 = skfmm.travel_time(phi0, speed=speed0/5 + np.log10(1+counts))
ttx2 = skfmm.travel_time(phi0, speed=speed)
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
kdt_x = sp.spatial.KDTree(px_locs)
```

```{code-cell} ipython3
#len(kdt_x)
```

```{code-cell} ipython3
%time ids =  kdt_x.query_ball_point((255,255),1)
```

```{code-cell} ipython3
px_locs[ids]
```

```{code-cell} ipython3
#portrait[*px_locs[ids].T]
```

```{code-cell} ipython3
def make_portrait_nx(G, shape, min_diam_show=0, fill_soma=False, do_threshold=False):
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
    if fill_soma:
        portrait[tm_mask] = np.percentile(portrait[ndi.binary_dilation(tm_mask)],99)
    #portrait = np.maximum(portrait, np.max(portrait)*gauss_blob((255,255), 10, counts.shape))
    return portrait
    
```

```{code-cell} ipython3
def make_portrait_nx2(G, shape, min_diam_show=0, fill_soma=False):
    portrait = np.zeros(shape)
    px_locs = np.indices(shape).reshape((2,-1)).T
    ktree = sp.spatial.KDTree(px_locs)
    for n in tqdm(G):
        diam = G.nodes[n]['diam']
        if diam >= min_diam_show:
            amp = np.log10(0.1+G.nodes[n]['count'])
            #amp = diam
            #portrait += amp*gauss_blob(n, diam/2, portrait.shape)
            knns = ktree.query_ball_point(n, diam/2)
            locs = px_locs[knns]
            for loc in locs:
                l = tuple(loc)
                portrait[l] = np.maximum(portrait[l],amp)
    if fill_soma:
        portrait[tm_mask] = np.percentile(portrait[ndi.binary_dilation(tm_mask)],99)
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
assign_diameters_nx(Gx, min_diam=0.1, gamma=1.1, max_diam=6)

%time portrait = make_portrait_nx2(Gx, speed.shape, fill_soma=True)

plt.imshow(portrait, cmap='plasma'); plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(portrait,cmap='gray'); plt.colorbar()
```

```{code-cell} ipython3
# x = np.linspace(0.01, 10, 200)
# for gamma in [ 0.5, 1., 1.5, 2]:
#     plt.plot(x, (2*x**gamma)**(1/gamma), label=gamma)
# plt.legend()
```

```{code-cell} ipython3
assign_diameters_nx(Gx2, min_diam=0.01, gamma=1.2, max_diam=16)
portrait2 = make_portrait_nx2(Gx2, speed.shape, min_diam_show=0.05, fill_soma=True)
```

```{code-cell} ipython3
plt.imshow(portrait2, cmap='plasma'); plt.colorbar()
```

```{code-cell} ipython3
plt.imshow(portrait2, cmap='gray_r')
plt.axis('off')
```

```{code-cell} ipython3
plt.imshow(portrait2 > 0.0, interpolation='nearest')
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
