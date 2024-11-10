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

for ax,tt in zip(axs[2],ttms,):
    tree, fails = iffm.build_tree(tt, init_pts, tm_mask =~phi0)
    if len(tree):
        plot_tree(tree,ax,random_colors=False,lw=0.75,linecolor='k')
        ax.axis([0,512,512,0])
        ax.axis('off')


for ax,letter in zip(np.ravel(axs), 'abcde'):
    ax.text(10,10, f'({letter})', backgroundcolor='w') 


plt.tight_layout()
vis.multi_savefig(fig, 'figures/speed-field-effect')
```

## Figure 2. Effect of speed field update

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
                                                 tm_mask=~phi0, batch_size=1)
    plot_tree(tree, axs[0,col], random_colors=False, mfc='k', linecolor='k', lw=0.75)
    axs[1,col].imshow(uc.clip_outliers(np.log2(1+speed)), cmap='viridis')
    show_tt_map(ttx, ax=axs[2,col])
    axs[0,col].axis([0,512, 512,0])



for ax,letter in zip(np.ravel(axs), 'abcd'):
    ax.text(10,10, f'({letter})', backgroundcolor='w') 

for ax in np.ravel(axs):    
    ax.axis('off')

plt.tight_layout()
vis.multi_savefig(fig, 'figures/ifmm-updates')
```

## Figure 3. Sampling strategies

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
# plt.figure()
# plt.imshow(np.ma.masked_less(periph_prob,1e-10), cmap='cool', alpha=0.25, )
# ax = plt.gca()
# plot_tree(tree, random_colors=False, mfc='r', linecolor='forestgreen', lw=1, max_lw=1, ax=ax)

# for p in periph_locs_dense:
#     if np.random.rand() < 0.15:
#         color = np.clip(1 - np.random.rand(3)**2 + (0.3,0.25,0.75),0,1)
#         angle = np.random.uniform(-60,60)
#         size = np.random.uniform(2,16)
#         zorder=np.random.uniform(0,20)
#         ax.plot(p[1],p[0], color=color, marker=(6, 2, angle), alpha=0.95,
#                 ms=size,zorder=zorder)
    
#     elif np.random.rand() < 0.025:
#         ax.plot(p[1],p[0], color='r', marker='o', ms=5, alpha=0.5)
        
# plt.tight_layout(); ax.axis('off')
# plt.savefig('/tmp/ny-ring.png', )
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
ax = axs[0,0]
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_s, 
                                             update_amp=1,
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)

ax.imshow(np.ma.masked_less(uniform_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25, )
plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
ax.text(10,10, '(a)')



# -- uniform prob, ordered sampling (periphery)
ax = axs[0,1]
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
ax.text(10,10, '(b)')

# -- taxis (uniform prob, ordered sampling, point)
ax = axs[0,2]
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
ax.text(10,10, '(c)')


# -- non-uniform prob (center), uniform sampling
ax = axs[1,0]
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
ax.text(10,10, '(d)')

# -- non-uniform prob (periphery), uniform sampling
ax = axs[1,1]
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
ax.text(10,10, '(e)')


# -- fundus
ax = axs[1,2]
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
ax.text(10,10, '(f)')


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

## Figures 4-5. Dense seeds and branch diameters

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
def make_portrait(tree, shape, min_diam_show=0, fill_soma=False, 
                  soma_mask=None,verbose=False):
    if soma_mask is None:
        soma_mask = np.zeros(shape, bool)
    portrait = np.zeros(shape)
    ndim = len(shape)
    px_locs = np.indices(shape).reshape((ndim,-1)).T
    ktree = sp.spatial.KDTree(px_locs)
    for loc, n in tqdm(tree.items(),disable=not verbose):
        diam = n.diam
        if diam >= min_diam_show:
            amp = np.log10(0.1+n.count)
            #amp = diam
            #portrait += amp*gauss_blob(n, diam/2, portrait.shape)
            knns = ktree.query_ball_point(loc, diam/2)
            locs = px_locs[knns]
            for loc_ in locs:
                dist = eu_dist(loc, loc_)
                ampr = amp*np.exp(-dist**2/(diam**2/4))
                l = tuple(loc_)
                portrait[l] = np.maximum(portrait[l],ampr)
    if fill_soma:
        portrait[soma_mask] = np.percentile(portrait[ndi.binary_dilation(soma_mask)],99)
    #portrait = np.maximum(portrait, np.max(portrait)*gauss_blob((255,255), 10, counts.shape))
    return portrait
    
```

```{code-cell} ipython3
print(np.sum(bmask_filt))
uniform_locs_dense = sample_points(uniform_prob, np.sum(bmask_filt)) 
```

```{code-cell} ipython3
len(uniform_locs_dense)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
reload(iffm)
```

```{code-cell} ipython3
tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_dense, 
                                             scaling='linear',
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
reload(iffm)
```

```{code-cell} ipython3
Ntotal = len(uniform_locs_dense)
```

```{code-cell} ipython3
#pts_fractions = 0.002*3**np.arange(6)
#pts_fractions = 0.002*4**np.arange(5)
pts_fractions = [0.002, 0.008, 0.03, 0.13, 0.5]
pts_fractions
```

```{code-cell} ipython3
#uniform_locs_dense
```

```{code-cell} ipython3
tree_kw = dict(scaling='linear', tm_mask=~phi0, batch_size=1, batch_size_alpha=1.1,)
                                            

params = [
    tree_kw,
    tree_kw | dict(do_phi0_update=True, max_count_phi0=32),
    tree_kw | dict(do_phi0_update=True, max_count_phi0=32),
    tree_kw | dict(do_phi0_update=True, max_count_phi0=32)]

pts_samplers = [
    lambda n: uniform_locs_dense[:round(int(n))],
    lambda n: uc.scramble.local_jitter(uniform_locs_dense)[:round(int(n))],
    lambda n: uc.scramble.local_jitter(np.array(sorted(uniform_locs_dense[:round(int(n))],
                                                       key=lambda x: -eu_dist(x, (255,255))))),
    lambda n: uc.scramble.local_jitter(np.array(sorted(uniform_locs_dense[:round(int(n))],
                                                       key=lambda x: eu_dist(x, (255,255))))),
]
    
    
    
```

```{code-cell} ipython3
%%time 

fig1, axs = plt.subplots(len(params)+1,len(pts_fractions), figsize=(9,9),
                        sharex='col', sharey='col',
                        gridspec_kw=dict(hspace=0.05, wspace=0.05),
                       )

final_trees = []
labels = ['CR', 'VR', 'VP','VC']
colors= ["#005f73", "#0a9396", "#ca6702", "#ae2012"]


for j, (pset, sampler) in enumerate(zip(tqdm(params,desc='params'), pts_samplers)):
    tp_ratios = []
    for k,frac in enumerate(pts_fractions):
        n_pts = int(round(Ntotal*frac))
        pts = sampler(n_pts)
        
        if j==0:
            ax = axs[0,k]
            ax.plot(pts[:,0], pts[:,1], 'k,')
            ax.axis('square')
            ax.axis([0,512,512,0])
            ax.axis('off')
            title = f'{100*frac:1.1f}%' if frac<0.01 else f'{100*frac:1.0f}%'
            ax.set_title(title)

        ax = axs[j+1,k]
        if k == 0:
            ax.text(10,10, labels[j], color=colors[j], va='top')
        
        tree, speed, ttx = iffm.iterative_build_tree(filaments_ms,phi0,pts, **pset)
        ttxf = skfmm.travel_time(phi0, speed=speed)
        counts = iffm.count_occurences(tree, speed.shape)
        
        if frac < 0.005:
            gamma = 1.75
        elif gamma < 0.1:
            gamma = 2
        else:
            gamma = 2.25
        
        iffm.assign_diameters(tree,min_diam=0.25,gamma=gamma,max_diam=9)
        portrait = make_portrait(tree, speed.shape, fill_soma=True, soma_mask=~phi0)
        top_p = 95 if frac < 0.01 else 99.5
        vmin,vmax=np.percentile(portrait[portrait>0],(1,top_p))
        #print(vmin,vmax)
        ax.imshow(portrait, vmin=0,vmax=vmax, cmap='BuPu')
        ax.plot(255,255, 'o', color='violet', mfc='none', ms=10)
        ax.axis('off')
        tips = iffm.get_tips(tree)
        tip_source_ratio = len(tips)/len(pts)
        tp_ratios.append(tip_source_ratio)
        print('---', j,frac,'tip/source ratio:',tip_source_ratio)
    final_trees.append((tree, ttxf, pts, tips, tp_ratios))
plt.tight_layout()
vis.multi_savefig(fig1, 'figures/updated-phi0-patterns')
```

```{code-cell} ipython3
fig1
```

```{code-cell} ipython3
fig2 = plt.figure(figsize=(3,3))

#colors= ["#005f73", "#0a9396", "#ca6702", "#ae2012"]
#labels = ['CR', 'VR', 'VP','VC']

for lab,coll,color in zip(labels,final_trees,colors):
    tpr = coll[-1]
    plt.plot(pts_fractions, tpr, 'o-', label=lab,mfc='none',color=color)
plt.legend(loc=(1.05, 0.5), ncol=1)
ax = plt.gca()
ax.set(xscale='log', xlabel='seed density', ylabel='tip fraction')
vis.lean_axes(ax)
vis.multi_savefig(fig2, 'figures/updated-phi0-tip-fractions')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
reload(vis)
```

```{code-cell} ipython3
#plt.figure()
fig3, axs = plt.subplots(2,4, sharey='row', sharex='row', 
                        figsize=(9,6), 
                        gridspec_kw=dict(wspace=0.25, hspace=0.5))

for j,lab,coll,color in zip(range(100), labels,final_trees,colors):
    tree, ttxf, pts, tips, tp_ratios = coll
    atips = np.array([t.v for t in tips])
    acc = []
    for t in atips:
        p = iffm.apath_to_root(tree[tuple(t)])
        acc.append((len(p), ttxf[tuple(t)]))
    acc = np.array(acc)
    ax = axs[0,j]
    ax.hist(ttxf[*atips.T],50, log=False, color=color, density=True, label=lab,lw=1.5);
    ax.set_xlabel('travel time, a.u.')
    #ax.legend(frameon=False)
    ax.text(300, 0.02, lab, color=color)
    ax = axs[1,j]
    ax.text(600, 350, lab, color=color)
    #ax.plot(acc[:,0], acc[:,1], '.',mfc='w',alpha=1,color='gray', label=lab)
    ax.hexbin(acc[:,0], acc[:,1], cmap='Blues', bins='log', rasterized=True)
    #plt.plot(acc[:,0], acc[:,1], ',',alpha=0.1, label=lab)
    ax.set(xlabel='path length, a.u.')

for ax in np.ravel(axs):
    vis.lean_axes(ax)
    
axs[0,0].set_ylabel('Prob. density')
axs[1,0].set_ylabel('travel time, a.u.')
#plt.legend()
vis.multi_savefig(fig3, 'figures/updated-phi0-travel-times')
```

```{code-cell} ipython3
#plt.hexbin?
```

```{code-cell} ipython3
# plt.figure()

# for lab,coll in zip(labels,final_trees):
#     tree, ttxf, pts, tips, tp_ratios = coll
#     atips = np.array([t.v for t in tips])
#     plt.hist(ttxf[*atips.T],50, log=False, range=(0,200), density=True, 
#              label=lab,
#              histtype='step',lw=1.5);

# plt.legend()
```

```{code-cell} ipython3

```

```{code-cell} ipython3
tree_kw | dict(do_phi0_update=True, max_count_phi0=32)
```

```{code-cell} ipython3
seeds = uniform_locs_dense[:int(round(Ntotal*0.03))]

tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             seeds, 
                                             **(tree_kw | dict(do_phi0_update=True, max_count_phi0=32)))
                                             
#plt.figure()
#plt.imshow(np.ma.masked_less(uniform_prob,1e-10), cmap='Wistia', alpha=0.25, )
#ax = plt.gca()
#plot_tree(tree, random_colors=False, mfc='k', linecolor='k', lw=0.75, ax=ax)
#plt.tight_layout(); ax.axis('off')
#plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='plasma')
#plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='BuPu')
counts = iffm.count_occurences(tree, speed.shape)
#plt.figure(); plt.imshow(np.log2(1+counts), interpolation='nearest', cmap='BuPu')
iffm.assign_diameters(tree,min_diam=0.25,gamma=2.25,max_diam=9)
portrait = make_portrait(tree, speed.shape, fill_soma=True, soma_mask=~phi0)

#plt.plot(255,255,'o',color='purple',mfc='none')
#plt.plot(255,255,'o',color='violet',mfc='none',ms=10)
plt.axis('off')
#plt.colorbar()
```

```{code-cell} ipython3
plt.figure(figsize=(3,3))
plt.imshow(uc.clip_outliers(portrait**0.5), vmin=0,  cmap='BuPu')
ax = plt.gca()
tips = np.array([t.v for t in iffm.get_tips(tree)])

plt.plot(seeds[:,1], seeds[:,0], color='k', ls='', marker='.',ms=8,mfc='none')
plt.plot(tips[:,1], tips[:,0], 'r.',ms=4)
ax.axis('off')
plt.tight_layout()
plt.axis([250,350, 150,50])
vis.multi_savefig(plt.gcf(), 'figures/updated-phi0-tip-fractions-illustration')
```

```{code-cell} ipython3
len(tips)/len(seeds)
```

```{code-cell} ipython3
#tips
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# plt.imshow(uc.clip_outliers(portrait),  cmap='BuPu'); plt.colorbar()
# plt.plot(255,255,'o',color='violet',mfc='none',ms=10)
# plt.axis('off')
```

```{code-cell} ipython3
# iffm.assign_diameters_logcounts(tree,min_diam=1,max_diam=16)

# portrait = make_portrait(tree, speed.shape, fill_soma=True, soma_mask=~phi0)

# plt.imshow(portrait, cmap='BuPu')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
#plot_tree(tree, random_colors=False)
```

```{code-cell} ipython3
#plt.figure(); plt.imshow(counts, interpolation='nearest', cmap='BuPu')
```

```{code-cell} ipython3
# reload(iffm)
```

```{code-cell} ipython3
# tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
#                                              phi0, 
#                                              uniform_locs_dense[:128], 
#                                              scaling='linear',
#                                              tm_mask=~phi0, 
#                                              batch_size=1,
#                                              batch_size_alpha=1.1,
#                                              do_phi0_update=True,
#                                              max_count_phi0=32)
# #plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='plasma')
# counts = iffm.count_occurences(tree, speed.shape)
# plt.figure(); plt.imshow(np.log2(1+counts),  cmap='BuPu'); plt.axis('off')
# plt.plot(255,255,'o',color='purple',mfc='none')
```

```{code-cell} ipython3
#import ucats as uc
```

```{code-cell} ipython3
#uc.scramble.local_jitter(list(np.arange(100)))
```

```{code-cell} ipython3
#2**16
```

```{code-cell} ipython3
#2**np.arange(8,17,2)
```

```{code-cell} ipython3
# tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
#                                              phi0, 
#                                              uc.scramble.local_jitter(
#                                                  np.array(
#                                                      sorted(uniform_locs_dense[:2**8],
#                                                             key=lambda x: -eu_dist(x, (255,255))))),
#                                              scaling='linear',
#                                              tm_mask=~phi0, 
#                                              batch_size=1,
#                                              batch_size_alpha=1.1,
#                                              do_phi0_update=True,
#                                              max_count_phi0=32)
# #plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='plasma')
# #plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='BuPu')
# counts = iffm.count_occurences(tree, speed.shape)
# plt.figure(); 
# plt.imshow(np.log2(1+counts), interpolation='nearest',   cmap='BuPu'); plt.axis('off')
# plt.plot(255,255,'o',color='purple',mfc='none')
```

```{code-cell} ipython3
# plt.figure(); plt.imshow(np.log2(1+counts), interpolation='nearest', cmap='BuPu'); plt.axis('off')
# plt.plot(255,255,'o',color='purple',mfc='none')
# plt.colorbar()
```

```{code-cell} ipython3
# tree, speed, ttx = iffm.iterative_build_tree(filaments_ms, 
#                                              phi0, 
#                                              sorted(uniform_locs_dense[:250], 
#                                                     key=lambda x: eu_dist(x, (255,255))),
#                                              scaling='linear',
#                                              tm_mask=~phi0, 
#                                              batch_size=1,
#                                              batch_size_alpha=1.1,
#                                              do_phi0_update=True,
#                                              max_count_phi0=32)
# #plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='plasma')
# #plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='BuPu')
# counts = iffm.count_occurences(tree, speed.shape)
# plt.figure(); plt.imshow(np.log2(1+counts), interpolation='nearest', cmap='BuPu'); plt.axis('off')
# plt.plot(255,255,'o',color='purple',mfc='none')
```

```{code-cell} ipython3
# iffm.assign_diameters(tree,min_diam=0.25,gamma=2.5,max_diam=9)

# portrait = make_portrait(tree, speed.shape, fill_soma=True, soma_mask=~phi0)

# plt.imshow(portrait,  cmap='BuPu')
```

**NB** make diameters proportional to log counts?

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
#plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='BuPu')
```

```{code-cell} ipython3
#reload(iffm)
```

```{code-cell} ipython3
#iffm.assign_diameters(tree, max_diam=12)
```

```{code-cell} ipython3
#counts = iffm.count_occurences(tree, speed.shape)
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
#plt.imshow(np.log2(1 + counts),cmap='BuPu')
```

```{code-cell} ipython3
#plt.imshow(np.log(1 + counts_nx),cmap='BuPu')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
#plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='BuPu')
```

```{code-cell} ipython3
#plt.figure(); plt.imshow(np.log2(1+speed), interpolation='nearest', cmap='BuPu')
```

```{code-cell} ipython3

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
# iffm.assign_diameters(tree, max_diam=12, gamma=1.25)
# portrait = make_portrait(tree, speed.shape, min_diam_show=0.02,
#                          fill_soma=True,soma_mask=~phi0)
```

```{code-cell} ipython3
#plt.imshow(portrait, cmap='gray_r')
```

```{code-cell} ipython3
#reload(iffm)
```

```{code-cell} ipython3
#print(len(tree), len(iffm.get_tips(tree)))
```

```{code-cell} ipython3
# twigs = []
# for tip in iffm.get_tips(tree):
#     twig = iffm.prune_twig(tip, min_length=10, max_count_diff=5)
#     if len(twig):
#         twigs.append(twig)
```

```{code-cell} ipython3
#len(twigs), len(iffm.get_tips(tree)), len(tree)
```

```{code-cell} ipython3
# iffm.assign_diameters(tree, max_diam=12, gamma=1.25)
# portrait2 = make_portrait(tree, speed.shape, min_diam_show=0.02)
# plt.imshow(portrait2, cmap='gray_r')
```

```{code-cell} ipython3
# twig_tree = dict()
# for twig in twigs:
#     for p in twig:
#         twig_tree[tuple(p.v)] = p
```

```{code-cell} ipython3
# clean_tree = dict()
# for loc, p in tree.items():
#     if len(iffm.follow_to_root(p))>10:
#         if not loc in twig_tree:
#             clean_tree[loc] = p
```

```{code-cell} ipython3
#len(twig_tree), len(clean_tree)
```

```{code-cell} ipython3
# portrait3 = make_portrait(twig_tree, speed.shape, min_diam_show=0.01)
# plt.imshow(portrait3, cmap='gray_r')
```

```{code-cell} ipython3
#plt.imshow(iffm.count_occurences(twig_tree, speed.shape))
```

```{code-cell} ipython3
# iffm.assign_diameters(clean_tree, max_diam=12, gamma=1.25)
# portrait_clean = make_portrait(clean_tree, speed.shape,
#                                fill_soma=True,
#                                soma_mask = ~phi0,
#                                min_diam_show=0.01)
# plt.imshow(portrait_clean, cmap='gray_r')
```

```{code-cell} ipython3
#plt.imshow(np.dstack([portrait, portrait_clean,portrait_clean]))
```

```{code-cell} ipython3
#plt.imshow(iffm.count_occurences(clean_tree, speed.shape)**0.25)
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

## Figure 6a. Multi-cellular network

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
tri_locs = grid2points(*tri_grid(3*3,3*2))
```

```{code-cell} ipython3
np.min(tri_locs,0),np.max(tri_locs,0)
```

```{code-cell} ipython3
plt.plot(tri_locs[:,0], tri_locs[:,1], '.')
plt.axis('equal')
```

```{code-cell} ipython3
tri_locsr = np.maximum(tri_locs + np.clip(np.random.randn(*tri_locs.shape)*0.15, -0.25,0.25),0)
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
field_shape_w = (512, 256*3)

filaments_ms_w = uc.utils.rescale(sum(s**2*morpho.sato2d(np.random.randn(*field_shape_w), s)
                                      for s in (1.5, 3, 6, 12)))
speed_w = filaments_ms_w
```

```{code-cell} ipython3
seeds[seeds[:,0] < 50]
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


seeds = tri_locsr[:,::-1]*100
seeds = seeds[(seeds[:,0]>10)*(seeds[:,1]>10)*(seeds[:,0]<500)*(seeds[:,1]<750)]

#seeds[seeds[:,0] < 50] += (50, 0)

seeds = np.array(seeds)
#seeds = np.concatenate(([(255,255)],seeds))
#seeds += np.random.randint(-16,16, size=(len(seeds),2))
plt.imshow(speed_w)
plt.plot(seeds[:,1], seeds[:,0], 'r.')
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
kdt = sp.spatial.KDTree(seeds)

px_locs = (np.indices(speed_w.shape)
           .reshape((2,-1))
           .T)

labels = kdt.query(px_locs)[1] + 1
labels = labels.reshape(speed_w.shape)
plt.imshow(labels)
```

```{code-cell} ipython3
valid_inits = kdt.query_ball_point(px_locs, 100,return_length=True)
```

```{code-cell} ipython3
somata_mask = kdt.query_ball_point(px_locs, 5,return_length=True).reshape(speed_w.shape)>0
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
phi0_w = somata_mask
plt.imshow(phi0_w, interpolation='nearest')

phi0_w = ~phi0_w
```

```{code-cell} ipython3
ttx = skfmm.travel_time(phi0_w, speed=speed_w)
#ttx = ttx*(ttx>0)
#ttx[ttx.mask] = np.max(ttx)
#ttx = np.array(ttx)
ttx = np.ma.filled(ttx,np.max(ttx))

#boundary_mask = labels == 1
boundary_mask = ttx < np.percentile(ttx,85)

plt.figure()
plt.imshow(ttx,cmap='BuPu', vmax=np.percentile(ttx[ttx<np.max(ttx)],99)); plt.colorbar()
#plt.contour(ttx, levels=[np.percentile(ttx, 25)], colors='c')
plt.contour(boundary_mask, levels=[0.5], colors='r',linewidths=0.75)
```

```{code-cell} ipython3
#np.array(ttx)
```

```{code-cell} ipython3
#plt.imshow(boundary_mask)
```

```{code-cell} ipython3
tm_mask_w = ndi.binary_dilation(ttx<=np.percentile(ttx,0.5),iterations=1)
plt.imshow(tm_mask_w)
```

```{code-cell} ipython3
uniform_prob_w = ndi.gaussian_filter(np.ones(field_shape_w)*boundary_mask,5)
uniform_prob_w /= uniform_prob_w.sum()
plt.imshow(uniform_prob_w + tm_mask_w)
```

```{code-cell} ipython3
seeds_w = sample_points(uniform_prob_w, np.sum(uniform_prob_w>0)) 
```

```{code-cell} ipython3
len(seeds_w)*0.1
```

```{code-cell} ipython3
tree_w, speedx_w, ttx_w = iffm.iterative_build_tree(speed_w, 
                                             phi0_w, 
                                             seeds_w[:35600], 
                                             scaling='linear',
                                             tm_mask=tm_mask_w, 
                                             batch_size=1,
                                             batch_size_alpha=1.1)
plt.figure(); plt.imshow(np.log2(1+speedx_w), interpolation='nearest', cmap='PuBu')
counts = iffm.count_occurences(tree_w, speedx_w.shape)
plt.figure(); plt.imshow(np.log2(1+counts), interpolation='nearest', cmap='BuPu')

iffm.assign_diameters(tree_w,min_diam=0.25,gamma=2.25,max_diam=12)
portrait_w = make_portrait(tree_w, speedx_w.shape, fill_soma=True, soma_mask=tm_mask_w)
plt.imshow(portrait_w,  cmap='BuPu')
#plt.plot(255,255,'o',color='purple',mfc='none')
#plt.plot(255,255,'o',color='violet',mfc='none',ms=10)

plt.axis('off')
#plt.colorbar()
```

```{code-cell} ipython3
fig = plt.figure()

iffm.assign_diameters(tree_w,min_diam=0.25,gamma=1.85,max_diam=12)
portrait_w = make_portrait(tree_w, speedx_w.shape, fill_soma=True, soma_mask=tm_mask_w)
plt.imshow(portrait_w,  cmap='BuPu')
plt.axis('off')
vis.multi_savefig(fig, 'figures/network-example')
```

```{code-cell} ipython3
soma_labels,ns = ndi.label(tm_mask_w)
ksoma = np.random.randint(ns)+1
random_soma = soma_labels == ksoma
plt.imshow(random_soma)
```

```{code-cell} ipython3
ttx_fin = skfmm.travel_time(~random_soma, portrait_w+0.01)
#ttx_fin = ttx_fin.filled(ttx_fin.max())
```

```{code-cell} ipython3
fig = plt.figure()
plt.imshow(portrait_w,  cmap='BuPu')
plt.axis('off')
tmax = 2718
plt.imshow(np.ma.masked_greater_equal(ttx_fin,tmax),vmax=tmax, cmap='Spectral', alpha=0.5); 
#plt.colorbar()
plt.tight_layout()
vis.multi_savefig(fig, 'figures/network-with-ttx')
```

```{code-cell} ipython3
#plt.imshow(speedx_w)
```

```{code-cell} ipython3
plt.imshow(filaments_ms[:256,:256])
```

## Figure 6b 3D

```{code-cell} ipython3
napari.view_image(np.random.randn(100,100,100))
```

```{code-cell} ipython3
%time 
field_shape_3d = (200, 200, 200)

filaments_ms_3d = uc.utils.rescale(sum(s**2*morpho.sato3d(np.random.randn(*field_shape_3d), s)
                                      for s in tqdm((1.5, 3, 6, 12))))
speed_3d = filaments_ms_3d
```

```{code-cell} ipython3
plt.imshow(filaments_ms_3d[100])
```

```{code-cell} ipython3
_=1
```

```{code-cell} ipython3
speed_3d.shape
```

```{code-cell} ipython3
phi03d = np.zeros(field_shape_3d)
phi03d[100-1:100+1,100-1:100+1,100-1:100+1] = 1
phi03d = ndi.binary_dilation(phi03d, iterations=2)
plt.imshow(phi03d.max(0))
```

```{code-cell} ipython3
plt.imshow(phi03d.max(1))
```

```{code-cell} ipython3
plt.imshow(phi03d.max(2))
```

```{code-cell} ipython3
phi03d = ~phi03d
```

```{code-cell} ipython3
plt.imshow(speed_3d[20])
```

```{code-cell} ipython3
plt.imshow(speed_3d[:,15])
```

```{code-cell} ipython3
%time ttx3d = skfmm.travel_time(phi03d, speed=speed_3d)
```

```{code-cell} ipython3
plt.imshow(ttx3d.max(0))
```

```{code-cell} ipython3
plt.imshow(ttx3d.mask[100])
```

```{code-cell} ipython3
ttx3df = ttx3d.filled(ttx3d.max())
```

```{code-cell} ipython3
plt.imshow(ttx3df[64])
```

```{code-cell} ipython3
#np.percentile(np.ravel(ttx3d),0.01)
```

```{code-cell} ipython3
tm_mask_3d = ndi.binary_dilation(ttx3d==0,iterations=1)
plt.imshow(tm_mask_3d[100])
```

```{code-cell} ipython3
boundary_mask = ndi.binary_dilation(ttx3df<=np.percentile(ttx3d,50),iterations=1)

plt.imshow(boundary_mask[100])
```

```{code-cell} ipython3
plt.imshow(boundary_mask[:,100])
```

```{code-cell} ipython3
pts_all3d = np.random.permutation(np.array(np.where(boundary_mask)).T)
pts_all3d.shape
```

```{code-cell} ipython3
reload(iffm)
```

```{code-cell} ipython3
tree_3d, speedx_3d, ttx_3d = iffm.iterative_build_tree(#speed_3d.T, 
                                             #np.ones(speed_3d.shape),
                                             np.random.rand(*speed_3d.shape),
                                             phi03d, 
                                             pts_all3d[:2000],
                                             scaling='linear',
                                             tm_mask=tm_mask_3d, 
                                             batch_size=1,
                                             batch_size_alpha=1.1)
plt.figure(); plt.imshow(np.log2(1+speedx_3d.max(0)), interpolation='nearest', cmap='PuBu')
```

```{code-cell} ipython3
plt.figure(); plt.imshow(np.log2(1+speedx_3d.max(1)), interpolation='nearest', cmap='PuBu')
```

```{code-cell} ipython3
plt.figure(); plt.imshow(np.log2(1+speedx_3d.max(2)), interpolation='nearest', cmap='PuBu')
```

```{code-cell} ipython3
counts = iffm.count_occurences(tree_3d, speedx_3d.shape)

#plt.figure(); plt.imshow(np.log2(1+counts.max(0)), interpolation='nearest', cmap='BuPu')

iffm.assign_diameters(tree_3d, min_diam=0.5,gamma=2.5,max_diam=12)
portrait_3d = make_portrait(tree_3d, speedx_3d.shape, fill_soma=True, soma_mask=tm_mask_3d)
#plt.figure(); plt.imshow(portrait_3d.max(0),  cmap='BuPu')
```

```{code-cell} ipython3

plt.figure(); plt.imshow(np.log2(1+counts.max(0)), interpolation='nearest', cmap='BuPu')
```

```{code-cell} ipython3
plt.figure(); plt.imshow(np.log2(1+counts.max(1)), interpolation='nearest', cmap='BuPu')
```

```{code-cell} ipython3
plt.figure(); plt.imshow(np.log2(1+counts.max(2)), interpolation='nearest', cmap='BuPu')
```

```{code-cell} ipython3
w = napari.view_image(np.log2(1 +counts))
w.add_image(portrait_3d)
```

```{code-cell} ipython3
#napari.view_image(speedx_3d)
```

```{code-cell} ipython3
#napari.view_image(phi03d)
```

```{code-cell} ipython3
np.ndim(speed_3d)
```

```{code-cell} ipython3
np.save('data/portrait_3d.npy', portrait_3d)
```

```{code-cell} ipython3
import pickle
```

```{code-cell} ipython3
pickle.dump(tree_3d, open('data/tree_3d.pickle', 'wb'))
```

```{code-cell} ipython3


import visvis as vv
```

```{code-cell} ipython3
%gui qt
```

```{code-cell} ipython3
vv.volshow(portrait_3d)
```

```{code-cell} ipython3
#tree.keys()
```

```{code-cell} ipython3
import pandas as pd

def to_swc_table(tree):
    "convert a tree to swc table"
    # start from 2 to allow for soma
    cell_ids = {loc:j+2 for j,loc in enumerate(tree)}
    acc = []
    visited = set()

    roots = iffm.get_roots(tree)
    center_loc = np.mean([r.v for r in roots],0)

    soma_radius = np.max([r.diam for r in roots])*1.5/2
    soma_entry =  dict(idx=1, ttype=1, 
                       x=center_loc[1], y=center_loc[2], z=center_loc[0], 
                       radius=soma_radius, parent = -1)
    
    for tip in iffm.get_tips(tree):
        for p in iffm.follow_to_root(tip):
            loc = tuple(p.v)
            if loc in visited:
                continue
            my_id = cell_ids[loc]
            parent = p.parent
            
            parent_id = cell_ids[tuple(parent.v)] if parent else -1
            radius = p.diam/2 if hasattr(p, 'diam') else 0.125
            row = dict(idx=my_id,
                       ttype=7, # glia (todo: add soma)
                       x = loc[1],
                       y = loc[2],
                       z = loc[0],
                       radius = radius,
                       parent = parent_id)
            acc.append(row)
            visited.add(loc)
    acc = acc[::-1] # parents come before children
    acc.append(soma_entry)
    return pd.DataFrame(acc)

def save_swc_table(path, df):
    "save pandas DataFrame with swc data to a file in '.swc' format"
    with open(path, 'w') as storage:
        storage.write('#iFM2B3D structure\n')
        storage.write('#id type x y z radius parent\n')
        df.to_csv(storage, sep=' ', header=False, index=False)
        
    
```

```{code-cell} ipython3
swc_df = to_swc_table(tree_3d)
```

```{code-cell} ipython3
save_swc_table('data/tree3d-wsoma-sparse.swc', swc_df)
```

---

```{code-cell} ipython3

```
