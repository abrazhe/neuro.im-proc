import numpy as np

import astromorpho as astro
from ucats import masks as umasks

from skimage import filters as skfilt

from imfun.core import extrema

from tqdm.auto import tqdm
from imfun.filt import l1spline, l2spline


def st_roll(arr):
    return np.moveaxis(arr, 2, 0)


def find_kink(v,pre_smooth=1.5):
    vs = l2spline(v,pre_smooth) if pre_smooth > 0 else v
    eout = extrema.extrema2(vs, refine=False)
    xfit,yfit = eout[0]
    #maxima,minima = [np.array(a) for a in eout[1]]
    minima = extrema.locextr(vs, output='min',refine=False)
    minima = np.array(minima)[:,0]

    gups,gdowns = [np.array(a) for a in eout[2]]
    print(gups[0])
    kink = np.max(minima[minima<=gups[0]+1])+1
    return int(np.round(kink))



def make_sato_mask(stack, sigmas=np.logspace(2,3,5,base=2), plow=99, phigh=99.9, **kwargs):
    out_sato = np.max([astro.morpho.sato3d(stack,s,**kwargs)*s**2 for s in tqdm(sigmas)],0).astype(np.float64)
    th_low_s, th_high_s = np.percentile(out_sato[out_sato>0], (plow, phigh))
    print('thresholds', th_low_s, th_high_s/2, th_high_s)
    sato_mask = umasks.largest_region(skfilt.apply_hysteresis_threshold(out_sato, th_high_s/2, th_high_s))
    return sato_mask

def make_simple_mask(stack):
    stack_rolled = np.moveaxis(stack, 2,0)
    th_low, th_high = np.percentile(stack, (99, 99.5))
    masks = np.array([umasks.refine_mask_by_percentile_filter(skfilt.apply_hysteresis_threshold(p, th_low, th_high),with_cleanup=True,min_obj_size=16)
                  for p in stack_rolled])
    return np.moveaxis(masks,0,2)

def combine_masks(stack,simple_mask,sato_mask):
    sato_m = st_roll(sato_mask)
    simple_m = st_roll(simple_mask)
    stack_rolled = st_roll(stack)
    masks3 = np.array([(m*(plane >= 0.25*np.mean(plane[m_s]))) if np.any(m_s) else np.zeros(plane.shape, np.bool)
                    for m,plane,m_s in zip(simple_m, stack_rolled, sato_m)])
    masks3 = umasks.select_overlapping(masks3,sato_m)
    return umasks.largest_region(masks3)


### EXAMPLE 1
#
# simple_mask = make_simple_mask(stack_z)
# sato_mask = make_sato_mask(stack_z)
# masks3 = combine_masks(stack_z, simple_mask, sato_mask)
#
# kink = find_kink(vx,pre_smooth=1.5)
#
# vx = masks3.sum(axis=(1,2))
# masks3a = masks3.copy()
# masks3a[kink:]=0
# masks3a = ucats.masks.largest_region(masks3a)
#
# use_kink = True
#
# masks3_final = masks3a if use_kink else masks3
# masks3_final = np.moveaxis(masks3_final,0,2)
#
# show_stack = 255*masks3_final.astype(int)


### EXAMPLE 2
#
# simple_mask = make_simple_mask(stack_z)
# sato_mask = make_sato_mask(stack_z)
# masks3 = combine_masks(stack_z, simple_mask, sato_mask)
#
# vx = masks3.sum(axis=(1,2))
# vx_sato = st_roll(sato_mask).sum(axis=(1,2))
#
# try:
#     kink = find_kink(vx)
# except :
#     #print(E)
#     kink = len(vx)
#
# masks3a = masks3.copy()
# masks3a[kink:]=0
# masks3a = ucats.masks.largest_region(masks3a)
#
# masks3_final = masks3a if use_kink else masks3
# masks3_final = np.moveaxis(masks3_final,0,2)
#
# show_stack = (stack_z*(~(ndi.binary_dilation(masks3_final,iterations=3)))).astype(float64)
