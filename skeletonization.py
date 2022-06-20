#!/usr/bin/env python
# coding: utf-8

import os
import sys

import cv2

import scipy
from scipy import ndimage as ndi
import numpy as np
import networkx as nx

from pathlib import Path
from tqdm.auto import tqdm

import pickle

import ccdb
import astromorpho as astro
from ucats import masks as umasks

from argparser import create_parser
from networx2napari import draw_edges, draw_nodes


# def graph_creation(vectors, soma_mask=None):
#     G = nx.Graph()
#     nodes = {}

#     ndim = vectors.shape[-1]
#     if ndim == 3:
#         i, j, k = np.indices(ndim)
#         idx = np.stack((i,j,k), axis=ndim)
#         for crop, acrop in crops3D:
#             G.add_weighted_edges_from(calc_edges(vectors[crop], vectors[acrop], idx[crop], idx[acrop]))
#     elif ndim == 2:
#         i, j = np.indices(ndim)
#         idx = np.stack((i,j), axis=ndim)
#         for crop, acrop in crops2D:
#             G.add_weighted_edges_from(calc_edges(vectors[crop], vectors[acrop], idx[crop], idx[acrop]))
#     else:
#         raise Exception('ERROR: wrong dimension of vectors array')

#     if soma_mask is not None:
#         soma = [tuple(i) for i in idx[soma_mask]]
#         G_s = nx.complete_graph(soma)
#         nx.set_edge_attributes(G_s, 0.7, name='weight')
#         for p1, p2, weight in G_s.edges(data=True):
#             try:
#                 old_weight = G.get_edge_data(p1, p2)['weight']
#             except:
#                 old_weight = 1
#             G.add_edge(p1, p2, weight=min(weight['weight'], old_weight))
#     nodes = {n:n for n in G.nodes()}
#     return G, nodes


def eu_dist(p1, p2):
    return np.sqrt(np.sum([(x - y)**2 for x, y in zip(p1, p2)]))


from collections import defaultdict


def count_points_paths(paths):
    acc = defaultdict(int)
    for path in paths:
        for n in path:
            acc[n] += 1
    return acc


def get_shell_mask(mask, do_skeletonize=False, as_points=False):
    out = ndi.binary_erosion(mask)^mask
    if do_skeletonize:
        out = skeletonize(out)
    if as_points:
        out = astro.morpho.mask2points(out)
    return out

from skimage.filters import threshold_li, threshold_minimum, threshold_triangle
from skimage.morphology import remove_small_objects


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


def filter_image(image, filter_func):
    threshold = filter_func(image)
    #img_filt = np.where(image > threshold, image, 0)
    binary_clean = remove_small_objects(image >= threshold, 5, connectivity=3)
    return np.where(binary_clean, image, 0)


def planewise_fill_holes(mask):
    for k,plane in enumerate(mask):
        mask[k] = ndi.binary_fill_holes(plane)
    return mask


import itertools as itt


def percentile_rescale(arr, plow=1, phigh=99.5):
    low, high = np.percentile(arr, (plow, phigh))
    if low == high:
        return np.zeros_like(arr)
    else:
        return np.clip((arr-low)/(high-low), 0, 1)


def flat_indices(shape):
    idx = np.indices(shape)
    return np.hstack([np.ravel(x_)[:,None] for x_ in idx])

from skimage.morphology import dilation, skeletonize, flood
from astromorpho import morpho


def prep_crops():
    "makes list of crops for edges"
    num2slice = {1: (slice(1,None), slice(None,-1)),
                 0: (slice(None), slice(None)),
                -1: (slice(None,-1), slice(1,None))}
    shifts = list(itt.product(*[(-1,0,1)]*3))
    # we only need one half of that
    cut = int(np.ceil(len(shifts)/2))
    crops_new = [list(zip(*[num2slice[n] for n in tuple])) for tuple in shifts[cut:]]
    return crops_new


def tensor_cosine_similarity(U, V, return_norms=False):
    "Calculate cosine similarity between vectors stored in the last dimension of some tensor"

    dprod = np.einsum('...ij,...ij->...i', U, V)

    #norm_U = np.linalg.norm(U, axis=-1)
    #norm_V = np.linalg.norm(V, axis=-1)

    # don't know why, but this is faster than linalg.norm
    norm_U = np.sum(U**2, axis=-1)**0.5
    norm_V = np.sum(V**2, axis=-1)**0.5

    normprod = norm_U*norm_V

    out = np.zeros(U.shape[:-1], dtype=np.float32)
    nonzero = normprod>0
    out[nonzero] = dprod[nonzero]/normprod[nonzero]

    if return_norms:
        return out, (norm_U, norm_V)
    else:
        return out


def calc_edges(U, V, index1, index2, alpha=0.0, do_threshold=True, return_W=False, verbose=False):

    # cовпадение направлений из Гессиана
    Sh, (normU,normV) = tensor_cosine_similarity(U,V, return_norms=True)
    Sh = np.abs(Sh)

    # совпадение направления из Гессиана и направления к соседу
    Se = tensor_cosine_similarity(U, index2-index1, return_norms=False)
    Se = np.abs(Se)

    N = (normU + normV)/2
    N /= N.max()

    #W = 1 - N*((1 - alpha)*Sh + alpha*Se)
    W = 1 - N*(Sh * Se**alpha)


    if return_W:
        return W

    Wflat = W.ravel()
    cond = Wflat < 1
    Sx = 1-Wflat[cond]
    #thresholds = [1-threshold_minimum(Sx),
    #              1-threshold_li(Sx),
    #              1-threshold_triangle(Sx)
    #             ]
    #th = np.max(thresholds)
    th = 1 - (threshold_li(Sx))
    #li = threshold_li(Wflat) if do_threshold else W.max()
    th = th if do_threshold else W.max()
    Wgood = Wflat < th

    if verbose:
        print('Thresholding done')
        print('Threshold: ', th)
        print('Max, min:', Wflat.max(), Wflat.min())
        print('% supra-threshold', 100*np.sum(Wgood)/len(Wflat))

    idx1 = (tuple(i) for i in index1.reshape((-1, index1.shape[-1]))[Wgood])
    idx2 = (tuple(i) for i in index2.reshape((-1, index2.shape[-1]))[Wgood])

    return zip(idx1, idx2,  Wflat[Wgood])


def get_mask_vals(idxs, mask):
    idx_mask = mask[idxs[:,0], idxs[:,1], idxs[:,2]]
    return idxs[idx_mask]


def get_edges(mask, index1, index2, weight):
    idx1 = [tuple(i) for i in get_mask_vals(index1.reshape((-1, index1.shape[-1])), mask)]
    idx2 = [tuple(i) for i in get_mask_vals(index2.reshape((-1, index2.shape[-1])), mask)]
    return zip(idx1, idx2, np.full(len(idx1), weight))


from copy import copy


def find_paths(G, targets, min_count=1, min_path_length=10):
    paths_dict = nx.multi_source_dijkstra_path(G, targets, )

    #reverse order of points in paths, so that they start at tips
    paths_dict = {path[-1]:path[::-1] for path in paths_dict.values() if len(path) >= min_path_length}
    paths = list(paths_dict.values())
    points = count_points_paths(paths)

    qstack = np.zeros(vectors.shape[:-1])  #Это встречаемость точек в путях
    for p, val in points.items():
        if val >= min_count:
            qstack[p] = np.log(val)
    return qstack, paths_dict


def path_to_graph(path):
    "Converts an ordered list of points (path) into a directed graph"
    g = nx.DiGraph()
    tp = tip_paths[0]

    for k,p in enumerate(path):
        tp = tuple(p)
        g.add_node(tp)
        if k > 0:
            g.add_edge(tp, tuple(path[k-1]), weight=1)
    return g


def get_tips(g):
    return {n for n in g.nodes if len(list(g.successors(n))) == 0}


def batch_compose_all(tip_paths, batch_size=10000):
    graphs = []
    print("Composing...")
    for i, tp in enumerate(tqdm(tip_paths)):
        graphs.append(path_to_graph(tp))
        if i % batch_size == 0:
            gx_all = nx.compose_all(graphs)
            graphs = [gx_all]
    return nx.compose_all(graphs)


def get_attrs_by_nodes(G, arr, func=None):
    nodesG = np.array(G.nodes())
    attrs = arr[nodesG[:,0], nodesG[:,1], nodesG[:,2]]
    if func is not None:
        func_vect = np.vectorize(func)
        attrs = func_vect(attrs)
    return {tuple(node): attr for node, attr in zip(nodesG, attrs)}


def filter_graph(graph, func = lambda node: True ):
    "returns a view on graph for the nodes satisfying the condition defined by func(node)"
    good_nodes = (node for node in graph.nodes if func(graph.nodes[node]))
    return graph.subgraph(good_nodes)


# def get_roots(g):
#     return {n for n in g.nodes if len(list(g.predecessors(n))) < 1}


# def get_branch_points(g):
#     return {n for n in gx.nodes if len(list(gx.successors(n))) > 1}


# def graph_to_paths(g, min_path_length=1):
#     """
#     given a directed graph, return a list of a lists of nodes, collected
#     as unbranched segments of the graph
#     """

#     roots = get_roots(g)

#     def _acc_segment(root, segm, accx):
#         if segm is None:
#             segm = []
#         if accx is None:
#             accx = []
#         children = list(g.successors(root))

#         if len(children) < 1:
#             accx.append(segm)
#             return

#         elif len(children) == 1:
#             c = children[0]
#             segm.append(c)
#             _acc_segment(c, segm, accx)

#         if len(children) > 1:
#             #segm.append(root)
#             accx.append(segm)
#             for c in children:
#                 _acc_segment(c, [root, c], accx)

#     acc = {}
#     for root in roots:
#         px = []
#         _acc_segment(root, [], px)
#         acc[root] = [s for s in px if len(s) >= min_path_length]
#     return acc



# def paths_to_colored_stack(paths, shape, change_color_at_branchpoints=False):
#     #colors = np.random.randint(0,255,size=(len(paths),3))
#     stack = np.zeros(shape + (3,), np.uint8)
#     for root in paths:
#         color =  np.random.randint(0,255, size=3)
#         for kc,pc in enumerate(paths[root]):
#             if change_color_at_branchpoints:
#                 color = np.random.randint(0,255, size=3)
#             for k,p in enumerate(pc):
#                 #print(k, p)
#                 stack[tuple(p)] = color
#     return stack

tup2str = lambda x: ','.join(list(map(str, x)))
str2tup = lambda x: tuple(map(int, x.split(',')))


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    # if not check_args(args):
    #     exit()

    # Считывание изображения
    filename = Path(args.data_dir).joinpath(args.filename)

    stack, meta = ccdb.read_pic(filename)
    dims = ccdb.get_axes(meta)

    if len(dims):
        zoom = (dims[-1][0]/dims[0][0])
    else:
        zoom = 4


    # Предобработка изображения
    ## CLAHE
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))

    if args.clahe:
        stack_shape = stack.shape
        img_clahe = np.zeros(stack.shape, np.float32)
        for k,plane in enumerate(stack):
            img_clahe[k] = clahe.apply(plane)
        img = img_clahe
    else:
        img = stack

    ## Кадрирование
    max_proj_xy = img.max(0)

    domain_mask_xy = ndi.binary_dilation(largest_region(remove_small_objects(max_proj_xy > 0.5*threshold_li(max_proj_xy))), iterations=3)
    domain_mask_xy = ndi.binary_closing(domain_mask_xy,iterations=3)
    img_cropped = np.array([crop_image(plane,domain_mask_xy, margin=10) for plane in img])

    max_proj_yz = img_cropped.max(1)
    domain_mask_yz = ndi.binary_dilation(largest_region(remove_small_objects(max_proj_yz > 0.5*threshold_li(max_proj_yz))), iterations=3)
    domain_mask_yz = ndi.binary_closing(domain_mask_yz,iterations=3)
    img_cropped = np.array([crop_image(img_cropped[:,i],domain_mask_yz, margin=10) for i in range(img_cropped.shape[1])]).swapaxes(0,1)

    ## Масштабирование
    downscale = 2
    img_noisy = ndi.zoom(img_cropped.astype(np.float32), (zoom/downscale, 1/downscale, 1/downscale), order=1)

    ## Фильтрация изображения
    img_clear = filter_image(img_noisy, threshold_li)
    final_image = img_clear


    # Сегментация сомы
    ## Определение центра
    X1a = flat_indices(final_image.shape)
    weights_s = percentile_rescale(np.ravel(ndi.gaussian_filter(final_image,5))**2,plow=99.5,phigh=99.99)
    center = tuple(map(int, np.sum(X1a*weights_s[:,None],axis=0)/np.sum(weights_s)))

    ## Выделение сомы
    smooth_stack = ndi.gaussian_filter(final_image, 3)
    tol = (smooth_stack.max() - smooth_stack[final_image>0].min())/10
    soma_seed_mask = flood(smooth_stack, center, tolerance=tol)
    soma_mask = morpho.expand_mask(soma_seed_mask, smooth_stack, iterations = 10)
    soma_shell = get_shell_mask(soma_mask, as_points=True)


    # Расчет матрицы Гессе для различных сигм
    sigmas = 2**np.arange(args.start, args.end+args.step, args.step)
    id2sigma = {i+1:sigma for i, sigma in enumerate(sigmas)} # shift by one, so that zero doesn't correspond to a cell
    sigma2id = {sigma:i+1 for i, sigma in enumerate(sigmas)}
    sato_coll = {}
    Vf_coll = {}
    print("Calculating Hessian...")
    for sigma in tqdm(sigmas):
        #astro.morpho.sato3d is newer and uses tensorflow (if it's installed)
        #optimally, the two variants of sato3d should be merged
        sato, Vf = astro.morpho.sato3d(final_image, sigma, hessian_variant='gradient_of_smoothed', do_brightness_correction=False, return_vectors=True)
        sato_coll[sigma] = (sato*sigma**2)*(final_image > 0)
        Vf_coll[sigma] = Vf[...,0][...,::-1]

    lengths_coll = {sigma: astro.enh.percentile_rescale(sato)**0.5 for sigma, sato in sato_coll.items()}
    vectors_coll = {}

    for sigma in Vf_coll:
        Vfx = Vf_coll[sigma]
        V = Vfx[..., 0]
        U = Vfx[..., 1]
        C = Vfx[..., 2]
        lengths = lengths_coll[sigma]
        vectors_coll[sigma] = np.stack((U*lengths, V*lengths, C*lengths), axis=3)


    # Расчет масок для различных сигм
    masks = {}
    print('Calculating masks...')
    for sigma in tqdm(sigmas):
        sato = sato_coll[sigma]
        threshold = threshold_li(sato[sato>0])*sigma**0.5
        masks[sigma] = remove_small_objects(sato > threshold, min_size=int(sigma*64))
    masks[sigmas[-1]] = umasks.select_overlapping(masks[sigmas[-1]], soma_mask)
    for k in range(len(sigmas)-2,-1,-1):
        sigma = sigmas[k]
        masks[sigma] = umasks.select_overlapping(masks[sigma], ndi.binary_dilation(masks[sigmas[k+1]], iterations=5))

    # Объединение результатов Сато для различных сигм
    sigma_sato = np.zeros(final_image.shape, dtype=int)
    hout = np.zeros(final_image.shape)
    mask_sum = np.zeros(final_image.shape, dtype=bool)

    print("Merging Sato...")
    for sigma, sato in tqdm(sorted(sato_coll.items(), reverse=True)):
        hcurr = sato
        mask_sum = masks[sigma] | mask_sum
        mask = (hcurr > hout)*mask_sum # restrict search for optimal sigmas by the corresponding mask

        hout[mask] = hcurr[mask]
        sigma_sato[mask] = sigma2id[sigma]


    # Объединение собственных векторов различных сигм
    vectors_best = np.zeros(vectors_coll[sigmas[0]].shape)
    mask_sum = np.zeros(final_image.shape,bool)
    masks_exclusive = {}

    for k in range(len(sigmas)-1,-1,-1):
        sigma = sigmas[k]
        mask = masks[sigma]
        if k < len(sigmas)-1:
            mask = mask & (mask ^ mask_sum)
        mask_sum += mask.astype(bool)
        masks_exclusive[sigma] = mask
        vectors_best[mask] = vectors_coll[sigma][mask]

    # Объединение масок для различных сигм
    sigma_mask = np.zeros(final_image.shape, dtype=int)
    for sigma_id, sigma in id2sigma.items():
        sigma_mask[masks_exclusive[sigma]] = sigma_id


    # Построение графа

    ## Выражение для весов ребер
    i, j, k = np.indices(final_image.shape)
    idx = np.stack((i,j,k), axis=3)
    crops = prep_crops()

    alpha = 0.0
    vectors = vectors_best
    graph = nx.Graph()
    print('Creating graph...')
    for crop, acrop in tqdm(crops):
             graph.add_weighted_edges_from(calc_edges(vectors[crop], vectors[acrop], idx[crop], idx[acrop], alpha=alpha))

    ## Добавление точек оболочки сомы в граф
    Gsoma = nx.Graph()
    soma_shell_mask = get_shell_mask(soma_mask)
    print("Adding soma...")
    for crop, acrop in tqdm(crops):
        Gsoma.add_weighted_edges_from(get_edges(soma_shell_mask, idx[crop], idx[acrop], 0.7))

    for p1, p2, weight in Gsoma.edges(data=True):
        try:
            old_weight = graph.get_edge_data(p1, p2)['weight']
        except Exception as exc:
            old_weight = 1
            graph.add_edge(p1, p2, weight=min(weight['weight'], old_weight))
    nodes = {n:n for n in graph.nodes()}


    # Расчет путей, встречаемости точек в путях и слияние графов по путям

    qstack, paths_best = find_paths(graph, soma_shell)
    all_tips = list(paths_best.keys())


    domain_mask3d = ndi.binary_fill_holes(final_image > 0)
    domain_shell_mask = get_shell_mask(domain_mask3d)

    domain_mask3d = planewise_fill_holes(domain_mask3d)

    domain_mask3d = np.moveaxis(domain_mask3d, 1, 0)
    domain_mask3d = planewise_fill_holes(domain_mask3d)
    domain_mask3d = np.moveaxis(domain_mask3d, 0, 1)

    domain_mask3d = np.moveaxis(domain_mask3d, 2, 0)
    domain_mask3d = planewise_fill_holes(domain_mask3d)
    domain_mask3d = np.moveaxis(domain_mask3d, 0, 2)

    domain_outer_shell_mask = get_shell_mask(domain_mask3d) & domain_shell_mask


    domain_outer_shell_pts = set(astro.morpho.mask2points(domain_outer_shell_mask))
    domain_shell_pts = set(astro.morpho.mask2points(domain_shell_mask))
    print('Making tips...')
    tips = [t for t in tqdm(all_tips) if t in domain_shell_pts]
    tip_paths = [np.array(paths_best[t]) for t in tips]

    gx_all = batch_compose_all(tip_paths) # УРА!!

    # Добавление сопутствующей информации
    nx.set_node_attributes(gx_all,
                           get_attrs_by_nodes(gx_all, sigma_mask, lambda x: id2sigma[x]),
                           'sigma_mask')
    nx.set_node_attributes(gx_all,
                           get_attrs_by_nodes(gx_all, sigma_sato, lambda x: id2sigma[x]),
                           'sigma_opt')
    nx.set_node_attributes(gx_all,
                           get_attrs_by_nodes(gx_all, qstack),
                           'occurence')

    # Распределения встречаемостей по сигме

    # Попробуем взать только те пути, где встречаемость больше порога на соотв. сигме

    occ_acc = {}
    print('Filtering occurance')
    for sigma in tqdm(sigmas):
        sub = filter_graph(gx_all, lambda node: node['sigma_mask']==sigma)
        occ_acc[sigma] = np.array([sub.nodes[n]['occurence'] for n in sub.nodes])

    occ_threshs = {}

    for sigma in sigmas:
        v_occ = occ_acc[sigma]
        th = threshold_li(v_occ)
        occ_threshs[sigma] = th

    high_occ_subs = [filter_graph(gx_all, lambda node: (node['occurence'] >=th) & (node['sigma_mask']==sigma)) for sigma, th in occ_threshs.items()]
    high_occurence_graph1  = nx.compose_all(high_occ_subs)
    # Перестроим еще раз граф, но уже используя только узлы из графа `high_occurence_graph1`

    high_occurence_graph1a = graph.subgraph(high_occurence_graph1.nodes) # узлы из изначального графа
    filtered_soma_shell = [p for p in soma_shell if p in high_occurence_graph1a]
    qstack2, filtered_paths = find_paths(high_occurence_graph1a, filtered_soma_shell)

    high_occurence_tips = get_tips(high_occurence_graph1) # Кочики веток из предыдущего графа
    filtered_tips = [t for t in filtered_paths if domain_shell_mask[t] and t in high_occurence_tips]

    # Теперь скомпозируем все обратно и запишем атрибуты
    print("Composing... Again...")
    high_occurence_graph1b = nx.compose_all(path_to_graph(filtered_paths[t]) for t in tqdm(filtered_tips))

    nx.set_node_attributes(high_occurence_graph1b,
                           get_attrs_by_nodes(high_occurence_graph1b, sigma_mask, lambda x: id2sigma[x]),
                           'sigma_mask')
    nx.set_node_attributes(high_occurence_graph1b,
                           get_attrs_by_nodes(high_occurence_graph1b, sigma_sato, lambda x: id2sigma[x]),
                           'sigma_opt')
    nx.set_node_attributes(high_occurence_graph1b,
                           get_attrs_by_nodes(high_occurence_graph1b, qstack),
                           'occurence')

    final_graph = high_occurence_graph1b


    # xpaths_all = graph_to_paths(gx_all)
    # colored_paths_all = paths_to_colored_stack(xpaths_all, final_image.shape, change_color_at_branchpoints=False)

    # w = napari.view_image(final_image, ndisplay=3, opacity=0.5)
    # #props = {'path-id': ['line'+str(i) for i in np.arange(len(xpaths))]}
    # w.add_image(colored_paths_all,  channel_axis=3, colormap=['red','green','blue'], name='cp_all')
    # napari.run()

    output_filename = '{}graph_{}{}'.format(args.prefix, os.path.basename(filename), args.suffix)
    if args.save_type == 'gml':
        nx.write_gml(final_graph, output_filename + '.gml', tup2str)
    elif args.save_type == 'pickle':
        with open(output_filename + '.pickle', 'wb') as f:
            pickle.dump(final_graph, f)
    else:
        print('ERROR!! Please choose save type: gml or pickle')
        exit()