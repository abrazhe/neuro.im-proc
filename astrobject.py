import itertools as itt
from functools import reduce

import numpy as np
from scipy import ndimage as ndi

from skimage import segmentation
from skimage import feature as skf
from skimage.morphology import flood, remove_small_objects
from skimage.filters import threshold_li

import networkx as nx

import astromorpho as astro
from astro_graph import AstroGraph as AG

## CENTER DETECTION
def percentile_rescale(arr, plow=1, phigh=99.5):
    low, high = np.percentile(arr, (plow, phigh))
    if low == high:
        return np.zeros_like(arr)
    else:
        return np.clip((arr-low)/(high-low), 0, 1)


def flat_indices(shape):
    idx = np.indices(shape)
    return np.hstack([np.ravel(x_)[:,None] for x_ in idx])


## SOMA SEGMENTATION
def get_shell_mask(mask, do_skeletonize=False, as_points=False):
    out = ndi.binary_erosion(mask)^mask
    if do_skeletonize:
        out = skeletonize(out)
    if as_points:
        out = astro.morpho.mask2points(out)
    return out

## BRANCH SEGMENTATION

def calc_vectors(image, sigma, scale):
    sato, Vf = astro.morpho.sato3d(image, (sigma/scale[0], sigma, sigma),
                                       hessian_variant='gradient_of_smoothed',
                                       do_brightness_correction=False,
                                       return_vectors=True)

    sato = (sato*sigma**2)*(image > 0)
    Vf = Vf[...,0][...,::-1] # z, r, c

    lengths = astro.enh.percentile_rescale(sato)**0.5

    Vfx = Vf
    C = Vfx[...,0] # -> Z (d)
    V = Vfx[...,1] # -> Y (r)
    U = Vfx[...,2] # -> X (c)
    vectors = np.stack((C*lengths, V*lengths, U*lengths), axis=3)

    return vectors, sato


def calc_sato_mask(sato, sigma):
    threshold = threshold_li(sato[sato>0])*sigma**0.5
    mask = remove_small_objects(sato > threshold, min_size=int(sigma*64))
    return mask


def masks_overlapping(*masks, reverse=False):
    if reverse:
        masks = masks[::-1]

    for k in range(len(masks), -1):
        masks[k] = umasks.select_overlapping(masks[k+1], ndi.binary_dilation(masks[k+1], iterations=5))

    if reverse:
        masks = masks[::-1]

    return masks


def mask_thresholding(image, mask, threshold_method=threshold_li):
    lightness = image[mask]
    th = threshold_method(lightness)
    pre_mask = remove_small_objects((image > th) & mask, 5, connectivity=3)
    return pre_mask


def clear_sigmas(sigma2del, sigmas, vectors, sato, masks):
    for sigma in sigma2del.values():
        del masks[sigma]
        del sato[sigma]
        del vectors[sigma]

    sigmas = np.delete(sigmas, list(sigma2del.keys()))
    return sigmas


def merge_sato(image, satos, masks, sigma2id):
    sato_best = np.zeros(image.shape, dtype=int)
    hout = np.zeros(image.shape, bool)
    mask_sum = np.zeros(image.shape, bool)
    for sigma, sato in sorted(satos.items(), reverse=True):
        hcurr = sato
        mask_sum = masks[sigma] | mask_sum
        mask = (hcurr > hout)*mask_sum
        hout[mask] = hcurr[mask]
        sato_best[mask] = sigma2id[sigma]
    return sato_best


def merge_vectors(vectors, sigmas, masks):
    vectors_best = np.zeros(vectors[sigmas[0]].shape)
    mask_sum = np.zeros(vectors[sigmas[0]].shape[:-1], bool)
    masks_exclusive = {}
    for k in range(len(sigmas)-1,-1,-1):
        sigma = sigmas[k]
        mask = masks[sigma]
        if k < len(sigmas)-1:
            mask = mask & (mask ^ mask_sum)
        mask_sum += mask.astype(bool)
        masks_exclusive[sigma] = mask
        vectors_best[mask] = vectors[sigma][mask]
    return vectors_best, masks_exclusive


## FULL GRAPH PLOTTING

def prep_crops(ndim=3):
    '''makes list of crops for edges'''
    num2slice = {1: (slice(1,None), slice(None,-1)),
                 0: (slice(None), slice(None)),
                -1: (slice(None,-1), slice(1,None))}
    shifts = list(itt.product(*[(-1,0,1)]*ndim))
    # we only need one half of that
    cut = int(np.ceil(len(shifts)/2))
    crops_new = [list(zip(*[num2slice[n] for n in tuple])) for tuple in shifts[cut:]]
    return crops_new


def tensor_cosine_similarity(U, V, return_norms=False):
    '''Calculate cosine similarity between vectors stored in the last dimension of some tensor'''

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


def calc_edges(U, V, index1, index2, alpha=0.1, beta=0.001, offset=1,
               do_threshold=True, return_W=False, verbose=False):

    # cовпадение направлений из Гессиана
    Sh, (normU,normV) = tensor_cosine_similarity(U,V, return_norms=True)
    Sh = np.abs(Sh)

    # совпадение направления из Гессиана и направления к соседу
    Se = tensor_cosine_similarity(U, (index2-index1), return_norms=False)
    Se = np.abs(Se)

    #Sx = np.sum((index2-index1)**2, axis=-1)#**0.5
    Sx = np.sum(np.abs(index2-index1), axis=-1)
    #Sx /= Sx.max()

    N = (normU + normV)/2
    N /= N.max()

    # if VERBOSE:
    #     print('N+ percentiles:', np.percentile(N[N>0], (2,25,50,75,95)))

    # Cosine similarity between Hessian eigenvectors orientations and
    # between Hessian vector and linkage vector
    S = (1-alpha)*Sh + alpha*Se

    # THIS IS THE MAIN THING IN THE NOTEBOOK
    W  = Sx*beta + offset - N*S

    if verbose:
        print('Negative weights?', np.any(W<offset))
        print('S stats:', np.percentile(np.exp(-N*S)[N>0], (2,25,50,75,95)))
        print('W stats:', np.percentile(W[N>0], (2,25,50,75,95)))
        print('Sx stats:', np.percentile(Sx[N>0], (2,25,50,75,95)))

    W = np.maximum(0, W) # just to be safe

    if return_W:
        return W

    Wflat = W.ravel()
    #cond = Wflat < Wflat.max()
    cond = np.ravel(N) > 0
    Sx = Wflat[cond]
    # Thresholding is the tricky bit: too little and it takes forever to compute paths
    # Too high and you can't build paths at all

    # The negative threshold of negative distribution trick
    # Rationale is that we want to take "dark" values rather than "bright"
    # So we take a negative of the  "picture" and flip over the threshold
    th = -threshold_li(-Sx)
    th = th if do_threshold else W.max()
    Wgood = (Wflat < th) & (np.ravel(N)>0) # was this

    # if VERBOSE:
    #     print('Thresholding done')
    #     print('Threshold: ', th)
    #     print('Max, min:', Wflat.max(), Wflat.min())
    #     print('% supra-threshold', 100*np.sum(Wgood)/len(Wflat))

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


def add_soma_points(graph, soma_mask, idx, crops):
    Gsoma = nx.Graph()
    soma_shell_mask = get_shell_mask(soma_mask)
    for crop, acrop in crops:
        Gsoma.add_weighted_edges_from(get_edges(soma_shell_mask, idx[crop], idx[acrop], 0.7))

    for p1, p2, weight in Gsoma.edges(data=True):
        try:
            old_weight = graph.get_edge_data(p1, p2)['weight']
        except Exception as exc:
            old_weight = 1
        graph.graph.add_edge(p1, p2, weight=min(weight['weight'], old_weight))


## ASTRO GRAPH PLOTTING

def trim_path(g, path, sigma_start, visited_set):
    acc = []
    for p in path:
        acc.append(p)
        if (g.nodes[p]['sigma_mask'] > sigma_start) and (p in visited_set):
            break
    return acc


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

def compose_path_segments(G, stack_shape, seq_paths, ultimate_targets, max_start_sigma=2, min_path_lenght=25):
    """
    Combine all multi-scale path segments to a graph, then take only paths
    starting a a small enough sigma and reaching for the soma, the ultimate target
    """
    gx_all = nx.compose_all([seq_paths[sigma] for sigma in sorted(seq_paths)])
    gx_all = AG(gx_all)

    all_tips = gx_all.get_tips()
    fine_tips = list({t for t in all_tips if G.nodes[t]['sigma_mask'] <= max_start_sigma})
    new_paths = (follow_to_root(gx_all, t) for t in fine_tips)
    # Can leave just min_path_length (?)
    new_paths = (p for p in new_paths
                 if p[-1] in ultimate_targets and len(p)>=min_path_lenght)
    new_paths = sorted(new_paths, key=lambda p: len(p), reverse=True)

    gx_all = AG.batch_compose_all(new_paths)

    counts = AG.count_points_paths(new_paths)
    qstack = np.zeros(stack_shape)
    for p,val in counts.items():
        if val >= 1:
            qstack[p] = np.log(val)

    # add the useful attributes
    nx.set_node_attributes(gx_all,
                           gx_all.get_attrs_by_nodes(qstack),
                           'occurence')
    nx.set_node_attributes(gx_all,
                           gx_all.get_attrs_by_nodes(sigma_mask, lambda x: id2sigma[x]),
                           'sigma_mask')

    nx.set_node_attributes(gx_all,
                           gx_all.get_attrs_by_nodes(sigma_sato, lambda x: id2sigma[x]),
                           'sigma_opt')
    return gx_all


def filter_fn_(G, n):
    ni = G.nodes[n]
    #is_high = ni['occurence'] > max(0, occ_threshs[ni['sigma_mask']])
    is_high = ni['occurence'] > 0 # very permissive, but some branches are valid and only occur once
    not_tip = len(list(G.successors(n)))
    return is_high and not_tip


class AstrObject:
    def __init__(self, image, soma_mask=None, soma_shell_mask=None):
        self.image = image
        self.soma_mask = soma_mask
        self.soma_shell_mask = soma_shell_mask
        self._graph = None


    @property
    def graph(self):
        return self._graph

    @graph.setter
    def set_graph(self, graph):
        if type(graph) is AstroGraph:
            self._graph = graph


    def center_detection(self):
        X1a = flat_indices(self.image.shape)
        weights_s = percentile_rescale(np.ravel(ndi.gaussian_filter(self.image,5))**2,plow=99.5,phigh=99.99)
        center = tuple(map(int, np.sum(X1a*weights_s[:,None],axis=0)/np.sum(weights_s)))
        self.center = center


    def soma_segmentation(self, iterations=10, return_shell=False):
        print('Smooth Stack')
        smooth_stack = ndi.gaussian_filter(self.image, 3)
        tol = (smooth_stack.max() - smooth_stack[self.image>0].min())/10
        print('Soma Seed')
        soma_seed_mask = flood(smooth_stack, self.center, tolerance=tol)
        print('Soma Mask')
        soma_mask = astro.morpho.expand_mask(soma_seed_mask, smooth_stack, iterations=iterations)
        print('Soma Shell')
        soma_shell = get_shell_mask(soma_mask, as_points=True)

        self.soma_mask=soma_mask
        self.soma_shell_mask=soma_shell


    def branch_segmentation(self, scale, sigma_start=0, sigma_end=3, sigma_step=0.5, sigmas=None):

        if sigmas is None:
            sigmas = 2**np.arange(sigma_start, sigma_end, sigma_step)

        ## VECTORS AND MASKS
        print('Vectors and Masks...')

        masks=[]
        vectors={}
        satos={}
        for sigma in sigmas:
            vectors[sigma], satos[sigma] = calc_vectors(self.image, sigma, scale)
            masks.append(calc_sato_mask(satos[sigma], sigma))

        masks = {sigma: mask for sigma, mask in zip([*sigmas, 0], masks_overlapping(*masks, self.soma_mask, reverse=True))}

        sigma2del = {}
        for sigma, mask in masks.items():
            masks[sigma] = mask_thresholding(self.image, mask)
            if np.sum(masks[sigma]) == 0:
                sigma2del[i] = sigma

        self.sigmas = clear_sigmas(sigma2del, sigmas, vectors, satos, masks)

        self.masks = masks
        self.id2sigma = {i+1:sigma for i, sigma in enumerate(sigmas)}
        self.sigma2id = {sigma:i+1 for i, sigma in enumerate(sigmas)}


        ## MERGING
        print('Merging...')

        self.sato = merge_sato(self.image, satos, masks, self.sigma2id)
        self.vectors, masks_exclusive = merge_vectors(vectors, sigmas, masks)

        sigma_mask = np.zeros(self.image.shape, dtype=int)
        for sigma_id, sigma in self.id2sigma.items():
            sigma_mask[masks_exclusive[sigma]] = sigma_id
        self.sigma_mask = sigma_mask


    def full_graph_plotting(self, alpha, beta, offset=1):
        i, j, k = np.indices(self.image.shape)
        idx = np.stack((i,j,k), axis=3)

        crops = prep_crops()
        graph = AG(nx.Graph())

        vectors = self.vectors

        for crop, acrop in crops:
            graph.graph.add_weighted_edges_from(calc_edges(vectors[crop], vectors[acrop],
                                                     idx[crop], idx[acrop],
                                                     alpha=alpha, beta=beta,
                                                     verbose=False))

        # no-no for too big sigma jumps
        for p1, p2, data in graph.edges(data=True):
            if np.abs(self.sigma_mask[p1]-self.sigma_mask[p2]) > 1:
                graph.graph.add_edge(p1,p2, weight=data['weight']*2)

        # Add soma points
        add_soma_points(graph, self.soma_mask, idx, crops)
        self.sigma_mask[self.soma_mask] = self.sigma2id[self.sigmas[-1]] # Soma is also the largest scale
        ssm = np.array(self.soma_shell_mask)
        self.sigma_mask[ssm[:,0], ssm[:,1], ssm[:,2]] = self.sigma2id[self.sigmas[-1]] # Soma is also the largest scale

        self.id2sigma[0] = 0
        nx.set_node_attributes(graph,
                               graph.get_attrs_by_nodes(self.sigma_mask, lambda x: self.id2sigma[x]),
                               'sigma_mask')
        self.full_graph = graph



    def scale_sequential_paths(self):
        """
        Starting with the largest spatial scale, first try to reach soma, then reach the set of the
        previous starting points, and so on. Some  black magic with stopping the path segments at the
        right place to prevent loops and cycles in the merged graphs.
        Cycles are bad, because they break the coloring/visualization code :)
        """
        sub_graphs = {sigma:self.full_graph.filter_graph(lambda n: n['sigma_mask']>=sigma) for sigma in self.sigmas}
        targets = set(self.soma_shell_mask)
        visited = set(self.soma_shell_mask)
        path_acc = {}
        for sigma in sorted(self.sigmas, reverse=True):
            _, paths = AG.find_paths(sub_graphs[sigma], targets, self.image.shape)
            targets = targets.union(set(paths.keys()))
            if sigma < np.max(self.sigmas):
                paths = {loc:trim_path(self.full_graph, path, sigma, visited)
                         for loc, path in paths.items()
                         if self.full_graph.nodes[loc]['sigma_mask'] == sigma}
            visited = visited.union(reduce(set.union, paths.values(), set()))
            non_empty_paths = [p for p in paths.values() if p]
            if len(non_empty_paths):
                path_acc[sigma] = AG.batch_compose_all(non_empty_paths, verbose=False)
            else:
                path_acc[sigma] = nx.DiGraph()
        return path_acc


    def astro_graph_plotting(self):
        seq_paths = self.scale_sequential_paths()
        gx_all = compose_path_segments(self.full_graph, self.image.shape, seq_paths, ultimate_targets=set(self.soma_shell_mask))
        gx_all.check_for_cycles(verbose=True)
        occ_acc = {}
        for sigma in self.sigmas:
            sub = gx_all.filter_graph(lambda node: node['sigma_mask']==sigma)
            occ_acc[sigma] = np.array([sub.nodes[n]['occurence'] for n in sub.nodes])

        gx_all_occ = gx_all

        for i in range(10):
            good_nodes = (node for node in gx_all_occ if filter_fn_(gx_all_occ, node))
            gx_all_occ = gx_all_occ.subgraph(good_nodes)

        self.graph = gx_all_occ