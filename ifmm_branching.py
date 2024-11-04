import numpy as np

from tqdm.auto import tqdm

from imfun.bwmorph import neighbours_2

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

# def follow_to_root_rec(tip):
#     if not tip.parent:
#         return [tip]
#     else:
#         return [tip] + follow_to_root(tip.parent)

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


def build_tree(ttm, seeds, tree=None, tm_mask=None):
    if tree is None:
        tree = dict()
    ttm = np.ma.filled(ttm, np.max(ttm))
    fails = []
    for p in tqdm(seeds):
        path,success = merging_rw_gd(ttm, p, 
                                     terminate_mask=tm_mask, 
                                     pjitter=0.0, 
                                     nsteps=10000, tree=tree)
        if not success:
            fails.append(path)    
    return tree, fails

import skfmm

def iterative_build_tree(speed, phi0, seeds, 
                         update_amp=1,
                         tm_mask = None,
                         scaling='linear',
                         batch_size=10, 
                         speed_gamma = 2,
                         alpha=1.0):
    speed0 = speed.copy()    
    speed = speed0.copy()
    
    ttx0 = skfmm.travel_time(phi0, speed=speed0)    
    ttx0 = np.ma.filled(ttx0, np.max(ttx0))
    ttx = ttx0.copy()
    
    tree = dict()
    
    speed_upd = np.zeros(speed.shape)    
    
    fails = []
    
    j = 0
    
    if scaling == 'linear':
        update_fn = lambda m:m
    elif scaling == 'power':
        update_fn = lambda m:m**speed_gamma
    elif scaling == 'log':
        update_fn = lambda m: np.log2(1 + m)
    elif scaling == 'exp':
        update_fn = lambda m: np.exp(m*speed_gamma)
    else:
        update_fn = lambda m:m
        
    
    for p0 in tqdm(seeds):
        p0 = tuple(map(int, p0))

        # skip unreacheable points
        if ttx[p0] == np.max(ttx):
            continue
        
        try_path, finished = merging_rw_gd(ttx, p0, 
                                           terminate_mask=tm_mask, 
                                           pjitter=0.0, 
                                           nsteps=10000, 
                                           tree=tree)
        if finished:
            apath = apath_to_root(tree[p0])
            speed_upd[tuple(apath[:-1,i] for i in (0,1))] +=\
                                                    update_amp
            #speed += update_fn(speed_upd)
            speed = speed0 + update_fn(speed_upd)
    
            if (not j%batch_size) and alpha**j > 1e-6:    
                #ttx = ttx + skfmm.travel_time(phi0, speed=speed)
                ttx = skfmm.travel_time(phi0, speed=speed)
                ttx = np.ma.filled(ttx, np.max(ttx))
            j += 1
        else:
            fails.append(np.array([n.v for n in try_path]))
            print('not finished for loc', p0)    
    return tree, speed, ttx
    