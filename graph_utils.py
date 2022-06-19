from collections import defaultdict

import numpy as np
import networkx as nx

from tqdm.auto import tqdm

def path_to_graph(path):
    "Converts an ordered list of points (path) into a directed graph"
    g = nx.DiGraph()
    
    root = tuple(path[-1])
    visited = set()
    for k,p in enumerate(path):
        tp = tuple(p)
        if not tp in visited:
            g.add_node(tp, root=root)            
            if k > 0:
                g.add_edge(tp, tprev, weight=1)
            tprev = tp
            visited.add(tp)
    return g


def check_for_cycles(g, verbose=False):
    try:
        cycle = nx.find_cycle(g)
        if verbose:
            print('Found a cycle:', cycle)
        return cycle
    except nx.exception.NetworkXNoCycle:
        if verbose:
            print('No cycles!')
        return None

def count_points_paths(paths):
    acc = defaultdict(int)
    for path in paths:
        for n in path:
            acc[n] += 1
    return acc


def find_paths(G, targets, stack_shape, min_count=1, min_path_length=10):
    paths_dict = nx.multi_source_dijkstra_path(G, targets, )
    
    #reverse order of points in paths, so that they start at tips 
    paths_dict = {path[-1]:path[::-1] for path in paths_dict.values() if len(path) >= min_path_length}
    paths = list(paths_dict.values())
    points = count_points_paths(paths)

    qstack = np.zeros(stack_shape)  #Это встречаемость точек в путях
    for p, val in points.items():
        if val >= min_count:
            qstack[p] = np.log(val)
    return qstack, paths_dict

def get_tips(g):
    return {n for n in g.nodes if len(list(g.successors(n))) == 0}
            
def get_roots(g):
    return {n for n in g.nodes if len(list(g.predecessors(n))) < 1}

def get_sorted_roots(g):
    return sorted(get_roots(g), 
                  key=lambda r: len(filter_graph(g, lambda n: n['root']==r)), 
                  reverse=True,)

def get_branch_points(g):
    return {n for n in gx.nodes if len(list(gx.successors(n))) > 1}     

def batch_compose_all(tip_paths, batch_size=10000, verbose=True):
    graphs = []
    for i, tp in enumerate(tqdm(tip_paths, disable=not verbose)):
        graphs.append(path_to_graph(tp))
        if i % batch_size == 0:
            gx_all = nx.compose_all(graphs)
            graphs = [gx_all]
    return nx.compose_all(graphs)    

def filter_graph(graph, func = lambda node: True ):
    "returns a view on graph for the nodes satisfying the condition defined by func(node)"
    good_nodes = (node for node in graph if func(graph.nodes[node]))
    return graph.subgraph(good_nodes)


def get_attrs_by_nodes(G, arr, func=None):
    nodesG = np.array(G.nodes())
    attrs = arr[nodesG[:,0], nodesG[:,1], nodesG[:,2]]
    if func is not None:
        func_vect = np.vectorize(func)
        attrs = func_vect(attrs)
    return {tuple(node): attr for node, attr in zip(nodesG, attrs)}