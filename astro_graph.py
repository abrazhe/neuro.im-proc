from collections import defaultdict

import numpy as np
import networkx as nx

from tqdm.auto import tqdm


class AstroGraph(nx.Graph):

    def __init__(self, graph):
        self.graph = graph


    @classmethod
    def batch_compose_all(cls, tip_paths, batch_size=10000, verbose=True):
        graphs = []
        for i, tp in enumerate(tqdm(tip_paths, disable=not verbose)):
            graphs.append(AstroGraph.path_to_graph(tp))
            if i % batch_size == 0:
                gx_all = nx.compose_all(graphs)
                graphs = [gx_all]
        return cls(nx.compose_all(graphs))


    @staticmethod
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

    @property
    def type(self):
        return type(self.graph)

    @property
    def nodes(self, data=False):
        return self.graph.nodes(data=data)

    # def nodes(self):
    #     return self.graph.nodes()

    @property
    def edges(self, data=False):
        return self.graph.edges(data=data)

    def predecessors(self, node):
        return self.graph.predecessors(node)

    def successors(self, node):
        return self.graph.successors(node)

    def get_tips(self):
        return {n for n in self.nodes if len(list(self.successors(n))) == 0}


    def get_roots(self):
        return {n for n in self.nodes if len(list(self.predecessors(n))) < 1}


    def get_sorted_roots(self):
        return sorted(self.get_roots(),
                  key=lambda r: len(self.filter_graph(lambda n: n['root']==r)),
                  reverse=True,)


    def get_branches(self):
        # self.graph
        raise Exception('ERROR')


    def get_branch_points(self):
        return {n for n in self.nodes if len(list(self.successors(n))) > 1}


    def get_attrs_by_nodes(self, arr, func=None):
        nodesG = np.array(self.nodes())
        attrs = arr[nodesG[:,0], nodesG[:,1], nodesG[:,2]]
        if func is not None:
            func_vect = np.vectorize(func)
            attrs = func_vect(attrs)
        return {tuple(node): attr for node, attr in zip(nodesG, attrs)}


    def subgraph(self, nodes):
        return self.graph.subgraph(nodes)


    def add_edge(self, start, end, **attr):
        self.graph.add_edge(start, end, **attr)


    def add_node(self, node, **attr):
        self.graph.add_node(node, **attr)



    def check_for_cycles(self, verbose=False):
        try:
            cycle = nx.find_cycle(self.graph)
            if verbose:
                print('Found a cycle:', cycle)
            return cycle
        except nx.exception.NetworkXNoCycle:
            if verbose:
                print('No cycles!')
            return None

    @staticmethod
    def find_paths(graph, targets, stack_shape, min_count=1, min_path_length=10):
        paths_dict = nx.multi_source_dijkstra_path(graph, targets, )

        #reverse order of points in paths, so that they start at tips
        paths_dict = {path[-1]:path[::-1] for path in paths_dict.values() if len(path) >= min_path_length}
        paths = list(paths_dict.values())
        points = AstroGraph.count_points_paths(paths)

        qstack = np.zeros(stack_shape)  #Это встречаемость точек в путях
        for p, val in points.items():
            if val >= min_count:
                qstack[p] = np.log(val)
        return qstack, paths_dict


    def filter_graph(self, func = lambda node: True):
        "returns a view on graph for the nodes satisfying the condition defined by func(node)"
        good_nodes = (node for node in self.graph if func(self.nodes[node]))
        return self.subgraph(good_nodes)


    #### VIZUALIZATIONS


    def view_graph_as_shapes(self, viewer, color=None, kind='points', name=None):
        """
        display nodes of graph g in napari viewer as points or as lines
        """
        if color is None:
            color = np.random.rand(3)
        pts = np.array(self.nodes)

        kw = dict(face_color=color, edge_color=color, blending='translucent_no_depth', name=name)
        #kw = dict(face_color=color, edge_color=color,  name=name)
        if kind == 'points':
            viewer.add_points(pts, size=1, symbol='square', **kw)
        elif kind == 'path':
            viewer.add_shapes(pts, edge_width=0.5, shape_type='path', **kw)

    def view_graph_as_colored_image(self, shape,
                                    viewer=None, name=None,
                                    root_chooser=lambda r: True,
                                    change_color_at_branchpoints=False):
        """
        Convert a graph to a colored 3D stack image and add it to a napari viewer.
        if the viewer instance is None, just return the colored 3D stack
        """
        paths = self.graph_to_paths(root_chooser=root_chooser)
        stack = self.paths_to_colored_stack(paths, shape, change_color_at_branchpoints)
        if viewer is not None:
            viewer.add_image(stack, channel_axis=3, colormap=['red','green','blue'], name=name)
            return viewer
        else:
            return stack

    def graph_to_paths(self, min_path_length=1, root_chooser=lambda r:True):
        """
        given a directed graph, return a list of a lists of nodes, collected
        as unbranched segments of the graph
        """

        roots = self.get_roots()

        def _acc_segment(root, segm, accx):
            if segm is None:
                segm = []
            if accx is None:
                accx = []
            children = list(self.successors(root))

            if len(children) < 1:
                accx.append(segm)
                return

            elif len(children) == 1:
                c = children[0]
                segm.append(c)
                _acc_segment(c, segm, accx)

            if len(children) > 1:
                #segm.append(root)
                accx.append(segm)
                for c in children:
                    _acc_segment(c, [root, c], accx)

        acc = {}
        for root in roots:
            if root_chooser(root):
                px = []
                _acc_segment(root, [], px)
                acc[root] = [s for s in px if len(s) >= min_path_length]
        return acc


    def __str__(self):
        return str(self.graph)


    ## USEFUL FUNCTIONS


    @staticmethod
    def count_points_paths(paths):
        acc = defaultdict(int)
        for path in paths:
            for n in path:
                acc[n] += 1
        return acc


    @staticmethod
    def paths_to_colored_stack(paths, shape, change_color_at_branchpoints=False):
        #colors = np.random.randint(0,255,size=(len(paths),3))
        stack = np.zeros(shape + (3,), np.uint8)
        for root in paths:
            color =  np.random.randint(0,255, size=3)
            for kc,pc in enumerate(paths[root]):
                if change_color_at_branchpoints:
                    color = np.random.randint(0,255, size=3)
                for k,p in enumerate(pc):
                    #print(k, p)
                    stack[tuple(p)] = color
        return stack
