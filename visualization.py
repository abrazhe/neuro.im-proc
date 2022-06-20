import graph_utils as gu    
import numpy as np
    
def view_graph_as_shapes(g, viewer, color=None, kind='points', name=None):
    """
    display nodes of graph g in napari viewer as points or as lines
    """
    if color is None:
        color = np.random.rand(3)
    pts = np.array(g.nodes)
    
    kw = dict(face_color=color, edge_color=color, blending='translucent_no_depth', name=name)
    #kw = dict(face_color=color, edge_color=color,  name=name)
    if kind == 'points':
        viewer.add_points(pts, size=1, symbol='square', **kw)
    elif kind == 'path':
        viewer.add_shapes(pts, edge_width=0.5, shape_type='path', **kw)
        
def view_graph_as_colored_image(g,  shape, 
                                viewer=None, name=None, 
                                root_chooser=lambda r: True,
                                change_color_at_branchpoints=False):
    """
    Convert a graph to a colored 3D stack image and add it to a napari viewer.
    if the viewer instance is None, just return the colored 3D stack
    """
    paths = graph_to_paths(g, root_chooser=root_chooser)
    stack = paths_to_colored_stack(paths, shape, change_color_at_branchpoints)
    if viewer is not None:
        viewer.add_image(stack, channel_axis=3, colormap=['red','green','blue'], name=name)
        return viewer
    else:
        return stack
        
def graph_to_paths(g, min_path_length=1, root_chooser=lambda r:True):
    """
    given a directed graph, return a list of a lists of nodes, collected
    as unbranched segments of the graph
    """

    roots = gu.get_roots(g)
    
    def _acc_segment(root, segm, accx):
        if segm is None:
            segm = []
        if accx is None:
            accx = []
        children = list(g.successors(root))
        
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