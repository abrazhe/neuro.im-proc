import numpy as np
import napari


def draw_nodes(pos, nodelist):
    return np.asarray([pos[n] for n in nodelist])


def draw_edges(pos, edgelist):
    return np.asarray([[pos[n1], pos[n2]] for n1, n2 in edgelist])