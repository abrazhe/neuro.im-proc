import numpy as np


def add_hessian_vectors(viewer, vectors, lengths, axis=0, index=1):
    Vfx = vectors[..., axis][..., ::-1]

    V = Vfx[...,0] # row directions (Y)
    U = Vfx[...,1] # col directions (X)
    C = Vfx[...,2]

    nr, nc, nd = (1, U.shape[0]) if U.ndim == 1 else U.shape
    indexgrid = np.meshgrid(np.arange(nc), np.arange(nr), np.arange(nd))
    x, y, z = [np.ravel(a)[::index] for a in indexgrid]

    u = U.ravel()[::index]
    v = V.ravel()[::index]
    c = C.ravel()[::index]
    length = lengths.ravel()[::index]

    x1, y1, z1 = u*length, v*length, c*length

    vectors = np.zeros((u.shape[0], 2, 3))
    vectors[...,0, 0] = y
    vectors[...,0, 1] = x
    vectors[...,0, 2] = z
    vectors[...,1, 0] = y1
    vectors[...,1, 1] = x1
    vectors[...,1, 2] = z1

    properties = {'length': length}

    viewer.add_vectors(vectors, edge_width=0.2,
                       length=1,  properties=properties,
                       edge_color='length', edge_colormap='inferno')
    return viewer
