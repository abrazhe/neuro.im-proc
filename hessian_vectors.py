import numpy as np

import hessian_cecp as hcecp


def plot_from_sato(viewer, data, vectors, axis=0, index=1):
    """
    *args : list
        list of arguments. Depending on their number they are parsed to::
            data/length, Vf/Vfx/vectors/xyzuvc/uvc
    """

    lengths = hcecp.percentile_rescale(data)**0.5
    length = lengths.ravel()[::index]

    Vf = vectors[..., axis][..., ::-1]
    if Vf.shape[-1] == 3:
        ndim = 3
        V = Vf[..., 0]
        U = Vf[..., 1]
        C = Vf[..., 2]
    elif Vf.shape[-1] == 2:
        ndim = 2
        V = Vf[..., 0]
        U = Vf[..., 1]


    if ndim == 2:
        u = U.ravel()[::index]
        v = V.ravel()[::index]

        nr, nc = (1, U.shape[0]) if U.ndim == 1 else U.shape
        indexgrid = np.meshgrid(np.arange(nc), np.arange(nr))

        x, y = [np.ravel(a)[::index] for a in indexgrid]
        x1, y1 = u*length, v*length

    elif ndim == 3:
        u = U.ravel()[::index]
        v = V.ravel()[::index]
        c = C.ravel()[::index]

        nr, nc, nd = (1, U.shape[0]) if U.ndim == 1 else U.shape
        indexgrid = np.meshgrid(np.arange(nc), np.arange(nr), np.arange(nd))

        x, y, z = [np.ravel(a)[::index] for a in indexgrid]
        x1, y1, z1 = u*length, v*length, c*length



    vectors = np.zeros((u.shape[0], 2, ndim))
    vectors[..., 0, 0] = y
    vectors[..., 0, 1] = x
    vectors[..., 1, 0] = y1
    vectors[..., 1, 1] = x1

    if ndim == 3:
        vectors[..., 0, 2] = z
        vectors[..., 1, 2] = z1

    add_hessian_vectors(viewer, vectors, length)

    return viewer


def add_hessian_vectors(viewer, vectors, length):
    properties = {'length': length}

    viewer.add_vectors(vectors, edge_width=0.2,
                       length=1,  properties=properties,
                       edge_color='length', edge_colormap='inferno')
    return viewer
