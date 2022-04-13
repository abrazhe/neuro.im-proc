import numpy as np

import astromorpho as astro


def sato2napari_vectors(data, vectors, axis=0, index=1):
    """
    *args : list
        list of arguments. Depending on their number they are parsed to::
            data/length, Vf/Vfx/vectors/xyzuvc/uvc
    """

    lengths = astro.enh.percentile_rescale(data)**0.5
    length = lengths.ravel()[::index]

    Vf = vectors[..., axis][..., ::-1]
    ndim = Vf.shape[-1]

    if ndim not in [2, 3]:
        return Exception('ERROR! Array of vectors should have 2 or 3 dimensions')


    V = Vf[..., 0]
    U = Vf[..., 1]
    if ndim == 3:
        C = Vf[..., 2]


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

    return vectors, length


def add_hessian_vectors(viewer, vectors, length):
    properties = {'length': length}

    viewer.add_vectors(vectors, edge_width=0.2,
                       length=1,  properties=properties,
                       edge_color='length', edge_colormap='inferno')
