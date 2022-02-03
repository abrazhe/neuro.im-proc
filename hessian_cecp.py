# Hessian-based complexity-entropy
# bonus: turbo-snails in 3D

import itertools as itt
import numpy as np
from numpy import linalg
from numpy import pi as π
from numpy import exp

from scipy import ndimage as ndi
from numba import jit

from matplotlib import pyplot as plt
from tqdm.auto import tqdm


from skimage import filters as skfilters
from skimage import feature as skf
from skimage import color as skcolor

_with_tensorflow = False

try:
    import tensorflow as tf
    _with_tensorflow = True
    eigh = tf.linalg.eigh
except:
    eigh = np.linalg.eigh
    pass

logfn = np.log2

def jmax(M):
    "Max Jensen-Shannon divergence in a system of M states"
    #return -0.5*((M+1)*logfn(M+1)/M -2*logfn(2*M) + logfn(M))
    return -0.5*((M+1)*logfn(M+1)/M -logfn(4*M))


def shannon(P):
    "Shannon entropy for distribution {P}"
    return -np.sum(P[P>0]*logfn(P[P>0]))


def cecp(P):
    "cecp: calculate complexity-entropy causality pair for distribution {P}"
    M = len(P)
    Pe = np.ones(M)/M
    Sp = shannon(P)
    Smax = logfn(M)
    J = shannon(0.5*(P+Pe)) - 0.5*(Sp + Smax)
    Hr = Sp/Smax
    Cr = (J/jmax(M))*Hr
    return (Hr,Cr)


def hessian_by_dog(img, sigma):
    ndim = np.ndim(img)
    ax_pairs = itt.combinations_with_replacement(range(ndim),2)
    sigma = sigma/np.sqrt(2)
    trunc = 6 # default
    if np.any(sigma*trunc < 3):
        trunc = 3/sigma
    def dog(m,k):
        o = np.zeros(ndim, np.int)
        o[k] = 1
        return ndi.gaussian_filter(m, sigma, order=o, truncate=trunc)
    double_dog = lambda axp: dog(dog(img, axp[0]),axp[1])
    out = [double_dog(axp) for axp in ax_pairs]
    return out

def hessian_eigen_decomp(H):
    Hmat = skf.corner._hessian_matrix_image(H)
    w,v = eigh(Hmat)
    return w,v


def sato3d(img, sigma,
           gamma12=0.5, gamma23=0.5, alpha=0.25,
           hessian_variant='gradient_of_smoothed',
           mode='nearest',
           do_brightness_correction=False,
           return_vectors=False,
          ):
    if hessian_variant.lower() == 'dog':
        H = hessian_by_dog(img, sigma)
    elif hessian_variant.lower() == 'gradient_of_smoothed':
        H = skf.hessian_matrix(img, sigma, mode=mode,order='rc')
    else:
        print('Unknown Hessian variant')
        return

    Hmat = skf.corner._hessian_matrix_image(H)
    #w = eigvalsh(Hmat)

    w,v = eigh(Hmat) # both vectors and values
    w = w[...,::-1]
    v = v[...,::-1]

    sh = img.shape
    out = np.zeros(sh)

    lam1,lam2,lam3 = [w[...,i] for i in range(3)]

    ratio1 = np.where(lam3!=0, lam2/(1e-6 + lam3),0)
    ratio2 = lam1/(1e-6 + np.abs(lam2))

    #ratio1 = np.abs(ratio1)
    #ratio2 = np.abs(ratio2)

    out = np.where(lam1 < 0, np.abs(lam3)*np.abs(ratio1)**gamma23*np.abs(1 + ratio2)**gamma12,
                np.where((lam2 < 0) & (lam1 < np.abs(lam2)/alpha), np.abs(lam3)*np.abs(ratio1)**gamma23*np.abs(1 - alpha*ratio2)**gamma12,0))

    if do_brightness_correction:
        S = (np.sum(w**2,axis=-1))**0.5
        gamma = np.percentile(S,99.9)/2
        gamma_sq = 2*gamma**2
        out = out*(1-exp(-S**2/gamma_sq))

    if return_vectors:
        return out, np.asarray(v)
    else:
        return out

from imfun import core
def frangi(img, sigma,
             Heigvalues=None,
             alpha=0.5, beta=0.5, gamma=None,
             hessian_variant='gradient_of_smoothed',
             mode='nearest',
             return_vectors=False,
             return_blobness=False
           ):

    have_Hvectors = False
    if Heigvalues is None:
        if hessian_variant.lower() == 'dog':
            H = hessian_by_dog(img, sigma)
        elif hessian_variant.lower() == 'gradient_of_smoothed':
            H = skf.hessian_matrix(img, sigma, mode=mode,order='rc')
        else:
            print('Unknown Hessian variant')
            return
        Hmat = skf.corner._hessian_matrix_image(H)
        #w = eigvalsh(Hmat)
        w,v = eigh(Hmat) # both vectors and values
        w = np.array(w[...,::-1])
        v = np.array(v[...,::-1])
        have_Hvectors = True
    else:
        w = Heigvalues
        v = None

    ksort = np.argsort(np.abs(w),axis=-1)
    w = np.take_along_axis(w,ksort,axis=-1)

    sh = img.shape
    contrast = np.zeros(sh)
    blobness = np.zeros(sh)

    S = (np.sum(w**2,axis=-1))**0.5
    if gamma is None:
        gamma = np.percentile(S,99.9)/2
        #gamma = np.max(S)/2


    alpha_sq = 2*alpha**2
    beta_sq = 2*beta**2
    gamma_sq = 2*gamma**2

    @jit
    def frangi_contrast3d(contrast,blobness):
        for z in range(sh[0]):
            for r in range(sh[1]):
                for c in range(sh[2]):
                    lam0,lam1,lam2 = w[z,r,c]
                    if (lam1 < 0) and (lam2 < 0):
                        Ra2 = (lam1/lam2)**2
                        Rb2 = (lam0/np.abs(lam1*lam2)**0.5)**2
                        Sx = (1-exp(-S[z,r,c]**2/gamma_sq))
                        expRb  = exp(-Rb2/beta_sq)
                        contrast[z,r,c] = (1 - exp(-Ra2/(alpha_sq)))*Sx*expRb
                        blobness[z,r,c] = Sx*(1-expRb)
    @jit
    def frangi_contrast2d(contrast,blobness):
            for r in range(sh[0]):
                for c in range(sh[1]):
                    lam0,lam1 = w[r,c]
                    if (lam1 < 0):
                        Rb2 = (lam0/lam1)**2
                        Sx = (1-exp(-S[r,c]**2/gamma_sq))
                        expRb  = exp(-Rb2/beta_sq)
                        contrast[r,c] = Sx*expRb
                        blobness[r,c] = Sx*(1-expRb)


    if len(sh) == 3:
        frangi_contrast3d(contrast, blobness)
    elif len(sh) == 2:
        frangi_contrast2d(contrast, blobness)
    else:
        print(f"Don't know how to process data with {np.ndim(img)} dimensions")
        return

    out = (contrast, )

    if return_vectors and have_Hvectors:
        out = out + (v, )

    if return_blobness:
        out = out + (blobness, )

    if len(out) < 2:
        out = out[0]

    return out


def sato2d(img, sigma, gamma12=0.5, alpha=0.25, hessian_variant='gradient_of_smoothed',return_vectors=False):
    if hessian_variant.lower() == 'dog':
        H = hessian_by_dog(img, sigma)
    elif hessian_variant.lower() == 'gradient_of_smoothed':
        H = skf.hessian_matrix(img, sigma,mode='nearest',order='rc')
    else:
        print("unknown hessian variant")
        return
    Hmat = skf.corner._hessian_matrix_image(H)
    w,v = eigh(Hmat) # both eigenvectors and eigenvalues (in ascending order)
    w = w[...,::-1]  # flip to descending order
    v = v[...,::-1]  # flip to descending order

    sh = img.shape
    out = np.zeros(sh)

    # here lam1 >= lam2
    lam1,lam2 = [w[...,i] for i in range(2)]

    ratio2 = lam1/(1e-6 + np.abs(lam2))

    out = np.where(lam1 < 0,
                   np.abs(lam2)*np.abs(1 + ratio2)**gamma12,
                   np.where((lam2 < 0) & (lam1 < np.abs(lam2)/alpha), np.abs(lam2)*np.abs(1 - alpha*ratio2)**gamma12,0))
    if return_vectors:
        return out, v
    else:
        return out

@jit
def cartesian_to_polar(x,y):
    r = (x**2 + y**2)**0.5
    phi = np.arctan2(x,y)
    return r, phi

def percentile_rescale(stack, low=0.5, high=99.95):
    px = np.percentile(stack[stack>0],(low, high))
    return np.clip((stack-px[0])/(px[1]-px[0]), 0,1)

def simple_rescale(stack):
    mn,mx = np.min(stack), np.max(stack)
    if mn == mx:
        return np.zeros_like(stack)
    return (stack - mn)/(mx-mn)


def corrupt_ppixels(img,p=0.1):
    nrows,ncols = img.shape
    Npx = nrows*ncols
    Ncorrupt = int(round(Npx*p))
    #rr = uniform(nrows,size=Ncorrupt).astype(int)
    #cc = uniform(ncols,size=Ncorrupt).astype(int)
    imgc = img.copy()
    for r in range(nrows):
        for c in range(ncols):
            if uniform() < p:
                imgc[r,c] = uniform()
    return imgc

@jit
def make_grid_pattern(size=512,freq=0.05,incl=0.5, rectify=True):
    sh = size,size
    out = np.zeros(sh)
    for r in range(size):
        for c in range(size):
            out[r,c] = sin(2*π*(incl*freq*c + (1-incl)*freq*r))
    if rectify:
        out = np.maximum(out, 0)
    return out

def hessian_cecp2d(img, sigma, nbins=200, hessian_variant='gradient_of_smoothed', j=0, gamma=1, reverse_weights=False):
    sato, vf = sato2d(img, sigma, hessian_variant=hessian_variant, return_vectors=True)
    sato_n = percentile_rescale(sato, 0.1, 99.9)
    if reverse_weights:
        sato_n = 1-sato_n
    _,angles = cartesian_to_polar(vf[...,0,:],vf[...,1,:])
    phi = angles[...,j]
    phi2 = np.where(phi < 0, π+phi, phi)
    hx,_ = np.histogram(np.ravel(phi2)**gamma, nbins, weights=np.ravel(sato_n))
    return cecp(hx/np.sum(hx))

def get_lower_envelope(N=200,step=0.01):
    # looks fine
    acc = []
    for p in np.arange(0,1+0.1*step,step):
        d = np.zeros(N)
        d[0] = p
        d[1:] = (1-p)/(N-1)
        acc.append(cecp(d))
    return np.array(acc)

def get_higher_envelope(N=200,step=1,verbose=False):
    "doesn't work..."
    acc = []
    for ncut in tqdm(range(1,N,step), disable=not verbose):
        p = np.zeros(N)
        p[:ncut] = 1/(ncut)
        acc.append(cecp(p))
    return np.array(acc)

def get_cecp_envelope(N,step_lower=0.01, step_upper=1):
    lower = get_lower_envelope(N,step_lower)
    upper = get_higher_envelope(N,step_upper)
    return np.vstack((lower,upper))

def test_powerlaw_distributions(N=200,alpha_range=np.linspace(-100,100,10000)):
    acc = []
    for alpha in alpha_range:
        p = np.linspace(1,N+1,N)**alpha
        acc.append(cecp(p/np.sum(p)))
    return np.array(acc)

@jit
def cartesian_to_polar3d(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2((x**2 + y**2)**0.5, z)
    phi = np.arctan2(y,x)
    return r, theta, phi


@jit
def polar_to_cartesian3d(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z


def flip_directions(alphas):
    return np.where(alphas < 0, alphas+π, alphas)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


from ucats.patches import make_grid

def hessian_cecp3d(stack, sigma,
                   nbins=50,
                   hessian_variant='gradient_of_smoothed',
                   j=0,
                   gamma=1,
                   reverse_weights=False,
                   spatial_binning=-1,
                   spatial_overlap=1,
                   verbose=False,
                   with_plot=False,
                   sato_n=None, Vf=None):
    if sato_n is None or Vf is None:
        sato, Vf = sato3d(stack, sigma, hessian_variant=hessian_variant, return_vectors=True)
        sato_n = percentile_rescale(sato, 0.1, 99.9)
        if reverse_weights:
            sato_n = 1-sato_n
    #_,angles = cartesian_to_polar(vf[...,0],vf[...,1])
    Vfj = Vf[...,j] # first vectors point along bright tubular structures


    #hx,_ = np.histogram(ravel(phi2)**gamma, nbins, weights=ravel(sato_n))

    #thetas = thetas.reshape(stack.shape)
    #rhos = rhos.reshape(stack.shape)
    if spatial_binning <= 0:
        slices = [(slice(None),slice(None),slice(None))]
    else:
        slices = make_grid(stack.shape, spatial_binning,spatial_overlap)

    hc_acc = []
    window_weights = np.array([np.mean(sato_n[w]) for w in slices])
    window_weights /= np.sum(window_weights)
    for w,ww in zip(tqdm(slices, disable=not verbose),window_weights):
        #print(w, thetas.shape, rhow.shape, thetas_w)
        # find collective directions of the vectors and use the rotated projection

        u,s,vh = linalg.svd(Vfj[w].reshape(-1,3), full_matrices=False)
        rr,thetas,rhos = cartesian_to_polar3d(u[...,1],u[...,2],u[...,0]) # x,y,z
        rhos = flip_directions(rhos)

        hx, xedges, yedges = np.histogram2d(np.ravel(thetas),
                                            np.ravel(rhos),
                                            bins=(nbins,nbins),
                                            weights=np.ravel(sato_n[w]))
        hx = np.ravel(hx)
        hc_acc.append(cecp(hx/(1e-6 + np.sum(hx))) + (ww,))

    if with_plot:

        f,axs = plt.subplots(1,2,sharex=True)
        axs[0].set_title('Theta')
        axs[1].set_title('Phi')
        _ = axs[0].hist(np.ravel(thetas), 200, weights=np.ravel(sato_n), density=True, histtype='step',lw=3,)
        _ = axs[1].hist(np.ravel(rhos), 200, weights=np.ravel(sato_n), density=True, histtype='step',lw=3,)

        selected_points = np.dot(u,np.diag(s))[np.ravel(sato_n) >= 0.99]
        N = len(selected_points)
        if N > 10000:
            selected_points = np.random.permutation(selected_points)[:10000]
        fig = plt.figure(figsize=(9,8))
        ax = fig.add_subplot(111, projection='3d')

        theta, phi = np.meshgrid(np.linspace(0,π,100), np.linspace(-π,π,100))
        r = np.ones(theta.size)

        x,y,z = polar_to_cartesian3d(r, np.ravel(theta), np.ravel(phi))

        ax.scatter(x,y,z,s=1,color='gray',alpha=0.5)

        ax.scatter(selected_points[:,2], selected_points[:,1], selected_points[:,0],s=10,alpha=0.5)

        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)



    if spatial_binning <=0:
        return hc_acc[0]
    else:
        return np.array(hc_acc)


def prep_hc_axes(nbins=100, ax=None):
    if ax is None:
        f,ax = plt.subplots(1,1)
    hc_env = get_cecp_envelope(nbins)
    ax.plot(hc_env[:,0],hc_env[:,1],color='gray',ls=':')
    ax.set_xlabel('H')
    ax.set_ylabel('C')
    return ax

# snails in 3d

@jit
def make_trajectory3d(x0, force_field, xtrails,
                      m=1, friction=1, T=100, dt=1,
                      tol=0.01,
                      max_speed=1,
                      speed_gamma=1,
                      wrap_bounds=False):
    traj = np.zeros((T,) + (len(x0),))
    #print(traj.shape)

    nz,nr,nc,nd = force_field.shape

    F = force_field
    a0 = F[int(x0[0]),int(x0[1]),int(x0[2])]/m
    #v0 = 0.1*np.random.rand(3)
    v0 = a0
    #print(a0)
    traj[0] = x0
    traj[1] = x0 + 0*v0*dt + a0*dt**2/2
    dt2 = dt**2
    for i in range(1, T-1):
        z,r,c = traj[i]

        #zi,ri,ci = np.int(z), np.int(r),np.int(c)
        zi,ri,ci = np.int(round(z)), np.int(round(r)),np.int(round(c))

        if wrap_bounds:
            zi = zi%nz
            ri = ri%nr
            ci = ci%nc
        else:
            if zi >= nz or ri >= nr or c >= nc or z < 0 or r < 0 or c < 0:
                break

        speed = np.sum((traj[i]-traj[i-1])**2)**0.5/dt
        if speed < tol:
            break
        if speed > 0.1:
            xtrails[zi,ri,ci] = xtrails[zi,ri,ci] + speed**speed_gamma
        a = F[zi,ri,ci]/m
        #traj[i+1] = traj[i] + a*dt
        #print(a)
        update = (2-friction*dt2/m)*traj[i] - (1-friction*dt2/m)*traj[i-1] + a*dt2 - traj[i]
        update_norm = np.sum(update**2)**0.5
        if update_norm > max_speed:
            update = update/update_norm
        traj[i+1] = traj[i] + update

    return traj#[:i]



def make_trails_stack(out_shape,
                      Nparticles=100,
                      sigma=1.5,
                      field_mag=1,
                      random_weight=0.5,
                      drift=(1,1,1),
                      with_progress=False):

    F = random_weight*np.random.randn(*(out_shape + (3,))) + (1-random_weight)*asarray(drift)
    F = F*field_mag
    trails = np.zeros(out_shape)
    starts = [tuple(np.random.randint(n) for n in trails.shape) for i in range(Nparticles)]
    for x0 in tqdm(starts, disable=not with_progress):
        make_trajectory3d(x0, F, trails, T=1000, dt=0.2, gamma_speed=0.01,wrap_bounds=True)
    #trails = percentile_rescale(trails)

    return percentile_rescale(ndi.gaussian_filter(trails, sigma)) + 0.05*np.random.randn(*out_shape)

@jit
def collect_trails(field, mask=None, m=1, mag_th=0.05, dt=0.25, friction=0.05, T=1000,
                   speed_gamma=1,
                   max_speed=1):
    sh = field.shape[:-1]
    trails = np.zeros(sh)
    mag = np.sum(field**2,axis=-1)**0.5
    if mask is None:
        mask = mag > mag_th
    for z in range(sh[0]):
        for r in range(sh[1]):
            for c in range(sh[2]):
                if mask[z,r,c]:
                    make_trajectory3d(np.array([z,r,c]), field, trails,
                                      m=m, dt=dt, T=T,
                                      speed_gamma=speed_gamma,
                                      max_speed=max_speed,
                                      friction=friction)
    return trails


def turbosnail_vesselness(stack, sigma, amp=5, mask=None,start_threshold=0.99, vfield_gamma=1.0, **kwargs):
    sato, Vf = sato3d(stack, sigma, hessian_variant='gradient_of_smoothed', return_vectors=True)
    Vfj = Vf[...,0][...,::-1]
    weights = percentile_rescale(sato)**vfield_gamma
    #mask = (lam[...,0] < 0) & ((lam[...,1] <= 0) | (lam[...,1] < np.abs(lam[...,0])/0.5))
    #weights = percentile_rescale(np.abs(lam[...,0]))*mask
    if mask is None:
        mask = weights >=start_threshold
    field = amp*Vfj*weights[...,np.newaxis]
    trails = 0.5*(collect_trails(field,mask, **kwargs) + collect_trails(-field,mask,**kwargs))
    return trails



@jit
def make_trajectory3d_food(x0, force_field, xtrails, speed_trails, food_field, vizit_counts,
                           m=1, friction=0.1, T=100, dt=1,tol=0.1,
                           max_speed=np.sqrt(2),
                           food_memory=3,
                           ignore_transitions=False,
                           max_stop_time=3,
                           wrap_bounds=False):
    traj = np.zeros((T,) + (len(x0),))

    nz,nr,nc,nd = force_field.shape

    F = force_field
    traj[0] = x0

    food_cargo = np.zeros(food_memory)
    food_idx = 0
    food_acc = 0
    food_count = 0
    zi_p, ri_p, ci_p = -1,-1,-1
    k = 0
    dt2 = dt**2

    min_speed_count = 0

    for i in range(0, T-1):
        z,r,c = traj[i]
        zi,ri,ci = np.int(round(z)), np.int(round(r)),np.int(round(c))


        if wrap_bounds:
            zi = zi%nz
            ri = ri%nr
            ci = ci%nc
        else:
            if zi >= nz or ri >= nr or c >= nc or z < 0 or r < 0 or c < 0:
                break

        a = F[zi,ri,ci]/m

        if i == 0:
            #next_state = traj[i] + 0*a*dt + a*dt**2/2
            #speed = np.sum((next_state - traj[i])**2)**0.5
            update = 0*a*dt + a*dt2/2
        else:
            #speed_prev = np.sum((traj[i-1] - traj[i])**2)**0.5
            #update = (2-friction*dt2/m)*traj[i] - (1-friction*dt2/m)*traj[i-1] + a*dt2 - traj[i]
            update = (2-friction)*traj[i] - (1-friction)*traj[i-1] + a*dt2 - traj[i]
            #update = 2*traj[i] - traj[i-1] + a*dt2 - traj[i]
            #speed = np.sum((next_state - traj[i-1])**2)**0.5/2

        update_norm = np.sum(update**2)**0.5
        speed = update_norm

        if speed > max_speed:
            update = max_speed*update/update_norm

        traj[i+1] = traj[i] + update

        if speed/dt < tol:
            min_speed_count +=1

        if min_speed_count > max_stop_time:
            break

        #food_idx = i%food_memory
        if ignore_transitions |  (zi != zi_p) | (ri != ri_p) | (ci != ci_p):

            food_new = food_field[zi,ri,ci]
            food_idx = (food_idx+1)%food_memory # point to next position
            food_tail = food_cargo[food_idx] # either zero or end of ring
            food_cargo[food_idx-1] = food_new

            food_count = min(food_count+1, food_memory) # visited or length of ring
            food_acc += food_new - food_tail

            if True | (food_count >= food_memory):
                xtrails[zi,ri,ci] += food_acc/food_count
                vizit_counts[zi,ri,ci] += 1
            food_count += 1

            speed_trails[zi,ri,ci] += speed

        #traj[i+1] = traj[]

        k+=1
        zi_p,ri_p,ci_p = zi,ri,ci

    return traj[:k]

@jit
def collect_trails_food(field, mask, food_field, food_memory=3,
                        max_speed=np.sqrt(2),
                        tol = 0.01,
                        m=1, dt=0.25, friction=1, T=1000):
    sh = field.shape[:-1]

    xtrails = np.zeros(sh)
    vtrails = np.zeros(sh)
    counts = np.zeros(sh)
    traj_lengths= np.zeros(sh)

    for z in range(sh[0]):
        for r in range(sh[1]):
            for c in range(sh[2]):
                if mask[z,r,c]:
                    t = make_trajectory3d_food(np.array([z,r,c]),
                                               field,
                                               xtrails,
                                               vtrails,
                                               food_field,
                                               counts,
                                               food_memory=food_memory,
                                               max_speed=max_speed,
                                               friction=friction,

                                               tol=tol,
                                               m=m, dt=dt, T=T)
                    traj_lengths[z,r,c] = len(t)
    return xtrails, vtrails,counts,traj_lengths

def mean_pair(x1,x2):
    return (x1+x2)/2

def outside_boundary(mask):
    return ndi.binary_dilation(mask)^mask

def inside_boundary(mask):
    return ndi.binary_erosion(mask)^mask


import ucats

def turbosnail_vesselness_food(stack, sigma, amp=1, mask=None,
                               start_threshold=0.5,
                               field=None,
                               vfield_gamma=1,
                               min_counts=1,
                               min_speed=1,
                               min_traj_len = 11,
                               niters=1,
                               vesselness='frangi',
                               verbose=False,
                               **kwargs):



    if field is None:
        if vesselness.lower() == 'sato':
            sato, Vf = sato3d(stack, sigma, hessian_variant='gradient_of_smoothed', return_vectors=True)
            weights = percentile_rescale(sato, 0.1, 99.9)**vfield_gamma
        elif vesselness.lower() == 'frangi':
            my_frangi,Vf,blobness = frangi3d(stack,sigma,return_vectors=True,beta=1.5,

                                             return_blobness=True)
            blobness2 = percentile_rescale(ndi.gaussian_filter(blobness, sigma), 0.1, 99.99)**2
            frangi_weights = percentile_rescale(my_frangi, 0.1, 99.9)
            mx = ucats.masks.threshold_object_size((frangi_weights > 0.5)*(blobness2 < 0.1), 27)
            saved_blobs = ucats.masks.select_overlapping(blobness2 > 0.1, ndi.binary_dilation(mx))
            frangi_weights = frangi_weights*(~(blobness2>0.1) + saved_blobs)
            weights = frangi_weights

        Vfj = Vf[...,0][...,::-1]


        field = amp*Vfj*weights[...,np.newaxis]
    else:
        weights = np.linalg.norm(field, axis=-1)


    if mask is None:
        mask = weights >=start_threshold


    xtrails_acc = np.zeros(stack.shape)
    counts_acc = np.zeros(stack.shape)
    tried_initials = mask.copy()
    mn = stack.min()

    for i in tqdm(range(niters)):
        if i > 0:
            mask = outside_boundary(xtrails_acc > mn) + inside_boundary(xtrails_acc > mn)
            mask[counts_acc > 1] = False
            mask[tried_initials] = False
            tried_initials[mask] = True
            if verbose:
                print(np.sum(mask))


        if not np.any(mask):
            break


        xtrails_fw, vtrails_fw,counts_fw,tl_fw = collect_trails_food(field,mask,stack, **kwargs)
        xtrails_bw, vtrails_bw,counts_bw,tl_bw = collect_trails_food(-field,mask,stack, **kwargs)

        vtrails = (vtrails_fw + vtrails_bw)/2
        #counts = mean_pair(counts_fw,counts_bw)
        counts = (counts_fw+counts_bw)/2

        cond = (counts >=min_counts)*(tl_fw + tl_bw > min_traj_len*2)*(vtrails>=min_speed)
        counts_fw = counts_fw + 1e-6
        counts_bw = counts_bw + 1e-6

        xtrails = np.where(cond, (xtrails_fw/counts_fw + xtrails_bw/counts_bw)/2, 0)
        vtrails[~cond] = 0

        xtrails_acc += xtrails
        counts_acc += xtrails > 0

        #xtrails = np.where(counts >= min_counts, mean_pair(xtrails_fw, xtrails_bw)/(1e-6+counts), 0)
        #vtrails = np.where(counts >= min_counts, mean_pair(vtrails_fw, vtrails_bw), 0)

    return xtrails_acc*(counts_acc>0)/(1e-6 + counts_acc), vtrails, counts


@jit
def local_vf_coherence3d(Vf, coh,nhood=1):
    nz,nr,nc,n_ = np.shape(Vf)

    #coh = np.zeros((nz,nr,nc))

    for z in range(nhood,nz-nhood):
        for r in range(nhood,nr-nhood):
            for c in range(nhood,nc-nhood):
                v = Vf[z,r,c]
                for zi in range(z-nhood,z+nhood+1):
                    for ri in range(r-1,r+nhood+1):
                        for ci in range(c-1,c+nhood+1):
                            vi = Vf[zi,ri,ci]
                            coh[z,r,c] += np.abs(np.dot(v,vi))
    return coh

@jit
def vf_diff3d(Vf, contrast, data, nhood=1):
    nz,nr,nc,n_ = np.shape(Vf)

    #coh = np.zeros((nz,nr,nc))
    upd = np.zeros((nz,nr,nc))
    upd_cnt = np.zeros((nz,nr,nc))

    cont_n = 0.0
    cont = 0.0
    dist = 0.0


    for z in range(nhood,nz-nhood):
        for r in range(nhood,nr-nhood):
            for c in range(nhood,nc-nhood):
                v = Vf[z,r,c]
                cont = contrast[z,r,c]
                for zi in range(z-nhood,z+nhood+1):
                    for ri in range(r-1,r+nhood+1):
                        for ci in range(c-1,c+nhood+1):
                            distsq = ((z-zi)**2 + (r-ri)**2 + (c-ci)**2)
                            vi = Vf[zi,ri,ci]
                            cont_n = contrast[zi,ri,ci]
                            edge_w = np.abs(np.dot(v,vi))
                            edge_w = edge_w*cont_n*exp(-(cont-cont_n)**2/100000)*exp(-distsq/3)
                            upd[z,r,c] += edge_w*data[z,r,c]
                            upd_cnt[z,r,c] += edge_w


    return (upd_cnt>1e-5)*upd/(1e-5 + upd_cnt)
