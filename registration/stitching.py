import numpy as np
import dask.array as da
from dask.delayed import delayed
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
import matplotlib as mpl
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import dask
import numba
import scipy.ndimage as ndi
from sklearn.neighbors import NearestNeighbors
import zarr

from scipy.optimize import bisect, minimize
import scipy.sparse as ssp

from .registration import *

def qhist(data, quality, binbins=20, cmap='viridis', ax=None, bins=20):
    """ A helper function to plot a histogram of data with the spread of quality
    over the bins indicated by color."""
    ds = 1/binbins
    binbin = np.arange(0,1,ds)
    cmapi = mpl.cm.get_cmap(cmap)
    rgbas = cmapi(binbin+ds)
    x = [data[np.logical_and(quality < q + ds, quality > q)] for q in binbin]
    if ax is None:
        plt.hist(x, stacked=True, color=rgbas, bins=bins);
    else:
        ax.hist(x, stacked=True, color=rgbas, bins=bins);
    

def connected_bisect(cutoff, weights, nbs):
    row_ind, col_ind = (nbs[:, [0]] * np.ones((1, nbs.shape[1]-1))).flatten(), nbs[:,1:].flatten()
    graph_mask = weights.flatten() > cutoff
    #print("connected_bisect", row_ind.shape, col_ind.shape, weights.shape)
    graph = ssp.csr_matrix((weights.flatten()[graph_mask], (row_ind[graph_mask], col_ind[graph_mask])), 
                           shape=[nbs.shape[0]]*2 )
    cc = ssp.csgraph.connected_components(graph, 
                                            directed=False, 
                                            return_labels=False)
    return cc-1.5


def find_maximum_spanning_tree(weights, nbs):
    graph = w_and_n_2_graph(-1 * weights, nbs)
    mst = ssp.csgraph.minimum_spanning_tree(graph)
    mst.data = -1 * mst.data
    return mst


def weights_and_neighbours(graph):
    wres = np.full((graph.shape[0], graph.getnnz(axis=1).max()), 0.)
    nbres = np.full((graph.shape[0], graph.getnnz(axis=1).max()), 0.)
    for i in range(graph.shape[0]):
        row = graph.getrow(i)
        wres[i, :len(row.data)] = row.data
        nbres[i, :len(row.indices)] = row.indices
    return wres, nbres.astype(int)


def w_and_n_2_graph(weights, nbs):
    row_ind, col_ind = (nbs[:, [0]] * np.ones((1, nbs.shape[1]-1))).flatten(), nbs[:,1:].flatten()
    graph = ssp.csr_matrix((weights.flatten(), (row_ind, col_ind)), 
                           shape=[nbs.shape[0]]*2 )
    return graph


def transform_to_mst(dist, mst, nbr):
    graph = w_and_n_2_graph(dist, nbr)
    graph.data = np.where(graph.data != 0, graph.data, np.nan)
    graph = (mst > 0.).multiply(graph)
    transformed, _ = weights_and_neighbours(graph)
    transformed.data = np.nan_to_num(transformed.data)
    return transformed

#TODO: rename
@numba.njit(nogil=True)
def error_func(x, indices, weights, target):
    """Given a range of positions x, 
    calculate the weighted distances to nneighbors indicated
    by indices and compute the total squared error 
    compared to the target differences.
    """
    error = 0.
    for i in range(indices.shape[1]):
        error += (weights[:,i] * (x - x[indices[:,i]] - target[:,i])**2).mean()
    return error


def base_transformation_error(A, r, rprime, weights=1):
    """Given two sets of 2D vectors r and rprime,
    and a basis transformation A (can be a linear vector to be converted to matrix),
    Calculate the mean of the norm of the difference between the transformed vectors Ar
    and rprime.
    """
    A = A.reshape((2,2))
    return np.linalg.norm((A@r - rprime)*weights, axis=0).mean()



def find_overlap_region(ref_image, image, estimate, mask=False, fftsize=256):
    d_ar = np.array(image.shape)
    # Put image inside boundaries of image
    e_clip = np.clip(estimate.astype(np.int), fftsize-d_ar, d_ar-fftsize)
    ref_region = sliced_region(ref_image, -e_clip, mask=mask)# * corr_mask)
    im_region = sliced_region(image, e_clip, mask=mask)#* corr_mask)
    return np.array([ref_region, im_region])


def sliced_region(image, estimate, mask=False, fftsize=256):
    d_ar = np.array(image.shape)
    center = (estimate.astype(np.int) + d_ar)//2
    # Put image inside boundaries of image, Superfluous if check is in find_correction_and_w
    # wrong thing to do in this region
    center = np.clip(center, fftsize//2, d_ar-fftsize//2)
    im = image[(center[0]-fftsize//2):(center[0]+fftsize//2),
                 (center[1]-fftsize//2):(center[1]+fftsize//2)]
    if mask is False:
        return im
    msk = mask[(center[0]-fftsize//2):(center[0]+fftsize//2),
             (center[1]-fftsize//2):(center[1]+fftsize//2)]
    return im, msk


#@da.as_gufunc(signature="(i,j),(i,j),(t),()->(t),()", output_dtypes=[np.float, np.float], vectorize=True)
#def find_overlap_regions_v(ref_image, image, estimate, fftsize=256):
def find_overlap_regions(images, estimates, mask=False):
    ref_image = images.squeeze()[0]
    res = [find_overlap_region(ref_image, image, estimate, mask)\
           for image, estimate in zip(images.squeeze()[1:], estimates.squeeze())]
    return np.array(res)[np.newaxis,...]


def fft_region(image, estimate, mask=False, fftsize=256):
    d_ar = np.array(image.shape)
    center = (estimate.astype(int) + d_ar)//2
    # Put image inside boundaries of image, Superfluous if check is in find_correction_and_w
    center = np.clip(center, fftsize//2, d_ar-fftsize//2) #Commented out for numba use
    im = image[(center[0]-fftsize//2):(center[0]+fftsize//2),
                 (center[1]-fftsize//2):(center[1]+fftsize//2)]
    if mask:
        im = im - np.average(im, weights=corr_mask)
    else:
        im = im - im.mean()
    return im


@numba.njit()
def n_fft_region(image, estimate, mask=False, fftsize=256):
    d_ar = np.array(image.shape)
    center = (np.floor(estimate) + d_ar)//2
    # Put image inside boundaries of image, Superfluous if check is in find_correction_and_w
    #center = np.clip(center, fftsize//2, d_ar-fftsize//2) #Commented out for numba use
    im = image[(center[0]-fftsize//2):(center[0]+fftsize//2),
                 (center[1]-fftsize//2):(center[1]+fftsize//2)]
    im = im - im.mean()
    return im


@da.as_gufunc(signature="(i,j),(i,j),(t),()->(t),()", output_dtypes=[np.float, np.float], vectorize=True)
def find_correction_and_w(ref_image, image, estimate, fftsize=256):
    d_ar = np.array(image.shape)
    # Put image inside boundaries of image
    e_clip = np.clip(estimate.astype(np.int), fftsize-d_ar, d_ar-fftsize)
    #if np.any(np.abs(estimate)-dims[1:]+fftsize > 0) or np.linalg.norm(estimate) > 2*855-np.sqrt(2)*fftsize:
    if False:
        return np.full((2,), np.nan), np.nan
    else:
        ref_fft = np.fft.rfft2(fft_region(ref_image, -e_clip, mask=False, fftsize=fftsize) )# * corr_mask)
        im_fft = np.fft.rfft2(fft_region(image, e_clip, mask=False, fftsize=fftsize) )#* corr_mask)
        FC = ref_fft*im_fft.conj()
        #FC = FC / np.absolute(FC)
        correl = np.fft.irfft2(FC).real
        w = np.sqrt(np.fft.irfft2(ref_fft*ref_fft.conj()).real[0,0]
                   * np.fft.irfft2(im_fft*im_fft.conj()).real[0,0])
        correl = np.fft.fftshift(correl)
        w = correl.max() / w
        correction = np.array(np.unravel_index(correl.argmax(), correl.shape)) - fftsize//2 + (estimate.astype(np.int)-e_clip)
        return correction, w


def plot_match(i,j):
    print("n", nn[i,j])
    fig = plt.figure(figsize=(16,9))
    ax0 = plt.subplot2grid((2, 3), (0, 0))
    ax1 = plt.subplot2grid((2, 3), (1, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2)
    ax0.imshow(fft_region(data[nn[i,0]], -diffvecs[i,j-1,:]).T)
    ax1.imshow(((fft_region(data[nn[i,j]], diffvecs[i,j-1,:], mask=False))).T, cmap='inferno')
    d = find_correction_and_w_pc(data[nn[i,0]], data[nn[i,j]], diffvecs[i,j-1,:])
    print("d,w:", d, diffvecs[i,j-1,:])
    ax2.imshow(data[nn[i,0]].T)
    ax2.imshow(shift(data[nn[i,j]],  d[:-1]-diffvecs[i,j-1,:]).T, alpha=0.5, cmap='inferno')


def to_nn_diffvecs(coords, nn=None, n_neighbors=5):
    """Generate difference vectors for each coord
    with its n_neighbors nearest neighbors excluding itself
    parameters:
    coords: N*M array, N coordinates of M dimensions
    returns: 
    diff_vecs, N*(n_neighbors-1)*M array of difference
    vectors
    neighbor_indices, N* n_neighbors array,
    as return my kneighbors"""
    if nn is None:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(coords.T)
        nn = nbrs.kneighbors(coords.T, return_distance=False)
    diffvecs = []
    # pc to diffvecs
    diffvecs = [coords.T[nn[:, 1+i]] - coords.T for i in range(nn.shape[1]-1)]
    diffvecs = np.stack(diffvecs, axis=1)
    return diffvecs, nn


def trim_nans(image):
    """Trim all rows and columns containing only nans from the image
    """
    xmask = np.all(np.isnan(image), axis=1)
    ymask = np.all(np.isnan(image), axis=0)
    if len(image.shape) >= 3:
        #Color channel handling
        xmask = np.any(xmask, axis=-1)
        ymask = np.any(ymask, axis=-1)
    return image[~xmask][:,~ymask]