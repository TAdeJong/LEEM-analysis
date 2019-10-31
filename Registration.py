import numpy as np
import dask.array as da
import dask
from dask.delayed import delayed
from dask.distributed import Client
from scipy.optimize import least_squares
from scipy.ndimage.interpolation import zoom, shift
from scipy.interpolate import interp1d
import scipy.sparse as sp
from skimage import filters
import numba

def filter_block(block, sigma, mode='nearest'):
    """Perform Gaussian and Sobel filtering on a block of images"""
    #return np.stack([GSfilter(block[i], sigma, mode) for i in range(block.shape[0])])
    return np.stack([GSfilter(image, sigma, mode) for image in block])

def GSfilter(image, sigma, mode):
    """Combine a Sobel and a Gaussian filter"""
    return filters.sobel(filters.gaussian(image, sigma=sigma, mode=mode))

def crop_and_filter(images, sigma=11, mode='nearest', finalsize=256):
    """Crop images to finalsize and apply the filters.
    Cropping is initially with a margin of sigma,
    to prevent edge effects of the filters."""
    cent = [dim//2 for dim in images.shape[1:]]
    halfsize = finalsize//2 + sigma
    result = images[:,(cent[0]-halfsize):(cent[0]+halfsize),
                    (cent[1]-halfsize):(cent[1]+halfsize)]
    result = result.map_blocks(filter_block, dtype=np.float64, 
                               sigma=sigma, mode=mode)
    if sigma > 0:
        result = result[:,sigma:-sigma,sigma:-sigma]
    return result

def dask_cross_corr(data):
    """Return the dask array with the crosscorrelations of data
    (uncomputed)
    """
    # Define the Correlation `Corr` via FFT's:
    F = da.fft.rfft2(data, axes=(1,2))
    Corr = da.fft.irfft2(F[:, np.newaxis, ...] * F.conj()[np.newaxis, ...], 
                         axes=(2,3)
                        ).real
    Corr = da.fft.fftshift(Corr, axes=(2,3))
    return Corr

def max_and_argmax(data):
    """Return the dask max and argmax of data along the last two axes,
    which corresponds to the x and y dimensions
    (uncomputed)
    """
    # Slap out a dimension to nicely apply argmax and max
    flatData = data.reshape(data.shape[:-2]+(-1,))
    argmax = da.argmax(flatData, axis=-1)
    # We can forego calculating both max and argmax as soon as 
    # we have da.take_along_axis() https://github.com/dask/dask/issues/3663 
    # Would a map_blocks of np.take_along_axis() work and be faster?
    weights = da.max(flatData, axis=-1)
    return weights, argmax

def calculate_halfmatrices(weights, argmax, fftsize=128):
    """Calculate the half matrices of the weights and the argmax
    and reconstruct the full arrays as numpy arrays now."""
    # Calculate half of the matrices only, because we know it is antisymmetric.
    uargmax = da.triu(argmax, 1) # Diagonal shifts are zero anyway, so 1. Reconstruct after computation

    uW = da.triu(weights, 1)
    uW = uW + uW.T + da.diag(da.diag(weights))

    # Do actual computations, get a cup of coffee
    Mc, Wc = da.compute(uargmax, uW)

    # Undo the flatten: Reconstruct 2D indices from global linear indices of argmax
    Mc = np.stack(np.unravel_index(Mc, (fftsize*2, fftsize*2))) 
    Mc -= np.triu(np.full_like(Mc, fftsize), 1)  # Compensate for the fft-shift
    Mc = Mc - Mc.swapaxes(1,2)  # Reconstruct full antisymmetric matrices
    # Mc = Mc / z_factor  # Compensate for zoomfactor
    return Wc, Mc


def threshold_and_mask(min_normed_weight, W, Mc, coords): #=np.arange(Wc.shape[0])*stride + start):
    """Normalize the weights W, threshold to min_normed_weight and remove diagonal,
    reduce DX and DY to the columns and rows still containing weights.
    Returns:
    the indices of these columns in terms of original image indices
    the thresholded weights
    The reduced DX and DY.
    The indices of these columns in terms of calculated arrays.
    """
    #coords = np.arange(Wc.shape[0])*stride + start
    wcdiag = np.atleast_2d(np.diag(W))
    W_n = W / np.sqrt(wcdiag.T*wcdiag)
    mask = W_n - np.diag(np.diag(W_n)) > min_normed_weight
    row_mask = np.any(mask, axis=0)
    W_n = np.where(mask, W_n, 0)
    DX, DY = Mc[0], Mc[1]
    W_n_m = W_n[:, row_mask][row_mask, :]
    coords = coords[row_mask]
    #mask_red = mask[row_mask, :][:, row_mask]
    DX_m, DY_m = DX[row_mask, :][:, row_mask], DY[row_mask, :][:, row_mask]
    return coords, W_n_m, DX_m, DY_m, row_mask


def construct_jac(W):
    """Construct a sparse Jacobian matrix of the least squares problem 
    from a weight matrix W. This Jacobian is independent of the position
    vector dx as the problem is actually linear"""
    n = W.shape[0]
    i = np.arange(n)
    j = i + n*i
    data = np.ones(n)
    A = sp.coo_matrix((data, (i, j)), shape=(n, n*n)).tocsr()
    wsp = sp.csr_matrix(W)
    J = wsp.dot(A)
    J = J.reshape((n*n, n))
    return J - wsp.T.dot(A).reshape((n*n, n), order='F')

def calc_shift_vectors(DX, DY, weightmatrix, wpower=4, lsqkwargs={}):
    wopt = weightmatrix**wpower
    @numba.jit(nopython=True, nogil=True)
    def err_func(x, result):
        x = np.atleast_2d(x)
        epsilon = x - x.T - result
        epsilon = epsilon * wopt
        return epsilon.ravel()
    
    Jac = construct_jac(wopt)
    res = least_squares(err_func, DX[DX.shape[0]//2,:], 
                        jac=lambda x, Jargs: Jac,
                        args=(DX,), **lsqkwargs)
    dx = res['x']
    res = least_squares(err_func, DY[DY.shape[0]//2,:], 
                        jac=lambda x, Jargs: Jac,
                        args=(DY,), **lsqkwargs)
    dy = res['x']
    dx -= dx.min()
    dy -= dy.min()
    return dx, dy

def interp_shifts(coords, shifts, n=None):
    """
    Interpolate shifts for all frames not in coords and create dask array

    :param coords: coordinates for shifts
    :param shifts: list of arrays to be interpolated
    :param n: final length (original size)
    :return: list of interpolated shifts
    """
    if n is None:
        ns = np.arange(coords.min(),coords.max()+1)
    else:
        ns = np.arange(n)
    shifts_interp = []
    for dr in shifts:
        f = interp1d(coords, dr, fill_value=(dr[0], dr[-1]),
                     kind='linear', assume_sorted=True, bounds_error=False)
        shifts_interp.append(f(ns))
    #shift = da.from_array(shift, chunks=(dE,1,1))
    return shifts_interp

# def shift_block(images, shifts, margins=(0,0)):
#     """Shift a block of images per image in the x,y plane by shifts[index].
#     Embed this in margins extra space"""
#     result = np.zeros((images.shape[0], 
#                        images.shape[1] + margins[0], 
#                        images.shape[2] + margins[1]))
#     for index in range(images.shape[0]): 
#         result[index, 
#                0:images.shape[1], 
#                0:images.shape[2]] = images[index]
#         result[index,...] = shift(result[index,...],
#                                   shift=shifts[index], 
#                                   order=1,
#                                   )
#     return result

# def syn_shift_blocks(shiftsX, shiftsY, image):
#     result = np.stack([image] * dE)
#     for index in range(dE): 
#         result[index,...] = shift(result[index,...],
#               shift=(shiftsX[index,...],shiftsY[index,...]), 
#               order=1,
#               )
#     return result


def only_filter(images, sigma=11, mode='nearest'):
    """Apply the filters.
    Cropped with a margin of sigma,
    to prevent edge effects of the filters."""
    result = images.map_blocks(filter_block, dtype=np.float64, 
                               sigma=sigma, mode=mode)
    if sigma > 0:
        si = int(np.ceil(sigma))
        result = result[:,si:-si,si:-si]
    return result


