"""Functions used for registering stacks of images, e.g. spectroscopic data"""
import numpy as np
import dask.array as da
import dask
from dask.delayed import delayed
from scipy.optimize import least_squares
import scipy.ndimage as ndi
from scipy.interpolate import interp1d
import scipy.sparse as ssp
from skimage import filters
import numba


def filter_block(block, sigma, mode='nearest'):
    """Perform Gaussian and Sobel filtering on a block of images
    TODO: replace with gaussian_gradient_magnitude as implemented in cupy, ndi and dask_image
    """
    # return np.stack([GSfilter(block[i], sigma, mode) for i in range(block.shape[0])])
    return np.stack([GSfilter(image, sigma, mode) for image in block])


def GSfilter(image, sigma, mode):
    """Combine a Sobel and a Gaussian filter"""
    return filters.sobel(filters.gaussian(image, sigma=sigma, mode=mode))


def crop_and_filter(images, sigma=11, mode='nearest', finalsize=256):
    """Crop images to finalsize and apply the filters.
    Cropping is initially with a margin of sigma,
    to prevent edge effects of the filters.
    """
    cent = [dim//2 for dim in images.shape[1:]]
    halfsize = finalsize//2 + sigma
    result = images[:, (cent[0]-halfsize):(cent[0]+halfsize),
                    (cent[1]-halfsize):(cent[1]+halfsize)]
    result = result.map_blocks(filter_block, dtype=np.float64,
                               sigma=sigma, mode=mode)
    if sigma > 0:
        result = result[:, sigma:-sigma, sigma:-sigma]
    return result


def dask_cross_corr(data):
    """Return the dask array with the crosscorrelations of data
    (uncomputed)
    """
    # Define the Correlation `Corr` via FFT's:
    F = da.fft.rfft2(data, axes=(1, 2))
    Corr = da.fft.irfft2(F[:, np.newaxis, ...] * F.conj()[np.newaxis, ...],
                         axes=(2, 3)
                         ).real
    Corr = da.fft.fftshift(Corr, axes=(2, 3))
    return Corr


def max_and_argmax(data):
    """Returns max and argmax along last two axes.

    Last two axes should correspond to the x and y dimensions.

    Parameters
    ----------
    data : dask array
        data with at least 3 dimensions

    Returns
    -------
    weights : dask array
        max of `data` along the last two axes
    argmax : dask array
        argmax of `data` along the last two axes
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
    """Compute the half matrices of the weights and the argmax
    and reconstruct the full arrays.

    Parameters
    ----------
    weights : dask array
    argmax : dask array
    fftsize : int, default: 128

    Returns
    -------
    Wc : numpy array
        Computed weights, symmetric
    Mc : numpy array
        Computed locations of maxima
    """
    # Calculate half of the matrices only, because we know it is antisymmetric.
    uargmax = da.triu(argmax, 1)  # Diagonal shifts are zero anyway, so 1. Reconstruct after computation

    uW = da.triu(weights, 1)
    uW = uW + uW.T + da.diag(da.diag(weights))

    # Do actual computations, get a cup of coffee
    Mc, Wc = da.compute(uargmax, uW)

    # Undo the flatten: Reconstruct 2D indices from global linear indices of argmax
    Mc = np.stack(np.unravel_index(Mc, (fftsize*2, fftsize*2)))
    Mc -= np.triu(np.full_like(Mc, fftsize), 1)  # Compensate for the fft-shift
    Mc = Mc - Mc.swapaxes(1, 2)  # Reconstruct full antisymmetric matrices
    # Mc = Mc / z_factor  # Compensate for zoomfactor
    return Wc, Mc


def threshold_and_mask(min_normed_weight, W, Mc, coords):  # =np.arange(Wc.shape[0])*stride + start):
    """Normalize the weights W, threshold to min_normed_weight and remove diagonal,
    reduce DX and DY to the columns and rows still containing weights.

    Returns
    -------
    coords : array_like
        the indices of these columns in terms of original image indices
    W_n_m : array_like
        the thresholded weights
    D_X_m : array_like
        The reduced DX
    D_Y_m : array_like
        The reduced DY
    row_mask : array_like
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
    from a weight matrix.

    This Jacobian is independent of the position
    vector dx as the problem is actually linear

    Parameters
    ----------
    W : array_like (N,N)
        weight matrix

    Returns
    -------
    sparse array (N, N*N)
        Jacobian in compressed sparse row format
    """
    N = W.shape[0]
    i = np.arange(N)
    j = i + N*i
    data = np.ones(N)
    A = ssp.coo_matrix((data, (i, j)), shape=(N, N*N)).tocsr()
    wsp = ssp.csr_matrix(W)
    J = wsp.dot(A)
    J = J.reshape((N*N, N))
    return J - wsp.T.dot(A).reshape((N*N, N), order='F')


def calc_shift_vectors(DX, DY, weightmatrix, wpower=4, lsqkwargs={}):
    """From relative displacement matrices, compute absolute displacement
    vectors.

    Parameters
    ----------
    DX : array_like (N,N)
        horizontal relative displacement matrix
    DY : array_like (N,N)
        vertical relative displacement matrix
    weightmatrix : array_like (N,N)
        weights for least sqaures minimization
    wpower : int or float, default=4
        weightmatrix is used to this power
        (componentwise)
    lsqkwargs : dict, default={}
        keyword arguments to pass to calls of
        `scipy.optimize.least_squares`

    Returns
    -------
    dx : array (N,)
        horizontal absolute shift vector
    dy : array (N,)
        vertical absolute shift vector
    """
    wopt = weightmatrix**wpower

    @numba.jit(nopython=True, nogil=True)
    def err_func(x, result):
        x = np.atleast_2d(x)
        epsilon = x - x.T - result
        epsilon = epsilon * wopt
        return epsilon.ravel()

    Jac = construct_jac(wopt)
    res = least_squares(err_func, DX[DX.shape[0]//2, :],
                        jac=lambda x, Jargs: Jac,
                        args=(DX,), **lsqkwargs)
    dx = res['x']
    res = least_squares(err_func, DY[DY.shape[0]//2, :],
                        jac=lambda x, Jargs: Jac,
                        args=(DY,), **lsqkwargs)
    dy = res['x']
    dx -= dx.min()
    dy -= dy.min()
    return dx, dy


def interp_shifts(coords, shifts, n=None):
    """
    Interpolate shifts for all frames not in coords and create dask array

    Parameters
    ----------
    coords : (N,)
        coordinates for shifts
    shifts : (d, N)
        list of arrays to be interpolated
    n : int or None, defaults=None
        final length (original size)

    Returns
    -------
    shifts_interp : (d, n)
        array of interpolated shifts
    """
    if n is None:
        ns = np.arange(coords.min(), coords.max()+1)
    else:
        ns = np.arange(n)
    shifts_interp = []
    for dr in shifts:
        f = interp1d(coords, dr, fill_value=(dr[0], dr[-1]),
                     kind='linear', assume_sorted=True, bounds_error=False)
        shifts_interp.append(f(ns))
    #shift = da.from_array(shift, chunks=(dE,1,1))
    return np.array(shifts_interp)

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
    to prevent edge effects of the filters.
    """
    result = images.map_blocks(filter_block, dtype=np.float64,
                               sigma=sigma, mode=mode)
    if sigma > 0:
        si = int(np.ceil(sigma))
        result = result[:, si:-si, si:-si]
    return result


def register_stack(data, sigma=5, fftsize=256, dE=10, min_norm=0.15):
    """Top level convenience function to register a stack of images.

    `data` should be a stack of images stacked along axis 0 in the form
    of anything convertible to a dask array by `da.asarray()`.
    Quick and dirty function, should only be used for small stacks, as
    not all parameters are exposed, in particular strides/interpolation
    are unavailable.
    """
    data = da.asarray(data, chunks=(dE, -1, -1))
    sobel = crop_and_filter(data.rechunk({0: dE}), sigma=sigma, finalsize=2*fftsize)
    sobel = (sobel - sobel.mean(axis=(1, 2), keepdims=True)).persist()
    corr = dask_cross_corr(sobel)
    W, M = calculate_halfmatrices(*max_and_argmax(corr), fftsize=fftsize)
    w_diag = np.atleast_2d(np.diag(W))
    W_n = W / np.sqrt(w_diag.T*w_diag)
    nr = np.arange(data.shape[0])
    coords, weightmatrix, DX, DY, row_mask = threshold_and_mask(min_norm, W, M, nr)
    dx, dy = calc_shift_vectors(DX, DY, weightmatrix)
    shifts = np.stack(interp_shifts(coords, [dx, dy], n=data.shape[0]), axis=1)
    neededMargins = np.ceil(shifts.max(axis=0)).astype(int)
    shifts = da.from_array(shifts, chunks=(dE, -1))

    @da.as_gufunc(signature="(i,j),(2)->(i,j)", output_dtypes=data.dtype, vectorize=True)
    def shift_images(image, shifts):
        """Shift `image` by `shift` pixels."""
        return ndi.shift(image, shift=shifts, order=1)
    padded = da.pad(data.rechunk({0: dE}),
                    ((0, 0),
                     (0, neededMargins[0]),
                     (0, neededMargins[1])
                     ),
                    mode='constant'
                    )
    corrected = shift_images(padded.rechunk({1: -1, 2: -1}), shifts)
    return corrected, shifts
