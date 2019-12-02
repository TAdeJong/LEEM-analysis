# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 4 - Accuracy testing
# To test the accuracy of the algorithm, we need some data of which the 'true shift' is known. For this purpose we created a dataset of images with a known relative shift, before adding a variable amount of noise.

# +
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import xarray as xr
import os

from scipy.ndimage.interpolation import shift
from dask.distributed import Client, LocalCluster
from Registration import *

folder = './data'

cluster = LocalCluster()
client = Client(cluster)
client.upload_file('Registration.py')
client
# -

# Here, the image size is the size of the fft (2*fftsize) and actually twice as large as the size used in benchmarking.
fftsize = 512 // 2
x,y = np.mgrid[-fftsize-4:fftsize+4,-fftsize-4:fftsize+4]
r_min = 50
r_max = 100
annulus = np.where((r_min**2 < x**2 + y**2) & (x**2 + y**2 < r_max**2), 1., 0.).astype(float)
plt.imshow(annulus)
plt.colorbar();

# The $x$ and $y$ coordinate are independent in the algorithm. We take advantage of this by testing two types of displacements at once: a parabolic displacement in the $x$-direction and a random one for $y$.

length = 100
dE = 10
x = np.arange(length)
xshifts = x**2/(4*length)
xshifts -= xshifts.mean()
np.random.seed(42)
yshifts = np.random.random(length)
yshifts -= yshifts.mean()
trueshifts = xr.DataArray(np.stack([xshifts, yshifts]), 
                          dims=('direction', 'n'), 
                          coords={'n':np.arange(length), 
                                  'direction': ['x','y']},
                          name='shift')
trueshifts.plot(hue='direction')


# Next, we apply the shifts to (copies of the annulus image) to create the dataset. Once again using `dask` and `gufunc`s allows for effortless streaming and parallelization.

# +
@da.as_gufunc(signature="(i,j),(2)->(i,j)", output_dtypes=float, vectorize=True)
def shift_images(image, shifts):
    """Shift copies of image over shiftsX and shiftsY."""
    return shift(image, shift=shifts, order=1)

da_annulus = da.from_array(annulus, chunks='auto' )
shifts = da.from_array(np.stack([xshifts, yshifts], axis=1), chunks=(dE,2))

synthetic = shift_images(da_annulus, shifts) 


# -

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


# Create the random noise to be added in different magnitudes to the data.
da.random.seed(42)
noise = da.random.normal(size=synthetic.shape, chunks=synthetic.chunks) 

# + {"jupyter": {"outputs_hidden": true}}
# Create an xarray.DataArray to store the results
As = np.arange(0., 4., 0.05)  # Noise amplitudes to calculate for
sigmas = np.arange(0, 20, 0.25)  # Smoothing values to calculate for
# iteration over the datapoints / images, as we will save the found shift for each image.
# We specify np.int32 as opendap does not support int64
ns = np.arange(length, dtype=np.int32) 
direction = ['x', 'y']
res = xr.DataArray(np.zeros((len(As), len(sigmas), 2, len(ns))), 
             coords={'A' : As, 's': sigmas, 'direction': direction, 'n': ns}, 
             dims=['A','s', 'direction', 'n'])
res
# -

for A in As:
    noisedata = synthetic + A * noise
    #noisedata = noisedata.persist()
    for s in sigmas:
        if(res.loc[dict(s=s,A=A,direction='x')].mean() == 0):
            sobel = only_filter(noisedata, 
                                    sigma=s, mode='nearest')
            sobel = sobel - sobel.mean(axis=(1,2), keepdims=True)  
            Corr = dask_cross_corr(sobel)
            weights, argmax = max_and_argmax(Corr)
            W, DX_DY = calculate_halfmatrices(weights, argmax, fftsize=fftsize)
            coords = np.arange(W.shape[0])
            coords, weightmatrix, DX, DY, row_mask = threshold_and_mask(0.0, W, DX_DY, coords=coords)
            dx, dy = calc_shift_vectors(DX, DY, weightmatrix, wpower=4)
            ddx, ddy = interp_shifts(coords, [dx,dy], length)
            res.loc[dict(s=s, A=A, direction='x')] = ddx
            res.loc[dict(s=s, A=A, direction='y')] = ddy
            print(f"A={A}, s={s} completed")
        else: 
            print(f"Skipping A={A}, s={s}")
    res.to_netcdf(os.path.join(folder, 'accuracy_results.nc'))

# ## Plotting
# Let's visualize the results! This produces Figure 8 of the paper.

# +
# One can remove _reference to view newly generated results
data = xr.open_dataset(os.path.join(folder, 'accuracy_result_reference.nc')) 
pltdat = np.abs(data['shift'] + trueshifts)
pltdat = pltdat.rename({'s':'$\sigma$'})
pltdat['$\sigma$'].attrs['units'] = 'pixels'
contourkwargs = {'levels': 15, 'colors': 'black', 'alpha': 0.5, 'x':'A'}

maxplotkwargs = {'vmax': 1, 'cbar_kwargs': {'label':'max error (pixels)'}, 'x':'A'}
meanplotkwargs = {'vmax': 1, 'cbar_kwargs': {'label':'mean error (pixels)'}, 'x':'A'}
fig, axs = plt.subplots(ncols=3, nrows=2, 
                        figsize=[8, 4.5], 
                        sharex='col',
                        constrained_layout=True
                       )
axs= axs.T.flatten()

absmax = pltdat.loc['x'].max('n')
# Use imshow to prevent white lines in some PDF-viewers
im = absmax.plot.imshow(ax=axs[0],**maxplotkwargs) 
CS = np.clip(absmax, 0, 1).plot.contour(ax=axs[0],
                                   **contourkwargs)
im.colorbar.add_lines(CS)
axs[0].plot(absmax['A'],
            absmax['$\sigma$'][absmax.argmin(dim='$\sigma$')],  
            '.', color='white')

absmax = pltdat.loc['y'].max('n')
im = absmax.plot.imshow(ax=axs[1], **maxplotkwargs)
CS = np.clip(absmax, 0, 1).plot.contour(ax=axs[1],
                                   **contourkwargs)
im.colorbar.add_lines(CS)
axs[1].plot(absmax['A'], 
            absmax['$\sigma$'][absmax.argmin(dim='$\sigma$')], 
            '.', color='white')

absmean = pltdat.loc['x'].mean('n')
im = absmean.plot.imshow(ax=axs[2], **meanplotkwargs)
CS = np.clip(absmean, 0, 0.29).plot.contour(ax=axs[2],
                                       **contourkwargs)
im.colorbar.add_lines(CS)
axs[2].plot(absmean['A'], 
            absmean['$\sigma$'][absmean.argmin(dim='$\sigma$')],
            '.', color='white')

absmean = pltdat.loc['y'].mean('n')
im = absmean.plot.imshow(ax=axs[3], **meanplotkwargs)
CS = np.clip(absmean, 0, 0.29).plot.contour(ax=axs[3],
                                       **contourkwargs)
im.colorbar.add_lines(CS)
axs[3].plot(absmean['A'], 
            absmean['$\sigma$'][absmean.argmin(dim='$\sigma$')], 
            '.', color='white')

for i,direction in enumerate(['x', 'y']):
    ax = axs[4 + i]
    ax.axhline(1. / length, c='black', alpha=0.5)
    ax.plot(pltdat['A'], pltdat.loc[direction].max('n').min(dim='$\sigma$'), c='black', alpha=0.5)
    optimal = pltdat.loc[direction][{'$\sigma$': pltdat.loc[direction].max('n').argmin(dim='$\sigma$')}]
    ax.fill_between(optimal['A'], *optimal.quantile([0.159, 0.841], dim='n'), color=f'C{i}', alpha=0.7)
    ax.fill_between(optimal['A'], *optimal.quantile([0.023, 0.977], dim='n'), color=f'C{i}', alpha=0.4)
    optimal.mean(dim='n').plot(ax=ax, color='white') # marker='.')
    optimal.max(dim='n').plot(ax=ax, c='gray')
    pltdat.loc[direction].max('n').min(dim='$\sigma$')

for ax in axs:
    ax.set_xlim([0, 1.95])
for ax in axs[:4]:
    ax.set_ylim([0, 10.8])
for ax in axs[4:]:
    ax.set_ylim([0, max(axs[4].get_ylim()[1], axs[5].get_ylim()[1])])
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='y', labelright=True, labelleft=False)
    ax.set_ylabel("Error (pixels)")
for ax in axs[1::2]:
    ax.set_title(None)
for ax in axs[::2]:
    ax.set_xlabel(None)
axs[0].set_title("Max error")
axs[2].set_title("Mean error")
axs[4].set_title("Optimal error spread")

plt.savefig('simulation_error.pdf')
# -
noisedata = synthetic + 1. * noise
sobel = only_filter(noisedata, sigma=5, mode='nearest')
sobel = sobel - sobel.mean(axis=(1,2), keepdims=True)
fig, axs = plt.subplots(ncols=3, figsize=[8,2.3])
im = axs[2].imshow(sobel[0,...], origin='lower', cmap='gray')
fig.colorbar(im, ax=axs[2])
im = axs[1].imshow(noisedata[0,...], origin='lower', cmap='gray')
fig.colorbar(im, ax=axs[1])
for ax in axs[1:]:
    ax.set_ylabel('y')
    ax.set_xlabel('x')
axs[0].plot(xshifts.squeeze(), label='x')
axs[0].plot(yshifts.squeeze(), label='y')
axs[0].set_ylabel('shift')
axs[0].set_xlabel('index')
axs[0].legend()
plt.tight_layout()
plt.savefig('Figure7.pdf')


