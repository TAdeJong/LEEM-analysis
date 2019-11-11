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

# # 3 - Benchmarking
# This notebook explores the performance of the driftcorrection algorithm as defined in `2 - Driftcorrection` by benchmarking the time it takes to driftcorrect stacks of different numbers of images.

# Needed imports
from Registration import *
import shutil
import xarray as xr
import time
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
import os

folder = r'./data'
name = '20171120_160356_3.5um_591.4_IVhdr'
original = xr.open_dataset(os.path.join(folder, name + '_detectorcorrected.nc'), chunks={'time': 1})
original = original.Intensity.data

# Define fftsize used for the drift correction algorithm actual size of the fft is twice this value.
fftsize = 256 // 2


# Next, we define the grid of parameters for which we will perform the benchmark and the timings we want to save as an empty `xarray.DataArray`.

iters = np.arange(5)
sigmas = [3, 7, 9, 11, 13, 17]
strides = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 35, 50, 70,])
ts = [0,1,2,3,4]
res = xr.DataArray(np.zeros((len(iters), len(sigmas), len(strides), len(ts))), 
             coords={'i':iters, 'sigma': sigmas, 'strides': strides, 't': ts}, 
             dims=['i','sigma', 'strides', 't'])
res.coords

# Before we can start, we connect to the dask-scheduler and upload the used functions

cluster = LocalCluster()
client = Client(cluster)
client.upload_file('Registration.py')
client

tstart = time.time()
t = np.zeros((5,))
for p, stride in enumerate(strides):
    for sigma in sigmas:
        if p == 0:
            break
        for i in iters:
            t[0] = time.time() - tstart
            #start, stride, dE = 40, 1, 10
            start, stop, dE = 40, 740, 10
            #stop = start + n
            Eslice = slice(start, stop, stride)
            sliced_data = original[Eslice,...].rechunk({0:dE})
            sobel = crop_and_filter(sliced_data, 
                                    sigma=sigma, finalsize=2*fftsize)
            sobel = sobel - sobel.mean(axis=(1,2), keepdims=True)  
            Corr = dask_cross_corr(sobel)
            weights, argmax = max_and_argmax(Corr)
            Wc, Mc = calculate_halfmatrices(weights, argmax, fftsize=fftsize)
            t[1] = (time.time() - (t[0]+tstart))
            coords = np.arange(sliced_data)
            coords, weightmatrix, DX, DY, row_mask = threshold_and_mask(0.0, Wc, Mc, coords=coords)
            t[2] = (time.time() - (t[0]+tstart))
            dx, dy = calc_shift_vectors(DX, DY, weightmatrix)
            t[3] = (time.time() - (t[0]+tstart))
            shifts = np.stack(interp_shifts(coords, [dx, dy], n=sliced_data.shape[0]), axis=1)
            neededMargins = np.ceil(shifts.max(axis=0)).astype(int)
            shifts = da.from_array(shifts[..., np.newaxis], chunks=(dE,-1,1))
            corrected = da.map_blocks(shift_block, sliced_data, shifts,
                          margins=neededMargins,
                          dtype=sliced_data.dtype, 
                          chunks=(dE,
                                  sliced_data.shape[1] + neededMargins[0], 
                                  sliced_data.shape[2] + neededMargins[1]),
                         )
            corrected[:sliced_data.shape[0]].to_zarr(r'./tempresult.zarr', overwrite=True)
            t[4] = (time.time() - (t[0]+tstart))
            res.loc[dict(i=i,sigma=sigma,strides=stride)] = t
            shutil.rmtree(r'tempresult.zarr')
            print(t, corrected.shape, weights.shape)
    res.to_netcdf(os.path.join(folder, 'benchmarkresult.nc'))

# ## Plotting
# We can plot the results of either the benchmark run above or the reference results. This is done using `xarray` plotting interface on the created datasets. First we do some cleaning up of the data and recombination:

data = xr.open_dataarray(os.path.join(folder, 'benchmarkresult_reference.nc')) # One can remove _reference to view newly generated results
data

# +
# We are interested in the times each individual step took, 
# instead of the time upto that step which is saved, so take a diff

data = xr.concat([data.isel(t=1), data.isel(t=slice(2,5)).diff(dim='t', label='lower'), data.isel(t=4)], 't')
data.attrs['long_name'] = 'Run time'
data.attrs['units'] = 's'

# Define a nicer 'Phase' of the algorithm dimension instead of 't', as it is saved in the file.
data.coords['Phase'] = ('t', ['Filter+CC', 'Least Squares', 'Shift and write', 'Total']) 
data = data.swap_dims({'t': 'Phase'})
data.coords['N'] = ('strides', 700//data.coords['strides'])
data = data.swap_dims({'strides': 'N'})
# -

# And now we are ready to actually plot.

# +
#Take the mean over the iterations
red = data.mean(dim='i', keep_attrs=True)

facetgrid = red.plot.line('.', col='Phase', hue='sigma', 
                          yscale='log', xscale='log', ylim=[0.1, 1000], 
                          figsize=[6,2], alpha=0.8)

# Add guidelines in red for quadratic in green for linear.
facetgrid.axes[0,0].plot(data['N'], 0.0012*data['N']**2, c='red', zorder=0, alpha=0.5)
facetgrid.axes[0,1].plot(data['N'], 0.00025*data['N']**2, c='red', zorder=0, alpha=0.5)
facetgrid.axes[0,1].plot(data['N'], 0.0035*data['N'], c='green', zorder=0, alpha=0.5)
facetgrid.axes[0,2].plot(data['N'], 0.02*data['N'], c='green', zorder=0, alpha=0.5)
facetgrid.axes[0,3].plot(data['N'], 0.0012*data['N']**2, c='red', zorder=0, alpha=0.5)
# Replace default plot titles with a somewhat nicer version
facetgrid.axes[0,0].set_title('Phase:\nFilter & CC')
facetgrid.axes[0,1].set_title('Phase:\nLeast Squares')
facetgrid.axes[0,2].set_title('Phase:\nShift & write')
facetgrid.axes[0,3].set_title('Total')
plt.subplots_adjust(top=0.8, bottom=0.18, left=0.08, wspace=0.1)
#plt.savefig('timebench.pdf')
# -


