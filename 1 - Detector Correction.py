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

# # 1 - Detector Correction
#
# In this first notebook, the effects of correcting for the artefacts in images caused by the detector system are explored. The relevant detector system consists of a microchannel plate (MCP) for electron amplification, a fluorescent screen to convert to photons and a CCD camera. Calibration of the gain vs MCP bias voltage, and subtraction of the dark current, followed by [flat fielding](https://en.wikipedia.org/wiki/Flat-field_correction), compensates artefacts and allows for conversion to true reflectivity spectra.

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import xarray as xr
import os
import dask.array as da
from IPython.display import Markdown

SAVEFIG = True
# -

# ## Dark count
# First we open the dark count dataset and extract both a mean dark count _image_ as well as a mean dark count _value.

folder = './data'
DC_dat =  xr.open_dataset(os.path.join(folder, '20171205_115846_0.66um_475.8_DC.nc'))
DC_dat

# +
# We always grab a square part of the center of the image for calibration
radius = 100
DC = DC_dat['Intensity'].mean(dim='time')

DCval = DC[(640-radius):(640+radius), (512-radius):(512+radius)].mean()
# -

# ## Calibration of $G(V_{MCP})$
#
# We calibrate the gain of the MCP, by fitting Image intensity versus gain $G(V_{MCP})$. First we load the reference data:

# +
CPvals = []
Ivals = []
Iorigs = []
EGY = []
Vars = []
dats=[]

# Highest intensity spot is smaller and slightly off-center
c = [750, 480]

for file in ['20171205_143305_31um_474.2_sweepCHANNELPLATE_SET_higherintensity', 
             '20171205_103440_0.66um_479.2_sweepCHANNELPLATE_SET_highintensity', 
             '20171205_100540_0.66um_479.2_sweepCHANNELPLATE_SET_lowintensity_v2', 
             ]:
    dset = xr.open_dataset(os.path.join(folder, file+'.nc'))
    CP = np.array(dset['MCP_bias'])
    Orig = dset['Intensity']
    DCcor = Orig - DC
    dats.append(Orig[:, (c[0]-radius):(c[0]+radius), (c[1]-radius):(c[1]+radius)])
    V = np.var(Orig[:, (640-radius):(c[0]+radius), (c[1]-radius):(c[1]+radius)].data, axis=(1,2))
    Orig = Orig[:, (c[0]-radius):(c[0]+radius), (c[1]-radius):(c[1]+radius)].mean(axis=(1,2))
    Vars.append(V)
    DCcor = DCcor[:, (c[0]-radius):(c[0]+radius), (c[1]-radius):(c[1]+radius)].mean(axis=(1,2))
    Iorigs.append(Orig)
    CPvals.append(CP)
    Ivals.append(DCcor)
    EGY.append(dset['Energy_set'][0])
Ivals = np.stack(Ivals)
CPvals = np.stack(CPvals)
Iorigs = np.stack(Iorigs)
# -

# As described in the paper, we fit a function of the following form:
#
# $G(V_\text{MCP}) = A_i\exp\left(\sum_{k=0}^8 c_k {V_\text{MCP}}^{2k+1}\right)$
#
# As initial guess we will assume 3 equal amplitudes $A_i$, a non-zero linear component $c_0$ and zero for the higher order terms

# +
# Fitting and error function definitions

def odd_polynomial(x, *coeffs):
    res = np.zeros_like(x)
    for index in range(len(coeffs)):
        res += coeffs[index]*np.power(x, 2*index + 1)
    return res

def polynomial(x, *coeffs):
    res = np.zeros_like(x)
    for index in range(len(coeffs)):
        res += coeffs[index]*np.power(x, index + 1)
    return res

def mod_exp(V, *coeffs):
    "Modified exponent function"
    return np.exp(-1*odd_polynomial(V, *coeffs))

def fit_func(CP, *params):
    """The joint fit function."""
    Amps = np.array(params[:CP.shape[0]])[:,np.newaxis]
    return Amps * mod_exp(CP, *params[CP.shape[0]:])

def err_func(params, CP, I):
    """Logarithmic error function"""
    return (np.log(fit_func(CP, *params)) - np.log(I)).ravel()
# -

initial_params = [1,1,1, 1,0,0,0,0,0,0,0,0]  # 3 amplitudes and linear exponent non-zero
fullres = least_squares(err_func, initial_params, args=(CPvals, Ivals), 
                         max_nfev=1000000)
res = fullres['x']
print(fullres['message'])
Markdown('$'+'$  \n$'.join(['c_{} = {:.5f}'.format(k, res[3+k]) for k in range(len(res[3:]))])+'$  \n$\sum_{k=0}'+'^{}c_k = {:.5f}$'.format(len(res[3:])-1, np.sum(res[3:])))


def multiplier_from_CP(V_MCP):
    """Multiplier based on MCP voltage V_MCP (in kV)
    Normalised to 1.0 at V_MCP = 1
    """
    return np.exp(-1*odd_polynomial(V_MCP, *res[3:]) + np.sum(res[3:]))

# +
fix, axs = plt.subplots(ncols=2, nrows=2, figsize=[6,6], sharex=True, constrained_layout=True)
axs = axs.flatten()
axs[0].axhline(DCval, color='black', alpha=0.5, linestyle='--')
axs[0].annotate('DC', xy=(1.7, DCval/1.8), color='black', alpha=0.5)
for i in range(len(EGY)):
    axs[0].semilogy(CPvals[i], Iorigs[i], '.', markersize=4, label=r"$E_0={:.1f}$".format(EGY[i].values))
axs[0].legend()

axs[1].semilogy(CPvals.T, Ivals.T, '.', markersize=4)
axs[1].semilogy(CPvals.T, fit_func(CPvals, *res).T, 
                  color='black', linewidth=0.7, label='fit')
axs[2].semilogy(CPvals.T, Ivals.T/res[:3], '.', markersize=4, alpha=0.8)
axs[2].semilogy(CPvals.T, fit_func(CPvals, *res).T/res[:3], 
              color='black', linewidth=0.7, label='fit')

axs[3].plot(CPvals.T, 
            ((Ivals - fit_func(CPvals, *res)).T/res[:3]) / (fit_func(CPvals, *res).T/res[:3]), 
            '.', alpha=0.8, markersize=4)
axs[3].set_ylim([-0.5,0.5])
for ax in axs:
    ax.set_xlabel('Channel Plate bias (kV)')
axs[0].set_ylim(axs[1].get_ylim())
axs[0].set_ylabel('Intensity (CCD counts)')
axs[1].set_ylabel('Corrected Intensity (CCD counts)')
axs[2].set_ylabel('Intensity (a.u.)')
axs[3].set_ylabel('Relative fitting residuals')
for ax in [axs[1], axs[3]]:
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='y', labelright=True, labelleft=False)
for ax in axs[:2]:
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis='x', labeltop=True, labelbottom=False)
if SAVEFIG:
    plt.savefig(f'Channelplate_calibration_kmax={len(res[3:])-1}.pdf')
# -

# ## Flatfielding 
# First we load a dataset, then we will use `dask` and `numpy`'s `gufunc` to define a function `correctImages()` that performs the detector correction and apply it. Using `dask` has the benefit of streaming and parallelizing the operation without large additional programming effort.

dataname = '20171120_160356_3.5um_591.4_IVhdr'
DFdataname = '20171120_215555_3.5um_583.1_IVhdr_DF2'
dataset = xr.open_dataset(os.path.join(folder, dataname + '.nc'), chunks={'time': 5})
DFdataset = xr.open_dataset(os.path.join(folder, DFdataname + '.nc'), chunks={'time': 5})
data = dataset['Intensity'].data
dataset


@da.as_gufunc(signature="(i,j), (i,j), (i,j) ->(i,j),()", output_dtypes=(np.uint16, float))
def correctImages(image, darkCount, FF):
    """Using a darkCount and a Flatfield image FF, perform all
    detector corrections, scale the image to use the full bitrange
    of 16 bits and return the needed multiplier and resulting image(s).
    """
    Corr_image = image.astype(np.int32) - darkCount
    # Set negative values to 0
    np.clip(Corr_image, 0, 2**16 - 1, out=Corr_image)
    Corr_image = (Corr_image / FF).astype(np.float64)
    multiplier = (2**16 - 1.) / Corr_image.max(axis=(-2, -1), keepdims=True)
    Corr_image = Corr_image * multiplier
    Corr_image = Corr_image.astype(np.uint16)
    return Corr_image, np.atleast_1d(multiplier.squeeze())

# +
G_MCP = multiplier_from_CP(dataset["MCP_bias"].compute())

# Use the mirror mode, where all electrons are reflected, as flat field
FF = (data[:32].mean(axis=0) - DC)

# Do not scale for pixels outside the channelplate
FF = da.where(FF > 0.1*FF.max(), FF, FF.max())
plt.imshow(FF.compute().T, cmap='gray')
plt.colorbar()
# -

corrected, multiplier = correctImages(data, DC, FF / G_MCP[:32].mean())
multiplier *= G_MCP
# Pick a specific nice image for the figure at index 77
rawimage = data[77]
CPcorimage = corrected[77] / multiplier[77]

fig, axs = plt.subplots(2,2, constrained_layout=True, figsize=[8, 4.9])
im = axs[0,0].imshow(DC.T/16., interpolation='none')
fig.colorbar(im, ax=axs[0,0], label='Intensity (mean CCD counts)', pad=0)
im = axs[0,1].imshow((data[:32].mean(axis=0) - DC).T, interpolation='none')
fig.colorbar(im, ax=axs[0,1], label='Intensity (CCD counts)')
im = axs[1,0].imshow(rawimage.compute().T, cmap='gray', interpolation='none')
fig.colorbar(im, ax=axs[1,0], label='Intensity (CCD counts)')
im2 = axs[1,1].imshow(CPcorimage.compute().T, cmap='gray', interpolation='none')
fig.colorbar(im2, ax=axs[1,1], label='Reflectivity')
axs[1,0].axvline(700, color='red', alpha=0.5)
axs[1,1].axvline(700, color='green', alpha=0.5)
for ax in axs.flatten()[1:]:
    ax.annotate('', xy=(705, 1024-887),  xycoords='data',
            xytext=(0.7, 0.8), textcoords='axes fraction',
            arrowprops=dict(facecolor='red', shrink=0.1),
            horizontalalignment='right', verticalalignment='top',
            )
if SAVEFIG:
    plt.savefig('DC_and_Flatfield.pdf', dpi=300)

fig, ax = plt.subplots(figsize=[4, 4.9])
line1 = plt.plot(rawimage[700, :], label='raw', color='red')
plt.ylabel('Intensity (CCD counts)')
plt.xlabel('y (pixels)')
plt.ylim(im.get_clim())
plt.twinx()
line2 = plt.plot(CPcorimage[700,:], label='DC and FF corrected', color='green')
plt.ylabel('Reflectivity')
plt.ylim(im2.get_clim())
ax.annotate('', xy=(1024-887, 23000),  xycoords='data',
            xytext=(0.28, 0.35), textcoords='axes fraction',
            arrowprops=dict(facecolor='red', shrink=0.1),
            horizontalalignment='right', verticalalignment='top',
            )
lns = line1 + line2
labels = [l.get_label() for l in lns]
plt.margins(x=0, y=0, tight=True)
plt.legend(lns, labels)
plt.tight_layout()
if SAVEFIG:
    plt.savefig('DC_and_Flatfield2.pdf', dpi=300)

# ## Saving data
#
# Finally, we save both the corrected images and the multiplier to a new netCDF dataset via xarray

xrcorrected = dataset.copy()
xrcorrected.Intensity.data = corrected
xrcorrected.Intensity.attrs['DetectorCorrected'] = 'True'
xrcorrected['multiplier'] = (('time'), multiplier)
xrcorrected.to_netcdf(os.path.join(folder, dataname + '_detectorcorrected.nc'))
xrcorrected

# ### Dark Field dataset
# We also correct the Dark Field dataset, using the mirror mode from the Bright Field dataset as Flat Field

DFcorrected, DFmultiplier = correctImages(DFdataset['Intensity'].data, DC, FF / G_MCP[:32].mean())
DFmultiplier *= multiplier_from_CP(DFdataset["MCP_bias"].compute())

xrcorrected = DFdataset.copy()
xrcorrected.Intensity.data = DFcorrected
xrcorrected.Intensity.attrs['DetectorCorrected'] = 'True'
xrcorrected['multiplier'] = (('time'), DFmultiplier)
xrcorrected.to_netcdf(os.path.join(folder, DFdataname + '_detectorcorrected.nc'))
xrcorrected

# ## Comparison of results
# Finally, we visualize the effect of active tuning of the gain by plotting a spectrum (from a different dataset) measured with regular settings and one with adaptive gain.

# +
hdr = xr.load_dataset(os.path.join(folder, '20190509_142203_3.5um_561.5_IVhdr_BILAYER.nc'))
ref = xr.load_dataset(os.path.join(folder, '20190509_155656_3.5um_562.1_IV_BILAYER.nc'))

fig, axs = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [0.8,2]}, sharex=True, figsize=[4,3])
axs[0].plot(hdr.Energy, hdr.MCP_bias)
axs[0].plot(ref.Energy, ref.MCP_bias)
axs[0].margins(x=0)
axs[0].set_ylabel('$V_{MCP}\ (kV)$')

axs[1].semilogy(hdr.Energy, hdr.Intensity, label='HDR corrected')
axs[1].semilogy(ref.Energy, ref.Intensity, label='Constant $V_{mcp}$, corrected')
axs[1].margins(x=0)
axs[1].set_xlabel('$E_0\ (eV)$')
axs[1].set_ylabel('Reflectivity')
axs[1].legend()
plt.tight_layout(h_pad=0.0, pad=0)
if SAVEFIG:
    plt.savefig('HDRcomparison.pdf')
