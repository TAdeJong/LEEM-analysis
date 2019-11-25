# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 0 - Data download
#
# Before testing out the notebooks, we need some data. The data will be available from the [4TU center for research data](https://data.4tu.nl/repository/). The data is available as [10.4121/uuid:7f672638-66f6-4ec3-a16c-34181cc45202](https://data.4tu.nl/repository/uuid:7f672638-66f6-4ec3-a16c-34181cc45202). It can be accessed using [OPeNDAP](https://researchdata.4tu.nl/en/use-4turesearchdata/opendap-and-netcdf/), enabling realtime interaction without full download of the data.
# However, it can be beneficial to create a local copy of the data. This Notebook does exactly this using `xarray`. 
#
# The **total** dataset containing both Bright Field and Dark Field spectroscopic data and for each the raw data, detector corrected data and driftcorrected is about **9.87 GiB**, and it is **not recommended** to download everything. 
#
# The corrected data can be generated locally and is only recommended for download if you plan to not run the detector correction or image registration notebooks.
#
# Finally, it is possible to use only a reduced landing energy range, to reduce download size and computation load. Suitable ranges are suggested below.
#
# But first, let's import some useful packages and define the dataset location:

import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

opendap_location =  'https://opendap.tudelft.nl/thredds/dodsC/data2/uuid'
dataset_UUID = '7f672638-66f6-4ec3-a16c-34181cc45202'
save_folder = './data'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# ## 0.1 Bright Field  & Dark Field dataset
# The cell below downloads the bright field and/or dark field datasets. If you don't look to run the detector correction notebook or the drift correction, add the respective suffix.
#
# The reduced energy range focuses on the layer counts and reduces the download size from **>1.5 GiB** to  **<90 MiB** per dataset.

# +
names = [
         '20171120_160356_3.5um_591.4_IVhdr', 
         '20171120_215555_3.5um_583.1_IVhdr_DF2'
        ]
#suffix = ''
#suffix = '_detectorcorrected'  # Needed for benchmarking and image registration
suffix = '_driftcorrected'  # Fully corrected data, used for dimension reduction and clustering

# Reduced BF range from 0 to 5 eV in 0.2 eV steps, DF range from ... to ...
red_ranges = [slice(42,92,2) , slice(42,92,2)]
for i, name in enumerate(names):
    location = os.path.join(opendap_location, dataset_UUID, name + suffix + '.nc')
    data = xr.open_dataset(location)
    if red_ranges:
        data = data.isel(time=red_ranges[i])
    data.to_netcdf(os.path.join(save_folder,
                                name + suffix + '.nc'))
# -

# ## 0.2 Detector Correction
#
# The cell below downloads all additional data needed for the Detector Correction notebook, providing Dark Count data and HDR calibration curves for the ESCHER setup. 
#
# Total download size **~ 600 MiB**.

names = ['20171205_115846_0.66um_475.8_DC',
         '20171205_100540_0.66um_479.2_sweepCHANNELPLATE_SET_lowintensity_v2',
         '20171205_103440_0.66um_479.2_sweepCHANNELPLATE_SET_highintensity',
         '20171205_143305_31um_474.2_sweepCHANNELPLATE_SET_higherintensity',
         '20190509_155656_3.5um_562.1_IV_BILAYER',
         '20190509_142203_3.5um_561.5_IVhdr_BILAYER']
for name in names:
    location = os.path.join(opendap_location, dataset_UUID, name + '.nc')
    data = xr.open_dataset(location)
    data.to_netcdf(os.path.join(save_folder,
                                name + '.nc'))

# ## 0.3 Benchmark reference
# The cells below downloads the reference results for Accureacy testing and Benchmarking notebooks. As the full parameter range of these notebooks takes **days** to compute on 2019's hardware, the download of **< 4 MiB** is recommended.

name = 'benchmarkresult_reference'
location = os.path.join(opendap_location, dataset_UUID, name + '.nc')
data = xr.open_dataset(location)
data.to_netcdf(os.path.join(save_folder, name + '.nc'))
data

name = 'accuracy_result_reference'
location = os.path.join(opendap_location, dataset_UUID, name + '.nc')
data = xr.open_dataset(location)
# We need to replace the string dtype for a proper unicode string
data = data.assign_coords(direction=np.char.decode(data.direction.data))
data.to_netcdf(os.path.join(save_folder, name + '.nc'))
data


