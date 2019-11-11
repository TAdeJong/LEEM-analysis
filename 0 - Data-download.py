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

# # 0 - Data download
#
# Before testing out the notebooks, we need some data. The data will be available from the [4TU center for research data](https://data.4tu.nl/repository/). The data will be available as [10.4121/uuid:7f672638-66f6-4ec3-a16c-34181cc45202](https://data.4tu.nl/repository/uuid:7f672638-66f6-4ec3-a16c-34181cc45202). It can be accessed using [OPeNDAP](https://researchdata.4tu.nl/en/use-4turesearchdata/opendap-and-netcdf/), enabling realtime interaction without full download of the data.
# However, it can be beneficial to create a local copy of the data. This Notebook does exactly this.

import os

if not os.path.exists('./data'):
    os.mkdir('./data')

# The data is available in a temporary download from https://stack.tadejong.nl/s/VyMhaeFB53ubzsD. Choose "Download Map" in top right and extract the contents to the data folder, or pick individual files.


