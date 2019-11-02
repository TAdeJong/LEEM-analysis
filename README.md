# Quantitative Data Analysis for spectroscopic LEEM.

This repository contains the code to showcase the methods and algorithms presented in the paper 
"[Quantitative analysis of spectroscopic Low Energy Electron Microscopy data: High-dynamic range imaging, drift correction and cluster analysis](https://arxiv.org/abs/1907.13510)".

**This is still very much a WORK IN PROGRESS**

It is organized as a set of notebooks, reproducing the different techniques and algorithms as presented in the paper, as well as the Figures. The notebooks are in some cases supported by a separate Python file with library functions.
For human readable diffs, each notebook is shadowed by a Python file using [jupytext](https://github.com/mwouts/jupytext).

## Implementation
The code makes extensive use of [`dask`](https://dask.org/) for lazy and parallel computation, the N-D labeled arrays and datasets library [`xarray`](http://xarray.pydata.org/), as well as the usual components of the scipy stack such as `numpy`, `matplotlib` and `skimage`.

## Data
The data will be available separately at https://researchdata.4tu.nl/. The zeroth notebook facilitates easy download of all related data.