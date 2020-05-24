# Quantitative Data Analysis for spectroscopic LEEM.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3539538.svg)](https://doi.org/10.5281/zenodo.3539538)

![Principal Components ANalysis](https://ars.els-cdn.com/content/image/1-s2.0-S0304399119302475-gr10.jpg)


This repository contains the code to showcase the methods and algorithms presented in the paper 
"[Quantitative analysis of spectroscopic Low Energy Electron Microscopy data: High-dynamic range imaging, drift correction and cluster analysis](https://doi.org/10.1016/j.ultramic.2019.112913)".

It is organized as a set of notebooks, reproducing the different techniques and algorithms as presented in the paper, as well as the Figures. The notebooks are in some cases supported by a separate Python file with library functions.
For human readable diffs, each notebook is shadowed by a Python file using [jupytext](https://github.com/mwouts/jupytext).

## Implementation
The code makes extensive use of [`dask`](https://dask.org/) for lazy and parallel computation, the N-D labeled arrays and datasets library [`xarray`](http://xarray.pydata.org/), as well as the usual components of the scipy stack such as `numpy`, `matplotlib` and `skimage`.

## Getting started
* Git clone or download this repository.
* Create a Python environment with the necessary packages, either from [requirements.txt](requirements.txt) or (for `conda` users) from [environment.yml](environment.yml).
* Activate the environment and start a Jupyter notebook and have a look at the notebooks

## Data
The data is available separately via [https://researchdata.4tu.nl/](http://doi.org/10.4121/uuid:7f672638-66f6-4ec3-a16c-34181cc45202). The [zeroth notebook](0%20-%20Data-download.ipynb) facilitates easy download of all (or parts of) related data.
