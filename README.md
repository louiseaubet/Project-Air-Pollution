# Project on IR Spectroscopy for PM2.5


### Description

This project's aim is to determine the corresponding concentration of ammonium sulfate, using he Fourier transform infrared (FT-IR) spectra of fine particulate matter (PM2.5). The strategy is to build a calibration model using both laboratory standards and collocated ambient measurements to build a common basis set.


### Getting Started

This version was designed for python 3.6.6 or higher. To run the model's calculation, use the Python Notebook `main.ipynb`. The parameters of the model can be set by the user in this file.


### Prerequisites

#### Libraries
The following librairies are used:
* [numpy](http://www.numpy.org/) 1.14.3, can be obtained through [anaconda](https://www.anaconda.com/download/)
* [pandas](https://pandas.pydata.org/), also available through anaconda
* [matplotlib](https://matplotlib.org/), also available through anaconda
* [imbalanced-learn](https://pypi.org/project/imbalanced-learn/) 0.7.0 : `pip install imbalanced-learn`
* [scikit-learn](https://scikit-learn.org/stable/index.html) : `pip install -U scikit-learn`
* [plotly](https://plotly.com/python/) : `pip install plotly==4.14.1`


#### Code

To launch the code `main.ipynb`, you need the following codes and files:
* `preprocessing.py` : Deal with loading of `.csv` files and clean the data
* `model.py` : Contains the definition of the class Model
* `helpers.py` : Contains some useful functions for upsampling or cross-validation

The `data/raw` folder is also needed to store the input data. The following files are used : `IMPROVE_2011-2013_filterSubtractionV2.csv`, `IMPROVE_2011_2013_XRF_OC_ions_mug_cm2.csv`, `matched_std_2784_baselined_filter_subtracted.csv`, `FG_Ruthenburg_std_responses.csv`, `wavenumbers_2_zerofilling.txt`.


### Additional content

The folder `Data_analysis` contains some Python Notebooks that allow to better understand the data and the models : `Distributions.ipynb`, `Interpretation_models.ipynb`, `Scatter_plot_sites.ipynb` and `Bias.ipynb`.

The folder `outputs` contains the numerical and graphical results of this project.

The file `read_rds_csv.R` converts the `.rds` files into `.csv` files so they can be loaded in Python. 

The file `report.pdf` contains further informations on the background, the mathematical definition of the models as well as the analysis of the results.


### Authors

Aubet Louise, louise.aubet@epfl.ch


### Project Status

The project was submitted on the 12th of Januray 2021, as a semester project at EPFL.