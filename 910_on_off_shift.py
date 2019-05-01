from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

# For home
sys.path.insert(0,"E:/Google Drive/Research/Lamb shift measurement/Code")

from exp_data_analysis import *
import fosof_data_set_analysis
from quenching_curve_analysis import *
from ZX47_Calibration_analysis import *

import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt

# Package for wrapping long string to a paragraph
import textwrap
#%%
saving_folder_location = 'E:/2017-10-17 Lamb Shift Measurement/Data/FOSOF analyzed data sets'
#%%
on_off_data_df = pd.read_csv('910_on_off_fosof_data.csv', index_col=[0])

# These data sets were acquired for different frequency ranges, which are not related to the experiment + there are some data sets acquired for different flow rates.

# '180610-215802 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 920-922 MHz' - has traces with very low SNR.

on_off_data_df = on_off_data_df.drop(['180613-181439 - FOSOF Acquisition 910 onoff (80 pct) P CGX HIGH - pi config, 8 V per cm PD 120V, 898-900 MHz', '180613-220307 - FOSOF Acquisition 910 onoff (80 pct) P CGX LOW - pi config, 8 V per cm PD 120V, 898-900 MHz', '180511-005856 - FOSOF Acquisition 910 onoff (80 pct), CGX small flow rate - 0 config, 18 V per cm PD 120V', '180609-131341 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 898-900 MHz', '180610-115642 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 920-922 MHz', '180610-215802 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 920-922 MHz', '180611-183909 - FOSOF Acquisition 910 onoff (80 pct) - 0 config, 8 V per cm PD 120V, 898-900 MHz'])

# This data set has very low SNR. For some reason there was almost 100% quenching fraction of the metastabke atoms.
on_off_data_df = on_off_data_df.drop(['180321-225850 - FOSOF Acquisition 910 onoff - 0 config, 14 V per cm PD 120V'])
#%%
# pi-configuration data has opposite slope. It should have opposite frequency shift. To make the frequency shift of the same sign, the phase differences are multiplied by -1.
pi_config_index = on_off_data_df[on_off_data_df ['Configuration'] == 'pi'].index
on_off_data_df.loc[pi_config_index, 'Weighted Mean'] = on_off_data_df.loc[pi_config_index, 'Weighted Mean'] * (-1)
#%%
# Expanding the error bars for the data sets with larger than 1 reduced chi squared

chi_squared_large_index = on_off_data_df[on_off_data_df['Reduced Chi Squared'] > 1].index

on_off_data_df.loc[chi_squared_large_index, 'Weighted STD'] = on_off_data_df.loc[chi_squared_large_index, 'Weighted STD'] * np.sqrt(on_off_data_df.loc[chi_squared_large_index, 'Reduced Chi Squared'])
#%%
on_off_data_grouped = on_off_data_df.groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])

on_off_av_df = on_off_data_grouped.apply(lambda df: straight_line_fit_params(df['Weighted Mean'], df['Weighted STD']))
#%%
on_off_av_df.to_csv('910_on_off_av.csv')
#%%
on_off_av_df
#%%
on_off_data_df
