'''
2018-11-10

In this script I am correcting zero-crossings from the FOSOF data for systematic effects. It is possible that this is the final analysis that is performed on the data.

Each zero-crossing needs to be corrected for several systematic effects:

1. Second-order Doppler shift (SOD). Depends on the accelerating voltage only.

2. AC Stark shift. Depends on the waveguide separation, accelerating voltage, waveguide RF power, beam rms radius, quench offset.

The analysis of the pre-quench 910 ON/OFF data is performed here as well (at the very bottom of the script)

Author: Nikita Bezginov
'''


from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

path_data_df = pd.read_csv(filepath_or_buffer='path_data.csv', delimiter=',', comment='#', header=[0], skip_blank_lines=True, index_col=[0])

code_folder_path = path_data_df.loc['Code Folder'].values[0].replace('\\', '/')
fosof_analyzed_data_folder_path = path_data_df.loc['FOSOF Analyzed Data Folder'].values[0].replace('\\', '/')
wvg_calib_data_folder_path = path_data_df.loc['Waveguide Calibration Folder'].values[0].replace('\\', '/')
krytar109B_pwr_det_calib_folder_path = path_data_df.loc['KRYTAR 109 B Power Detector Calibration Data Folder'].values[0].replace('\\', '/')
travis_data_folder_path = path_data_df.loc['Travis Data Folder'].values[0].replace('\\', '/')
these_high_n_folder_path = path_data_df.loc['Thesis High-n Shift Folder'].values[0].replace('\\', '/')
these_ac_folder_path = path_data_df.loc['Thesis AC Shift Folder'].values[0].replace('\\', '/')
these_beam_speed_folder_path = path_data_df.loc['Thesis Speed Measurement Folder'].values[0].replace('\\', '/')
these_phase_control_folder_path = path_data_df.loc['Thesis Phase Control Folder'].values[0].replace('\\', '/')


sys.path.insert(0, code_folder_path)

from exp_data_analysis import *
from beam_speed import *
import fosof_data_set_analysis
import hydrogen_sim_data

os.chdir(code_folder_path)
from travis_data_import import *
os.chdir(code_folder_path)
import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate
import scipy.optimize

import matplotlib.pyplot as plt


import threading
from queue import Queue

# Package for wrapping long string to a paragraph
import textwrap

from tkinter import *
from tkinter import ttk

from tkinter import messagebox

from uncertainties import ufloat
#import wvg_power_calib_analysis
#%%
# Load the zero-crossing frequencies
saving_folder_location = fosof_analyzed_data_folder_path

fosof_lineshape_param_file_name = 'fosof_lineshape_param.csv'

os.chdir(saving_folder_location)

grouped_exp_fosof_lineshape_param_df = pd.read_csv(filepath_or_buffer=fosof_lineshape_param_file_name, delimiter=',', comment='#', header=[0, 1, 2, 3, 4], skip_blank_lines=True, index_col=[0, 1, 2, 3, 4, 5, 6, 7])
#%%
''' Statistical analysis of the lineshape data. All of the frequency range multiples are averaged. Results obtained for each of the RF combiners are analyzed separately. At the end it is assumed that the uncertainty due to the difference in the resonant frequencies obtained from the combiners is systematic, not statistical.
'''

# In case if a lineshape has reduced chi-squared of larger than 1 then we expand the uncertainty in the fit parameters to make the chi-squared to be one.

# Addidng the columns of normalized = corrected for large reduced chi-squared, uncertainties in the fit parameters.
fosof_lineshape_param_df = grouped_exp_fosof_lineshape_param_df.join(grouped_exp_fosof_lineshape_param_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Zero-crossing Frequency STD [MHz]', 'Slope STD [Rad/MHz]', 'Offset STD [MHz]'])].rename(columns={'Zero-crossing Frequency STD [MHz]': 'Zero-crossing Frequency STD (Normalized) [MHz]', 'Slope STD [Rad/MHz]': 'Slope STD (Normalized) [Rad/MHz]', 'Offset STD [MHz]': 'Offset STD (Normalized) [MHz]'})).sort_index(axis='columns')
#%%
# We now adjust the uncertainties for getting the reduced chi-squared of at least 1.
def normalize_std(df):

    ref_type = df.columns.get_level_values('Phase Reference Type')[0]
    harm_type = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]

    large_chi_squared_index = df[df.loc[slice(None), (ref_type, harm_type, av_type, std_type, 'Reduced Chi-Squared')] > 1].index

    df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Zero-crossing Frequency STD (Normalized) [MHz]')] = df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Zero-crossing Frequency STD [MHz]')] * np.sqrt(df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Reduced Chi-Squared')])

    df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Slope STD (Normalized) [Rad/MHz]')] = df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Slope STD [Rad/MHz]')] * np.sqrt(df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Reduced Chi-Squared')])

    df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Offset STD (Normalized) [MHz]')] = df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Offset STD [MHz]')] * np.sqrt(df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Reduced Chi-Squared')])

    return df

fosof_lineshape_param_norm_df = fosof_lineshape_param_df.groupby(level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns').apply(normalize_std)
#%%
# Statistical averaging of the FOSOF data.

def calc_av_freq_for_analysis_type(df):

    ref_type = df.columns.get_level_values('Phase Reference Type')[0]
    harm_type = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]

    df = df[ref_type, harm_type, av_type, std_type]

    data_arr = df['Zero-crossing Frequency [MHz]']
    data_std_arr = df['Zero-crossing Frequency STD [MHz]']

    av_s = straight_line_fit_params(data_arr, data_std_arr)

    av_s['Weighted STD (Normalized)'] = av_s['Weighted STD']
    if av_s['Reduced Chi Squared'] > 1:
        av_s['Weighted STD (Normalized)'] = av_s['Weighted STD'] * np.sqrt(av_s['Reduced Chi Squared'])

    av_s.name = 'Not Normalized'

    data_arr = df['Zero-crossing Frequency [MHz]']
    data_std_arr = df['Zero-crossing Frequency STD (Normalized) [MHz]']

    av_norm_s = straight_line_fit_params(data_arr, data_std_arr)

    av_norm_s['Weighted STD (Normalized)'] = av_norm_s['Weighted STD']
    if av_norm_s['Reduced Chi Squared'] > 1:
        av_norm_s['Weighted STD (Normalized)'] = av_norm_s['Weighted STD'] * np.sqrt(av_norm_s['Reduced Chi Squared'])

    av_norm_s.name = 'Normalized'

    return pd.DataFrame(av_s).join(pd.DataFrame(av_norm_s))

def calc_av_zero_cross_freq(group_df):

    return group_df.groupby(level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns').apply(calc_av_freq_for_analysis_type)

def calc_av_slope_for_analysis_type(df):

    ref_type = df.columns.get_level_values('Phase Reference Type')[0]
    harm_type = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]

    df = df[ref_type, harm_type, av_type, std_type]

    data_arr = df['Slope [Rad/MHz]']
    data_std_arr = df['Slope STD [Rad/MHz]']

    av_s = straight_line_fit_params(data_arr, data_std_arr)

    av_s['Weighted STD (Normalized)'] = av_s['Weighted STD']
    if av_s['Reduced Chi Squared'] > 1:
        av_s['Weighted STD (Normalized)'] = av_s['Weighted STD'] * np.sqrt(av_s['Reduced Chi Squared'])

    av_s.name = 'Not Normalized'

    data_arr = df['Slope [Rad/MHz]']
    data_std_arr = df['Slope STD (Normalized) [Rad/MHz]']

    av_norm_s = straight_line_fit_params(data_arr, data_std_arr)

    av_norm_s['Weighted STD (Normalized)'] = av_norm_s['Weighted STD']
    if av_norm_s['Reduced Chi Squared'] > 1:
        av_norm_s['Weighted STD (Normalized)'] = av_norm_s['Weighted STD'] * np.sqrt(av_norm_s['Reduced Chi Squared'])

    av_norm_s.name = 'Normalized'

    return pd.DataFrame(av_s).join(pd.DataFrame(av_norm_s))

def calc_av_slope(group_df):

    return group_df.groupby(level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns').apply(calc_av_slope_for_analysis_type)
#%%
# First we average the data for given beam rms radius, waveguide separation, accelerating voltage, Proton deflector voltage, RF E field amplitude, and the Frequency Range Multiple. We look at the waveguide carrier frequency sweep-type experiments only.

zero_cross_av_df = fosof_lineshape_param_norm_df.loc[('Waveguide Carrier Frequency Sweep'), (slice(None))].groupby(['Beam RMS Radius [mm]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple']).apply(calc_av_zero_cross_freq).unstack(level=-1)

zero_cross_av_df.columns.names = ['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

zero_cross_av_df.rename(columns={'Weighted Mean': 'Zero-crossing Frequency [MHz]', 'Weighted STD': 'Zero-crossing Frequency STD [MHz]', 'Weighted STD (Normalized)': 'Zero-crossing Frequency STD (Normalized) [MHz]'}, level='Data Field', inplace=True)

slope_av_df = fosof_lineshape_param_norm_df.loc[('Waveguide Carrier Frequency Sweep'), (slice(None))].groupby(['Beam RMS Radius [mm]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple']).apply(calc_av_slope).unstack(level=-1)

slope_av_df.columns.names = ['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

slope_av_df.rename(columns={'Weighted Mean': 'Slope [Rad/MHz]', 'Weighted STD': 'Slope STD [Rad/MHz]', 'Weighted STD (Normalized)': 'Slope STD (Normalized) [Rad/MHz]'}, level='Data Field', inplace=True)
#%%
# There are some data sets that were acquired with the PD turned off. We do not want to include them into the analysis.

zero_cross_av_no_pd_df = zero_cross_av_df.loc[zero_cross_av_df.loc[(slice(None), slice(None), slice(None), 0), (slice(None))].index]

zero_cross_av_df = zero_cross_av_df.loc[zero_cross_av_df.index.difference(zero_cross_av_no_pd_df.index)]

slope_av_no_pd_df = slope_av_df.loc[slope_av_df.loc[(slice(None), slice(None), slice(None), 0), (slice(None))].index]

slope_av_df = slope_av_df.loc[slope_av_df.index.difference(slope_av_no_pd_df.index)]
#%%
''' Determining the average zero-crossing frequency using both of the RF Combiners, the RMS STD, and the systematic uncertainty due to the difference in the frequencies obtained with the two combiners.

In principle, it is better to perform this averaging not on the zero-crossings, but on the FOSOF phases for each of the 0-pi data sets. This is explained on p 36-53 in Lab Notes #4 written on August 31, 2018. However, our method, of analyzing the lineshapes for different combiners separately, should be sufficient. In this case, the only downside, is that the systematic shift due to combiners will be somewhat larger.
'''
rf_combiner_I_zero_cross_df = zero_cross_av_df['RF Combiner I Reference'].loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]'])]

rf_combiner_R_zero_cross_df = zero_cross_av_df['RF Combiner R Reference'].loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]'])]

zero_cross_freq_df = zero_cross_av_df['RF Combiner I Reference'].copy()

zero_cross_freq_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')] = (rf_combiner_I_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')]  + rf_combiner_R_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')] ) / 2

zero_cross_freq_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD [MHz]')] = np.sqrt((rf_combiner_I_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD [MHz]')]**2  + rf_combiner_R_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD [MHz]')]**2 ) / 2)

zero_cross_freq_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD (Normalized) [MHz]')] = np.sqrt((rf_combiner_I_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD (Normalized) [MHz]')]**2  + rf_combiner_R_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD (Normalized) [MHz]')]**2 ) / 2)

zero_cross_freq_df = zero_cross_freq_df.join(np.abs(rf_combiner_I_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')] - zero_cross_freq_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')]).rename(columns={'Zero-crossing Frequency [MHz]': 'Combiner Uncertainty [MHz]'}, level='Data Field')).sort_index(axis='columns')

#%%
# Performing similar steps for the FOSOF slopes.
rf_combiner_I_slope_df = slope_av_df['RF Combiner I Reference'].loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Slope [Rad/MHz]', 'Slope STD [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]'])]

rf_combiner_R_slope_df = slope_av_df['RF Combiner R Reference'].loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Slope [Rad/MHz]', 'Slope STD [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]'])]

slope_df = slope_av_df['RF Combiner I Reference'].copy()

slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')] = (rf_combiner_I_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')]  + rf_combiner_R_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')] ) / 2

slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD [Rad/MHz]')] = np.sqrt((rf_combiner_I_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD [Rad/MHz]')]**2  + rf_combiner_R_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD [Rad/MHz]')]**2 ) / 2)

slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD (Normalized) [Rad/MHz]')] = np.sqrt((rf_combiner_I_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD (Normalized) [Rad/MHz]')]**2  + rf_combiner_R_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD (Normalized) [Rad/MHz]')]**2 ) / 2)

slope_df = slope_df.join(np.abs(rf_combiner_I_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')] - slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')]).rename(columns={'Slope [Rad/MHz]': 'Combiner Uncertainty [Rad/MHz]'}, level='Data Field')).sort_index(axis='columns')
#%%
zero_cross_freq_df
#%%
os.chdir()
