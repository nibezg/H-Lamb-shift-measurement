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

# os.chdir(code_folder_path)
# import travis_data_import
# os.chdir(code_folder_path)
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
# Blind offset in kHz
BLIND_OFFSET = 0.03174024731 * 1E3
#%%
# Load the zero-crossing frequencies
saving_folder_location = fosof_analyzed_data_folder_path

fosof_lineshape_param_file_name = 'fosof_lineshape_param.csv'

os.chdir(saving_folder_location)

grouped_exp_fosof_lineshape_param_df = pd.read_csv(filepath_or_buffer=fosof_lineshape_param_file_name, delimiter=',', comment='#', header=[0, 1, 2, 3, 4], skip_blank_lines=True, index_col=[0, 1, 2, 3, 4, 5, 6, 7])
#%%
# Frequencie multiple range to use for the data analysis
freq_mult_to_use_list = [1]
#%%
''' Statistical analysis of the lineshape data. All of the frequency range multiples are averaged. Results obtained for each of the RF combiners are analyzed separately. At the end it is assumed that the uncertainty due to the difference in the resonant frequencies obtained from the combiners is systematic, not statistical.
'''

# Best estimate of the beam rms radius that was determined from the Monte Carlo simulation.
beam_rms_rad_best_est = 1.6

# Beam RMS radius to use in the statistical average of the data. If the value of -1 is used, then it corresponds to the uncorrected data for imperfect power flatness observed during data acquisitions.
beam_rms_rad_best_est_to_use = 1.6

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
# First we average the data for given waveguide separation, accelerating voltage, Proton deflector voltage, RF E field amplitude, and the Frequency Range Multiple. We only look at the data for the best estimate of the beam RMS radius, because the uncertainty due to this parameter is already included in the AC shift. We also look at the waveguide carrier frequency sweep-type experiments only.

zero_cross_av_df = fosof_lineshape_param_norm_df.loc[('Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est_to_use), (slice(None))].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple']).apply(calc_av_zero_cross_freq).unstack(level=-1)

zero_cross_av_df.columns.names = ['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

zero_cross_av_df.rename(columns={'Weighted Mean': 'Zero-crossing Frequency [MHz]', 'Weighted STD': 'Zero-crossing Frequency STD [MHz]', 'Weighted STD (Normalized)': 'Zero-crossing Frequency STD (Normalized) [MHz]'}, level='Data Field', inplace=True)

slope_av_df = fosof_lineshape_param_norm_df.loc[('Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est_to_use), (slice(None))].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple']).apply(calc_av_slope).unstack(level=-1)

slope_av_df.columns.names = ['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

slope_av_df.rename(columns={'Weighted Mean': 'Slope [Rad/MHz]', 'Weighted STD': 'Slope STD [Rad/MHz]', 'Weighted STD (Normalized)': 'Slope STD (Normalized) [Rad/MHz]'}, level='Data Field', inplace=True)
#%%
# There are some data sets that were acquired with the PD turned off. We do not want to include them into the analysis.
zero_cross_av_no_pd_df = zero_cross_av_df.loc[zero_cross_av_df.loc[(slice(None), slice(None), 0), (slice(None))].index]

zero_cross_av_df = zero_cross_av_df.loc[zero_cross_av_df.index.difference(zero_cross_av_no_pd_df.index)]

slope_av_no_pd_df = slope_av_df.loc[slope_av_df.loc[(slice(None), slice(None), 0), (slice(None))].index]

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
''' Frequency shift due to higher-n states or imperfect quenching of the 2S_{1/2} f=1 states.

We assume that the phasor from the higher-n states due to FOSOF that gets added to the 2S_1/2 -> 2P_1/2 f=0 transition is always of the same size, independent of RF power, beam speed, and waveguide separation. For the RF power: we assume that the transitions involving high-n states are saturated at any RF powers used in the experiment, since they involves the states with high n => dipole moments are large. For the waveguide separations: again, since the states have high n values => the lifetimes are very large => decay is not an issue.

We also assume that this phasor is always perpendicular to the 2S_1/2 -> 2P_1/2 f=0 phasor, which is the worse case scenario, since in this case the frequency shift is the largest. But this allows us to estimate the frequency shift for the case when pre-quench 910 cavity is OFF (usual FOSOF). But in addition, we assume that this normality is the case for all of the RF powers, beam speeds and waveguide separations. This lets us estimate the shifts for other experiment parameters, by scaling them by the FOSOF amplitude factor, determined from the simulations.

'''

# We first look at the pre-quench 910 cavity On?Off switching data.
os.chdir(saving_folder_location)

on_off_data_df = pd.read_csv('910_on_off_fosof_data.csv', index_col=[0])

# These data sets were acquired for different frequency ranges, which are not related to the experiment + there are some data sets acquired for different flow rates.

# '180610-215802 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 920-922 MHz' - has traces with very low SNR.

on_off_data_df = on_off_data_df.drop(['180613-181439 - FOSOF Acquisition 910 onoff (80 pct) P CGX HIGH - pi config, 8 V per cm PD 120V, 898-900 MHz', '180613-220307 - FOSOF Acquisition 910 onoff (80 pct) P CGX LOW - pi config, 8 V per cm PD 120V, 898-900 MHz', '180511-005856 - FOSOF Acquisition 910 onoff (80 pct), CGX small flow rate - 0 config, 18 V per cm PD 120V', '180609-131341 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 898-900 MHz', '180610-115642 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 920-922 MHz', '180610-215802 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 920-922 MHz', '180611-183909 - FOSOF Acquisition 910 onoff (80 pct) - 0 config, 8 V per cm PD 120V, 898-900 MHz'])

# This data set has very low SNR. For some reason there was almost 100% quenching fraction of the metastabke atoms.
on_off_data_df = on_off_data_df.drop(['180321-225850 - FOSOF Acquisition 910 onoff - 0 config, 14 V per cm PD 120V'])

# pi-configuration data has opposite slope. It should have opposite frequency shift. To make the frequency shift of the same sign, the phase differences are multiplied by -1.
pi_config_index = on_off_data_df[on_off_data_df ['Configuration'] == 'pi'].index
on_off_data_df.loc[pi_config_index, 'Weighted Mean'] = on_off_data_df.loc[pi_config_index, 'Weighted Mean'] * (-1)

# Expanding the error bars for the data sets with larger than 1 reduced chi squared

chi_squared_large_index = on_off_data_df[on_off_data_df['Reduced Chi Squared'] > 1].index

on_off_data_df.loc[chi_squared_large_index, 'Weighted STD'] = on_off_data_df.loc[chi_squared_large_index, 'Weighted STD'] * np.sqrt(on_off_data_df.loc[chi_squared_large_index, 'Reduced Chi Squared'])

on_off_data_df = on_off_data_df.reset_index().set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])

# Join the averaged slopes. The pre-910 on/off switching data was acquired only for 49.86 kV accelerating voltage at 120 V proton deflector voltage with +-2MHz range around 910MHz.
on_off_data_df = on_off_data_df.join(slope_df.loc[(slice(None), 49.86, 120, slice(None), 1), slice(None)].reset_index().set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD'].loc[on_off_data_df.index.drop_duplicates()]['Normalized']['Slope [Rad/MHz]'])

# Calculate the frequency shift
on_off_data_df['Frequency Shift [MHz]'] = -on_off_data_df['Weighted Mean'] / on_off_data_df['Slope [Rad/MHz]'] * (1-on_off_data_df['F=0 Fraction Quenched'])

on_off_data_df['Frequency Shift STD [MHz]'] = -on_off_data_df['Weighted STD'] / on_off_data_df['Slope [Rad/MHz]'] * (1-on_off_data_df['F=0 Fraction Quenched'])

# Determine average frequency shift for each combination of the waveguide separation, accelerating voltage and the electric field amplitude.
on_off_data_grouped = on_off_data_df.groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])

on_off_av_df = on_off_data_grouped.apply(lambda df: straight_line_fit_params(df['Frequency Shift [MHz]'], df['Frequency Shift STD [MHz]']))

# Normalize the averaged data to reduced chi-squared of 1.

on_off_av_large_chi_squared_index = on_off_av_df[on_off_av_df['Reduced Chi Squared'] > 1].index
on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Weighted STD'] = on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Weighted STD'] * np.sqrt(on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Reduced Chi Squared'])

# Average shift in kHz
on_off_av_df = on_off_av_df[['Weighted Mean', 'Weighted STD']] * 1E3
on_off_av_df.rename(columns={'Weighted Mean': 'Frequency Shift [kHz]', 'Weighted STD': 'Frequency Shift STD [kHz]'}, inplace=True)

# We now want to scale the shifts by their amplitudes, as discussed in the beginning, to be able to estimate the average shift at one of the RF powers and waveguide separations, which can later be mapped to any other beam configuration by knowing the FOSOF amplitude ratios for these configurations.

# Obtaining the amplitude ratios:

# Unique experiment configuration parameters from all of the data
param_arr = zero_cross_freq_df.reset_index(['Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple'], drop=True).index.drop_duplicates().values

# Hydrogen FOSOF simulations
fosof_sim_data = hydrogen_sim_data.FOSOFSimulation()
ampl_mean_df = pd.DataFrame()
freq_arr = np.linspace(908, 912, 41)

# Determine average FOSOF amplitude for each experiment configuration
for sim_param_tuple in param_arr:

    wvg_sep = sim_param_tuple[0]
    acc_val = sim_param_tuple[1]

    sim_params_dict = { 'Frequency Array [MHz]': freq_arr,
                        'Waveguide Separation [cm]': wvg_sep,
                        'Accelerating Voltage [kV]': acc_val,
                        'Off-axis Distance [mm]': beam_rms_rad_best_est_to_use}

    zero_crossing_vs_off_axis_dist_chosen_df, fosof_sim_info_chosen_df = fosof_sim_data.filter_fosof_sim_set(sim_params_dict, blind_freq_Q=False)

    ampl_mean_chosen_df = fosof_sim_data.fosof_sim_data_df.loc[fosof_sim_info_chosen_df.index][['Amplitude']].groupby('Simulation Key').aggregate('mean').join(fosof_sim_data.fosof_sim_info_df.loc[fosof_sim_info_chosen_df.index][['Waveguide Separation [cm]', 'Waveguide Electric Field [V/cm]']])

    ampl_mean_chosen_df['Accelerating Voltage [kV]'] = acc_val
    ampl_mean_chosen_df.set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'], inplace=True)

    ampl_mean_df = ampl_mean_df.append(ampl_mean_chosen_df)

ampl_mean_df = ampl_mean_df.loc[ampl_mean_df.index.intersection(zero_cross_freq_df.reset_index(['Proton Deflector Voltage [V]', 'Frequency Range Multiple'], drop=True).index.drop_duplicates())]

# Normalize the amplitude to their minimal value.
ampl_mean_df['Amplitude (Normalized)'] = ampl_mean_df['Amplitude'] / ampl_mean_df.min().values

# Change the type of the Waveguide Electric Field index level to a float from integers
on_off_av_df = on_off_av_df.reset_index('Waveguide Electric Field [V/cm]')
on_off_av_df['Waveguide Electric Field [V/cm]'] = on_off_av_df['Waveguide Electric Field [V/cm]'].astype(np.float64)
on_off_av_df.set_index('Waveguide Electric Field [V/cm]', append=True, inplace=True)

# Join with the normalized-amplitude-scaling factors
on_off_av_df = on_off_av_df.join(ampl_mean_df.loc[on_off_av_df.index])

# The larger the FOSOF amplitude the smaller the shift should be (in a proportional way)
# Calculate scaled frequency shifts
on_off_av_df['Scaled Frequency Shift [kHz]'] = on_off_av_df['Frequency Shift [kHz]'] * on_off_av_df['Amplitude (Normalized)']
on_off_av_df['Scaled Frequency Shift STD [kHz]'] = on_off_av_df['Frequency Shift STD [kHz]'] * on_off_av_df['Amplitude (Normalized)']

av_high_n_shift_max_s = straight_line_fit_params(on_off_av_df['Scaled Frequency Shift [kHz]'], on_off_av_df['Scaled Frequency Shift STD [kHz]'])

if av_high_n_shift_max_s['Reduced Chi Squared'] > 1:
    av_high_n_shift_max_s['Weighted STD'] = np.sqrt(av_high_n_shift_max_s['Reduced Chi Squared'])*(av_high_n_shift_max_s['Weighted STD'])

av_shift_high_n_df = ampl_mean_df.copy()

av_shift_high_n_df['Pre-910 On-Off Frequency Shift [kHz]'] = av_high_n_shift_max_s['Weighted Mean'] / av_shift_high_n_df['Amplitude (Normalized)']
av_shift_high_n_df['Pre-910 On-Off Frequency Shift STD [kHz]'] = av_high_n_shift_max_s['Weighted STD'] / av_shift_high_n_df['Amplitude (Normalized)']
#%%
''' Analysis of the high-pressure data. The statistical analysis is identical to that of the usual data sets. The data was acquired at 10 x the pressure in the Box. It is assumed that whatever the shift obsereved is supposed to be 10 times smaller at the usual pressure.
'''

fosof_lineshape_param_file_high_press_name = 'fosof_lineshape_high_press_param.csv'

os.chdir(saving_folder_location)

grouped_exp_fosof_lineshape_high_press_param_df = pd.read_csv(filepath_or_buffer=fosof_lineshape_param_file_high_press_name, delimiter=',', comment='#', header=[0, 1, 2, 3, 4], skip_blank_lines=True, index_col=[0, 1, 2, 3, 4, 5, 6, 7])
#%%

''' Statistical analysis of the lineshape data for high pressure
'''

# In case if a lineshape has reduced chi-squared of larger than 1 then we expand the uncertainty in the fit parameters to make the chi-squared to be one.

# Addidng the columns of normalized = corrected for large reduced chi-squared, uncertainties in the fit parameters.
fosof_lineshape_high_press_param_df = grouped_exp_fosof_lineshape_high_press_param_df.join(grouped_exp_fosof_lineshape_high_press_param_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Zero-crossing Frequency STD [MHz]', 'Slope STD [Rad/MHz]', 'Offset STD [MHz]'])].rename(columns={'Zero-crossing Frequency STD [MHz]': 'Zero-crossing Frequency STD (Normalized) [MHz]', 'Slope STD [Rad/MHz]': 'Slope STD (Normalized) [Rad/MHz]', 'Offset STD [MHz]': 'Offset STD (Normalized) [MHz]'})).sort_index(axis='columns')

# We now adjust the uncertainties for getting the reduced chi-squared of at least 1.

fosof_lineshape_high_press_param_norm_df = fosof_lineshape_high_press_param_df.groupby(level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns').apply(normalize_std)

#%%
# Statistaical averaging of the FOSOF data.

# First we average the data for given waveguide separation, accelerating voltage, Proton deflector voltage, RF E field amplitude, and the frequency range multiple. We only look at the data for the best estimate of the beam RMS radius, because the uncertainty due to this parameter is already included in the AC shift. We also look at the waveguide carrier frequency sweep-type experiments only.

zero_cross_av_high_press_df = fosof_lineshape_high_press_param_norm_df.loc[('Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est), (slice(None))].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple']).apply(calc_av_zero_cross_freq).unstack(level=-1)

zero_cross_av_high_press_df.columns.names = ['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

zero_cross_av_high_press_df.rename(columns={'Weighted Mean': 'Zero-crossing Frequency [MHz]', 'Weighted STD': 'Zero-crossing Frequency STD [MHz]', 'Weighted STD (Normalized)': 'Zero-crossing Frequency STD (Normalized) [MHz]'}, level='Data Field', inplace=True)

slope_av_high_press_df = fosof_lineshape_high_press_param_norm_df.loc[('Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est), (slice(None))].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple']).apply(calc_av_slope).unstack(level=-1)

slope_av_high_press_df.columns.names = ['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

slope_av_high_press_df.rename(columns={'Weighted Mean': 'Slope [Rad/MHz]', 'Weighted STD': 'Slope STD [Rad/MHz]', 'Weighted STD (Normalized)': 'Slope STD (Normalized) [Rad/MHz]'}, level='Data Field', inplace=True)

# Determine difference between the zero-crossing frequencies determined from the high-pressure data and usual FOSOF data
zero_cross_high_press_diff_df = zero_cross_av_high_press_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized'] - zero_cross_av_df.loc[zero_cross_av_high_press_df.index]['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized']

zero_cross_high_press_diff_df['Zero-crossing Frequency STD (Normalized) [MHz]'] = np.sqrt(zero_cross_av_high_press_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized', 'Zero-crossing Frequency STD (Normalized) [MHz]']**2 + zero_cross_av_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized', 'Zero-crossing Frequency STD (Normalized) [MHz]']**2)

zero_cross_high_press_diff_df = zero_cross_high_press_diff_df[['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]']]

zero_cross_high_press_diff_df = zero_cross_high_press_diff_df * 1E3
zero_cross_high_press_diff_df.rename(columns={'Zero-crossing Frequency [MHz]': 'Frequency Shift [kHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]': 'Frequency Shift STD [kHz]'}, inplace=True)

zero_cross_high_press_diff_df.reset_index(['Proton Deflector Voltage [V]', 'Frequency Range Multiple'], drop=True, inplace=True)

# The statistical analysis is finished. Time to determine the pressure-dependent shift
# Join with the normalized-amplitude-scaling factors
high_press_df = zero_cross_high_press_diff_df.join(ampl_mean_df.loc[zero_cross_high_press_diff_df.index])

# The larger the FOSOF amplitude the smaller the shift should be (in a proportional way)
# Calculate scaled frequency shifts
high_press_df['Scaled Frequency Shift [kHz]'] = high_press_df['Frequency Shift [kHz]'] * high_press_df['Amplitude (Normalized)']
high_press_df['Scaled Frequency Shift STD [kHz]'] = high_press_df['Frequency Shift STD [kHz]'] * high_press_df['Amplitude (Normalized)']

av_high_press_shift_max_s = straight_line_fit_params(high_press_df['Scaled Frequency Shift [kHz]'], high_press_df['Scaled Frequency Shift STD [kHz]'])

if av_high_press_shift_max_s['Reduced Chi Squared'] > 1:
    av_high_press_shift_max_s['Weighted STD'] = np.sqrt(av_high_press_shift_max_s['Reduced Chi Squared'])*(av_high_press_shift_max_s['Weighted STD'])

av_high_press_shift_df = ampl_mean_df.copy()

# Additional factor of 10 is due to pressure being 10 times larger.

press_mult = 10

av_high_press_shift_df['High-pressure Frequency Shift [kHz]'] = av_high_press_shift_max_s['Weighted Mean'] / av_high_press_shift_df['Amplitude (Normalized)'] / press_mult
av_high_press_shift_df['High-pressure Frequency Shift STD [kHz]'] = av_high_press_shift_max_s['Weighted STD'] / av_high_press_shift_df['Amplitude (Normalized)'] / press_mult

# Combine the pre-910 ON/OFF switching shifts with the pressure-dependent shifts
av_shift_high_n_df = av_shift_high_n_df.join(av_high_press_shift_df[['High-pressure Frequency Shift [kHz]','High-pressure Frequency Shift STD [kHz]']])
#%%
''' Comparison of the data with the Proton Deflector disabled to usual FOSOF data (PD ON).
'''

zero_cross_no_pd_df = zero_cross_av_no_pd_df.reset_index(['Proton Deflector Voltage [V]'], drop=True)

# Determine difference between the zero-crossing frequencies determined from the case when the proton deflector was OFF nad ON. Notice that we are using here the averaged zero-phase crossing frequencies determined for RF Combiner I, not the average of the two combiners. This is so, because we use the RF Combiner I reference for the Proton Deflector OFF data. I assume that it really should not matter, whether we use the average of the combiners, or any of two combiners.
pd_on_off_diff_df = zero_cross_no_pd_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized'] - zero_cross_av_df.reset_index(['Proton Deflector Voltage [V]'], drop=True).loc[zero_cross_no_pd_df.index]['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized']

pd_on_off_diff_df['Zero-crossing Frequency STD (Normalized) [MHz]'] = np.sqrt(zero_cross_no_pd_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized', 'Zero-crossing Frequency STD (Normalized) [MHz]']**2 + zero_cross_av_df.reset_index(['Proton Deflector Voltage [V]'], drop=True).loc[zero_cross_no_pd_df.index]['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized', 'Zero-crossing Frequency STD (Normalized) [MHz]']**2)

pd_on_off_diff_df = pd_on_off_diff_df[['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]']]

pd_on_off_diff_df = pd_on_off_diff_df * 1E3
pd_on_off_diff_df.rename(columns={'Zero-crossing Frequency [MHz]': 'Frequency Shift [kHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]': 'Frequency Shift STD [kHz]'}, inplace=True)

# We select only the data with the frequency range multiple of 1 - since this is the data we are interested in for this experiment.
pd_on_off_diff_df = pd_on_off_diff_df.loc[(slice(None), slice(None), slice(None), 1), slice(None)].reset_index(['Frequency Range Multiple'], drop=True)

# Join with the normalized-amplitude-scaling factors
pd_on_off_df = pd_on_off_diff_df.join(ampl_mean_df.loc[pd_on_off_diff_df.index])

# The larger the FOSOF amplitude the smaller the shift should be (in a proportional way)
# Calculate scaled frequency shifts
pd_on_off_df['Scaled Frequency Shift [kHz]'] = pd_on_off_df['Frequency Shift [kHz]'] * pd_on_off_df['Amplitude (Normalized)']
pd_on_off_df['Scaled Frequency Shift STD [kHz]'] = pd_on_off_df['Frequency Shift STD [kHz]'] * pd_on_off_df['Amplitude (Normalized)']

av_pd_on_off_shift_max_s = straight_line_fit_params(pd_on_off_df['Scaled Frequency Shift [kHz]'], pd_on_off_df['Scaled Frequency Shift STD [kHz]'])

if av_pd_on_off_shift_max_s['Reduced Chi Squared'] > 1:
    av_pd_on_off_shift_max_s['Weighted STD'] = np.sqrt(av_pd_on_off_shift_max_s['Reduced Chi Squared'])*(av_pd_on_off_shift_max_s['Weighted STD'])

av_pd_on_off_shift_df = ampl_mean_df.copy()

av_pd_on_off_shift_df['PD On-Off Frequency Shift [kHz]'] = av_pd_on_off_shift_max_s['Weighted Mean'] / av_pd_on_off_shift_df['Amplitude (Normalized)']
av_pd_on_off_shift_df['PD On-Off Frequency Shift STD [kHz]'] = av_pd_on_off_shift_max_s['Weighted STD'] / av_pd_on_off_shift_df['Amplitude (Normalized)']

# Combine the pre-910 ON/OFF switching shifts with the proton-deflector-dependent shifts
av_shift_high_n_df = av_shift_high_n_df.join(av_pd_on_off_shift_df[['PD On-Off Frequency Shift [kHz]','PD On-Off Frequency Shift STD [kHz]']])
#%%
av_shift_high_n_df.sort_index()
#%%
''' Correction for AC shift, Fractional Offset, and Beam profile.

We want to determine the effect of the fractional offset on the zero-crossing frequencies. For this, to be independent of the calibration data analysis I determine the effect of the fractional offset on the simulated quench curves for given beam speed, and the beam profile. However, it is so that the beam profile has no significant effect on the simulated quench curve, and thus it can be ignored. I am simply using the simulation obtained for the off-axis distance that is the closest to the beam rms radius determined using Monte Carlo simulation.

The reason we need to look at the quench curves is to understand by how much the RF power that we thought we had for given surviving fraction changes when the fractional offset is different.

Now, this effect on the quench curve, in principle, should be determined for every RF frequency, such that one can later determine the effect on each corresponding FOSOF phase for every data set and correct the data accordingly. However, it turns out that all of the frequencies, for which the FOSOF data was acquired, get affected by the same amount. I.e., at each RF frequency for given RF power, the shift in the power for given fractional offset is essentially the same for all frequencies. This allows us to simply construct the FOSOF lineshape that has the same power for all of the frequencies. We do not have to worry about having different power change at different frequencies, which would complicate the analysis.

Notice that the simulated FOSOF lineshapes are sensitive to different beam profiles. Since we do not know the beam profile well, we use the beam RMS radius determined from the Monte Carlo simulation, and assign 50% uncertainty to it. For generality, for each beam RMS radius, I construct the interpolated FOSOF lineshapes, using the simulated data, and determine the AC shift for each of the fractional offsets.
'''

# This is the fractional offset that is our best estimate
fract_offset_best_est = 0.045

low_fract_offset = fract_offset_best_est * 0.5

high_fract_offset = fract_offset_best_est * 1.5

# Array of different possible quench offsets. We assign 50% uncertainty to our best estimate. I also include the fractional offset of 0 (for no real reason).
fract_quench_offset_arr = np.array([0, fract_offset_best_est, high_fract_offset, low_fract_offset])

# The trickiest correction to apply is the AC shift correction. We first create an index that has all of the experiment parameters that are important for this shift
exp_params_AC_shift_df = grouped_exp_fosof_lineshape_param_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD'].reset_index()[['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Beam RMS Radius [mm]']].drop_duplicates()

#exp_params_AC_shift_df = exp_params_AC_shift_df[exp_params_AC_shift_df['Beam RMS Radius [mm]'] != -1].drop_duplicates()

# Beam speed to use for the simulation [cm/ns]. All of the waveguide calibration data that is used for the experiments was acquired at 49.87 kV of accelerating voltage. This was later measured to correspond to roughly 0.3223 cm/ns.
v_speed = 0.3223

# Load power scan simulation data that corresponds to the particular beam speed.
old_sim_info_df = hydrogen_sim_data.OldSimInfo().get_info()

old_quench_sim_info_df = old_sim_info_df[old_sim_info_df['Waveguide Electric Field [V/cm]'] == 'Power Scan']

old_quench_sim_info_df = old_quench_sim_info_df[old_quench_sim_info_df['Speed [cm/ns]'] == v_speed]

# In general there are several available simulations. Pick one of them.
# Waveguide Quenching simulation to use
old_quench_sim_info_s = old_quench_sim_info_df.iloc[3]

# Use the simulation file to obtain the analyzed simulation class instance.
old_quench_sim_set = hydrogen_sim_data.WaveguideOldQuenchCurveSimulationSet(old_quench_sim_info_s)
sim_name_list = old_quench_sim_set.get_list_of_simulations()

# We use 1.8 mm off-axis simulation, because this is the RMS beam radius that is closes to the one determined by using Monte Carlo simulation using distances between and diameters of apertures.
quench_sim_vs_freq_df = old_quench_sim_set.get_simulation_data(sim_name_list[2])

# The quench simulations assume 910 MHz to be the resonant frequency. We pick this frequency for the quench curve for which we determine the effect of the fractional offset
rf_freq = 910

# Obtaining the simulated quench curves corrected for different fractional offsets.
quench_data_df_copy = quench_sim_vs_freq_df.loc[(rf_freq), (slice(None))].copy()

quench_data_vs_offset_dict = {}

for fract_quench_offset in fract_quench_offset_arr:

    # Transformation of the simulated quench data with no offset to the quench curve with the specified offset.
    quench_data_fract_offset_df = quench_data_df_copy*(1-fract_quench_offset) + fract_quench_offset

    quench_sim_data_set_offset = hydrogen_sim_data.OldWaveguideQuenchCurveSimulation(quench_data_fract_offset_df)
    quench_sim_data_set_offset.analyze_data()

    quench_data_vs_offset_dict[fract_quench_offset] = quench_sim_data_set_offset

# These are the electric field values for which that FOSOF data was acquired for the whole Lamb Shift experiment
rf_e_field_arr = exp_params_AC_shift_df['Waveguide Electric Field [V/cm]'].drop_duplicates().values

# To calculate the effect of the fractional offset on the RF power that we were suppose to use for the data sets, we first use the quench curve for our best estimate fractional offset and determine the surviving fractions at the field powers that were used in the experiments.
surv_frac_0_arr = quench_data_vs_offset_dict[fract_offset_best_est].get_surv_frac_with_poly_fit(rf_e_field_arr**2)

# For each of the fractional offsets we now want to find to what RF power this surviving fraction corresponds to. (we apply inverse operation here, in a sense)
e_field_vs_offset_dict = {}
for fract_quench_offset, quench_sim_data_set in quench_data_vs_offset_dict.items():
    e_field_vs_offset_dict[fract_quench_offset] = np.sqrt(quench_sim_data_set.get_RF_power_with_poly_fit(surv_frac_0_arr))

# We construct the dataframe that has the fractional offsets, and the electric fields that we (hopefully) had in the waveguides obtained by assuming having the best estimate fractional offset. For each fractional offset we have the related electric field that we actually had (if we indeed had this particular fractional offset)
rf_pow_vs_offset_df = pd.DataFrame(np.array(list(e_field_vs_offset_dict.values())), columns=rf_e_field_arr, index=list(e_field_vs_offset_dict.keys())).T

rf_pow_vs_offset_df.index.names = ['Electric Field [V/cm]']
rf_pow_vs_offset_df.columns.names = ['Fractional Offset']
#%%
# We have the set of sets of the experiment parameters used for the FOSOF experiments. For each of the set of the parameters we obtain the deviation of the zero-crossing frequency from the resonant frequency. The parameters include different beam rms radii. For each of the set of the parameters the frequency deviation is determined for the specified set of the fractional offsets.
fosof_lineshape_vs_offset_set_df = pd.DataFrame()

# This dataframe is for recording the names of the simulations that are used for the data. Later they will be used to calculate uncertainty associated with all of the interpolations performed for the simulation data.
fosof_sim_data_chosen_df = pd.DataFrame()

# These are the frequencies used to construct interpolated FOSOF lineshape for given RF power, accelerating voltage, waveguide separation, and off-axis distance, which is equivalent to the beam RMS radius. We do not need to specify many frequencies here. We should limit the range of frequencies to 908-912, so that we can use the NEW simulations with the proper phase averaging. Also, for the accelerations of around 16.67 kV, we do not have OLD simulation data at all.
freq_arr = np.array([908, 910, 912])

for index, data_s in exp_params_AC_shift_df.iterrows():
    print(index)

    beam_rms_rad_to_use = data_s['Beam RMS Radius [mm]']

    if data_s['Beam RMS Radius [mm]'] == -1:
         beam_rms_rad_to_use = 0
    # if data_s['Beam RMS Radius [mm]'] == 1.6:
    #      beam_rms_rad_to_use = 1.64
    # if data_s['Beam RMS Radius [mm]'] == 0.8:
    #      beam_rms_rad_to_use = 1.64 / 2
    # if data_s['Beam RMS Radius [mm]'] == 2.4:
    #      beam_rms_rad_to_use = 1.64 * 1.5

    # Load the simulation data

    fosof_sim_data = hydrogen_sim_data.FOSOFSimulation(load_Q=True)

    # Use the appropriate simulation parameters and find the interpolated phase values

    sim_params_dict = { 'Frequency Array [MHz]': freq_arr,
                        'Waveguide Separation [cm]': data_s['Waveguide Separation [cm]'],
                        'Accelerating Voltage [kV]': data_s['Accelerating Voltage [kV]'],
                        'Off-axis Distance [mm]': beam_rms_rad_to_use}

    zero_crossing_vs_off_axis_dist_chosen_df, fosof_sim_info_chosen_df = fosof_sim_data.filter_fosof_sim_set(sim_params_dict, blind_freq_Q=False)

    phase_vs_e_field_poly_fit_df = fosof_sim_data.get_e_field_func()

    e_field_needed = data_s['Waveguide Electric Field [V/cm]']

    fosof_lineshape_vs_offset_dict = {}
    for fract_offset, e_field_to_use in rf_pow_vs_offset_df.loc[e_field_needed].items():

    # Determine interpolated simulation FOSOF lineshape parameters.
        zero_cross_params_s = fosof_sim_data.calc_interp_FOSOF_lineshape_params(e_field_to_use)
        fosof_lineshape_vs_offset_dict[fract_offset] = zero_cross_params_s

    fosof_lineshape_vs_offset_df = pd.DataFrame(fosof_lineshape_vs_offset_dict)
    fosof_lineshape_vs_offset_df.index.names = ['FOSOF Lineshape Fit Parameters']

    fosof_lineshape_vs_offset_df['Waveguide Separation [cm]'] = data_s['Waveguide Separation [cm]']
    fosof_lineshape_vs_offset_df['Accelerating Voltage [kV]'] = data_s['Accelerating Voltage [kV]']
    fosof_lineshape_vs_offset_df['Beam RMS Radius [mm]'] = data_s['Beam RMS Radius [mm]']
    fosof_lineshape_vs_offset_df['Waveguide Electric Field [V/cm]'] = data_s['Waveguide Electric Field [V/cm]']

    fosof_lineshape_vs_offset_set_df = fosof_lineshape_vs_offset_set_df.append(fosof_lineshape_vs_offset_df)

    index_to_append_in = fosof_sim_info_chosen_df.index.difference(fosof_sim_data_chosen_df.index)

    fosof_sim_data_chosen_df = fosof_sim_data_chosen_df.append(fosof_sim_info_chosen_df.loc[index_to_append_in])

fosof_lineshape_vs_offset_set_df.set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Beam RMS Radius [mm]'], append=True, inplace=True)

fosof_lineshape_vs_offset_set_df.columns.names = ['Fractional Offset']

fosof_lineshape_vs_offset_set_df = fosof_lineshape_vs_offset_set_df.unstack(level='FOSOF Lineshape Fit Parameters').stack(level='Fractional Offset')

fosof_lineshape_vs_offset_set_df = fosof_lineshape_vs_offset_set_df.reset_index().set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Beam RMS Radius [mm]', 'Fractional Offset'])

fosof_lineshape_vs_offset_set_df = fosof_lineshape_vs_offset_set_df.astype(dtype={'Fractional Slope Deviation [ppt]': np.float64, 'Largest Absolute Residual [mrad]': np.float64, 'Resonant Frequency Offset [kHz]': np.float64, 'Slope [Rad/MHz]': np.float64, 'Resonant Frequency Deviation [kHz]': np.float64})
#%%
# The interpolation routine for extracting the FOSOF lineshape is not perfect. We have an RMS error of 0.41 kHz (and the maximum error of 0.85 kHz) in determining the resonant frequency by using this technique as opposed to calculating the resonant frequency of the simulated data directly. This error gets added to the AC shift list of errors. The reason the interpolation does not give the perfect agreement is probably, because, the field power shift is not dependent on the 2-order polynomial exactly, but it has some local maxima and minima. It might be also due to numerical uncertainty in the simulation, but since I do not know, I will simply assume that this is the independent uncertainty.

interpolation_unc_df = fosof_sim_data.get_fosof_interpolation_unc().reset_index().set_index('Simulation Key').loc[fosof_sim_data_chosen_df.index].set_index('Off-axis Distance [mm]', append=True).sort_index()

rms_sim_interp_unc = interpolation_unc_df.aggregate(lambda x: np.sqrt(np.sum(x**2)/x.shape[0]))['Resonant Frequency Deviation [kHz]']
#%%
# We now calculate the effect on the resonant frequency due to the beam RMS radius.

# Firstly, the resonant frequency deviations are calculated for each fractional offset. The deviation calculated for different beam rms radii for each fractional offset. The analysis is similar to the one for the fractional offset.

rms_rad_shift_df = fosof_lineshape_vs_offset_set_df.reset_index('Beam RMS Radius [mm]').sort_index()

# This is the frequency shift that we get due to different beam rms radii from the best estimate beam rms radius.
rms_rad_shift_df['Frequency Shift From Best Estimate Beam RMS Radius [kHz]'] = rms_rad_shift_df['Resonant Frequency Offset [kHz]'] - rms_rad_shift_df[rms_rad_shift_df['Beam RMS Radius [mm]'] == beam_rms_rad_best_est]['Resonant Frequency Offset [kHz]']

rms_rad_shift_df = rms_rad_shift_df.set_index(['Beam RMS Radius [mm]'], append=True).sort_index()

small_beam_rms_rad = beam_rms_rad_best_est * 0.5

large_beam_rms_rad = round(beam_rms_rad_best_est * 1.5, 3)
# RMS uncertainty due to the max and min limits in the beam RMS radii for each fractional offset.

rms_beam_rms_rad_unc_s = np.sqrt(((rms_rad_shift_df.loc[(slice(None), slice(None), slice(None), slice(None), small_beam_rms_rad), ('Frequency Shift From Best Estimate Beam RMS Radius [kHz]')].reset_index('Beam RMS Radius [mm]', drop=True))**2 + (rms_rad_shift_df.loc[(slice(None), slice(None), slice(None), slice(None), large_beam_rms_rad), ('Frequency Shift From Best Estimate Beam RMS Radius [kHz]')].reset_index('Beam RMS Radius [mm]', drop=True))**2) / 2)

rms_beam_rms_rad_unc_s = rms_beam_rms_rad_unc_s.groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).aggregate(lambda x: np.sqrt(np.sum(x**2)/x.shape[0]))

rms_beam_rms_rad_unc_s.name = 'Beam RMS Radius Uncertainty [kHz]'
rms_beam_rms_rad_unc_df = pd.DataFrame(rms_beam_rms_rad_unc_s)
#%%
# Another way to calculate the uncertainty due to the beam radius is to simply assign the uncertainty to be equal to 50% of the resonant frequency difference with the best estimate for the rms beam radius and on-axis case. If one compares the shifts obtained this way, then they are almost 50% smaller.
beam_rms_rad_fract_unc = 0.5

rms_rad_shift_df = fosof_lineshape_vs_offset_set_df.reset_index('Beam RMS Radius [mm]').sort_index()

# This is the frequency shift that we get due to fractional offsets.
rms_rad_shift_df['Beam RMS Radius Frequency Shift [kHz]'] = rms_rad_shift_df['Resonant Frequency Offset [kHz]'] - rms_rad_shift_df[rms_rad_shift_df['Beam RMS Radius [mm]'] == -1]['Resonant Frequency Offset [kHz]']

rms_rad_shift_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] = rms_rad_shift_df['Beam RMS Radius Frequency Shift [kHz]'] * beam_rms_rad_fract_unc

rms_rad_shift_df.set_index('Beam RMS Radius [mm]', append=True, inplace=True)

# We select the data corresponding to no fractional offset.
rms_rad_shift_df = rms_rad_shift_df.loc[(slice(None), slice(None), slice(None), 0,beam_rms_rad_best_est), (['Beam RMS Radius Frequency Shift [kHz]', 'Beam RMS Radius Frequency Shift Uncertainty [kHz]'])].reset_index(['Beam RMS Radius [mm]', 'Fractional Offset'], drop=True)
#%%
# We now want to calculate the deviation the resonant frequency determined for each fractional offset from that of the best estimate for the fractional offset. For each beam RMS radius this is done separately.

offset_shift_df = fosof_lineshape_vs_offset_set_df.reset_index('Fractional Offset')

# This is the frequency shift that we get due to different fractional offsets from the best estimate fractional offset.
offset_shift_df['Frequency Shift From Best Estimate Offset [kHz]'] = offset_shift_df['Resonant Frequency Offset [kHz]'] - offset_shift_df[offset_shift_df['Fractional Offset'] == fract_offset_best_est]['Resonant Frequency Offset [kHz]']

offset_shift_df.set_index('Fractional Offset', append=True, inplace=True)

# This calculated deviation from low and high limits for the fractional offset are taken as the + and - uncertainty in the resonant frequency due to the offset. These limits are not equal. However, we assume that both are equally likely. Thus we take the RMS uncertainty by combining the two deviations.

rms_fract_offset_unc_s = np.sqrt(((offset_shift_df.loc[(slice(None), slice(None), slice(None), slice(None), low_fract_offset), ('Frequency Shift From Best Estimate Offset [kHz]')].reset_index('Fractional Offset', drop=True))**2 + (offset_shift_df.loc[(slice(None), slice(None), slice(None), slice(None), high_fract_offset), ('Frequency Shift From Best Estimate Offset [kHz]')].reset_index('Fractional Offset', drop=True))**2) / 2)

# This is calculated for every beam RMS radius. We combine all of these RMS uncertainties for all of the beam radii and find their RMS to give the final uncertainty due to the fractional offset.

rms_fract_offset_unc_s = rms_fract_offset_unc_s.groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).aggregate(lambda x: np.sqrt(np.sum(x**2)/x.shape[0]))

rms_fract_offset_unc_s.name = 'Fractional Offset Uncertainty [kHz]'

rms_fract_offset_unc_df = pd.DataFrame(rms_fract_offset_unc_s)
#%%
# Another way to calculate the uncertainty due to the fractional offset is to simply assign the uncertainty to be equal to 50% of the resonant frequency difference with the best estimate for the fractional offset and with no offset.
fract_offset_fract_unc = 0.5

offset_shift_set_unc_df = fosof_lineshape_vs_offset_set_df.reset_index('Fractional Offset')

# This is the frequency shift that we get due to fractional offsets.
offset_shift_set_unc_df['Fractional Offset Frequency Shift [kHz]'] = offset_shift_set_unc_df['Resonant Frequency Offset [kHz]'] - offset_shift_set_unc_df[offset_shift_set_unc_df['Fractional Offset'] == 0]['Resonant Frequency Offset [kHz]']

offset_shift_set_unc_df['Fractional Offset Frequency Shift Uncertainty [kHz]'] = offset_shift_set_unc_df['Fractional Offset Frequency Shift [kHz]'] * fract_offset_fract_unc

offset_shift_set_unc_df.set_index('Fractional Offset', append=True, inplace=True)

offset_shift_set_unc_df = offset_shift_set_unc_df.loc[(slice(None), slice(None), slice(None), beam_rms_rad_best_est, fract_offset_best_est), (['Fractional Offset Frequency Shift [kHz]', 'Fractional Offset Frequency Shift Uncertainty [kHz]'])].reset_index(['Beam RMS Radius [mm]', 'Fractional Offset'], drop=True)

#%%
# Calculation of the resonant frequency shift due to RF power. This calculation is performed for the best estimate for the beam rms radius and the fractional offset

# Fractional uncertainty in the simulation
simul_unc = 0.05
add_unc = 0.00

field_power_shift_unc = simul_unc + add_unc

# The Field power shift
field_power_shift_s = fosof_lineshape_vs_offset_set_df.loc[(slice(None), slice(None), slice(None), beam_rms_rad_best_est, fract_offset_best_est), ('Resonant Frequency Offset [kHz]')].reset_index(['Beam RMS Radius [mm]', 'Fractional Offset'], drop=True)

field_power_shift_s.name = 'AC Shift [kHz]'

# The uncertainty in the shift
field_power_shift_unc_s = field_power_shift_s * field_power_shift_unc

field_power_shift_unc_s.name = 'AC Shift Simulation Uncertainty [kHz]'

field_power_shift_df = pd.DataFrame(field_power_shift_s)
field_power_shift_unc_df = pd.DataFrame(field_power_shift_unc_s)

ac_shift_df = field_power_shift_unc_df.join([field_power_shift_df, rms_beam_rms_rad_unc_df, rms_fract_offset_unc_df])

# ac_shift_df = ac_shift_df * 1E-3
#
# ac_shift_df.rename(columns={'AC Shift Simulation Uncertainty [kHz]': 'AC Shift Simulation Uncertainty [MHz]', 'AC Shift [kHz]': 'AC Shift [MHz]', 'Beam RMS Radius Uncertainty [kHz]': 'Beam RMS Radius Uncertainty [MHz]', 'Fractional Offset Uncertainty [kHz]': 'Fractional Offset Uncertainty [MHz]'}, inplace=True)

ac_shift_df['Interpolation Uncertainty [kHz]'] = rms_sim_interp_unc

# Calculating the total uncertainty in the AC shift by considering the uncertainties due to uncertainties in knowing the values beam rms radius and the fractional offset.
ac_shift_df['AC Shift Uncertainty (Unknown Parameters) [kHz]'] = np.sqrt(ac_shift_df['AC Shift Simulation Uncertainty [kHz]']**2 + ac_shift_df['Beam RMS Radius Uncertainty [kHz]']**2 + ac_shift_df['Fractional Offset Uncertainty [kHz]']**2 + ac_shift_df['Interpolation Uncertainty [kHz]']**2)

# AC shift formed by combining individual shifts from the power shift, beam rms radius shift, and the fractional offset shift with corresponding uncertainties.
ac_shift_sep_unc_df = fosof_lineshape_vs_offset_set_df.loc[(slice(None), slice(None), slice(None), -1, 0,), (['Resonant Frequency Offset [kHz]'])].reset_index(['Beam RMS Radius [mm]', 'Fractional Offset'], drop=True).join([offset_shift_set_unc_df, rms_rad_shift_df])

ac_shift_sep_unc_df.rename(columns={'Resonant Frequency Offset [kHz]': 'Field Power Shift [kHz]'}, inplace=True)

ac_shift_df = ac_shift_df.join(ac_shift_sep_unc_df)

# This uncertainty is be considering the fractional uncertainties in the total shift due to the beam rms radius and the fractional offset.
ac_shift_df['AC Shift Uncertainty [kHz]'] = np.sqrt(ac_shift_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + ac_shift_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]']**2 + ac_shift_df['Interpolation Uncertainty [kHz]']**2 + ac_shift_df['AC Shift Simulation Uncertainty [kHz]']**2)

# Shifts needed for correcting the frequencies in MHz.
ac_shift_MHz_df = ac_shift_df[['AC Shift [kHz]', 'AC Shift Uncertainty [kHz]']]
ac_shift_MHz_df.rename(columns={'AC Shift [kHz]': 'AC Shift [MHz]', 'AC Shift Uncertainty [kHz]': 'AC Shift Uncertainty [MHz]'}, inplace=True)
ac_shift_MHz_df = ac_shift_MHz_df * 1E-3
#%%
ac_shift_MHz_df
#%%
ac_shift_df
#%%
''' Correction for the Second-order Doppler Shift (SOD)
'''
bs_4_22 = BeamSpeed(acc_volt=22.17, wvg_sep=4)
bs_4_16 = BeamSpeed(acc_volt=16.27, wvg_sep=4)
bs_4_50 = BeamSpeed(acc_volt=49.86, wvg_sep=4)
bs_5_50 = BeamSpeed(acc_volt=49.86, wvg_sep=5)
bs_7_50 = BeamSpeed(acc_volt=49.86, wvg_sep=7)

# fig = plt.figure()
# fig.set_size_inches(10, 7)
# ax = fig.add_subplot(111)
#
# ax = bs_7_50.get_plot(ax)
# fig.tight_layout()
# plt.show()

# These are the speeds that were experimentally measured
beam_speed_data_df = pd.DataFrame(np.array([bs_4_22.calc_speed(), bs_4_16.calc_speed(), bs_4_50.calc_speed(), bs_5_50.calc_speed(), bs_7_50.calc_speed()]), columns=['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Beam Speed [cm/ns]', 'Beam Speed STD [cm/ns]']).set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]'])

# While taking data for different separation, for the same accelerating voltages, it is true that we are not keeping all of the Source parameters (=voltages) the same all the time. The spread in the values that we got for the beam speeds is the good indicator of the variability of the Source parameters that were used for the experiment. Thus the average of these values gives us the best estimate for the speed. The STDOM of the spread is added with quadruate to the RMS uncertainty in the speed values to give us the average uncertainty in the average beam speed.
def get_av_speed(df):

    if df.shape[0] > 1:
        av_s = df[['Beam Speed [cm/ns]']].aggregate(lambda x: np.mean(x))
        av_s['Beam Speed STD [cm/ns]'] = np.sqrt((np.std(df['Beam Speed [cm/ns]'], ddof=1)/np.sqrt(df['Beam Speed STD [cm/ns]'].shape[0]))**2 + np.sum(df['Beam Speed STD [cm/ns]']**2)/df['Beam Speed STD [cm/ns]'].shape[0]**2)
    else:
        av_s = df.iloc[0]
    return av_s

beam_speed_df = beam_speed_data_df.groupby('Accelerating Voltage [kV]').apply(get_av_speed)

beam_speed_df = beam_speed_data_df.reset_index('Waveguide Separation [cm]').join(beam_speed_df, lsuffix='_Delete').drop(columns=['Beam Speed [cm/ns]_Delete', 'Beam Speed STD [cm/ns]_Delete'])

# We add the 6 cm and 49.86 kV point to the dataframe of beam speeds.

wvg_sep_6_s = beam_speed_df.loc[49.86].iloc[0].copy()
wvg_sep_6_s['Waveguide Separation [cm]'] = 6

beam_speed_df = beam_speed_df.append(wvg_sep_6_s)

# Calculate the SOD
# Assumed resonant frequency [MHz]
freq_diff = 909.872
# Speed of light [m/s]
c_light = 299792458

sod_shift_df = beam_speed_df.copy()
sod_shift_df['SOD Shift [MHz]'] = (1/np.sqrt(1-(beam_speed_df['Beam Speed [cm/ns]'] * 1E-2 * 1E9/c_light)**2) - 1) * freq_diff

sod_shift_df['SOD Shift STD [MHz]'] = freq_diff * beam_speed_df['Beam Speed [cm/ns]'] * 1E-2 * 1E9 * beam_speed_df['Beam Speed STD [cm/ns]'] * 1E-2 * 1E9 / ((1 - (beam_speed_df['Beam Speed [cm/ns]']/c_light)**2)**(1.5) * c_light**2)

sod_shift_df = sod_shift_df.set_index('Waveguide Separation [cm]', append=True).swaplevel(0, 1)

#%%
''' Shift due to imperfect phase control upon waveguide reversal. This shift was measured to be about delta_phi = 0.2 mrad. In frequency units it corresponds to the frequency shift of delta_phi / slope. Instead of correcting the data, we add this as the additional type of the uncertainty.
'''

# Shift due to imperfect phase control [Rad]
delta_phi = 0.18 * 1E-3

phase_control_unc_df = np.abs(delta_phi / slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')]).rename(columns={'Slope [Rad/MHz]': 'Phase Control Uncertainty [MHz]'}).copy()
#%%
zero_cross_freq_with_unc_df = zero_cross_freq_df.copy()
# We now want to determine the (blinded) resonant frequencies = zero-crossing frequencies corrected for the systematic shifts and also include all of the systematic uncertainties for each of the determined resonant frequencies.

# Adding the columns to the specific level that will store the systematic shifts
col_name_list = list(ac_shift_MHz_df.columns.union(sod_shift_df.columns))
col_name_list.append('Resonant Frequency (Blinded) [MHz]')
for col_name in col_name_list:

    zero_cross_freq_with_unc_df = zero_cross_freq_with_unc_df.join(zero_cross_freq_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')].rename(columns={'Zero-crossing Frequency [MHz]': col_name}, level='Data Field')).sort_index(axis='columns')

# Addidng the uncertainty due to imperfect phase control
zero_cross_freq_with_unc_df = zero_cross_freq_with_unc_df.join(phase_control_unc_df).sort_index(axis='columns')
#%%
zero_cross_freq_with_unc_df
#%%
def correct_for_sys_shift(df):
    ''' Corrects the zero-crossing frequencies for the systematic shifts, and assign the respective systematic shift uncertainties.
    '''
    harm_type = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]
    std_state = df.columns.get_level_values('STD State')[0]

    df = df.reset_index(['Proton Deflector Voltage [V]', 'Frequency Range Multiple'])

    df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] = df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] - ac_shift_MHz_df['AC Shift [MHz]']

    for col_name in ac_shift_MHz_df.columns:
        df[harm_type, av_type, std_type, std_state, col_name] = ac_shift_MHz_df[col_name]

    df = df.reset_index('Waveguide Electric Field [V/cm]')

    df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] = df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] + sod_shift_df['SOD Shift [MHz]']

    for col_name in sod_shift_df.columns:
        df[harm_type, av_type, std_type, std_state, col_name] = sod_shift_df[col_name]

    df.set_index(['Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple'], append=True, inplace=True)

    return df

# List of corrected frequencies (blinded) with the respective systematic and statistical uncertainties included.
res_freq_df = zero_cross_freq_with_unc_df.groupby(level=['Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State'], axis='columns').apply(correct_for_sys_shift)
#%%
res_freq_df
#%%
''' Calculation of the final average + uncertainty
'''

# Select the averaging type to use for the final frequency determination
#av_type_to_use = 'Phasor Averaging Relative To DC'
av_type_to_use = 'Phasor Averaging'
#av_type_to_use = 'Phase Averaging'

# Whether to use the data sets, uncertainty of which was expanded for chi-squared > 1.
normalized_data_set_Q = True

if normalized_data_set_Q:
    norm_q_col_name = 'Normalized'
else:
    norm_q_col_name = 'Not Normalized'

# Whether to use the data normalized for the reduced chi-squared of > 1. This boolean controls the chi-squared expansion for the case when we average the data sets for the same experiment parameters.
normalized_sets_Q = True

if normalized_sets_Q:
    freq_std_col_name = 'Zero-crossing Frequency STD (Normalized) [MHz]'
    freq_std_col_rename = 'Zero-crossing Frequency STD (Normalized) [kHz]'

    slope_std_col_name = 'Slope STD (Normalized) [Rad/MHz]'
else:
    freq_std_col_name = 'Zero-crossing Frequency STD [MHz]'
    freq_std_col_rename = 'Zero-crossing Frequency STD [kHz]'
    slope_std_col_name = 'Slope STD [Rad/MHz]'

# Here we select the data needed.
df = res_freq_df.loc[(slice(None), slice(None), slice(None), slice(None), 1), slice(None)]['First Harmonic', av_type_to_use, 'Phase RMS Repeat STD', norm_q_col_name].reset_index(['Proton Deflector Voltage [V]', 'Frequency Range Multiple'], drop=True)

# The row with 19.0 V/cm is not needed
df.drop(index=(4.0, 49.86, 19.0), inplace=True)

df = df[['AC Shift Uncertainty [MHz]', 'Combiner Uncertainty [MHz]', 'SOD Shift STD [MHz]', 'Phase Control Uncertainty [MHz]', freq_std_col_name, 'Resonant Frequency (Blinded) [MHz]']]

data_to_use_1_df = df

def minimize_unc(w_arr, df):
    w_sum = np.sum(w_arr)
    #beam_rms_rad_unc_tot = np.sum(df['Beam RMS Radius Uncertainty [MHz]'] * w_arr) / w_sum
    #field_power_shift_unc_tot = np.sum(df['Field Power Shift Uncertainty [MHz]'] * w_arr) / w_sum
    field_power_shift_unc_tot = np.sum(df['AC Shift Uncertainty [MHz]'] * w_arr) / w_sum
    #fract_offset_unc_tot = np.sum(df['Fractional Offset Uncertainty [MHz]'] * w_arr) / w_sum
    #interp_unc_tot = np.sum(df['Interpolation Uncertainty [MHz]'] * w_arr) / w_sum
    comb_unc_tot = np.sum(df['Combiner Uncertainty [MHz]'] * w_arr) / w_sum
    sod_unc_tot = np.sum(df['SOD Shift STD [MHz]'] * w_arr) / w_sum
    phase_control_unc_tot = np.sum(df['Phase Control Uncertainty [MHz]'] * w_arr) / w_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((w_arr*df[freq_std_col_name])**2) / w_sum**2)

    #tot_unc = np.sqrt(beam_rms_rad_unc_tot**2 + field_power_shift_unc_tot**2 + fract_offset_unc_tot**2 + interp_unc_tot**2 + comb_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)

    tot_unc = np.sqrt(field_power_shift_unc_tot**2 + comb_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)
    #tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)

    return tot_unc

def w_tot_constraint(w_arr):
    return np.sum(w_arr) - 1

def find_unc_weights(df):
    w_arr = np.linspace(1, 10, df.shape[0])
    w_arr = w_arr/np.sum(w_arr)

    bounds_to_use_list = [(0, 1) for i in range(w_arr.shape[0])]

    w_min_arr = scipy.optimize.minimize(fun=minimize_unc, x0=w_arr, args=(df), bounds=bounds_to_use_list, constraints={'type': 'eq', 'fun': w_tot_constraint}, tol=1E-5)

    df['Weight'] = w_min_arr['x']

    return df

#df_with_weights = df.loc[(slice(None), slice(None), slice(None), slice(None), 1), (slice(None))].groupby('Waveguide Separation [cm]').apply(find_unc_weights)

df_with_weights = df.groupby('Waveguide Separation [cm]').apply(find_unc_weights)

df_with_weights['Weight'] = df_with_weights['Weight'] / df_with_weights.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().shape[0]

weight_arr_sum = df_with_weights['Weight'].sum()

#beam_rms_rad_unc_tot = np.sum(df_with_weights['Beam RMS Radius Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

#field_power_shift_unc_tot = np.sum(df_with_weights['Field Power Shift Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

field_power_shift_unc_tot = np.sum(df_with_weights['AC Shift Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

#fract_offset_unc_tot = np.sum(df_with_weights['Fractional Offset Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

#interp_unc_tot = np.sum(df_with_weights['Interpolation Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

comb_unc_tot = np.sum(df_with_weights['Combiner Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

sod_unc_tot = np.sum(df_with_weights['SOD Shift STD [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

phase_control_unc_tot = np.sum(df_with_weights['Phase Control Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

zero_cross_stat_unc_tot = np.sqrt(np.sum((df_with_weights['Weight']*df_with_weights[freq_std_col_name])**2) / weight_arr_sum**2)

#tot_unc = np.sqrt(beam_rms_rad_unc_tot**2 + field_power_shift_unc_tot**2 + fract_offset_unc_tot**2 + interp_unc_tot**2 + comb_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)

tot_unc = np.sqrt(field_power_shift_unc_tot**2 + comb_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)

#tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)
#%%
df_with_weights
#%%
phase_control_unc_tot
#%%
zero_cross_stat_unc_tot
#%%
comb_unc_tot
#%%
sod_unc_tot
#%%
field_power_shift_unc_tot
#%%
freq = np.sum(df_with_weights['Resonant Frequency (Blinded) [MHz]'] * df_with_weights['Weight']) / weight_arr_sum
#%%
tot_unc
#%%
f0 = freq + BLIND_OFFSET / 1E3
f0
#%%
(f0-909.8717) * 1E3
#%%
3.2
0.8
#%%
3.3
1.2
#%%
'''
=================================
Different analysis of the data, by combining different (chosen) frequency range multiples together. Also, the RF combiners' difference is assumed to be a statistical uncertainty that gets computed for each data set (grouped data set). I want to compare if we get significantly different result by performing this type of analysis.
=================================
'''
# We want to use only the data for the frequency multiple range of 1 and 2.
fosof_lineshape_param_norm_df = fosof_lineshape_param_norm_df.loc[(slice(None), slice(None), slice(None), slice(None), slice(None), slice(None), freq_mult_to_use_list), slice(None)]
#%%
''' Determination of the average frequency difference between the combiners, acting as different beatnote references for FOSOF.
'''

# Zero-phase-crossing frequencies determined from both of the combiners. The beam RMS radius is of no important. We want the data acquired at 1 x the standard frequency range.
rf_comb_comp_df = grouped_exp_fosof_lineshape_param_df.loc[(slice(None), -1, slice(None), slice(None), slice(None), slice(None), [1, 2]), (slice(None), 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', ['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD [MHz]'])]

rf_comb_I_df = rf_comb_comp_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD']
rf_comb_R_df = rf_comb_comp_df['RF Combiner R Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD']

# Calculation of the frequency differences for all of the data sets
rf_comb_freq_diff_df = rf_comb_I_df.copy()
rf_comb_freq_diff_df['Frequency Difference [kHz]'] = (rf_comb_I_df['Zero-crossing Frequency [MHz]'] - rf_comb_R_df['Zero-crossing Frequency [MHz]']) * 1E3

rf_comb_freq_diff_df['Frequency Difference STD [kHz]'] = rf_comb_freq_diff_df['Zero-crossing Frequency STD [MHz]'] * 1E3

rf_comb_freq_diff_df.drop(columns=['Zero-crossing Frequency STD [MHz]', 'Zero-crossing Frequency [MHz]'], inplace=True)

# Average frequency difference between two combiners. Notice that there is no uncertainty for this number, since the beatnote reference has extremely high SNR.
av_freq_diff = rf_comb_freq_diff_df['Frequency Difference [kHz]'].mean()
av_freq_diff
#%%
fig, ax = plt.subplots()

rf_comb_freq_diff_df.reset_index().reset_index().plot(kind='scatter', x='index', y='Frequency Difference [kHz]', ax=ax)
plt.show()
#%%
''' It seems that on average the RF combiners give the same resonant frequency. Therefore, there seems to be no systematic coupling (that can be detected by their respective resonant frequency difference). We therefore assume that the uncertainty due to this difference for each data set is not systematic, but statistical.

We then can determine the average zero-crossing frequency using both of the RF Combiners, the RMS STD, and the uncertainty due to the difference in the frequencies obtained with the two combiners.

In principle, it is better to perform this averaging not on the zero-crossings, but on the FOSOF phases for each of the 0-pi data sets. This is explained on p 36-53 in Lab Notes #4 written on August 31, 2018. However, our method, of analyzing the lineshapes for different combiners separately, should be sufficient. In this case, the only downside, is that the shift due to combiners will be somewhat larger.
'''

col_rf_comb_to_use = ['Offset [MHz]', 'Slope [Rad/MHz]', 'Zero-crossing Frequency [MHz]', 'Offset STD [MHz]', 'Offset STD (Normalized) [MHz]', 'Slope STD [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]', 'Zero-crossing Frequency STD [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]']

col_data_val_to_use = ['Offset [MHz]', 'Slope [Rad/MHz]', 'Zero-crossing Frequency [MHz]']
col_data_unc_to_use = ['Offset STD [MHz]', 'Offset STD (Normalized) [MHz]', 'Slope STD [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]', 'Zero-crossing Frequency STD [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]']

rf_combiner_I_df = fosof_lineshape_param_norm_df['RF Combiner I Reference'].loc[slice(None), (slice(None), slice(None), slice(None), col_rf_comb_to_use)]

rf_combiner_R_df = fosof_lineshape_param_norm_df['RF Combiner R Reference'].loc[slice(None), (slice(None), slice(None), slice(None), col_rf_comb_to_use)]

fosof_lineshape_param_av_comb_df = rf_combiner_I_df.copy()

fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_val_to_use)] = (rf_combiner_I_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_val_to_use)]  + rf_combiner_R_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_val_to_use)] ) / 2

fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_unc_to_use)] = np.sqrt((rf_combiner_I_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_unc_to_use)]**2  + rf_combiner_R_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_unc_to_use)]**2 ) / 2)

# Adding in quadrature the difference in the parameters between the combiners.
rf_comb_diff_df = (rf_combiner_I_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_val_to_use)] - fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_val_to_use)])

fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset STD [MHz]')] = np.sqrt(fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset STD [MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset [MHz]')].values**2)

fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset STD (Normalized) [MHz]')] = np.sqrt(fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset STD (Normalized) [MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset [MHz]')].values**2)

fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope STD [Rad/MHz]')] = np.sqrt(fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope STD [Rad/MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')].values**2)

fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope STD (Normalized) [Rad/MHz]')] = np.sqrt(fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope STD (Normalized) [Rad/MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')].values**2)

fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD [MHz]')] = np.sqrt(fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD [MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')].values**2)

fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD (Normalized) [MHz]')] = np.sqrt(fosof_lineshape_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD (Normalized) [MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')].values**2)
#%%
# Statistical averaging of the FOSOF data.

def calc_av_freq_for_analysis_type(df):

    harm_type = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]

    df = df[harm_type, av_type, std_type]

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

    return group_df.groupby(level=['Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns').apply(calc_av_freq_for_analysis_type)


def calc_av_slope_for_analysis_type(df):

    harm_type = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]

    df = df[harm_type, av_type, std_type]

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

    return group_df.groupby(level=['Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns').apply(calc_av_slope_for_analysis_type)
#%%
# First we average the data for given waveguide separation, accelerating voltage, Proton deflector voltage, RF E field amplitude. We only look at the data for the best estimate of the beam RMS radius, because the uncertainty due to this parameter is already included in the AC shift. We also look at the waveguide carrier frequency sweep-type experiments only.

# Note that both the frequency range multiple of 1 and 2 are getting averaged together.

zero_cross_av_df = fosof_lineshape_param_av_comb_df.loc[('Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est_to_use), (slice(None))].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]']).apply(calc_av_zero_cross_freq).unstack(level=-1)

zero_cross_av_df.columns.names = ['Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

zero_cross_av_df.rename(columns={'Weighted Mean': 'Zero-crossing Frequency [MHz]', 'Weighted STD': 'Zero-crossing Frequency STD [MHz]', 'Weighted STD (Normalized)': 'Zero-crossing Frequency STD (Normalized) [MHz]'}, level='Data Field', inplace=True)

slope_av_df = fosof_lineshape_param_av_comb_df.loc[('Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est_to_use), (slice(None))].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]']).apply(calc_av_slope).unstack(level=-1)

slope_av_df.columns.names = ['Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

slope_av_df.rename(columns={'Weighted Mean': 'Slope [Rad/MHz]', 'Weighted STD': 'Slope STD [Rad/MHz]', 'Weighted STD (Normalized)': 'Slope STD (Normalized) [Rad/MHz]'}, level='Data Field', inplace=True)

# There are some data sets that were acquired with the PD turned off. We do not want to include them into the analysis.
zero_cross_av_no_pd_df = zero_cross_av_df.loc[zero_cross_av_df.loc[(slice(None), slice(None), 0), (slice(None))].index]

zero_cross_av_df = zero_cross_av_df.loc[zero_cross_av_df.index.difference(zero_cross_av_no_pd_df.index)]

slope_av_no_pd_df = slope_av_df.loc[slope_av_df.loc[(slice(None), slice(None), 0), (slice(None))].index]

slope_av_df = slope_av_df.loc[slope_av_df.index.difference(slope_av_no_pd_df.index)]
#%%
''' Frequency shift due to higher-n states or imperfect quenching of the 2S_{1/2} f=1 states.

We assume that the phasor from the higher-n states due to FOSOF that gets added to the 2S_1/2 -> 2P_1/2 f=0 transition is always of the same size, independent of RF power, beam speed, and waveguide separation. For the RF power: we assume that the transitions involving high-n states are saturated at any RF powers used in the experiment, since they involves the states with high n => dipole moments are large. For the waveguide separations: again, since the states have high n values => the lifetimes are very large => decay is not an issue.

We also assume that this phasor is always perpendicular to the 2S_1/2 -> 2P_1/2 f=0 phasor, which is the worse case scenario, since in this case the frequency shift is the largest. But this allows us to estimate the frequency shift for the case when pre-quench 910 cavity is OFF (usual FOSOF). But in addition, we assume that this normality is the case for all of the RF powers, beam speeds and waveguide separations. This lets us estimate the shifts for other experiment parameters, by scaling them by the FOSOF amplitude factor, determined from the simulations.

'''

# We first look at the pre-quench 910 cavity On?Off switching data.
os.chdir(saving_folder_location)

on_off_data_df = pd.read_csv('910_on_off_fosof_data.csv', index_col=[0])

# These data sets were acquired for different frequency ranges, which are not related to the experiment + there are some data sets acquired for different flow rates.

# '180610-215802 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 920-922 MHz' - has traces with very low SNR.

on_off_data_df = on_off_data_df.drop(['180613-181439 - FOSOF Acquisition 910 onoff (80 pct) P CGX HIGH - pi config, 8 V per cm PD 120V, 898-900 MHz', '180613-220307 - FOSOF Acquisition 910 onoff (80 pct) P CGX LOW - pi config, 8 V per cm PD 120V, 898-900 MHz', '180511-005856 - FOSOF Acquisition 910 onoff (80 pct), CGX small flow rate - 0 config, 18 V per cm PD 120V', '180609-131341 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 898-900 MHz', '180610-115642 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 920-922 MHz', '180610-215802 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 920-922 MHz', '180611-183909 - FOSOF Acquisition 910 onoff (80 pct) - 0 config, 8 V per cm PD 120V, 898-900 MHz'])

# This data set has very low SNR. For some reason there was almost 100% quenching fraction of the metastabke atoms.
on_off_data_df = on_off_data_df.drop(['180321-225850 - FOSOF Acquisition 910 onoff - 0 config, 14 V per cm PD 120V'])

# pi-configuration data has opposite slope. It should have opposite frequency shift. To make the frequency shift of the same sign, the phase differences are multiplied by -1.
pi_config_index = on_off_data_df[on_off_data_df ['Configuration'] == 'pi'].index
on_off_data_df.loc[pi_config_index, 'Weighted Mean'] = on_off_data_df.loc[pi_config_index, 'Weighted Mean'] * (-1)

# Expanding the error bars for the data sets with larger than 1 reduced chi squared

chi_squared_large_index = on_off_data_df[on_off_data_df['Reduced Chi Squared'] > 1].index

on_off_data_df.loc[chi_squared_large_index, 'Weighted STD'] = on_off_data_df.loc[chi_squared_large_index, 'Weighted STD'] * np.sqrt(on_off_data_df.loc[chi_squared_large_index, 'Reduced Chi Squared'])

on_off_data_df = on_off_data_df.reset_index().set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])

# Join the averaged slopes. The pre-910 on/off switching data was acquired only for 49.86 kV accelerating voltage at 120 V proton deflector voltage with +-2MHz range around 910MHz.
on_off_data_df = on_off_data_df.join(slope_av_df.loc[(slice(None), 49.86, 120, slice(None)), slice(None)].reset_index().set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD'].loc[on_off_data_df.index.drop_duplicates()]['Normalized']['Slope [Rad/MHz]'])

# Calculate the frequency shift
on_off_data_df['Frequency Shift [MHz]'] = -on_off_data_df['Weighted Mean'] / on_off_data_df['Slope [Rad/MHz]'] * (1-on_off_data_df['F=0 Fraction Quenched'])

on_off_data_df['Frequency Shift STD [MHz]'] = -on_off_data_df['Weighted STD'] / on_off_data_df['Slope [Rad/MHz]'] * (1-on_off_data_df['F=0 Fraction Quenched'])

# Determine average frequency shift for each combination of the waveguide separation, accelerating voltage and the electric field amplitude.
on_off_data_grouped = on_off_data_df.groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])

on_off_av_df = on_off_data_grouped.apply(lambda df: straight_line_fit_params(df['Frequency Shift [MHz]'], df['Frequency Shift STD [MHz]']))

# Normalize the averaged data to reduced chi-squared of 1.

on_off_av_large_chi_squared_index = on_off_av_df[on_off_av_df['Reduced Chi Squared'] > 1].index
on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Weighted STD'] = on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Weighted STD'] * np.sqrt(on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Reduced Chi Squared'])

# Average shift in kHz
on_off_av_df = on_off_av_df[['Weighted Mean', 'Weighted STD']] * 1E3
on_off_av_df.rename(columns={'Weighted Mean': 'Frequency Shift [kHz]', 'Weighted STD': 'Frequency Shift STD [kHz]'}, inplace=True)

# We now want to scale the shifts by their amplitudes, as discussed in the beginning, to be able to estimate the average shift at one of the RF powers and waveguide separations, which can later be mapped to any other beam configuration by knowing the FOSOF amplitude ratios for these configurations.

# Obtaining the amplitude ratios:

# Unique experiment configuration parameters from all of the data
param_arr = zero_cross_av_df.reset_index(['Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]'], drop=True).index.drop_duplicates().values

# Hydrogen FOSOF simulations
fosof_sim_data = hydrogen_sim_data.FOSOFSimulation()
ampl_mean_df = pd.DataFrame()
freq_arr = np.linspace(908, 912, 41)

# Determine average FOSOF amplitude for each experiment configuration
for sim_param_tuple in param_arr:

    wvg_sep = sim_param_tuple[0]
    acc_val = sim_param_tuple[1]

    sim_params_dict = { 'Frequency Array [MHz]': freq_arr,
                        'Waveguide Separation [cm]': wvg_sep,
                        'Accelerating Voltage [kV]': acc_val,
                        'Off-axis Distance [mm]': beam_rms_rad_best_est}

    zero_crossing_vs_off_axis_dist_chosen_df, fosof_sim_info_chosen_df = fosof_sim_data.filter_fosof_sim_set(sim_params_dict, blind_freq_Q=False)

    ampl_mean_chosen_df = fosof_sim_data.fosof_sim_data_df.loc[fosof_sim_info_chosen_df.index][['Amplitude']].groupby('Simulation Key').aggregate('mean').join(fosof_sim_data.fosof_sim_info_df.loc[fosof_sim_info_chosen_df.index][['Waveguide Separation [cm]', 'Waveguide Electric Field [V/cm]']])

    ampl_mean_chosen_df['Accelerating Voltage [kV]'] = acc_val
    ampl_mean_chosen_df.set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'], inplace=True)

    ampl_mean_df = ampl_mean_df.append(ampl_mean_chosen_df)

ampl_mean_df = ampl_mean_df.loc[ampl_mean_df.index.intersection(zero_cross_av_df.reset_index(['Proton Deflector Voltage [V]'], drop=True).index.drop_duplicates())]

# Normalize the amplitude to their minimal value.
ampl_mean_df['Amplitude (Normalized)'] = ampl_mean_df['Amplitude'] / ampl_mean_df.min().values

# Change the type of the Waveguide Electric Field index level to a float from integers
on_off_av_df = on_off_av_df.reset_index('Waveguide Electric Field [V/cm]')
on_off_av_df['Waveguide Electric Field [V/cm]'] = on_off_av_df['Waveguide Electric Field [V/cm]'].astype(np.float64)
on_off_av_df.set_index('Waveguide Electric Field [V/cm]', append=True, inplace=True)

# Join with the normalized-amplitude-scaling factors
on_off_av_df = on_off_av_df.join(ampl_mean_df.loc[on_off_av_df.index])

# The larger the FOSOF amplitude the smaller the shift should be (in a proportional way)
# Calculate scaled frequency shifts
on_off_av_df['Scaled Frequency Shift [kHz]'] = on_off_av_df['Frequency Shift [kHz]'] * on_off_av_df['Amplitude (Normalized)']
on_off_av_df['Scaled Frequency Shift STD [kHz]'] = on_off_av_df['Frequency Shift STD [kHz]'] * on_off_av_df['Amplitude (Normalized)']

av_high_n_shift_max_s = straight_line_fit_params(on_off_av_df['Scaled Frequency Shift [kHz]'], on_off_av_df['Scaled Frequency Shift STD [kHz]'])

if av_high_n_shift_max_s['Reduced Chi Squared'] > 1:
    av_high_n_shift_max_s['Weighted STD'] = np.sqrt(av_high_n_shift_max_s['Reduced Chi Squared'])*(av_high_n_shift_max_s['Weighted STD'])

av_shift_high_n_df = ampl_mean_df.copy()

av_shift_high_n_df['Pre-910 On-Off Frequency Shift [kHz]'] = av_high_n_shift_max_s['Weighted Mean'] / av_shift_high_n_df['Amplitude (Normalized)']
av_shift_high_n_df['Pre-910 On-Off Frequency Shift STD [kHz]'] = av_high_n_shift_max_s['Weighted STD'] / av_shift_high_n_df['Amplitude (Normalized)']
#%%
''' Analysis of the high-pressure data. The statistical analysis is identical to that of the usual data sets. The data was acquired at 10 x the pressure in the Box. It is assumed that whatever the shift obsereved is supposed to be 10 times smaller at the usual pressure.
'''

fosof_lineshape_param_file_high_press_name = 'fosof_lineshape_high_press_param.csv'

os.chdir(saving_folder_location)

grouped_exp_fosof_lineshape_high_press_param_df = pd.read_csv(filepath_or_buffer=fosof_lineshape_param_file_high_press_name, delimiter=',', comment='#', header=[0, 1, 2, 3, 4], skip_blank_lines=True, index_col=[0, 1, 2, 3, 4, 5, 6, 7])
#%%
''' Statistical analysis of the lineshape data for high pressure
'''

# In case if a lineshape has reduced chi-squared of larger than 1 then we expand the uncertainty in the fit parameters to make the chi-squared to be one.

# Addidng the columns of normalized = corrected for large reduced chi-squared, uncertainties in the fit parameters.
fosof_lineshape_high_press_param_df = grouped_exp_fosof_lineshape_high_press_param_df.join(grouped_exp_fosof_lineshape_high_press_param_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Zero-crossing Frequency STD [MHz]', 'Slope STD [Rad/MHz]', 'Offset STD [MHz]'])].rename(columns={'Zero-crossing Frequency STD [MHz]': 'Zero-crossing Frequency STD (Normalized) [MHz]', 'Slope STD [Rad/MHz]': 'Slope STD (Normalized) [Rad/MHz]', 'Offset STD [MHz]': 'Offset STD (Normalized) [MHz]'})).sort_index(axis='columns')

# We now adjust the uncertainties for getting the reduced chi-squared of at least 1.

fosof_lineshape_high_press_param_norm_df = fosof_lineshape_high_press_param_df.groupby(level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns').apply(normalize_std)

# We want to use only the data for the frequency multiple range of 1 and 2.
fosof_lineshape_high_press_param_norm_df  = fosof_lineshape_high_press_param_norm_df.loc[(slice(None), slice(None), slice(None), slice(None), slice(None), slice(None), [1, 2]), slice(None)]
#%%
''' Determination of the average zero-crossing frequency using both of the RF Combiners, the RMS STD, and the uncertainty due to the difference in the frequencies obtained with the two combiners.
'''

rf_combiner_I_df = fosof_lineshape_high_press_param_norm_df['RF Combiner I Reference'].loc[slice(None), (slice(None), slice(None), slice(None), col_rf_comb_to_use)]

rf_combiner_R_df = fosof_lineshape_high_press_param_norm_df['RF Combiner R Reference'].loc[slice(None), (slice(None), slice(None), slice(None), col_rf_comb_to_use)]

fosof_lineshape_high_press_param_av_comb_df = rf_combiner_I_df.copy()

fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_val_to_use)] = (rf_combiner_I_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_val_to_use)]  + rf_combiner_R_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_val_to_use)] ) / 2

fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_unc_to_use)] = np.sqrt((rf_combiner_I_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_unc_to_use)]**2  + rf_combiner_R_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_unc_to_use)]**2 ) / 2)

# Adding in quadrature the difference in the parameters between the combiners.
rf_comb_diff_df = (rf_combiner_I_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_val_to_use)] - fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), col_data_val_to_use)])

fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset STD [MHz]')] = np.sqrt(fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset STD [MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset [MHz]')].values**2)

fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset STD (Normalized) [MHz]')] = np.sqrt(fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset STD (Normalized) [MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Offset [MHz]')].values**2)

fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope STD [Rad/MHz]')] = np.sqrt(fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope STD [Rad/MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')].values**2)

fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope STD (Normalized) [Rad/MHz]')] = np.sqrt(fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope STD (Normalized) [Rad/MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')].values**2)

fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD [MHz]')] = np.sqrt(fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD [MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')].values**2)

fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD (Normalized) [MHz]')] = np.sqrt(fosof_lineshape_high_press_param_av_comb_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD (Normalized) [MHz]')]**2 + rf_comb_diff_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')].values**2)
#%%
# Statistical averaging of the FOSOF data.

# Note that both the frequency range multiple of 1 and 2 are getting averaged together.

zero_cross_av_high_press_df = fosof_lineshape_high_press_param_av_comb_df.loc[('Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est), (slice(None))].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]']).apply(calc_av_zero_cross_freq).unstack(level=-1)

zero_cross_av_high_press_df.columns.names = ['Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

zero_cross_av_high_press_df.rename(columns={'Weighted Mean': 'Zero-crossing Frequency [MHz]', 'Weighted STD': 'Zero-crossing Frequency STD [MHz]', 'Weighted STD (Normalized)': 'Zero-crossing Frequency STD (Normalized) [MHz]'}, level='Data Field', inplace=True)

slope_av_high_press_df = fosof_lineshape_high_press_param_av_comb_df.loc[('Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est), (slice(None))].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]']).apply(calc_av_slope).unstack(level=-1)

slope_av_high_press_df.columns.names = ['Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

slope_av_high_press_df.rename(columns={'Weighted Mean': 'Slope [Rad/MHz]', 'Weighted STD': 'Slope STD [Rad/MHz]', 'Weighted STD (Normalized)': 'Slope STD (Normalized) [Rad/MHz]'}, level='Data Field', inplace=True)

# Determine difference between the zero-crossing frequencies determined from the high-pressure data and usual FOSOF data
zero_cross_high_press_diff_df = zero_cross_av_high_press_df['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized'] - zero_cross_av_df.loc[zero_cross_av_high_press_df.index]['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized']

zero_cross_high_press_diff_df['Zero-crossing Frequency STD (Normalized) [MHz]'] = np.sqrt(zero_cross_av_high_press_df['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized', 'Zero-crossing Frequency STD (Normalized) [MHz]']**2 + zero_cross_av_df['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized', 'Zero-crossing Frequency STD (Normalized) [MHz]']**2)

zero_cross_high_press_diff_df = zero_cross_high_press_diff_df[['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]']]

zero_cross_high_press_diff_df = zero_cross_high_press_diff_df * 1E3
zero_cross_high_press_diff_df.rename(columns={'Zero-crossing Frequency [MHz]': 'Frequency Shift [kHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]': 'Frequency Shift STD [kHz]'}, inplace=True)

zero_cross_high_press_diff_df.reset_index(['Proton Deflector Voltage [V]'], drop=True, inplace=True)

# The statistical analysis is finished. Time to determine the pressure-dependent shift
# Join with the normalized-amplitude-scaling factors
high_press_df = zero_cross_high_press_diff_df.join(ampl_mean_df.loc[zero_cross_high_press_diff_df.index])

# The larger the FOSOF amplitude the smaller the shift should be (in a proportional way)
# Calculate scaled frequency shifts
high_press_df['Scaled Frequency Shift [kHz]'] = high_press_df['Frequency Shift [kHz]'] * high_press_df['Amplitude (Normalized)']
high_press_df['Scaled Frequency Shift STD [kHz]'] = high_press_df['Frequency Shift STD [kHz]'] * high_press_df['Amplitude (Normalized)']

av_high_press_shift_max_s = straight_line_fit_params(high_press_df['Scaled Frequency Shift [kHz]'], high_press_df['Scaled Frequency Shift STD [kHz]'])

if av_high_press_shift_max_s['Reduced Chi Squared'] > 1:
    av_high_press_shift_max_s['Weighted STD'] = np.sqrt(av_high_press_shift_max_s['Reduced Chi Squared'])*(av_high_press_shift_max_s['Weighted STD'])

av_high_press_shift_df = ampl_mean_df.copy()

# Additional factor of 10 is due to pressure being 10 times larger.

press_mult = 10

av_high_press_shift_df['High-pressure Frequency Shift [kHz]'] = av_high_press_shift_max_s['Weighted Mean'] / av_high_press_shift_df['Amplitude (Normalized)'] / press_mult
av_high_press_shift_df['High-pressure Frequency Shift STD [kHz]'] = av_high_press_shift_max_s['Weighted STD'] / av_high_press_shift_df['Amplitude (Normalized)'] / press_mult

# Combine the pre-910 ON/OFF switching shifts with the pressure-dependent shifts
av_shift_high_n_df = av_shift_high_n_df.join(av_high_press_shift_df[['High-pressure Frequency Shift [kHz]','High-pressure Frequency Shift STD [kHz]']])
#%%
''' Comparison of the data with the Proton Deflector disabled to usual FOSOF data (PD ON).
'''
zero_cross_no_pd_df = zero_cross_av_no_pd_df.reset_index(['Proton Deflector Voltage [V]'], drop=True)

# Determine difference between the zero-crossing frequencies determined from the case when the proton deflector was OFF nad ON.
pd_on_off_diff_df = zero_cross_no_pd_df['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized']-zero_cross_av_df.reset_index(['Proton Deflector Voltage [V]'], drop=True).loc[zero_cross_no_pd_df.index]['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized']

pd_on_off_diff_df['Zero-crossing Frequency STD (Normalized) [MHz]'] = np.sqrt(zero_cross_no_pd_df['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized', 'Zero-crossing Frequency STD (Normalized) [MHz]']**2 + zero_cross_av_df.reset_index(['Proton Deflector Voltage [V]'], drop=True).loc[zero_cross_no_pd_df.index]['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized', 'Zero-crossing Frequency STD (Normalized) [MHz]']**2)

pd_on_off_diff_df = pd_on_off_diff_df[['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]']]

pd_on_off_diff_df = pd_on_off_diff_df * 1E3
pd_on_off_diff_df.rename(columns={'Zero-crossing Frequency [MHz]': 'Frequency Shift [kHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]': 'Frequency Shift STD [kHz]'}, inplace=True)

# # We select only the data with the frequency range multiple of 1 - since this is the data we are interested in for this experiment.
# pd_on_off_diff_df = pd_on_off_diff_df.loc[(slice(None), slice(None), slice(None), 1), slice(None)].reset_index(['Frequency Range Multiple'], drop=True)

# Join with the normalized-amplitude-scaling factors
pd_on_off_df = pd_on_off_diff_df.join(ampl_mean_df.loc[pd_on_off_diff_df.index])

# The larger the FOSOF amplitude the smaller the shift should be (in a proportional way)
# Calculate scaled frequency shifts
pd_on_off_df['Scaled Frequency Shift [kHz]'] = pd_on_off_df['Frequency Shift [kHz]'] * pd_on_off_df['Amplitude (Normalized)']
pd_on_off_df['Scaled Frequency Shift STD [kHz]'] = pd_on_off_df['Frequency Shift STD [kHz]'] * pd_on_off_df['Amplitude (Normalized)']

av_pd_on_off_shift_max_s = straight_line_fit_params(pd_on_off_df['Scaled Frequency Shift [kHz]'], pd_on_off_df['Scaled Frequency Shift STD [kHz]'])

if av_pd_on_off_shift_max_s['Reduced Chi Squared'] > 1:
    av_pd_on_off_shift_max_s['Weighted STD'] = np.sqrt(av_pd_on_off_shift_max_s['Reduced Chi Squared'])*(av_pd_on_off_shift_max_s['Weighted STD'])

av_pd_on_off_shift_df = ampl_mean_df.copy()

av_pd_on_off_shift_df['PD On-Off Frequency Shift [kHz]'] = av_pd_on_off_shift_max_s['Weighted Mean'] / av_pd_on_off_shift_df['Amplitude (Normalized)']
av_pd_on_off_shift_df['PD On-Off Frequency Shift STD [kHz]'] = av_pd_on_off_shift_max_s['Weighted STD'] / av_pd_on_off_shift_df['Amplitude (Normalized)']

# Combine the pre-910 ON/OFF switching shifts with the proton-deflector-dependent shifts
av_shift_high_n_df = av_shift_high_n_df.join(av_pd_on_off_shift_df[['PD On-Off Frequency Shift [kHz]','PD On-Off Frequency Shift STD [kHz]']])
#%%
''' Shift due to imperfect phase control upon waveguide reversal. This shift was measured to be about delta_phi = 0.2 mrad. In frequency units it corresponds to the frequency shift of delta_phi / slope. Instead of correcting the data, we add this as the additional type of the uncertainty.
'''

# Shift due to imperfect phase control [Rad]
delta_phi = 0.18 * 1E-3

phase_control_unc_df = np.abs(delta_phi / slope_av_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')]).rename(columns={'Slope [Rad/MHz]': 'Phase Control Uncertainty [MHz]'}).copy()
#%%
phase_control_unc_df
#%%
zero_cross_freq_with_unc_df = zero_cross_av_df.copy()
#%%
zero_cross_freq_with_unc_df
#%%
# We now want to determine the (blinded) resonant frequencies = zero-crossing frequencies corrected for the systematic shifts and also include all of the systematic uncertainties for each of the determined resonant frequencies.

# Adding the columns to the specific level that will store the systematic shifts
col_name_list = list(ac_shift_MHz_df.columns.union(sod_shift_df.columns))
col_name_list.append('Resonant Frequency (Blinded) [MHz]')
for col_name in col_name_list:

    zero_cross_freq_with_unc_df = zero_cross_freq_with_unc_df.join(zero_cross_av_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')].rename(columns={'Zero-crossing Frequency [MHz]': col_name}, level='Data Field')).sort_index(axis='columns')

# Addidng the uncertainty due to imperfect phase control
zero_cross_freq_with_unc_df = zero_cross_freq_with_unc_df.join(phase_control_unc_df).sort_index(axis='columns')
#%%
def correct_for_sys_shift(df):
    ''' Corrects the zero-crossing frequencies for the systematic shifts, and assign the respective systematic shift uncertainties.
    '''
    harm_type = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]
    std_state = df.columns.get_level_values('STD State')[0]

    df = df.reset_index(['Proton Deflector Voltage [V]'])

    df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] = df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] - ac_shift_MHz_df['AC Shift [MHz]']

    for col_name in ac_shift_MHz_df.columns:
        df[harm_type, av_type, std_type, std_state, col_name] = ac_shift_MHz_df[col_name]

    df = df.reset_index('Waveguide Electric Field [V/cm]')

    df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] = df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] + sod_shift_df['SOD Shift [MHz]']

    for col_name in sod_shift_df.columns:
        df[harm_type, av_type, std_type, std_state, col_name] = sod_shift_df[col_name]

    df.set_index(['Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]'], append=True, inplace=True)

    return df

# List of corrected frequencies (blinded) with the respective systematic and statistical uncertainties included.
res_freq_df = zero_cross_freq_with_unc_df.groupby(level=['Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State'], axis='columns').apply(correct_for_sys_shift)
#%%
res_freq_df
#%%
''' Calculation of the final average + uncertainty. Here one can select the type of analysis with which the data was averaged.
'''

# Select the averaging type to use for the final frequency determination
#av_type_to_use = 'Phasor Averaging Relative To DC'
av_type_to_use = 'Phasor Averaging'
#av_type_to_use = 'Phase Averaging'

# Whether to use the data sets, uncertainty of which was expanded for chi-squared > 1.
normalized_data_set_Q = True

if normalized_data_set_Q:
    norm_q_col_name = 'Normalized'
else:
    norm_q_col_name = 'Not Normalized'

# Whether to use the data normalized for the reduced chi-squared of > 1. This boolean controls the chi-squared expansion for the case when we average the data sets for the same experiment parameters.
normalized_sets_Q = True

if normalized_sets_Q:
    freq_std_col_name = 'Zero-crossing Frequency STD (Normalized) [MHz]'
    freq_std_col_rename = 'Zero-crossing Frequency STD (Normalized) [kHz]'

    slope_std_col_name = 'Slope STD (Normalized) [Rad/MHz]'
else:
    freq_std_col_name = 'Zero-crossing Frequency STD [MHz]'
    freq_std_col_rename = 'Zero-crossing Frequency STD [kHz]'
    slope_std_col_name = 'Slope STD [Rad/MHz]'

# Here we select the data needed.
df = res_freq_df.loc[(slice(None), slice(None), slice(None), slice(None)), (slice(None))]['First Harmonic', av_type_to_use, 'Phase RMS Repeat STD', norm_q_col_name].reset_index(['Proton Deflector Voltage [V]'], drop=True)

# The row with 19.0 V/cm is not needed
df.drop(index=(4.0, 49.86, 19.0), inplace=True)

df['Phase Control Shift [MHz]'] = 0

df = df[['AC Shift Uncertainty [MHz]', 'SOD Shift STD [MHz]', 'Phase Control Uncertainty [MHz]', freq_std_col_name, 'AC Shift [MHz]', 'SOD Shift [MHz]', 'Phase Control Shift [MHz]', 'Resonant Frequency (Blinded) [MHz]']]

# Converting all of the shifts in to kHz from MHz
df = df * 1E3

df.rename(columns={'AC Shift Uncertainty [MHz]': 'AC Shift Uncertainty [kHz]', 'SOD Shift STD [MHz]': 'SOD Shift STD [kHz]', 'Phase Control Uncertainty [MHz]': 'Phase Control Uncertainty [kHz]', freq_std_col_name: freq_std_col_rename, 'AC Shift [MHz]': 'AC Shift [kHz]', 'SOD Shift [MHz]': 'SOD Shift [kHz]', 'Phase Control Shift [MHz]': 'Phase Control Shift [kHz]', 'Resonant Frequency (Blinded) [MHz]': 'Resonant Frequency (Blinded) [kHz]'}, inplace=True)
#%%
# The weights are first determined for the case when there is no uncertainty due to the phase control. The reason is that before unblinding we did not have such an uncertainty. It is not fair to change the weights AFTER the unblinding was done.
data_to_use_2_df = df

def minimize_unc(w_arr, df):
    w_sum = np.sum(w_arr)

    field_power_shift_unc_tot = np.sum(df['AC Shift Uncertainty [kHz]'] * w_arr) / w_sum

    sod_unc_tot = np.sum(df['SOD Shift STD [kHz]'] * w_arr) / w_sum

    #phase_control_unc_tot = np.sum(df['Phase Control Uncertainty [kHz]'] * w_arr) / w_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((w_arr*df[freq_std_col_rename])**2) ) / w_sum

    #tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)
    tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2)

    return tot_unc

def w_tot_constraint(w_arr):
    return np.sum(w_arr) - 1

def find_unc_weights(df):
    w_arr = np.linspace(1, 10, df.shape[0])
    #w_arr = np.ones(df.shape[0])
    w_arr = w_arr/np.sum(w_arr)

    bounds_to_use_list = [(0, 1) for i in range(w_arr.shape[0])]

    w_min_arr = scipy.optimize.minimize(fun=minimize_unc, x0=w_arr, args=(df), bounds=bounds_to_use_list, constraints={'type': 'eq', 'fun': w_tot_constraint}, tol=1E-15)

    df['Weight'] = w_min_arr['x']

    return df

df_with_weights = df.groupby('Waveguide Separation [cm]').apply(find_unc_weights)

df_with_weights['Weight'] = df_with_weights['Weight'] / df_with_weights.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().shape[0]

weight_arr_sum = df_with_weights['Weight'].sum()

field_power_shift_unc_tot = np.sum(df_with_weights['AC Shift Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

sod_unc_tot = np.sum(df_with_weights['SOD Shift STD [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

#phase_control_unc_tot = np.sum(df_with_weights['Phase Control Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

zero_cross_stat_unc_tot = np.sqrt(np.sum((df_with_weights['Weight']*df_with_weights[freq_std_col_rename])**2) / weight_arr_sum**2)

#tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)

tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2)

freq_av = np.sum(df_with_weights['Resonant Frequency (Blinded) [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

AC_shift_av = np.sum(df_with_weights['AC Shift [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

SOD_shift_av = np.sum(df_with_weights['SOD Shift [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

#phase_control_shift_av = np.sum(df_with_weights['Phase Control Shift [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

f0 = freq_av + BLIND_OFFSET

f0_0_s = pd.Series({'Resonant Frequency [kHz]': f0, 'Statistical Uncertainty [kHz]': zero_cross_stat_unc_tot, 'Second-order Doppler Shift [kHz]': SOD_shift_av, 'Second-order Doppler Shift Uncertainty [kHz]': sod_unc_tot, 'AC Shift [kHz]': AC_shift_av, 'AC Shift Uncertainty [kHz]': field_power_shift_unc_tot, 'Total Uncertainty [kHz]': tot_unc})

#f0_1_s = pd.Series({'Resonant Frequency [kHz]': f0, 'Statistical Uncertainty [kHz]': zero_cross_stat_unc_tot, 'Phase Control Shift [kHz]': phase_control_shift_av, 'Phase Control Uncertainty [kHz]': phase_control_unc_tot, 'Second-order Doppler Shift [kHz]': SOD_shift_av, 'Second-order Doppler Shift Uncertainty [kHz]': sod_unc_tot, 'AC Shift [kHz]': AC_shift_av, 'AC Shift Uncertainty [kHz]': field_power_shift_unc_tot, 'Total Uncertainty [kHz]': tot_unc})

df_with_weights['Weight [%]'] = df_with_weights['Weight'] * 100
#%%
df_with_weights
#%%
f0_0_s
#%%
# Using the same weights, but adding the phase control shift + its associated uncertainty.

phase_control_shift_av = np.sum(df_with_weights['Phase Control Shift [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

phase_control_unc_tot = np.sum(df_with_weights['Phase Control Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)

f0_1_s = pd.Series({'Resonant Frequency [kHz]': f0, 'Statistical Uncertainty [kHz]': zero_cross_stat_unc_tot, 'Phase Control Shift [kHz]': phase_control_shift_av, 'Phase Control Uncertainty [kHz]': phase_control_unc_tot, 'Second-order Doppler Shift [kHz]': SOD_shift_av, 'Second-order Doppler Shift Uncertainty [kHz]': sod_unc_tot, 'AC Shift [kHz]': AC_shift_av, 'AC Shift Uncertainty [kHz]': field_power_shift_unc_tot, 'Total Uncertainty [kHz]': tot_unc})
#%%
f0_1_s['Second-order Doppler Shift [kHz]'] - f0_1_s['AC Shift [kHz]']
#%%
f0_1_s['Second-order Doppler Shift [kHz]'] - f0_1_s['AC Shift [kHz]'] - (52.6-31.4)
#%%
28.8-1.33
#%%
29.5-1.6
#%%
np.sqrt(2.3**2 + 1.0**2 + 1.4**2 + 1.5**2)
#%%
np.sqrt(2.25**2 + 0.9**2 + 1.33**2 + 1.5**2)
#%%
f0_1_s
#%%
52.2 - 31.7
#%%
51.6 - 29.5
#%%
f0_1_s['Resonant Frequency [kHz]'] - 909871.7
#%%
1.1
1.3
#%%
f0_1_s['Resonant Frequency [kHz]'] - 909871.7 - 0.533
#%%
# Phasor + 1 range
1.055
#%%
# Phase + 1 range
1.31
#%%
# Phasor + 2 range
1.74
#%%
# Phase + 2 range
1.992
#%%
'''
====================
Plot of the resonant frequency vs inverse slope. This is for the microwave-related phase shifts.
====================
'''
f0_to_use_s = f0_1_s

common_mode_phase_slope_df = data_to_use_2_df.join(slope_av_df.loc[(slice(None), slice(None), slice(None), slice(None)), slice(None)]['First Harmonic'][av_type_to_use]['Phase RMS Repeat STD']['Normalized'][['Slope [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]']])

#common_mode_phase_slope_df = common_mode_phase_slope_df.loc[(slice(None), slice(None), slice(None), slice(None), 1), slice(None)]

common_mode_phase_slope_df['Inverse Slope [kHz/mrad]'] = 1 / common_mode_phase_slope_df['Slope [Rad/MHz]']
common_mode_phase_slope_df['Inverse Slope STD [kHz/mrad]'] = common_mode_phase_slope_df['Slope STD (Normalized) [Rad/MHz]'] / common_mode_phase_slope_df['Slope [Rad/MHz]'] * common_mode_phase_slope_df['Inverse Slope [kHz/mrad]']

common_mode_phase_slope_df['Resonant Frequency [kHz]'] = common_mode_phase_slope_df['Resonant Frequency (Blinded) [kHz]'] + BLIND_OFFSET

common_mode_phase_slope_df.reset_index(['Proton Deflector Voltage [V]'], drop=True, inplace=True)

common_mode_phase_slope_df['Inverse Slope [kHz/mrad]'] = common_mode_phase_slope_df['Inverse Slope [kHz/mrad]'] * (-1)

common_mode_phase_slope_df['Total Uncertainty [kHz]'] = np.sqrt(common_mode_phase_slope_df['AC Shift Uncertainty [kHz]']**2 + common_mode_phase_slope_df['SOD Shift STD [kHz]']**2 + common_mode_phase_slope_df['Phase Control Uncertainty [kHz]']**2 + common_mode_phase_slope_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

#%%

def calc_fosof_lineshape_param(x_data_arr, y_data_arr, y_sigma_arr):
    ''' Fits the FOSOF data (phase vs frequency) to the first-order polynomial. Extracts the slope, offset, zero-crossing frequency, and the associated uncertainties. Gives the reduced chi-squared parameter.
    '''
    w_arr = 1/y_sigma_arr**2

    delta_arr = np.sum(w_arr) * np.sum(w_arr*x_data_arr**2) - (np.sum(w_arr*x_data_arr))**2

    offset = (np.sum(w_arr*y_data_arr) * np.sum(w_arr*x_data_arr**2) - np.sum(w_arr*x_data_arr*y_data_arr) * np.sum(w_arr*x_data_arr)) / delta_arr

    offset_sigma = np.sqrt(np.sum(w_arr*x_data_arr**2) / delta_arr)

    slope = (np.sum(w_arr*x_data_arr*y_data_arr) * np.sum(w_arr) - np.sum(w_arr*x_data_arr) * np.sum(w_arr*y_data_arr)) / delta_arr

    slope_sigma = np.sqrt(np.sum(w_arr) / delta_arr)

    # zero-crossing frequency.
    x0 = -offset/slope

    delta2_arr = x_data_arr.shape[0] * np.sum(x_data_arr*y_data_arr) - np.sum(x_data_arr) * np.sum(y_data_arr)

    # This formula was derived analytically. It gives the answer that is, as expected, independent of the bias added to the frequencies. If one does not use this formula, but assumes that there is no correlation between the offset and the slope, then one gets wrong uncertainty - it depends on the amount of the bias added to the frequency.
    sigma_x0 = np.sqrt(np.sum(((np.sum(x_data_arr) * x_data_arr - np.sum(x_data_arr**2)) / (delta2_arr) + (np.sum(x_data_arr**2) * np.sum(y_data_arr) - np.sum(x_data_arr) * np.sum(x_data_arr * y_data_arr)) * (x_data_arr.shape[0] * x_data_arr - np.sum(x_data_arr)) / (delta2_arr**2))**2 * y_sigma_arr**2))

    fit_param_dict = {'Slope [Rad/MHz]': slope, 'Slope STD [Rad/MHz]': slope_sigma, 'Offset [MHz]': offset, 'Offset STD [MHz]': offset_sigma, 'Zero-crossing Frequency [MHz]': x0, 'Zero-crossing Frequency STD [MHz]': sigma_x0}

    # For the chi-squared determination.
    fit_data_arr = slope * x_data_arr + offset
    n_constraints = 2
    fit_param_dict = {**fit_param_dict, **get_chi_squared(y_data_arr, y_sigma_arr, fit_data_arr, n_constraints)}

    return fit_param_dict

#%%
# Plotting
wvg_e_field_list = list(common_mode_phase_slope_df.index.get_level_values('Waveguide Electric Field [V/cm]').drop_duplicates().values)

wvg_sep_list = list(common_mode_phase_slope_df.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().values)

acc_volt_list = list(common_mode_phase_slope_df.index.get_level_values('Accelerating Voltage [kV]').drop_duplicates().values)

plot_marker_list = ['^', 'o', 'D', 's', 'o']
plot_marker_dict = dict(list(zip(*[wvg_sep_list, plot_marker_list])))

plot_marker_fill_style_list = ['none', 'top', 'full']
plot_marker_fill_style_dict = dict(list(zip(*[acc_volt_list, plot_marker_fill_style_list])))

plot_marker_edge_color_list = ['brown', 'red', 'purple', 'green', 'blue']
plot_marker_edge_color_dict = dict(list(zip(*[wvg_e_field_list, plot_marker_edge_color_list])))

common_mode_phase_slope_df['Plot Marker Type'] = None
common_mode_phase_slope_df['Plot Marker Fill Style'] = None
common_mode_phase_slope_df['Plot Marker Edge Color'] = None

for wvg_e_field in wvg_e_field_list:
    common_mode_phase_slope_df.loc[(slice(None), slice(None), wvg_e_field), ('Plot Marker Edge Color')] = plot_marker_edge_color_dict[wvg_e_field]

for wvg_sep in wvg_sep_list:
    common_mode_phase_slope_df.loc[(wvg_sep), ('Plot Marker Type')] = plot_marker_dict[wvg_sep]

for acc_volt in acc_volt_list:
    common_mode_phase_slope_df.loc[(slice(None), acc_volt, slice(None)), ('Plot Marker Fill Style')] = plot_marker_fill_style_dict[acc_volt]
#%%
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

for marker in list(plot_marker_dict.values()):

    data_marker_df = common_mode_phase_slope_df[common_mode_phase_slope_df['Plot Marker Type'] == marker]

    for color in list(plot_marker_edge_color_dict.values()):
        data_color_df = data_marker_df[data_marker_df['Plot Marker Edge Color'] == color]
        if data_color_df.shape[0] > 0:

            for fill_style in list(plot_marker_fill_style_dict.values()):

                data_fill_style_df = data_color_df[data_color_df['Plot Marker Fill Style'] == fill_style]

                if data_fill_style_df.shape[0] > 0:
                    x_data_arr = data_fill_style_df['Inverse Slope [kHz/mrad]']
                    x_data_err_arr = data_fill_style_df['Inverse Slope STD [kHz/mrad]']
                    y_data_arr = data_fill_style_df['Resonant Frequency [kHz]']
                    y_data_err_arr = data_fill_style_df['Total Uncertainty [kHz]']

                    ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, fillstyle=fill_style, capsize=5, capthick=2, linestyle='', markersize=13, markerfacecoloralt='cyan')

                # if face_color != 'full':
                #     ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, markerfacecolor='none', capsize=5, capthick=2, linestyle='', markersize=10)
                # else:
                #     ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, capsize=5, capthick=2, linestyle='', markersize=10)
        #data_marker_df.plot(kind='scatter', x='Inverse Slope [kHz/mrad]', xerr='Inverse Slope STD [kHz/mrad]', y='Resonant Frequency [kHz]', yerr='Total Uncertainty [kHz]', ax=ax, marker=marker, color=data_marker_df['Plot Color'].values)

x_data_arr = common_mode_phase_slope_df['Inverse Slope [kHz/mrad]']
y_data_arr = common_mode_phase_slope_df['Resonant Frequency [kHz]']
y_sigma_arr = common_mode_phase_slope_df['Total Uncertainty [kHz]']

fit_dict = calc_fosof_lineshape_param(x_data_arr, y_data_arr, y_sigma_arr)

fit_dict['Slope [Rad/MHz]'] * x_data_arr + fit_dict['Offset [MHz]']

ax.set_xlim((0, ax.get_xlim()[1]))
ax_x_lim = ax.get_xlim()

x_plot_data = np.array([0, np.max(x_data_arr)+10])

ax.plot(x_plot_data, fit_dict['Slope [Rad/MHz]'] * x_plot_data + fit_dict['Offset [MHz]'], color='black')

ax.plot(x_plot_data, (fit_dict['Slope [Rad/MHz]']+fit_dict['Slope STD [Rad/MHz]']) * x_plot_data + fit_dict['Offset [MHz]']-fit_dict['Offset STD [MHz]'], color='black')

ax.plot(x_plot_data, (fit_dict['Slope [Rad/MHz]']-fit_dict['Slope STD [Rad/MHz]']) * x_plot_data + fit_dict['Offset [MHz]']+fit_dict['Offset STD [MHz]'], color='black')

plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

rect_f0 = Rectangle((x_plot_data[0], f0_1_s['Resonant Frequency [kHz]']-f0_1_s['Total Uncertainty [kHz]']), x_plot_data[1]-x_plot_data[0]+10, 2*f0_1_s['Total Uncertainty [kHz]'], color='wheat', fill=True, alpha=1)
ax.add_patch(rect_f0)

rect_f_slope = Rectangle((x_plot_data[0], fit_dict['Offset [MHz]']-fit_dict['Offset STD [MHz]']), 1-x_plot_data[0], 2*fit_dict['Offset STD [MHz]'], color='gray', fill=True, alpha=0.5)
ax.add_patch(rect_f_slope)

ax.set_xlim(ax_x_lim)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax.set_xlabel(r'$1/S$ (kHz/mrad)')
ax.set_ylabel(r'$f_0$ (kHz)')

fig.tight_layout()

#os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\FOSOF_ph_cntrl')

#plt_name = 'inverse_slope.pdf'
#plt.savefig(plt_name)
plt.show()
#%%
fit_dict
#%%
f0_1_s
#%%
'''
 ===================================
 Comparison between the experiment slopes and the simulation slopes
 ===================================
'''
slope_comp_df = fosof_lineshape_vs_offset_set_df.loc[(slice(None), slice(None), slice(None), beam_rms_rad_best_est, fract_offset_best_est), (['Slope [Rad/MHz]'])].reset_index(['Beam RMS Radius [mm]', 'Fractional Offset'], drop=True).rename(columns={'Slope [Rad/MHz]': 'Simulation Slope [Rad/MHz]'}).join(slope_av_df.loc[(slice(None), slice(None), slice(None), slice(None)), (slice(None))]['First Harmonic', av_type_to_use, 'Phase RMS Repeat STD', 'Normalized'].reset_index(['Proton Deflector Voltage [V]'], drop=True)[['Slope [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]']])

slope_comp_df['Simulation Slope [Rad/MHz]'] = -1 * slope_comp_df['Simulation Slope [Rad/MHz]']
slope_comp_df['Slope [Rad/MHz]'] = -1 * slope_comp_df['Slope [Rad/MHz]']

slope_comp_df['Simulation Slope STD [Rad/MHz]'] = simul_unc * np.abs(slope_comp_df['Simulation Slope [Rad/MHz]'])

slope_comp_df['Slope Fractional Deviation'] = (slope_comp_df['Slope [Rad/MHz]'] - slope_comp_df['Simulation Slope [Rad/MHz]'])/slope_comp_df['Simulation Slope [Rad/MHz]']

slope_comp_df['Slope Fractional Deviation STD'] = np.sqrt(slope_comp_df['Slope STD (Normalized) [Rad/MHz]']**2 + slope_comp_df['Simulation Slope STD [Rad/MHz]']**2) / np.abs(slope_comp_df['Simulation Slope [Rad/MHz]'])


slope_comp_df
#%%
'''
==========================
910 On-Off data with changing Mass Flow Rate setting
==========================
'''
# We first look at the pre-quench 910 cavity On/Off switching data.
os.chdir(fosof_analyzed_data_folder_path)

on_off_data_df = pd.read_csv('910_on_off_m_flow_fosof_data.csv', index_col=[0])

# pi-configuration data has opposite slope. It should have opposite frequency shift. To make the frequency shift of the same sign, the phase differences are multiplied by -1.
pi_config_index = on_off_data_df[on_off_data_df ['Configuration'] == 'pi'].index
on_off_data_df.loc[pi_config_index, 'Weighted Mean'] = on_off_data_df.loc[pi_config_index, 'Weighted Mean'] * (-1)

# Expanding the error bars for the data sets with larger than 1 reduced chi squared

chi_squared_large_index = on_off_data_df[on_off_data_df['Reduced Chi Squared'] > 1].index

on_off_data_df.loc[chi_squared_large_index, 'Weighted STD'] = on_off_data_df.loc[chi_squared_large_index, 'Weighted STD'] * np.sqrt(on_off_data_df.loc[chi_squared_large_index, 'Reduced Chi Squared'])

on_off_data_df = on_off_data_df.reset_index().set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])

# Join the averaged slopes. The pre-910 on/off switching data was acquired only for 49.86 kV accelerating voltage at 120 V proton deflector voltage with +-2MHz range around 910MHz.
on_off_data_df = on_off_data_df.join(slope_df.loc[(slice(None), 49.86, 120, slice(None), 1), slice(None)].reset_index().set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD'].loc[on_off_data_df.index.drop_duplicates()]['Normalized']['Slope [Rad/MHz]'])

# Calculate the frequency shift
on_off_data_df['Frequency Shift [MHz]'] = -on_off_data_df['Weighted Mean'] / on_off_data_df['Slope [Rad/MHz]'] * (1-on_off_data_df['F=0 Fraction Quenched'])

on_off_data_df['Frequency Shift STD [MHz]'] = -on_off_data_df['Weighted STD'] / on_off_data_df['Slope [Rad/MHz]'] * (1-on_off_data_df['F=0 Fraction Quenched'])

# Average pressure in the charge exchange for each mass flow setting.
on_off_data_cgx_press_df = on_off_data_df.set_index('Mass Flow Rate [sccm]', append=True)[['Charge Exchange Pressure [Torr]']].groupby(['Mass Flow Rate [sccm]']).aggregate(lambda x: np.mean(x)*1E6).rename(columns={'Charge Exchange Pressure [Torr]': 'Charge Exchange Pressure [uTorr]'})

# Determine average frequency shift for each combination of the waveguide separation, accelerating voltage and the electric field amplitude.
on_off_data_grouped = on_off_data_df.set_index('Mass Flow Rate [sccm]', append=True).groupby(['Mass Flow Rate [sccm]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])

on_off_av_df = on_off_data_grouped.apply(lambda df: straight_line_fit_params(df['Frequency Shift [MHz]'], df['Frequency Shift STD [MHz]']))

# Normalize the averaged data to reduced chi-squared of 1.

on_off_av_large_chi_squared_index = on_off_av_df[on_off_av_df['Reduced Chi Squared'] > 1].index
on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Weighted STD'] = on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Weighted STD'] * np.sqrt(on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Reduced Chi Squared'])

# Average shift in kHz
on_off_av_df = on_off_av_df[['Weighted Mean', 'Weighted STD']] * 1E3
on_off_av_df.rename(columns={'Weighted Mean': 'Frequency Shift [kHz]', 'Weighted STD': 'Frequency Shift STD [kHz]'}, inplace=True)

on_off_cgx_flow_rate_df = on_off_av_df.reset_index().set_index('Mass Flow Rate [sccm]').join(on_off_data_cgx_press_df).set_index(['Waveguide Separation [cm]', 'Charge Exchange Pressure [uTorr]']).drop(columns=['Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).sort_index()

on_off_cgx_flow_rate_df
#%%
#%%
'''
=========================
 For thesis. Various tables
=========================
'''

def for_latex_unc(num_arr, unc_arr, unc_only_Q=False, u_format_Q=False):
    ''' Convert number + uncertainty into a string in a proper format for latex.
    '''
    num_str_arr = np.zeros(num_arr.shape)
    num_str_arr = num_str_arr.astype(np.str)

    for i in range(num_arr.shape[0]):
        x = ufloat(num_arr[i], unc_arr[i])
        if u_format_Q:
            unc_str = '{:L}'.format(x)
        else:
            unc_str = '{:.1fL}'.format(x)
        if not unc_only_Q:
            num_str_arr[i] = unc_str
        else:
            unc_str = '{:.1fL}'.format(x)
            # Regular expression
            reg_exp = re.compile( r"""(?P<num>[\d\.\s]+)  # number
             \\\\pm (?P<unc>[\s\d\.]+)      # uncertainty
             """, # Digitizer channel (positive integer)
                            re.VERBOSE)
            num_unc_object = reg_exp.search("%r"%unc_str
            )

            num_str_arr[i] = num_unc_object.group('unc').strip()

    return num_str_arr
#%%
'''Export the table of high-n shifts to thesis
'''
# Thesis folder
os.chdir(these_high_n_folder_path)

av_shift_high_n_s_df = av_shift_high_n_df.copy().sort_index()

num_arr = av_shift_high_n_s_df['Pre-910 On-Off Frequency Shift [kHz]'].values
unc_arr = av_shift_high_n_s_df['Pre-910 On-Off Frequency Shift STD [kHz]'].values
av_shift_high_n_s_df['Pre-910 On-Off Frequency Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)

num_arr = av_shift_high_n_s_df['High-pressure Frequency Shift [kHz]'].values
unc_arr = av_shift_high_n_s_df['High-pressure Frequency Shift STD [kHz]'].values
av_shift_high_n_s_df['High-pressure Frequency Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)

num_arr = av_shift_high_n_s_df['PD On-Off Frequency Shift [kHz]'].values
unc_arr = av_shift_high_n_s_df['PD On-Off Frequency Shift STD [kHz]'].values
av_shift_high_n_s_df['PD On-Off Frequency Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)

av_shift_high_n_s_df.drop(columns=['Amplitude', 'Pre-910 On-Off Frequency Shift STD [kHz]', 'High-pressure Frequency Shift STD [kHz]', 'PD On-Off Frequency Shift STD [kHz]'], inplace=True)

av_shift_high_n_s_df.reset_index(inplace=True)

av_shift_high_n_s_df['Waveguide Separation [cm]'] = av_shift_high_n_s_df['Waveguide Separation [cm]'].astype(np.int16)
av_shift_high_n_s_df['Waveguide Electric Field [V/cm]'] = av_shift_high_n_s_df['Waveguide Electric Field [V/cm]'].astype(np.int16)

#av_shift_high_n_s_df.rename(columns={'Waveguide Separation [cm]': r'$D$ \si{(cm)}', 'Accelerating Voltage [kV]': r'$V_{\mathrm{HV}}$ \si{(\kilo\volt)}', 'Waveguide Electric Field [V/cm]': r'$E_0$ \si{(\volt/\centi\meter)}'}, inplace=True)

# av_shift_high_n_s_df.set_index([r'$D$ \si{(cm)}', r'$V_{\mathrm{HV}}$ \si{(\kilo\volt)}', r'$E_0$ \si{(\volt/\centi\meter)}'], inplace=True)

av_shift_high_n_s_df['Amplitude (Normalized)'] = av_shift_high_n_s_df['Amplitude (Normalized)'].transform(lambda x: list(map(lambda y: '{:.1f}'.format(y), x)))
#%%
av_shift_high_n_s_df
#%%
av_shift_high_n_s_df.to_latex(buf='high_n_shift.txt', column_format='lllSSSS', multirow=True, escape=False, index=False, index_names=True, header=[r'$D$ \si{(cm)}', r'$V_{\mathrm{HV}}$ \si{(\kilo\volt)}', r'$E_0$ \si{(\volt/\centi\meter)}', r'$A$', r'$\Delta_{n>2}^{(910)}$ \si{(\kilo\hertz)}', r'$\Delta_{n>2}^{(P)}$ \si{(\kilo\hertz)}', r'$\Delta_{n>2}^{(PD)}$ \si{(\kilo\hertz)}'])

#%%
''' AC Shift table
'''
# Thesis folder
os.chdir(these_ac_folder_path)

ac_shift_thesis_df = ac_shift_df.copy()
ac_shift_thesis_df = ac_shift_thesis_df[['Field Power Shift [kHz]', 'Beam RMS Radius Frequency Shift [kHz]', 'Beam RMS Radius Frequency Shift Uncertainty [kHz]', 'Fractional Offset Frequency Shift [kHz]', 'Fractional Offset Frequency Shift Uncertainty [kHz]', 'AC Shift [kHz]', 'AC Shift Uncertainty [kHz]']]
#%%
# The row with 19.0 V/cm is not needed
ac_shift_thesis_df.drop(index=(4, 49.86, 19.0), inplace=True)
#%%
num_arr = ac_shift_thesis_df['Field Power Shift [kHz]'].values
unc_arr = ac_shift_thesis_df['Field Power Shift [kHz]'].values * field_power_shift_unc

ac_shift_thesis_df['Field Power Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)

num_arr = ac_shift_thesis_df['Beam RMS Radius Frequency Shift [kHz]'].values
unc_arr = ac_shift_thesis_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'].values
ac_shift_thesis_df['Beam RMS Radius Frequency Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)

num_arr = ac_shift_thesis_df['Fractional Offset Frequency Shift [kHz]'].values
unc_arr = ac_shift_thesis_df['Fractional Offset Frequency Shift Uncertainty [kHz]'].values
ac_shift_thesis_df['Fractional Offset Frequency Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)

num_arr = ac_shift_thesis_df['AC Shift [kHz]'].values
unc_arr = ac_shift_thesis_df['AC Shift Uncertainty [kHz]'].values
ac_shift_thesis_df['AC Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)
#%%
ac_shift_thesis_df = ac_shift_thesis_df[['Field Power Shift [kHz]', 'Beam RMS Radius Frequency Shift [kHz]', 'Fractional Offset Frequency Shift [kHz]', 'AC Shift [kHz]']]

ac_shift_thesis_df.reset_index(inplace=True)

ac_shift_thesis_df['Waveguide Separation [cm]'] = ac_shift_thesis_df['Waveguide Separation [cm]'].astype(np.int16)
ac_shift_thesis_df['Waveguide Electric Field [V/cm]'] = ac_shift_thesis_df['Waveguide Electric Field [V/cm]'].astype(np.int16)
#%%
ac_shift_thesis_df.to_latex(buf='ac_shift.txt', column_format='lllcccc', multirow=True, escape=False, index=False, index_names=True, header=[r'$D$ \si{(cm)}', r'$V_{\mathrm{HV}}$ \si{(\kilo\volt)}', r'$E_0$ \si{(\volt/\centi\meter)}', r'$\Delta_{\mathrm{AC}}^{(E_0)}$ \si{(\kilo\hertz)}', r'$\Delta_{\mathrm{AC}}^{(r)}$ \si{(\kilo\hertz)}', r'$\Delta_{\mathrm{AC}}^{\mathrm{(offset)}}$ \si{(\kilo\hertz)}', r'$\Delta_{\mathrm{AC}}$ \si{(\kilo\hertz)}'])
#%%
ac_shift_thesis_df
#%%
''' Imperfect phase control shifts table
'''

phase_control_thesis_df = slope_av_df.loc[(slice(None), slice(None), slice(None), slice(None)), (slice(None))]['First Harmonic', av_type_to_use, 'Phase RMS Repeat STD', 'Normalized'].reset_index(['Proton Deflector Voltage [V]'], drop=True)[['Slope [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]']].join(data_to_use_2_df[['Phase Control Shift [kHz]', 'Phase Control Uncertainty [kHz]']])

phase_control_thesis_df.drop(index=(4.0, 49.86, 19.0), inplace=True)

num_arr = np.abs(phase_control_thesis_df['Slope [Rad/MHz]'].values)
unc_arr = phase_control_thesis_df['Slope STD (Normalized) [Rad/MHz]'].values
phase_control_thesis_df['Slope [mrad/kHz]'] = for_latex_unc(num_arr, unc_arr, u_format_Q=True)

num_arr = phase_control_thesis_df['Phase Control Shift [kHz]'].values
unc_arr = phase_control_thesis_df['Phase Control Uncertainty [kHz]'].values
phase_control_thesis_df['Imperfect Phase Control Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)

phase_control_thesis_df.drop(columns=['Slope [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]', 'Phase Control Shift [kHz]', 'Phase Control Uncertainty [kHz]'], inplace=True)

phase_control_thesis_df.reset_index(inplace=True)

phase_control_thesis_df['Waveguide Separation [cm]'] = phase_control_thesis_df['Waveguide Separation [cm]'].astype(np.int16)
phase_control_thesis_df['Waveguide Electric Field [V/cm]'] = phase_control_thesis_df['Waveguide Electric Field [V/cm]'].astype(np.int16)
#%%
os.chdir(these_phase_control_folder_path)

phase_control_thesis_df.to_latex(buf='phase_control.txt', column_format='lllSS', multirow=True, escape=False, index=False, index_names=True, header=[r'$D$\,\si{(cm)}', r'$V_{\mathrm{HV}}$\,\si{(\kilo\volt)}', r'$E_0$\,\si{(\volt/\centi\meter)}', r'$S$\,\si{(\milli\radian/\kilo\hertz)}', r'$\Delta_{c}^{(\mathrm{rf})}$\,\si{(\kilo\hertz)}'])

#%%

''' Table of the 910 on-off data for different Mass flow rate settings
'''

on_off_cgx_flow_rate_thesis_df = on_off_cgx_flow_rate_df.reset_index()

on_off_cgx_flow_rate_thesis_df['Charge Exchange Pressure [uTorr]'] = on_off_cgx_flow_rate_thesis_df['Charge Exchange Pressure [uTorr]'].transform(lambda x: list(map(lambda y: '{:.1f}'.format(y), x)))

on_off_cgx_flow_rate_thesis_df['Waveguide Separation [cm]'] = on_off_cgx_flow_rate_thesis_df['Waveguide Separation [cm]'].astype(np.int16)

num_arr = on_off_cgx_flow_rate_thesis_df['Frequency Shift [kHz]'].values
unc_arr = on_off_cgx_flow_rate_thesis_df['Frequency Shift STD [kHz]'].values
on_off_cgx_flow_rate_thesis_df['Frequency Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)

on_off_cgx_flow_rate_thesis_df = on_off_cgx_flow_rate_thesis_df.drop(columns=['Frequency Shift STD [kHz]'])

#%%

os.chdir(these_high_n_folder_path)

on_off_cgx_flow_rate_thesis_df.to_latex(buf='910_on_off_cgx_press_table.txt', column_format='llS', multirow=True, escape=False, index=False, index_names=True, header=[r'$D$\,\si{(cm)}', r'$P_\mathrm{CGX}\.\mu\mathrm{Torr}$', r'$\Delta_{n>2}^{(P_{\mathrm{CGX}})}$\,\si{(\kilo\hertz)}'])
#%%
on_off_cgx_flow_rate_thesis_df

#%%
''' Table for the comparison of the slopes from the FOSOF experiments and simulations
'''
slope_comp_thesis_df = slope_comp_df.reset_index()

slope_comp_thesis_df['Waveguide Separation [cm]'] = slope_comp_thesis_df['Waveguide Separation [cm]'].astype(np.int16)
slope_comp_thesis_df['Waveguide Electric Field [V/cm]'] = slope_comp_thesis_df['Waveguide Electric Field [V/cm]'].astype(np.int16)

num_arr = slope_comp_thesis_df['Simulation Slope [Rad/MHz]'].values
unc_arr = slope_comp_thesis_df['Simulation Slope STD [Rad/MHz]'].values
slope_comp_thesis_df['Simulation Slope [mrad/kHz]'] = for_latex_unc(num_arr, unc_arr, u_format_Q=True)

num_arr = slope_comp_thesis_df['Slope [Rad/MHz]'].values
unc_arr = slope_comp_thesis_df['Slope STD (Normalized) [Rad/MHz]'].values
slope_comp_thesis_df['Slope [mrad/kHz]'] = for_latex_unc(num_arr, unc_arr, u_format_Q=True)

num_arr = slope_comp_thesis_df['Slope Fractional Deviation'].values
unc_arr = slope_comp_thesis_df['Slope Fractional Deviation STD'].values
slope_comp_thesis_df['Slope Fractional Deviation'] = for_latex_unc(num_arr, unc_arr, u_format_Q=True)

slope_comp_thesis_df = slope_comp_thesis_df.drop(columns=['Simulation Slope [Rad/MHz]', 'Simulation Slope STD [Rad/MHz]', 'Slope [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]', 'Slope Fractional Deviation STD'])

slope_comp_thesis_df = slope_comp_thesis_df[['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Slope [mrad/kHz]', 'Simulation Slope [mrad/kHz]', 'Slope Fractional Deviation']]
#%%
slope_comp_thesis_df
#%%
os.chdir(these_beam_speed_folder_path)

slope_comp_thesis_df.to_latex(buf='slope_comp_table.txt', column_format='lllccc', multirow=True, escape=False, index=False, index_names=True, header=[r'$D$\,\si{(cm)}', r'$V_{\mathrm{HV}}$\,\si{(\kilo\volt)}', r'$E_0$\,\si{(\volt/\centi\meter)}', r'$S$\,\si{(\milli\radian/\kilo\hertz)}', r'$S_\mathrm{sim}$\,\si{(\milli\radian/\kilo\hertz)}', r'$\Delta S/S_{\mathrm{sim}}$'])
#%%
fosof_lineshape_param_av_comb_df
#%%
''' Table for the zero-crossing frequencies used to find the resonant frequency
'''
data_set_data_thesis_df = fosof_lineshape_param_av_comb_df.loc['Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est_to_use]['First Harmonic', av_type_to_use, 'Phase RMS Repeat STD'].reset_index(['Proton Deflector Voltage [V]', 'Frequency Range Multiple', 'Group ID'], drop=True)[['Zero-crossing Frequency [MHz]', freq_std_col_name, 'Slope [Rad/MHz]', slope_std_col_name]].reset_index()

#data_set_data_thesis_df['Zero-crossing Frequency [kHz]'] = data_set_data_thesis_df['Zero-crossing Frequency [MHz]']
#data_set_data_thesis_df['Zero-crossing Frequency STD [kHz]'] = data_set_data_thesis_df[freq_std_col_name]

# Turning negative slopes into positive slopes
data_set_data_thesis_df['Slope [Rad/MHz]'] = data_set_data_thesis_df['Slope [Rad/MHz]'] * (-1)

data_set_data_thesis_df['Waveguide Separation [cm]'] = data_set_data_thesis_df['Waveguide Separation [cm]'].astype(np.int16)
data_set_data_thesis_df['Waveguide Electric Field [V/cm]'] = data_set_data_thesis_df['Waveguide Electric Field [V/cm]'].astype(np.int16)

num_arr = data_set_data_thesis_df['Zero-crossing Frequency [MHz]'].values
unc_arr = data_set_data_thesis_df[freq_std_col_name].values
data_set_data_thesis_df['Zero-crossing Frequency [MHz]'] = for_latex_unc(num_arr, unc_arr)

num_arr = data_set_data_thesis_df['Slope [Rad/MHz]'].values
unc_arr = data_set_data_thesis_df[slope_std_col_name].values
data_set_data_thesis_df['Slope [Rad/MHz]'] = for_latex_unc(num_arr, unc_arr)

data_set_data_thesis_df = data_set_data_thesis_df[['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Zero-crossing Frequency [MHz]', 'Slope [Rad/MHz]']]

#%%
os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Thesis\exp_acq')

data_set_data_thesis_df.to_latex(buf='zero_cross_data_sets.txt', column_format='lllSS', multirow=True, escape=False, index=False, index_names=True, longtable=True, header=[r'$D$\,\si{(cm)}', r'$V_{\mathrm{HV}}$\,\si{(\kilo\volt)}', r'$E_0$\,\si{(\volt/\centi\meter)}', r'$f_{zc}$\,\si{(\mega\hertz)}', r'$S$\,\si{(\radian/\mega\hertz)}'])
#%%
'''
=============================
Extracting # of data sets (0 + pi pairs) used to extract the resonant frequencies. Primarily needed for the thesis.
=============================
'''
fosof_lineshape_param_av_comb_df.loc['Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est_to_use].shape[0]

# index_for_data = fosof_lineshape_param_av_comb_df.index.difference(fosof_lineshape_param_av_comb_df.loc[('Waveguide Carrier Frequency Sweep', slice(None), slice(None), slice(None), 0), slice(None)].index)
#
# group_id_used_arr = index_for_data.get_level_values('Group ID').drop_duplicates().values
#
# fosof_grouped_phase_file_name = 'fosof_phase_grouped_list.csv'

#os.chdir(saving_folder_location)

#grouped_phase = pd.read_csv(filepath_or_buffer=fosof_grouped_phase_file_name, delimiter=',', comment='#', skip_blank_lines=True, header=[0, 1, 2, 3, 4], index_col=[0, 1, 2, 3])

#experiment_id_size = grouped_phase.loc[group_id_used_arr].index.get_level_values('Experiment ID').drop_duplicates().values.shape[0]

#experiment_id_size
#%%
#%%

#%%
# Number of experiment parameters
exp_param_num = fosof_lineshape_param_av_comb_df.loc[index_for_data].loc['Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est].reset_index(['Proton Deflector Voltage [V]', 'Frequency Range Multiple', 'Group ID'], drop=True).drop(index=(4, 49.86, 19)).index.drop_duplicates().values.shape[0]
exp_param_num
#%%


fosof_lineshape_param_df
#%%
df_to_use = fosof_lineshape_param_df.loc[('Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est)].loc[(slice(None), slice(None), slice(None), slice(None), [1,2]), (slice(None), 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD')]
#%%
df_to_use
df_to_use[df_to_use['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Number Of Degrees Of Freedom'] != 39]
#%%
'''
=============================
Comparison with Travis's data
=============================
'''
# Comparison with Travis's data for the case when the correction (due to imperfect power flatness) was applied to each data set.

# Comparison of the resonant frequencies
np.sum((df_with_weights['Resonant Frequency (Blinded) [kHz]'] + BLIND_OFFSET - (travis_corr_df['Resonant Frequency [kHz]'])) *  travis_corr_df['Weight'])
#%%
# Difference between the statistical frequencies (AC shift + SOD shift corrections are subtracted out). The comparison is made by using the same weights from Travis's analysis. The resonant frequencies in Travis's case have the blind offset already added to them.
np.sum((df_with_weights['Resonant Frequency (Blinded) [kHz]'] + df_with_weights['AC Shift [kHz]'] - df_with_weights['SOD Shift [kHz]'] + BLIND_OFFSET - (travis_corr_df['Resonant Frequency [kHz]'] + travis_corr_df['Doppler Shift [kHz]'] + travis_corr_df['AC Shift [kHz]'])) *  travis_corr_df['Weight'])

weights_arr_sum_travis = np.sum(travis_corr_df['Weight'])

f0_stat_travis = np.sqrt(np.sum((travis_corr_df['Weight']*travis_corr_df['Statistical Uncertainty [kHz]'])**2)) / weights_arr_sum_travis

sod_unc_travis = np.sum(travis_corr_df['Doppler Uncertainty [kHz]'] * travis_corr_df['Weight']) / weights_arr_sum_travis
comb_unc_travis = np.sum(travis_corr_df['Combiner Uncertainty [kHz]'] * travis_corr_df['Weight']) / weights_arr_sum_travis
alain_unc_travis = np.sum(travis_corr_df['Alain Uncertainty [kHz]'] * travis_corr_df['Weight']) / weights_arr_sum_travis
det_offset_unc_travis = np.sum(travis_corr_df['Detector Offset Uncertainty [kHz]'] * travis_corr_df['Weight']) / weights_arr_sum_travis
off_axis_unc_travis = np.sum(travis_corr_df['Off-axis Uncertainty [kHz]'] * travis_corr_df['Weight']) / weights_arr_sum_travis

tot_unc_travis = np.sqrt(f0_stat_travis**2 + sod_unc_travis**2 + comb_unc_travis**2 + alain_unc_travis**2 + det_offset_unc_travis**2 + off_axis_unc_travis**2)
#%%
np.sum(np.sqrt(travis_corr_df['Combiner Uncertainty [kHz]']**2 + travis_corr_df['Alain Uncertainty [kHz]']**2 + travis_corr_df['Detector Offset Uncertainty [kHz]']**2 + travis_corr_df['Off-axis Uncertainty [kHz]']**2)*travis_corr_df['Weight'])
#%%
f0_stat_travis
#%%
tot_unc_travis
#%%
np.sqrt(comb_unc_travis**2 + alain_unc_travis**2 + det_offset_unc_travis**2 + off_axis_unc_travis**2)
#%%
# Total AC and SOF shifts for each of the analyses

sod_shift_tot_travis = np.sum(travis_corr_df['Doppler Shift [kHz]'] * travis_corr_df['Weight'])

sod_shift_tot = np.sum(df_with_weights['SOD Shift [kHz]'] * travis_corr_df['Weight'])

ac_shift_tot_travis = np.sum(travis_corr_df['AC Shift [kHz]'] * travis_corr_df['Weight'])
ac_shift_tot = np.sum(df_with_weights['AC Shift [kHz]'] * travis_corr_df['Weight'])

# AC shift is positive - it needs to be subtracted from the statistical frequencies.
# SOD shift is negative - it needs to be added to the statistical frequencies
correct_tot = (-ac_shift_tot) + sod_shift_tot

# In Travis's analysis, the SOD shift is negative - thus the minus sign
correct_tot_travis = (-sod_shift_tot_travis) + (-ac_shift_tot_travis)
#%%
sod_shift_tot - (-sod_shift_tot_travis)
#%%
(-ac_shift_tot) - (-ac_shift_tot_travis)
#%%
correct_tot - correct_tot_travis
#%%
sod_shift_tot
#%%
-sod_shift_tot_travis
#%%
-ac_shift_tot
#%%
-ac_shift_tot_travis
#%%
0.75 + 0.81
#%%
travis_corr_df
#%%
sod_unc_np.sum(travis_corr_df['Doppler Uncertainty [kHz]']*travis_corr_df['Weight'])

#%%
ac_shift_tot_travis
#%%
df_with_weights['Zero-crossing Frequency STD (Normalized) [kHz]']
#%%
travis_corr_df['Statistical Uncertainty [kHz]']
#%%
df_with_weights[freq_std_col_rename] -travis_corr_df['Statistical Uncertainty [kHz]']
#%%

#%%
df_with_weights['Weight [%]']-travis_corr_df['Weight [%]']
#%%
# My final results using Travis's weights
f0_stat_nikita = np.sqrt(np.sum((travis_corr_df['Weight']*df_with_weights['Zero-crossing Frequency STD (Normalized) [kHz]'])**2)) / weights_arr_sum_travis

sod_unc_nikita = np.sum(df_with_weights['SOD Shift STD [kHz]'] * travis_corr_df['Weight']) / weights_arr_sum_travis
ac_unc_nikita = np.sum(df_with_weights['AC Shift Uncertainty [kHz]'] * travis_corr_df['Weight']) / weights_arr_sum_travis

phase_control_unc_nikita = np.sum(df_with_weights['Phase Control Uncertainty [kHz]'] * travis_corr_df['Weight']) / weights_arr_sum_travis

tot_unc_nikita = np.sqrt(f0_stat_nikita**2 + sod_unc_nikita**2 + ac_unc_nikita**2)
#%%
tot_unc_nikita
#%%
df_with_weights
#%%
df
#%%
grouped_exp_fosof_lineshape_param_df
#%%
grouped_exp_fosof_lineshape_param_df.loc['Waveguide Carrier Frequency Sweep', -1.0, 7, 49.86, 120, 24, 1]
#%%
grouped_exp_fosof_lineshape_param_df.loc[('Waveguide Carrier Frequency Sweep', -1.0, 6, 49.86, 120, 18, 1), (slice(None), 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD')]
#%%
rf_comb_1 = grouped_exp_fosof_lineshape_param_df.loc[('Waveguide Carrier Frequency Sweep', -1.0, 7, 49.86, 120, 24, 1), (slice(None), 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD')]['RF Combiner I Reference']['First Harmonic']['Phase Averaging']['Phase RMS Repeat STD']

rf_comb_2 = grouped_exp_fosof_lineshape_param_df.loc[('Waveguide Carrier Frequency Sweep', -1.0, 7, 49.86, 120, 24, 1), (slice(None), 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD')]['RF Combiner I Reference']['First Harmonic']['Phase Averaging']['Phase RMS Repeat STD']
#%%
rf_comb_1
#%%
av_freq_arr = (rf_comb_1['Zero-crossing Frequency [MHz]'] + rf_comb_2['Zero-crossing Frequency [MHz]']) / 2
#%%
av_std_arr = np.sqrt((rf_comb_1['Zero-crossing Frequency STD [MHz]']**2 + rf_comb_2['Zero-crossing Frequency STD [MHz]']**2)/2)
#%%
av_std_arr
#%%
np.average(a=av_freq_arr, weights=1/av_std_arr**2)
#%%
av_data = straight_line_fit_params(av_freq_arr, av_std_arr)
av_data
#%%
np.sqrt(av_data['Reduced Chi Squared']) * av_data['Weighted STD']
#%%
zero_cross_av_df.loc[6, 49.86, 120]['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD']
#%%
travis_no_corr_df.loc[6, 49.86]
#%%
travis_corr_df['AC Shift [kHz]']
#%%
comb_2 = np.array([909.8638047, 909.8870046, 909.889247699999])
comb_2_std = np.array([0.009252033, 0.009094745, 0.007538792])
comb_2_chi = np.array([0.884144604, 0.899242449, 1.21162947])

comb_1 = np.array([909.865203099999, 909.8883161, 909.890837099999])
comb_1_std = np.array([0.009247169, 0.009074835, 0.007535271])
comb_1_chi = np.array([0.883928077, 0.896106374, 1.21131583099999])

comb_2_std = comb_2_std / np.sqrt(comb_2_chi)
comb_1_std = comb_1_std / np.sqrt(comb_1_chi)
comb_av = (comb_1 + comb_2)/2

comb_std = np.sqrt(((comb_1_std**2 + comb_2_std**2)/2))

#comb_std = np.array([np.sqrt(np.average([comb_1_std[0], comb_1_std[0]])**2), np.sqrt(np.average([comb_1_std[1], comb_1_std[1]])**2), np.sqrt(np.average([comb_1_std[2], comb_1_std[2]])**2)])
#%%
comb_av
comb_std
#%%
av_data2 = straight_line_fit_params(comb_av, comb_std)
av_data2
#%%
np.average(a=comb_av, weights=1/comb_std**2)
#%%
np.sqrt(np.average([comb_1_std[1], comb_1_std[1]])**2)
#%%
np.sqrt(np.average([comb_1_std[0], comb_1_std[0]])**2)
#%%
comb_2 = np.array([909.8634703, 909.886695799999, 909.8890318])
comb_2_std = np.array([0.009198799, 0.008980343, 0.007576358])
comb_2_chi = np.array([0.874383688, 0.877207154, 1.224501189])

comb_1 = np.array([909.8648684, 909.888007, 909.890620799999])
comb_1_std = np.array([0.009194198, 0.008960869, 0.00757317699999999])
comb_1_chi = np.array([0.874214690999999, 0.874184498, 1.224299869])

comb_2_std = comb_2_std / np.sqrt(comb_2_chi)
comb_1_std = comb_1_std / np.sqrt(comb_1_chi)
comb_av = (comb_1 + comb_2)/2

#comb_std = np.sqrt(((comb_1_std**2 + comb_2_std**2)/2))

comb_std = np.array([np.sqrt(np.average([comb_1_std[0], comb_1_std[0]])**2), np.sqrt(np.average([comb_1_std[1], comb_1_std[1]])**2), np.sqrt(np.average([comb_1_std[2], comb_1_std[2]])**2)])
#%%
comb_av
comb_std
#%%
av_data2 = straight_line_fit_params(comb_av, comb_std)
av_data2
#%%
np.average(a=comb_av, weights=1/comb_std**2)
#%%
np.sqrt(np.average([comb_1_std[1], comb_1_std[1]])**2)
#%%
np.sqrt(np.average([comb_1_std[0], comb_1_std[0]])**2)
#%%
travis_corr_df
#%%
df['Resonant Frequency (Blinded) [kHz]'] + df['AC Shift [kHz]'] - df['SOD Shift [kHz]'] - (travis_corr_df['Resonant Frequency [kHz]'] + travis_corr_df['Doppler Shift [kHz]'] + travis_corr_df['AC Shift [kHz]'] - BLIND_OFFSET)
#%%

#%%
travis_no_corr_df['Resonant Frequency [kHz]']
#%%
df['Resonant Frequency (Blinded) [kHz]'] + df['AC Shift [kHz]'] - df['SOD Shift [kHz]'] - travis_no_corr_df['Resonant Frequency [kHz]']
#%%
df[freq_std_col_rename] - travis_no_corr_df['Statistical Uncertainty [kHz]']
#%%
grouped_exp_fosof_lineshape_param_df.loc[('Waveguide Carrier Frequency Sweep', -1.0, 4, 49.86, 120, 8, 1), (slice(None), 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD')]
#%%
df
#%%
'''
========================
Plot of the resonant frequency for different experiment parameters
========================
'''
res_freq_for_plotting_df = df.copy()

wvg_e_field_list = list(res_freq_for_plotting_df.index.get_level_values('Waveguide Electric Field [V/cm]').drop_duplicates().values)

wvg_sep_list = list(res_freq_for_plotting_df.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().values)

acc_volt_list = list(res_freq_for_plotting_df.index.get_level_values('Accelerating Voltage [kV]').drop_duplicates().values)

plot_marker_list = ['^', 'o', 'D', 's', 'o']
plot_marker_dict = dict(list(zip(*[wvg_sep_list, plot_marker_list])))

plot_marker_fill_style_list = ['none', 'top', 'full']
plot_marker_fill_style_dict = dict(list(zip(*[acc_volt_list, plot_marker_fill_style_list])))

plot_marker_edge_color_list = ['brown', 'red', 'purple', 'green', 'blue']
plot_marker_edge_color_dict = dict(list(zip(*[wvg_e_field_list, plot_marker_edge_color_list])))

res_freq_for_plotting_df['Plot Marker Type'] = None
res_freq_for_plotting_df['Plot Marker Fill Style'] = None
res_freq_for_plotting_df['Plot Marker Edge Color'] = None

for wvg_e_field in wvg_e_field_list:
    res_freq_for_plotting_df.loc[(slice(None), slice(None), wvg_e_field), ('Plot Marker Edge Color')] = plot_marker_edge_color_dict[wvg_e_field]

for wvg_sep in wvg_sep_list:
    res_freq_for_plotting_df.loc[(wvg_sep), ('Plot Marker Type')] = plot_marker_dict[wvg_sep]

for acc_volt in acc_volt_list:
    res_freq_for_plotting_df.loc[(slice(None), acc_volt, slice(None)), ('Plot Marker Fill Style')] = plot_marker_fill_style_dict[acc_volt]


res_freq_for_plotting_df['Total Uncertainty [kHz]'] = np.sqrt(res_freq_for_plotting_df[freq_std_col_rename]**2 + res_freq_for_plotting_df['AC Shift Uncertainty [kHz]']**2 + res_freq_for_plotting_df['SOD Shift STD [kHz]']**2 + res_freq_for_plotting_df['Phase Control Uncertainty [kHz]']**2)

#res_freq_for_plotting_df['Total Uncertainty [kHz]'] = np.sqrt(res_freq_for_plotting_df[freq_std_col_rename]**2)

res_freq_for_plotting_df['Resonant Frequency [kHz]'] = res_freq_for_plotting_df['Resonant Frequency (Blinded) [kHz]'] + BLIND_OFFSET

res_freq_for_plotting_df = res_freq_for_plotting_df.reorder_levels(['Waveguide Electric Field [V/cm]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]']).sort_index().reset_index().reset_index()

index_mean_vs_e_field_list = []
index_color_list = []
for group_name, grouped_df in res_freq_for_plotting_df.set_index('Waveguide Electric Field [V/cm]').groupby('Waveguide Electric Field [V/cm]'):

    min_freq = ((grouped_df['Resonant Frequency [kHz]'] - grouped_df['Total Uncertainty [kHz]']).min()) / 1E3

    max_freq = ((grouped_df['Resonant Frequency [kHz]'] + grouped_df['Total Uncertainty [kHz]']).max()) / 1E3

    index_mean_vs_e_field_list.append([group_name, grouped_df['index'].mean(), grouped_df['index'].max() - grouped_df['index'].min() + 0.4, min_freq-2/1E3, max_freq+2/1E3, grouped_df.shape[0]])

    index_color_list.append(grouped_df['Plot Marker Edge Color'].values[0])

index_mean_vs_e_field_arr = np.array(index_mean_vs_e_field_list)
index_top_bottom = ['top', 'top', 'top', 'top', 'top']

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(18, 8)

for marker in list(plot_marker_dict.values()):

    data_marker_df = res_freq_for_plotting_df[res_freq_for_plotting_df['Plot Marker Type'] == marker]

    for color in list(plot_marker_edge_color_dict.values()):
        data_color_df = data_marker_df[data_marker_df['Plot Marker Edge Color'] == color]
        if data_color_df.shape[0] > 0:

            for fill_style in list(plot_marker_fill_style_dict.values()):

                data_fill_style_df = data_color_df[data_color_df['Plot Marker Fill Style'] == fill_style]

                if data_fill_style_df.shape[0] > 0:
                    x_data_arr = data_fill_style_df['index']
                    y_data_arr = data_fill_style_df['Resonant Frequency [kHz]'] / 1E3
                    y_data_err_arr = data_fill_style_df['Total Uncertainty [kHz]'] / 1E3

                    axes[0].errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, fillstyle=fill_style, capsize=5, capthick=2, linestyle='', markersize=13, markerfacecoloralt='cyan')

                # if face_color != 'full':
                #     ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, markerfacecolor='none', capsize=5, capthick=2, linestyle='', markersize=10)
                # else:
                #     ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, capsize=5, capthick=2, linestyle='', markersize=10)
        #data_marker_df.plot(kind='scatter', x='Inverse Slope [kHz/mrad]', xerr='Inverse Slope STD [kHz/mrad]', y='Resonant Frequency [kHz]', yerr='Total Uncertainty [kHz]', ax=ax, marker=marker, color=data_marker_df['Plot Color'].values)

ax_x_lim = axes[0].get_xlim()

axes[0].ticklabel_format(useOffset=False)

rect_f0 = Rectangle((ax_x_lim[0]-2, (f0_1_s['Resonant Frequency [kHz]']-f0_1_s['Total Uncertainty [kHz]'])/1E3), ax_x_lim[1]-ax_x_lim[0]+10, 2*f0_1_s['Total Uncertainty [kHz]']/1E3, color='wheat', fill=True, alpha=1)
axes[0].add_patch(rect_f0)

axes[0].set_xlim([-2, ax_x_lim[1]+1])

font_size = 15
for item in ([axes[0].title, axes[0].xaxis.label, axes[0].yaxis.label] +
             axes[0].get_xticklabels() + axes[0].get_yticklabels()):
    item.set_fontsize(font_size)

axes[0].set_ylabel(r'$f_0$ (MHz)')

for i in range(index_mean_vs_e_field_arr.shape[0]):
    arrow_style = '-[, widthB=' + str(index_mean_vs_e_field_arr[i,2]) + ', lengthB='+str(0.3)

    if index_top_bottom[i] == 'top':
        xy=(index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,3])
        xy_text = (index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,3] - 2/1E3)
    if index_top_bottom[i] == 'bottom':
        xy=(index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,4])
        xy_text = (index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,4] + 2/1E3)

    axes[0].annotate(str(index_mean_vs_e_field_arr[i,0]) + ' V/cm', xy=xy, xytext=xy_text, fontsize=font_size, ha='center', va=index_top_bottom[i], arrowprops=dict(arrowstyle=arrow_style, lw=2.0, color=index_color_list[i]), color=index_color_list[i])

ax_y_lim = axes[0].get_ylim()

ax_y_lim = np.array([ax_y_lim[0], ax_y_lim[1]])

ax_y_lim[0] = ax_y_lim[0] - 7/1E3
axes[0].set_ylim(ax_y_lim)
axes[0].set_xticklabels([])
axes[0].set_xticks([])

# Plotting by combining data, for same particular exp parameter, together.
res_freq_vs_e_field_df = res_freq_for_plotting_df.set_index('Waveguide Electric Field [V/cm]').sort_index().groupby('Waveguide Electric Field [V/cm]').apply(lambda df: straight_line_fit_params(df['Resonant Frequency [kHz]'], df['Total Uncertainty [kHz]']))

res_freq_vs_acc_volt_df = res_freq_for_plotting_df.set_index('Accelerating Voltage [kV]').sort_index().groupby('Accelerating Voltage [kV]').apply(lambda df: straight_line_fit_params(df['Resonant Frequency [kHz]'], df['Total Uncertainty [kHz]']))

res_freq_vs_wvg_sep_df = res_freq_for_plotting_df.set_index('Waveguide Separation [cm]').sort_index().groupby('Waveguide Separation [cm]').apply(lambda df: straight_line_fit_params(df['Resonant Frequency [kHz]'], df['Total Uncertainty [kHz]']))

index_e_field_arr = np.arange(0, res_freq_vs_e_field_df.shape[0])
res_freq_vs_e_field_df['index'] = index_e_field_arr

index_acc_volt_arr = np.arange(0, res_freq_vs_acc_volt_df.shape[0]) + index_e_field_arr[-1] + 1
res_freq_vs_acc_volt_df['index'] = index_acc_volt_arr

index_wvg_sep_arr = np.arange(0, res_freq_vs_wvg_sep_df.shape[0]) + index_acc_volt_arr[-1] + 1
res_freq_vs_wvg_sep_df['index'] = index_wvg_sep_arr

df_list = [res_freq_vs_e_field_df, res_freq_vs_acc_volt_df, res_freq_vs_wvg_sep_df]
group_name_list = [r'$E_0$ (V/cm)', r'$V_\mathrm{HV}$ (kV)', r'$D$ (cm)']

group_color_list = ['green', 'red', 'blue', 'purple']


index_mean_vs_e_field_list = []
index_color_list = []

for i in range(len(df_list)):

    grouped_df = df_list[i]
    group_name = group_name_list[i]

    min_freq = ((grouped_df['Weighted Mean'] - grouped_df['Weighted STD']).min()) / 1E3

    max_freq = ((grouped_df['Weighted Mean'] + grouped_df['Weighted STD']).max()) / 1E3

    index_mean_vs_e_field_list.append([0, grouped_df['index'].mean(), grouped_df['index'].max() - grouped_df['index'].min() + 0.4, min_freq-2/1E3, max_freq+2/1E3, grouped_df.shape[0]])

    index_color_list.append(group_color_list[i])

index_mean_vs_e_field_arr = np.array(index_mean_vs_e_field_list)
index_top_bottom = ['top', 'top', 'top']

x_data_arr = res_freq_vs_e_field_df['index']
y_data_arr = res_freq_vs_e_field_df['Weighted Mean'] / 1E3
y_data_err_arr = res_freq_vs_e_field_df['Weighted STD'] / 1E3

axes[1].errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker='.', color='black', capsize=5, capthick=2, linestyle='', markersize=13)

x_data_arr = res_freq_vs_acc_volt_df['index']
y_data_arr = res_freq_vs_acc_volt_df['Weighted Mean'] / 1E3
y_data_err_arr = res_freq_vs_acc_volt_df['Weighted STD'] / 1E3

axes[1].errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker='.', color='black', capsize=5, capthick=2, linestyle='', markersize=13)

x_data_arr = res_freq_vs_wvg_sep_df['index']
y_data_arr = res_freq_vs_wvg_sep_df['Weighted Mean'] / 1E3
y_data_err_arr = res_freq_vs_wvg_sep_df['Weighted STD'] / 1E3

axes[1].errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker='.', color='black', capsize=5, capthick=2, linestyle='', markersize=13)

rect_f0 = Rectangle((ax_x_lim[0]-2, (f0_1_s['Resonant Frequency [kHz]']-f0_1_s['Total Uncertainty [kHz]'])/1E3), ax_x_lim[1]-ax_x_lim[0]+10, 2*f0_1_s['Total Uncertainty [kHz]']/1E3, color='wheat', fill=True, alpha=1)
axes[1].add_patch(rect_f0)

font_size = 15

for i in range(index_mean_vs_e_field_arr.shape[0]):
    arrow_style = '-[, widthB=' + str(index_mean_vs_e_field_arr[i,2]*1.7) + ', lengthB='+str(0.3)

    if index_top_bottom[i] == 'top':
        xy=(index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,3])
        xy_text = (index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,3] - 2/1E3)
    if index_top_bottom[i] == 'bottom':
        xy=(index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,4])
        xy_text = (index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,4] + 2/1E3)

    axes[1].annotate(group_name_list[i], xy=xy, xytext=xy_text, fontsize=font_size, ha='center', va=index_top_bottom[i], arrowprops=dict(arrowstyle=arrow_style, lw=2.0, color=index_color_list[i]), color=index_color_list[i])


# Annotation for data vs e field amplitude
for i in np.arange(np.min(index_e_field_arr), np.max(index_e_field_arr)+1):
    freq = (res_freq_vs_e_field_df[res_freq_vs_e_field_df['index'] == i]['Weighted Mean'].values[0] + res_freq_vs_e_field_df[res_freq_vs_e_field_df['index'] == i]['Weighted STD'].values[0]) / 1E3

    val = res_freq_vs_e_field_df[res_freq_vs_e_field_df['index'] == i].index.values[0]

    axes[1].annotate(str(int(val)) + ' V/cm', xy=[i, freq], xytext=[i, freq + 1.25/1E3], fontsize=font_size, ha='center', va='bottom', color=index_color_list[0], rotation=90)

# Annotation for data vs accelerating voltage
for i in np.arange(np.min(index_acc_volt_arr), np.max(index_acc_volt_arr)+1):
    freq = (res_freq_vs_acc_volt_df[res_freq_vs_acc_volt_df['index'] == i]['Weighted Mean'].values[0] + res_freq_vs_acc_volt_df[res_freq_vs_acc_volt_df['index'] == i]['Weighted STD'].values[0]) / 1E3

    val = res_freq_vs_acc_volt_df[res_freq_vs_acc_volt_df['index'] == i].index.values[0]

    axes[1].annotate(str(val) + ' kV', xy=[i, freq], xytext=[i, freq + 1.25/1E3], fontsize=font_size, ha='center', va='bottom', color=index_color_list[1], rotation=90)

# Annotation for data vs separation
for i in np.arange(np.min(index_wvg_sep_arr), np.max(index_wvg_sep_arr)+1):
    freq = (res_freq_vs_wvg_sep_df[res_freq_vs_wvg_sep_df['index'] == i]['Weighted Mean'].values[0] + res_freq_vs_wvg_sep_df[res_freq_vs_wvg_sep_df['index'] == i]['Weighted STD'].values[0]) / 1E3

    val = res_freq_vs_wvg_sep_df[res_freq_vs_wvg_sep_df['index'] == i].index.values[0]

    axes[1].annotate(str(int(val)) + ' cm', xy=[i, freq], xytext=[i, freq + 1.25/1E3], fontsize=font_size, ha='center', va='bottom', color=index_color_list[2], rotation=90)

for item in ([axes[1].title, axes[1].xaxis.label, axes[1].yaxis.label] +
             axes[1].get_xticklabels() + axes[1].get_yticklabels()):
    item.set_fontsize(font_size)

#ax.set_ylabel(r'$f_0$ (MHz)')

axes[1].set_ylim(ax_y_lim)

axes[1].set_xticklabels([])
axes[1].set_xticks([])
axes[1].set_yticklabels([])

from matplotlib.ticker import FormatStrFormatter
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

fig.tight_layout()


plt.show()


# os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Thesis\exp_acq')
#
# plt_name = 'res_freq.pdf'
# plt.savefig(plt_name)
plt.show()
#%%
index_mean_vs_e_field_arr[1,2]*0.2
#%%
res_freq_vs_e_field_df[res_freq_vs_e_field_df['index'] == i]['Weighted Mean'].values[0]
#%%
travis_corr_df['Resonant Frequency [kHz]'] + travis_corr_df['AC Shift [kHz]'] + travis_corr_df['Doppler Shift [kHz]'] - BLIND_OFFSET

#%%
df_with_weights['Resonant Frequency (Blinded) [kHz]'] + df_with_weights['AC Shift [kHz]'] - df_with_weights['SOD Shift [kHz]']
#%%
df_with_weights
#%%
np.sqrt(travis_corr_df['Doppler Uncertainty [kHz]']**2 + travis_corr_df['Combiner Uncertainty [kHz]']**2 + travis_corr_df['Alain Uncertainty [kHz]']**2 + travis_corr_df['Detector Offset Uncertainty [kHz]']**2 + travis_corr_df['Off-axis Uncertainty [kHz]']**2 + df_with_weights['Phase Control Uncertainty [kHz]']**2)
#%%
index_mean_vs_e_field_arr.shape
#%%
df_to_use = grouped_exp_fosof_lineshape_param_df.loc['Waveguide Carrier Frequency Sweep', -1.0, slice(None), slice(None), slice(None), slice(None), [2]]['RF Combiner I Reference']['First Harmonic']['Phasor Averaging']['Phase RMS Repeat STD']

df_to_use[df_to_use['Number Of Degrees Of Freedom'] > 37]
#%%
os.chdir()
zero_cross_av_df.reorder_levels(order=['STD State', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'Data Field'], axis=1)['Normalized']

slope_av_df.reorder_levels(order=['STD State', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'Data Field'], axis=1)['Normalized']
#%%
grouped_exp_fosof_lineshape_param_df
#%%
fosof_lineshape_param_norm_df['RF Combiner I Reference', 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD']
#%%
fosof_lineshape_param_av_comb_df
#%%
fosof_lineshape_param_av_comb_df
#%%
'Beam RMS Radius Frequency Shift Uncertainty [kHz]'
#%%
ac_shift_df['Fractional Offset Frequency Shift [kHz]'] / ac_shift_df['AC Shift [kHz]'] * 1E2
#%%
64/24
#%%

#%%
ac_shift_df['Beam RMS Radius Frequency Shift [kHz]'] / ac_shift_df['AC Shift [kHz]'] * 1E2
#%%
ac_shift_df
#%%
ac_shift_MHz_df
