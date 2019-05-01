'''
2019-04-05

Import of Travis's data with formatting it in a way that makes it easy to compare this data with my analysis.
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

from exp_data_analysis import *
from beam_speed import *
import fosof_data_set_analysis
import hydrogen_sim_data

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

# Blind offset in kHz
BLIND_OFFSET = 0.03174024731 * 1E3
#%%
'''
=================================
Data from Travis (Final weights)
=================================
'''
os.chdir(travis_data_folder_path)

# Data without correcting each data set for imperfect power in the RF system during its acquisition.

travis_no_corr_df = pd.read_csv(filepath_or_buffer='final_weights_nocorrection_to_import.csv', delimiter=',', comment='#', header=[0], skip_blank_lines=True, index_col=[0, 1, 2,])

travis_no_corr_df = travis_no_corr_df.reset_index().rename(columns={'Separation [cm]': 'Waveguide Separation [cm]', 'RF Amplitude [V/cm]': 'Waveguide Electric Field [V/cm]', 'Weights': 'Weight'})

travis_no_corr_df['Waveguide Electric Field [V/cm]'] = travis_no_corr_df['Waveguide Electric Field [V/cm]'].astype(np.float32)

travis_no_corr_df['Waveguide Separation [cm]'] = travis_no_corr_df['Waveguide Separation [cm]'].astype(np.float32)

acc_volt_16_index = travis_no_corr_df[travis_no_corr_df['Accelerating Voltage [kV]'] == 16].index
acc_volt_22_index = travis_no_corr_df[travis_no_corr_df['Accelerating Voltage [kV]'] == 22].index
acc_volt_50_index = travis_no_corr_df[travis_no_corr_df['Accelerating Voltage [kV]'] == 50].index

travis_no_corr_df.loc[(acc_volt_16_index), ('Accelerating Voltage [kV]')] = 16.27
travis_no_corr_df.loc[(acc_volt_22_index), ('Accelerating Voltage [kV]')] = 22.17
travis_no_corr_df.loc[(acc_volt_50_index), ('Accelerating Voltage [kV]')] = 49.86

travis_no_corr_df.set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'], inplace=True)

travis_no_corr_df['Weight'] = travis_no_corr_df['Weight'] / 4

travis_no_corr_df = travis_no_corr_df * 1E3


travis_no_corr_df['Weight'] = travis_no_corr_df['Weight'] / 1E3

travis_no_corr_df['Weight [%]'] = travis_no_corr_df['Weight'] * 1E2

travis_no_corr_df.rename(columns={'Doppler Shift [MHz]': 'Doppler Shift [kHz]', 'Resonant Frequency [MHz]': 'Resonant Frequency [kHz]', 'AC Shift [MHz]': 'AC Shift [kHz]', 'Statistical Uncertainty [MHz]': 'Statistical Uncertainty [kHz]', 'Doppler Uncertainty [MHz]': 'Doppler Uncertainty [kHz]', 'Combiner Uncertainty [MHz]': 'Combiner Uncertainty [kHz]', 'Alain Uncertainty 5% [MHz]': 'Alain Uncertainty [kHz]', 'Detector Offset Uncerainty [MHz]': 'Detector Offset Uncertainty [kHz]', 'Off-Axis Uncertainty [MHz]': 'Off-axis Uncertainty [kHz]'}, inplace=True)


# Final weights with the correction applied
travis_corr_df = pd.read_csv(filepath_or_buffer='final_weights.csv', delimiter=',', comment='#', header=[0], skip_blank_lines=True, index_col=[0, 1, 2,])

travis_corr_df = travis_corr_df.reset_index().rename(columns={'Separation [cm]': 'Waveguide Separation [cm]', 'RF Amplitude [V/cm]': 'Waveguide Electric Field [V/cm]', 'Weights': 'Weight'})

travis_corr_df['Waveguide Electric Field [V/cm]'] = travis_corr_df['Waveguide Electric Field [V/cm]'].astype(np.float32)

travis_corr_df['Waveguide Separation [cm]'] = travis_corr_df['Waveguide Separation [cm]'].astype(np.float32)

acc_volt_16_index = travis_corr_df[travis_corr_df['Accelerating Voltage [kV]'] == 16].index
acc_volt_22_index = travis_corr_df[travis_corr_df['Accelerating Voltage [kV]'] == 22].index
acc_volt_50_index = travis_corr_df[travis_corr_df['Accelerating Voltage [kV]'] == 50].index

travis_corr_df.loc[(acc_volt_16_index), ('Accelerating Voltage [kV]')] = 16.27
travis_corr_df.loc[(acc_volt_22_index), ('Accelerating Voltage [kV]')] = 22.17
travis_corr_df.loc[(acc_volt_50_index), ('Accelerating Voltage [kV]')] = 49.86

travis_corr_df.set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'], inplace=True)

travis_corr_df['Weight'] = travis_corr_df['Weight']

travis_corr_df = travis_corr_df * 1E3


travis_corr_df['Weight'] = travis_corr_df['Weight'] / 1E3

travis_corr_df['Weight [%]'] = travis_corr_df['Weight'] * 1E2

travis_corr_df.rename(columns={'Doppler Shift [MHz]': 'SOD Shift [kHz]', 'Resonant Frequency [MHz]': 'Resonant Frequency [kHz]', 'AC Shift [MHz]': 'AC Shift [kHz]', 'Statistical Uncertainty [MHz]': 'Statistical Uncertainty [kHz]', 'Doppler Uncertainty [MHz]': 'SOD Shift STD [kHz]', 'Combiner Uncertainty [MHz]': 'Combiner Uncertainty [kHz]', 'Alain Uncertainty 5% [MHz]': 'AC Shift Simulation Uncertainty [kHz]', 'Detector Offset Uncerainty [MHz]': 'Fractional Offset Frequency Shift Uncertainty [kHz]', 'Off-Axis Uncertainty [MHz]': 'Beam RMS Radius Frequency Shift Uncertainty [kHz]'}, inplace=True)

travis_corr_df['SOD Shift [kHz]'] = -1 * travis_corr_df['SOD Shift [kHz]']
travis_corr_df = travis_corr_df.sort_index()

# Another set of files with weights. Frequencies seem to be more correct
os.chdir(travis_data_folder_path)
os.chdir('FOSOFDataAnalysis')

#os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Data\FOSOF analyzed data sets\Travis_data_and_code\FOSOFDataAnalysis')

# Final weights with the correction applied
travis_corr_2_df = pd.read_csv(filepath_or_buffer='final_weights.csv', delimiter=',', comment='#', header=[0], skip_blank_lines=True, index_col=[0, 1, 2,])

travis_corr_2_df = travis_corr_2_df.reset_index().rename(columns={'Separation [cm]': 'Waveguide Separation [cm]', 'RF Amplitude [V/cm]': 'Waveguide Electric Field [V/cm]', 'Weights': 'Weight'})

travis_corr_2_df['Waveguide Electric Field [V/cm]'] = travis_corr_2_df['Waveguide Electric Field [V/cm]'].astype(np.float32)

travis_corr_2_df['Waveguide Separation [cm]'] = travis_corr_2_df['Waveguide Separation [cm]'].astype(np.float32)

acc_volt_16_index = travis_corr_2_df[travis_corr_2_df['Accelerating Voltage [kV]'] == 16].index
acc_volt_22_index = travis_corr_2_df[travis_corr_2_df['Accelerating Voltage [kV]'] == 22].index
acc_volt_50_index = travis_corr_2_df[travis_corr_2_df['Accelerating Voltage [kV]'] == 50].index

travis_corr_2_df.loc[(acc_volt_16_index), ('Accelerating Voltage [kV]')] = 16.27
travis_corr_2_df.loc[(acc_volt_22_index), ('Accelerating Voltage [kV]')] = 22.17
travis_corr_2_df.loc[(acc_volt_50_index), ('Accelerating Voltage [kV]')] = 49.86

travis_corr_2_df.set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'], inplace=True)

travis_corr_2_df['Weight'] = travis_corr_2_df['Weight'] / 4

travis_corr_2_df = travis_corr_2_df * 1E3


travis_corr_2_df['Weight'] = travis_corr_2_df['Weight'] / 1E3

travis_corr_2_df['Weight [%]'] = travis_corr_2_df['Weight'] * 1E2

travis_corr_2_df.rename(columns={'Doppler Shift [MHz]': 'SOD Shift [kHz]', 'Resonant Frequency [MHz]': 'Resonant Frequency [kHz]', 'AC Shift [MHz]': 'AC Shift [kHz]', 'Statistical Uncertainty [MHz]': 'Statistical Uncertainty [kHz]', 'Doppler Uncertainty [MHz]': 'SOD Shift STD [kHz]', 'Combiner Uncertainty [MHz]': 'Combiner Uncertainty [kHz]', 'Alain Uncertainty 5% [MHz]': 'AC Shift Simulation Uncertainty [kHz]', 'Detector Offset Uncerainty [MHz]': 'Fractional Offset Frequency Shift Uncertainty [kHz]', 'Off-Axis Uncertainty [MHz]': 'Beam RMS Radius Frequency Shift Uncertainty [kHz]'}, inplace=True)

travis_corr_2_df['SOD Shift [kHz]'] = -1 * travis_corr_2_df['SOD Shift [kHz]']
travis_corr_2_df = travis_corr_2_df.sort_index()

#%%
''' Summary of all of the data
'''

os.chdir(travis_data_folder_path)
os.chdir('FOSOFDataAnalysis')

# List of data sets and associated frequencies.
summ_df = pd.read_csv(filepath_or_buffer='fosof_summary_all.csv', delimiter=',', comment='#', header=[0], skip_blank_lines=True, index_col=[0])

# All of the data sets are of 4 MHz range in this file.
summ_df['Frequency Range [MHz]'].drop_duplicates().values

# We are only interested in the 'RMS Uncertainty' averaging type.
summ_df = summ_df.set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Dataset Type', 'Frequency Range [MHz]', 'Peak RF Field Amplitude [V/cm]', 'Dataset Timestamp', 'Averaging Method', 'Combiner/Harmonic', 'Corrected']).loc[(slice(None), slice(None), slice(None), 4, slice(None), slice(None), 'RMS Uncertainty', slice(None)), slice(None)].sort_index().reset_index(['Averaging Method', 'Frequency Range [MHz]'], drop=True)

summ_df.columns.names = ['Data Field']

summ_df.rename(columns={'Resonant Frequency [MHz]':'Zero-crossing Frequency [MHz]', 'Error in Resonant Frequency [MHz]': 'Zero-crossing Frequency STD [MHz]', 'Slope [rad/MHz]': 'Slope [Rad/MHz]', 'Error in Slope [rad/MHz]': 'Slope STD [Rad/MHz]', 'Y-Intercept [rad]': 'Offset [Rad]', 'Error in Y-Intercept [rad]': 'Offset STD [Rad]', 'Chi Squared': 'Reduced Chi-Squared'}, inplace=True)

# Notice that the uncertainties are already expanded for the reduced chi-squared. However, the expansion is not performed correctly: irrespective of the chi-squared, all of the uncertaities are multiplied by np.sqrt(chi_squared); even if the reduced chi-squared < 1. I need to fix this
summ_df['Zero-crossing Frequency STD [MHz]'] = summ_df['Zero-crossing Frequency STD [MHz]'] / np.sqrt(summ_df['Reduced Chi-Squared'])
summ_df['Slope STD [Rad/MHz]'] = summ_df['Slope STD [Rad/MHz]'] / np.sqrt(summ_df['Reduced Chi-Squared'])
summ_df['Offset STD [Rad]'] = summ_df['Offset STD [Rad]'] / np.sqrt(summ_df['Reduced Chi-Squared'])

large_chi_squared_index = summ_df[summ_df['Reduced Chi-Squared'] > 1].index

summ_df['Zero-crossing Frequency STD (Normalized) [MHz]'] = summ_df['Zero-crossing Frequency STD [MHz]']
summ_df['Slope STD (Normalized) [Rad/MHz]'] = summ_df['Slope STD [Rad/MHz]']
summ_df['Offset STD (Normalized) [Rad]'] = summ_df['Offset STD [Rad]']

summ_df.loc[large_chi_squared_index, 'Zero-crossing Frequency STD (Normalized) [MHz]'] = summ_df.loc[large_chi_squared_index, 'Zero-crossing Frequency STD [MHz]'] * np.sqrt(summ_df.loc[large_chi_squared_index, 'Reduced Chi-Squared'])

summ_df.loc[large_chi_squared_index, 'Slope STD (Normalized) [Rad/MHz]'] = summ_df.loc[large_chi_squared_index, 'Slope STD [Rad/MHz]'] * np.sqrt(summ_df.loc[large_chi_squared_index, 'Reduced Chi-Squared'])

summ_df.loc[large_chi_squared_index, 'Offset STD (Normalized) [Rad]'] = summ_df.loc[large_chi_squared_index, 'Offset STD [Rad]'] * np.sqrt(summ_df.loc[large_chi_squared_index, 'Reduced Chi-Squared'])

# Whether to use the data sets, uncertainty of which was expanded for chi-squared > 1.
normalized_data_set_Q = False

if not normalized_data_set_Q:
    summ_df['Zero-crossing Frequency STD (Normalized) [MHz]'] = summ_df['Zero-crossing Frequency STD [MHz]']
    summ_df['Slope STD (Normalized) [Rad/MHz]'] = summ_df['Slope STD [Rad/MHz]']
    summ_df['Offset STD (Normalized) [Rad]'] = summ_df['Offset STD [Rad]']


summ_df.drop(columns=['Folder', 'B_x [Gauss]', 'B_y [Gauss]', 'Mass Flow Rate [CC]', 'Pre-Quench 910 State'], inplace=True)

summ_df = summ_df.unstack(level=['Combiner/Harmonic']).reorder_levels(axis='columns', order=['Combiner/Harmonic', 'Data Field']).sort_index(axis='columns').rename(mapper={'Combiner 1, Harmonic 1': 'RF Combiner I Reference', 'Combiner 2, Harmonic 1': 'RF Combiner R Reference'}, level='Combiner/Harmonic', axis='columns')

summ_df = summ_df.reset_index().rename(columns={'Separation [cm]': 'Waveguide Separation [cm]', 'Peak RF Field Amplitude [V/cm]': 'Waveguide Electric Field [V/cm]', 'Weights': 'Weight'})

summ_df['Waveguide Electric Field [V/cm]'] = summ_df['Waveguide Electric Field [V/cm]'].astype(np.float32)

summ_df['Waveguide Separation [cm]'] = summ_df['Waveguide Separation [cm]'].astype(np.float32)

acc_volt_16_index = summ_df[summ_df['Accelerating Voltage [kV]'] == 16].index
acc_volt_22_index = summ_df[summ_df['Accelerating Voltage [kV]'] == 22].index
acc_volt_50_index = summ_df[summ_df['Accelerating Voltage [kV]'] == 50].index

summ_df.loc[(acc_volt_16_index), ('Accelerating Voltage [kV]')] = 16.27
summ_df.loc[(acc_volt_22_index), ('Accelerating Voltage [kV]')] = 22.17
summ_df.loc[(acc_volt_50_index), ('Accelerating Voltage [kV]')] = 49.86

summ_df.set_index(['Dataset Type', 'Corrected', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Dataset Timestamp'], inplace=True)


rf_comb_I_df = summ_df['RF Combiner I Reference']
rf_comb_R_df = summ_df['RF Combiner R Reference']

fosof_lineshape_param_av_comb_t_df = rf_comb_I_df.copy()

fosof_lineshape_param_av_comb_t_df['Zero-crossing Frequency [MHz]'] = (rf_comb_I_df['Zero-crossing Frequency [MHz]'] + rf_comb_R_df['Zero-crossing Frequency [MHz]']) / 2

fosof_lineshape_param_av_comb_t_df['Slope [Rad/MHz]'] = (rf_comb_I_df['Slope [Rad/MHz]'] + rf_comb_R_df['Slope [Rad/MHz]']) / 2

fosof_lineshape_param_av_comb_t_df['Zero-crossing Frequency STD (Normalized) [MHz]'] = np.sqrt((rf_comb_I_df['Zero-crossing Frequency STD (Normalized) [MHz]']**2 + rf_comb_R_df['Zero-crossing Frequency STD (Normalized) [MHz]']**2)) / np.sqrt(2)

fosof_lineshape_param_av_comb_t_df['Slope STD (Normalized) [Rad/MHz]'] = np.sqrt((rf_comb_I_df['Slope STD (Normalized) [Rad/MHz]']**2 + rf_comb_R_df['Slope STD (Normalized) [Rad/MHz]']**2)) / np.sqrt(2)

fosof_lineshape_param_av_comb_t_df['Combiner Uncertainty [MHz]'] = np.abs(fosof_lineshape_param_av_comb_t_df['Zero-crossing Frequency [MHz]'] - rf_comb_I_df['Zero-crossing Frequency [MHz]'])

# We need to shift all of the frequency by 1/2 of the frequency offset
fosof_lineshape_param_av_comb_t_df['Zero-crossing Frequency [MHz]'] = fosof_lineshape_param_av_comb_t_df['Zero-crossing Frequency [MHz]'] + fosof_lineshape_param_av_comb_t_df['Offset Frequency [Hz]'] / 1E6 / 2

fosof_lineshape_param_av_comb_t_df['Reduced Chi-Squared'] = (rf_comb_I_df['Reduced Chi-Squared'] + rf_comb_R_df['Reduced Chi-Squared']) / 2
#%%
# fosof_lineshape_param_av_comb_t_df['Zero-crossing Frequency STD (Normalized) [MHz]'] = fosof_lineshape_param_av_comb_t_df['Zero-crossing Frequency STD (Normalized) [MHz]'] * 1.02

#%%
# Whether to use the corrected or uncorrected data for imperfect power flatness in the waveguides

corrected_Q = True

if corrected_Q:
    corrected = 1
else:
    corrected = 0

zero_cross_av_df = fosof_lineshape_param_av_comb_t_df.loc[('Normal', corrected), slice(None)].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).apply(lambda df: straight_line_fit_params(df['Zero-crossing Frequency [MHz]'], np.sqrt(df['Zero-crossing Frequency STD (Normalized) [MHz]']**2+df['Combiner Uncertainty [MHz]']**2)))


slope_av_df = fosof_lineshape_param_av_comb_t_df.loc[('Normal', corrected), slice(None)].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).apply(lambda df: straight_line_fit_params(df['Slope [Rad/MHz]'], df['Slope STD (Normalized) [Rad/MHz]']))

zero_cross_av_df.rename(columns={'Weighted Mean': 'Zero-crossing Frequency [MHz]', 'Weighted STD': 'Zero-crossing Frequency STD (Normalized) [MHz]', 'Reduced Chi Squared': 'Reduced Chi-Squared'}, inplace=True)

slope_av_df.rename(columns={'Weighted Mean': 'Slope [Rad/MHz]', 'Weighted STD': 'Slope STD (Normalized) [Rad/MHz]'}, inplace=True)
#%%
''' Calculating average chi-squared for the averaged data sets.
'''
# Adding the number of data sets that was averaged for each set of experiment parameters
zero_cross_av_df = zero_cross_av_df.join(fosof_lineshape_param_av_comb_t_df.loc[('Normal', corrected), slice(None)].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).apply(lambda df: pd.Series({'Number Of Points': df.shape[0]})))
#%%
# #==========================================
# # Expanding a single data point by sqrt(chi-squared) MAKE SURE TO REMOVE THIS, WHEN NOT NEEDED
# zero_cross_av_df.loc[(4, 49.86, 8), ('Zero-crossing Frequency STD (Normalized) [MHz]')] = zero_cross_av_df.loc[(4, 49.86, 8), ('Zero-crossing Frequency STD (Normalized) [MHz]')] * np.sqrt(zero_cross_av_df.loc[(4, 49.86, 8), ('Reduced Chi-Squared')])
#
# zero_cross_av_df.loc[(4, 49.86, 8), ('Reduced Chi-Squared')] = 1
# #==========================================
#%%
# In total we determined 18 averages. There are in total 116 data sets, therefore we should have 116-18 = 98 degree os freedom.
dof = np.sum(zero_cross_av_df['Number Of Points']-1)

chi_squared_reduced = np.sum(zero_cross_av_df['Reduced Chi-Squared'] * (zero_cross_av_df['Number Of Points'] - 1)) / dof

chi_squared = chi_squared_reduced*dof

prob_large_chi_squared = scipy.stats.chi2.sf(x=chi_squared, df=dof)
#%%
# 1 sigma prob - check whether the chi-squared distribution is close enough the the gaussian distribution
scipy.stats.chi2.sf(x=dof-np.sqrt(2 * dof), df=dof) - scipy.stats.chi2.sf(x=dof+np.sqrt(2 * dof), df=dof)
#%%
dof
#%%
np.sqrt(2 * dof)
#%%
chi_squared
#%%
(chi_squared - dof)/(np.sqrt(2 * dof))
#%%
prob_large_chi_squared
#%%
# Data for different frequency ranges (Half width, Quarter width, and Eighth width)
zero_cross_av_half_w_df = fosof_lineshape_param_av_comb_t_df.loc[('Half Width', corrected), slice(None)].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).apply(lambda df: straight_line_fit_params(df['Zero-crossing Frequency [MHz]'], np.sqrt(df['Zero-crossing Frequency STD (Normalized) [MHz]']**2+df['Combiner Uncertainty [MHz]']**2))).rename(columns={'Weighted Mean': 'Zero-crossing Frequency [MHz]', 'Weighted STD': 'Zero-crossing Frequency STD (Normalized) [MHz]'})

zero_cross_av_quarter_w_df = fosof_lineshape_param_av_comb_t_df.loc[('Quarter Width', corrected), slice(None)].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).apply(lambda df: straight_line_fit_params(df['Zero-crossing Frequency [MHz]'], np.sqrt(df['Zero-crossing Frequency STD (Normalized) [MHz]']**2+df['Combiner Uncertainty [MHz]']**2))).rename(columns={'Weighted Mean': 'Zero-crossing Frequency [MHz]', 'Weighted STD': 'Zero-crossing Frequency STD (Normalized) [MHz]'})

zero_cross_av_eighth_w_df = fosof_lineshape_param_av_comb_t_df.loc[('Eighth Width', corrected), slice(None)].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).apply(lambda df: straight_line_fit_params(df['Zero-crossing Frequency [MHz]'], np.sqrt(df['Zero-crossing Frequency STD (Normalized) [MHz]']**2+df['Combiner Uncertainty [MHz]']**2))).rename(columns={'Weighted Mean': 'Zero-crossing Frequency [MHz]', 'Weighted STD': 'Zero-crossing Frequency STD (Normalized) [MHz]'})
#%%
'''
AC Shift
'''
travis_corr_df['Beam RMS Radius Frequency Shift [kHz]'] = travis_corr_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * 2

travis_corr_df['Fractional Offset Frequency Shift [kHz]'] = travis_corr_df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * 2

travis_corr_df['Field Power Shift [kHz]'] = travis_corr_df['AC Shift [kHz]'] - travis_corr_df['Beam RMS Radius Frequency Shift [kHz]'] - travis_corr_df['Fractional Offset Frequency Shift [kHz]']

field_power_shift_unc = 0.05

travis_corr_df['AC Shift Uncertainty [kHz]'] = np.sqrt(travis_corr_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + travis_corr_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]']**2)
travis_corr_df['AC Shift Uncertainty [kHz]'] = np.sqrt(travis_corr_df['AC Shift Uncertainty [kHz]']**2 + (travis_corr_df['AC Shift [kHz]'] * field_power_shift_unc)**2)

ac_shift_df = travis_corr_df[['Field Power Shift [kHz]', 'Beam RMS Radius Frequency Shift [kHz]', 'Beam RMS Radius Frequency Shift Uncertainty [kHz]', 'Fractional Offset Frequency Shift [kHz]', 'Fractional Offset Frequency Shift Uncertainty [kHz]', 'AC Shift [kHz]', 'AC Shift Uncertainty [kHz]', 'AC Shift Simulation Uncertainty [kHz]']]

ac_shift_df['AC Shift Simulation Uncertainty [kHz]'] = ac_shift_df['Field Power Shift [kHz]'] * field_power_shift_unc

# Shifts needed for correcting the frequencies in MHz.
ac_shift_MHz_df = ac_shift_df[['AC Shift [kHz]', 'AC Shift Uncertainty [kHz]', 'Fractional Offset Frequency Shift Uncertainty [kHz]', 'AC Shift Simulation Uncertainty [kHz]', 'Beam RMS Radius Frequency Shift Uncertainty [kHz]']]
ac_shift_MHz_df.rename(columns={'AC Shift [kHz]': 'AC Shift [MHz]', 'AC Shift Uncertainty [kHz]': 'AC Shift Uncertainty [MHz]', 'Fractional Offset Frequency Shift Uncertainty [kHz]': 'Fractional Offset Frequency Shift Uncertainty [MHz]', 'AC Shift Simulation Uncertainty [kHz]': 'AC Shift Simulation Uncertainty [MHz]', 'Beam RMS Radius Frequency Shift Uncertainty [kHz]': 'Beam RMS Radius Frequency Shift Uncertainty [MHz]'}, inplace=True)
ac_shift_MHz_df = ac_shift_MHz_df * 1E-3
#%%
'''
Second-order Doppler shift correction
'''
sod_shift_df = travis_corr_df[['SOD Shift [kHz]', 'SOD Shift STD [kHz]']].rename(columns={'SOD Shift [kHz]': 'SOD Shift [MHz]', 'SOD Shift STD [kHz]': 'SOD Shift STD [MHz]'}) / 1E3
sod_shift_df = sod_shift_df.reset_index('Waveguide Electric Field [V/cm]', drop=True)

sod_shift_df = sod_shift_df [~sod_shift_df.index.duplicated(keep='first')]
#%%
''' Shift due to imperfect phase control upon waveguide reversal. This shift was measured to be about delta_phi = 0.2 mrad. In frequency units it corresponds to the frequency shift of delta_phi / slope. Instead of correcting the data, we add this as the additional type of the uncertainty.
'''

# Shift due to imperfect phase control [Rad]
delta_phi = 0.18 * 1E-3

phase_control_unc_df = np.abs(delta_phi / slope_av_df[['Slope [Rad/MHz]']]).rename(columns={'Slope [Rad/MHz]': 'Phase Control Uncertainty [MHz]'}).copy()
#%%
zero_cross_freq_with_unc_df = zero_cross_av_df.copy()
zero_cross_freq_with_unc_half_w_df = zero_cross_av_half_w_df.copy()
zero_cross_freq_with_unc_quarter_w_df = zero_cross_av_quarter_w_df.copy()
zero_cross_freq_with_unc_eighth_w_df = zero_cross_av_eighth_w_df.copy()
# We now want to determine the (blinded) resonant frequencies = zero-crossing frequencies corrected for the systematic shifts and also include all of the systematic uncertainties for each of the determined resonant frequencies.

# Adding the columns to the specific level that will store the systematic shifts
col_name_list = list(ac_shift_MHz_df.columns.union(sod_shift_df.columns))
col_name_list.append('Resonant Frequency (Blinded) [MHz]')

for col_name in col_name_list:

    zero_cross_freq_with_unc_df = zero_cross_freq_with_unc_df.join(zero_cross_av_df[['Zero-crossing Frequency [MHz]']].rename(columns={'Zero-crossing Frequency [MHz]': col_name})).sort_index(axis='columns')

    zero_cross_freq_with_unc_half_w_df = zero_cross_freq_with_unc_half_w_df.join(zero_cross_av_half_w_df[['Zero-crossing Frequency [MHz]']].rename(columns={'Zero-crossing Frequency [MHz]': col_name})).sort_index(axis='columns')

    zero_cross_freq_with_unc_quarter_w_df = zero_cross_freq_with_unc_quarter_w_df.join(zero_cross_av_quarter_w_df[['Zero-crossing Frequency [MHz]']].rename(columns={'Zero-crossing Frequency [MHz]': col_name})).sort_index(axis='columns')

    zero_cross_freq_with_unc_eighth_w_df = zero_cross_freq_with_unc_eighth_w_df.join(zero_cross_av_eighth_w_df[['Zero-crossing Frequency [MHz]']].rename(columns={'Zero-crossing Frequency [MHz]': col_name})).sort_index(axis='columns')

# Addidng the uncertainty due to imperfect phase control
zero_cross_freq_with_unc_df = zero_cross_freq_with_unc_df.join(phase_control_unc_df).sort_index(axis='columns')

zero_cross_freq_with_unc_half_w_df = zero_cross_freq_with_unc_half_w_df.join(phase_control_unc_df).sort_index(axis='columns')

zero_cross_freq_with_unc_quarter_w_df = zero_cross_freq_with_unc_quarter_w_df.join(phase_control_unc_df).sort_index(axis='columns')

zero_cross_freq_with_unc_eighth_w_df = zero_cross_freq_with_unc_eighth_w_df.join(phase_control_unc_df).sort_index(axis='columns')
#%%
def correct_for_sys_shift(df):
    ''' Corrects the zero-crossing frequencies for the systematic shifts, and assign the respective systematic shift uncertainties.
    '''
    df['Resonant Frequency (Blinded) [MHz]'] = df['Resonant Frequency (Blinded) [MHz]'] - ac_shift_MHz_df['AC Shift [MHz]']

    for col_name in ac_shift_MHz_df.columns:
        df[col_name] = ac_shift_MHz_df[col_name]

    df = df.reset_index('Waveguide Electric Field [V/cm]')

    df['Resonant Frequency (Blinded) [MHz]'] = df['Resonant Frequency (Blinded) [MHz]'] + sod_shift_df['SOD Shift [MHz]']

    for col_name in sod_shift_df.columns:
        df[col_name] = sod_shift_df[col_name]

    df.set_index(['Waveguide Electric Field [V/cm]'], append=True, inplace=True)

    return df

res_freq_df = correct_for_sys_shift(zero_cross_freq_with_unc_df.copy())
res_freq_half_w_df = correct_for_sys_shift(zero_cross_freq_with_unc_half_w_df.copy())
res_freq_quarter_w_df = correct_for_sys_shift(zero_cross_freq_with_unc_quarter_w_df.copy())
res_freq_eighth_w_df = correct_for_sys_shift(zero_cross_freq_with_unc_eighth_w_df.copy())
#%%
#res_freq_df['Total Uncertainty [MHz]'] = np.sqrt(res_freq_df['AC Shift Uncertainty [MHz]']**2 + res_freq_df['Phase Control Uncertainty [MHz]']**2 + res_freq_df['SOD Shift STD [MHz]']**2 + res_freq_df['Zero-crossing Frequency STD (Normalized) [MHz]']**2)

#res_freq_half_w_df['Total Uncertainty [MHz]'] = np.sqrt(res_freq_half_w_df['AC Shift Uncertainty [MHz]']**2 + res_freq_half_w_df['Phase Control Uncertainty [MHz]']**2 + res_freq_half_w_df['SOD Shift STD [MHz]']**2 + res_freq_half_w_df['Zero-crossing Frequency STD (Normalized) [MHz]']**2)

#res_freq_quarter_w_df['Total Uncertainty [MHz]'] = np.sqrt(res_freq_quarter_w_df['AC Shift Uncertainty [MHz]']**2 + res_freq_quarter_w_df['Phase Control Uncertainty [MHz]']**2 + res_freq_quarter_w_df['SOD Shift STD [MHz]']**2 + res_freq_quarter_w_df['Zero-crossing Frequency STD (Normalized) [MHz]']**2)

#res_freq_eighth_w_df['Total Uncertainty [MHz]'] = np.sqrt(res_freq_eighth_w_df['AC Shift Uncertainty [MHz]']**2 + res_freq_eighth_w_df['Phase Control Uncertainty [MHz]']**2 + res_freq_eighth_w_df['SOD Shift STD [MHz]']**2 + res_freq_eighth_w_df['Zero-crossing Frequency STD (Normalized) [MHz]']**2)
#%%
''' Calculation of the final average + uncertainty. Here one can select the type of analysis with which the data was averaged.
'''
freq_std_col_name = 'Zero-crossing Frequency STD (Normalized) [MHz]'
freq_std_col_rename = 'Zero-crossing Frequency STD (Normalized) [kHz]'

slope_std_col_name = 'Slope STD (Normalized) [Rad/MHz]'

# Here we select the data needed.
df = res_freq_df

df['Phase Control Shift [MHz]'] = 0

df = df[['AC Shift Uncertainty [MHz]', 'SOD Shift STD [MHz]', 'Phase Control Uncertainty [MHz]', freq_std_col_name, 'AC Shift [MHz]', 'SOD Shift [MHz]', 'Phase Control Shift [MHz]', 'Resonant Frequency (Blinded) [MHz]', 'Fractional Offset Frequency Shift Uncertainty [MHz]', 'AC Shift Simulation Uncertainty [MHz]', 'Beam RMS Radius Frequency Shift Uncertainty [MHz]', 'Reduced Chi-Squared', 'Number Of Points']]

# Converting all of the shifts in to kHz from MHz
df[['AC Shift Uncertainty [MHz]', 'SOD Shift STD [MHz]', 'Phase Control Uncertainty [MHz]', freq_std_col_name, 'AC Shift [MHz]', 'SOD Shift [MHz]', 'Phase Control Shift [MHz]', 'Resonant Frequency (Blinded) [MHz]', 'Fractional Offset Frequency Shift Uncertainty [MHz]', 'AC Shift Simulation Uncertainty [MHz]', 'Beam RMS Radius Frequency Shift Uncertainty [MHz]']] = df[['AC Shift Uncertainty [MHz]', 'SOD Shift STD [MHz]', 'Phase Control Uncertainty [MHz]', freq_std_col_name, 'AC Shift [MHz]', 'SOD Shift [MHz]', 'Phase Control Shift [MHz]', 'Resonant Frequency (Blinded) [MHz]', 'Fractional Offset Frequency Shift Uncertainty [MHz]', 'AC Shift Simulation Uncertainty [MHz]', 'Beam RMS Radius Frequency Shift Uncertainty [MHz]']] * 1E3

df.rename(columns={'AC Shift Uncertainty [MHz]': 'AC Shift Uncertainty [kHz]', 'SOD Shift STD [MHz]': 'SOD Shift STD [kHz]', 'Phase Control Uncertainty [MHz]': 'Phase Control Uncertainty [kHz]', freq_std_col_name: freq_std_col_rename, 'AC Shift [MHz]': 'AC Shift [kHz]', 'SOD Shift [MHz]': 'SOD Shift [kHz]', 'Phase Control Shift [MHz]': 'Phase Control Shift [kHz]', 'Resonant Frequency (Blinded) [MHz]': 'Resonant Frequency (Blinded) [kHz]', 'Fractional Offset Frequency Shift Uncertainty [MHz]': 'Fractional Offset Frequency Shift Uncertainty [kHz]', 'AC Shift Simulation Uncertainty [MHz]': 'AC Shift Simulation Uncertainty [kHz]', 'Beam RMS Radius Frequency Shift Uncertainty [MHz]': 'Beam RMS Radius Frequency Shift Uncertainty [kHz]'}, inplace=True)
#%%
#df['Zero-crossing Frequency STD (Normalized) [kHz]'] = travis_corr_2_df['Statistical Uncertainty [kHz]']

#df['Combiner Uncertainty [kHz]'] = travis_corr_2_df['Combiner Uncertainty [kHz]']

#df['Resonant Frequency (Blinded) [kHz]'] = travis_corr_2_df['Resonant Frequency [kHz]'] - BLIND_OFFSET
#%%
# The weights are first determined for the case when there is no uncertainty due to the phase control. The reason is that before unblinding we did not have such an uncertainty. It is not fair to change the weights AFTER the unblinding was done.
data_to_use_2_df = df

def minimize_unc(w_arr, df):
    w_sum = np.sum(w_arr)

    ac_sim_unc_tot = np.sum(df['AC Shift Simulation Uncertainty [kHz]'] * w_arr) / w_sum
    frac_offset_unc_tot = np.sum(df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum
    beam_rms_rad_unc_tot = np.sum(df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum

    phase_control_unc_tot = np.sum(df['Phase Control Uncertainty [kHz]'] * w_arr) / w_sum

    sod_unc_tot = np.sum(df['SOD Shift STD [kHz]'] * w_arr) / w_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((w_arr*df[freq_std_col_rename])**2) / w_sum**2)

    tot_unc = np.sqrt(ac_sim_unc_tot**2 + frac_offset_unc_tot**2 + beam_rms_rad_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2)

    #tot_unc = np.sqrt(ac_sim_unc_tot**2 + frac_offset_unc_tot**2 + beam_rms_rad_unc_tot**2 + phase_control_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2)

    return tot_unc

def w_tot_constraint(w_arr):
    return np.sum(w_arr) - 1

def find_unc_weights(df):
    #w_arr = np.linspace(1, 10, df.shape[0])
    w_arr = np.ones(df.shape[0])
    w_arr = w_arr/np.sum(w_arr) / 10

    bounds_to_use_list = [(0, 1) for i in range(w_arr.shape[0])]

    w_min_arr = scipy.optimize.minimize(fun=minimize_unc, x0=w_arr, args=(df), bounds=bounds_to_use_list, constraints={'type': 'eq', 'fun': w_tot_constraint}, tol = 1E-15)

    df['Weight'] = w_min_arr['x']

    return df

df_with_weights = df.groupby('Waveguide Separation [cm]').apply(find_unc_weights)

# Using the weights from Travis
df_with_weights['Weight'] = travis_corr_df['Weight']

df_with_weights['Weight'] = df_with_weights['Weight'] / df_with_weights.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().shape[0]

weight_arr_sum = df_with_weights['Weight'].sum()

fract_offset_unc = np.sum(df_with_weights['Fractional Offset Frequency Shift Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

beam_rms_rad_unc = np.sum(df_with_weights['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

ac_sim_unc = np.sum(df_with_weights['AC Shift Simulation Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

field_power_shift_unc_tot = np.sqrt(fract_offset_unc**2 + beam_rms_rad_unc**2 + ac_sim_unc**2)

sod_unc_tot = np.sum(df_with_weights['SOD Shift STD [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

#comb_unc_tot = np.sum(df_with_weights['Combiner Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

zero_cross_stat_unc_tot = np.sqrt(np.sum((df_with_weights['Weight']*df_with_weights[freq_std_col_rename])**2) / weight_arr_sum**2)

tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2)

freq_av = np.sum(df_with_weights['Resonant Frequency (Blinded) [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

AC_shift_av = np.sum(df_with_weights['AC Shift [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

SOD_shift_av = np.sum(df_with_weights['SOD Shift [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

syst_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2)
f0 = freq_av + BLIND_OFFSET

f0_0_s = pd.Series({'Resonant Frequency [kHz]': f0, 'Statistical Uncertainty [kHz]': zero_cross_stat_unc_tot, 'Second-order Doppler Shift [kHz]': SOD_shift_av, 'Second-order Doppler Shift Uncertainty [kHz]': sod_unc_tot, 'AC Shift [kHz]': AC_shift_av, 'AC Shift Uncertainty [kHz]': field_power_shift_unc_tot, 'Total Uncertainty [kHz]': tot_unc, 'Systematic Uncertainty [kHz]': syst_unc})

df_with_weights['Resonant Frequency [kHz]'] = df_with_weights['Resonant Frequency (Blinded) [kHz]'] + BLIND_OFFSET

df_with_weights['Weight [%]'] = df_with_weights['Weight'] * 100
#%%
f0_0_s
#%%
# Using the same weights, but adding the phase control shift + its associated uncertainty.

phase_control_shift_av = np.sum(df_with_weights['Phase Control Shift [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

phase_control_unc_tot = np.sum(df_with_weights['Phase Control Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)

syst_unc = np.sqrt(tot_unc**2 - zero_cross_stat_unc_tot**2)
f0 = freq_av + BLIND_OFFSET

f0_1_s = pd.Series({'Resonant Frequency [kHz]': f0, 'Statistical Uncertainty [kHz]': zero_cross_stat_unc_tot, 'Phase Control Shift [kHz]': phase_control_shift_av, 'Phase Control Uncertainty [kHz]': phase_control_unc_tot, 'Second-order Doppler Shift [kHz]': SOD_shift_av, 'Second-order Doppler Shift Uncertainty [kHz]': sod_unc_tot, 'AC Shift [kHz]': AC_shift_av, 'AC Shift Uncertainty [kHz]': field_power_shift_unc_tot, 'Total Uncertainty [kHz]': tot_unc, 'Systematic Uncertainty [kHz]': syst_unc})

df_with_weights['Systematic Uncertainty [kHz]'] = np.sqrt(df_with_weights['AC Shift Uncertainty [kHz]']**2 + df_with_weights['SOD Shift STD [kHz]']**2 + df_with_weights['Phase Control Uncertainty [kHz]']**2)
#%%
# With chi-squared expansion
# Corrected data: 909871.9, 3.2
# Uncorrected data: 909874.5, 3.2
# Without chi-squared expansion
# Corrected data: 909871.7, 3.2
# Uncorrected data: 909874.3, 3.2
#%%
f0_1_s
#%%
final_weights_df = df_with_weights.copy()
final_weights_df['Total Uncertainty [kHz]'] = np.sqrt(final_weights_df['AC Shift Uncertainty [kHz]']**2 + final_weights_df['SOD Shift STD [kHz]']**2 + final_weights_df['Phase Control Uncertainty [kHz]']**2 + final_weights_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)
#%%
final_weights_df
#%%
final_weights_df
#%%
df_with_weights
#%%
np.sqrt(final_weights_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2 + (final_weights_df['AC Shift [kHz]'] * 0.05)**2)
#%%
straight_line_fit_params(final_weights_df['Resonant Frequency [kHz]'], np.sqrt(final_weights_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2))
#%%
straight_line_fit_params(final_weights_df['Resonant Frequency [kHz]'], np.sqrt(final_weights_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2 + final_weights_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]']**2 + final_weights_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2))
#%%
'''
====================
Plot of the resonant frequency vs inverse slope. This is for the microwave-related phase shifts.
====================
'''
f0_to_use_s = f0_1_s

common_mode_phase_slope_df = data_to_use_2_df.join(slope_av_df[['Slope [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]']])

#common_mode_phase_slope_df = common_mode_phase_slope_df.loc[(slice(None), slice(None), slice(None), slice(None), 1), slice(None)]

common_mode_phase_slope_df['Inverse Slope [kHz/mrad]'] = 1 / common_mode_phase_slope_df['Slope [Rad/MHz]']
common_mode_phase_slope_df['Inverse Slope STD [kHz/mrad]'] = common_mode_phase_slope_df['Slope STD (Normalized) [Rad/MHz]'] / common_mode_phase_slope_df['Slope [Rad/MHz]'] * common_mode_phase_slope_df['Inverse Slope [kHz/mrad]']

common_mode_phase_slope_df['Resonant Frequency [kHz]'] = common_mode_phase_slope_df['Resonant Frequency (Blinded) [kHz]'] + BLIND_OFFSET

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

os.chdir(these_phase_control_folder_path)

# plt_name = 'inverse_slope_t.pdf'
# plt.savefig(plt_name)
plt.show()
#%%
fit_dict
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

#res_freq_for_plotting_df['Total Uncertainty [kHz]'] = np.sqrt(res_freq_for_plotting_df[freq_std_col_rename]**2 + res_freq_for_plotting_df['AC Shift Uncertainty [kHz]']**2)

res_freq_for_plotting_df['Resonant Frequency [kHz]'] = res_freq_for_plotting_df['Resonant Frequency (Blinded) [kHz]'] + BLIND_OFFSET

res_freq_for_plotting_df = res_freq_for_plotting_df.reorder_levels(['Waveguide Electric Field [V/cm]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]']).sort_index().reset_index().reset_index()

index_mean_vs_e_field_list = []
index_color_list = []
for group_name, grouped_df in res_freq_for_plotting_df.set_index('Waveguide Electric Field [V/cm]').groupby('Waveguide Electric Field [V/cm]'):

    min_freq = ((grouped_df['Resonant Frequency [kHz]'] - grouped_df['Total Uncertainty [kHz]']).min())

    max_freq = ((grouped_df['Resonant Frequency [kHz]'] + grouped_df['Total Uncertainty [kHz]']).max())

    index_mean_vs_e_field_list.append([group_name, grouped_df['index'].mean(), grouped_df['index'].max() - grouped_df['index'].min() + 0.4, min_freq-2, max_freq+2, grouped_df.shape[0]])

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
                    y_data_arr = data_fill_style_df['Resonant Frequency [kHz]']
                    y_data_err_arr = data_fill_style_df['Total Uncertainty [kHz]']

                    axes[0].errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, fillstyle=fill_style, capsize=5, capthick=2, linestyle='', markersize=13, markerfacecoloralt='cyan')

                # if face_color != 'full':
                #     ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, markerfacecolor='none', capsize=5, capthick=2, linestyle='', markersize=10)
                # else:
                #     ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, capsize=5, capthick=2, linestyle='', markersize=10)
        #data_marker_df.plot(kind='scatter', x='Inverse Slope [kHz/mrad]', xerr='Inverse Slope STD [kHz/mrad]', y='Resonant Frequency [kHz]', yerr='Total Uncertainty [kHz]', ax=ax, marker=marker, color=data_marker_df['Plot Color'].values)

ax_x_lim = axes[0].get_xlim()

axes[0].ticklabel_format(useOffset=False)

rect_f0 = Rectangle((ax_x_lim[0]-2, (f0_1_s['Resonant Frequency [kHz]']-f0_1_s['Total Uncertainty [kHz]'])), ax_x_lim[1]-ax_x_lim[0]+10, 2*f0_1_s['Total Uncertainty [kHz]'], color='wheat', fill=True, alpha=1)
axes[0].add_patch(rect_f0)

axes[0].set_xlim([-2, ax_x_lim[1]+1])

font_size = 15
for item in ([axes[0].title, axes[0].xaxis.label, axes[0].yaxis.label] +
             axes[0].get_xticklabels() + axes[0].get_yticklabels()):
    item.set_fontsize(font_size)

axes[0].set_ylabel(r'$f_0$ (kHz)')

for i in range(index_mean_vs_e_field_arr.shape[0]):
    arrow_style = '-[, widthB=' + str(index_mean_vs_e_field_arr[i,2]) + ', lengthB='+str(0.3)

    if index_top_bottom[i] == 'top':
        xy=(index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,3])
        xy_text = (index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,3] - 2)
    if index_top_bottom[i] == 'bottom':
        xy=(index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,4])
        xy_text = (index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,4] + 2)

    axes[0].annotate(str(index_mean_vs_e_field_arr[i,0]) + ' V/cm', xy=xy, xytext=xy_text, fontsize=font_size, ha='center', va=index_top_bottom[i], arrowprops=dict(arrowstyle=arrow_style, lw=2.0, color=index_color_list[i]), color=index_color_list[i])

ax_y_lim = axes[0].get_ylim()

ax_y_lim = np.array([ax_y_lim[0], ax_y_lim[1]])

ax_y_lim[0] = ax_y_lim[0] - 7
axes[0].set_ylim(ax_y_lim)
axes[0].set_xticklabels([])
axes[0].set_xticks([])

# Plotting by combining data, for same particular exp parameter, together.

def minimize_unc_2(w_arr, df):
    w_sum = np.sum(w_arr)

    field_power_shift_unc_tot = np.sum(df['AC Shift Uncertainty [kHz]'] * w_arr) / w_sum

    sod_unc_tot = np.sum(df['SOD Shift STD [kHz]'] * w_arr) / w_sum

    phase_control_unc_tot = np.sum(df['Phase Control Uncertainty [kHz]'] * w_arr) / w_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((w_arr*df[freq_std_col_rename])**2) / w_sum**2)

    tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)

    return tot_unc

def find_unc_weights_2(df):
    #w_arr = np.linspace(1, 10, df.shape[0])
    w_arr = np.ones(df.shape[0])
    w_arr = w_arr/np.sum(w_arr) / 10

    bounds_to_use_list = [(0, 1) for i in range(w_arr.shape[0])]

    w_min_arr = scipy.optimize.minimize(fun=minimize_unc_2, x0=w_arr, args=(df), bounds=bounds_to_use_list, constraints={'type': 'eq', 'fun': w_tot_constraint}, tol = 1E-15)

    df['Weight'] = w_min_arr['x']

    return df

def get_res_freq(df_with_weights):
    weight_arr_sum = df_with_weights['Weight'].sum()

    field_power_shift_unc_tot = np.sum(df_with_weights['AC Shift Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

    sod_unc_tot = np.sum(df_with_weights['SOD Shift STD [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

    phase_control_unc_tot = np.sum(df_with_weights['Phase Control Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((df_with_weights['Weight']*df_with_weights[freq_std_col_rename])**2) / weight_arr_sum**2)

    f0 = np.sum(df_with_weights['Resonant Frequency [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

    tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)

    AC_shift_av = np.sum(df_with_weights['AC Shift [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

    SOD_shift_av = np.sum(df_with_weights['SOD Shift [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

    phase_control_shift_av = np.sum(df_with_weights['Phase Control Shift [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

    f0_s = pd.Series({'Resonant Frequency [kHz]': f0, 'Statistical Uncertainty [kHz]': zero_cross_stat_unc_tot, 'Phase Control Shift [kHz]': phase_control_shift_av, 'Phase Control Uncertainty [kHz]': phase_control_unc_tot, 'Second-order Doppler Shift [kHz]': SOD_shift_av, 'Second-order Doppler Shift Uncertainty [kHz]': sod_unc_tot, 'AC Shift [kHz]': AC_shift_av, 'AC Shift Uncertainty [kHz]': field_power_shift_unc_tot, 'Total Uncertainty [kHz]': tot_unc, 'Systematic Uncertainty [kHz]': syst_unc})

    return f0_s


res_freq_vs_e_field_df = res_freq_for_plotting_df.set_index('Waveguide Electric Field [V/cm]').sort_index().groupby('Waveguide Electric Field [V/cm]').apply(find_unc_weights_2).groupby('Waveguide Electric Field [V/cm]').apply(get_res_freq)

res_freq_vs_e_field_df.rename(columns={'Resonant Frequency [kHz]': 'Weighted Mean', 'Total Uncertainty [kHz]': 'Weighted STD'}, inplace=True)

res_freq_vs_acc_volt_df = res_freq_for_plotting_df.set_index('Accelerating Voltage [kV]').sort_index().groupby('Accelerating Voltage [kV]').apply(find_unc_weights_2).groupby('Accelerating Voltage [kV]').apply(get_res_freq)

res_freq_vs_acc_volt_df.rename(columns={'Resonant Frequency [kHz]': 'Weighted Mean', 'Total Uncertainty [kHz]': 'Weighted STD'}, inplace=True)

res_freq_vs_wvg_sep_df = res_freq_for_plotting_df.set_index('Waveguide Separation [cm]').sort_index().groupby('Waveguide Separation [cm]').apply(find_unc_weights_2).groupby('Waveguide Separation [cm]').apply(get_res_freq)

res_freq_vs_wvg_sep_df.rename(columns={'Resonant Frequency [kHz]': 'Weighted Mean', 'Total Uncertainty [kHz]': 'Weighted STD'}, inplace=True)

freq_std_col_name = 'Zero-crossing Frequency STD (Normalized) [MHz]'
freq_std_col_rename = 'Zero-crossing Frequency STD (Normalized) [kHz]'

slope_std_col_name = 'Slope STD (Normalized) [Rad/MHz]'

# Here we select the data corresponding to different frequency ranges
one_range_df = res_freq_df.copy()
one_range_df['Frequency Range [MHz]'] = 4
one_range_df['Weight'] = df_with_weights['Weight']
one_range_df.set_index('Frequency Range [MHz]', append=True, inplace=True)
one_range_df['Phase Control Shift [MHz]'] = 0

half_range_df = res_freq_half_w_df.copy()
half_range_df['Frequency Range [MHz]'] = 2
half_range_df['Weight'] = df_with_weights['Weight']
half_range_df.set_index('Frequency Range [MHz]', append=True, inplace=True)
half_range_df['Phase Control Shift [MHz]'] = 0

quarter_range_df = res_freq_quarter_w_df.copy()
quarter_range_df['Frequency Range [MHz]'] = 1
quarter_range_df['Weight'] = df_with_weights['Weight']
quarter_range_df.set_index('Frequency Range [MHz]', append=True, inplace=True)
quarter_range_df['Phase Control Shift [MHz]'] = 0

eighth_range_df = res_freq_eighth_w_df.copy()
eighth_range_df['Frequency Range [MHz]'] = 0.5
eighth_range_df['Weight'] = df_with_weights['Weight']
eighth_range_df.set_index('Frequency Range [MHz]', append=True, inplace=True)
eighth_range_df['Phase Control Shift [MHz]'] = 0

one_range_df = one_range_df.reorder_levels(['Frequency Range [MHz]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).sort_index()
half_range_df = half_range_df.reorder_levels(['Frequency Range [MHz]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).sort_index()
quarter_range_df = quarter_range_df.reorder_levels(['Frequency Range [MHz]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).sort_index()
eighth_range_df = eighth_range_df.reorder_levels(['Frequency Range [MHz]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).sort_index()

freq_range_df = freq_range_df.reorder_levels(['Frequency Range [MHz]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])

freq_range_df = freq_range_df[['AC Shift Uncertainty [kHz]', 'SOD Shift STD [kHz]', 'Phase Control Uncertainty [kHz]', freq_std_col_rename, 'AC Shift [kHz]', 'SOD Shift [kHz]', 'Phase Control Shift [kHz]', 'Resonant Frequency (Blinded) [kHz]', 'Weight']]

# Converting all of the shifts in to kHz from MHz
#freq_range_df = freq_range_df * 1E3

# freq_range_df.rename(columns={'AC Shift Uncertainty [MHz]': 'AC Shift Uncertainty [kHz]', 'SOD Shift STD [MHz]': 'SOD Shift STD [kHz]', 'Phase Control Uncertainty [MHz]': 'Phase Control Uncertainty [kHz]', freq_std_col_name: freq_std_col_rename, 'AC Shift [MHz]': 'AC Shift [kHz]', 'SOD Shift [MHz]': 'SOD Shift [kHz]', 'Phase Control Shift [MHz]': 'Phase Control Shift [kHz]', 'Resonant Frequency (Blinded) [MHz]': 'Resonant Frequency (Blinded) [kHz]'}, inplace=True)

freq_range_df['Resonant Frequency [kHz]'] = freq_range_df['Resonant Frequency (Blinded) [kHz]'] + BLIND_OFFSET

freq_range_av_df = freq_range_df.groupby('Frequency Range [MHz]').apply(get_res_freq)

freq_range_av_df.rename(columns={'Resonant Frequency [kHz]': 'Weighted Mean', 'Total Uncertainty [kHz]': 'Weighted STD'}, inplace=True)

index_e_field_arr = np.arange(0, res_freq_vs_e_field_df.shape[0])
res_freq_vs_e_field_df['index'] = index_e_field_arr

index_acc_volt_arr = np.arange(0, res_freq_vs_acc_volt_df.shape[0]) + index_e_field_arr[-1] + 1
res_freq_vs_acc_volt_df['index'] = index_acc_volt_arr

index_wvg_sep_arr = np.arange(0, res_freq_vs_wvg_sep_df.shape[0]) + index_acc_volt_arr[-1] + 1
res_freq_vs_wvg_sep_df['index'] = index_wvg_sep_arr

index_freq_range_arr = np.arange(0, freq_range_av_df.shape[0]) + index_wvg_sep_arr[-1] + 1
freq_range_av_df['index'] = index_freq_range_arr

df_list = [res_freq_vs_e_field_df, res_freq_vs_acc_volt_df, res_freq_vs_wvg_sep_df, freq_range_av_df]
group_name_list = [r'$E_0$ (V/cm)', r'$V_\mathrm{HV}$ (kV)', r'$D$ (cm)', r'$f_{\mathrm{max}}-f_{\mathrm{min}}$ (MHz)']

group_color_list = ['green', 'red', 'blue', 'purple', 'blue']

index_mean_vs_e_field_list = []
index_color_list = []

for i in range(len(df_list)):

    grouped_df = df_list[i]
    group_name = group_name_list[i]

    min_freq = ((grouped_df['Weighted Mean'] - grouped_df['Weighted STD']).min())

    max_freq = ((grouped_df['Weighted Mean'] + grouped_df['Weighted STD']).max())

    index_mean_vs_e_field_list.append([0, grouped_df['index'].mean(), 0.7*(grouped_df['index'].max() - grouped_df['index'].min()) + 0.4, min_freq-2, max_freq+2, grouped_df.shape[0]])

    index_color_list.append(group_color_list[i])

index_mean_vs_e_field_arr = np.array(index_mean_vs_e_field_list)
index_top_bottom = ['top', 'top', 'top', 'top']

x_data_arr = res_freq_vs_e_field_df['index']
y_data_arr = res_freq_vs_e_field_df['Weighted Mean']
y_data_err_arr = res_freq_vs_e_field_df['Weighted STD']

axes[1].errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker='.', color='black', capsize=5, capthick=2, linestyle='', markersize=13)

x_data_arr = res_freq_vs_acc_volt_df['index']
y_data_arr = res_freq_vs_acc_volt_df['Weighted Mean']
y_data_err_arr = res_freq_vs_acc_volt_df['Weighted STD']

axes[1].errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker='.', color='black', capsize=5, capthick=2, linestyle='', markersize=13)

x_data_arr = res_freq_vs_wvg_sep_df['index']
y_data_arr = res_freq_vs_wvg_sep_df['Weighted Mean']
y_data_err_arr = res_freq_vs_wvg_sep_df['Weighted STD']

axes[1].errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker='.', color='black', capsize=5, capthick=2, linestyle='', markersize=13)

x_data_arr = freq_range_av_df['index']
y_data_arr = freq_range_av_df['Weighted Mean']
y_data_err_arr = freq_range_av_df['Weighted STD']

axes[1].errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker='.', color='black', capsize=5, capthick=2, linestyle='', markersize=13)

rect_f0 = Rectangle((ax_x_lim[0]-2, (f0_1_s['Resonant Frequency [kHz]']-f0_1_s['Total Uncertainty [kHz]'])), ax_x_lim[1]-ax_x_lim[0]+10, 2*f0_1_s['Total Uncertainty [kHz]'], color='wheat', fill=True, alpha=1)
axes[1].add_patch(rect_f0)

font_size = 15

for i in range(index_mean_vs_e_field_arr.shape[0]):
    arrow_style = '-[, widthB=' + str(index_mean_vs_e_field_arr[i,2]*1.7) + ', lengthB='+str(0.3)

    if index_top_bottom[i] == 'top':
        xy=(index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,3])
        xy_text = (index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,3] - 2)
    if index_top_bottom[i] == 'bottom':
        xy=(index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,4])
        xy_text = (index_mean_vs_e_field_arr[i,1], index_mean_vs_e_field_arr[i,4] + 2)

    axes[1].annotate(group_name_list[i], xy=xy, xytext=xy_text, fontsize=font_size, ha='center', va=index_top_bottom[i], arrowprops=dict(arrowstyle=arrow_style, lw=2.0, color=index_color_list[i]), color=index_color_list[i])


# Annotation for data vs e field amplitude
for i in np.arange(np.min(index_e_field_arr), np.max(index_e_field_arr)+1):
    freq = (res_freq_vs_e_field_df[res_freq_vs_e_field_df['index'] == i]['Weighted Mean'].values[0] + res_freq_vs_e_field_df[res_freq_vs_e_field_df['index'] == i]['Weighted STD'].values[0])

    val = res_freq_vs_e_field_df[res_freq_vs_e_field_df['index'] == i].index.values[0]

    axes[1].annotate(str(int(val)), xy=[i, freq], xytext=[i, freq + 1.25], fontsize=font_size, ha='center', va='bottom', color=index_color_list[0], rotation=90)

# Annotation for data vs accelerating voltage
for i in np.arange(np.min(index_acc_volt_arr), np.max(index_acc_volt_arr)+1):
    freq = (res_freq_vs_acc_volt_df[res_freq_vs_acc_volt_df['index'] == i]['Weighted Mean'].values[0] + res_freq_vs_acc_volt_df[res_freq_vs_acc_volt_df['index'] == i]['Weighted STD'].values[0])

    val = res_freq_vs_acc_volt_df[res_freq_vs_acc_volt_df['index'] == i].index.values[0]

    axes[1].annotate(str(val), xy=[i, freq], xytext=[i, freq + 1.25], fontsize=font_size, ha='center', va='bottom', color=index_color_list[1], rotation=90)

# Annotation for data vs separation
for i in np.arange(np.min(index_wvg_sep_arr), np.max(index_wvg_sep_arr)+1):
    freq = (res_freq_vs_wvg_sep_df[res_freq_vs_wvg_sep_df['index'] == i]['Weighted Mean'].values[0] + res_freq_vs_wvg_sep_df[res_freq_vs_wvg_sep_df['index'] == i]['Weighted STD'].values[0])

    val = res_freq_vs_wvg_sep_df[res_freq_vs_wvg_sep_df['index'] == i].index.values[0]

    axes[1].annotate(str(int(val)), xy=[i, freq], xytext=[i, freq + 1.25], fontsize=font_size, ha='center', va='bottom', color=index_color_list[2], rotation=90)


# Annotation for data vs frequency range
for i in np.arange(np.min(index_freq_range_arr), np.max(index_freq_range_arr)+1):
    freq = (freq_range_av_df[freq_range_av_df['index'] == i]['Weighted Mean'].values[0] + freq_range_av_df[freq_range_av_df['index'] == i]['Weighted STD'].values[0])

    val = freq_range_av_df[freq_range_av_df['index'] == i].index.values[0]
    if int(val) == val:
        val = int(val)
    axes[1].annotate(str(val), xy=[i, freq], xytext=[i, freq + 1.25], fontsize=font_size, ha='center', va='bottom', color=index_color_list[3], rotation=90)

for item in ([axes[1].title, axes[1].xaxis.label, axes[1].yaxis.label] +
             axes[1].get_xticklabels() + axes[1].get_yticklabels()):
    item.set_fontsize(font_size)

#ax.set_ylabel(r'$f_0$ (MHz)')

axes[1].set_ylim(ax_y_lim)

axes[1].set_xticklabels([])
axes[1].set_xticks([])
axes[1].set_yticklabels([])

axes[0].text(0.15, 909847, r'(a)', fontsize=25, horizontalalignment='center')
axes[1].text(0.9, 909847, r'(b)', fontsize=25, horizontalalignment='center')

from matplotlib.ticker import FormatStrFormatter
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

fig.tight_layout()

#os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Thesis\exp_acq')

# plt_name = 'data_consist.pdf'
# plt.savefig(plt_name)


plt.show()
#%%
'''
==================
Plot of reduced chi-squared values for data sets
==================
'''

def gauss_fit_func(x, a, b, sigma, x0):
    return a * np.exp(-(x-x0)**2/(2*sigma**2)) + b

# Number of bins to use for the histogram of reduced chi-squared
bin_n = 10
hist_arr, edge_arr = np.histogram(a=fosof_lineshape_param_av_comb_t_df.loc[('Normal', corrected), slice(None)]['Reduced Chi-Squared'].values, bins=bin_n, density=False)

x_arr = edge_arr[:-1] + np.diff(edge_arr)/2
hist_std_arr = np.sqrt(hist_arr)

# Fit function for the nonlinear least-squares fitting routine
fit, cov_raw = scipy.optimize.curve_fit(f=gauss_fit_func, xdata=x_arr, ydata=hist_arr, p0=(20, 0, 0.3, 1), sigma=hist_std_arr, absolute_sigma=True)

sigma_arr = np.sqrt(np.diag(cov_raw))

x_fit_arr = np.linspace(np.min(x_arr), np.max(x_arr), x_arr.shape[0]*10)
fit_arr = gauss_fit_func(x_fit_arr, *fit)

fig = plt.figure()
fig.set_size_inches(10, 6)

ax = fig.add_subplot(111)

fosof_lineshape_param_av_comb_t_df.loc[('Normal', corrected), slice(None)]['Reduced Chi-Squared'].hist(ax=ax, bins=bin_n, fill=True, facecolor='gray', alpha=1, edgecolor='black')

ax.plot(x_fit_arr, fit_arr, color='blue')
ax.errorbar(x_arr, hist_arr, hist_std_arr, marker='.', capsize=5, capthick=2, linestyle='', markersize=13, color='blue')

ax.grid(False)

ax.set_xlabel(r'$\chi_r^2$')
ax.set_ylabel(r'$N$')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(17)

fig.tight_layout()

# os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Thesis\exp_acq')

# plt_name = 'chi_squared.pdf'
# plt.savefig(plt_name)

plt.show()
#%%
fit
#%%
sigma_arr
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
''' Table for the zero-crossing frequencies used to find the resonant frequency using Travis's data
'''

data_set_data_thesis_df = fosof_lineshape_param_av_comb_t_df.loc[('Normal', 1), slice(None)][['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]', 'Slope [Rad/MHz]', 'Combiner Uncertainty [MHz]', 'Slope STD (Normalized) [Rad/MHz]', 'Reduced Chi-Squared']].reset_index()

data_set_data_thesis_df.rename(columns={'Zero-crossing Frequency [MHz]': 'Zero-crossing Frequency [kHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]': 'Zero-crossing Frequency STD (Normalized) [kHz]', 'Slope [Rad/MHz]': 'Slope [mrad/kHz]', 'Combiner Uncertainty [MHz]': 'Combiner Uncertainty [kHz]', 'Slope STD (Normalized) [Rad/MHz]': 'Slope STD (Normalized) [mrad/kHz]'}, inplace=True)

data_set_data_thesis_df[['Zero-crossing Frequency [kHz]', 'Zero-crossing Frequency STD (Normalized) [kHz]', 'Combiner Uncertainty [kHz]']] = data_set_data_thesis_df[['Zero-crossing Frequency [kHz]', 'Zero-crossing Frequency STD (Normalized) [kHz]', 'Combiner Uncertainty [kHz]']] * 1E3

data_set_data_thesis_df['Zero-crossing Frequency [kHz]'] = data_set_data_thesis_df['Zero-crossing Frequency [kHz]'] + BLIND_OFFSET

data_set_data_thesis_df['Zero-crossing Frequency STD (Normalized) [kHz]'] = np.sqrt(data_set_data_thesis_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2 + data_set_data_thesis_df['Combiner Uncertainty [kHz]']**2)

print('Number of data sets: ' + str(data_set_data_thesis_df.shape[0]) + '\n')

data_set_data_thesis_df['Waveguide Separation [cm]'] = data_set_data_thesis_df['Waveguide Separation [cm]'].astype(np.int16)
data_set_data_thesis_df['Waveguide Electric Field [V/cm]'] = data_set_data_thesis_df['Waveguide Electric Field [V/cm]'].astype(np.int16)

num_arr = data_set_data_thesis_df['Zero-crossing Frequency [kHz]'].values
unc_arr = data_set_data_thesis_df['Combiner Uncertainty [kHz]'].values
data_set_data_thesis_df['Combiner Uncertainty [kHz]'] = for_latex_unc(num_arr, unc_arr, unc_only_Q=True)

num_arr = data_set_data_thesis_df['Zero-crossing Frequency [kHz]'].values
unc_arr = data_set_data_thesis_df['Zero-crossing Frequency STD (Normalized) [kHz]'].values
data_set_data_thesis_df['Zero-crossing Frequency [kHz]'] = for_latex_unc(num_arr, unc_arr)

num_arr = data_set_data_thesis_df['Slope [mrad/kHz]'].values
unc_arr = data_set_data_thesis_df['Slope STD (Normalized) [mrad/kHz]'].values
data_set_data_thesis_df['Slope [mrad/kHz]'] = for_latex_unc(num_arr, unc_arr, u_format_Q=True)

for index, row in data_set_data_thesis_df.iterrows():
    data_set_data_thesis_df.loc[index, ('Reduced Chi-Squared')] = "{:.2f}".format(row['Reduced Chi-Squared'])

data_set_data_thesis_df = data_set_data_thesis_df[['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Zero-crossing Frequency [kHz]', 'Combiner Uncertainty [kHz]', 'Reduced Chi-Squared', 'Slope [mrad/kHz]' ]]
#%%
data_set_data_thesis_df
#%%
os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Thesis\exp_acq')

data_set_data_thesis_df.to_latex(buf='zero_cross_data_sets_t.txt', column_format='lllcccc', multirow=True, escape=False, index=False, index_names=True, longtable=True, header=[r'$D$\,\si{(cm)}', r'$V_{\mathrm{HV}}$\,\si{(\kilo\volt)}', r'$E_0$\,\si{(\volt/\centi\meter)}', r'$f_{zc}$\,\si{(\kilo\hertz)}', r'$\sigma_{f_{zc}}^{\mathrm{(C)}}$\,\si{(\kilo\hertz)}',r'$\chi_{r}^2$', r'$S$\,\si{(\milli\radian/\kilo\hertz)}'])
#%%
''' AC Shift table
'''
# Thesis folder
os.chdir(these_ac_folder_path)

ac_shift_thesis_df = travis_corr_df.copy()
ac_shift_thesis_df = ac_shift_thesis_df[['Field Power Shift [kHz]', 'Beam RMS Radius Frequency Shift [kHz]', 'Beam RMS Radius Frequency Shift Uncertainty [kHz]', 'Fractional Offset Frequency Shift [kHz]', 'Fractional Offset Frequency Shift Uncertainty [kHz]', 'AC Shift [kHz]', 'AC Shift Uncertainty [kHz]']]
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
ac_shift_thesis_df
#%%
ac_shift_thesis_df.to_latex(buf='ac_shift_t.txt', column_format='lllSSSS', multirow=True, escape=False, index=False, index_names=True, header=[r'$D$\,\si{(cm)}', r'$V_{\mathrm{HV}}$\,\si{(\kilo\volt)}', r'$E_0$\,\si{(\volt/\centi\meter)}', r'$\Delta_{\mathrm{AC}}^{(E_0)}$\,\si{(\kilo\hertz)}', r'$\Delta_{\mathrm{AC}}^{(r)}$\,\si{(\kilo\hertz)}', r'$\Delta_{\mathrm{AC}}^{\mathrm{(offset)}}$\,\si{(\kilo\hertz)}', r'$\Delta_{\mathrm{AC}}$\,\si{(\kilo\hertz)}'])
#%%
'''
FINAL WEIGHTS TABLE
'''
final_weights_thesis_df = final_weights_df.copy()

num_arr = final_weights_thesis_df['AC Shift [kHz]'].values
unc_arr = final_weights_thesis_df['AC Shift Uncertainty [kHz]'].values

final_weights_thesis_df['AC Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)

num_arr = final_weights_thesis_df['SOD Shift [kHz]'].values
unc_arr = final_weights_thesis_df['SOD Shift STD [kHz]'].values
final_weights_thesis_df['SOD Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)

num_arr = final_weights_thesis_df['Phase Control Shift [kHz]'].values
unc_arr = final_weights_thesis_df['Phase Control Uncertainty [kHz]'].values
final_weights_thesis_df['Phase Control Shift [kHz]'] = for_latex_unc(num_arr, unc_arr)

for index, row in final_weights_thesis_df.iterrows():
    final_weights_thesis_df.loc[index, ('Weight [%]')] = "{:.1f}".format(row['Weight [%]'])

for index, row in final_weights_thesis_df.iterrows():
    final_weights_thesis_df.loc[index, ('Reduced Chi-Squared')] = "{:.2f}".format(row['Reduced Chi-Squared'])

num_arr = final_weights_thesis_df['Resonant Frequency [kHz]'].values
unc_arr = final_weights_thesis_df['Systematic Uncertainty [kHz]'].values
final_weights_thesis_df['Systematic Uncertainty [kHz]'] = for_latex_unc(num_arr, unc_arr, unc_only_Q=True)

num_arr = final_weights_thesis_df['Resonant Frequency [kHz]'].values
unc_arr = final_weights_thesis_df[freq_std_col_rename].values
final_weights_thesis_df['Resonant Frequency [kHz]'] = for_latex_unc(num_arr, unc_arr)

final_weights_thesis_df['Resonant Frequency [kHz]'] = '$' + final_weights_thesis_df['Resonant Frequency [kHz]'] + r' \pm ' + final_weights_thesis_df['Systematic Uncertainty [kHz]'] + '$'

final_weights_thesis_df = final_weights_thesis_df[['Weight [%]', 'AC Shift [kHz]', 'SOD Shift [kHz]', 'Phase Control Shift [kHz]', 'Reduced Chi-Squared', 'Number Of Points', 'Resonant Frequency [kHz]']]

final_weights_thesis_df.reset_index(inplace=True)

final_weights_thesis_df['Waveguide Separation [cm]'] = final_weights_thesis_df['Waveguide Separation [cm]'].astype(np.int16)
final_weights_thesis_df['Waveguide Electric Field [V/cm]'] = final_weights_thesis_df['Waveguide Electric Field [V/cm]'].astype(np.int16)

#%%
os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Thesis\exp_acq')

final_weights_thesis_df.to_latex(buf='final_weights.txt', column_format='lllcSSSccc', multirow=True, escape=False, index=False, index_names=True, header=[r'$D$\,\si{(cm)}', r'$V_{\mathrm{HV}}$\,\si{(\kilo\volt)}', r'$E_0$\,\si{(\volt/\centi\meter)}', r'Weight (\%)', r'$\Delta_{\mathrm{AC}}$\,\si{(\kilo\hertz)}', r'$\Delta_\mathrm{SOD}$\,\si{(\kilo\hertz)}', r'$\Delta_{c}^{(\mathrm{rf})}$\,\si{(\kilo\hertz)}', r'$\chi^2_{r}$', r'N', r'$f_0 \pm \sigma_\mathrm{stat} \pm \sigma_{\mathrm{sys}}$\,\si{(\kilo\hertz)}'])
#%%
'''
FINAL RESULTS TABLE
'''
f0_thesis_s = f0_1_s.copy()

syst_unc_str = for_latex_unc(np.array([f0_thesis_s['Resonant Frequency [kHz]']]), np.array([f0_thesis_s['Systematic Uncertainty [kHz]']]), unc_only_Q=True)[0]

f0_thesis_s['Resonant Frequency [kHz]'] =  "$" + for_latex_unc(np.array([f0_thesis_s['Resonant Frequency [kHz]']]), np.array([f0_thesis_s['Statistical Uncertainty [kHz]']]))[0] + r" \pm " + syst_unc_str + r"\.\si{\kilo\hertz}$"

f0_thesis_s['Phase Control Shift [kHz]'] =  "$" + for_latex_unc(np.array([f0_thesis_s['Phase Control Shift [kHz]']]), np.array([f0_thesis_s['Phase Control Uncertainty [kHz]']]))[0] + r"$"

f0_thesis_s['Second-order Doppler Shift [kHz]'] = "$" +  for_latex_unc(np.array([f0_thesis_s['Second-order Doppler Shift [kHz]']]), np.array([f0_thesis_s['Second-order Doppler Shift Uncertainty [kHz]']]))[0] + r"$"

f0_thesis_s['AC Shift [kHz]'] =  "$" + for_latex_unc(np.array([f0_thesis_s['AC Shift [kHz]']]), np.array([f0_thesis_s['AC Shift Uncertainty [kHz]']]))[0] + r"$"

f0_thesis_df = pd.DataFrame(f0_thesis_s[['AC Shift [kHz]', 'Second-order Doppler Shift [kHz]', 'Phase Control Shift [kHz]', 'Resonant Frequency [kHz]']]).rename(index={'AC Shift [kHz]': r'AC shift, $\Delta_\mathrm{AC}$',  'Second-order Doppler Shift [kHz]': r'Second-order Doppler shift, $\Delta_\mathrm{SOD}$', 'Phase Control Shift [kHz]': r'Phase error, $\Delta_c^\mathrm{rf}$', 'Resonant Frequency [kHz]': r'$2S_{1/2},\,f=0,\,\rightarrow 2P_{1/2},\,f=1,\,m_f=0$'})

#f0_thesis_df.index.names=['Correction']

f0_thesis_df = f0_thesis_df.reset_index()
#%%
f0_thesis_df['index'].values
#%%
os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Thesis\exp_acq')

f0_thesis_df.to_latex(buf='final_answer.txt', column_format='lr', multirow=True, escape=False, index=False, index_names=False, header=['Correction', r'Weighted average (\si{\kilo\hertz})'])
#%%
'''
==========================
Data analysis by removing 7-cm data
==========================
'''
df_with_weights_no_7cm = df.drop(df.loc[[7]].index).groupby('Waveguide Separation [cm]').apply(find_unc_weights)

# Using the weights from Travis
df_with_weights_no_7cm['Weight'] = travis_corr_df['Weight'].drop(travis_corr_df.loc[[7]].index)

# df_with_weights_no_7cm = df.groupby('Waveguide Separation [cm]').apply(find_unc_weights)
#
# # Using the weights from Travis
# df_with_weights_no_7cm['Weight'] = travis_corr_df['Weight']

df_with_weights_no_7cm['Weight'] = df_with_weights_no_7cm['Weight'] / df_with_weights_no_7cm.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().shape[0]

weight_arr_sum = df_with_weights_no_7cm['Weight'].sum()

fract_offset_unc = np.sum(df_with_weights_no_7cm['Fractional Offset Frequency Shift Uncertainty [kHz]'] * df_with_weights_no_7cm['Weight']) / weight_arr_sum

beam_rms_rad_unc = np.sum(df_with_weights_no_7cm['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * df_with_weights_no_7cm['Weight']) / weight_arr_sum

ac_sim_unc = np.sum(df_with_weights_no_7cm['AC Shift Simulation Uncertainty [kHz]'] * df_with_weights_no_7cm['Weight']) / weight_arr_sum

field_power_shift_unc_tot = np.sqrt(fract_offset_unc**2 + beam_rms_rad_unc**2 + ac_sim_unc**2)

sod_unc_tot = np.sum(df_with_weights_no_7cm['SOD Shift STD [kHz]'] * df_with_weights_no_7cm['Weight']) / weight_arr_sum

#comb_unc_tot = np.sum(df_with_weights['Combiner Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

zero_cross_stat_unc_tot = np.sqrt(np.sum((df_with_weights_no_7cm['Weight']*df_with_weights_no_7cm[freq_std_col_rename])**2) / weight_arr_sum**2)

tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2)

freq_av = np.sum(df_with_weights_no_7cm['Resonant Frequency (Blinded) [kHz]'] * df_with_weights_no_7cm['Weight']) / weight_arr_sum

AC_shift_av = np.sum(df_with_weights_no_7cm['AC Shift [kHz]'] * df_with_weights_no_7cm['Weight']) / weight_arr_sum

SOD_shift_av = np.sum(df_with_weights_no_7cm['SOD Shift [kHz]'] * df_with_weights_no_7cm['Weight']) / weight_arr_sum

syst_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2)
f0 = freq_av + BLIND_OFFSET

f0_0_no_7cm_s = pd.Series({'Resonant Frequency [kHz]': f0, 'Statistical Uncertainty [kHz]': zero_cross_stat_unc_tot, 'Second-order Doppler Shift [kHz]': SOD_shift_av, 'Second-order Doppler Shift Uncertainty [kHz]': sod_unc_tot, 'AC Shift [kHz]': AC_shift_av, 'AC Shift Uncertainty [kHz]': field_power_shift_unc_tot, 'Total Uncertainty [kHz]': tot_unc, 'Systematic Uncertainty [kHz]': syst_unc})

df_with_weights_no_7cm['Resonant Frequency [kHz]'] = df_with_weights_no_7cm['Resonant Frequency (Blinded) [kHz]'] + BLIND_OFFSET

df_with_weights_no_7cm['Weight [%]'] = df_with_weights_no_7cm['Weight'] * 100
#%%
f0_0_no_7cm_s
#%%
# Using the same weights, but adding the phase control shift + its associated uncertainty.

phase_control_shift_av = np.sum(df_with_weights_no_7cm['Phase Control Shift [kHz]'] * df_with_weights_no_7cm['Weight']) / weight_arr_sum

phase_control_unc_tot = np.sum(df_with_weights_no_7cm['Phase Control Uncertainty [kHz]'] * df_with_weights_no_7cm['Weight']) / weight_arr_sum

tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)

syst_unc = np.sqrt(tot_unc**2 - zero_cross_stat_unc_tot**2)
f0 = freq_av + BLIND_OFFSET

f0_1_no_7_cm_s = pd.Series({'Resonant Frequency [kHz]': f0, 'Statistical Uncertainty [kHz]': zero_cross_stat_unc_tot, 'Phase Control Shift [kHz]': phase_control_shift_av, 'Phase Control Uncertainty [kHz]': phase_control_unc_tot, 'Second-order Doppler Shift [kHz]': SOD_shift_av, 'Second-order Doppler Shift Uncertainty [kHz]': sod_unc_tot, 'AC Shift [kHz]': AC_shift_av, 'AC Shift Uncertainty [kHz]': field_power_shift_unc_tot, 'Total Uncertainty [kHz]': tot_unc, 'Systematic Uncertainty [kHz]': syst_unc})

df_with_weights_no_7cm['Systematic Uncertainty [kHz]'] = np.sqrt(df_with_weights_no_7cm['AC Shift Uncertainty [kHz]']**2 + df_with_weights_no_7cm['SOD Shift STD [kHz]']**2 + df_with_weights_no_7cm['Phase Control Uncertainty [kHz]']**2)
#%%
f0_1_no_7_cm_s
#%%
weight_arr_sum
#%%
(fosof_lineshape_param_av_comb_t_df.loc[('Normal', corrected), slice(None)]['Reduced Chi-Squared']  * (39)).hist(bins=10)
#%%
one_range_df = one_range_df.reorder_levels(['Frequency Range [MHz]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).sort_index()
half_range_df = half_range_df.reorder_levels(['Frequency Range [MHz]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).sort_index()
quarter_range_df = quarter_range_df.reorder_levels(['Frequency Range [MHz]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).sort_index()
eighth_range_df = eighth_range_df.reorder_levels(['Frequency Range [MHz]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).sort_index()
#%%
freq_range_df = one_range_df.append(half_range_df).append(quarter_range_df).append(eighth_range_df)
#%%
freq_range_df = freq_range_df.reorder_levels(['Frequency Range [MHz]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])

freq_range_df = freq_range_df[['AC Shift Uncertainty [kHz]', 'SOD Shift STD [kHz]', 'Phase Control Uncertainty [kHz]', freq_std_col_rename, 'AC Shift [kHz]', 'SOD Shift [kHz]', 'Phase Control Shift [kHz]', 'Resonant Frequency (Blinded) [kHz]', 'Weight']]
#%%
'''
========================
Plot of the resonant frequency for different experiment parameters with using different AC shift uncertainty. Only the statistical + previously-used AC shift uncertainties are added in quadrature, to calculate the total uncertainty for each point.
========================
'''

ac_shift_unc_to_try_arr = np.array([[-0.2, 0.2], [0.2, 0.2]])

fig, axes = plt.subplots(nrows=ac_shift_unc_to_try_arr.shape[0], ncols=ac_shift_unc_to_try_arr[0].shape[0])
fig.set_size_inches(22, 16)

for i in range(ac_shift_unc_to_try_arr.shape[0]):
    for j in range(ac_shift_unc_to_try_arr[0].shape[0]):

        ac_shift_unc_to_try = ac_shift_unc_to_try_arr[i, j]

        ax = axes[i, j]

        res_freq_for_plotting_df = final_weights_df.copy()

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

        # res_freq_for_plotting_df['Total Uncertainty [kHz]'] = np.sqrt(res_freq_for_plotting_df[freq_std_col_rename]**2 + res_freq_for_plotting_df['AC Shift Uncertainty [kHz]']**2 + res_freq_for_plotting_df['SOD Shift STD [kHz]']**2 + res_freq_for_plotting_df['Phase Control Uncertainty [kHz]']**2)

        res_freq_for_plotting_df['Total Uncertainty [kHz]'] = np.sqrt(res_freq_for_plotting_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]']**2 + res_freq_for_plotting_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + res_freq_for_plotting_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)
        #res_freq_for_plotting_df['Total Uncertainty [kHz]'] = np.sqrt(res_freq_for_plotting_df[freq_std_col_rename]**2 + res_freq_for_plotting_df['AC Shift Uncertainty [kHz]']**2)

        res_freq_for_plotting_df['Resonant Frequency [kHz]'] = res_freq_for_plotting_df['Resonant Frequency (Blinded) [kHz]'] + BLIND_OFFSET


        res_freq_for_plotting_df['Resonant Frequency [kHz]'] = res_freq_for_plotting_df['Resonant Frequency [kHz]'] + res_freq_for_plotting_df['AC Shift [kHz]'] * ac_shift_unc_to_try


        res_freq_for_plotting_df = res_freq_for_plotting_df.reorder_levels(['Waveguide Electric Field [V/cm]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]']).sort_index().reset_index().reset_index()

        index_mean_vs_e_field_list = []
        index_color_list = []
        for group_name, grouped_df in res_freq_for_plotting_df.set_index('Waveguide Electric Field [V/cm]').groupby('Waveguide Electric Field [V/cm]'):

            min_freq = ((grouped_df['Resonant Frequency [kHz]'] - grouped_df['Total Uncertainty [kHz]']).min())

            max_freq = ((grouped_df['Resonant Frequency [kHz]'] + grouped_df['Total Uncertainty [kHz]']).max())

            index_mean_vs_e_field_list.append([group_name, grouped_df['index'].mean(), grouped_df['index'].max() - grouped_df['index'].min() + 0.4, min_freq-2, max_freq+2, grouped_df.shape[0]])

            index_color_list.append(grouped_df['Plot Marker Edge Color'].values[0])

        index_mean_vs_e_field_arr = np.array(index_mean_vs_e_field_list)
        index_top_bottom = ['top', 'top', 'top', 'top', 'top']

        for marker in list(plot_marker_dict.values()):

            data_marker_df = res_freq_for_plotting_df[res_freq_for_plotting_df['Plot Marker Type'] == marker]

            for color in list(plot_marker_edge_color_dict.values()):
                data_color_df = data_marker_df[data_marker_df['Plot Marker Edge Color'] == color]
                if data_color_df.shape[0] > 0:

                    for fill_style in list(plot_marker_fill_style_dict.values()):

                        data_fill_style_df = data_color_df[data_color_df['Plot Marker Fill Style'] == fill_style]

                        if data_fill_style_df.shape[0] > 0:
                            x_data_arr = data_fill_style_df['index']
                            y_data_arr = data_fill_style_df['Resonant Frequency [kHz]']
                            y_data_err_arr = data_fill_style_df['Total Uncertainty [kHz]']

                            ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, fillstyle=fill_style, capsize=5, capthick=2, linestyle='', markersize=13, markerfacecoloralt='cyan')

                        # if face_color != 'full':
                        #     ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, markerfacecolor='none', capsize=5, capthick=2, linestyle='', markersize=10)
                        # else:
                        #     ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, capsize=5, capthick=2, linestyle='', markersize=10)
                #data_marker_df.plot(kind='scatter', x='Inverse Slope [kHz/mrad]', xerr='Inverse Slope STD [kHz/mrad]', y='Resonant Frequency [kHz]', yerr='Total Uncertainty [kHz]', ax=ax, marker=marker, color=data_marker_df['Plot Color'].values)

        ax_x_lim = ax.get_xlim()

        ax.ticklabel_format(useOffset=False)

        rect_f0 = Rectangle((ax_x_lim[0]-2, (f0_1_s['Resonant Frequency [kHz]']-f0_1_s['Total Uncertainty [kHz]'])), ax_x_lim[1]-ax_x_lim[0]+10, 2*f0_1_s['Total Uncertainty [kHz]'], color='wheat', fill=True, alpha=1)
        ax.add_patch(rect_f0)

        ax.set_xlim([-2, ax_x_lim[1]+1])

        font_size = 15
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(font_size)

        ax.set_ylabel(r'$f_0$ (kHz)')

        for k in range(index_mean_vs_e_field_arr.shape[0]):
            arrow_style = '-[, widthB=' + str(index_mean_vs_e_field_arr[k,2]) + ', lengthB='+str(0.3)

            if index_top_bottom[k] == 'top':
                xy=(index_mean_vs_e_field_arr[k,1], index_mean_vs_e_field_arr[k,3])
                xy_text = (index_mean_vs_e_field_arr[k,1], index_mean_vs_e_field_arr[k,3] - 2)
            if index_top_bottom[k] == 'bottom':
                xy=(index_mean_vs_e_field_arr[k,1], index_mean_vs_e_field_arr[k,4])
                xy_text = (index_mean_vs_e_field_arr[k,1], index_mean_vs_e_field_arr[k,4] + 2)

            ax.annotate(str(index_mean_vs_e_field_arr[k,0]) + ' V/cm', xy=xy, xytext=xy_text, fontsize=font_size, ha='center', va=index_top_bottom[k], arrowprops=dict(arrowstyle=arrow_style, lw=2.0, color=index_color_list[k]), color=index_color_list[k])

        ax_y_lim = ax.get_ylim()

        ax_y_lim = np.array([ax_y_lim[0], ax_y_lim[1]])

        ax_y_lim[0] = ax_y_lim[0] - 7
        ax.set_ylim(ax_y_lim)
        ax.set_xticklabels([])
        ax.set_xticks([])

        ax.set_ylim(909830, 909905)

axes[0, 1].set_yticklabels([])
axes[0, 1].set_yticks([])
axes[0, 1].set_ylabel('')
axes[1, 1].set_yticklabels([])
axes[1, 1].set_yticks([])
axes[1, 1].set_ylabel('')
fig.tight_layout()

# os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Thesis\ac_shift')
#
# plt_name = 'ac_shift_unc.pdf'
# plt.savefig(plt_name)

plt.show()
#%%
def minimize_unc_no_ac_shift(w_arr, df):
    w_sum = np.sum(w_arr)

    frac_offset_unc_tot = np.sum(df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum
    beam_rms_rad_unc_tot = np.sum(df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum

    phase_control_unc_tot = np.sum(df['Phase Control Uncertainty [kHz]'] * w_arr) / w_sum

    sod_unc_tot = np.sum(df['SOD Shift STD [kHz]'] * w_arr) / w_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((w_arr*df[freq_std_col_rename])**2) / w_sum**2)

    tot_unc = np.sqrt(frac_offset_unc_tot**2 + beam_rms_rad_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2 + phase_control_unc_tot**2)

    return tot_unc

def find_unc_weights_no_ac_shift(df):
    #w_arr = np.linspace(1, 10, df.shape[0])
    w_arr = np.ones(df.shape[0])
    w_arr = w_arr/np.sum(w_arr) / 10

    bounds_to_use_list = [(0, 1) for i in range(w_arr.shape[0])]

    w_min_arr = scipy.optimize.minimize(fun=minimize_unc_no_ac_shift, x0=w_arr, args=(df), bounds=bounds_to_use_list, constraints={'type': 'eq', 'fun': w_tot_constraint}, tol = 1E-15)

    df['Weight'] = w_min_arr['x']

    return df

fig, ax = plt.subplots()

fig.set_size_inches(10, 6)

ac_shift_unc_to_try = 0.2


res_freq_for_plotting_df = final_weights_df.copy()

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

# res_freq_for_plotting_df['Total Uncertainty [kHz]'] = np.sqrt(res_freq_for_plotting_df[freq_std_col_rename]**2 + res_freq_for_plotting_df['AC Shift Uncertainty [kHz]']**2 + res_freq_for_plotting_df['SOD Shift STD [kHz]']**2 + res_freq_for_plotting_df['Phase Control Uncertainty [kHz]']**2)

res_freq_for_plotting_df['Total Uncertainty [kHz]'] = np.sqrt(res_freq_for_plotting_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]']**2 + res_freq_for_plotting_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + res_freq_for_plotting_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2 + res_freq_for_plotting_df['Phase Control Uncertainty [kHz]']**2 + res_freq_for_plotting_df['SOD Shift STD [kHz]']**2)
#res_freq_for_plotting_df['Total Uncertainty [kHz]'] = np.sqrt(res_freq_for_plotting_df[freq_std_col_rename]**2 + res_freq_for_plotting_df['AC Shift Uncertainty [kHz]']**2)

res_freq_for_plotting_df['Resonant Frequency [kHz]'] = res_freq_for_plotting_df['Resonant Frequency (Blinded) [kHz]'] + BLIND_OFFSET


res_freq_for_plotting_df['Resonant Frequency [kHz]'] = res_freq_for_plotting_df['Resonant Frequency [kHz]'] + res_freq_for_plotting_df['AC Shift [kHz]'] * ac_shift_unc_to_try


res_freq_for_plotting_df = res_freq_for_plotting_df.reorder_levels(['Waveguide Electric Field [V/cm]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]']).sort_index().reset_index().reset_index()

index_mean_vs_e_field_list = []
index_color_list = []
for group_name, grouped_df in res_freq_for_plotting_df.set_index('Waveguide Electric Field [V/cm]').groupby('Waveguide Electric Field [V/cm]'):

    min_freq = ((grouped_df['Resonant Frequency [kHz]'] - grouped_df['Total Uncertainty [kHz]']).min())

    max_freq = ((grouped_df['Resonant Frequency [kHz]'] + grouped_df['Total Uncertainty [kHz]']).max())

    index_mean_vs_e_field_list.append([group_name, grouped_df['index'].mean(), grouped_df['index'].max() - grouped_df['index'].min() + 0.4, min_freq-2, max_freq+2, grouped_df.shape[0]])

    index_color_list.append(grouped_df['Plot Marker Edge Color'].values[0])

index_mean_vs_e_field_arr = np.array(index_mean_vs_e_field_list)
index_top_bottom = ['top', 'top', 'top', 'top', 'top']

for marker in list(plot_marker_dict.values()):

    data_marker_df = res_freq_for_plotting_df[res_freq_for_plotting_df['Plot Marker Type'] == marker]

    for color in list(plot_marker_edge_color_dict.values()):
        data_color_df = data_marker_df[data_marker_df['Plot Marker Edge Color'] == color]
        if data_color_df.shape[0] > 0:

            for fill_style in list(plot_marker_fill_style_dict.values()):

                data_fill_style_df = data_color_df[data_color_df['Plot Marker Fill Style'] == fill_style]

                if data_fill_style_df.shape[0] > 0:
                    x_data_arr = data_fill_style_df['index']
                    y_data_arr = data_fill_style_df['Resonant Frequency [kHz]']
                    y_data_err_arr = data_fill_style_df['Total Uncertainty [kHz]']

                    ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, fillstyle=fill_style, capsize=5, capthick=2, linestyle='', markersize=13, markerfacecoloralt='cyan')

                # if face_color != 'full':
                #     ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, markerfacecolor='none', capsize=5, capthick=2, linestyle='', markersize=10)
                # else:
                #     ax.errorbar(x_data_arr, y_data_arr, y_data_err_arr, marker=marker, color=color, capsize=5, capthick=2, linestyle='', markersize=10)
        #data_marker_df.plot(kind='scatter', x='Inverse Slope [kHz/mrad]', xerr='Inverse Slope STD [kHz/mrad]', y='Resonant Frequency [kHz]', yerr='Total Uncertainty [kHz]', ax=ax, marker=marker, color=data_marker_df['Plot Color'].values)

ax_x_lim = ax.get_xlim()

ax.ticklabel_format(useOffset=False)

flat_line_fit_params = straight_line_fit_params(res_freq_for_plotting_df['Resonant Frequency [kHz]'], res_freq_for_plotting_df['Total Uncertainty [kHz]'])

# rect_f0 = Rectangle((ax_x_lim[0]-2, flat_line_fit_params['Weighted Mean']-flat_line_fit_params['Weighted STD']), ax_x_lim[1]-ax_x_lim[0]+10, 2*flat_line_fit_params['Weighted STD'], color='darksalmon', fill=True, alpha=1)
# ax.add_patch(rect_f0)

res_freq_for_plotting_with_weights_df = res_freq_for_plotting_df.drop(columns='index').set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).sort_index().groupby('Waveguide Separation [cm]').apply(find_unc_weights_no_ac_shift)

res_freq_for_plotting_with_weights_df['Weight'] = res_freq_for_plotting_with_weights_df['Weight'] / res_freq_for_plotting_with_weights_df.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().shape[0]

weight_arr_sum = res_freq_for_plotting_with_weights_df['Weight'].sum()

fract_offset_unc = np.sum(res_freq_for_plotting_with_weights_df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * res_freq_for_plotting_with_weights_df['Weight']) / weight_arr_sum

beam_rms_rad_unc = np.sum(res_freq_for_plotting_with_weights_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * res_freq_for_plotting_with_weights_df['Weight']) / weight_arr_sum

field_power_shift_unc_tot = np.sqrt(fract_offset_unc**2 + beam_rms_rad_unc**2)

sod_unc_tot = np.sum(res_freq_for_plotting_with_weights_df['SOD Shift STD [kHz]'] * res_freq_for_plotting_with_weights_df['Weight']) / weight_arr_sum

zero_cross_stat_unc_tot = np.sqrt(np.sum((res_freq_for_plotting_with_weights_df['Weight']*res_freq_for_plotting_with_weights_df[freq_std_col_rename])**2) / weight_arr_sum**2)

phase_control_unc_tot = np.sum(df_with_weights['Phase Control Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2 + phase_control_unc_tot**2)

freq_av = np.sum(res_freq_for_plotting_with_weights_df['Resonant Frequency [kHz]'] * res_freq_for_plotting_with_weights_df['Weight']) / weight_arr_sum

dof = res_freq_for_plotting_with_weights_df.shape[0] - 1

chi_squared_reduced = np.sum(((res_freq_for_plotting_with_weights_df['Resonant Frequency [kHz]']-freq_av)/res_freq_for_plotting_with_weights_df['Total Uncertainty [kHz]'])**2)/dof

prob_large_chi_squared = scipy.stats.chi2.sf(x=chi_squared_reduced*dof, df=dof)

print(dof, chi_squared_reduced, prob_large_chi_squared)

rect_f0 = Rectangle((ax_x_lim[0]-2, freq_av-tot_unc), ax_x_lim[1]-ax_x_lim[0]+10, 2*tot_unc, color='darksalmon', fill=True, alpha=0.7)
ax.add_patch(rect_f0)

ax.set_xlim([-2, ax_x_lim[1]+1])

font_size = 15
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(font_size)

ax.set_ylabel(r'$f_0$ (kHz)')

for k in range(index_mean_vs_e_field_arr.shape[0]):
    arrow_style = '-[, widthB=' + str(index_mean_vs_e_field_arr[k,2]) + ', lengthB='+str(0.3)

    if index_top_bottom[k] == 'top':
        xy=(index_mean_vs_e_field_arr[k,1], index_mean_vs_e_field_arr[k,3])
        xy_text = (index_mean_vs_e_field_arr[k,1], index_mean_vs_e_field_arr[k,3] - 2)
    if index_top_bottom[k] == 'bottom':
        xy=(index_mean_vs_e_field_arr[k,1], index_mean_vs_e_field_arr[k,4])
        xy_text = (index_mean_vs_e_field_arr[k,1], index_mean_vs_e_field_arr[k,4] + 2)

    ax.annotate(str(index_mean_vs_e_field_arr[k,0]) + ' V/cm', xy=xy, xytext=xy_text, fontsize=font_size, ha='center', va=index_top_bottom[k], arrowprops=dict(arrowstyle=arrow_style, lw=2.0, color=index_color_list[k]), color=index_color_list[k])

ax_y_lim = ax.get_ylim()

ax_y_lim = np.array([ax_y_lim[0], ax_y_lim[1]])

ax_y_lim[0] = ax_y_lim[0] - 7
ax.set_ylim(ax_y_lim)
ax.set_xticklabels([])
ax.set_xticks([])

ax.set_ylim(909850, 909917)

fig.tight_layout()

os.chdir(these_ac_folder_path)

plt_name = 'ac_shift_err_20.pdf'
plt.savefig(plt_name)

plt.show()
#%%
res_freq_for_plotting_df
#%%
res_freq_for_plotting_with_weights_df
#%%
flat_line_fit_params
#%%
ac_shift_unc_df = final_weights_df.copy()
ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

# ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

#%%
ac_shift_unc_df.reset_index().reset_index().plot(kind='scatter', x='index', y='Resonant Frequency [kHz]', yerr='Total Uncertainty [kHz]')

#%%
ac_shift_unc_to_try_arr = np.linspace(-0.2, 0.2, 201)
index_0 = np.where(ac_shift_unc_to_try_arr == 0)

chi_squared_dev_arr = np.zeros(ac_shift_unc_to_try_arr.shape[0])
prob_arr = np.zeros(ac_shift_unc_to_try_arr.shape[0])
# Full uncertainty

ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

for i in range(ac_shift_unc_to_try_arr.shape[0]):
    ac_shift_unc_to_try = ac_shift_unc_to_try_arr[i]

    av_ac_shift_study = straight_line_fit_params(ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]'], ac_shift_unc_df['Total Uncertainty [kHz]'])

    dof = ac_shift_unc_df['Resonant Frequency [kHz]'].shape[0]-1
    sigma1_prob = scipy.stats.chi2.sf(x=dof, df=dof) - scipy.stats.chi2.sf(x=dof+np.sqrt(2 * dof), df=dof)

    prob_arr[i] = scipy.stats.chi2.sf(x=av_ac_shift_study['Reduced Chi Squared'] * dof, df=dof)

    chi_squared_dev_arr[i] = (av_ac_shift_study['Reduced Chi Squared'] * dof - dof) / np.sqrt(2 * dof)

three_unc_chi_squared_dev_arr = chi_squared_dev_arr.copy()
three_prob_arr = prob_arr.copy()

# Two source of uncertainty (No Beam RMS Radius uncertainty)
ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

for i in range(ac_shift_unc_to_try_arr.shape[0]):
    ac_shift_unc_to_try = ac_shift_unc_to_try_arr[i]
    av_ac_shift_study = straight_line_fit_params(ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]'], ac_shift_unc_df['Total Uncertainty [kHz]'])

    dof = ac_shift_unc_df['Resonant Frequency [kHz]'].shape[0]-1
    sigma1_prob = scipy.stats.chi2.sf(x=dof, df=dof) - scipy.stats.chi2.sf(x=dof+np.sqrt(2 * dof), df=dof)

    prob_arr[i] = scipy.stats.chi2.sf(x=av_ac_shift_study['Reduced Chi Squared'] * dof, df=dof)

    chi_squared_dev_arr[i] = (av_ac_shift_study['Reduced Chi Squared'] * dof - dof) / np.sqrt(2 * dof)

two_unc_chi_squared_dev_arr = chi_squared_dev_arr.copy()
two_prob_arr = prob_arr.copy()

# Only statistical uncertainty
ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

for i in range(ac_shift_unc_to_try_arr.shape[0]):
    ac_shift_unc_to_try = ac_shift_unc_to_try_arr[i]
    av_ac_shift_study = straight_line_fit_params(ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]'], ac_shift_unc_df['Total Uncertainty [kHz]'])

    dof = ac_shift_unc_df['Resonant Frequency [kHz]'].shape[0]-1
    sigma1_prob = scipy.stats.chi2.sf(x=dof, df=dof) - scipy.stats.chi2.sf(x=dof+np.sqrt(2 * dof), df=dof)

    prob_arr[i] = scipy.stats.chi2.sf(x=av_ac_shift_study['Reduced Chi Squared'] * dof, df=dof)

    chi_squared_dev_arr[i] = (av_ac_shift_study['Reduced Chi Squared'] * dof - dof) / np.sqrt(2 * dof)

one_unc_chi_squared_dev_arr = chi_squared_dev_arr.copy()
one_prob_arr = prob_arr.copy()
#%%
prob_1_sigma = scipy.stats.chi2.sf(x=dof+np.sqrt(2 * dof), df=dof)
#%%
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
ax.plot(ac_shift_unc_to_try_arr*100, one_prob_arr/one_prob_arr[index_0], color='black')

ax.plot(ac_shift_unc_to_try_arr*100, two_prob_arr/two_prob_arr[index_0], color='red')
ax.plot(ac_shift_unc_to_try_arr*100, three_prob_arr/three_prob_arr[index_0], color='blue')

ax.set_xlabel(r'$\alpha_{\mathrm{AC}}$ (%)')
ax.set_ylabel(r'$P(\alpha_{\mathrm{AC}})\;/\;P(0)$')
ax.set_xlim(-10, 20)

font_size = 15
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(font_size)
y_ax_lim = ax.get_ylim()

rect_unc = Rectangle((-5, -5), 10, 10, color='darkorange', fill=True, alpha=0.3)
ax.add_patch(rect_unc)

ax.set_ylim(y_ax_lim)
fig.tight_layout()

# os.chdir(these_ac_folder_path)
#
# plt_name = 'ac_shift_unc.pdf'
# plt.savefig(plt_name)
plt.show()
#%%
fig, ax = plt.subplots()

ax.plot(ac_shift_unc_to_try_arr*100, one_unc_chi_squared_dev_arr/one_unc_chi_squared_dev_arr [index_0], color='black')

ax.plot(ac_shift_unc_to_try_arr*100, two_unc_chi_squared_dev_arr/two_unc_chi_squared_dev_arr[index_0], color='red')
ax.plot(ac_shift_unc_to_try_arr*100, three_unc_chi_squared_dev_arr/three_unc_chi_squared_dev_arr[index_0], color='blue')2

plt.show()
#%%
ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

e0_24_df = ac_shift_unc_df.loc[(slice(None), 49.86, (14)), slice(None)]
e0_5_8_df = ac_shift_unc_df.loc[(slice(None), 49.86, (5)), slice(None)]
#e0_5_8_df = ac_shift_unc_df.loc[(slice(None), slice(None), (8)), slice(None)]
ac_shift_unc_to_try_arr = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])

dev_arr = np.zeros(ac_shift_unc_to_try_arr.shape[0])
dev_std_arr = np.zeros(ac_shift_unc_to_try_arr.shape[0])

for i in range(ac_shift_unc_to_try_arr.shape[0]):
    ac_shift_unc_to_try = ac_shift_unc_to_try_arr[i]

    e0_5_8_av_shift = straight_line_fit_params(e0_5_8_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * e0_5_8_df['AC Shift [kHz]'], e0_5_8_df['Total Uncertainty [kHz]'])

    e0_24_av_shift = straight_line_fit_params(e0_24_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * e0_24_df['AC Shift [kHz]'], e0_24_df['Total Uncertainty [kHz]'])

    dev_arr[i] = e0_24_av_shift['Weighted Mean'] - e0_5_8_av_shift['Weighted Mean']

    # dev_arr[i] = np.average(e0_24_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * e0_24_df['AC Shift [kHz]']) - np.average(e0_5_8_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * e0_5_8_df['AC Shift [kHz]'])

    dev_std_arr[i] = np.sqrt(e0_24_av_shift['Weighted STD']**2 + e0_5_8_av_shift['Weighted STD']**2)
#%%
fig, ax = plt.subplots()

ax.plot(ac_shift_unc_to_try_arr, dev_arr, color='black')
ax.plot(ac_shift_unc_to_try_arr, dev_arr + dev_std_arr, color='red')
ax.plot(ac_shift_unc_to_try_arr, dev_arr - dev_std_arr, color='red')
plt.show()
#%%
fit_param = np.polyfit(ac_shift_unc_to_try_arr, dev_arr, 1)
zero_dev = fit_param[1]
slope_dev = fit_param[0]

(dev_arr[0] + dev_std_arr[0] - zero_dev) / slope_dev
#%%

#%%
(0 - zero_dev) / slope_dev
#%%
np.mean(np.array([0.0224, 0.107, 0.044]))
#%%
straight_line_fit_params(e0_5_8_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * e0_5_8_df['AC Shift [kHz]'], e0_5_8_df['Total Uncertainty [kHz]'])
#%%
np.average(e0_5_8_df['Resonant Frequency [kHz]'])
#%%
straight_line_fit_params(e0_24_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * e0_24_df['AC Shift [kHz]'], e0_24_df['Total Uncertainty [kHz]'])
#%%
e0_24_df
#%%
ac_shift_unc_to_try_arr = np.linspace(-0.2, 0.2, 101)
index_0 = np.where(ac_shift_unc_to_try_arr == 0)
#%%
chi_squared_dev_arr = np.zeros(ac_shift_unc_to_try_arr.shape[0])
prob_arr = np.zeros(ac_shift_unc_to_try_arr.shape[0])
# Full uncertainty

def minimize_unc_no_ac_shift(w_arr, df):
    w_sum = np.sum(w_arr)

    frac_offset_unc_tot = np.sum(df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum
    beam_rms_rad_unc_tot = np.sum(df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum

    #phase_control_unc_tot = np.sum(df['Phase Control Uncertainty [kHz]'] * w_arr) / w_sum

    sod_unc_tot = np.sum(df['SOD Shift STD [kHz]'] * w_arr) / w_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((w_arr*df[freq_std_col_rename])**2) / w_sum**2)

    tot_unc = np.sqrt(frac_offset_unc_tot**2 + beam_rms_rad_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2)

    return tot_unc

def find_unc_weights_no_ac_shift(df):
    #w_arr = np.linspace(1, 10, df.shape[0])
    w_arr = np.ones(df.shape[0])
    w_arr = w_arr/np.sum(w_arr) / 10

    bounds_to_use_list = [(0, 1) for i in range(w_arr.shape[0])]

    w_min_arr = scipy.optimize.minimize(fun=minimize_unc_no_ac_shift, x0=w_arr, args=(df), bounds=bounds_to_use_list, constraints={'type': 'eq', 'fun': w_tot_constraint}, tol = 1E-15)

    df['Weight'] = w_min_arr['x']

    return df


ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2 + ac_shift_unc_df['Phase Control Uncertainty [kHz]']**2 + ac_shift_unc_df['SOD Shift STD [kHz]']**2)

for i in range(ac_shift_unc_to_try_arr.shape[0]):
    ac_shift_unc_to_try = ac_shift_unc_to_try_arr[i]

    av_ac_shift_study = straight_line_fit_params(ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]'], ac_shift_unc_df['Total Uncertainty [kHz]'])

    dof = ac_shift_unc_df['Resonant Frequency [kHz]'].shape[0]-1
    sigma1_prob = scipy.stats.chi2.sf(x=dof, df=dof) - scipy.stats.chi2.sf(x=dof+np.sqrt(2 * dof), df=dof)

    prob_arr[i] = scipy.stats.chi2.sf(x=av_ac_shift_study['Reduced Chi Squared'] * dof, df=dof)

    chi_squared_dev_arr[i] = (av_ac_shift_study['Reduced Chi Squared'] * dof - dof) / np.sqrt(2 * dof)

    ac_shift_unc_df_shifted = ac_shift_unc_df.copy()
    ac_shift_unc_df_shifted['Resonant Frequency [kHz]'] = ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]']

    ac_shift_unc_with_weights_df = ac_shift_unc_df_shifted.groupby('Waveguide Separation [cm]').apply(find_unc_weights_no_ac_shift)

    ac_shift_unc_with_weights_df['Weight'] = ac_shift_unc_with_weights_df['Weight'] / ac_shift_unc_with_weights_df.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().shape[0]

    weight_arr_sum = ac_shift_unc_with_weights_df['Weight'].sum()

    fract_offset_unc = np.sum(ac_shift_unc_with_weights_df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    beam_rms_rad_unc = np.sum(ac_shift_unc_with_weights_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    field_power_shift_unc_tot = np.sqrt(fract_offset_unc**2 + beam_rms_rad_unc**2)

    sod_unc_tot = np.sum(ac_shift_unc_with_weights_df['SOD Shift STD [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((ac_shift_unc_with_weights_df['Weight']*ac_shift_unc_with_weights_df[freq_std_col_rename])**2) / weight_arr_sum**2)

    phase_control_unc_tot = np.sum(df_with_weights['Phase Control Uncertainty [kHz]'] * df_with_weights['Weight']) / weight_arr_sum

    tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2 + phase_control_unc_tot**2)

    freq_av = np.sum(ac_shift_unc_with_weights_df['Resonant Frequency [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    dof = ac_shift_unc_with_weights_df.shape[0] - 1

    chi_squared_reduced = np.sum(((ac_shift_unc_with_weights_df['Resonant Frequency [kHz]']-freq_av)/ac_shift_unc_with_weights_df['Total Uncertainty [kHz]'])**2)/dof

    prob_large_chi_squared = scipy.stats.chi2.sf(x=chi_squared_reduced*dof, df=dof)

    prob_arr[i] = prob_large_chi_squared

    chi_squared_dev_arr[i] = (chi_squared_reduced * dof - dof) / np.sqrt(2 * dof)

one_unc_chi_squared_dev_arr = chi_squared_dev_arr.copy()
one_prob_arr = prob_arr.copy()

# Two source of uncertainty (No phase control uncertainty)
def minimize_unc_no_ac_shift(w_arr, df):
    w_sum = np.sum(w_arr)

    frac_offset_unc_tot = np.sum(df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum
    beam_rms_rad_unc_tot = np.sum(df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum

    phase_control_unc_tot = np.sum(df['Phase Control Uncertainty [kHz]'] * w_arr) / w_sum

    sod_unc_tot = np.sum(df['SOD Shift STD [kHz]'] * w_arr) / w_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((w_arr*df[freq_std_col_rename])**2) / w_sum**2)

    tot_unc = np.sqrt(frac_offset_unc_tot**2 + beam_rms_rad_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2)

    return tot_unc

def find_unc_weights_no_ac_shift(df):
    #w_arr = np.linspace(1, 10, df.shape[0])
    w_arr = np.ones(df.shape[0])
    w_arr = w_arr/np.sum(w_arr) / 10

    bounds_to_use_list = [(0, 1) for i in range(w_arr.shape[0])]

    w_min_arr = scipy.optimize.minimize(fun=minimize_unc_no_ac_shift, x0=w_arr, args=(df), bounds=bounds_to_use_list, constraints={'type': 'eq', 'fun': w_tot_constraint}, tol = 1E-15)

    df['Weight'] = w_min_arr['x']

    return df


ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2 + ac_shift_unc_df['SOD Shift STD [kHz]']**2)

for i in range(ac_shift_unc_to_try_arr.shape[0]):
    ac_shift_unc_to_try = ac_shift_unc_to_try_arr[i]

    av_ac_shift_study = straight_line_fit_params(ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]'], ac_shift_unc_df['Total Uncertainty [kHz]'])

    dof = ac_shift_unc_df['Resonant Frequency [kHz]'].shape[0]-1
    sigma1_prob = scipy.stats.chi2.sf(x=dof, df=dof) - scipy.stats.chi2.sf(x=dof+np.sqrt(2 * dof), df=dof)

    prob_arr[i] = scipy.stats.chi2.sf(x=av_ac_shift_study['Reduced Chi Squared'] * dof, df=dof)

    chi_squared_dev_arr[i] = (av_ac_shift_study['Reduced Chi Squared'] * dof - dof) / np.sqrt(2 * dof)

    ac_shift_unc_df_shifted = ac_shift_unc_df.copy()
    ac_shift_unc_df_shifted['Resonant Frequency [kHz]'] = ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]']

    ac_shift_unc_with_weights_df = ac_shift_unc_df_shifted.groupby('Waveguide Separation [cm]').apply(find_unc_weights_no_ac_shift)

    ac_shift_unc_with_weights_df['Weight'] = ac_shift_unc_with_weights_df['Weight'] / ac_shift_unc_with_weights_df.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().shape[0]

    weight_arr_sum = ac_shift_unc_with_weights_df['Weight'].sum()

    fract_offset_unc = np.sum(ac_shift_unc_with_weights_df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    beam_rms_rad_unc = np.sum(ac_shift_unc_with_weights_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    field_power_shift_unc_tot = np.sqrt(fract_offset_unc**2 + beam_rms_rad_unc**2)

    sod_unc_tot = np.sum(ac_shift_unc_with_weights_df['SOD Shift STD [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((ac_shift_unc_with_weights_df['Weight']*ac_shift_unc_with_weights_df[freq_std_col_rename])**2) / weight_arr_sum**2)

    tot_unc = np.sqrt(field_power_shift_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2)

    freq_av = np.sum(ac_shift_unc_with_weights_df['Resonant Frequency [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    dof = ac_shift_unc_with_weights_df.shape[0] - 1

    chi_squared_reduced = np.sum(((ac_shift_unc_with_weights_df['Resonant Frequency [kHz]']-freq_av)/ac_shift_unc_with_weights_df['Total Uncertainty [kHz]'])**2)/dof

    prob_large_chi_squared = scipy.stats.chi2.sf(x=chi_squared_reduced*dof, df=dof)

    prob_arr[i] = prob_large_chi_squared

    chi_squared_dev_arr[i] = (chi_squared_reduced * dof - dof) / np.sqrt(2 * dof)

two_unc_chi_squared_dev_arr = chi_squared_dev_arr.copy()
two_prob_arr = prob_arr.copy()

# Three source of uncertainty (no SOD Shift uncertainty)
def minimize_unc_no_ac_shift(w_arr, df):
    w_sum = np.sum(w_arr)

    frac_offset_unc_tot = np.sum(df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum
    beam_rms_rad_unc_tot = np.sum(df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum

    phase_control_unc_tot = np.sum(df['Phase Control Uncertainty [kHz]'] * w_arr) / w_sum

    sod_unc_tot = np.sum(df['SOD Shift STD [kHz]'] * w_arr) / w_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((w_arr*df[freq_std_col_rename])**2) / w_sum**2)

    #tot_unc = np.sqrt(frac_offset_unc_tot**2 + beam_rms_rad_unc_tot**2 + sod_unc_tot**2 + zero_cross_stat_unc_tot**2)

    tot_unc = np.sqrt(frac_offset_unc_tot**2 + beam_rms_rad_unc_tot**2 + zero_cross_stat_unc_tot**2)

    return tot_unc

def find_unc_weights_no_ac_shift(df):
    #w_arr = np.linspace(1, 10, df.shape[0])
    w_arr = np.ones(df.shape[0])
    w_arr = w_arr/np.sum(w_arr) / 10

    bounds_to_use_list = [(0, 1) for i in range(w_arr.shape[0])]

    w_min_arr = scipy.optimize.minimize(fun=minimize_unc_no_ac_shift, x0=w_arr, args=(df), bounds=bounds_to_use_list, constraints={'type': 'eq', 'fun': w_tot_constraint}, tol = 1E-15)

    df['Weight'] = w_min_arr['x']

    return df


ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

for i in range(ac_shift_unc_to_try_arr.shape[0]):
    ac_shift_unc_to_try = ac_shift_unc_to_try_arr[i]

    av_ac_shift_study = straight_line_fit_params(ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]'], ac_shift_unc_df['Total Uncertainty [kHz]'])

    dof = ac_shift_unc_df['Resonant Frequency [kHz]'].shape[0]-1
    sigma1_prob = scipy.stats.chi2.sf(x=dof, df=dof) - scipy.stats.chi2.sf(x=dof+np.sqrt(2 * dof), df=dof)

    prob_arr[i] = scipy.stats.chi2.sf(x=av_ac_shift_study['Reduced Chi Squared'] * dof, df=dof)

    chi_squared_dev_arr[i] = (av_ac_shift_study['Reduced Chi Squared'] * dof - dof) / np.sqrt(2 * dof)

    ac_shift_unc_df_shifted = ac_shift_unc_df.copy()
    ac_shift_unc_df_shifted['Resonant Frequency [kHz]'] = ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]']

    ac_shift_unc_with_weights_df = ac_shift_unc_df_shifted.groupby('Waveguide Separation [cm]').apply(find_unc_weights_no_ac_shift)

    ac_shift_unc_with_weights_df['Weight'] = ac_shift_unc_with_weights_df['Weight'] / ac_shift_unc_with_weights_df.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().shape[0]

    weight_arr_sum = ac_shift_unc_with_weights_df['Weight'].sum()

    fract_offset_unc = np.sum(ac_shift_unc_with_weights_df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    beam_rms_rad_unc = np.sum(ac_shift_unc_with_weights_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    field_power_shift_unc_tot = np.sqrt(fract_offset_unc**2 + beam_rms_rad_unc**2)

    sod_unc_tot = np.sum(ac_shift_unc_with_weights_df['SOD Shift STD [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((ac_shift_unc_with_weights_df['Weight']*ac_shift_unc_with_weights_df[freq_std_col_rename])**2) / weight_arr_sum**2)

    tot_unc = np.sqrt(field_power_shift_unc_tot**2 + zero_cross_stat_unc_tot**2)

    freq_av = np.sum(ac_shift_unc_with_weights_df['Resonant Frequency [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    dof = ac_shift_unc_with_weights_df.shape[0] - 1

    chi_squared_reduced = np.sum(((ac_shift_unc_with_weights_df['Resonant Frequency [kHz]']-freq_av)/ac_shift_unc_with_weights_df['Total Uncertainty [kHz]'])**2)/dof

    prob_large_chi_squared = scipy.stats.chi2.sf(x=chi_squared_reduced*dof, df=dof)

    prob_arr[i] = prob_large_chi_squared

    chi_squared_dev_arr[i] = (chi_squared_reduced * dof - dof) / np.sqrt(2 * dof)

three_unc_chi_squared_dev_arr = chi_squared_dev_arr.copy()
three_prob_arr = prob_arr.copy()

# Four source of uncertainty (no Beam RMS radius uncertainty)
def minimize_unc_no_ac_shift(w_arr, df):
    w_sum = np.sum(w_arr)

    frac_offset_unc_tot = np.sum(df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum
    beam_rms_rad_unc_tot = np.sum(df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum

    phase_control_unc_tot = np.sum(df['Phase Control Uncertainty [kHz]'] * w_arr) / w_sum

    sod_unc_tot = np.sum(df['SOD Shift STD [kHz]'] * w_arr) / w_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((w_arr*df[freq_std_col_rename])**2) / w_sum**2)

    tot_unc = np.sqrt(frac_offset_unc_tot**2 + zero_cross_stat_unc_tot**2)

    return tot_unc

def find_unc_weights_no_ac_shift(df):
    #w_arr = np.linspace(1, 10, df.shape[0])
    w_arr = np.ones(df.shape[0])
    w_arr = w_arr/np.sum(w_arr) / 10

    bounds_to_use_list = [(0, 1) for i in range(w_arr.shape[0])]

    w_min_arr = scipy.optimize.minimize(fun=minimize_unc_no_ac_shift, x0=w_arr, args=(df), bounds=bounds_to_use_list, constraints={'type': 'eq', 'fun': w_tot_constraint}, tol = 1E-15)

    df['Weight'] = w_min_arr['x']

    return df


ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Fractional Offset Frequency Shift Uncertainty [kHz]']**2 + ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

for i in range(ac_shift_unc_to_try_arr.shape[0]):
    ac_shift_unc_to_try = ac_shift_unc_to_try_arr[i]

    av_ac_shift_study = straight_line_fit_params(ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]'], ac_shift_unc_df['Total Uncertainty [kHz]'])

    dof = ac_shift_unc_df['Resonant Frequency [kHz]'].shape[0]-1
    sigma1_prob = scipy.stats.chi2.sf(x=dof, df=dof) - scipy.stats.chi2.sf(x=dof+np.sqrt(2 * dof), df=dof)

    prob_arr[i] = scipy.stats.chi2.sf(x=av_ac_shift_study['Reduced Chi Squared'] * dof, df=dof)

    chi_squared_dev_arr[i] = (av_ac_shift_study['Reduced Chi Squared'] * dof - dof) / np.sqrt(2 * dof)

    ac_shift_unc_df_shifted = ac_shift_unc_df.copy()
    ac_shift_unc_df_shifted['Resonant Frequency [kHz]'] = ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]']

    ac_shift_unc_with_weights_df = ac_shift_unc_df_shifted.groupby('Waveguide Separation [cm]').apply(find_unc_weights_no_ac_shift)

    ac_shift_unc_with_weights_df['Weight'] = ac_shift_unc_with_weights_df['Weight'] / ac_shift_unc_with_weights_df.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().shape[0]

    weight_arr_sum = ac_shift_unc_with_weights_df['Weight'].sum()

    fract_offset_unc = np.sum(ac_shift_unc_with_weights_df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    beam_rms_rad_unc = np.sum(ac_shift_unc_with_weights_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    sod_unc_tot = np.sum(ac_shift_unc_with_weights_df['SOD Shift STD [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((ac_shift_unc_with_weights_df['Weight']*ac_shift_unc_with_weights_df[freq_std_col_rename])**2) / weight_arr_sum**2)

    tot_unc = np.sqrt(fract_offset_unc**2 + zero_cross_stat_unc_tot**2)

    freq_av = np.sum(ac_shift_unc_with_weights_df['Resonant Frequency [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    dof = ac_shift_unc_with_weights_df.shape[0] - 1

    chi_squared_reduced = np.sum(((ac_shift_unc_with_weights_df['Resonant Frequency [kHz]']-freq_av)/ac_shift_unc_with_weights_df['Total Uncertainty [kHz]'])**2)/dof

    prob_large_chi_squared = scipy.stats.chi2.sf(x=chi_squared_reduced*dof, df=dof)

    prob_arr[i] = prob_large_chi_squared

    chi_squared_dev_arr[i] = (chi_squared_reduced * dof - dof) / np.sqrt(2 * dof)

four_unc_chi_squared_dev_arr = chi_squared_dev_arr.copy()
four_prob_arr = prob_arr.copy()
#%%

# Five source of uncertainty (no fractional offset uncertainty) More points is needed here

ac_shift_unc_to_try_arr_5 = np.linspace(-0.2, 0.2, 401)
index_05 = np.where(ac_shift_unc_to_try_arr_5 == 0)
#%%
chi_squared_dev_arr = np.zeros(ac_shift_unc_to_try_arr_5.shape[0])
prob_arr = np.zeros(ac_shift_unc_to_try_arr_5.shape[0])

def minimize_unc_no_ac_shift(w_arr, df):
    w_sum = np.sum(w_arr)

    frac_offset_unc_tot = np.sum(df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum
    beam_rms_rad_unc_tot = np.sum(df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * w_arr) / w_sum

    phase_control_unc_tot = np.sum(df['Phase Control Uncertainty [kHz]'] * w_arr) / w_sum

    sod_unc_tot = np.sum(df['SOD Shift STD [kHz]'] * w_arr) / w_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((w_arr*df[freq_std_col_rename])**2) / w_sum**2)

    tot_unc = np.sqrt(zero_cross_stat_unc_tot**2)

    return tot_unc

def find_unc_weights_no_ac_shift(df):
    #w_arr = np.linspace(1, 10, df.shape[0])
    w_arr = np.ones(df.shape[0])
    w_arr = w_arr/np.sum(w_arr) / 10

    bounds_to_use_list = [(0, 1) for i in range(w_arr.shape[0])]

    w_min_arr = scipy.optimize.minimize(fun=minimize_unc_no_ac_shift, x0=w_arr, args=(df), bounds=bounds_to_use_list, constraints={'type': 'eq', 'fun': w_tot_constraint}, tol = 1E-15)

    df['Weight'] = w_min_arr['x']

    return df


ac_shift_unc_df['Total Uncertainty [kHz]'] = np.sqrt(ac_shift_unc_df['Zero-crossing Frequency STD (Normalized) [kHz]']**2)

for i in range(ac_shift_unc_to_try_arr_5.shape[0]):
    ac_shift_unc_to_try = ac_shift_unc_to_try_arr_5[i]

    av_ac_shift_study = straight_line_fit_params(ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]'], ac_shift_unc_df['Total Uncertainty [kHz]'])

    dof = ac_shift_unc_df['Resonant Frequency [kHz]'].shape[0]-1
    sigma1_prob = scipy.stats.chi2.sf(x=dof, df=dof) - scipy.stats.chi2.sf(x=dof+np.sqrt(2 * dof), df=dof)

    prob_arr[i] = scipy.stats.chi2.sf(x=av_ac_shift_study['Reduced Chi Squared'] * dof, df=dof)

    chi_squared_dev_arr[i] = (av_ac_shift_study['Reduced Chi Squared'] * dof - dof) / np.sqrt(2 * dof)

    ac_shift_unc_df_shifted = ac_shift_unc_df.copy()
    ac_shift_unc_df_shifted['Resonant Frequency [kHz]'] = ac_shift_unc_df['Resonant Frequency [kHz]'] + ac_shift_unc_to_try * ac_shift_unc_df['AC Shift [kHz]']

    ac_shift_unc_with_weights_df = ac_shift_unc_df_shifted.groupby('Waveguide Separation [cm]').apply(find_unc_weights_no_ac_shift)

    ac_shift_unc_with_weights_df['Weight'] = ac_shift_unc_with_weights_df['Weight'] / ac_shift_unc_with_weights_df.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().shape[0]

    weight_arr_sum = ac_shift_unc_with_weights_df['Weight'].sum()

    fract_offset_unc = np.sum(ac_shift_unc_with_weights_df['Fractional Offset Frequency Shift Uncertainty [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    beam_rms_rad_unc = np.sum(ac_shift_unc_with_weights_df['Beam RMS Radius Frequency Shift Uncertainty [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    sod_unc_tot = np.sum(ac_shift_unc_with_weights_df['SOD Shift STD [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((ac_shift_unc_with_weights_df['Weight']*ac_shift_unc_with_weights_df[freq_std_col_rename])**2) / weight_arr_sum**2)

    tot_unc = np.sqrt(zero_cross_stat_unc_tot**2)

    freq_av = np.sum(ac_shift_unc_with_weights_df['Resonant Frequency [kHz]'] * ac_shift_unc_with_weights_df['Weight']) / weight_arr_sum

    dof = ac_shift_unc_with_weights_df.shape[0] - 1

    chi_squared_reduced = np.sum(((ac_shift_unc_with_weights_df['Resonant Frequency [kHz]']-freq_av)/ac_shift_unc_with_weights_df['Total Uncertainty [kHz]'])**2)/dof

    prob_large_chi_squared = scipy.stats.chi2.sf(x=chi_squared_reduced*dof, df=dof)

    prob_arr[i] = prob_large_chi_squared

    chi_squared_dev_arr[i] = (chi_squared_reduced * dof - dof) / np.sqrt(2 * dof)

five_unc_chi_squared_dev_arr = chi_squared_dev_arr.copy()
five_prob_arr = prob_arr.copy()
#%%
prob_1_sigma = scipy.stats.chi2.sf(x=dof+np.sqrt(2 * dof), df=dof)
#%%
fig, ax = plt.subplots()
fig.set_size_inches(7, 4.5)

#ax.plot(ac_shift_unc_to_try_arr*100, one_prob_arr/one_prob_arr[index_0], color='brown')
#ax.plot(ac_shift_unc_to_try_arr*100, two_prob_arr/two_prob_arr[index_0], color='blue')
ax.plot(ac_shift_unc_to_try_arr*100, three_prob_arr/three_prob_arr[index_0], color='black')
#ax.plot(ac_shift_unc_to_try_arr*100, four_prob_arr/four_prob_arr[index_0], color='green')
#ax.plot(ac_shift_unc_to_try_arr_5*100, five_prob_arr/five_prob_arr[index_05], color='red')

ax.set_xlabel(r'$\alpha_{\mathrm{AC}}$ (%)')
ax.set_ylabel(r'$P(\alpha_{\mathrm{AC}})\;/\;P(0)$')
ax.set_xlim(-10, 20)

font_size = 15
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(font_size)
y_ax_lim = ax.get_ylim()

rect_unc = Rectangle((-5, -5), 10, 20, color='darkorange', fill=True, alpha=0.3)
ax.add_patch(rect_unc)

ax.plot([0, 0], [-10, 10], linestyle='dashed', color='black')

ax.set_ylim(y_ax_lim)
fig.tight_layout()

os.chdir(these_ac_folder_path)

plt_name = 'ac_shift_unc.pdf'
plt.savefig(plt_name)
plt.show()
