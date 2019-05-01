''' 2018-10-05 (Author: Nikita Bezginov)

Here I can compare different waveguide calibrations against each other. This is especially useful for comparing the calibration performed for different accelerating voltages.

In this particular case I am comparing the calibrations obtained for 4 cm waveguide separation. One is for the case when the accelerating votlage is 49.86 kV, and another is for 22.17 kV.
'''

from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil
import datetime

path_data_df = pd.read_csv(filepath_or_buffer='path_data.csv', delimiter=',', comment='#', header=[0], skip_blank_lines=True, index_col=[0])

code_folder_path = path_data_df.loc['Code Folder'].values[0].replace('\\', '/')
fosof_analyzed_data_folder_path = path_data_df.loc['FOSOF Analyzed Data Folder'].values[0].replace('\\', '/')
wvg_calib_data_folder_path = path_data_df.loc['Waveguide Calibration Folder'].values[0].replace('\\', '/')
krytar109B_pwr_det_calib_folder_path = path_data_df.loc['KRYTAR 109 B Power Detector Calibration Data Folder'].values[0].replace('\\', '/')
sim_data_folder_path = path_data_df.loc['Simulation Data Folder'].values[0].replace('\\', '/')

sys.path.insert(0, code_folder_path)

from exp_data_analysis import *
from fosof_data_set_analysis import *
from ZX47_Calibration_analysis import *
from KRYTAR_109_B_Calib_analysis import *

from hydrogen_sim_data import *

os.chdir(code_folder_path)
from wvg_power_calib_raw_data_analysis_Old import *
os.chdir(code_folder_path)
from wvg_power_calib_analysis import *

import re
import time
import math
import copy
import pickle

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt

# Package for wrapping long string to a paragraph
import textwrap

#%%
# 22.17 kV

fract_DC_offset = 0.02

# Settings for the Waveguides power calibration.
wvg_calib_param_dict = {
            'Date [date object]': datetime.date(year=2018, month=5, day=22),
            'Waveguide Separation [cm]': 4,
            'Accelerating Voltage [kV]': 22.17,
            'RF Frequency Scan Range [MHz]': '894-926',
            'Atom Off-Axis Distance (Simulation) [mm]': 1.8,
            'Fractional DC Offset': fract_DC_offset,
            'Minimum RF E Field Amplitude [V/cm]': 5,
            'Maximum RF E Field Amplitude [V/cm]': 27,
            'Use Boundary Conditions': False,
            'Polynomial Fit Order': 4
                    }
quench_sim_vs_freq_df = None
surv_frac_av_df = None

wvg_calib_analysis_22 = WaveguideCalibrationAnalysis(load_Q=True, wvg_calib_param_dict=wvg_calib_param_dict)

quench_sim_data_sets_df_22 = wvg_calib_analysis_22.analyze_simulation_quench_curves()
surv_frac_converted_df_22 = wvg_calib_analysis_22.extract_E_fields()

# RF E Field amplitude vs RF power parameters fit curves.
extracted_E_field_vs_RF_power_fits_set_df_22 = wvg_calib_analysis_22.get_converted_E_field_curve_fits()
# DC On/Off Ratio vs RF power parameters fit curves
surv_frac_vs_RF_power_fits_set_df_22 = wvg_calib_analysis_22.get_quench_curve_fits()

# Perform RF power parameters' calibration for each RF E field requested.
rf_e_field_calib_df_22 = wvg_calib_analysis_22.perform_power_calib()
calib_av_df_22 = wvg_calib_analysis_22.get_av_calib_data()
#%%
# 49.87 kV

fract_DC_offset = 0.03

# Settings for the Waveguides power calibration.
wvg_calib_param_dict =    {
            'Date [date object]': datetime.date(year=2018, month=5, day=12),
            'Waveguide Separation [cm]': 4,
            'Accelerating Voltage [kV]': 49.87,
            'RF Frequency Scan Range [MHz]': '894-926',
            'Atom Off-Axis Distance (Simulation) [mm]': 1.8,
            'Fractional DC Offset': fract_DC_offset,
            'Minimum RF E Field Amplitude [V/cm]': 5,
            'Maximum RF E Field Amplitude [V/cm]': 27,
            'Use Boundary Conditions': True
                        }
wvg_calib_analysis = WaveguideCalibrationAnalysis(load_Q=True, quench_sim_vs_freq_df=quench_sim_vs_freq_df, surv_frac_av_df=surv_frac_av_df, wvg_calib_param_dict=wvg_calib_param_dict)

quench_sim_data_sets_df = wvg_calib_analysis.analyze_simulation_quench_curves()
surv_frac_converted_df = wvg_calib_analysis.extract_E_fields()

# RF E Field amplitude vs RF power parameters fit curves.
extracted_E_field_vs_RF_power_fits_set_df = wvg_calib_analysis.get_converted_E_field_curve_fits()
# DC On/Off Ratio vs RF power parameters fit curves
surv_frac_vs_RF_power_fits_set_df = wvg_calib_analysis.get_quench_curve_fits()

# Perform RF power parameters' calibration for each RF E field requested.
rf_e_field_calib_df = wvg_calib_analysis.perform_power_calib()
calib_av_df = wvg_calib_analysis.get_av_calib_data()
#%%
rf_e_field_ampl = 18.0
rf_channel = 'A'

fig, axes = plt.subplots(nrows=1, ncols=4)

fig.set_size_inches(24, 8)

axes = wvg_calib_analysis.get_calibration_plot(rf_channel, rf_e_field_ampl, axes)
plt.show()
#%%
# Calculate the intersection of the indeces of the two calibrations
common_index = calib_av_df.index.intersection(calib_av_df_22.index)

# Fractional difference between the two calibrations
calib_fract_diff_df = (calib_av_df_22.loc[common_index] - calib_av_df.loc[common_index])/calib_av_df_22.loc[common_index]

# Calculate the STD of the fractional difference
calib_fract_diff_df.loc[slice(None), (slice(None), 'Data STD')] = np.sqrt((calib_av_df_22.loc[common_index].loc[slice(None), (slice(None), 'Data STD')].values/calib_av_df_22.loc[common_index].loc[slice(None), (slice(None), 'Mean Value')].values)**2 + (calib_av_df.loc[common_index].loc[slice(None), (slice(None), 'Data STD')].values/calib_av_df.loc[common_index].loc[slice(None), (slice(None), 'Mean Value')].values)**2)
#%%
# Waveguide
rf_channel = 'A'

# RF E field amplitude inside the wavegudie [V/cm]
rf_e_field = 5

calib_av_22_picked_df = calib_av_df_22.loc[common_index].loc[rf_channel, rf_e_field].reset_index().set_index('Waveguide Carrier Frequency [MHz]')

calib_av_picked_df = calib_av_df.loc[common_index].loc[rf_channel, rf_e_field].reset_index().set_index('Waveguide Carrier Frequency [MHz]')

calib_fract_diff_picked_df = calib_fract_diff_df.loc[rf_channel, rf_e_field].reset_index().set_index('Waveguide Carrier Frequency [MHz]')

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(16,12)

calib_av_22_picked_df['RF Generator Power Setting [mW]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=axes[0,0], label='22.17 kV')

calib_av_picked_df['RF Generator Power Setting [mW]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=axes[0,0], color='red', label='49.86 kV')

axes[0,0].set_ylabel('RF Generator Power Setting [mW]')

calib_av_22_picked_df['RF System Power Sensor Reading [V]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=axes[0,1])
#
calib_av_picked_df['RF System Power Sensor Reading [V]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=axes[0,1], color='red')

axes[0,1].set_ylabel('RF System Power Sensor Reading [V]')

calib_av_22_picked_df['RF System Power Sensor Detected Power [mW]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=axes[1,0])

calib_av_picked_df['RF System Power Sensor Detected Power [mW]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=axes[1,0], color='red')

axes[1,0].set_ylabel('RF System Power Sensor Detected Power [mW]')

calib_av_22_picked_df['RF Generator Power Setting [dBm]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=axes[1,1])

calib_av_picked_df['RF Generator Power Setting [dBm]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=axes[1,1], color='red')

axes[1,1].set_ylabel('RF Generator Power Setting [dBm]')

plt.show()
#%%
fig, ax = plt.subplots()
fig.set_size_inches(8,6)

calib_fract_diff_picked_df['RF Generator Power Setting [mW]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=ax, label='RF Generator Power Setting [mW]', color='blue')

calib_fract_diff_picked_df['RF System Power Sensor Reading [V]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=ax, label='RF System Power Sensor Reading [V]', color='red')

calib_fract_diff_picked_df['RF System Power Sensor Detected Power [mW]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=ax, label='RF System Power Sensor Detected Power [mW]', color='purple')

calib_fract_diff_picked_df['RF Generator Power Setting [dBm]'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Mean Value', yerr='Data STD', ax=ax, label='RF Generator Power Setting [dBm]', color='black')

ax.set_ylabel('Fractional difference')

plt.show()
#%%
# One can see that the fractional difference is almost the same vs RF frequency. Thus I can simply calculate the average fractional difference for each RF frequency and look at the difference between the calibrations versus RF power.

rf_channel = 'A'

calib_fract_diff_mean_df = calib_fract_diff_df.groupby(['Generator Channel', 'E Field [V/cm]']).aggregate(['mean', lambda x: np.std(x, ddof=1)/np.sqrt(x.shape[0])]).loc[rf_channel].drop(columns='Data STD', level=1).rename(columns={'<lambda>': 'Data STDOM', 'mean': 'Mean Value'}, level=2).reorder_levels([1, 0, 2], axis='columns')['Mean Value']

calib_fract_diff_mean_df = calib_fract_diff_mean_df.reset_index()
calib_fract_diff_mean_df['E Field Squared [V^2/cm^2]'] = calib_fract_diff_mean_df['E Field [V/cm]']**2

calib_fract_diff_mean_df.set_index('E Field Squared [V^2/cm^2]', inplace=True)
#%%
fig, ax = plt.subplots()
fig.set_size_inches(8,6)

calib_fract_diff_mean_df['RF Generator Power Setting [mW]'].reset_index().plot(kind='scatter', x='E Field Squared [V^2/cm^2]', y='Mean Value', yerr='Data STDOM', ax=ax, label='RF Generator Power Setting [mW]', color='blue')

calib_fract_diff_mean_df['RF System Power Sensor Reading [V]'].reset_index().plot(kind='scatter', x='E Field Squared [V^2/cm^2]', y='Mean Value', yerr='Data STDOM', ax=ax, label='RF System Power Sensor Reading [V]', color='red')

calib_fract_diff_mean_df['RF System Power Sensor Detected Power [mW]'].reset_index().plot(kind='scatter', x='E Field Squared [V^2/cm^2]', y='Mean Value', yerr='Data STDOM', ax=ax, label='RF System Power Sensor Detected Power [mW]', color='purple')

calib_fract_diff_mean_df['RF Generator Power Setting [dBm]'].reset_index().plot(kind='scatter', x='E Field Squared [V^2/cm^2]', y='Mean Value', yerr='Data STDOM', ax=ax, label='RF Generator Power Setting [dBm]', color='black')

ax.set_ylabel('Average fractional difference')

plt.show()

# It seems that we have not the greatest agreement at low RF powers. I am not very worried about the RF generator setting agreement, since we have seen that we can get different RF power after the RF amplifier for the same RF generator setting on different days. At about 8 V/cm the difference in RF power needed to be detected on the RF power detector is about 1% - we can definitely tolerate this. As for the lower powers, the difference is possible due to lower quality data, but I am not entirely sure.
