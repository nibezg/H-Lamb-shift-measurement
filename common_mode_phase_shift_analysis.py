''' Analysis of FOSOF data sets taken without the atomic beam. Instead of the Lyman-alpha detector signal for Digi 1 CH 1, we have KRYTAR 109B power detector connected to it, which is in turn connected to the RF power detector that gets the signal from RF probes inserted into each waveguide's RF choke tube.
'''

from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import pytz
import datetime

# For the lab
sys.path.insert(0,"C:/Users/Helium1/Google Drive/Research/Lamb shift measurement/Code") #
from exp_data_analysis import *
from fosof_data_set_analysis import *
import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt
#%%
# Folder for storing the phase drift data
analysis_data_folder = 'C:/Research/Lamb shift measurement/Data/Common-mode phase systematic study'

# Phase drift data csv file
phase_analysis_file_name = 'phase_analysis'+'.txt'

# File name containing list of the experiments acquired to test for the common-mode systematic phase shift effect.
experiments_list_file_name = 'Experiments_list.csv'
#%%
# Get list of the experiments
os.chdir(analysis_data_folder)

exp_folder_name_df = pd.read_csv(filepath_or_buffer=experiments_list_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True, index_col=None)

exp_folder_name_list = list(exp_folder_name_df.values.flatten())

# If the file with pre-saved analysis exists, then it gets loaded and any new experiments that were not analyzed, get analyzed, otherwise all of the experiments are analyzed
os.chdir(analysis_data_folder)

if os.path.isfile(phase_analysis_file_name):
    # Load the phase data file
    phase_drift_data_set_df = pd.read_csv(filepath_or_buffer=phase_analysis_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True, index_col=0)
    # We now want to see if there are any new experiments that are not in this folder.
    exp_folder_name_to_analyze_list = list(set(exp_folder_name_list) - set(phase_drift_data_set_df.index.values))

    # This is needed for adding data later to this dataframe
    phase_drift_data_set_df = phase_drift_data_set_df.T
else:
    exp_folder_name_to_analyze_list = exp_folder_name_list
    phase_drift_data_set_df = None

# Perform the analysis on the experiments for which the analysis was not performed before.

if len(exp_folder_name_to_analyze_list) > 0:
    for exp_folder_name in exp_folder_name_to_analyze_list:

        data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_data_Q=True)
        data_set.save_analysis_data()

        digi_df = data_set.get_digitizers_data()
        comb_phase_diff_df = data_set.get_combiners_phase_diff_data()
        digi_delay_df = data_set.get_inter_digi_delay_data()
        phase_diff_df = data_set.get_phase_diff_data()
        phase_av_set_averaged_df = data_set.average_av_sets()

        phase_A_minus_B_df, phase_freq_response_df = data_set.cancel_out_freq_response()
        #fosof_ampl_df, fosof_phase_df = data_set.average_FOSOF_over_repeats()

        inter_comb_phase_diff_df = phase_A_minus_B_df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging', ['Phase [Rad]', 'Phase STD [Rad]'])]

        inter_comb_phase_diff_grouped_df = inter_comb_phase_diff_df.reorder_levels(axis='columns', order=['Averaging Type', 'Fourier Harmonic', 'Phase Reference Type', 'Data Field']).sort_index(axis='columns').groupby(level=['Fourier Harmonic'], axis='columns')

        df = inter_comb_phase_diff_grouped_df.get_group(('First Harmonic'))
        df.columns = df.columns.droplevel(level=['Averaging Type', 'Fourier Harmonic'])

        # Calculate the phase difference between inter-waveguide combiner and other combiners + the standard deviation. The logic behind the calculation is shown on p.30 of Lab Notes #4 (August 31, 2018).
        # The standard deviation is not calculated correctly, due to correlations hidden in the phase difference (inter-waveguide phase difference is common to both of the calculations). Probably this is not even that important.

        df.loc[slice(None), 'Phase Difference [Rad]'] = (1/4*df['RF Combiner I Reference', 'Phase [Rad]'].values + 1/4*df['RF Combiner R Reference', 'Phase [Rad]'].values)

        df.loc[slice(None), 'Phase Difference STD [Rad]'] = np.sqrt((1/4*df['RF Combiner I Reference', 'Phase STD [Rad]'].values)**2 + (1/4*df['RF Combiner R Reference', 'Phase STD [Rad]'].values)**2)

        # Notice that the FOSOF phases that we had are from RF CHA - RF CH B calculation, WITHOUT division by 2. Thus we have to divide it here by 2.
        df.loc[slice(None), 'Phase Difference [Rad]'] = 1/2 * df.loc[slice(None), 'Phase Difference [Rad]']

        df.loc[slice(None), 'Phase Difference STD [Rad]'] = 1/2 * df.loc[slice(None), 'Phase Difference STD [Rad]']

        exp_params_s =  data_set.get_exp_parameters()
        exp_params_s['Experiment Start Time [s]']
        exp_params_s['Configuration']
        # 1/2 factor for Mean Combiner I, R Phase is due to not dividing by 2 when calculating RF CH A - RF CH B in fosof analysis code.

        phase_drift_df = pd.DataFrame(pd.Series({
            'Data Set': data_set,
            'Mean Combiner Phase Difference [mrad]': 1/2*np.mean(comb_phase_diff_df['First Harmonic', 'Fourier Phase [Rad]']),
            'Mean Phase Difference [mrad]': np.mean(df['Phase Difference [Rad]']),
            'Mean Combiner I Phase [mrad]': 1/2*np.mean(df['RF Combiner I Reference', 'Phase [Rad]']),
            'Mean Combiner R Phase [mrad]': 1/2*np.mean(df['RF Combiner R Reference', 'Phase [Rad]']),
            'Acquisition Start Time UTC [s]': exp_params_s['Experiment Start Time [s]'],
            'Acquisition End Time UTC [s]': exp_params_s['Experiment Start Time [s]'] + exp_params_s['Experiment Duration [s]'],
            'Configuration': exp_params_s['Configuration'],
            'Waveguide Separation [cm]': exp_params_s['Waveguide Separation [cm]']
            }, name=exp_folder_name))

        # A factor of 1E3 to convert radians to miliradians.
        phase_drift_df.loc[['Mean Combiner Phase Difference [mrad]', 'Mean Phase Difference [mrad]', 'Mean Combiner I Phase [mrad]', 'Mean Combiner R Phase [mrad]']] =  1E3 * phase_drift_df.loc[['Mean Combiner Phase Difference [mrad]', 'Mean Phase Difference [mrad]', 'Mean Combiner I Phase [mrad]', 'Mean Combiner R Phase [mrad]']]

        if phase_drift_data_set_df is None:
            phase_drift_data_set_df = phase_drift_df
        else:
            phase_drift_data_set_df = phase_drift_data_set_df.join(phase_drift_df)

    os.chdir(analysis_data_folder)

    # Saving the data frame containing information about loading the data.
    phase_drift_data_set_df.T.to_csv(path_or_buf='phase_analysis'+'.txt', header=True)
else:
    print("No new experiments were found.")

phase_drift_data_df = phase_drift_data_set_df.drop(axis='index', labels='Data Set')
#%%
# These data sets were acquired with the Box at atmospheric pressure. We do not want to look at this data.
exp_folder_name_df.drop(index=[0, 1, 2, 7, 8, 9, 10], inplace=True)

# Calculate the mean phase drift between 0 and pi configurations
# Time zone where the data was acquired
eastern_tz = pytz.timezone('US/Eastern')

# Dataframe containing phase data only.
phase_drift_df = phase_drift_data_df.drop(['Configuration', 'Acquisition Start Time UTC [s]', 'Acquisition End Time UTC [s]', 'Waveguide Separation [cm]'])

phase_drift_data_s_list = []
for index, row in exp_folder_name_df.iterrows():

    # Determining the average time for when the acqusition was taking place
    data_set_type_list = ['Experiment Folder 0-config', 'Experiment Folder pi-config']
    time_start_0_config = phase_drift_data_df[row['Experiment Folder 0-config']]['Acquisition Start Time UTC [s]']
    time_start_pi_config = phase_drift_data_df[row['Experiment Folder pi-config']]['Acquisition Start Time UTC [s]']

    max_time_index = np.argmax([time_start_0_config, time_start_pi_config])
    min_time_index = np.argmin([time_start_0_config, time_start_pi_config])
    data_set_type_max = data_set_type_list[max_time_index]
    data_set_type_min = data_set_type_list[min_time_index]

    time_start = phase_drift_data_df[row[data_set_type_min]]['Acquisition End Time UTC [s]']
    time_end = phase_drift_data_df[row[data_set_type_max]]['Acquisition End Time UTC [s]']
    mean_time = np.mean([time_start, time_end])

    exp_datetime = datetime.datetime.fromtimestamp(mean_time, tz=eastern_tz)

    # Calculating phase difference between 0-configuration and pi-configuration
    data_s = (phase_drift_df[row['Experiment Folder 0-config']] - phase_drift_df[row['Experiment Folder pi-config']]).append(pd.Series({'Date': exp_datetime}))

    phase_drift_data_s_list.append(data_s)

phase_shift_systematic_df = pd.DataFrame(phase_drift_data_s_list)
#%%
phase_shift_systematic_df
#%%
fig, ax = plt.subplots()
fig.set_size_inches(15,9)
phase_shift_systematic_df.set_index('Date').plot(ax=ax, style='.')
plt.show()
#%%
fig, ax = plt.subplots()
fig.set_size_inches(15,9)
phase_shift_systematic_df.set_index('Date').plot(ax=ax, style='.', y='Mean Phase Difference [mrad]')
plt.show()
#%%
fig, ax = plt.subplots()

phase_shift_systematic_df.plot(kind='scatter', x='Mean Phase Difference [mrad]', y='Mean Combiner Phase Difference [mrad]', ax=ax, label='Combiner I - Combiner R')

phase_shift_systematic_df.plot(kind='scatter', x='Mean Phase Difference [mrad]', y=['Mean Combiner I Phase [mrad]'], ax=ax, color='Red', label='Combiner I')

phase_shift_systematic_df.plot(kind='scatter', x='Mean Phase Difference [mrad]', y=['Mean Combiner R Phase [mrad]'], ax=ax, color='DarkGreen', label='Combiner R')

ax.set_ylabel('$\Delta \phi$ [mrad]')

plt.show()

#%%
# Compute the average change in various types of phases
av_phase_shift_systematic_df = phase_shift_systematic_df.drop('Date', axis='columns').aggregate(lambda x: np.mean(x))
#%%
av_phase_shift_systematic_df
#%%
shift = av_phase_shift_systematic_df['Shift [kHz]']
final_uncertainty = np.sqrt(2.9**2+shift**2)

print('Phase systematic shift [kHz]: ' + str(shift))
print('Final uncertainty [kHz]: ' + str(final_uncertainty))
#%%
#%%
# ========================================
# Useful code to look at individual data sets
# ========================================
exp_folder_name = '180914-135544 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'

data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_data_Q=True)
#data_set.save_analysis_data()

digi_df = data_set.get_digitizers_data()
comb_phase_diff_df = data_set.get_combiners_phase_diff_data()
digi_delay_df = data_set.get_inter_digi_delay_data()
phase_diff_df = data_set.get_phase_diff_data()
phase_av_set_averaged_df = data_set.average_av_sets()

phase_A_minus_B_df, phase_freq_response_df = data_set.cancel_out_freq_response()
#fosof_ampl_df, fosof_phase_df = data_set.average_FOSOF_over_repeats()

inter_comb_phase_diff_df = phase_A_minus_B_df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging', ['Phase [Rad]', 'Phase STD [Rad]'])]

inter_comb_phase_diff_grouped_df = inter_comb_phase_diff_df.reorder_levels(axis='columns', order=['Averaging Type', 'Fourier Harmonic', 'Phase Reference Type', 'Data Field']).sort_index(axis='columns').groupby(level=['Fourier Harmonic'], axis='columns')

df = inter_comb_phase_diff_grouped_df.get_group(('First Harmonic'))
df.columns = df.columns.droplevel(level=['Averaging Type', 'Fourier Harmonic'])

# Calculate the phase difference between inter-waveguide combiner and other combiners + the standard deviation. The standard deviation is not calculated correctly, due to correlations hidden in the phase difference (inter-waveguide phase difference is common to both of the calculations). Probably this is not even that important.

df.loc[slice(None), 'Phase Difference [Rad]'] = 1/4*df['RF Combiner I Reference', 'Phase [Rad]'].values + 1/4*df['RF Combiner R Reference', 'Phase [Rad]'].values

df.loc[slice(None), 'Phase Difference STD [Rad]'] = np.sqrt((1/4*df['RF Combiner I Reference', 'Phase STD [Rad]'].values)**2 + (1/4*df['RF Combiner R Reference', 'Phase STD [Rad]'].values)**2)

#%%
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(16,12)

df['RF Combiner I Reference'].reset_index().plot(kind='scatter', x='Repeat', y='Phase [Rad]', yerr='Phase STD [Rad]', ax=axes[0, 0])

data_arr_max = np.max((df['RF Combiner I Reference', 'Phase [Rad]']+df['RF Combiner I Reference', 'Phase STD [Rad]']).values)
data_arr_min = np.min((df['RF Combiner I Reference', 'Phase [Rad]']-df['RF Combiner I Reference', 'Phase STD [Rad]']).values)

axes[0, 0].set_ylim(data_arr_min, data_arr_max)

df['RF Combiner R Reference'].reset_index().plot(kind='scatter', x='Repeat', y='Phase [Rad]', yerr='Phase STD [Rad]', ax=axes[0, 1])

data_arr_max = np.max((df['RF Combiner R Reference', 'Phase [Rad]']+df['RF Combiner I Reference', 'Phase STD [Rad]']).values)
data_arr_min = np.min((df['RF Combiner R Reference', 'Phase [Rad]']-df['RF Combiner I Reference', 'Phase STD [Rad]']).values)

axes[0, 1].set_ylim(data_arr_min, data_arr_max)

df.reset_index().plot(kind='scatter', x='Repeat', y='Phase Difference [Rad]', yerr='Phase Difference STD [Rad]', ax=axes[1, 0])

data_arr_max = np.max((df['Phase Difference [Rad]']+df['Phase Difference STD [Rad]']).values)
data_arr_min = np.min((df['Phase Difference [Rad]']-df['Phase Difference STD [Rad]']).values)
data_arr_max - data_arr_min
axes[1, 0].set_ylim(data_arr_min, data_arr_max)

comb_phase_diff_df['First Harmonic'].reset_index().plot(kind='scatter', x='Elapsed Time [s]', y='Fourier Phase [Rad]', ax=axes[1, 1])

data_arr_max = np.max((comb_phase_diff_df['First Harmonic', 'Fourier Phase [Rad]']).values)
data_arr_min = np.min((comb_phase_diff_df['First Harmonic', 'Fourier Phase [Rad]']).values)

axes[1, 1].set_ylim(data_arr_min, data_arr_max)

plt.show()
#%%
