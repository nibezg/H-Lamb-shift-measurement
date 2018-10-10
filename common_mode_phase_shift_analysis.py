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
import seaborn as sns
#%%
# Folder for storing the phase drift data
analysis_data_folder = 'C:/Research/Lamb shift measurement/Data/Common-mode phase systematic study'


# Phase drift data csv file
phase_analysis_file_name = 'phase_drifts'+'.csv'
# Experiment parameters csv file
exp_parameters_file_name = 'exp_params'+'.csv'

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

    # Load the phase data file and the experiment parameters files
    phase_drift_data_df = pd.read_csv(filepath_or_buffer=phase_analysis_file_name, delimiter=',', comment='#', header=[0, 1], skip_blank_lines=True, index_col=[0, 1])

    exp_params_df = pd.read_csv(filepath_or_buffer=exp_parameters_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True, index_col=0)

    # We now want to see if there are any new experiments that are not in this folder.
    exp_folder_name_to_analyze_list = list(set(exp_folder_name_list) - set(phase_drift_data_df.index.get_level_values('Experiment Folder Name').values))

else:
    exp_folder_name_to_analyze_list = exp_folder_name_list
    phase_drift_data_df = pd.DataFrame()
    exp_params_df = pd.DataFrame()

if len(exp_folder_name_to_analyze_list) > 0:

    phase_drift_data_list = []
    exp_params_list = []

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

        # Calculate the phase difference between inter-waveguide combiner and other combiners + the standard deviation. The logic behind the calculation is shown on p.30 of Lab Notes #4 (August 31, 2018). Notice, however, that we are not dividing here by the required factor of 4. This is because we want to do the division at the very end, when 0 and pi configurations are subtracted from each other. Make sure to remember that there is also another factor of 2, by which we have to divide this phase. It comes from A - B subtaction, performed in the FOSOF data analysis code, where we purposedly did not divide by 2.

        df.loc[slice(None), 'Phase Difference [Rad]'] = (df['RF Combiner I Reference', 'Phase [Rad]'].values + df['RF Combiner R Reference', 'Phase [Rad]'].values)

        df['Phase Difference [Rad]'] = df['Phase Difference [Rad]'].transform(convert_phase_to_2pi_range)

        # The standard deviation is not calculated correctly, due to correlations hidden in the phase difference (inter-waveguide phase difference is common to both of the calculations). Probably this is not even that important.

        df.loc[slice(None), 'Phase Difference STD [Rad]'] = np.sqrt((df['RF Combiner I Reference', 'Phase STD [Rad]'].values)**2 + (df['RF Combiner R Reference', 'Phase STD [Rad]'].values)**2)

        df = df.drop(columns=['Phase STD [Rad]'], level='Data Field').swaplevel(axis='columns', i='Data Field', j='Phase Reference Type')
        df.columns = df.columns.droplevel('Data Field')
        df.drop(columns='Phase Difference STD [Rad]', inplace=True)

        # Notice the - sign in front of the difference. This is because we have Combiner Inter-Waveguide - Combiner I and Combiner Inter-Waveguide - Combiner R as our phase differences.
        df['Combiners I-R Phase Difference [Rad]'] = (-(df['RF Combiner I Reference'] - df['RF Combiner R Reference'])).transform(lambda x: convert_phase_to_2pi_range(x))

        data_df = df.groupby('Waveguide Carrier Frequency [MHz]').aggregate(['mean', lambda x: np.std(x, ddof=1)/np.sqrt(x.shape[0])])

        data_df.rename(columns={'mean': 'Phase [Rad]', '<lambda>': 'Phase STDOM [Rad]'}, level=1, inplace=True)

        exp_params_s =  data_set.get_exp_parameters()

        data_df['Experiment Folder Name'] = exp_folder_name

        data_df = data_df.reset_index().set_index(['Experiment Folder Name', 'Waveguide Carrier Frequency [MHz]'])

        exp_params_s['Experiment Folder Name'] = exp_folder_name

        # Phase shifts data and experiment parameters data are stored in different dataframes for convenience. Later one can simply join the two dataframe together.
        exp_params_list.append(exp_params_s)
        phase_drift_data_list.append(data_df)

    # Add the new analysis data to the dataframes
    exp_params_df = exp_params_df.append(pd.DataFrame(exp_params_list).set_index('Experiment Folder Name'))
    phase_drift_data_df = phase_drift_data_df.append(phase_drift_data_list)

    os.chdir(analysis_data_folder)

    # Saving the data frames
    phase_drift_data_df.to_csv(path_or_buf=phase_analysis_file_name, header=True)
    exp_params_df.to_csv(path_or_buf=exp_parameters_file_name, header=True)

else:
    print("No new experiments were found.")

#%%
# These data sets were acquired with the Box at atmospheric pressure. We do not want to look at this data.
exp_folder_name_filtered_df = exp_folder_name_df.drop(index=[0, 1, 2, 7, 8, 9, 10])

# Calculate the mean phase drift between 0 and pi configurations
# Time zone where the data was acquired
eastern_tz = pytz.timezone('US/Eastern')

phase_drift_data_s_list = []
for index, row in exp_folder_name_filtered_df.iterrows():

    # Determining the average time for when the acqusition was taking place
    data_set_type_list = ['Experiment Folder 0-config', 'Experiment Folder pi-config']

    exp_params_0_config_s =  exp_params_df.loc[row['Experiment Folder 0-config']]
    exp_params_pi_config_s =  exp_params_df.loc[row['Experiment Folder pi-config']]

    time_start_0_config = exp_params_0_config_s['Experiment Start Time [s]']
    time_start_pi_config = exp_params_pi_config_s['Experiment Start Time [s]']

    max_time_index = np.argmax([time_start_0_config, time_start_pi_config])
    min_time_index = np.argmin([time_start_0_config, time_start_pi_config])
    data_set_type_max = data_set_type_list[max_time_index]
    data_set_type_min = data_set_type_list[min_time_index]

    time_start =  exp_params_df.loc[row[data_set_type_min]]['Experiment Start Time [s]']
    time_end =  exp_params_df.loc[row[data_set_type_max]]['Experiment Start Time [s]'] + exp_params_df.loc[row[data_set_type_max]]['Experiment Duration [s]']
    mean_time = np.mean([time_start, time_end])

    exp_datetime = pd.Timestamp(datetime.datetime.fromtimestamp(mean_time, tz=eastern_tz))

    # Calculating phase difference between 0-configuration and pi-configuration
    data_df = (phase_drift_data_df.loc[row['Experiment Folder 0-config']] - phase_drift_data_df.loc[row['Experiment Folder pi-config']])

    # Calculate the STDOM of the differences
    data_df.loc[slice(None), (slice(None), 'Phase STDOM [Rad]')] = np.sqrt(phase_drift_data_df.loc[row['Experiment Folder 0-config']].loc[slice(None), (slice(None), 'Phase STDOM [Rad]')]**2 + phase_drift_data_df.loc[row['Experiment Folder pi-config']].loc[slice(None), (slice(None), 'Phase STDOM [Rad]')]**2)

    data_df.loc[slice(None), ('Date')] = exp_datetime
    data_df.loc[slice(None), ('Experiment Folder Name 0-config')] = exp_params_0_config_s.name

    data_df = data_df.reset_index().set_index(['Experiment Folder Name 0-config', 'Waveguide Carrier Frequency [MHz]', 'Date']).sort_index()

    phase_drift_data_s_list.append(data_df)

# Combine the individual dataframes together
phase_config_diff_df = pd.DataFrame().append(phase_drift_data_s_list)

# Correct for A and B difference (division by two that was not done before)
phase_shift_df = phase_config_diff_df/2

# Divide the phase difference by a factor of 4 as required (see comments above in the first loop of the code)
phase_shift_df['Phase Difference [Rad]'] = phase_shift_df['Phase Difference [Rad]'] / 4

# Convert radians to miliradians
phase_shift_df = phase_shift_df * 1E3

phase_shift_df.rename(columns={'Phase [Rad]': 'Phase [mrad]', 'Phase STDOM [Rad]': 'Phase STDOM [mrad]'}, inplace=True)
phase_shift_df.rename(columns={'Phase Difference [Rad]': 'Phase Difference', 'Combiners I-R Phase Difference [Rad]': 'Combiners I-R Phase Difference'}, inplace=True)
#%%
# ==================
# Organizing the data
# ==================

# Select the required experiments to match the required experiment parameters

exp_folder_name_to_use_list = exp_params_df[(exp_params_df['Waveguide Separation [cm]'] == 7) & (exp_params_df['Configuration'] == '0') & (exp_params_df['Waveguide Electric Field [V/cm]'] == 18)].index.values

phase_shift_chosen_df = phase_shift_df.reset_index().set_index('Experiment Folder Name 0-config').loc[exp_folder_name_to_use_list].reset_index().drop(columns=['Experiment Folder Name 0-config']).set_index(['Waveguide Carrier Frequency [MHz]', 'Date']).sort_index()

#%%
sns.relplot(x='Waveguide Carrier Frequency [MHz]', y='Phase [mrad]', hue='Date', data=phase_shift_chosen_df['Phase Difference'].reset_index())

plt.show()
#%%
phase_shift_chosen_df.columns = phase_shift_chosen_df.columns.remove_unused_levels()

phase_shift_chosen_avg_df = phase_shift_chosen_df.groupby(['Waveguide Carrier Frequency [MHz]']).aggregate(['mean', lambda x: np.std(x, ddof=1)/np.sqrt(x.shape[0])]).drop(columns=['Phase STDOM [mrad]'], level=1).rename(columns={'mean': 'Phase [mrad]', '<lambda>': 'Phase STDOM [mrad]'}).reorder_levels([1, 0, 2], axis='columns')['Phase [mrad]']
#%%
phase_shift_chosen_avg_df
#%%

data_df = phase_shift_chosen_avg_df['Phase Difference'].reset_index()

x_data_arr = data_df['Waveguide Carrier Frequency [MHz]']
y_data_arr = data_df['Phase [mrad]']
y_data_std_arr = data_df['Phase STDOM [mrad]']

poly_fit_params = np.polyfit(x=x_data_arr, y=y_data_arr, deg=4, w=1/y_data_std_arr**2)

phase_shift_fit_func = np.poly1d(poly_fit_params)

x_arr = np.linspace(np.min(x_data_arr), np.max(x_data_arr), 10*x_data_arr.shape[0])

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(15, 10)

axes[0, 0].plot(x_arr, phase_shift_fit_func(x_arr), color='blue')

data_df.plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Phase [mrad]', yerr='Phase STDOM [mrad]', ax=axes[0, 0])

phase_shift_chosen_avg_df['Combiners I-R Phase Difference'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Phase [mrad]', yerr='Phase STDOM [mrad]', ax=axes[0, 1])

phase_shift_chosen_avg_df['RF Combiner I Reference'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Phase [mrad]', yerr='Phase STDOM [mrad]', ax=axes[1, 0])

phase_shift_chosen_avg_df['RF Combiner R Reference'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Phase [mrad]', yerr='Phase STDOM [mrad]', ax=axes[1, 1])

plt.show()
#%%
# ====================================
# Effect of the frequency-dependent phase shift on FOSOF lineshape.
# ====================================
# I model the FOSOF lineshape as the first-order polynomial crossing 910 MHz (defined as 0). The slope is taken as the average value of the slope obtained experimentally at given separation.

# Waveguide separation [cm] and corresponding slope in Rad/MHz = mrad/kHz
slope_dict = {4: 1/10.5, 5: 1/9, 6: 1/7.5, 7: 1/6.7, 8: 10**2}

# Value by which to shift the frequencies [MHz]
f_shift = 910

# Resonant frequency [MHz]
f0 = 910 - f_shift

# List of frequencies shifted by 910 MHz
f_arr = data_df['Waveguide Carrier Frequency [MHz]'].values - f_shift

# Waveguide separation [cm]
waveguide_separation = 4

slope = slope_dict[waveguide_separation]

phi0 = -slope * f0
phase_arr = phi0 + slope * f_arr

# Adding the systematic phase shift to the FOSOF lineshape. Notice that the fit function for the phase shift gives miliradians.
phase_syst_shift_arr = phi0 + slope * f_arr + phase_shift_fit_func(f_arr + f_shift) * 1E-3

fosof_poly_fit_params = np.polyfit(x=f_arr, y=phase_arr, deg=1)
f0 = -fosof_poly_fit_params[1]/fosof_poly_fit_params[0]

fosof_poly_fit_shift_params = np.polyfit(x=f_arr, y=phase_syst_shift_arr, deg=1)
f0_shifted = -fosof_poly_fit_shift_params[1]/fosof_poly_fit_shift_params[0]

# Frequency shift [kHz]
freq_shift = (f0_shifted - f0) * 1E3
print('Frequency shift [kHz]: ' + str(freq_shift))
#%%
# Average phase shift converting to frequency [kHz]
av_shift = np.mean(phase_shift_fit_func(x_arr))/slope
av_shift
#%%
fig, ax = plt.subplots()
ax.scatter(x=f_arr, y=phase_arr)
plt.show()
#%%
# ==================
# Organizing the data for 7 cm separation and 8 V/cm of the E field Amplitude inside the waveguides
# ==================

# Select the required experiments to match the required experiment parameters

exp_folder_name_to_use_list = exp_params_df[(exp_params_df['Waveguide Separation [cm]'] == 7) & (exp_params_df['Configuration'] == '0') & (exp_params_df['Waveguide Electric Field [V/cm]'] == 8)].index.values

phase_shift_chosen_df = phase_shift_df.reset_index().set_index('Experiment Folder Name 0-config').loc[exp_folder_name_to_use_list].reset_index().drop(columns=['Experiment Folder Name 0-config']).set_index(['Waveguide Carrier Frequency [MHz]', 'Date']).sort_index()

#%%
sns.relplot(x='Waveguide Carrier Frequency [MHz]', y='Phase [mrad]', hue='Date', data=phase_shift_chosen_df['Phase Difference'].reset_index())

plt.show()
#%%
phase_shift_chosen_df.columns = phase_shift_chosen_df.columns.remove_unused_levels()

phase_shift_chosen_avg_df = phase_shift_chosen_df.groupby(['Waveguide Carrier Frequency [MHz]']).aggregate(['mean', lambda x: np.std(x, ddof=1)/np.sqrt(x.shape[0])]).drop(columns=['Phase STDOM [mrad]'], level=1).rename(columns={'mean': 'Phase [mrad]', '<lambda>': 'Phase STDOM [mrad]'}).reorder_levels([1, 0, 2], axis='columns')['Phase [mrad]']
#%%
phase_shift_chosen_avg_df
#%%

data_df = phase_shift_chosen_avg_df['Phase Difference'].reset_index()

x_data_arr = data_df['Waveguide Carrier Frequency [MHz]']
y_data_arr = data_df['Phase [mrad]']
y_data_std_arr = data_df['Phase STDOM [mrad]']

poly_fit_params = np.polyfit(x=x_data_arr, y=y_data_arr, deg=4, w=1/y_data_std_arr**2)

phase_shift_fit_func = np.poly1d(poly_fit_params)

x_arr = np.linspace(np.min(x_data_arr), np.max(x_data_arr), 10*x_data_arr.shape[0])

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(15, 10)

axes[0, 0].plot(x_arr, phase_shift_fit_func(x_arr), color='blue')

data_df.plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Phase [mrad]', yerr='Phase STDOM [mrad]', ax=axes[0, 0])

phase_shift_chosen_avg_df['Combiners I-R Phase Difference'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Phase [mrad]', yerr='Phase STDOM [mrad]', ax=axes[0, 1])

phase_shift_chosen_avg_df['RF Combiner I Reference'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Phase [mrad]', yerr='Phase STDOM [mrad]', ax=axes[1, 0])

phase_shift_chosen_avg_df['RF Combiner R Reference'].reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y='Phase [mrad]', yerr='Phase STDOM [mrad]', ax=axes[1, 1])

plt.show()
