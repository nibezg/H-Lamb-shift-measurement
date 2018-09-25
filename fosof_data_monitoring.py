from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string

sys.path.insert(0,"C:/Users/Helium1/Google Drive/Code/Python/Testing/Blah") #
from exp_data_analysis import *

import re
import numpy.fft
import time
import scipy.fftpack
import scipy.interpolate
import matplotlib.pyplot as plt
import math

import threading
from Queue import Queue

# Package for wrapping long string to a paragraph
import textwrap

from Tkinter import *
import ttk
import tkMessageBox

import copy
#%%
# Analysis version
version_number = 0.1

exp_folder_name = '180524-185405 - FOSOF Acquisition - pi config, 18 V per cm PD ON 82.5 V, 22.17 kV, B_y scan'

exp_folder_name = '180627-214753 - FOSOF Acquisition - 0 config, 24 V per cm PD 120 V, 49.86 kV, 908-912 MHz. B_x scan'

#exp_folder_name = '180610-115642 - FOSOF Acquisition 910 onoff (80 pct) - pi config, 8 V per cm PD 120V, 920-922 MHz'

#exp_folder_name = '180615-164059 - FOSOF Acquisition - pi config, 8 V per cm PD 120 V, 49.86 kV, 908-912 MHz'
# Location where the analyzed experiment is saved
saving_folder_location = 'C:/Research/Lamb shift measurement/Data/FOSOF analyzed data sets'
# Analyzed data file name
data_analyzed_file_name = 'data_analyzed v' + str(version_number) + '.txt'

os.chdir(saving_folder_location)
os.chdir(exp_folder_name)

exp_data_frame = pd.read_csv(filepath_or_buffer=data_analyzed_file_name, delimiter=',', comment='#', header=0)

# Add a column of elapsed time since the start of the acquisition
exp_data_frame['Elapsed Time [s]'] = exp_data_frame['Time'].transform(lambda x: x-x[0])

# There is additional space before [V] in the column below. Fixing it here
exp_data_frame = exp_data_frame.rename(columns={'Waveguide A Power Reading  [V]': 'Waveguide A Power Reading [V]'})


exp_params_dict, comment_string_arr = get_experiment_params(data_analyzed_file_name)
#%%

# Some of the column column
detector_first_harmonic_phase_col = 'Detector First Harmonic Fourier Phase [Rad]'
detector_second_harmonic_phase_col = 'Detector Second Harmonic Fourier Phase [Rad]'
combiner_I_digi_1_first_harmonic_phase_col = 'RF Power Combiner I Digi 1 First Harmonic Fourier Phase [Rad]'
combiner_I_digi_1_second_harmonic_phase_col = 'RF Power Combiner I Digi 1 Second Harmonic Fourier Phase [Rad]'

combiner_I_digi_2_first_harmonic_phase_col = 'RF Power Combiner I Digi 2 First Harmonic Fourier Phase [Rad]'
combiner_I_digi_2_second_harmonic_phase_col = 'RF Power Combiner I Digi 2 Second Harmonic Fourier Phase [Rad]'

detector_SNR_col = 'Detector First Harmonic SNR'
#%%
#exp_data_frame = exp_data_frame.iloc[:312]
exp_data_frame[detector_SNR_col]
#%%
# Series containing flags to determine what kind of data set has been acquired. It is possible for several flags to be True, however, not for all of the combinations of flags there exists analysis at the moment.
data_set_type_s = pd.Series(
            {'B Field Scan': False,
            'Pre-910 Switching': False,
            'Waveguide Carrier Frequency Sweep': False,
            'Offset Frequency Switching': False})

# Important experiment parameters
n_Bx_steps = exp_params_dict['Number of B_x Steps']
n_By_steps = exp_params_dict['Number of B_y Steps']
n_averages = exp_params_dict['Number of Averages']
sampling_rate = exp_params_dict['Digitizer Sampling Rate [S/s]']
n_digi_samples = exp_params_dict['Number of Digitizer Samples']
n_freq_steps = exp_params_dict['Number of Frequency Steps']
n_repeats = exp_params_dict['Number of Repeats']
digi_ch_range = exp_params_dict['Digitizer Channel Range [V]']
offset_freq_array = np.array(exp_params_dict['Offset Frequency [Hz]'])
pre_910_on_n_digi_samples = exp_params_dict['Pre-Quench 910 On Number of Digitizer Samples']

# Expected number of rows in the data file
rows_expected = n_repeats * n_freq_steps * len(offset_freq_array) * n_averages * 2

if n_freq_steps > 1:
    data_set_type_s['Waveguide Carrier Frequency Sweep'] = True

# B field values for different axes are assumed to be part of the same parameter: B field in such a way that when the B field along one axis is getting scanned, B field along another axis is set to 0. Thus the total number of B field scan steps is the sum of the scan steps along each axis.  When both the Bx and By are set to zero and are not getting scanned though, the B field parameter is not getting changed, thus the total number of B field steps is 1 (one).
if exp_params_dict['B_x Min [Gauss]'] == 0 and exp_params_dict['B_x Max [Gauss]'] == 0 and exp_params_dict['B_y Min [Gauss]'] == 0 and exp_params_dict['B_y Max [Gauss]'] == 0:
    n_B_field_steps = 1

else:
    data_set_type_s['B Field Scan'] = True
    if n_Bx_steps == 1 and n_By_steps > 1:
        n_B_field_steps = n_By_steps
    if n_By_steps == 1 and n_Bx_steps > 1:
        n_B_field_steps = n_Bx_steps
    if n_Bx_steps > 1 and n_By_steps > 1:
        n_B_field_steps = n_Bx_steps + n_By_steps

rows_expected = rows_expected * n_B_field_steps

if exp_params_dict['Pre-Quench 910 On/Off']:
    data_set_type_s['Pre-910 Switching'] = True
    rows_expected = rows_expected * 2

if offset_freq_array.shape[0] > 1:
    data_set_type_s['Offset Frequency Switching'] = True
#%%
data_set_type_s
#%%

data_set_type = None
if data_set_type_s['Waveguide Carrier Frequency Sweep'] == False:
    print('There is no frequency scan for the data set. No analysis will be performed.')
else:
    data_set_type_no_freq_s = data_set_type_s.drop(labels=['Waveguide Carrier Frequency Sweep'])
    if data_set_type_no_freq_s[data_set_type_no_freq_s==True].size > 1:
        print('In addition to the Waveguide Carrier Frequency scan more than one parameters were scanned. The analysis will not be performed')
    else:
        data_set_type = data_set_type_s[data_set_type_s==True].index.values[0]
#%%
data_set_type
#%%
#%%
# Size check for the data set.

# We want to make sure that our analysis will proceed as expected. This means that we must definitely be sure that there is not more data than expected.
if exp_data_frame.shape[0] > rows_expected:
    raise FosofAnalysisError("Number of rows in the analyzed data file is larger than the expected number of rows")

if exp_data_frame.shape[0] < rows_expected:
    print('Seems that the data set has not been fully acquired/analyzed')
#%%
# Fourier harmonics of the offset frequency that are present in the data set. This list will be used throughout the analysis.
harmonic_name_list = ['First Harmonic', 'Second Harmonic']
#%%
# Convenience functions used to reorganize the data set into convenient shape.
def match_string_start(array_str, match_str):
    ''' Returns boolean index for the array of strings, array_str, of whether the string match_str was a subset of the given string at its start.
    '''
    reg_exp = re.compile(match_str)
    matched_arr = np.array(map(lambda x: reg_exp.match(x), array_str)) != None

    return matched_arr

def remove_matched_string_from_start(array_str, match_str):
    ''' Removes first encounter (most left) of 'match_str' in the strings of the array_str. Each member of the array of strings, array_str, must contain at least one occurence of match_str.
    '''

    reg_exp = re.compile(match_str)

    # We split the column names to get the part that has no 'match_str' characters in it. We should only allow for single splitting to happen, to allow for the match_str string to be contained somewhere in the bulk of the string.

    stripped_column_arr = np.array(map(lambda x: reg_exp.split(x, maxsplit=1)[1], array_str))

    # The leftover string has whitespace (most likely) as the first character for every split column name. We remove it.
    stripped_column_arr = map(lambda x: x.strip(), stripped_column_arr)


    return dict(np.array([array_str, stripped_column_arr]).T)

def add_level_data_frame(df, level_name, level_value_arr):
    ''' Adds a new level to otherwise flat (non MultiIndex) dataframe.

    The level can have column names specified in an array. For each member of the array the columns are determined that have match in the beginning of the name with that particular array element. Column names that do not match any of the array elements are assigned value of 'Other'.

    Inputs:
    :df: pandas DataFrame
    :level_name: Name of the level to add
    :level_value_arr: Sublevel column names
    '''
    df_T = df.T
    df_T[level_name] = np.nan
    for level_value in level_value_arr:
        df_T.loc[df_T.index.values[match_string_start(df_T.index.values, level_value)], level_name] = level_value

    # There are also indeces that do not correspond to any level_value.
    df_T.loc[df_T[level_name].isnull(), level_name] = 'Other'

    updated_df = df_T.reset_index().set_index([level_name, 'index']).sort_index().T

    for level_value in level_value_arr:
        updated_df.rename(columns=remove_matched_string_from_start(updated_df[level_value].columns.values, level_value), level=1, inplace=True)

    return updated_df
#%%
# The data set contains many columns. Some columns are related to each other by sharing one or several common properties. It is convenient to group these columns together. Each of the groups might have its unique structure for subsequent analysis. Thus it is inconvenient to have single data frame. Thus I now categorize the data set into several smaller subsets. Each subset corresponds to data for specific category. The categories at the moment are digitizers, quenching cavities, beam end faraday cup, and RF power in the waveguides.

# Each subset contains the same general index that depends on the type of the experiment.
# 'index' and 'Elapsed Time [s]' are position at the end of the list to make sure that the resulting multiindex can be sorted.
index_column_list = ['Repeat', 'Configuration', 'Average', 'index', 'Elapsed Time [s]']

if data_set_type_s['Waveguide Carrier Frequency Sweep']:
    index_column_list.insert(index_column_list.index('Configuration'), 'Waveguide Carrier Frequency [MHz]')

if data_set_type_s['Offset Frequency Switching']:
    index_column_list.insert(index_column_list.index('Configuration'), 'Offset Frequency [Hz]')

if data_set_type_s['Offset Frequency Switching']:
    index_column_list.insert(index_column_list.index('Configuration'), 'Offset Frequency [Hz]')

if data_set_type_s['B Field Scan']:
    if n_Bx_steps > 1:
        index_column_list.insert(index_column_list.index('Configuration'), 'B_x [Gauss]')
    if n_By_steps > 1:
        index_column_list.insert(index_column_list.index('Configuration'), 'B_y [Gauss]')
if data_set_type_s['Pre-910 Switching']:
    index_column_list.insert(index_column_list.index('Configuration'), 'Pre-Quench 910 State')

general_index = exp_data_frame.reset_index().set_index(index_column_list).index
# Renaming 'index' name to 'Index' for enforcing first letter = capital letter rule
general_index.set_names('Index',level=list(general_index.names).index('index'), inplace=True)
#%%
# Data for the beam end Faraday cup:

beam_end_dc_df = exp_data_frame[exp_data_frame.columns[match_string_start(exp_data_frame.columns, 'fc')]].set_index(general_index)
#%%

fig, axes = plt.subplots(ncols=2, nrows=5)
fig.set_size_inches(12,24)

for i in range(beam_end_dc_df.columns.values.size):
    y_column = beam_end_dc_df.columns.values[i]
    beam_end_dc_df.reset_index().plot(kind='scatter', x='Elapsed Time [s]', y=y_column, ax=axes.flat[i])

plt.show()

#%%
# Data from the RF Power detectors for each RF system (A and B).
rf_system_power_df = exp_data_frame[['Waveguide A Power Reading [V]','Waveguide B Power Reading [V]']]

# Restructure the dataframe a bit
rf_system_power_df = add_level_data_frame(rf_system_power_df.reset_index(), 'RF System', ['Waveguide A', 'Waveguide B']).drop(columns=['Other']).rename(columns={'Waveguide A': 'RF System A', 'Waveguide B': 'RF System B'}).rename(columns={'Power Reading [V]': 'RF Power Detector Reading [V]'}, level=1).set_index(general_index).sort_index(level='Elapsed Time [s]')

rf_system_power_df.columns.names = ['RF System', 'Data Field']

max_det_v = rf_system_power_df.max().max()
min_det_v = rf_system_power_df.min().min()

# Obtain cubic spline for the power detector calibration data
fig, ax1 = plt.subplots()
det_calib_cspline_func, power_calib_pl = get_krytar_109B_calib(min_det_v, max_det_v, rf_frequency_MHz = 910, ax=ax1)
plt.show()

# Convert Detector Voltages to RF power in dBm
rf_system_power_df = rf_system_power_df.join(rf_system_power_df.transform(det_calib_cspline_func).rename(columns={'RF Power Detector Reading [V]':'Detected RF Power [dBm]'}, level='Data Field')).sort_index(axis='columns')

# Convert dBm to mW
rf_system_power_df = rf_system_power_df.join(rf_system_power_df.loc[slice(None), (slice(None), ['Detected RF Power [dBm]'])].transform(lambda x: 10**(x/10)).rename(columns={'Detected RF Power [dBm]':'Detected RF Power [mW]'}, level='Data Field')).sort_index(axis='columns')

#%%
# We now want to see by how much the power in each system has changed/drifted while acquiring data. The field with power in dBm is not needed for this calculation

# We first select data that corresponds to the first repeat only. Then we select only first occurences (in time) of each RF frequency.
rf_system_power_repeat_1 = rf_system_power_df.loc[slice(None),(slice(None), ['Detected RF Power [mW]', 'RF Power Detector Reading [V]'])].reset_index().set_index('Elapsed Time [s]').sort_index()

rf_system_power_initial_df = rf_system_power_repeat_1.loc[rf_system_power_repeat_1[rf_system_power_repeat_1['Repeat'] == 1]['Waveguide Carrier Frequency [MHz]'].sort_index().drop_duplicates(keep='first').sort_index().index]

# We need to remove all other index levels, except that of the 'Waveguide Carrier Frequency [MHz]'.

rf_system_power_initial_column_list = list(general_index.names)
rf_system_power_initial_column_list.remove('Waveguide Carrier Frequency [MHz]')

rf_system_power_initial_df = rf_system_power_initial_df.reset_index().set_index('Waveguide Carrier Frequency [MHz]').sort_index().drop(columns=rf_system_power_initial_column_list, level='RF System')

# We now calculate the fractional change in the RF power detector voltage and detector power given in parts per thousand [ppt]

rf_system_power_fract_change_df = (rf_system_power_df.loc[slice(None),(slice(None), ['Detected RF Power [mW]', 'RF Power Detector Reading [V]'])].reset_index().set_index(list(general_index.names)) - rf_system_power_initial_df)/rf_system_power_initial_df * 1E3

# Renaming columns and joining with the main rf power dataframe.
rf_system_power_df = rf_system_power_df.join(rf_system_power_fract_change_df.rename(columns={'Detected RF Power [mW]': 'Fractional Change In RF Power [ppt]', 'RF Power Detector Reading [V]': 'Fractional Change In RF Detector Voltage [ppt]'}, level='Data Field')).sort_index(axis='columns')
#%%
rf_system_power_df
#%%
fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(10,10)

rf_system_power_df.reset_index().set_index('Index').plot.scatter(x='Elapsed Time [s]', y=('RF System A', 'Fractional Change In RF Power [ppt]'), grid=True, label='RF System A',color='C2', ax=axes[0])
axes[0].set_ylabel('Change in RF Power [ppt]')
axes[0].legend()
axes[0].xaxis.set_ticklabels([])
#axes[0].set(xticks=[])
axes[0].set_ylabel('Change in RF Power [ppt]')
axes[0].set_xlabel('')

rf_system_power_df.reset_index().set_index('Index').plot.scatter(x='Elapsed Time [s]', y=('RF System B', 'Fractional Change In RF Power [ppt]'), grid=True, label='RF System B', color='C0', ax=axes[1])

axes[1].set_ylabel('Change in RF Power [ppt]')
axes[1].legend()

plt.show()
#%%
# Grouping Quenching cavities data together
level_value = 'Post-Quench'

# Selected dataframe
post_quench_df = exp_data_frame[exp_data_frame.columns.values[match_string_start(exp_data_frame.columns.values, level_value)]]

post_quench_df = post_quench_df.rename(columns=remove_matched_string_from_start(post_quench_df.columns.values, level_value))

post_quench_df = add_level_data_frame(post_quench_df, 'Quenching Cavity', ['910', '1088', '1147'])
post_quench_df.columns.set_names('Data Field', level=1, inplace=True)

#%%
level_value = 'Pre-Quench'
# Selected dataframe
pre_quench_df = exp_data_frame[exp_data_frame.columns.values[match_string_start(exp_data_frame.columns.values, level_value)]]

pre_quench_df = pre_quench_df.rename(columns=remove_matched_string_from_start(pre_quench_df.columns.values, level_value))

pre_quench_df = add_level_data_frame(pre_quench_df, 'Quenching Cavity', ['910', '1088', '1147'])
pre_quench_df.columns.names = ['Quenching Cavity', 'Data Field']

# No need to have the state of pre-910 quenching cavity. If the state is changing, then it will be part of the index.
pre_quench_df.drop(columns=[('910','State')], inplace=True)
#%%
# Combine pre-quench and post-quench data frames together
quenching_cavities_df = pd.concat([pre_quench_df.T, post_quench_df.T], keys=['Pre-Quench','Post-Quench'], names=['Cavity Stack Type']).T.set_index(general_index)
#%%
quenching_cavities_df
#%%
# Now we group the data from the digitizers together.
# Pick all the columns that have the 'level_value' as the first characters.
level_value = 'Detector'

# Selected dataframe
detector_df = exp_data_frame[exp_data_frame.columns.values[match_string_start(exp_data_frame.columns.values, level_value)]]

detector_df = detector_df.rename(columns=remove_matched_string_from_start(detector_df.columns.values, level_value))

detector_df = add_level_data_frame(detector_df, 'Data Type', harmonic_name_list)
#%%
# Pick all the columns that have the 'level_value' as the first characters.
level_value = 'RF Power Combiner I Digi 1'

# Selected dataframe
combiner_I_digi_1_df = exp_data_frame[exp_data_frame.columns.values[match_string_start(exp_data_frame.columns.values, level_value)]]

combiner_I_digi_1_df = combiner_I_digi_1_df.rename(columns=remove_matched_string_from_start(combiner_I_digi_1_df.columns.values, level_value))

combiner_I_digi_1_df = add_level_data_frame(combiner_I_digi_1_df, 'Data Type', harmonic_name_list)
#%%
# Pick all the columns that have the 'level_value' as the first characters.
level_value = 'RF Power Combiner I Digi 2'

# Selected dataframe
combiner_I_digi_2_df = exp_data_frame[exp_data_frame.columns.values[match_string_start(exp_data_frame.columns.values, level_value)]]

combiner_I_digi_2_df = combiner_I_digi_2_df.rename(columns=remove_matched_string_from_start(combiner_I_digi_2_df.columns.values, level_value))

combiner_I_digi_2_df = add_level_data_frame(combiner_I_digi_2_df, 'Data Type', harmonic_name_list)
#%%
# Pick all the columns that have the 'level_value' as the first characters.
level_value = 'RF Power Combiner R'

# Selected dataframe
combiner_R_df = exp_data_frame[exp_data_frame.columns.values[match_string_start(exp_data_frame.columns.values, level_value)]]

combiner_R_df = combiner_R_df.rename(columns=remove_matched_string_from_start(combiner_R_df.columns.values, level_value))

combiner_R_df = add_level_data_frame(combiner_R_df, 'Data Type', harmonic_name_list)
#%%
# Combine the phasor data together
phasor_data_df = pd.concat([detector_df.T, combiner_I_digi_1_df.T, combiner_I_digi_2_df.T, combiner_R_df.T], keys = ['Detector', 'RF Combiner I Digi 1', 'RF Combiner I Digi 2', 'RF Combiner R']).T

phasor_data_df = phasor_data_df.set_index(general_index).sort_index()
phasor_data_df.columns.rename(['Source','Data Type','Data Field'], inplace=True)
#%%
phasor_data_df
#%%
def group_apply_transform(grouped, func):
    ''' I either do not understand how the grouped.transform function works or there is a glitch in its implementation. I want to apply transformation operating to every group. When I do it thought using grouped.transform then it takes forever - the editor freezes. However, if I just iterate through groups and join the resulting transformed data frames together, then it takes little time. It is possible that the transform function somehow operated on each group (= a data frame), instead of operating on the columns of the group.

    Inputs:
    :grouped: pd.grouped object
    :func: lambda function to use as the transformation

    Outputs:
    :joined_df: combined dataframe with its elements transformed
    '''
    group_list = []
    joined_df = pd.DataFrame()
    for name, group in grouped:
        group_trasformed_df = group.transform(func).T
        if joined_df.size == 0:
            joined_df = group_trasformed_df
        else:
            joined_df = joined_df.join(group_trasformed_df)
    return joined_df
#%%
# Forming the data frame with the combiners' phase difference stability vs time.
# In principle the power detectors (and the digitizer) that we use on the combiners (ZX47-**LN+ from Mini Circuits) has phase shift due to its frequency response at the offset frequency. This phase shift should be cancelled out in a similar way as for the Detector signal. The difference is that we cannot average phases for each combiner for the given averaging set separately, because these phases cannot be referenced to anything. Thus we need to calculate phase difference between RF Combiner I and RF Combiner R for every average and then find the difference between these phase differences for the same average of configurations 'A' and 'B'.
# This quantity divided by two should cancel out the phase shift due to the frequency response of the RF power detectors at the offset frequency. Overall, we do not really expect the frequency response of the power detectors to change by any appreciable amount, because they are in a temperature stabilized enclosure, but nevertheless it the correct way of looking at the phase difference between the combiners.

# We first select the combiner phase data and sort it by the elapsed time
combiner_phase_df = phasor_data_df.loc[slice(None), (['RF Combiner I Digi 2', 'RF Combiner R'], harmonic_name_list, ['Fourier Phase [Rad]'])].sort_index(level='Elapsed Time [s]')

# Consider having Combiner I phase of 0.1 rad and Combiner R phase of 6.2 rad. One can calculate the phase difference to be 0.1-6.2 = -6.1 = roughly 0.1 rad. How do I know that this is actually not 0.1+2*pi*(k+1) rad for Combiner I and 6.2 + 2*pi*k rad for combiner R? Well, even with this assumption the phase difference is still 0.1 rad. Thus it seems that it does not matter.
# What about the next set of combiner traces? They will start getting acquired at a random phase. If for the next trace we get Combiner I phase of 1 rad, then we will see Combiner R phase of 1 - 0.1 = 0.9 rad. In this case the phase difference is 1-0.9 = 0.1, which is the same answer from the one above. Thus the conclusion is that we do not have to worry about this. We just need to make sure that every calculated phase is in the 0 to 2*pi range.

# Calculate phase difference between the combiners.
combiner_phase_diff_df = convert_phase_to_2pi_range((combiner_phase_df.loc[slice(None), ('RF Combiner I Digi 2', harmonic_name_list, 'Fourier Phase [Rad]')] - combiner_phase_df.loc[slice(None), ('RF Combiner R', harmonic_name_list, 'Fourier Phase [Rad]')].values)['RF Combiner I Digi 2'])

#%%
def eliminate_combiner_freq_response(df):
    '''Perform phase subtraction of RF CH A from RF CH B. The result is 2 * (phi_FOSOF + phi_RF) with the phase shift due to frequency response of the detection system at the offset frequency eliminated.

    Analysis is performed for the specified types of averaging set averaging.
    '''

    df_A = df[df['Configuration'] == 'A']
    df_B = df[df['Configuration'] == 'B']
    ## If the data has not been fully acquired, it is possible that only 'A' or 'B' configuration data is available for one of the df's. In this case we simply set df_A and df_B to be equal.
    #if df_A.size == 0:
    #    df_A = df_B
    #if df_B.size == 0:
    #    df_B = df_A
    phase_diff_df = df_A
    phase_diff_df.drop('Configuration', axis='columns', inplace=True)

    phase_diff_df.loc[slice(None),(slice(None), 'Fourier Phase [Rad]')] = ((df_A.loc[slice(None),(slice(None), 'Fourier Phase [Rad]')] - df_B.loc[slice(None), (slice(None), 'Fourier Phase [Rad]')]).transform(convert_phase_to_2pi_range))

    #phase_diff_df.loc[slice(None),(slice(None), 'Fourier Phase [Rad]')] = ((df_A.loc[slice(None),(slice(None), 'Fourier Phase [Rad]')] - df_B.loc[slice(None), (slice(None), 'Fourier Phase [Rad]')]).transform(convert_phase_to_2pi_range)).transform(divide_and_minimize_phase, div_number=2)

    #phase_diff_df.loc[slice(None), 'Elapsed Time [s]'] = df_A.loc[slice(None), 'Elapsed Time [s]']

    #phase_diff_df.loc[slice(None),'Index'] = df_A.loc[slice(None), 'Index']

    return phase_diff_df.iloc[0]
#%%
# We now form convenient group such that the 'Configuration' is one of the columns, but not part of the group. We simply want to group the combiner phase different by the same average (+ any other parameters, except the Configuration)
combiner_phase_diff_index = list(combiner_phase_diff_df.index.names)
combiner_phase_diff_index.remove('Configuration')
combiner_phase_diff_index.remove('Elapsed Time [s]')
combiner_phase_diff_index.remove('Index')

combiner_phase_diff_df = combiner_phase_diff_df.reset_index().set_index(combiner_phase_diff_index)
combiner_phase_diff_df.columns = combiner_phase_diff_df.columns.remove_unused_levels()

# Cancel out Power detector frequency response.
combiner_phase_diff_no_freq_response_df = combiner_phase_diff_df.groupby(combiner_phase_diff_index).apply(eliminate_combiner_freq_response)


#combiner_phase_diff_no_freq_response_df = combiner_phase_diff_no_freq_response_df.transform(divide_and_minimize_phase, div_number=2)
#%%
combiner_phase_diff_no_freq_response_df
#%%
# Different waveguide frequencies have different phase difference between the combiners. For typical data set we do not have enough data for each waveguide frequency to see smooth variation of the combiners' phase difference vs time. However, we can look at the deviation of the Combiners' phase difference from its initial phase difference for each RF frequency.

phase_diff_repeat_1 = combiner_phase_diff_no_freq_response_df.reset_index().set_index('Elapsed Time [s]').sort_index()

# We first select data that corresponds to the first repeat only. Then we select only first occurences (in time) of each RF frequency.
phase_diff_initial_df = phase_diff_repeat_1.loc[phase_diff_repeat_1[phase_diff_repeat_1['Repeat'] == 1]['Waveguide Carrier Frequency [MHz]'].sort_index().drop_duplicates(keep='first').sort_index().index]

# We need to remove all other index levels, except that of the 'Waveguide Carrier Frequency [MHz]', because when subtracting from the list of phase differences, we want to subtract the same phase difference from each phase, independent of other index levels.

phase_diff_initial_column_list = list(general_index.names)
phase_diff_initial_column_list.remove('Configuration')
phase_diff_initial_column_list.remove('Waveguide Carrier Frequency [MHz]')

phase_diff_initial_df = phase_diff_initial_df.reset_index().set_index('Waveguide Carrier Frequency [MHz]').drop(columns=phase_diff_initial_column_list, level='Data Type')
# We now subtract this list of initial combiners' phase differences from the rest of the phase differences.
combiner_diff_index_list = list(general_index.names)
combiner_diff_index_list.remove('Configuration')

#%%
# The difference is given in mrad
combiner_phase_diff_variation_df = (combiner_phase_diff_no_freq_response_df.reset_index().set_index(combiner_diff_index_list) - phase_diff_initial_df).transform(convert_phase_to_2pi_range).transform(divide_and_minimize_phase, div_number=2) * 1E3

#combiner_phase_diff_variation_df = (combiner_phase_diff_no_freq_response_df.reset_index().set_index(combiner_diff_index_list) - phase_diff_initial_df)* 1E3
#%%
combiner_phase_diff_variation_df
#%%
combiner_phase_diff_variation_df
#%%
combiner_phase_diff_variation_df.rename( columns={'Fourier Phase [Rad]': 'Phase Change [mrad]'}, level='Data Field', inplace=True)
#%%

combiner_difference_df = combiner_phase_diff_no_freq_response_df.reset_index().set_index(combiner_diff_index_list).join(combiner_phase_diff_variation_df).sort_index(axis='columns')
#%%

plt.figure()
combiner_difference_df.reset_index().set_index('Index').plot.scatter(x='Elapsed Time [s]', y=('First Harmonic', 'Phase Change [mrad]'))
plt.show()
#%%
# Another group with the delay between the Digitizers

# Calculate the delay between triggering the digitizers. We are calculating Digitizer 2 - Digitizer 1 phase delay. The reason for this is that we expect the phase delay to be positive: Digitizer 1 should trigger faster than Digitizer 2, thus the initial phase of Digitizer 2 should be larger than of Digitizer 1, thus Digitizer 2 - Digitizer 1 has to be positive.

digi_2_from_digi_1_delay_df = phasor_data_df.loc[slice(None), ('RF Combiner I Digi 2', harmonic_name_list, ['Fourier Frequency [Hz]', 'Fourier Phase [Rad]'])]['RF Combiner I Digi 2']

# Shift the phases into proper multiples of 2pi.

phasor_data_for_delay_df_group = phasor_data_df.loc[slice(None), (['RF Combiner I Digi 1','RF Combiner I Digi 2'], harmonic_name_list, 'Fourier Phase [Rad]')].T.groupby('Data Type')

# This is how I was shifting the phases initially. However, I either do not understand how the grouped.transform function works or there is a glitch in it - but the editor glitches when I use it on the large data set. Because of that I had to make my own function that uses .transform on each group = data frame of the grouped object.
#phasor_data_for_delay_df = phasor_data_df.loc[slice(None), (['RF Combiner I Digi 1','RF Combiner I Digi 2'], harmonic_name_list, 'Fourier Phase [Rad]')].T.groupby('Data Type').transform(lambda x: phases_shift(x)[0]).T

phasor_data_for_delay_df = group_apply_transform(phasor_data_for_delay_df_group, lambda x: phases_shift(x)[0])


# Calculate the phase difference between the Digitizers at the offset frequency harmonics
digi_2_from_digi_1_delay_df.loc[slice(None), (harmonic_name_list, 'Fourier Phase [Rad]')] = (phasor_data_for_delay_df.loc[slice(None), ('RF Combiner I Digi 2', harmonic_name_list, 'Fourier Phase [Rad]')] - phasor_data_for_delay_df.loc[slice(None), ('RF Combiner I Digi 1', harmonic_name_list, 'Fourier Phase [Rad]')].values)['RF Combiner I Digi 2']


# Calculate the Delay in microseconds and samples between the digitizers
for harmonic_value in harmonic_name_list:
    digi_2_from_digi_1_delay_df[harmonic_value,'Delay [Sample]'] = digi_2_from_digi_1_delay_df[harmonic_value, 'Fourier Phase [Rad]'] / (2*np.pi*digi_2_from_digi_1_delay_df[harmonic_value, 'Fourier Frequency [Hz]']) * sampling_rate

    digi_2_from_digi_1_delay_df[harmonic_value,'Delay [us]'] = digi_2_from_digi_1_delay_df[harmonic_value, 'Fourier Phase [Rad]'] / (2*np.pi*digi_2_from_digi_1_delay_df[harmonic_value, 'Fourier Frequency [Hz]'])*1E6

digi_2_from_digi_1_delay_df = digi_2_from_digi_1_delay_df.sort_index(axis='columns')


# The averaging is done for both the Delay in digitizer samples and the Delay in microseconds.
digi_2_from_digi_1_mean_delay_df = digi_2_from_digi_1_delay_df.loc[slice(None), (harmonic_name_list, 'Delay [Sample]')].T.aggregate([np.mean, lambda x: np.std(x, ddof=1)/np.sqrt(x.shape[0])]).T.rename(columns={'mean':'Mean Inter Digitizer Delay [Sample]', '<lambda>':'Mean Inter Digitizer Delay STD [Sample]'}).join(
digi_2_from_digi_1_delay_df.loc[slice(None), (harmonic_name_list, 'Delay [us]')].T.aggregate([np.mean, lambda x: np.std(x, ddof=1)/np.sqrt(x.shape[0])]).T.rename(columns={'mean':'Mean Inter Digitizer Delay [us]', '<lambda>':'Mean Inter Digitizer Delay STD [us]'}))

digi_2_from_digi_1_mean_delay_df = digi_2_from_digi_1_mean_delay_df.set_index(general_index).sort_index()
digi_2_from_digi_1_mean_delay_df.columns.set_names('Data Field', inplace=True)

#%%
digi_2_from_digi_1_mean_delay_df
#%%
plt.figure()
digi_2_from_digi_1_mean_delay_df.reset_index().set_index('Index').plot.scatter(x='Elapsed Time [s]', y='Mean Inter Digitizer Delay [Sample]')
plt.show()
#%%
# Main FOSOF data frame. Here we calculate Detector phases relative to various phase references.
phase_diff_data_df = phasor_data_df[['Detector']].rename(columns={'Detector': 'RF Combiner I Reference'},level=0)

phase_diff_data_df = phase_diff_data_df.join(phasor_data_df[['Detector']].rename(columns={'Detector': 'RF Combiner R Reference'},level=0))

# Trace Filename is not needed here.
phase_diff_data_df
phase_diff_data_df.drop('Trace Filename', axis='columns', level=2, inplace=True)

# Calculate Detector in RF Combiner I and RF Combiner R phase differences.
phase_diff_data_df.loc[slice(None), ('RF Combiner I Reference', harmonic_name_list, 'Fourier Phase [Rad]')] = (phasor_data_df.loc[slice(None), ('Detector', harmonic_name_list, 'Fourier Phase [Rad]')] - phasor_data_df.loc[slice(None), ('RF Combiner I Digi 1', harmonic_name_list, 'Fourier Phase [Rad]')].values).rename(columns={'Detector': 'RF Combiner I Reference'}).transform(lambda x: convert_phase_to_2pi_range(x))

phase_diff_data_df.loc[slice(None), ('RF Combiner R Reference', harmonic_name_list, 'Fourier Phase [Rad]')] = (
phasor_data_df.loc[slice(None), ('Detector', harmonic_name_list, 'Fourier Phase [Rad]')] - \
phasor_data_df.loc[slice(None), ('RF Combiner R', harmonic_name_list, 'Fourier Phase [Rad]')].values +\
digi_2_from_digi_1_delay_df.loc[slice(None), (harmonic_name_list, 'Fourier Phase [Rad]')].values
).rename(columns={'Detector': 'RF Combiner R Reference'}).transform(lambda x: convert_phase_to_2pi_range(x))

# We dropped the "Trace Filename" column, however, for some reason it is still contained in the index of columns. This function gets rid of it.
phase_diff_data_df.columns = phase_diff_data_df.columns.remove_unused_levels()

#%%
# List of columns by which to group the phase data for averaging set averaging
averaging_set_grouping_list = list(general_index.names)
averaging_set_grouping_list.remove('Average')
averaging_set_grouping_list.remove('Index')
averaging_set_grouping_list.remove('Elapsed Time [s]')

phase_diff_group = phase_diff_data_df.groupby(averaging_set_grouping_list)
# For the subsequent analysis we assume that the phases of the phasors are normally distributed (for the Phase Averaging method), as well as A*cos(phi) and A*sin(phi) of the phasors - its x and y components, where A = amplitude of the given phasor and phi is its phase (for the Phasor Averaging and Phasor Averaging Relative To DC ). We also assume that the average amplitude of the phasors relative to DC are normally distributed (For Phasor Averaging Relative To DC, but not for calculation of the phase, but for estimation of FOSOF relative amplitude, when averaging amplitude relative to dc obtained from each averaging set for the given Waveguide carrier frequency).
# These assumptions is how we can use the formulas for the best estimate of the mean and standard deviation (with N-1 factor instead of N). If this is not true, then we might need to use different type of analysis.
# Notice that it seems that NONE of these quantities exactly normally distributed. But it seems to be intuitive, that if the signal is not too weak, then the errors have to be small and in that case the quantities are approximately normally distributed.
#%%
def analyze_averaging_set(df):
    ''' Averages data for the phasors in the given averaging set.

    Various types of averaging are performed. So far I have three averaging methods: Phasor Averaging, Phase Averaging, and Phasor Averaging Relative To DC.
    '''


    # Columns names of the phase references. Mostly we expect only two elements here: Combiner I and Combiner R reference
    reference_type_list = df.columns.levels[0].values

    # Column names corresponding to phasors at various Fourier harmonics
    harmonics_names_list = df.columns.levels[1].drop('Other').values

    # For each reference and harmonic perform various averaging schemes
    av_container_list = []
    for reference_type in reference_type_list:

        reference_data_df = df[reference_type]

        # Series with the DC values common to all harmonics, of course.
        dc_s = reference_data_df['Other', 'DC [V]']

        av_set_harmonic_type_list = []

        for harmonic_level in harmonics_names_list:

            harmonic_data_df = reference_data_df[harmonic_level]

            amp_to_dc_ratio_s = harmonic_data_df['Fourier Amplitude [V]']/dc_s

            # List for storing Series with different types of averaging
            phase_av_type_list = []
            # Different types of averaging are performed below
            # The methods presented are not perfect: the Lyman-alpha detector frequency response at the offset frequency seems to have some DC dependence and thus the average phase changes with DC, thus making it quite impossible to perform any averaging if we have large DC changes (what is large? I do not know) between traces of the averaging set. Thus all of the methods here assume that this issue does not exist.
            # ----------------------------------------
            # Phasor Averaging
            # The assumption is that DC of the signal is the same for the duration of the averaging set. This way we can simply take the amplitudes and phases of the phasors and average them together to obtain single averaged phasor.
            phasor_av_s = pd.Series(mean_phasor(amp_arr=harmonic_data_df['Fourier Amplitude [V]'], phase_arr=harmonic_data_df['Fourier Phase [Rad]']))
            phasor_av_s.rename(index={'Number Of Averaged Phasors': 'Number Of Averaged Data Points'}, inplace=True)
            # Adding SNR of the average phasor. This is just an estimate. We assume that the true SNR is the same for each phasor in the averaging set. Thus, we can calculate the mean of the SNR. However, averaging N phasors results in the SNR going up by a factor of N**0.5. The reason for this is that the resulting phasor has, real and imaginary components as the mean of the set {A_n_i * cos(phi_n_i), A_n_i * sin(phi_n_i)}, where phi_n_i is random phase, and A_n_i is the noise amplitude. In this case the amplitude of the resulting averaged phasor is smaller by a factor of N**0.5, assuming that A_n_i is a constant. I have tested this in Mathematica and it seems indeed to be the case, except that it seems to go down as about 1.1*N**0.5.

            # We are testing if there are any SNR values that are NaN. In this case it is assumed that the averaging set has large noise (or the signal of interest is not there)
            if harmonic_data_df['SNR'][harmonic_data_df['SNR'].isnull()].size == 0:
                phasor_av_s['SNR'] = harmonic_data_df['SNR'].mean() * np.sqrt(harmonic_data_df['SNR'].size)
            else:
                phasor_av_s['SNR'] = np.nan

            phasor_av_s.index.name = 'Data Field'

            phasor_av_s = pd.concat([phasor_av_s], keys=['Phasor Averaging'], names=['Averaging Type'])

            phase_av_type_list.append(phasor_av_s)
            # ----------------------------------------
            # Phase averaging
            # Here we simply average the phases together. Amplitudes are not used. We can, however, calculate average Amplitude-to-DC ratio. This, when SNR is high, should, in principle, give better results, compared to phasor averaging, when DC is not stable during the averaging set, but changes from one trace to another.
            phase_data_s = phases_shift(harmonic_data_df['Fourier Phase [Rad]'])
            phase_av_s = pd.Series({
                'Phase [Rad]': np.mean(phase_data_s),
                'Phase STD [Rad]': np.std(phase_data_s, ddof=1),
                'Number Of Averaged Data Points': phase_data_s.size,
                'Range Of Phases [Deg]': (np.max(phase_data_s)-np.min(phase_data_s))*180/np.pi})

            phase_av_s.index.name = 'Data Field'

            phase_av_s = pd.concat([phase_av_s], keys=['Phase Averaging'], names=['Averaging Type'])


            phase_av_type_list.append(phase_av_s)

            # ----------------------------------------
            # Phasor averaging relative to DC signal level.
            # Same as phasor averaging, however instead of the phasor amplitudes we use amplitude-to-DC ratios for individual phasors. The reason for doing this is to eliminate the possibility of DC changing between traces for the given averaging set and skewing our average, because it would mean that the amplitude of the average phasor is not the same for all the phasors of the averaging set\. Now we should be insensitive to this, assuming that the mean phase (true phase) does not depend on DC. We still have the issue that the signal can be more or less noisy for different DC levels, therefore changing the standard deviation from trace to trace, hence making our formula for the best estimate of mean and standard deviation not entirely correct. (We assume Gaussian distribution here).
            phasor_av_relative_to_dc_s = pd.Series(mean_phasor(amp_arr=amp_to_dc_ratio_s, phase_arr=harmonic_data_df['Fourier Phase [Rad]']))

            phasor_av_relative_to_dc_s.rename({'Amplitude': 'Amplitude Relative To DC', 'Amplitude STD': 'Amplitude Relative To DC STD', 'Number Of Averaged Phasors': 'Number Of Averaged Data Points'}, inplace=True)

            phasor_av_relative_to_dc_s.index.name = 'Data Field'

            phasor_av_relative_to_dc_s = pd.concat([phasor_av_relative_to_dc_s], keys=['Phasor Averaging Relative To DC'], names=['Averaging Type'])

            phase_av_type_list.append(phasor_av_relative_to_dc_s)
            # ----------------------------------------

            av_set_harmonic_type_list.append(pd.concat([pd.concat(phase_av_type_list)], keys=[harmonic_level], names=['Fourier Harmonic']))

        av_set_harmonic_type_s = pd.concat([pd.concat(av_set_harmonic_type_list)], keys=[reference_type], names=['Phase Reference Type'])

        av_container_list.append(av_set_harmonic_type_s)

    av_set_combined_s = pd.concat(av_container_list)

    return av_set_combined_s

#%%
# Average phasor data for every averaging set.
phase_av_set_df = phase_diff_group.apply(analyze_averaging_set)
#%%
phase_av_set_df
#%%
# We assume that for the given repeat, offset frequency, B Field, and pre-910 state all of the phasors obtained should have the same average SNR and thus the same true standard deviation for phase and amplitude at the given offset frequency. That is why we calculate the RMS quantities below. Now, this assumption might be wrong, especially when we have the beam that deteriorates over time or when the beam is such that it abruptly changes its mode of operation (we see it sometimes). When we have scan through offset frequencies and other additional parameters, then it takes even more time to acquire single repeat, thus the chance for the standard deviation to change is even larger. The additional assumption is that the RF scan range is small enough for there to be no appreciable variation in SNR with the RF frequency.

rms_repeat_grouping_list = list(general_index.names)
rms_repeat_grouping_list.remove('Index')
rms_repeat_grouping_list.remove('Elapsed Time [s]')
rms_repeat_grouping_list.remove('Waveguide Carrier Frequency [MHz]')
rms_repeat_grouping_list.remove('Average')
rms_repeat_grouping_list.remove('Configuration')
#%%

phase_av_set_df_index_names_list = list(phase_av_set_df.index.names)

phase_av_set_group = phase_av_set_df.groupby(rms_repeat_grouping_list)

data_rms_std_df = phase_av_set_group.apply(lambda df: pd.Series(pooled_std(df.loc[slice(None), (slice(None), slice(None), slice(None), 'Phase STD [Rad]')].values
, df.loc[slice(None), (slice(None), slice(None), slice(None), 'Number Of Averaged Data Points')].values), index=df.loc[slice(None), (slice(None), slice(None), slice(None), 'Phase STD [Rad]')].columns))


data_rms_std_df.rename(columns={'Phase STD [Rad]': 'Phase RMS Repeat STD [Rad]'}, level='Data Field', inplace=True)

# We have to make sure that the data frames we are trying to join have the same index columns
phase_av_set_next_df = phase_av_set_df.reset_index().set_index(rms_repeat_grouping_list).join(data_rms_std_df, how='inner').sort_index(axis='columns').reset_index().set_index(phase_av_set_df_index_names_list)

phase_av_set_next_df = phase_av_set_next_df.sort_index(axis='columns')
phase_av_set_next_df.columns = phase_av_set_next_df.columns.remove_unused_levels()

#%%
# Another way to calculate the STD of the phases is to assume that the standard deviation is the same for the data for the same averaging set, which includes phases obtained for the same B field, pre-quench 910 state, repeat and waveguide carrier frequency. The configuration can be either A and B, of course.

# Note that later on, when calculating A-B phases, the standard deviation that we get from combining this types of STD and simply STD determining for each configuration and averaging set, are exactly the same. Thus in a sense we are not getting any advantage of performing this type of calculation.

rms_av_set_grouping_list = list(general_index.names)
rms_av_set_grouping_list.remove('Index')
rms_av_set_grouping_list.remove('Elapsed Time [s]')
rms_av_set_grouping_list.remove('Average')
rms_av_set_grouping_list.remove('Configuration')

phase_av_set_group = phase_av_set_next_df.groupby(rms_av_set_grouping_list)

data_rms_std_df = phase_av_set_group.apply(
            lambda df: pd.Series(pooled_std(df.loc[slice(None), (slice(None), slice(None), slice(None), 'Phase STD [Rad]')].values
            , df.loc[slice(None), (slice(None), slice(None), slice(None), 'Number Of Averaged Data Points')].values), index=df.loc[slice(None), (slice(None), slice(None), slice(None), 'Phase STD [Rad]')].columns)
            )

data_rms_std_df.rename(columns={'Phase STD [Rad]': 'Phase RMS Averaging Set STD [Rad]'}, level='Data Field', inplace=True)


phase_av_set_next_df = phase_av_set_next_df.reset_index().set_index(rms_av_set_grouping_list).join(data_rms_std_df, how='inner').reset_index().set_index(phase_av_set_df_index_names_list).sort_index(axis='columns')

phase_av_set_next_df.columns = phase_av_set_next_df.columns.remove_unused_levels()

#%%
# These calculations (determination of RMS STD) are also performed for the Amplitude Relative To DC. Here we expect that this quantity should have uncertainty independent of DC level, thus it makes sense to use it as relative FOSOF amplitude.

# RMS Repeat STD
phase_av_set_group = phase_av_set_next_df.groupby(rms_repeat_grouping_list)

data_rms_std_df = phase_av_set_group.apply(lambda df: pd.Series(pooled_std(df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC STD')].values
, df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC', 'Number Of Averaged Data Points')].values), index=df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC','Amplitude Relative To DC STD')].columns))

data_rms_std_df.rename(columns={'Amplitude Relative To DC STD': 'Amplitude Relative To DC RMS Repeat STD'}, level='Data Field', inplace=True)

phase_av_set_next_df = phase_av_set_next_df.reset_index().set_index(rms_repeat_grouping_list).join(data_rms_std_df, how='inner').reset_index().set_index(phase_av_set_df_index_names_list).sort_index(axis='columns')
#%%
# RMS Averaging Set STD
phase_av_set_group = phase_av_set_df.groupby(rms_av_set_grouping_list)

data_rms_std_df = phase_av_set_group.apply(lambda df: pd.Series(pooled_std(df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC STD')].values
, df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC', 'Number Of Averaged Data Points')].values), index=df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC','Amplitude Relative To DC STD')].columns)
            )

data_rms_std_df.rename(columns={'Amplitude Relative To DC STD': 'Amplitude Relative To DC RMS Averaging Set STD'}, level='Data Field', inplace=True)

phase_av_set_next_df = phase_av_set_next_df.reset_index().set_index(rms_av_set_grouping_list).join(data_rms_std_df, how='inner').reset_index().set_index(phase_av_set_df_index_names_list).sort_index(axis='columns')

phase_av_set_next_df.columns = phase_av_set_next_df.columns.remove_unused_levels()
#%%
# Calculate phase STD of the mean.

column_sigma_list = [
        'Phase RMS Repeat STD [Rad]',
        #'Phase RMS STD With Covariance [Rad]',
        'Phase STD [Rad]',
        'Phase RMS Averaging Set STD [Rad]']#,
        #'Phase STD With Covariance [Rad]']

column_mean_sigma_list = [
        'Phase RMS Repeat STDOM [Rad]',
        #'Phase Mean RMS STD With Covariance [Rad]',
        'Phase STDOM [Rad]',
        'Phase RMS Averaging Set STDOM [Rad]']#,
        #'Phase Mean STD With Covariance [Rad]']

# List of STD's for Phasor Averaging Relative to DC method. We include Ampltude To DC STD's here.
column_sigma_rel_to_dc_list = [
        'Phase RMS Repeat STD [Rad]',
        #'Phase RMS STD With Covariance [Rad]',
        'Phase STD [Rad]',
        'Phase RMS Averaging Set STD [Rad]',
        'Amplitude Relative To DC RMS Repeat STD',
        #'Phase RMS STD With Covariance [Rad]',
        'Amplitude Relative To DC STD',
        'Amplitude Relative To DC RMS Averaging Set STD']#,
        #'Phase STD With Covariance [Rad]']

column_mean_sigma_rel_to_dc_list = [
        'Phase RMS Repeat STDOM [Rad]',
        #'Phase Mean RMS STD With Covariance [Rad]',
        'Phase STDOM [Rad]',
        'Phase RMS Averaging Set STDOM [Rad]',
        'Amplitude Relative To DC RMS Repeat STDOM',
        #'Phase Mean RMS STD With Covariance [Rad]',
        'Amplitude Relative To DC STDOM',
        'Amplitude Relative To DC RMS Averaging Set STDOM']#,
        #'Phase Mean STD With Covariance [Rad]']

averaging_type_list = ['Phase Averaging', 'Phasor Averaging', 'Phasor Averaging Relative To DC']

n_averages_col_name = 'Number Of Averaged Data Points'

for reference_type in phase_av_set_next_df.columns.levels[0].values:
    for harmonic_value in harmonic_name_list:
        for averaging_type in averaging_type_list:

            if averaging_type == 'Phasor Averaging Relative To DC':
                column_sigma_to_use_list = column_sigma_rel_to_dc_list
                column_mean_sigma_to_use_list = column_mean_sigma_rel_to_dc_list
            else:
                column_sigma_to_use_list = column_sigma_list
                column_mean_sigma_to_use_list = column_mean_sigma_list

            for column_sigma_index in range(len(column_sigma_to_use_list)):
                phase_av_set_next_df[reference_type, harmonic_value, averaging_type, column_mean_sigma_to_use_list[column_sigma_index]] = phase_av_set_next_df[reference_type, harmonic_value, averaging_type, column_sigma_to_use_list[column_sigma_index]] / np.sqrt(phase_av_set_next_df[reference_type, harmonic_value, averaging_type, n_averages_col_name])

phase_av_set_next_df = phase_av_set_next_df.sort_index(axis='columns')
#%%
phase_av_set_next_df
#%%
# We now want to eliminate the frequency response of the detection system at the offset frequency and its harmonics.

rms_no_config_grouping_list = list(phase_av_set_next_df.index.names)
rms_no_config_grouping_list.remove('Configuration')
rms_no_config_grouping_list

# Transform dataframe to convenient multi index by disregarding the Configuration index
phase_av_set_next_df = phase_av_set_next_df.reset_index().set_index(rms_no_config_grouping_list)

# For some reason, after reindexing, the 'Repeat' is contained as a column in the phase_av_set_next_df.columns. It causes a problem with the following .groupby function, since the function sees that the Repeat is contained in both indeces and columns.
phase_av_set_next_df.columns = phase_av_set_next_df.columns.remove_unused_levels()

phase_av_set_group = phase_av_set_next_df.groupby(rms_no_config_grouping_list)

#%%
def eliminate_freq_response(df, columns_phase_std_dict, column_phase):
    '''Perform phase subtraction of RF CH A from RF CH B. The result is 2*(phi_FOSOF + phi_RF) with the phase shift due to frequency response of the detection system at the offset frequency eliminated.

    Analysis is performed for the specified types of averaging set averaging.
    '''

    df_A = df[df['Configuration'] == 'A']
    df_B = df[df['Configuration'] == 'B']

    phase_diff_df = (df_A.loc[slice(None),(slice(None), slice(None), slice(None), column_phase)] - df_B.loc[slice(None), (slice(None), slice(None), slice(None), column_phase)]).transform(convert_phase_to_2pi_range)

    std_df = np.sqrt(df_A.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2 + df_B.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2)

    std_df = std_df.rename(columns=columns_phase_std_dict, level='Data Field')

    return phase_diff_df.join(std_df).sort_index(axis='columns').iloc[0]
#%%
def calculate_freq_response(self, df, columns_phase_std_dict, column_phase):
    '''Perform phase addition of RF CH A from RF CH B. The result is (phi_FOSOF + phi_RF)/2 = delta_phi, which is the phase shift due to the frequency response of the detection system at the offset frequency.

    Analysis is performed for the specified types of averaging set averaging.
    '''

    df_A = df[df['Configuration'] == 'A']
    df_B = df[df['Configuration'] == 'B']

    phase_sum_df = (df_A.loc[slice(None),(slice(None), slice(None), slice(None), column_phase)] + df_B.loc[slice(None), (slice(None), slice(None), slice(None), column_phase)]).transform(convert_phase_to_2pi_range).transform(divide_and_minimize_phase, div_number=2)

    std_df = np.sqrt(df_A.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2 + df_B.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2)/2

    std_df = std_df.rename(columns=columns_phase_std_dict, level='Data Field')

    return phase_sum_df.join(std_df).sort_index(axis='columns').iloc[0]
#%%
columns_phase_std_dict = {
        'Phase STDOM [Rad]': 'Phase STD [Rad]',
        'Phase RMS Repeat STDOM [Rad]': 'Phase RMS Repeat STD [Rad]',
        'Phase RMS Averaging Set STDOM [Rad]': 'Phase RMS Averaging Set STD [Rad]'}
column_phase = 'Phase [Rad]'

rms_no_config_grouping_list.remove('Waveguide Carrier Frequency [MHz]')
rms_no_config_grouping_list.insert(0, 'Waveguide Carrier Frequency [MHz]')
rms_no_config_grouping_list

# Perform RF CH A and RF CH B phase subtraction to get rid of the detection system frequency response at the offset frequency.
# We also reverse the order of some of the indeces for convenience.
phase_A_minus_B_df = phase_av_set_group.apply(eliminate_freq_response, columns_phase_std_dict, column_phase).reset_index().set_index(rms_no_config_grouping_list).sort_index(level=0)

phase_A_plus_B_df = phase_av_set_group.apply(calculate_freq_response, columns_phase_std_dict, column_phase).reset_index().set_index(rms_no_config_grouping_list).sort_index(level=0)
#%%
phase_A_minus_B_df
#%%
phase_A_plus_B_df
#%%
# Perform averaging of phases across all of the Repeats for every RF Carrier Frequency.
# First we group the data frame by the carrier frequency to collect the phases from all of the repeats together for every carrier frequency.
rms_av_over_repeat_grouping_list = list(phase_A_minus_B_df.index.names)
rms_av_over_repeat_grouping_list.remove('Repeat')

rms_final_averaged_phasor_data_column_list = copy.copy(rms_av_over_repeat_grouping_list)

rms_final_averaged_phasor_data_column_list.remove('Waveguide Carrier Frequency [MHz]')
rms_final_averaged_phasor_data_column_list.append('Waveguide Carrier Frequency [MHz]')

phase_A_minus_B_df.columns = phase_A_minus_B_df.columns.remove_unused_levels()
phase_A_minus_B_df_group = phase_A_minus_B_df.groupby(rms_av_over_repeat_grouping_list)
#%%
def average_data_field(df, reference_type_list, harmonics_names_list, averaging_type_list, data_column, columns_dict, data_is_phases=False):
    ''' Perform averaging of the given data field.

    The averaging of data field is performed by performing a weighted least-squares straight line (slope = 0) fit to the data. This is equivalent to weighted average. We also extract Reduced Chi-Squared that tells us if our estimates for standard deviation are reasonable.

    It is assumed that the data set has the following structure: first level of columns is the reference_type_list = phase reference type for FOSOF. Second level consists of the Fourier Harmonics used, specified in the harmonics_names_list. Third level is the list of different types of averaging that was performed on the data frame before, specified in the averaging_type_list. The variable data_column is the column name of the variable that needs to get averaged. The columns_dict is the dictionary of what standard deviations to use from 4th level. Key = how this type of standard deviation will be named in the output series, value = current name of that standard deviation type in the data frame. If we are trying to average phases, then they need to be properly shifted: set the data_is_phases boolean to True.
    '''

    average_reference_type_list = []
    for reference_type in reference_type_list:
        reference_data_df = df[reference_type]

        average_harmonic_list = []
        for harmonic_name in harmonics_names_list:
            harmonic_data_df = reference_data_df[harmonic_name]


            averaging_type_data_list = []
            for averaging_type in averaging_type_list:

                data_df = harmonic_data_df[averaging_type]
                std_type_list = []
                if data_is_phases:
                    data_arr = phases_shift(data_df[data_column])[0]
                else:
                    data_arr = data_df[data_column]

                for std_output_type, std_input_type in columns_dict.iteritems():

                    std_arr = data_df[std_input_type]
                    av_s = straight_line_fit_params(data_arr, std_arr)
                    #av_s.rename({'Weighted Mean':'RF CH A - RF CH B Weighted Averaged Phase [Rad]','Weighted STD':'RF CH A - RF CH B Weighted Averaged Phase STD [Rad]'}, inplace=True)
                    av_s.index.name = 'Data Field'

                    av_s = pd.concat([av_s], keys=[std_output_type], names=['STD Type'])
                    std_type_list.append(av_s)

                averaging_type_data_list.append(pd.concat([pd.concat(std_type_list)], keys=[averaging_type], names=['Averaging Type']))

            average_harmonic_list.append(pd.concat([pd.concat(averaging_type_data_list)], keys=[harmonic_name], names=['Fourier Harmonic']))

        average_reference_type_list.append(pd.concat([pd.concat(average_harmonic_list)], keys=[reference_type], names=['Phase Reference Type']))

    average_reference_type_s = pd.concat(average_reference_type_list)

    return average_reference_type_s
#%%
reference_type_list = ['RF Combiner I Reference', 'RF Combiner R Reference']
averaging_type_list = ['Phase Averaging', 'Phasor Averaging', 'Phasor Averaging Relative To DC']

data_column = 'Phase [Rad]'

columns_dict = {
    'Phase RMS Repeat STD': 'Phase RMS Repeat STD [Rad]',
    'Phase RMS Averaging Set STD': 'Phase RMS Averaging Set STD [Rad]',
    'Phase STD': 'Phase STD [Rad]'}

# Average the phases over all repeats for every Waveguide Carrier Frequency.
fosof_phases_df = phase_A_minus_B_df_group.apply(average_data_field, reference_type_list, harmonic_name_list, averaging_type_list, data_column, columns_dict, True)

fosof_phases_df = fosof_phases_df.reset_index().set_index(rms_final_averaged_phasor_data_column_list).sort_index(axis='index')
#%%
fosof_phases_df
#%%
# We now want to average set of amplitude relative to dc values obtained for every averaging set (both 'A' and 'B' configurations) for all repeats for given RF frequency.
ampl_av_set_df = phase_av_set_next_df.reset_index().set_index(rms_no_config_grouping_list).sort_index(axis='index').loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC')]
ampl_av_set_df.columns = ampl_av_set_df.columns.remove_unused_levels()
ampl_av_set_df_group = ampl_av_set_df.groupby(rms_av_over_repeat_grouping_list)

#%%
averaging_type_list = ['Phasor Averaging Relative To DC']
data_column = 'Amplitude Relative To DC'
columns_dict = {
    'Amplitude Relative To DC RMS Repeat STD': 'Amplitude Relative To DC RMS Repeat STDOM',
    'Amplitude Relative To DC RMS Averaging Set STD': 'Amplitude Relative To DC RMS Averaging Set STDOM',
    'Amplitude Relative To DC STD': 'Amplitude Relative To DC STDOM'}

fosof_ampl_df = ampl_av_set_df_group.apply(average_data_field, reference_type_list, harmonic_name_list, averaging_type_list, data_column, columns_dict).reset_index().set_index(rms_final_averaged_phasor_data_column_list).sort_index(axis='index')
#%%
fosof_ampl_df
#%%
# For B field scan and pre-quench 910 ON/OFF data set we are not taking 0 and pi configurations, since we are not interested in the absolute resonant frequencies, but their difference for different values of B field and pre-quench 910 state.

# For pre-910 state switching data set.

# We want to calculate for every averaging set the difference in obtained phases for the case when pre-quench 910 state is ON and OFF.

pre_910_state_index_list = list(phase_A_minus_B_df.index.names)
pre_910_state_index_list.remove('Pre-Quench 910 State')

phase_A_minus_B_for_pre_910_state_df = phase_A_minus_B_df.reset_index().set_index(pre_910_state_index_list).sort_index()

phase_A_minus_B_for_pre_910_state_df.columns = phase_A_minus_B_for_pre_910_state_df.columns.remove_unused_levels()

pre_910_state_switching_df_group = phase_A_minus_B_for_pre_910_state_df.groupby(pre_910_state_index_list)

#%%
def pre_910_state_subtract(df, columns_phase_std_dict, column_phase):
    '''Perform phase subtraction of phases for pre-quench 910 state being 'on' and 'off' for each averaging set of data.

    Analysis is performed for the specified types of averaging set averaging.
    '''

    df_on = df[df['Pre-Quench 910 State'] == 'on']
    df_off = df[df['Pre-Quench 910 State'] == 'off']

    phase_diff_df = (df_on.loc[slice(None), (slice(None), slice(None), slice(None), column_phase)] - df_off.loc[slice(None),(slice(None), slice(None), slice(None), column_phase)]).transform(convert_phase_to_2pi_range)

    std_df = np.sqrt(df_on.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2 + df_off.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2)

    std_df = std_df.rename(columns=columns_phase_std_dict, level='Data Field')

    return phase_diff_df.join(std_df).sort_index(axis='columns').iloc[0]
#%%
columns_phase_std_dict = {
        'Phase STD [Rad]': 'Phase STD [Rad]',
        'Phase RMS Repeat STD [Rad]': 'Phase RMS Repeat STD [Rad]',
        'Phase RMS Averaging Set STD [Rad]': 'Phase RMS Averaging Set STD [Rad]'}
column_phase = 'Phase [Rad]'

pre_910_states_subtracted_df = pre_910_state_switching_df_group.apply(pre_910_state_subtract, columns_phase_std_dict, column_phase)
#%%
# We now have obtained for every RF frequency and repeat (and possibly other experiment parameters) phase difference between two states of pre-quench 910 cavity. We want now want to average for every RF frequency all of this data for all of the repeats.

pre_910_states_subtracted_index_list = list(pre_910_states_subtracted_df.index.names)
pre_910_states_subtracted_index_list.remove('Repeat')
pre_910_states_subtracted_df_group = pre_910_states_subtracted_df.groupby(pre_910_states_subtracted_index_list)

reference_type_list = ['RF Combiner I Reference', 'RF Combiner R Reference']
averaging_type_list = ['Phase Averaging', 'Phasor Averaging', 'Phasor Averaging Relative To DC']

data_column = 'Phase [Rad]'

columns_dict = {
    'Phase RMS Repeat STD': 'Phase RMS Repeat STD [Rad]',
    'Phase RMS Averaging Set STD': 'Phase RMS Averaging Set STD [Rad]',
    'Phase STD': 'Phase STD [Rad]'}

# Average the phases over all repeats for every Waveguide Carrier Frequency.

pre_910_states_averaged_df = pre_910_states_subtracted_df_group.apply(average_data_field, reference_type_list, harmonic_name_list, averaging_type_list, data_column, columns_dict, True)

# This difference now needs to get divided by two, because while calculating RF CH A - RF CH B we left it without the division by 2.

pre_910_states_averaged_df.loc[slice(None),(slice(None), slice(None), slice(None), slice(None), ['Weighted Mean'])] = pre_910_states_averaged_df.loc[slice(None),(slice(None), slice(None), slice(None), slice(None), ['Weighted Mean'])].transform(divide_and_minimize_phase, div_number=2)

# Of course we have to divide the error by 2 as well.
pre_910_states_averaged_df.loc[slice(None),(slice(None), slice(None), slice(None), slice(None), ['Weighted STD'])] = pre_910_states_averaged_df.loc[slice(None),(slice(None), slice(None), slice(None), slice(None), ['Weighted STD'])]/2


# We now finally average the deviations for all RF frequencies for each analysis method.
pre910_grouping_columns_list = list(pre_910_states_averaged_df.columns.names)
pre910_grouping_columns_list.remove('Data Field')

pre_910_av_difference_df = pre_910_states_averaged_df.groupby(level=pre910_grouping_columns_list, axis='columns').apply(lambda x: straight_line_fit_params(data_arr=x.xs(axis='columns',key='Weighted Mean', level='Data Field').values[:,0], sigma_arr=x.xs(axis='columns',key='Weighted STD', level='Data Field').values[:,0]))
#%%
pre_910_av_difference_df
#%%
y_data_arr = pre_910_states_averaged_df[('RF Combiner I Reference', 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD', 'Weighted Mean')]
y_data_err_arr = pre_910_states_averaged_df[('RF Combiner I Reference', 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD', 'Weighted STD')]
np.polyfit(x=pre_910_states_averaged_df.index.values, y=y_data_arr, deg=0, w=1/y_data_err_arr**2)
#%%
np.average(y_data_arr, weights=1/y_data_err_arr**2)
#%%
np.sqrt(1/np.sum(1/y_data_err_arr**2))
#%%

#%%
pre_910_av_difference_df
#%%
plt.figure()
pre_910_states_averaged_df.reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y=('RF Combiner I Reference', 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD', 'Weighted Mean'), yerr=pre_910_states_averaged_df.reset_index()['RF Combiner I Reference', 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD', 'Weighted STD'])
plt.show()

av = np.average(pre_910_states_averaged_df['RF Combiner I Reference', 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD', 'Weighted Mean'].values, weights=1/pre_910_states_averaged_df['RF Combiner I Reference', 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD', 'Weighted STD']**2)
#%%
av
#%%
np.sum(((pre_910_states_averaged_df['RF Combiner I Reference', 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD', 'Weighted Mean'].values-av)/pre_910_states_averaged_df['RF Combiner I Reference', 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD', 'Weighted STD'])**2)/(pre_910_states_averaged_df['RF Combiner I Reference', 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD', 'Weighted Mean'].values.size-1)
#%%
plt.figure()
pre_910_states_averaged_df.reset_index().plot(kind='scatter', x='Waveguide Carrier Frequency [MHz]', y=('RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Averaging Set STD', 'Weighted Mean'), yerr=pre_910_states_averaged_df.reset_index()['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Averaging Set STD', 'Weighted STD'])
plt.show()
#%%
# Plotting
pre_910_states_averaged_chosen_df = pre_910_states_averaged_df['RF Combiner I Reference']['First Harmonic']['Phasor Averaging']['Phase RMS Repeat STD']

y_data_arr = fosof_phases_chosen_df['Weighted Mean'].values
y_data_err_arr = fosof_phases_chosen_df['Weighted STD'].values

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(24,12)
axes[0,0].errorbar(x=fosof_phases_chosen_df.index.values, y=y_data_arr, color='C0', marker='.', yerr=y_data_err_arr, linestyle='None')
#
fit_coeff, fit_cov = np.polyfit(x=fosof_phases_chosen_df.index.values, y=y_data_arr, deg=1, w=1/y_data_err_arr**2, cov=True)

fit_p = np.poly1d(fit_coeff)

axes[0,0].plot(fosof_phases_chosen_df.index.values, fit_p(fosof_phases_chosen_df.index.values), color='C0')

#axes[0].set_xlabel('Waveguide Carrier Frequency [MHz]')
axes[0,0].set_ylabel('RF CH A - RF CH B Phase [Rad]')
axes[0,0].set_title('Atomic + RF Phase Vs Waveguide Frequency')
axes[0,0].grid()
#axes[0].xaxis.set_ticklabels([])
#axes[0].set(xticks=[])

axes[1,0].errorbar(x=fosof_phases_chosen_df.index.values, y=(y_data_arr-fit_p(fosof_phases_chosen_df.index.values))*1E3, color='C2', marker='.', yerr=y_data_err_arr*1E3, linestyle='None')

axes[1,0].set_xlabel('Waveguide Carrier Frequency [MHz]')
axes[1,0].set_ylabel('Residual [mrad]')
axes[1,0].grid()

axes[0,1].errorbar(x=fosof_ampl_chosen_df.index.values, y=y_ampl_data_arr, color='C0', marker='.', yerr=y_ampl_data_err_arr, linestyle='None')

axes[0,1].set_xlabel('Waveguide Carrier Frequency [MHz]')
axes[0,1].set_ylabel('FOSOF Fractional Amplitude')
axes[0,1].grid()
#axes[1].set_title('Atomic + RF Phase Vs Waveguide Frequency')

plt.show()
#%%
# Plotting
fosof_phases_chosen_df = fosof_phases_df.loc['on']['RF Combiner I Reference']['First Harmonic']['Phasor Averaging']['Phase STD']

fosof_ampl_chosen_df = fosof_ampl_df.loc['on']['RF Combiner I Reference']['First Harmonic']['Phasor Averaging Relative To DC']['Amplitude Relative To DC RMS Repeat STD']

y_data_arr = phases_shift(fosof_phases_chosen_df['Weighted Mean'].values)[0]
y_data_err_arr = fosof_phases_chosen_df['Weighted STD'].values

y_ampl_data_arr = fosof_ampl_chosen_df['Weighted Mean'].values
y_ampl_data_err_arr = fosof_ampl_chosen_df['Weighted STD'].values

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(24,12)
axes[0,0].errorbar(x=fosof_phases_chosen_df.index.values, y=y_data_arr, color='C0', marker='.', yerr=y_data_err_arr, linestyle='None')
#
fit_coeff, fit_cov = np.polyfit(x=fosof_phases_chosen_df.index.values, y=y_data_arr, deg=1, w=1/y_data_err_arr**2, cov=True)

fit_p = np.poly1d(fit_coeff)

axes[0,0].plot(fosof_phases_chosen_df.index.values, fit_p(fosof_phases_chosen_df.index.values), color='C0')

#axes[0].set_xlabel('Waveguide Carrier Frequency [MHz]')
axes[0,0].set_ylabel('RF CH A - RF CH B Phase [Rad]')
axes[0,0].set_title('Atomic + RF Phase Vs Waveguide Frequency')
axes[0,0].grid()
#axes[0].xaxis.set_ticklabels([])
#axes[0].set(xticks=[])

axes[1,0].errorbar(x=fosof_phases_chosen_df.index.values, y=(y_data_arr-fit_p(fosof_phases_chosen_df.index.values))*1E3, color='C2', marker='.', yerr=y_data_err_arr*1E3, linestyle='None')

axes[1,0].set_xlabel('Waveguide Carrier Frequency [MHz]')
axes[1,0].set_ylabel('Residual [mrad]')
axes[1,0].grid()

axes[0,1].errorbar(x=fosof_ampl_chosen_df.index.values, y=y_ampl_data_arr, color='C0', marker='.', yerr=y_ampl_data_err_arr, linestyle='None')

axes[0,1].set_xlabel('Waveguide Carrier Frequency [MHz]')
axes[0,1].set_ylabel('FOSOF Fractional Amplitude')
axes[0,1].grid()
#axes[1].set_title('Atomic + RF Phase Vs Waveguide Frequency')

plt.show()
#%%
axes
#%%
axes[1,0]
#%%
fit_p
#%%
fit_p
#%%
fit_p
#%%
fit_p
#%%
fit_p

#%%
fit_params = np.polyfit(x=fosof_data_df.index.values, y=phases_shift(fosof_data_df['RF CH A - RF CH B Weighted Averaged Phase [Rad]'].values)[0], deg=1, w=1/fosof_data_df['RF CH A - RF CH B Weighted Averaged Phase STD [Rad]'].values**2, cov=True)
#%%
fit_p(fosof_phases_df.index.values)
#%%
y_data_arr-fit_p(fosof_phases_df.index.values)
#%%
(y_data_arr-fit_p(fosof_phases_df.index.values))*1E3
#%%
phases_shift(fosof_data_df['RF CH A - RF CH B Weighted Averaged Phase [Rad]'].values)[0]-(fosof_phases_df.index.values*fit_params[0][0]+fit_params[0][1])
#%%
phases_shift(fosof_data_df['RF CH A - RF CH B Weighted Averaged Phase [Rad]'].values)[0]
#%%

#%%
fit_p(fosof_data_df.index.values)
#%%
phases_shift(fosof_data_df['RF CH A - RF CH B Weighted Averaged Phase [Rad]'].values)[0]-(fosof_phases_df.index.values*fit_params[0][0]+fit_params[0][1])
#%%
# Various column names
combiner_R_first_harmonic_phase_col = 'RF Power Combiner R First Harmonic Fourier Phase [Rad]'
combiner_R_second_harmonic_phase_col = 'RF Power Combiner R Second Harmonic Fourier Phase [Rad]'

fosof_phase_I_harm_1_col = 'Detector - Combiner I First Harmonic Phase Difference [Rad]'
fosof_phase_R_harm_1_col = 'Detector - Combiner R First Harmonic Phase Difference [Rad]'

fosof_phase_I_harm_2_col = 'Detector - Combiner I Second Harmonic Phase Difference [Rad]'
fosof_phase_R_harm_2_col = 'Detector - Combiner R Second Harmonic Phase Difference [Rad]'

fosof_ampl_harm_1_col = 'Detector First Harmonic Fourier Amplitude [V]'
fosof_ampl_harm_2_col = 'Detector Second Harmonic Fourier Amplitude [V]'

# Add columns with various phase differences

exp_data_frame[fosof_phase_I_harm_1_col] = convert_phase_to_2pi_range(exp_data_frame[detector_first_harmonic_phase_col]-exp_data_frame[combiner_I_digi_1_first_harmonic_phase_col])

exp_data_frame['Combiner I Digi 2 - Combiner R First Harmonic Phase Difference [Rad]'] = convert_phase_to_2pi_range(exp_data_frame[combiner_I_digi_1_first_harmonic_phase_col]-exp_data_frame[combiner_R_first_harmonic_phase_col])

# We assume that the Combiner R on Digi 2 is different from imaginary Combiner R on Digi 1 by the same phase difference as the Combiner I on Digi 2 from the Combiner I on Digi 1.
exp_data_frame[fosof_phase_R_harm_1_col] = convert_phase_to_2pi_range(exp_data_frame['Detector - Combiner I First Harmonic Phase Difference [Rad]']+exp_data_frame['Combiner I Digi 2 - Combiner R First Harmonic Phase Difference [Rad]'])

exp_data_frame['Combiner I Digi 1 - Combiner I Digi 2 First Harmonic Phase Difference [Rad]'] = exp_data_frame[combiner_I_digi_1_first_harmonic_phase_col]-exp_data_frame[combiner_I_digi_2_first_harmonic_phase_col]

exp_data_frame['Lag Between Digi 1 and Digi 2 [sample]'] = exp_data_frame['Combiner I Digi 1 - Combiner I Digi 2 First Harmonic Phase Difference [Rad]'] / (2*np.pi * exp_data_frame['Offset Frequency [Hz]']) * sampling_rate
#%%

# Construct multi index dataframe that simplifies data analysis.
exp_data_mdf = exp_data_frame.set_index(['Repeat', 'Waveguide Carrier Frequency [MHz]', 'Configuration','Average'])

# Group the data for analysis
exp_data_group = exp_data_mdf.groupby(level=['Repeat','Waveguide Carrier Frequency [MHz]','Configuration'])

#%%
def average_phasor(df):
    ''' Averages phasors for each averaging set of the data set, given a dataframe = member of a group
    '''

    # Average phasors for the 1st FOSOF harmonic with the Combiner I
    fosof_I_harm_1_s =pd.Series(mean_phasor(amp_arr=df[fosof_ampl_harm_1_col], phase_arr=df[fosof_phase_I_harm_1_col]))
    fosof_I_harm_1_s.rename(index={'Amplitude': 'FOSOF First Harmonic Amplitude [V]', 'Amplitude STD': 'FOSOF First Harmonic Amplitude STD [V]', 'Phase [Rad]':'FOSOF I First Harmonic Phase [Rad]', 'Phase STD [Rad]': 'FOSOF I First Harmonic Phase STD [Rad]', 'Phase STD With Covariance [Rad]': 'FOSOF I First Harmonic Phase STD With Covariance [Rad]'}, inplace=True)

    # Adding SNR of the average phasor. This is just an estimate. We assume that the true SNR is the same for each phasor in the averaging set. Thus, we can calculate the mean of the SNR. However, averaging N phasors results in the SNR going up by a factor of N**0.5. The reason for this is that the resulting phasor has, real and imaginary components as the mean of the set {A_n_i * cos(phi_n_i), A_n_i * sin(phi_n_i)}, where phi_n_i is random phase, and A_n_i is the noise amplitude. In this case the amplitude of the resulting averaged phasor is smaller by a factor of N**0.5, assuming that A_n_i is a constant. I have tested this in Mathematica and it seems indeed to be the case, except that it seems to go down as about 1.1*N**0.5.
    fosof_I_harm_1_s['FOSOF First Harmonic SNR'] = df['Detector First Harmonic SNR'].mean() * np.sqrt(df['Detector First Harmonic SNR'].size)
    #fosof_I_harm_1_s['FOSOF First Harmonic SNR STD'] = np.std(x['Detector First Harmonic SNR'], ddof=1)

    # # We are making sure that the averaged phasor does not have SNR of less than 2, since this corresponds to the range of the phase noise of about 60 deg.
    # if fosof_I_harm_1_s['FOSOF First Harmonic SNR'] < 2:
    #     print('SNR of the averaged phasor of less than 2 has been found! The amplitude and phase information of the phasor is set to np.nan.')

    # Average phasors for the 1st FOSOF harmonic with the Combiner R
    fosof_R_harm_1_s =pd.Series(mean_phasor(amp_arr=df[fosof_ampl_harm_1_col], phase_arr=df[fosof_phase_R_harm_1_col]))
    fosof_R_harm_1_s.rename(index={'Amplitude': 'FOSOF First Harmonic Amplitude [V]', 'Amplitude STD': 'FOSOF First Harmonic Amplitude STD [V]', 'Phase [Rad]':'FOSOF R First Harmonic Phase [Rad]','Phase STD [Rad]': 'FOSOF R First Harmonic Phase STD [Rad]', 'Phase STD With Covariance [Rad]':'FOSOF R First Harmonic Phase STD With Covariance [Rad]'}, inplace=True)

    # fosof_I_harm_2_s =pd.Series(mean_phasor(amp_arr=x[fosof_ampl_harm_2_col], phase_arr=x[fosof_phase_I_harm_2_col]))
    # fosof_I_harm_2_s.rename(index={'Amplitude': 'Amplitude STD': 'FOSOF Second Harmonic Amplitude STD [V]', 'Phase [Rad]':'FOSOF I Second Harmonic Phase [Rad]','Phase STD [Rad]': 'FOSOF I Second Harmonic Phase STD [Rad]'}, inplace=True)

    # fosof_R_harm_2_s =pd.Series(mean_phasor(amp_arr=x[fosof_ampl_harm_2_col], phase_arr=x[fosof_phase_R_harm_2_col]))
    # fosof_R_harm_2_s.rename(index={'Amplitude': 'FOSOF Second Harmonic Amplitude [V]','Amplitude STD': 'FOSOF Second Harmonic Amplitude STD [V]','Phase [Rad]':'FOSOF R Second Harmonic Phase [Rad]','Phase STD [Rad]': 'FOSOF R Second Harmonic Phase STD [Rad]'}, inplace=True)

    # return fosof_I_harm_1_s.append([fosof_R_harm_1_s,fosof_I_harm_2_s,fosof_R_harm_2_s])
    # Joining the series together
    combined_s = fosof_I_harm_1_s.append([fosof_R_harm_1_s])

    # Number Of Averaged Phasors is the common index to the series that has the same value for all of them. We do not want to have the same index with the same value repeating multiple times in the combined series. That is why we remove all the duplicated indeces from the combined series
    return combined_s[~combined_s.index.duplicated(keep='first')]
#%%
phase_av_set_df = exp_data_group.apply(average_phasor)

#%%
# We assume that for the given repeat all of the phasors obtained should have the same average SNR and thus the same true standard deviation for phase and amplitude at the given offset frequency. That is why we calculate the RMS quantities below. Now, this assumption might be wrong, especially when we have the beam that deteriorates over time or when the beam is such that it abruptly changes its mode of operation (we see it sometimes)
phase_av_set_group = phase_av_set_df.groupby(['Repeat'])

phasor_rms_std_df = phase_av_set_group[['FOSOF I First Harmonic Phase STD [Rad]','FOSOF R First Harmonic Phase STD [Rad]', 'FOSOF First Harmonic Amplitude STD [V]', 'FOSOF I First Harmonic Phase STD With Covariance [Rad]', 'FOSOF R First Harmonic Phase STD With Covariance [Rad]','FOSOF First Harmonic SNR']].aggregate(lambda x: np.sqrt(sum(x**2)/x.size))

phasor_rms_std_df.rename(columns={'FOSOF I First Harmonic Phase STD [Rad]':'FOSOF I First Harmonic Phase RMS STD [Rad]','FOSOF R First Harmonic Phase STD [Rad]':'FOSOF R First Harmonic Phase RMS STD [Rad]', 'FOSOF First Harmonic Amplitude STD [V]': 'FOSOF First Harmonic Amplitude RMS STD [V]', 'FOSOF I First Harmonic Phase STD With Covariance [Rad]': 'FOSOF I First Harmonic Phase RMS STD With Covariance [Rad]', 'FOSOF R First Harmonic Phase STD With Covariance [Rad]': 'FOSOF R First Harmonic Phase RMS STD With Covariance [Rad]', 'FOSOF First Harmonic SNR': 'FOSOF First Harmonic RMS SNR'}, inplace=True)

phase_av_set_df = phase_av_set_df.join(phasor_rms_std_df, how='inner')



#phase_av_set_df = phase_av_set_df.reset_index().set_index(['Repeat','FOSOF I First Harmonic Phase RMS STD [Rad]','FOSOF R First Harmonic Phase RMS STD [Rad]','FOSOF First Harmonic Amplitude RMS STD [V]','Waveguide Carrier Frequency [MHz]'])

# Transform dataframe to convenient multi index
phase_av_set_df = phase_av_set_df.reset_index().set_index(['Repeat','Waveguide Carrier Frequency [MHz]'])

# Just because I am not sure if the assumption that the standard deviation of the probability distributions for all of the data for the given repeat is the same, I will also do all the calculations with the estimate for the standard deviation calculated for each averaging set.

#%%
phase_av_set_df
#%%
# Calculate phase STD of the mean.

columns_sigma = [
        'FOSOF I First Harmonic Phase RMS STD [Rad]',
        'FOSOF I First Harmonic Phase RMS STD With Covariance [Rad]',
        'FOSOF I First Harmonic Phase STD [Rad]',
        'FOSOF R First Harmonic Phase RMS STD [Rad]',
        'FOSOF R First Harmonic Phase RMS STD With Covariance [Rad]',
        'FOSOF R First Harmonic Phase STD [Rad]']

columns_mean_sigma = [
        'FOSOF I First Harmonic Phase RMS Mean STD [Rad]',
        'FOSOF I First Harmonic Phase RMS Mean STD With Covariance [Rad]',
        'FOSOF I First Harmonic Phase Mean STD [Rad]',
        'FOSOF R First Harmonic Phase RMS Mean STD [Rad]',
        'FOSOF R First Harmonic Phase RMS Mean STD With Covariance [Rad]',
        'FOSOF R First Harmonic Phase Mean STD [Rad]']

for i in range(len(columns_sigma)):
    phase_av_set_df[columns_mean_sigma[i]] = phase_av_set_df[columns_sigma[i]]/np.sqrt(phase_av_set_df['Number Of Averaged Phasors'])
#%%
phase_av_set_df
#%%
phase_av_set_group = phase_av_set_df.groupby(['Repeat','Waveguide Carrier Frequency [MHz]'])

#%%
def eliminate_freq_response(x):
    '''Perform phase subtraction of RF CH A from RF CH B. The result is 2*(phi_FOSOF + phi_RF) with the phase shift due to frequency response of the detection system at the offset frequency eliminated.
    '''

    rf_ch_A_df = x[x['Configuration'] == 'A']
    rf_ch_B_df = x[x['Configuration'] == 'B']
    s = pd.Series(
    {
    'FOSOF I First Harmonic RF CHA - RF CHB Phase [Rad]': convert_phase_to_2pi_range(rf_ch_A_df['FOSOF I First Harmonic Phase [Rad]'].values[0]-rf_ch_B_df['FOSOF I First Harmonic Phase [Rad]'].values[0]),

    'FOSOF R First Harmonic RF CHA - RF CHB Phase [Rad]': convert_phase_to_2pi_range(rf_ch_A_df['FOSOF R First Harmonic Phase [Rad]'].values[0]-rf_ch_B_df['FOSOF R First Harmonic Phase [Rad]'].values[0]),

    'FOSOF I First Harmonic RF CHA - RF CHB Phase RMS STD [Rad]':np.sqrt((rf_ch_A_df['FOSOF I First Harmonic Phase RMS Mean STD [Rad]'].values[0])**2+(rf_ch_B_df['FOSOF I First Harmonic Phase RMS Mean STD [Rad]'].values[0])**2),

    'FOSOF I First Harmonic RF CHA - RF CHB Phase RMS STD With Covariance [Rad]':np.sqrt((rf_ch_A_df['FOSOF I First Harmonic Phase RMS Mean STD With Covariance [Rad]'].values[0])**2+(rf_ch_B_df['FOSOF I First Harmonic Phase RMS Mean STD With Covariance [Rad]'].values[0])**2),

    'FOSOF R First Harmonic RF CHA - RF CHB Phase RMS STD [Rad]':np.sqrt((rf_ch_A_df['FOSOF R First Harmonic Phase RMS Mean STD [Rad]'].values[0])**2+(rf_ch_B_df['FOSOF R First Harmonic Phase RMS Mean STD [Rad]'].values[0])**2),

    'FOSOF R First Harmonic RF CHA - RF CHB Phase RMS STD With Covariance [Rad]':np.sqrt((rf_ch_A_df['FOSOF R First Harmonic Phase RMS Mean STD With Covariance [Rad]'].values[0])**2+(rf_ch_B_df['FOSOF R First Harmonic Phase RMS Mean STD With Covariance [Rad]'].values[0])**2)
    }
    )

    return s

#%%
# Perform RF CH A and RF CH B phase subtraction to get rid of the detection system frequency response at the offset frequency.
# We also reverse the order of some of the indeces for convenience.
phase_A_minus_B_df = phase_av_set_group.apply(eliminate_freq_response).reset_index().set_index(['Waveguide Carrier Frequency [MHz]', 'Repeat']).sort_index(level=0)

phase_A_minus_B_df
#%%
phase_A_minus_B_df_group = phase_A_minus_B_df.groupby(['Waveguide Carrier Frequency [MHz]'])
#%%
def average_phases(df):
    ''' Perform phase averaging across all of the repeats of the given data frame.
    '''
    data_arr = phases_shift(df['FOSOF I First Harmonic RF CHA - RF CHB Phase [Rad]'])[0]
    sigma_arr = df['FOSOF I First Harmonic RF CHA - RF CHB Phase RMS STD With Covariance [Rad]']
    phase_I_average_s = straight_line_fit_params(data_arr, sigma_arr)

    phase_I_average_s.rename({'Weighted Mean':'FOSOF I First Harmonic RF CHA - RF CHB Average Phase [Rad]','Weighted STD':'FOSOF I First Harmonic RF CHA - RF CHB Average Phase STD [Rad]', 'Reduced Chi Squared': 'FOSOF I First Harmonic RF CHA - RF CHB Phase Reduced Chi Squared', 'P(>Chi Squared)': 'FOSOF I First Harmonic RF CHA - RF CHB Phase P(>Chi Squared)'}, inplace=True)

    data_arr = phases_shift(df['FOSOF R First Harmonic RF CHA - RF CHB Phase [Rad]'])[0]
    sigma_arr = df['FOSOF R First Harmonic RF CHA - RF CHB Phase RMS STD With Covariance [Rad]']
    phase_R_average_s = straight_line_fit_params(data_arr, sigma_arr)

    phase_R_average_s.rename({'Weighted Mean':'FOSOF R First Harmonic RF CHA - RF CHB Average Phase [Rad]','Weighted STD':'FOSOF R First Harmonic RF CHA - RF CHB Average Phase STD [Rad]', 'Reduced Chi Squared': 'FOSOF R First Harmonic RF CHA - RF CHB Phase Reduced Chi Squared', 'P(>Chi Squared)': 'FOSOF R First Harmonic RF CHA - RF CHB Phase P(>Chi Squared)'}, inplace=True)

    combined_s = phase_I_average_s.append([phase_R_average_s])

    return combined_s
#%%
# Average the phases over all repeats for every Waveguide Carrier Frequency.
fosof_phases_df = phase_A_minus_B_df_group.apply(average_phases)
fosof_phases_df

#%%
fig = plt.figure(figsize=(12.0,10.0))
ax = fig.add_subplot(111)
ax.errorbar(x=fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase [Rad]'].index.values, y=phases_shift(fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase [Rad]'].values)[0], color='C0', marker='.', yerr=fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase STD [Rad]'].values, linestyle='None')
#ax.set_ylim(0, np.pi)

fit_params = np.polyfit(x=fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase [Rad]'].index.values, y=phases_shift(fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase [Rad]'].values)[0], deg=1, w=1/fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase STD [Rad]'].values**2, cov=True)

ax.plot(fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase [Rad]'].index.values,fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase [Rad]'].index.values*fit_params[0][0]+fit_params[0][1], color='C0')

ax.set_xlabel('Waveguide Carrier Frequency [MHz]')
ax.set_ylabel('RF CH A - RF CH B Phase [Rad]')
ax.set_title('Atomic + RF Phase Vs Waveguide Frequency')
plt.show()


#%%
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set(title='Upper Left')
axes[0,1].set(title='Upper Right')
axes[1,0].set(title='Bottom Left')
axes[1,1].set(title='Bottom Right')

for ax in axes.flat:
    ax.set(xticks=[], yticks=[])

plt.show()
#%%
fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase [Rad]'].values
#%%
fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase [Rad]'].index.values
#%%
fig, ax = plt.subplots()
ax.plot([1,2,3,4], [10,20,25,30], label='Phil')
ax.plot([1,2,3,4], [30,23,15,5], label='dfdf')

ax.legend()
plt.show()
#%%
fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase STD [Rad]'].values
#%%
fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase [Rad]'].values
#%%

#%%
fit_params
fosof_phases_df['FOSOF I First Harmonic RF CHA - RF CHB Average Phase [Rad]'].index.values*fit_params[0][0]+fit_params[0][1]
#%%
fit_params[0][0]
#%%
exp_data_frame_grouped1 = exp_data_frame.groupby('Configuration')
exp_data_A = exp_data_frame_grouped1.get_group('A')
exp_data_snr=exp_data_A[exp_data_A['Repeat']==2]
#%%
fig = plt.figure(figsize=(12.0,10.0))
ax = fig.add_subplot(111)
ax.hist(x=exp_data_snr['Detector First Harmonic SNR'].values)
#ax.set_ylim(0, np.pi)

plt.show()
#%%

#%%
exp_data_snr['Detector Second Harmonic SNR'].values
#%%

fig = plt.figure(figsize=(12.0,10.0))
ax = fig.add_subplot(111)
ax.hist(x=phase_av_set_df.reset_index()[phase_av_set_df.reset_index()['Repeat']==2]['FOSOF I First Harmonic Phase STD [Rad]'].values)
#ax.set_ylim(0, np.pi)

plt.show()
#%%
data = exp_data_snr['Detector First Harmonic SNR'].values
#data = phase_av_set_df.reset_index()[phase_av_set_df.reset_index()['Repeat']==2]['FOSOF I First Harmonic Phase STD [Rad]'].values

freq, edges = np.histogram(a=data, bins=7)
data_freq_sigma = np.sqrt(freq)
edges_arr = np.linspace(np.min(edges), np.max(edges), edges.shape[0])
prob_arr = scipy.stats.norm.cdf(edges_arr[1::1], data_mean, data_std)-scipy.stats.norm.cdf(edges_arr[:-1:1], data_mean, data_std)
num_events_arr = np.sum(freq)*prob_arr.shape[0]/freq.shape[0] * prob_arr
bins_centers = np.diff(edges_arr)/2+edges[:-1]

fig, ax = plt.subplots()
ax.bar(edges[:-1], height=freq, width=np.diff(edges), align='edge', yerr=data_freq_sigma, ecolor='black', edgecolor='black')
ax.plot(bins_centers, num_events_arr, color='black')
plt.show()
#%%
data_mean = np.mean(data)
data_std = np.std(data, ddof=1)
#%%
np.sum((freq - num_events_arr)**2/freq)/(freq.shape[0]-2)
#%%
scipy.stats.norm(loc=data_mean, scale=data_std)
#%%
scipy.stats.normaltest(data)
#%%
scipy.stats.skewtest(data)
#%%
scipy.stats.skew(data)
#%%
prob_arr
#%%
#%%
freq
#%%

#%%


#%%
edges
#%%

#%%

#%%

#%%
scipy.stats.chisquare(f_obs=freq, f_exp=num_events_arr, ddof=0)
#%%
edges
#%%
edges[:-1:1]
#%%
edges[1::1]
#%%
phase_av_set_df.reset_index()[phase_av_set_df.reset_index()['Repeat']==2]['FOSOF I First Harmonic Phase STD [Rad]'].values


#%%
fig = plt.figure(figsize=(12.0,10.0))
ax = fig.add_subplot(111)
ax.scatter(x=exp_data_frame['Elapsed Time [s]'].values,y=exp_data_frame['Detector First Harmonic SNR'].values)
plt.show()
#%%
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=plt.figaspect(0.5))

ax1.plot([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])
ax2.scatter([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])

plt.show()
#%%
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=plt.figaspect(0.5))

ax1.plot([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])
ax2.scatter([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])

ax1.margins(x=0.0, y=0.1)
ax2.margins(0.5)
plt.show()
#%%
ax1.axis()
#%%
fig, axes = plt.subplots(nrows=3)

for ax in axes:
    ax.plot([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])

axes[0].set_title('Normal autoscaling', y=0.7, x=0.8)

axes[1].set_title('ax.axis("tight")', y=0.7, x=0.8)
axes[1].axis('tight')

axes[2].set_title('ax.axis("equal")', y=0.7, x=0.8)
axes[2].axis('equal')
plt.show()
#%%
# Good -- setting limits after plotting is done
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
ax1.plot([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])
ax2.scatter([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])
ax1.set_ylim(bottom=-10)
ax2.set_xlim(right=25)
plt.show()
#%%
# Bad -- Setting limits before plotting is done
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
ax1.set_ylim(bottom=-10)
ax2.set_xlim(right=25)
ax1.plot([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])
ax2.scatter([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])
plt.show()
#%%
fig, ax = plt.subplots()
ax.plot([1,2,3,4], [10,20,25,30], label='abc')
ax.plot([1,2,3,4], [30,10,15,6], label='def')
ax.set(ylabel='Temp',xlabel='Time', title='TATATAT')
ax.legend(loc='best')
plt.show()
#%%
np.sum(freq)
#%%
82/4
#%%
1/(np.sqrt(2*8*41-1))
#%%
index_names = ['Amplitude', 'Amplitude STD', 'Phase [Rad]', 'Phase STD [Rad]', 'Phase STD With Covariance [Rad]', 'Number Of Averaged Phasors']
#%%
map(lambda x: 'FOSOF '+x,index_names)
#%%
df = pd.DataFrame({'A': np.array([1,2,3,4]), 'B':np.array([5,6,7,8])})
df
#%%
df = pd.DataFrame({
    'class' : ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
    'number' : [1,2,3,4,5,1,2,3,4,5],
    'math' : [90, 20, 50, 30, 57, 67, 89, 79, 45, 23],
    'english' : [40, 21, 68, 89, 90, 87, 89, 54, 21, 23]
})
df
#%%
df1 = df.set_index(['number','class'])
df1
#%%
df1.unstack()
#%%
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
df.rename(index=str, columns={"A": ('C',"a"), "B": ('C',"c")}).columns
#%%
exp_data_mdf.rename(columns={fosof_phase_I_harm_1_col: ('Combiner I', 'SHIT'), fosof_phase_I_harm_2_col: ('Combiner I', 'SHIT2')}).columns
#%%
fosof_phase_I_harm_1_col
#%%
exp_data_mdf.columns
#%%
detector_harm_1_columns = ['Detector First Harmonic Average Total Noise Fourier Amplitude [V]', 'Detector First Harmonic Fourier Amplitude [V]', 'Detector First Harmonic Fourier Frequency [Hz]', 'Detector First Harmonic Fourier Phase [Rad]', 'Detector First Harmonic SNR']
detector_harm_2_columns = ['Detector Second Harmonic Average Total Noise Fourier Amplitude [V]', 'Detector Second Harmonic Fourier Amplitude [V]',  'Detector Second Harmonic Fourier Frequency [Hz]', 'Detector Second Harmonic Fourier Phase [Rad]', 'Detector Second Harmonic SNR']

combiner_I_harm_1_digi_1_columns = ['RF Power Combiner I Digi 1 First Harmonic Average Total Noise Fourier Amplitude [V]', 'RF Power Combiner I Digi 1 First Harmonic Fourier Amplitude [V]', 'RF Power Combiner I Digi 1 First Harmonic Fourier Frequency [Hz]', 'RF Power Combiner I Digi 1 First Harmonic Fourier Phase [Rad]', 'RF Power Combiner I Digi 1 First Harmonic SNR']
combiner_I_harm_2_digi_1_columns = ['RF Power Combiner I Digi 1 Second Harmonic Average Total Noise Fourier Amplitude [V]', 'RF Power Combiner I Digi 1 Second Harmonic Fourier Amplitude [V]', 'RF Power Combiner I Digi 1 Second Harmonic Fourier Frequency [Hz]', 'RF Power Combiner I Digi 1 Second Harmonic Fourier Phase [Rad]', 'RF Power Combiner I Digi 1 Second Harmonic SNR']

combiner_I_harm_1_digi_2_columns = ['RF Power Combiner I Digi 2 First Harmonic Average Total Noise Fourier Amplitude [V]', 'RF Power Combiner I Digi 2 First Harmonic Fourier Amplitude [V]', 'RF Power Combiner I Digi 2 First Harmonic Fourier Frequency [Hz]', 'RF Power Combiner I Digi 2 First Harmonic Fourier Phase [Rad]', 'RF Power Combiner I Digi 2 First Harmonic SNR']
combiner_I_harm_2_digi_2_columns = ['RF Power Combiner I Digi 2 Second Harmonic Average Total Noise Fourier Amplitude [V]', 'RF Power Combiner I Digi 2 Second Harmonic Fourier Amplitude [V]', 'RF Power Combiner I Digi 2 Second Harmonic Fourier Frequency [Hz]', 'RF Power Combiner I Digi 2 Second Harmonic Fourier Phase [Rad]', 'RF Power Combiner I Digi 2 Second Harmonic SNR']

combiner_R_harm_1_columns = ['RF Power Combiner R First Harmonic Average Total Noise Fourier Amplitude [V]', 'RF Power Combiner R First Harmonic Fourier Amplitude [V]', 'RF Power Combiner R First Harmonic Fourier Frequency [Hz]', 'RF Power Combiner R First Harmonic Fourier Phase [Rad]', 'RF Power Combiner R First Harmonic SNR']
combiner_R_harm_2_columns = ['RF Power Combiner R Second Harmonic Average Total Noise Fourier Amplitude [V]', 'RF Power Combiner R Second Harmonic Fourier Amplitude [V]', 'RF Power Combiner R Second Harmonic Fourier Frequency [Hz]', 'RF Power Combiner R Second Harmonic Fourier Phase [Rad]', 'RF Power Combiner R Second Harmonic SNR']

quenching_cavity_column_arr = ['Pre-Quench 910 Attenuator Voltage Reading [V]',
    'Pre-Quench 1088 Attenuator Voltage Reading [V]',
    'Pre-Quench 1147 Attenuator Voltage Reading [V]',
    'Pre-Quench 910 Attenuator Voltage Reading [V]',
    'Post-Quench 910 Attenuator Voltage Reading [V]',
    'Post-Quench 1088 Attenuator Voltage Reading [V]',
    'Post-Quench 1147 Attenuator Voltage Reading [V]',
    'Pre-Quench 910 Power Detector Reading [V]',
    'Pre-Quench 1088 Power Detector Reading [V]',
    'Pre-Quench 1147 Power Detector Reading [V]',
    'Post-Quench 910 Power Detector Reading [V]',
    'Post-Quench 1088 Power Detector Reading [V]',
    'Post-Quench 1147 Power Detector Reading [V]']
#%%
['Detector', 'RF Power Combiner I', 'RF Power Combiner R']
['First Harmonic', 'Second Harmonic']
['Average Total Noise Fourier Amplitude [V]', '']
#%%
quenching_cavities_column_arr = pd.MultiIndex.from_product([['Quenching cavities'], quenching_cavity_column_arr]).values
quenching_cavities_rename_dict = dict(map(lambda x: (x[1], x), quenching_cavities_column_arr))

detector_column_arr = pd.MultiIndex.from_product([['Detector'], np.concatenate((detector_harm_1_columns, detector_harm_2_columns))]).values
detector_rename_dict = dict(map(lambda x: (x[1], x), detector_column_arr))

combiner_I_column_arr = pd.MultiIndex.from_product([['RF Power Combiner I'], np.concatenate((combiner_I_harm_1_digi_1_columns, combiner_I_harm_2_digi_1_columns, combiner_I_harm_1_digi_2_columns, combiner_I_harm_2_digi_2_columns))]).values
combiner_I_rename_dict = dict(map(lambda x: (x[1], x), combiner_I_column_arr))

combiner_R_column_arr = pd.MultiIndex.from_product([['RF Power Combiner R'], np.concatenate((combiner_R_harm_1_columns, combiner_R_harm_2_columns))]).values
combiner_R_rename_dict = dict(map(lambda x: (x[1], x), combiner_R_column_arr))
#%%
columns_joined = np.concatenate((detector_harm_1_columns, detector_harm_2_columns, combiner_I_harm_1_digi_1_columns, combiner_I_harm_2_digi_1_columns, combiner_I_harm_2_digi_1_columns, combiner_I_harm_1_digi_2_columns, combiner_I_harm_2_digi_2_columns, combiner_R_harm_1_columns, combiner_R_harm_2_columns, quenching_cavity_column_arr, quenching_cavity_column_arr))

other_column_arr = exp_data_mdf.columns.difference(pd.Index(columns_joined)).values
#%%
other_column_arr
#%%
other_column_arr = pd.MultiIndex.from_product([['Other'], other_column_arr]).values
combiner_R_rename_dict = dict(map(lambda x: (x[1], x), combiner_R_column_arr))


other_rename_dict = dict(map(lambda x: (x[1], x), other_column_arr))
#%%
other_rename_dict.update(quenching_cavities_rename_dict)
other_rename_dict.update(detector_rename_dict)
other_rename_dict.update(combiner_I_rename_dict)
other_rename_dict.update(combiner_R_rename_dict)


#%%
len(other_rename_dict)
#%%
exp_data_mdf.columns = [map(lambda x: x[0], other_rename_dict.values()),map(lambda x: x[1], other_rename_dict.values())]
#%%
[map(lambda x: x[0], other_rename_dict.values()),map(lambda x: x[1], other_rename_dict.values())]
#%%
exp_data_mdf = exp_data_mdf.sort_index(axis=1)
#%%
exp_data_mdf['Detector'].columns
#%%
exp_data_mdf[]
#%%
exp_data_mdf['Detector'].columns = pd.MultiIndex.from_arrays([np.concatenate([np.repeat('First Harmonic', 5),np.repeat('Second Harmonic', 5)]), ['Detector First Harmonic Average Total Noise Fourier Amplitude [V]', 'Detector First Harmonic Fourier Amplitude [V]', 'Detector First Harmonic Fourier Frequency [Hz]', 'Detector First Harmonic Fourier Phase [Rad]', 'Detector First Harmonic SNR', 'Detector Second Harmonic Average Total Noise Fourier Amplitude [V]', 'Detector Second Harmonic Fourier Amplitude [V]',  'Detector Second Harmonic Fourier Frequency [Hz]', 'Detector Second Harmonic Fourier Phase [Rad]', 'Detector Second Harmonic SNR']])
#%%
exp_data_mdf['Detector'].columns
#%%
data
#%%
pd.MultiIndex.from_arrays([np.concatenate([np.repeat('First Harmonic', 5),np.repeat('Second Harmonic', 5)]), ['Detector First Harmonic Average Total Noise Fourier Amplitude [V]', 'Detector First Harmonic Fourier Amplitude [V]', 'Detector First Harmonic Fourier Frequency [Hz]', 'Detector First Harmonic Fourier Phase [Rad]', 'Detector First Harmonic SNR', 'Detector Second Harmonic Average Total Noise Fourier Amplitude [V]', 'Detector Second Harmonic Fourier Amplitude [V]',  'Detector Second Harmonic Fourier Frequency [Hz]', 'Detector Second Harmonic Fourier Phase [Rad]', 'Detector Second Harmonic SNR']])
#%%
idx =pd.MultiIndex.from_tuples([(1, u'one'), (1, u'two'),
                                  (2, u'one'), (2, u'two')],
                                  names=['foo', 'bar'])
idx
#%%
idx.set_levels([['a','b'], [1,2]])
#%%
exp_data_mdf.columns.values
#%%
idx.set_levels(['a','b'], level=2)
#%%
 arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue'], ['one','one','two','two']]
pd.MultiIndex.from_arrays(arrays, names=('number', 'color', 'string')).values
#%%
df1 = exp_data_mdf.T.reset_index()
#%%
df1['level_2'] = np.nan
#%%
df1[df1['level_0'] == 'Detector']['level_2'] = 'det'
#%%
df1
#%%
np.linspace(-100,50,76)
#%%
76*2*4/3600*16
#%%
(3+5)
#%%
exp_data_mdf.columns.values
#%%
df1 = exp_data_frame.T.reset_index()
#%%
df1['level_1'] = None
#%%
df1
#%%
# Regular expression
reg_exp = re.compile( r"""Detector""", # Digitizer channel (positive integer)
                re.VERBOSE)

#%%
exp_data_frame.columns.values
#%%
level_value_arr = ['Detector', 'RF Power Combiner I', 'RF Power Combiner R', 'Faraday Cup']

reg_exp_value_arr = ['Detector', 'RF Power Combiner I', 'RF Power Combiner R', 'fc']
for index in range(len(level_value_arr)):
    reg_exp = re.compile(reg_exp_value_arr[index])
    matched_column_arr = np.array(map(lambda x: reg_exp.match(x), exp_data_frame.columns.values))
    exp_data_frame.columns.values[info != None]

    matched_index = df1['index'][np.array(map(lambda x: reg_exp.match(x),exp_data_frame.columns.values)) != None].index
df1.loc[matched_index,'level_1'] = level_value_arr[index]
#%%
df1.set_index(['level_1', 'index']).sort_index().T
#%%
level_value_arr = ['Detector', 'RF Power Combiner I', 'RF Power Combiner R', 'Faraday Cup']

reg_exp_value_arr = ['Detector', 'RF Power Combiner I', 'RF Power Combiner R', 'fc']
for index in range(len(level_value_arr)):
    reg_exp = re.compile(reg_exp_value_arr[index])
    matched_column_arr = np.array(map(lambda x: reg_exp.match(x), exp_data_frame.columns.values))
    exp_data_frame.columns.values[info != None]

    matched_index = df1['index'][np.array(map(lambda x: reg_exp.match(x),exp_data_frame.columns.values)) != None].index
    df1.loc[matched_index,'level_1'] = level_value_arr[index]
#%%
df1.set_index(['level_1', 'index']).sort_index().T
#%%
#%%
s1 = pd.Series([1,2,3])
s1.size
#%%
#%%
def analyze_averaging_set(df):
    ''' Averages data for the phasors in the given averaging set.
    '''
    average_type_list = [
            'Phasor Averaging',
            'Phase Averaging',
            'Phasor Averaging With Fractional Amplitudes']


    # Columns names of the phase references. Mostly we expect only two elements here: Combiner I and Combiner R reference
    reference_type_list = df.columns.levels[0].values

    # Column names corresponding to phasors at various Fourier harmonics
    harmonics_names_list = df.columns.levels[1].drop('Other').values

    # For each reference and harmonic find the resulting average phasor
    phasor_av_container_list = []
    for reference_type in reference_type_list:

        reference_data_df = df[reference_type]

        dc_mean = np.mean(reference_data_df['Other', 'DC [V]'])
        dc_std = np.std(reference_data_df['Other', 'DC [V]'], ddof=1)
        phasor_av_list = []

        for harmonic_level in harmonics_names_list:

            harmonic_data_df = reference_data_df[harmonic_level]

            phasor_av_s = pd.Series(mean_phasor(amp_arr=harmonic_data_df['Fourier Amplitude [V]'], phase_arr=harmonic_data_df['Fourier Phase [Rad]']))

            # Adding SNR of the average phasor. This is just an estimate. We assume that the true SNR is the same for each phasor in the averaging set. Thus, we can calculate the mean of the SNR. However, averaging N phasors results in the SNR going up by a factor of N**0.5. The reason for this is that the resulting phasor has, real and imaginary components as the mean of the set {A_n_i * cos(phi_n_i), A_n_i * sin(phi_n_i)}, where phi_n_i is random phase, and A_n_i is the noise amplitude. In this case the amplitude of the resulting averaged phasor is smaller by a factor of N**0.5, assuming that A_n_i is a constant. I have tested this in Mathematica and it seems indeed to be the case, except that it seems to go down as about 1.1*N**0.5.
            phasor_av_s['SNR'] = harmonic_data_df['SNR'].mean() * np.sqrt(harmonic_data_df['SNR'].size)

            #phasor_av_s['Amplitude-to-DC Ratio'] = harmonic_data_df['Fourier Amplitude [V]'].mean() * np.sqrt(harmonic_data_df['SNR'].size)

            phasor_av_s = pd.concat([phasor_av_s], keys=[harmonic_level])
            phasor_av_list.append(phasor_av_s)

        phasor_av_s = pd.concat(phasor_av_list)
        phasor_av_s = pd.concat([phasor_av_s], keys=[reference_type])
        phasor_av_container_list.append(phasor_av_s)

    phasor_av_combined_s = pd.concat(phasor_av_container_list)

    return phasor_av_combined_s
#%%
def eliminate_freq_response(df):
    '''Perform phase subtraction of RF CH A from RF CH B. The result is 2*(phi_FOSOF + phi_RF) with the phase shift due to frequency response of the detection system at the offset frequency eliminated.
    '''

    rf_ch_A_df = df[df['Configuration'] == 'A'].iloc[0]
    rf_ch_B_df = df[df['Configuration'] == 'B'].iloc[0]

    phase_difference_reference_type_list = []

    reference_type_list = ['RF Combiner I Reference', 'RF Combiner R Reference']
    harmonics_names_list = ['First Harmonic', 'Second Harmonic']

    column_sigma_list = [
            'RF CH A - RF CH B Phase RMS STD [Rad]',
            #'RF CH A - RF CH B Phase RMS STD With Covariance [Rad]',
            'RF CH A - RF CH B Phase STD [Rad]']#,
            #'RF CH A - RF CH B Phase STD With Covariance [Rad]']

    column_mean_sigma_list = [
            'Phase Mean RMS STD [Rad]',
            #'Phase Mean RMS STD With Covariance [Rad]',
            'Phase Mean STD [Rad]']#,
            #'Phase Mean STD With Covariance [Rad]']

    for reference_type in reference_type_list:

        reference_data_A_df = rf_ch_A_df[reference_type]
        reference_data_B_df = rf_ch_B_df[reference_type]

        phase_difference_harmonic_type_list = []

        for harmonic_level in harmonics_names_list:

            harmonic_data_A_df = reference_data_A_df[harmonic_level]
            harmonic_data_B_df = reference_data_B_df[harmonic_level]

            phase_difference_s = pd.Series()

            phase_difference_s['RF CH A - RF CH B Phase [Rad]'] = convert_phase_to_2pi_range(harmonic_data_A_df['Phase [Rad]'] - harmonic_data_B_df['Phase [Rad]'])

            for sigma_type_index in range(len(column_mean_sigma_list)):
                phase_difference_s[column_sigma_list[sigma_type_index]] = np.sqrt(harmonic_data_A_df[column_mean_sigma_list[sigma_type_index]]**2 + harmonic_data_B_df[column_mean_sigma_list[sigma_type_index]]**2)
            phase_difference_s.index.name = 'Data Field'
            phase_difference_s = pd.concat([phase_difference_s], keys=[harmonic_level], names=['Fourier Harmonic'])

            phase_difference_harmonic_type_list.append(phase_difference_s)
        phase_difference_reference_type_list.append(pd.concat([pd.concat(phase_difference_harmonic_type_list)], keys=[reference_type]))
    phase_difference_combined_s = pd.concat(phase_difference_reference_type_list)
    return phase_difference_combined_s
#%%
s1 = pd.Series({'a':1,'b':2})
#%%
s1
#%%
s1.rename({'a':'ba'})
#%%
join(['a','b'], 'c')
#%%
import copy
#%%
x = ['a','b','c']
x1 = copy.copy(x)
#%%
id(x)
id(x1)
#%%
x1
#%%
x1.append('d')
#%%
x1
#%%
a = {'a':1, 'b':2}
#%%
for std_type_name, std_type_str in a.iteritems():
    print(std_type_name)
#%%
#%%
def average_phases(df):
    ''' Perform phase averaging across all of the repeats of the given data frame.

    The averaging of phases is performed by performing a weighted least-squares straight line (slope = 0) fit to the data. This is equivalent to weighted average. We also extract Reduced Chi-Squared that tells us if our estimates for standard deviation are reasonable.
    For every type of averaging set averaging the averaging is performed for each type of phase uncertainty separately.
    '''

    reference_type_list = ['RF Combiner I Reference', 'RF Combiner R Reference']
    harmonics_names_list = ['First Harmonic', 'Second Harmonic']

    averaging_type_list = ['Phase Averaging', 'Phasor Averaging', 'Phasor Averaging Relative To DC']
    column_phase_std_type_list = [
            'Phase RMS Repeat STD',
            'Phase RMS Averaging Set STD',
            #'Phase RMS STD With Covariance',
            'Phase STD']#,
            #'Phase STD With Covariance']

    column_sigma_list = [
            'RF CH A - RF CH B Phase RMS Repeat STD [Rad]',
            'RF CH A - RF CH B Phase RMS Averaging Set STD [Rad]',
            #'RF CH A - RF CH B Phase RMS STD With Covariance [Rad]',
            'RF CH A - RF CH B Phase STD [Rad]']#,
            #'RF CH A - RF CH B Phase STD With Covariance [Rad]']

    phase_average_reference_type_list = []
    for reference_type in reference_type_list:
        reference_data_df = df[reference_type]

        phase_average_harmonic_list = []
        for harmonic_name in harmonics_names_list:
            harmonic_data_df = reference_data_df[harmonic_name]


            phase_averaging_type_list = []
            for averaging_type in averaging_type_list:

                phase_data_df = harmonic_data_df[averaging_type]
                phase_std_type_list = []

                phase_arr = phases_shift(phase_data_df['RF CH A - RF CH B Phase [Rad]'])[0]

                for sigma_type_index in range(len(column_sigma_list)):

                    sigma_arr = phase_data_df[column_sigma_list[sigma_type_index]]

                    phase_av_s = straight_line_fit_params(phase_arr, sigma_arr)
                    phase_av_s.rename({'Weighted Mean':'RF CH A - RF CH B Weighted Averaged Phase [Rad]','Weighted STD':'RF CH A - RF CH B Weighted Averaged Phase STD [Rad]'}, inplace=True)
                    phase_av_s.index.name = 'Data Field'

                    phase_av_s = pd.concat([phase_av_s], keys=[column_phase_std_type_list[sigma_type_index]], names=['Phase STD Type'])
                    phase_std_type_list.append(phase_av_s)

                phase_averaging_type_list.append(pd.concat([pd.concat(phase_std_type_list)], keys=[averaging_type], names=['Averaging Type']))

            phase_average_harmonic_list.append(pd.concat([pd.concat(phase_averaging_type_list)], keys=[harmonic_name], names=['Fourier Harmonic']))

        phase_average_reference_type_list.append(pd.concat([pd.concat(phase_average_harmonic_list)], keys=[reference_type], names=['Phase Reference Type']))

    phase_average_reference_type_s = pd.concat(phase_average_reference_type_list)

    return phase_average_reference_type_s
#%%'
def eliminate_freq_response(df):

    rf_ch_A_df = df[df['Configuration'] == 'A'].iloc[0]
    rf_ch_B_df = df[df['Configuration'] == 'B'].iloc[0]

    phase_difference_reference_type_list = []

    reference_type_list = ['RF Combiner I Reference', 'RF Combiner R Reference']
    harmonics_names_list = ['First Harmonic', 'Second Harmonic']
    averaging_type_list = ['Phase Averaging', 'Phasor Averaging', 'Phasor Averaging Relative To DC']

    column_sigma_list = [
            'RF CH A - RF CH B Phase STD [Rad]',
            'RF CH A - RF CH B Phase RMS Repeat STD [Rad]',
            'RF CH A - RF CH B Phase RMS Averaging Set STD [Rad]']
            #'RF CH A - RF CH B Phase RMS STD With Covariance [Rad]',
            #'RF CH A - RF CH B Phase STD With Covariance [Rad]']

    column_mean_sigma_list = [
            'Phase STDOM [Rad]',
            'Phase RMS Repeat STDOM [Rad]',
            #'Phase Mean RMS STD With Covariance [Rad]',
            'Phase RMS Averaging Set STDOM [Rad]']#,
            #'Phase Mean STD With Covariance [Rad]']

    for reference_type in reference_type_list:

        reference_data_A_df = rf_ch_A_df[reference_type]
        reference_data_B_df = rf_ch_B_df[reference_type]

        phase_difference_harmonic_type_list = []

        for harmonic_level in harmonics_names_list:

            harmonic_data_A_df = reference_data_A_df[harmonic_level]
            harmonic_data_B_df = reference_data_B_df[harmonic_level]

            phase_diff_av_type_list = []

            for averaging_type in averaging_type_list:
                data_A_df = harmonic_data_A_df[averaging_type]
                data_B_df = harmonic_data_B_df[averaging_type]

                phase_difference_s = pd.Series()

                phase_difference_s['RF CH A - RF CH B Phase [Rad]'] = convert_phase_to_2pi_range(data_A_df['Phase [Rad]'] - data_B_df['Phase [Rad]'])

                for sigma_type_index in range(len(column_mean_sigma_list)):
                    phase_difference_s[column_sigma_list[sigma_type_index]] = np.sqrt(data_A_df[column_mean_sigma_list[sigma_type_index]]**2 + data_B_df[column_mean_sigma_list[sigma_type_index]]**2)

                phase_difference_s.index.name = 'Data Field'

                phase_difference_s = pd.concat([phase_difference_s], keys=[averaging_type], names=['Averaging Type'])

                phase_diff_av_type_list.append(phase_difference_s)

            phase_difference_harmonic_type_list.append(pd.concat([pd.concat(phase_diff_av_type_list)], keys=[harmonic_level], names=['Fourier Harmonic']))

        phase_difference_reference_type_list.append(pd.concat([pd.concat(phase_difference_harmonic_type_list)], keys=[reference_type], names=['Phase Reference Type']))
    phase_difference_combined_s = pd.concat(phase_difference_reference_type_list)

    return phase_difference_combined_s
fosof_phases_df
#%%
os.chdir('C:/Users/Helium1/Google Drive/Research/Lamb shift measurement/Test')
fosof_phases_df.to_csv(path_or_buf='test.txt', header=True)
#%%
fosof_phases_df
#%%

fosof_phases_df_2 = pd.read_csv(filepath_or_buffer='test.txt', header=[0,1,2,3,4], index_col=0)
#%%

fosof_phases_df_2
#%%
comment_string_arr
#%%
exp_info_file_loc = 'C:\Research\Lamb shift measurement\Data\FOSOF analyzed data sets'
os.chdir(exp_info_file_loc)
exp_info_file_name = 'fosof_data_sets_info.csv'
exp_info_df = pd.read_csv(filepath_or_buffer=exp_info_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True)
#exp_info_df = exp_info_df.set_index(exp_index_name)
# Convert index to timestamps, otherwise they are read as strings
#exp_info_df.index = pd.to_datetime(exp_info_df.index, infer_datetime_format=True)
#%%
exp_info_df = exp_info_df.set_index('Acquisition Start Date')
#%%
exp_info_df
#%%
os.chdir('C:/Research/Lamb shift measurement/Data/FOSOF analyzed data sets/180511-054606 - FOSOF Acquisition 910 onoff (80 pct), CGX usual flow rate - 0 config, 18 V per cm PD 120V')
data_file = 'data_analyzed v0.1.txt'
exp_params_dict, comment_string_arr = get_experiment_params(data_file)
#%%
exp_params_dict
#%%
def get_experiment_params(data_file_name):
    ''' Extract experiment parameters from the data file

        The parameters are converted to their respective type (int, float, or string) automatically.

        Inputs:
        :data_file_name: name of the data file (txt)

        Outputs:
        :exp_params_dict: dictionary with the experiment parameters
        :comment_string_arr: array of comment strings.
    '''
    # Regular expression for the comment lines
    reg_exp = re.compile(r"""
                            [#]
                            (?P<comment_name>[\S\s]+)
                            =
                            (?P<comment_value>[\S\s]+)$ # Match up to the newline character
                        """
                            , re.VERBOSE)

    # Number of lines in the file. This is monitor if the end of the file is reached
    num_lines = get_file_n_lines(data_file_name)

    # Open the data file in the read only mode to prevent accidental writes to the file
    data_file_object = open(data_file_name, mode='r')

    # Read lines from the file from the beginning of the file until the line does not start with the '#' character

    # Dictionary to store experiment parameters
    exp_params_dict = {}
    line_index = 0
    comments_end_bool = False
    comment_string_arr = []
    while comments_end_bool == False or line_index < num_lines:
        read_line = data_file_object.readline()
        line_index = line_index + 1

        # If we have empty line = '\n', then it is simply ignored. If we did not include this 'if' statement, then trying to perform .lstrip()[0] will result in the error, since '\n'.lstrip() = '' string has zero size = no elements.
        if read_line != '\n':
            if read_line.lstrip()[0] == '#':
                comment_string_arr.append(read_line)
                read_line = read_line.lstrip()
                # Get experiment parameter and name from the file line. Whitespace characters are stripped from left and right ends of the string
                comment_line_parsed_object = reg_exp.search(read_line)
                comment_name = comment_line_parsed_object.group('comment_name').strip()
                comment_value = comment_line_parsed_object.group('comment_value').strip()

                # Converting parameter values to proper type. First we try to convert the value to integer. If it fails, then we try to convert to float. If this also fails, then the value is a string. However, we can also have booleans - we check for that as well.
                # If the parameter name is 'Configuration', then at '0' configuration we do not want '0' to turn into integer 0.
                # If the parameter name is 'Offset Frequency [Hz]', then it is possible for having comma separated integer offset frequencies. Separate analysis
                if comment_name != 'Configuration':
                    if comment_name == 'Offset Frequency [Hz]':
                        comment_value = [int(i) for i in comment_value.split(',')]
                    else:
                        try:
                            comment_value = int(comment_value)
                        except ValueError:
                            try:
                                comment_value = float(comment_value)
                            except ValueError:
                                comment_value = str(comment_value)
                                if comment_value == 'True':
                                    comment_value = True
                                if comment_value == 'False':
                                    comment_value = False
                # Appending the experiment parameter to the dictionary
                exp_params_dict[comment_name] = comment_value
            else:
                # It is possible that the comments are not necesserily in the beginning of the file, thus we do not want to stop the search for the comments right away.
                if len(exp_params_dict) > 0:
                    comments_end_bool = True

    # Close the handle to the file
    data_file_object.flush()
    data_file_object.close()
    return exp_params_dict, comment_string_arr
#%%
# Regular expression for the comment lines
reg_exp = re.compile(r"""
                        [#]
                        (?P<comment_name>[\S\s][^=]+)
                        =
                        (?P<comment_value>[\S\s]+)$ # Match up to the newline character
                    """
                        , re.VERBOSE)

#%%
read_line = '# Notes = Usual = Mass = FLow ahsfjsdhfjsdfj dfjsdkl jf'
comment_line_parsed_object = reg_exp.search(read_line)
comment_line_parsed_object
#%%
comment_name = comment_line_parsed_object.group('comment_name').strip()
comment_value = comment_line_parsed_object.group('comment_value').strip()
#%%
comment_name
comment_value
#%%
1/(1.25*10)/np.sqrt(41*2*4*2*2)*1000*20
#%%
data_set_type_s
#%%
from numpy import linspace,exp
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
x = linspace(-3, 3, 100)
y = exp(-x**2) + randn(100)/10
s = UnivariateSpline(x, y, s=1)
xs = linspace(-3, 3, 1000)
ys = s(xs)
plt.plot(x, y, '.-')
plt.plot(xs, ys)
plt.show()
#%%
phase_arr = np.array([6.287665])
#%%
phase_arr[phase_arr > np.pi] % (2*np.pi)
#%%
def get_exp_tim_info(self):
    ''' Outputs acquisition start time in UNIX format [s] and duration of the experiment [s].
    '''
    return self.exp_data_frame['Time'].min(), self.exp_data_frame['Elapsed Time [s]'].max()
#%%
os.chdir(saving_folder_location)

exp_info_index_name = 'Experiment Folder Name'
exp_info_file_name = 'fosof_data_sets_info.csv'
exp_info_df = pd.read_csv(filepath_or_buffer=exp_info_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True)

#%%
exp_info_df = exp_info_df.set_index(exp_info_index_name)
#%%
exp_folder_name = '180607-183458 - FOSOF Acquisition - 0 config, 8 V per cm PD ON 120 V, 49.86 kV, 898-900 MHz'
#%%
if exp_folder_name in exp_info_df.index:
    exp_params_dict['Comments'] = exp_info_df.loc[exp_folder_name]['Comments']
#%%
'Comments' in exp_params_dict.keys()
