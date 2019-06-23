'''
2019-06-04

Data required by the Science journal. It includes averaged FOSOF phases (rf frequency vs FOSOF phase for both the 0 and pi configuration) for all of the datasets that were used for determination of the final result. The data was provided by Travis
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

travis_data_folder_path = path_data_df.loc['Travis Data Folder'].values[0].replace('\\', '/')

code_folder_path = path_data_df.loc['Code Folder'].values[0].replace('\\', '/')

from exp_data_analysis import *
#%%
os.chdir(code_folder_path)
data_df = pd.read_csv('all_data.csv')

# Sanity checks
if data_df[((data_df['Offset Frequency [Hz]zero'] - data_df['Offset Frequency [Hz]pi']) != 0)].shape[0] != 0:
    print('WARNING!!!')

if data_df[((data_df['rf Field Amplitude [V/cm]zero'] - data_df['rf Field Amplitude [V/cm]pi']) != 0)].shape[0] != 0:
    print('WARNING!!!')

if data_df[((data_df['Accelerating Voltage [kV]zero'] - data_df['Accelerating Voltage [kV]pi']) != 0)].shape[0] != 0:
    print('WARNING!!!')

if data_df[(data_df['Proton Deflector Statezero'] != data_df['Proton Deflector Statepi'])].shape[0] != 0:
    print('WARNING!!!')

if data_df[((data_df['B_x [Gauss]zero'] - data_df['B_x [Gauss]pi']) != 0)].shape[0] != 0:
    print('WARNING!!!')

if data_df[((data_df['B_y [Gauss]zero'] - data_df['B_y [Gauss]pi']) != 0)].shape[0] != 0:
    print('WARNING!!!')

if data_df[(data_df['Pre-Quench 910 Statezero'] != data_df['Pre-Quench 910 Statepi'])].shape[0] != 0:
    print('WARNING!!!')

if data_df[(data_df['Proton Deflector Voltage [V]zero'] != data_df['Proton Deflector Voltage [V]pi'])].shape[0] != 0:
    print('WARNING!!!')

if data_df[(data_df['Mass Flow Rate [CC]zero'] != data_df['Mass Flow Rate [CC]pi'])].shape[0] != 0:
    print('WARNING!!!')

BLIND_OFFSET = 0.03174024731

# Add 1/2 of the offset frequency to each of the rf carrier frequencies. We also add the blind offset to the frequencies
data_df['Waveguide Carrier Frequency [MHz]'] = data_df['Waveguide Carrier Frequency [MHz]'] + 1/2 * data_df['Offset Frequency [Hz]zero'] * 1E-6 + BLIND_OFFSET

data_chosen_df = data_df[['Waveguide Separation [cm]', 'Accelerating Voltage [kV]zero', 'rf Field Amplitude [V/cm]zero', 'Date Stringzero', 'Date Stringpi', 'Waveguide Carrier Frequency [MHz]', 'Combiner/Harmonic', 'Average Phase [rad]zero', 'Phase Error [rad]zero', 'Average Phase [rad]pi', 'Phase Error [rad]pi']]

data_chosen_df.rename(columns={'Accelerating Voltage [kV]zero': 'Accelerating Voltage [kV]', 'rf Field Amplitude [V/cm]zero': 'rf Field Amplitude [V/cm]', 'Date Stringzero': 'Dataset (0) Acquisition Start Date [yymmdd-hhmmss]', 'Date Stringpi': 'Dataset (180) Acquisition Start Date [yymmdd-hhmmss]', 'Waveguide Carrier Frequency [MHz]': 'rf Frequency [MHz]', 'Average Phase [rad]zero': 'FOSOF phase (0) [rad]', 'Phase Error [rad]zero': 'FOSOF Phase Error (0) [rad]', 'Average Phase [rad]pi': 'FOSOF phase (180) [rad]', 'Phase Error [rad]pi': 'FOSOF Phase Error (180) [rad]', 'Combiner/Harmonic': 'Reference Combiner'}, inplace=True)

# Changing accelerating voltage to beam speed
acc_volt_unique_arr = data_chosen_df['Accelerating Voltage [kV]'].drop_duplicates().values
for i in range(acc_volt_unique_arr.shape[0]):
    acc_volt_s = acc_volt_unique_arr[i]
    if acc_volt_s == 49.86:
        beam_speed = 3.22
    if acc_volt_s == 22.17:
        beam_speed = 2.25
    if acc_volt_s == 16.27:
        beam_speed = 1.99

    data_chosen_df.loc[data_chosen_df[data_chosen_df['Accelerating Voltage [kV]'] == acc_volt_s].index, 'Accelerating Voltage [kV]'] = beam_speed

data_chosen_df.rename(columns={'Accelerating Voltage [kV]': 'Beam Speed [mm/ns]'}, inplace=True)

# 19 V/cm data is not needed
data_chosen_df.drop(data_chosen_df[data_chosen_df['rf Field Amplitude [V/cm]'] == 19].index, inplace=True)

# Another data set needs to be removed (Travis: 'I believe the other dataset I included on accident was one of the mass flow controller tests. Here are the timestamps for the datasets: 180329-171158 (zero) and 180329-163541 (pi).')
# I think that Travis is wrong. He has some rogue dataset that I have no idea where it came from. The data set (0 config) is '180706-170901'. I am removing it instead.

data_chosen_df.drop(data_chosen_df[data_chosen_df['Dataset (0) Acquisition Start Date [yymmdd-hhmmss]'] == '180706-170901'].index, inplace=True)

# Adding the dataset Index column
data_chosen_df['Dataset Index'] = 0

data_set_0_name_arr = data_chosen_df['Dataset (0) Acquisition Start Date [yymmdd-hhmmss]'].drop_duplicates().values

data_set_index = 0
for i in range(data_set_0_name_arr.shape[0]):
    data_set_index = data_set_index + 1
    data_chosen_df.loc[data_chosen_df[data_chosen_df['Dataset (0) Acquisition Start Date [yymmdd-hhmmss]'] == data_set_0_name_arr[i]].index, 'Dataset Index'] = data_set_index

# Renaming combiner labels
data_chosen_df.loc[data_chosen_df[data_chosen_df['Reference Combiner'] == 'Combiner 1, Harmonic 1'].index, 'Reference Combiner'] = 'Combiner C1'
data_chosen_df.loc[data_chosen_df[data_chosen_df['Reference Combiner'] == 'Combiner 2, Harmonic 1'].index, 'Reference Combiner'] = 'Combiner C2'

data_chosen_df = data_chosen_df.set_index(['Dataset Index', 'Waveguide Separation [cm]', 'Beam Speed [mm/ns]', 'rf Field Amplitude [V/cm]', 'Dataset (0) Acquisition Start Date [yymmdd-hhmmss]', 'Dataset (180) Acquisition Start Date [yymmdd-hhmmss]', 'Reference Combiner', 'rf Frequency [MHz]']).sort_index()
#%%
# Calculation of selected zero-crossing frequencies, to make sure that we get the same answers as in the paper.

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

data_grouped = data_chosen_df.groupby('Dataset (0) Acquisition Start Date [yymmdd-hhmmss]')

zero_cross_df = pd.DataFrame()

for group_name, current_df in data_grouped:

    current_df = current_df.reset_index()
    wvg_sep = current_df['Waveguide Separation [cm]'][0]
    beam_speed = current_df['Beam Speed [mm/ns]'][0]
    rf_field = current_df['rf Field Amplitude [V/cm]'][0]
    dataset_timestamp =  group_name + ' - ' + current_df['Dataset (180) Acquisition Start Date [yymmdd-hhmmss]'][0]
    single_data_set_df = current_df.set_index('Reference Combiner').drop(columns=['Dataset Index', 'Waveguide Separation [cm]', 'Beam Speed [mm/ns]', 'rf Field Amplitude [V/cm]', 'Dataset (0) Acquisition Start Date [yymmdd-hhmmss]', 'Dataset (180) Acquisition Start Date [yymmdd-hhmmss]'])

    comb_c1_df = single_data_set_df.loc['Combiner C1'].reset_index()
    comb_c2_df = single_data_set_df.loc['Combiner C2'].reset_index()

    # Calculate zero-crossing frequencies for two combiners and then average the results.
    comb_selected_df = comb_c1_df

    x_data_arr = comb_selected_df['rf Frequency [MHz]']
    y_data_arr = 1/2 * (comb_selected_df['FOSOF phase (0) [rad]'] - comb_selected_df['FOSOF phase (180) [rad]']).values

    y_sigma_arr = 1/2 * np.sqrt(comb_selected_df['FOSOF Phase Error (0) [rad]']**2 + comb_selected_df['FOSOF Phase Error (180) [rad]']**2)

    #y_data_arr = correct_FOSOF_phases_zero_crossing(x_data_arr, y_data_arr, y_sigma_arr)

    fit_param_C1_dict = calc_fosof_lineshape_param(x_data_arr, y_data_arr, y_sigma_arr)

    comb_selected_df = comb_c2_df

    x_data_arr = comb_selected_df['rf Frequency [MHz]']
    y_data_arr = 1/2 * (comb_selected_df['FOSOF phase (0) [rad]'] - comb_selected_df['FOSOF phase (180) [rad]']).values

    y_sigma_arr = 1/2 * np.sqrt(comb_selected_df['FOSOF Phase Error (0) [rad]']**2 + comb_selected_df['FOSOF Phase Error (180) [rad]']**2)

    #y_data_arr = correct_FOSOF_phases_zero_crossing(x_data_arr, y_data_arr, y_sigma_arr)

    fit_param_C2_dict = calc_fosof_lineshape_param(x_data_arr, y_data_arr, y_sigma_arr)

    comb_diff = np.abs(fit_param_C1_dict['Zero-crossing Frequency [MHz]'] - fit_param_C2_dict['Zero-crossing Frequency [MHz]']) / 2

    comb_diff = 0

    result_s = pd.Series({'Dataset Timestamp': dataset_timestamp, 'Waveguide Separation [cm]': wvg_sep, 'Beam Speed [mm/ns]': beam_speed, 'rf Field Amplitude [V/cm]': rf_field, 'Zero-crossing Frequency [MHz]': 1/2 * (fit_param_C1_dict['Zero-crossing Frequency [MHz]'] + fit_param_C2_dict['Zero-crossing Frequency [MHz]']), 'Zero-crossing Frequency STD [MHz]': np.sqrt(1/2 * (fit_param_C1_dict['Zero-crossing Frequency STD [MHz]']**2 + fit_param_C2_dict['Zero-crossing Frequency STD [MHz]']**2) + comb_diff**2), 'Reduced Chi-Squared': fit_param_C1_dict['Reduced Chi-Squared']})

    result_s.name = group_name

    zero_cross_df = zero_cross_df.append(result_s)

# Changing beam speed to accelerating voltage
beam_speed_unique_arr = zero_cross_df['Beam Speed [mm/ns]'].drop_duplicates().values
for i in range(beam_speed_unique_arr.shape[0]):
    beam_speed_s = beam_speed_unique_arr[i]
    if beam_speed_s == 3.22:
        acc_volt_s = 49.86
    if beam_speed_s == 2.25:
        acc_volt_s = 22.17
    if beam_speed_s == 1.99:
        acc_volt_s = 16.27
    zero_cross_df.loc[zero_cross_df[zero_cross_df['Beam Speed [mm/ns]'] == beam_speed_s].index, 'Beam Speed [mm/ns]'] = acc_volt_s

zero_cross_df.rename(columns={'Beam Speed [mm/ns]': 'Accelerating Voltage [kV]', 'rf Field Amplitude [V/cm]': 'Waveguide Electric Field [V/cm]'}, inplace=True)

zero_cross_df = zero_cross_df.set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Dataset Timestamp']).sort_index()

#%%
'''Travis's data
'''
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

fosof_lineshape_param_av_comb_t_df['Combiner Uncertainty [MHz]'] = 0

# We need to shift all of the frequency by 1/2 of the frequency offset
fosof_lineshape_param_av_comb_t_df['Zero-crossing Frequency [MHz]'] = fosof_lineshape_param_av_comb_t_df['Zero-crossing Frequency [MHz]'] + fosof_lineshape_param_av_comb_t_df['Offset Frequency [Hz]'] / 1E6 / 2

fosof_lineshape_param_av_comb_t_df['Reduced Chi-Squared'] = (rf_comb_I_df['Reduced Chi-Squared'] + rf_comb_R_df['Reduced Chi-Squared']) / 2

zero_cross_orig_df = fosof_lineshape_param_av_comb_t_df.loc['Normal', 1]
zero_cross_orig_df['Zero-crossing Frequency STD [MHz]'] = np.sqrt(zero_cross_orig_df['Zero-crossing Frequency STD (Normalized) [MHz]']**2 + zero_cross_orig_df['Combiner Uncertainty [MHz]']**2)

zero_cross_orig_df['Zero-crossing Frequency [MHz]'] = zero_cross_orig_df['Zero-crossing Frequency [MHz]'] + BLIND_OFFSET
#%%
# Datasets NOT in the original dataset (from the summary file)
zero_cross_df.index.get_level_values('Dataset Timestamp').drop_duplicates().difference(zero_cross_orig_df.index.get_level_values('Dataset Timestamp').drop_duplicates())
#%%
# Datasets NOT in the list of phases provided by Travis for the paper.
zero_cross_orig_df.index.get_level_values('Dataset Timestamp').drop_duplicates().difference(zero_cross_df.index.get_level_values('Dataset Timestamp').drop_duplicates())
#%%
# Difference between the zero-crossing frequencies obtain from the summary file that Travis provided to me, and from the phases of all of the datasets that Travis sent to me. Looking at the absolute value. The difference is in Hz
diff_df = np.abs((zero_cross_orig_df[['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD [MHz]', 'Reduced Chi-Squared']] - zero_cross_df) * 1E6)

print('Maximum zero-crossing frequency difference [Hz]: ' + str(diff_df['Zero-crossing Frequency [MHz]'].max()))
print('Maximum zero-crossing frequency STD difference [kHz]: ' + str(diff_df['Zero-crossing Frequency STD [MHz]'].max()))
print('Maximum Reduced Chi-squared difference: ' + str(diff_df['Reduced Chi-Squared'].max()*1E-6))
#%%
# After performing this type of comparison we now want to average the FOSOF phases (not the resulting zero-crossing frequencies) for two combiners and make sure that we still get the same answers

# Firstly, we need to average the phases for the two combiners

data_comb_avg_df = data_chosen_df.reset_index('Reference Combiner')

data_c1_df = data_comb_avg_df[data_comb_avg_df['Reference Combiner'] == 'Combiner C1'].drop(columns=['Reference Combiner'])
data_c2_df = data_comb_avg_df[data_comb_avg_df['Reference Combiner'] == 'Combiner C2'].drop(columns=['Reference Combiner'])

data_comb_avg_df = data_c1_df.copy()

data_comb_avg_df['FOSOF phase (0) [rad]'] = 1/2 * (data_c1_df['FOSOF phase (0) [rad]'] + data_c2_df['FOSOF phase (0) [rad]'])
data_comb_avg_df['FOSOF phase (180) [rad]'] = 1/2 * (data_c1_df['FOSOF phase (180) [rad]'] + data_c2_df['FOSOF phase (180) [rad]'])

data_comb_avg_df['FOSOF Phase Error (0) [rad]'] = 1/np.sqrt(2) * np.sqrt(data_c1_df['FOSOF Phase Error (0) [rad]']**2 + data_c2_df['FOSOF Phase Error (0) [rad]']**2)
data_comb_avg_df['FOSOF Phase Error (180) [rad]'] = 1/np.sqrt(2) * np.sqrt(data_c1_df['FOSOF Phase Error (180) [rad]']**2 + data_c2_df['FOSOF Phase Error (180) [rad]']**2)

zero_cross_avg_comb_df = pd.DataFrame()

data_grouped = data_comb_avg_df.groupby('Dataset (0) Acquisition Start Date [yymmdd-hhmmss]')

for group_name, current_df in data_grouped:

    current_df = current_df.reset_index()
    wvg_sep = current_df['Waveguide Separation [cm]'][0]
    beam_speed = current_df['Beam Speed [mm/ns]'][0]
    rf_field = current_df['rf Field Amplitude [V/cm]'][0]
    dataset_timestamp =  group_name + ' - ' + current_df['Dataset (180) Acquisition Start Date [yymmdd-hhmmss]'][0]
    single_data_set_df = current_df.reset_index().drop(columns=['Dataset Index', 'Waveguide Separation [cm]', 'Beam Speed [mm/ns]', 'rf Field Amplitude [V/cm]', 'Dataset (0) Acquisition Start Date [yymmdd-hhmmss]', 'Dataset (180) Acquisition Start Date [yymmdd-hhmmss]'])

    x_data_s = single_data_set_df['rf Frequency [MHz]']

    y_data_avg_0_s = single_data_set_df['FOSOF phase (0) [rad]']
    y_data_avg_180_s = single_data_set_df['FOSOF phase (180) [rad]']
    y_data_s = 1/2 * (y_data_avg_0_s - y_data_avg_180_s)

    y_sigma_0_s = single_data_set_df['FOSOF Phase Error (0) [rad]']

    y_sigma_180_s = single_data_set_df['FOSOF Phase Error (180) [rad]']

    y_sigma_s = 1/2 * np.sqrt(y_sigma_0_s**2 + y_sigma_180_s**2)

    #y_data_arr = correct_FOSOF_phases_zero_crossing(x_data_arr, y_data_arr, y_sigma_arr)

    fit_param_C_avg_dict = calc_fosof_lineshape_param(x_data_s, y_data_s, y_sigma_s)

    result_s = pd.Series({'Dataset Timestamp': dataset_timestamp, 'Waveguide Separation [cm]': wvg_sep, 'Beam Speed [mm/ns]': beam_speed, 'rf Field Amplitude [V/cm]': rf_field, 'Zero-crossing Frequency [MHz]': fit_param_C_avg_dict['Zero-crossing Frequency [MHz]'], 'Zero-crossing Frequency STD [MHz]': fit_param_C_avg_dict['Zero-crossing Frequency STD [MHz]'], 'Reduced Chi-Squared': fit_param_C_avg_dict['Reduced Chi-Squared']})

    result_s.name = group_name

    zero_cross_avg_comb_df = zero_cross_avg_comb_df.append(result_s)

# Changing beam speed to accelerating voltage
beam_speed_unique_arr = zero_cross_avg_comb_df['Beam Speed [mm/ns]'].drop_duplicates().values
for i in range(beam_speed_unique_arr.shape[0]):
    beam_speed_s = beam_speed_unique_arr[i]
    if beam_speed_s == 3.22:
        acc_volt_s = 49.86
    if beam_speed_s == 2.25:
        acc_volt_s = 22.17
    if beam_speed_s == 1.99:
        acc_volt_s = 16.27
    zero_cross_avg_comb_df.loc[zero_cross_avg_comb_df[zero_cross_avg_comb_df['Beam Speed [mm/ns]'] == beam_speed_s].index, 'Beam Speed [mm/ns]'] = acc_volt_s

zero_cross_avg_comb_df.rename(columns={'Beam Speed [mm/ns]': 'Accelerating Voltage [kV]', 'rf Field Amplitude [V/cm]': 'Waveguide Electric Field [V/cm]'}, inplace=True)

zero_cross_avg_comb_df = zero_cross_avg_comb_df.set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Dataset Timestamp']).sort_index()
#%%
# Difference between the zero-crossing frequencies obtain from the summary file that Travis provided to me, and from the phases of all of the datasets that Travis sent to me. Looking at the absolute value. The difference is in Hz
diff_df = np.abs((zero_cross_orig_df[['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD [MHz]', 'Reduced Chi-Squared']] - zero_cross_avg_comb_df) * 1E6)

print('Maximum zero-crossing frequency difference [Hz]: ' + str(diff_df['Zero-crossing Frequency [MHz]'].max()))
print('Maximum zero-crossing frequency STD difference [kHz]: ' + str(diff_df['Zero-crossing Frequency STD [MHz]'].max()))
print('Maximum Reduced Chi-squared difference: ' + str(diff_df['Reduced Chi-Squared'].max()*1E-6))
#%%
# Export the data

# Round the data to a given number of decimal places
n_dec = 5
col_to_round_list = ['rf Frequency [MHz]', 'FOSOF phase (0) [rad]', 'FOSOF Phase Error (0) [rad]', 'FOSOF phase (180) [rad]', 'FOSOF Phase Error (180) [rad]']

data_comb_avg_to_export_df = data_comb_avg_df.reset_index()


data_comb_avg_to_export_df[col_to_round_list] = data_comb_avg_to_export_df[col_to_round_list].transform(lambda x: np.around(x, n_dec))

for col_name in col_to_round_list:
    data_comb_avg_to_export_df[col_name] = np.array(['{:.5f}'.format(data_comb_avg_to_export_df[col_name].values[i]) for i in range(data_comb_avg_to_export_df[col_name].shape[0])])

# Export to txt file
os.chdir(code_folder_path)

data_comb_avg_to_export_df.rename(columns={'FOSOF phase (0) [rad]': 'FOSOF Phase (0) [rad]', 'FOSOF phase (180) [rad]': 'FOSOF Phase (180) [rad]'}).to_csv(path_or_buf='paper_data.txt', index=False, sep='\t')
#%%
os.chdir(r'C:\Users\Nikita\Downloads')
# Checking if after exporting the data is still the same
data_import_df = pd.read_csv('paper_data (2).txt', sep='\t')
data_import_df = data_import_df.set_index(['Dataset Index', 'Waveguide Separation [cm]', 'Beam Speed [mm/ns]', 'rf Field Amplitude [V/cm]', 'Dataset (0) Acquisition Start Date [yymmdd-hhmmss]', 'Dataset (180) Acquisition Start Date [yymmdd-hhmmss]', 'rf Frequency [MHz]']).sort_index()
#%%
zero_cross_avg_comb_df = pd.DataFrame()

data_grouped = data_import_df.groupby('Dataset (0) Acquisition Start Date [yymmdd-hhmmss]')

for group_name, current_df in data_grouped:
    current_df = current_df.reset_index()
    wvg_sep = current_df['Waveguide Separation [cm]'][0]
    beam_speed = current_df['Beam Speed [mm/ns]'][0]
    rf_field = current_df['rf Field Amplitude [V/cm]'][0]
    dataset_timestamp =  group_name + ' - ' + current_df['Dataset (180) Acquisition Start Date [yymmdd-hhmmss]'][0]
    single_data_set_df = current_df.reset_index().drop(columns=['Dataset Index', 'Waveguide Separation [cm]', 'Beam Speed [mm/ns]', 'rf Field Amplitude [V/cm]', 'Dataset (0) Acquisition Start Date [yymmdd-hhmmss]', 'Dataset (180) Acquisition Start Date [yymmdd-hhmmss]'])

    x_data_s = single_data_set_df['rf Frequency [MHz]']

    y_data_avg_0_s = single_data_set_df['FOSOF Phase (0) rad']
    y_data_avg_180_s = single_data_set_df['FOSOF Phase (180) [rad]']
    y_data_s = 1/2 * (y_data_avg_0_s - y_data_avg_180_s)

    y_sigma_0_s = single_data_set_df['FOSOF Phase Error (0) [rad]']

    y_sigma_180_s = single_data_set_df['FOSOF Phase Error (180) [rad]']

    y_sigma_s = 1/2 * np.sqrt(y_sigma_0_s**2 + y_sigma_180_s**2)

    #y_data_arr = correct_FOSOF_phases_zero_crossing(x_data_arr, y_data_arr, y_sigma_arr)

    fit_param_C_avg_dict = calc_fosof_lineshape_param(x_data_s, y_data_s, y_sigma_s)

    result_s = pd.Series({'Dataset Timestamp': dataset_timestamp, 'Waveguide Separation [cm]': wvg_sep, 'Beam Speed [mm/ns]': beam_speed, 'rf Field Amplitude [V/cm]': rf_field, 'Zero-crossing Frequency [MHz]': fit_param_C_avg_dict['Zero-crossing Frequency [MHz]'], 'Zero-crossing Frequency STD [MHz]': fit_param_C_avg_dict['Zero-crossing Frequency STD [MHz]'], 'Reduced Chi-Squared': fit_param_C_avg_dict['Reduced Chi-Squared']})

    result_s.name = group_name

    zero_cross_avg_comb_df = zero_cross_avg_comb_df.append(result_s)

# Changing beam speed to accelerating voltage
beam_speed_unique_arr = zero_cross_avg_comb_df['Beam Speed [mm/ns]'].drop_duplicates().values
for i in range(beam_speed_unique_arr.shape[0]):
    beam_speed_s = beam_speed_unique_arr[i]
    if beam_speed_s == 3.22:
        acc_volt_s = 49.86
    if beam_speed_s == 2.25:
        acc_volt_s = 22.17
    if beam_speed_s == 1.99:
        acc_volt_s = 16.27
    zero_cross_avg_comb_df.loc[zero_cross_avg_comb_df[zero_cross_avg_comb_df['Beam Speed [mm/ns]'] == beam_speed_s].index, 'Beam Speed [mm/ns]'] = acc_volt_s

zero_cross_avg_comb_df.rename(columns={'Beam Speed [mm/ns]': 'Accelerating Voltage [kV]', 'rf Field Amplitude [V/cm]': 'Waveguide Electric Field [V/cm]'}, inplace=True)

zero_cross_avg_comb_df = zero_cross_avg_comb_df.set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Dataset Timestamp']).sort_index()
#%%
# Difference between the zero-crossing frequencies obtain from the summary file that Travis provided to me, and from the phases of all of the datasets that Travis sent to me. Looking at the absolute value. The difference is in Hz
diff_df = np.abs((zero_cross_orig_df[['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD [MHz]', 'Reduced Chi-Squared']] - zero_cross_avg_comb_df) * 1E6)

print('Maximum zero-crossing frequency difference [Hz]: ' + str(diff_df['Zero-crossing Frequency [MHz]'].max()))
print('Maximum zero-crossing frequency STD difference [kHz]: ' + str(diff_df['Zero-crossing Frequency STD [MHz]'].max()))
print('Maximum Reduced Chi-squared difference: ' + str(diff_df['Reduced Chi-Squared'].max()*1E-6))
