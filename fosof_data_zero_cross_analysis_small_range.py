'''
2019-04-12

Same as the fosof_data_zero_cross_analysis.py, but for frequency ranges of less than 4 MHz. In this case, especially for very small ranges, there is not enough of data points to reliably determine the slope and the offset of the linear fit. However, we can assume that the slope is equal to the average slope of the data, acquired at range multiple of 1. We also assume that for all of the range the data is following linear trend. In this case, it is easy to show, that the zero-crossing frequency for small range, f_zc_small is equal to:

f_zc_small = f_zc - <Delta Theta>/T, where f_zc is the average zero-crossing frequency for the range multiple of 1 data. <Delta Theta> is the average difference of the data from the expected trend, given by T(f-f_zc); T is the average slope (for the range multiple of 1 data).

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
import fosof_data_set_analysis
import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt


import threading
from queue import Queue

# Package for wrapping long string to a paragraph
import textwrap

from tkinter import *
from tkinter import ttk

from tkinter import messagebox

import wvg_power_calib_analysis

#%%
def get_freq_range_data(df, range_multiple):
    ''' We want to select the data in certain ranges symmetrically about the 'central' frequency of 910 MHz (blind offset is not taken into account). This way we can test if the zero-crossing is consistent for various ranges of data. This function returns the data corresponding the the chosen range_multiple value, where a value of 1 corresponds to roughly +-2 MHz range about 910 MHz, value of 2 is roughly 906-908 and 912-914 MHz, etc.
    '''
    central_freq = 910

    # In principle we want to take a look at the data in 2 MHz wide chunk on each side (left and right) of the central frequency. However, because of the up to 100 kHz jitters, we want to make sure that we can cover the additional 0.1 MHz on each side about 910 MHz. Since we took the data at most at 8 x the normal range = 32 MHz = 16 MHz in each direction, we need to make sure that we pick the range is such a way that we are 16.1 MHz on each direction at 8 x the range. This translates into the 2.0125 MHz range.
    side_range = 2.1

    if range_multiple >= 1:
        # Ranges of frequencies on each side to pick.
        left_side = [central_freq-side_range*range_multiple, central_freq-side_range*range_multiple+side_range]
        right_side = [side_range*range_multiple + central_freq-side_range, side_range*range_multiple + central_freq]

        # Notice the <= and > signs. These are picked to have non-overlapping limits for different range multiples.
        return df.loc[((df.index.get_level_values('Waveguide Carrier Frequency [MHz]') > left_side[0]) & (df.index.get_level_values('Waveguide Carrier Frequency [MHz]') <= left_side[1])) | ((df.index.get_level_values('Waveguide Carrier Frequency [MHz]') > right_side[0]) & (df.index.get_level_values('Waveguide Carrier Frequency [MHz]') <= right_side[1]))]

    # In case the range multiple is less than 1 then we have a simple situation, when we just have the data around 910 MHz, without any gaps in the range.

    else:
        left_side = central_freq - range_multiple * side_range
        right_side = central_freq + range_multiple * side_range

        return df.loc[((df.index.get_level_values('Waveguide Carrier Frequency [MHz]') >= left_side) & (df.index.get_level_values('Waveguide Carrier Frequency [MHz]') <= right_side))]

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

def get_fosof_lineshape(phase_data_df):
    ''' Given the dataframe of the phases, gives the fosof lineshape parameters.
    '''

    ph_ref_type = phase_data_df.columns.get_level_values('Phase Reference Type')[0]
    fourier_harm = phase_data_df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = phase_data_df.columns.get_level_values('Averaging Type')[0]
    std_type = phase_data_df.columns.get_level_values('STD Type')[0]

    phase_data_df = phase_data_df[ph_ref_type, fourier_harm, av_type, std_type]

    x_data_arr = phase_data_df.index.get_level_values('Waveguide Carrier Frequency [MHz]').values
    y_data_arr = phase_data_df['Weighted Mean'].values
    y_sigma_arr = phase_data_df['Weighted STD'].values

    # Correct for 0-2*np.pi jumps

    y_data_arr = correct_FOSOF_phases_zero_crossing(x_data_arr, y_data_arr, y_sigma_arr)

    #Important division by a factor of 4 that was explained before
    y_data_arr = y_data_arr / 4
    y_sigma_arr = y_sigma_arr / 4

    fit_param_dict = calc_fosof_lineshape_param(x_data_arr, y_data_arr, y_sigma_arr)

    return pd.Series(fit_param_dict)

# Two following functions are for averaging phases when there are several experiments combined that had a subset of identical frequencies. It was checked directly that the answers obtained by not averaging the frequencies (simply including all of the points in the line fit) or by averaging the frequencies gave the same result (as one would expect intuitively) FOR THE CASE WHEN THERE WAS NO EXPANSION DUE TO >1 REDUCED CHI-SQUARED. I.e., sometimes, by averaging the phases for the same frequencies, the reduced chi-squared of the resulting average was larger than one, which forced me to expand the uncertainty. In this case the answers obtained for the zero-phase crossing frequencies are tiny bit different. I guess, since, at most, we have only two sets of data with equal frequencies that are getting combined like that, then the prob. for obtaining that specific Reduced Chi-squared > 1 is quite large. Therefore, there is no reason for expanding the error bars.
def av_freq_data(df):

    ph_ref_type = df.columns.get_level_values('Phase Reference Type')[0]
    fourier_harm = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]

    selected_df = df.reset_index('Waveguide Carrier Frequency [MHz]', drop=True)[ph_ref_type, fourier_harm, av_type, std_type]

    # Shift the phases before performing the weighted average
    selected_df['Weighted Mean'] = selected_df[['Weighted Mean']].transform(lambda x: phases_shift(x)[0])

    # Determine the weighted average
    av_s = straight_line_fit_params(selected_df['Weighted Mean'].values, selected_df['Weighted STD'].values)

    # # Expand the weighted uncertainty for the chi-squared of larger than 1
    # if av_s['Reduced Chi Squared'] > 1:
    #     #print('Reduced Chi Squared of more than 1 was found: ' + str(av_s['Reduced Chi Squared']))
    #     av_s['Weighted STD'] = av_s['Weighted STD'] * np.sqrt(av_s['Reduced Chi Squared'])

    to_return_df = df.copy()

    to_return_df = to_return_df.iloc[[0]]
    to_return_df[ph_ref_type, fourier_harm, av_type, std_type, 'Weighted Mean'] = av_s['Weighted Mean']
    to_return_df[ph_ref_type, fourier_harm, av_type, std_type, 'Weighted STD'] = av_s['Weighted STD']

    return to_return_df[ph_ref_type, fourier_harm, av_type, std_type]

def get_av_phase_dupl_freq(df):
    if df.shape[0] > 1:
        freq_dupl = df.index.get_level_values('Waveguide Carrier Frequency [MHz]').drop_duplicates()[0]
        #print('Frequency ' + str(freq_dupl) + ' MHz is duplicated. Averaging the phases...')

        return df.groupby(level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns').apply(av_freq_data).reset_index('Waveguide Carrier Frequency [MHz]', drop=True)
    else:
        return df.reset_index('Waveguide Carrier Frequency [MHz]', drop=True)

def get_fosof_lineshape_for_param_group(fosof_phase_exp_group_df):
    ''' Gives the lineshape parameters for a set of grouped experiments.
    '''
    #print(fosof_phase_exp_group_df.index.values)
    #fosof_phase_exp_av_freq_df = fosof_phase_exp_group_df
    fosof_phase_exp_av_freq_df = fosof_phase_exp_group_df.groupby('Waveguide Carrier Frequency [MHz]').apply(get_av_phase_dupl_freq)

    fosof_phase_exp_av_freq_df = fosof_phase_exp_av_freq_df.reorder_levels(['Group ID', 'Beam RMS Radius [mm]', 'Experiment ID', 'Waveguide Carrier Frequency [MHz]'])

    fosof_phase_exp_grouped = fosof_phase_exp_av_freq_df.groupby(level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns')

    return fosof_phase_exp_grouped.apply(get_fosof_lineshape)
#%%
saving_folder_location = fosof_analyzed_data_folder_path

fosof_phase_data_file_name = 'fosof_phase_grouped_list.csv'
grouped_exp_param_file_name = 'grouped_exp_param_list.csv'

os.chdir(saving_folder_location)

fosof_phase_grouped_df = pd.read_csv(filepath_or_buffer=fosof_phase_data_file_name, delimiter=',', comment='#', header=[0, 1, 2, 3, 4], skip_blank_lines=True, index_col=[0, 1, 2, 3])

grouped_exp_df = pd.read_csv(filepath_or_buffer=grouped_exp_param_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True, index_col=0)



# We now group the data by the variaty of parameters and find the zero-crossing frequency for each grouped experiment. The important point is that we also group the data by the non-overalapping range of frequencies acquired for a given grouped experiment. Thus a given grouped experiment might appear in several different parameter groups, since its frequency range might cover several ranges of frequencies specified by us.

# Group ID 121 and 122 contain the data set obtained at high Box pressure. There is no way to determine that the pressure in the Box was high from the parameters recorded. Thus I need to deal with these data sets manually. Other than that, there is no point to group the data by the charge exchange mass flow rate, because it was adjusted only to maximize the signal-to-noise ratio.

grouped_exp_df.drop([121, 122], inplace=True)

grouped_exp_df['Charge Exchange Mass Flow Rate [sccm]'].drop_duplicates()

# Columns by which the grouped experiments are grouped
grouping_col_list = ['Experiment Type', 'Accelerating Voltage [kV]', 'Waveguide Separation [cm]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]']

grouped_exp_grouped = grouped_exp_df.groupby(grouping_col_list)

grouped_exp_fosof_lineshape_param_df = pd.DataFrame()

for name, group_df in grouped_exp_grouped:
    print(name)
    grouped_exp_index_list = list(group_df.index)
    fosof_phase_subgroup_df = fosof_phase_grouped_df.loc[grouped_exp_index_list]

    # For each parameters group we have the dataframe that stores the FOSOF lineshape fit parameters of the grouped experiments.
    fosof_lineshape_param_df = pd.DataFrame()

    # # There are 1, 2, 3, 4, 5, 6, 7, 8 multiples of central (4 MHz) frequency range that were acquired in the experiment. We also want to look at the data for 1/16, 1/8, 1/4, 1/2 multiple of 4 MHz.
    #
    # for range_multiple in np.append(np.array([1/16, 1/8, 1/4, 1/2]), np.arange(1, 9)):

    #There are 1, 2, 3, 4, 5, 6, 7, 8 multiples of central (4 MHz) frequency range that were acquired in the experiment.
    #for range_multiple in np.arange(1, 9):
    for range_multiple in np.array([1/8, 1, 2]):
         freq_range_df = get_freq_range_data(fosof_phase_subgroup_df, range_multiple)
         # The analysis is performed if there is data that was acquired for the given frequency range multiple.
         if freq_range_df.shape[0] > 0:
            print(freq_range_df.shape[0])
            freq_range_grouped = freq_range_df.groupby(['Group ID', 'Beam RMS Radius [mm]'])

            freq_range_fosof_lineshape_data_df = freq_range_grouped.apply(get_fosof_lineshape_for_param_group)

            freq_range_fosof_lineshape_data_df.index.names = ['Group ID', 'Beam RMS Radius [mm]', 'FOSOF Lineshape Parameter']

            freq_range_fosof_lineshape_data_df['Frequency Range Multiple'] = range_multiple
            freq_range_fosof_lineshape_data_df = freq_range_fosof_lineshape_data_df.set_index('Frequency Range Multiple', append=True)
            freq_range_fosof_lineshape_data_df = freq_range_fosof_lineshape_data_df.unstack(level='FOSOF Lineshape Parameter')

            fosof_lineshape_param_df = fosof_lineshape_param_df.append(freq_range_fosof_lineshape_data_df)

    # Adding the parameters of the group to the dataframe containing the FOSOF lineshape fit parameters.

    for col in grouping_col_list:
        fosof_lineshape_param_df[col] = group_df.iloc[0][col]

    fosof_lineshape_param_df = fosof_lineshape_param_df.set_index(grouping_col_list, append=True).reorder_levels(['Experiment Type', 'Beam RMS Radius [mm]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple', 'Group ID'])

    grouped_exp_fosof_lineshape_param_df = grouped_exp_fosof_lineshape_param_df.append(fosof_lineshape_param_df)

grouped_exp_fosof_lineshape_param_df.sort_index(inplace=True)
#%%
# Save the fosof data
fosof_lineshape_param_file_name = 'fosof_lineshape_param.csv'

os.chdir(saving_folder_location)

grouped_exp_fosof_lineshape_param_df.to_csv(path_or_buf=fosof_lineshape_param_file_name, mode='w', header=True)
#%%
freq_range_df
#%%
freq_range_df.shape[0]
