'''
2018-11-04

One full experiment = the paired experiment is the combination of the '0' and 'pi' configuration experiments, taken one after another. Sometimes we might want to combine sets of these paired experiments to construct one grouped experiment. This is definitely needed for the case when we do not take all of the desired frequencies at once, but across several full experiments. That is why, to be general, I make it such that I should only determine the resonant frequency of the grouped experiment, not the paired experiment. These groupes are constructed in this script. Then these groups can be used to determine the zero-crossing frequencies.

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

# For home
sys.path.insert(0,"E:/Google Drive/Research/Lamb shift measurement/Code")

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
#%%
saving_folder_location = 'E:/2017-10-17 Lamb Shift Measurement/Data/FOSOF analyzed data sets'
exp_paired_with_params_file_name = 'fosof_exp_paired_with_params.csv'

os.chdir(saving_folder_location)

exp_paired_param_df = pd.read_csv(filepath_or_buffer=exp_paired_with_params_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True, index_col=0)
#%%
exp_paired_param_df
#%%
# Form the grouped experiments
group_id_col_name = 'Group ID'
exp_paired_param_df[group_id_col_name] = np.nan

# The paired experiments corresponding to the central frequency range (+- 2 MHz about 910 MHz with additional 0.1 MHz expansion in the range due to up to 100 kHz frequency jitters) form the set of grouped experiments, each containing a single paired experiment. Also these experiments have to cover at about 4 MHz range (at least 3.6 MHz. again, due to possible 0.1 kHz of an offset added to each frequency).

central_freq_range_max = 912.1
central_freq_range_min = 907.9
#min_freq_diff = (912-0.2)-(908+0.2)

#central_freq_range_index = exp_paired_param_df[(exp_paired_param_df['Minimum Frequency [MHz]'] > central_freq_range_min) &  (exp_paired_param_df['Maximum Frequency [MHz]'] < central_freq_range_max) & (exp_paired_param_df['Maximum Frequency [MHz]'] - exp_paired_param_df['Minimum Frequency [MHz]'] >= min_freq_diff)].index

central_freq_range_index = exp_paired_param_df[(exp_paired_param_df['Minimum Frequency [MHz]'] > central_freq_range_min) &  (exp_paired_param_df['Maximum Frequency [MHz]'] < central_freq_range_max)].index

group_id = 0

for exp_id in central_freq_range_index.values:
    exp_paired_param_df.loc[exp_id, (group_id_col_name)] = group_id
    group_id = group_id + 1

# There are also sets of the paired experiments that are needed to determine the resonant frequency. These are the experiments that cover the range of frequencies beyond 908-912 MHz range.
# 6 x the range (4 MHz-wide scan)
grouped_exp_index = [132, 133]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

grouped_exp_index = [134, 135]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

grouped_exp_index = [134, 135]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

grouped_exp_index = [136, 137]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

grouped_exp_index = [138, 139]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

grouped_exp_index = [140, 141, 142, 143]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

grouped_exp_index = [140, 141, 142, 143]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

grouped_exp_index = [144, 145]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

grouped_exp_index = [150, 153]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

# This is a single experiment that obtained 4 MHz range of frequencies at 6 x the range.
grouped_exp_index = [154]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

grouped_exp_index = [174, 175]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

grouped_exp_index = [177, 178]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

grouped_exp_index = [179, 180]
exp_paired_param_df.loc[grouped_exp_index, (group_id_col_name)] = group_id
group_id = group_id + 1

# There were some data sets that acquired data in 4 MHz chunks to cover, for instance, 2 x the central range of 4 MHz = 8 MHz. These data sets are a bit problematic, because eventually we do not want to look at the resonant frequency due to whole 8 MHz range, but only due to a subset of this data that does not belong to the central range and covers 2 MHz of the range symmetrically on each side about 910 MHz to give 4 MHz total range. This way we can test if there are any problems with the larger range data = some  in the lineshapes, for instance.

# 2 x the range (8 MHz-wide scan)
to_adjust_index_list = [63, 64, 65]
exp_paired_param_df.loc[to_adjust_index_list, (group_id_col_name)] = group_id
group_id = group_id + 1

to_adjust_index_list = [66, 67]
exp_paired_param_df.loc[to_adjust_index_list, (group_id_col_name)] = group_id
group_id = group_id + 1

to_adjust_index_list = [68, 69, 70]
exp_paired_param_df.loc[to_adjust_index_list, (group_id_col_name)] = group_id
group_id = group_id + 1

to_adjust_index_list = [71, 72]
exp_paired_param_df.loc[to_adjust_index_list, (group_id_col_name)] = group_id
group_id = group_id + 1

to_adjust_index_list = [73, 74]
exp_paired_param_df.loc[to_adjust_index_list, (group_id_col_name)] = group_id
group_id = group_id + 1

# 4 x the range (16 MHz-wide scan)
to_adjust_index_list = [93, 94, 95, 96]
exp_paired_param_df.loc[to_adjust_index_list, (group_id_col_name)] = group_id
group_id = group_id + 1

to_adjust_index_list = [97, 98, 99, 100]
exp_paired_param_df.loc[to_adjust_index_list, (group_id_col_name)] = group_id
group_id = group_id + 1

# 6 x the range (24 MHz-wide scan)
to_adjust_index_list = [101, 102, 103, 104, 105, 106]
exp_paired_param_df.loc[to_adjust_index_list, (group_id_col_name)] = group_id
group_id = group_id + 1

# 8 x the range (32 MHz-wide scan)
to_adjust_index_list = [109, 112, 113, 114, 115, 116, 117, 118]
exp_paired_param_df.loc[to_adjust_index_list, (group_id_col_name)] = group_id
group_id = group_id + 1

# Convert the grouping column type to integer
exp_paired_param_df[group_id_col_name] = exp_paired_param_df[group_id_col_name].astype(np.int16)

exp_paired_param_df = exp_paired_param_df.reset_index().set_index([group_id_col_name, 'Experiment ID'])
#%%
# We now check whether all of the paired experiments for each grouped experiment have consistent=same parameters. In addition for each grouped experiment I load the fosof phases for each paired experiment in the group and calculate 0-pi phase difference. This is done for the range of beam rms radii.
# IMPORTANT
# Notice that the fosof phases need to be divided by 4 for the final lineshape. Factor of 2 is due to not dividing by 2 during RF Channel A and B phase averaging. And another factor of 2 is by having to divide 0-pi by this factor.

# The columns used for the comparison
col_to_compare_list = ['Accelerating Voltage [kV]', 'B_x Max [Gauss]', 'B_y Max [Gauss]', 'B_x Min [Gauss]', 'B_y Min [Gauss]', 'Experiment Type', 'Number of B_x Steps', 'Number of B_y Steps', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Waveguide Separation [cm]', 'Charge Exchange Mass Flow Rate [sccm]']

# Dataframe containing parameters for the group
grouped_exp_df = pd.DataFrame()

# Datafram containing the FOSOF phases for the group
fosof_phase_grouped_df = pd.DataFrame()

for group_id, df in exp_paired_param_df.groupby(group_id_col_name):
    # This check is needed only for the groups that contain more than 1 paired experiment
    if df.shape[0] > 1:
        for col_name in col_to_compare_list:
            if df[col_name].drop_duplicates().shape[0] > 1:
                raise FosofAnalysisError("The grouped experiment #" + str(group_id) + ' has non-duplicate data for the parameter ' + "'"+ col_name + "'.")

    grouped_exp_df = grouped_exp_df.append(df.reset_index('Experiment ID', drop=True)[col_to_compare_list])

    # We now calculate the 0-pi phase difference = fosof phase
    fosof_phase_exp_set_df = pd.DataFrame()

    for exp_id, data_df in df.groupby('Experiment ID'):

        data_s = data_df.iloc[0]

        fosof_phase_set_df = pd.DataFrame()

        # The beam rms radii are used only for the 'Waveguide Carrier Frequency Sweep' experiment type
        if data_s['Experiment Type'] == 'Waveguide Carrier Frequency Sweep':
            beam_rms_rad_list = [None, 0.8, 1.6, 2.4]
        else:
            beam_rms_rad_list = [None]

        for beam_rms_rad in beam_rms_rad_list:
            data_set_0 = fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=data_s['Experiment Name (0-config)'], load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

            data_set_pi = fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=data_s['Experiment Name (pi-config)'], load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

            fosof_ampl_0_df, fosof_phase_0_df = data_set_0.average_FOSOF_over_repeats()
            fosof_ampl_pi_df, fosof_phase_pi_df = data_set_pi.average_FOSOF_over_repeats()

            # Calculate fosof phases + their uncertainties
            fosof_phase_df = (fosof_phase_0_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')] - fosof_phase_pi_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')]).join(np.sqrt(fosof_phase_0_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted STD')]**2 + fosof_phase_pi_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted STD')]**2)).sort_index(axis='columns')

            # Convert the phases to the 0-2pi range
            fosof_phase_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')] = fosof_phase_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')].transform(convert_phase_to_2pi_range)

            index_name_list = list(fosof_phase_df.index.names)
            index_name_list.insert(0, 'Beam RMS Radius [mm]')

            if beam_rms_rad is None:
                beam_rms_rad = -1

            fosof_phase_df['Beam RMS Radius [mm]'] = beam_rms_rad
            fosof_phase_df = fosof_phase_df.reset_index().set_index(index_name_list)

            fosof_phase_set_df = fosof_phase_set_df.append(fosof_phase_df)

        index_name_list = list(fosof_phase_set_df.index.names)
        index_name_list.insert(1, 'Experiment ID')
        fosof_phase_set_df['Experiment ID'] = exp_id
        fosof_phase_set_df = fosof_phase_set_df.reset_index().set_index(index_name_list)

        fosof_phase_exp_set_df = fosof_phase_exp_set_df.append(fosof_phase_set_df)

    index_name_list = list(fosof_phase_exp_set_df.index.names)
    index_name_list.insert(0, group_id_col_name)
    fosof_phase_exp_set_df[group_id_col_name] = group_id
    fosof_phase_exp_set_df = fosof_phase_exp_set_df.reset_index().set_index(index_name_list)

    fosof_phase_grouped_df = fosof_phase_grouped_df.append(fosof_phase_exp_set_df)
#%%
# Save the fosof data
fosof_phase_data_file_name = 'fosof_phase_grouped_list.csv'
grouped_exp_param_file_name = 'grouped_exp_param_list.csv'

os.chdir(saving_folder_location)

fosof_phase_grouped_df.to_csv(path_or_buf=fosof_phase_data_file_name, mode='w', header=True)
grouped_exp_df.to_csv(path_or_buf=grouped_exp_param_file_name, mode='w', header=True)

#%%
fosof_phase_grouped_df
#%%
exp_paired_param_df.index.get_level_values('Experiment ID')
#%%
exp_paired_param_df.loc[(slice(None), 172), (slice(None))]

#%%
data_set_0 = fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=exp_paired_param_df.loc[(slice(None), 171), (slice(None))]['Experiment Name (0-config)'].values[0], load_Q=True, beam_rms_rad_to_load=None)
#%%
data_set_0.get_exp_parameters()
#%%
data_set_0.get_exp_parameters()
#%%
exp_paired_param_df.loc[123]['Experiment Name (0-config)'].values
