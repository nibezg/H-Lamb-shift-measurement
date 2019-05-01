'''
2018-11-03

I manually constructed the list of '0' and 'pi'-configuratiob paired data sets. I want to be able to check whether the pairing is correct, by comparing various parameters of the two experiments for each paired experiment. After that the important parameters for the paired experiments are recorded in a separate .csv file

One full experiment = the paired experiment is the combination of the '0' and 'pi' configuration experiments, taken one after another. This combination can be used to cancel out (in theory) all of the undesired phase shifts, leaving only the atomic phase.

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

#%%
saving_folder_location = fosof_analyzed_data_folder_path

exp_paired_file_name = 'fosof_exp_paired.xlsx'

exp_paired_with_params_file_name = 'fosof_exp_paired_with_params.csv'

os.chdir(saving_folder_location)

exp_paired_xlsx = pd.ExcelFile(exp_paired_file_name)
exp_paired_df = pd.read_excel(pd.ExcelFile(exp_paired_xlsx), sheet_name='Sheet1', header=[0], index_col=[0])
#%%
exp_pair_id_arr = exp_paired_df.index.values

exp_pair_id_arr = exp_pair_id_arr

exp_paired_param_df = pd.DataFrame()

for exp_par_id in exp_pair_id_arr:

    print('Experiment ID: ' + str(exp_par_id))

    exp_folder_name_1 = exp_paired_df.loc[exp_par_id]['Experiment Folder Name 1']
    exp_folder_name_2 = exp_paired_df.loc[exp_par_id]['Experiment Folder Name 2']

    data_set_1 = fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=exp_folder_name_1, load_Q=True, beam_rms_rad_to_load=None)
    data_set_2 = fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=exp_folder_name_2, load_Q=True, beam_rms_rad_to_load=None)

    exp_param_1_s = data_set_1.get_exp_parameters().copy()
    exp_param_2_s = data_set_2.get_exp_parameters().copy()

    # We first check whether the two experiments are of 0- and pi-configuration type.
    if set([exp_param_1_s['Configuration'], exp_param_2_s['Configuration']]) != set(['0', 'pi']):
        raise FosofAnalysisError("Not both of '0'- and 'pi' configurations are present")

    # Determine which of the experiments is '0' and 'pi' configuration respectively

    exp_object_config_dict = {}

    exp_pair_config_dict = {}
    if exp_param_1_s['Configuration'] == '0':
        exp_pair_config_dict['0'] = exp_folder_name_1
        exp_pair_config_dict['pi'] = exp_folder_name_2

        exp_object_config_dict['0'] = data_set_1
        exp_object_config_dict['pi'] = data_set_2
    else:
        exp_pair_config_dict['0'] = exp_folder_name_2
        exp_pair_config_dict['pi'] = exp_folder_name_1

        exp_object_config_dict['0'] = data_set_2
        exp_object_config_dict['pi'] = data_set_1

    # Now we compare if the data sets have exactly the same RF frequencies acquired

    fosof_ampl_1_df, fosof_phase_1_df = data_set_1.average_FOSOF_over_repeats()
    fosof_ampl_2_df, fosof_phase_2_df = data_set_2.average_FOSOF_over_repeats()

    freq_1_index = fosof_phase_1_df.index.get_level_values('Waveguide Carrier Frequency [MHz]').drop_duplicates()
    freq_2_index = fosof_phase_2_df.index.get_level_values('Waveguide Carrier Frequency [MHz]').drop_duplicates()

    if not(freq_1_index.equals(freq_2_index)):
        raise FosofAnalysisError("The data sets have different frequencies acquired")

    # Check if the number of frequencies scanned matches the number specified in the parameters. This is done because there are a couple of data sets that had frequencies picked not from the np.linspace() function, but manually. I want to make sure that they had correct number of frequency steps specified in their parameters


    if exp_param_1_s['Number of Frequency Steps'] != freq_1_index.shape[0]:
        raise FosofAnalysisError("The data sets have different number of frequencies acquired than the specified number in the run parameters: " + str(exp_param_1_s['Number of Frequency Steps']) + ' vs ' + str(freq_1_index.shape[0]))


    # We want now to check if the data sets had reasonably small amount of time between the acquisitions. In other words, between making the waveguide reversals I do not want to have hours passed, but something within a half an hour, for instance. This check is especially important for making sure that we did not pair data sets acquired on different days.

    if exp_param_1_s['Experiment Start Time [s]'] < exp_param_2_s['Experiment Start Time [s]']:

        t_short_dur = exp_param_1_s['Experiment Duration [s]']

        delta_t = exp_param_2_s['Experiment Start Time [s]'] - (exp_param_1_s['Experiment Start Time [s]'] + t_short_dur)

    else:

        t_short_dur = exp_param_2_s['Experiment Duration [s]']

        delta_t = exp_param_1_s['Experiment Start Time [s]'] - (exp_param_2_s['Experiment Start Time [s]'] + t_short_dur)

    if not((delta_t < 0.5 * t_short_dur) or (delta_t < 60 * 60)):
        raise FosofAnalysisError("The time difference between finishing acquiring the first data set and starting to acquire the second data set is too large: " + str(delta_t/60) + ' min.')

    # Now it is important to make sure that the following parameters are identical:

    param_to_match_list = ['Waveguide Separation [cm]', 'Proton Deflector Voltage [V]', 'Accelerating Voltage [kV]', 'Experiment Type', 'Offset Frequency [Hz]', 'Waveguide Electric Field [V/cm]', 'Charge Exchange Mass Flow Rate [sccm]', 'Number Of Frequency Steps', 'B_x Min [Gauss]', 'B_x Max [Gauss]', 'Number of B_x Steps', 'B_y Min [Gauss]', 'B_y Max [Gauss]', 'Number of B_y Steps']

    # Some data sets do not have all of the parameters listed above available. We need to take care of that
    param_to_use_1_set = set(param_to_match_list) - set(list(exp_param_1_s.index.values))

    param_to_use_2_set = set(param_to_match_list) - set(list(exp_param_2_s.index.values))

    if param_to_use_1_set != param_to_use_2_set:
        raise FosofAnalysisError("The data sets do not have the same subsets of available parameters of interest")

    param_to_use_list = list(set(param_to_match_list) - param_to_use_1_set)

    for key in param_to_use_list:

        if exp_param_1_s[key] != exp_param_2_s[key]:
            raise FosofAnalysisError("The experiments have different values for the parameters" + key)

    exp_type = exp_param_1_s['Experiment Type']

    # We can now construct the dataframe

    exp_param_to_record_s = exp_param_1_s[param_to_use_list]

    exp_param_to_record_s['Minimum Frequency [MHz]'] = freq_1_index.min()
    exp_param_to_record_s['Maximum Frequency [MHz]'] = freq_1_index.max()

    exp_param_to_record_s['Experiment Name (0-config)'] = exp_pair_config_dict['0']
    exp_param_to_record_s['Experiment Name (pi-config)'] = exp_pair_config_dict['pi']

    exp_param_to_record_s['Notes (0-config)'] = exp_object_config_dict['0'].get_exp_parameters()['Notes']

    exp_param_to_record_s['Notes (pi-config)'] = exp_object_config_dict['pi'].get_exp_parameters()['Notes']

    exp_param_to_record_s['Comments (0-config)'] = exp_object_config_dict['0'].get_exp_parameters()['Comments']

    exp_param_to_record_s['Comments (pi-config)'] = exp_object_config_dict['pi'].get_exp_parameters()['Comments']

    exp_param_to_record_s.name = exp_par_id

    exp_paired_param_df = exp_paired_param_df.append(exp_param_to_record_s)

    print('==========================')

exp_paired_param_df.index.names = exp_paired_df.index.names

os.chdir(saving_folder_location)

exp_paired_param_df.to_csv(path_or_buf=exp_paired_with_params_file_name, mode='w', header=True)
