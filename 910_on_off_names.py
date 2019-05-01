''' Going through the analyzed data sets info file and selecting only the ON/OFF data sets, which will later be used in the analysis of the effect of the higher-n states on the resonant frequency.
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

# Package for wrapping long string to a paragraph
import textwrap
#%%
# Location where the analyzed experiment is saved
# For Home
saving_folder_location = 'E:/2017-10-17 Lamb Shift Measurement/Data/FOSOF analyzed data sets'

# Analysis version. Needed for checking if the data set has been analyzed before.
version_number = 0.1

# File containing parameters and comments about all of the data sets.
exp_info_file_name = 'fosof_data_sets_info.csv'
exp_info_index_name = 'Experiment Folder Name'

os.chdir(saving_folder_location)
exp_info_df = pd.read_csv(filepath_or_buffer=exp_info_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True, dtype={'Error(s) During Analysis': np.bool})

exp_info_df[exp_info_index_name] = exp_info_df[exp_info_index_name].transform(lambda x: x.strip())

exp_info_df = exp_info_df.set_index(exp_info_index_name)

# Pick only fully analyzed data sets that had no errors during the analysis/acquisition.
exp_info_chosen_df = exp_info_df[exp_info_df['Data Set Fully Acquired'] & ~(exp_info_df['Error(s) During Analysis']) & (exp_info_df['Analysis Finished'])]

# After this date I was acquiring only the data sets without the atoms present. These data sets were analyzed with the different code.
max_datetime = pd.to_datetime('2018-08-31')

exp_info_chosen_df['Acquisition Start Date'] = pd.to_datetime(exp_info_chosen_df['Acquisition Start Date'])

# Selecting the data sets before the max_datetime
exp_info_chosen_df = exp_info_chosen_df[exp_info_chosen_df['Acquisition Start Date'] < max_datetime].sort_values(by='Acquisition Start Date')

exp_name_list = exp_info_chosen_df.index

exp_910_on_off_list = []

os.chdir(saving_folder_location)

for exp_folder_name in exp_name_list:

    os.chdir(saving_folder_location)
    os.chdir(exp_folder_name)
    print(exp_folder_name)
    data_set = fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=True, beam_rms_rad_to_load=None)

    if data_set.get_exp_parameters()['Experiment Type'] == 'Pre-910 Switching':
        exp_910_on_off_list.append(exp_folder_name)
        print('Pre-910 on-off switching')

    print('--------------------')
#%%
# This particular data set is removed, since as the comments say, the DC range of the digitizer was not set properly. (I am not sure why this data set was marked as valid in the first place in the info file)
exp_910_on_off_list.remove('180405-220057 - FOSOF Acquisition 910 onoff CGX pressure change - pi config, 18 V per cm PD 120V, P_CGX change')
#%%
# save the file names
os.chdir(saving_folder_location)
pd.Series(exp_910_on_off_list).to_csv(path='910_on_off_fosof_list.csv')
#%%
