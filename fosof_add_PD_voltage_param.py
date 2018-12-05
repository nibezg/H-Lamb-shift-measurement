'''
2018-11-02

While taking data we did not explicitly add the Proton deflector voltage as one of the parameters. The proton deflector voltage was, however, incorporated into the name of the file. I made a separate .csv sheet that has the Proton deflector voltage listed for the respective experiments.

This script simply goes through all of the experiments in the fosof_exp_list_PD_voltage.csv file, loads the analyzed object file, appends the PD setting and then saves the object.
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
# Location where the analyzed experiment is saved
#saving_folder_location = 'C:/Research/Lamb shift measurement/Data/FOSOF analyzed data sets'
# For Home
saving_folder_location = 'E:/2017-10-17 Lamb Shift Measurement/Data/FOSOF analyzed data sets'

exp_pd_volt_file_name = 'fosof_exp_list_PD_voltage.csv'

os.chdir(saving_folder_location)
exp_pd_volt_df = pd.read_csv(filepath_or_buffer=exp_pd_volt_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True, index_col=0)

pd_volt_col_name = 'Proton Deflector Voltage [V]'

# List of beam rms radia used for the experiment analysis used for imperfect RF power correction to the FOSOF phases.
beam_rms_rad_list = [0.8, 1.6, 2.4]

exp_folder_name_arr = exp_pd_volt_df.index.values
#%%

i = 0
for exp_folder_name in exp_folder_name_arr:

    print(str(i) + '/' + str(exp_folder_name_arr.shape[0]))
    print(exp_folder_name)
    print('=====================')

    pd_volt = exp_pd_volt_df.loc[exp_folder_name][pd_volt_col_name]

    data_set = fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=True, beam_rms_rad_to_load=None)
    data_set.get_exp_parameters()[pd_volt_col_name] = pd_volt
    data_set.save_instance(rewrite_Q=True)

    if data_set.get_exp_parameters()['Experiment Type'] == 'Waveguide Carrier Frequency Sweep':

        for beam_rms_rad in beam_rms_rad_list:

            fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)
            data_set.get_exp_parameters()[pd_volt_col_name] = pd_volt
            data_set.save_instance(rewrite_Q=True)

    else:
        print('FOSOF phase correction due to imperfect RF power is assumed to not have been applied.')

    i = i + 1

#%%
