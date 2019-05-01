'''
2019-03-10

I am trying to understand the reason for getting different results between my calculation of the resonant frequency and the one that Travis calculated.

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
# For the lab
sys.path.insert(0,"'C:/Users/Helium1/Google Drive/Research/Lamb shift measurement/Code'")

from exp_data_analysis import *
from beam_speed import *
import fosof_data_set_analysis
import hydrogen_sim_data


import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate
import scipy.optimize

import matplotlib.pyplot as plt


import threading
from queue import Queue

# Package for wrapping long string to a paragraph
import textwrap

from tkinter import *
from tkinter import ttk

from tkinter import messagebox

from uncertainties import ufloat
#import wvg_power_calib_analysis
#%%
# Blind offset in MHz
BLIND_OFFSET = 0.03174024731
#%%
# Load the zero-crossing frequencies
saving_folder_location = r'E:\Google Drive\Research\Lamb shift measurement\Data\FOSOF analyzed data sets'

fosof_lineshape_param_file_name = 'fosof_lineshape_param.csv'

fosof_grouped_phase_file_name = 'fosof_phase_grouped_list.csv'

os.chdir(saving_folder_location)
#%%
grouped_exp_fosof_lineshape_param_df = pd.read_csv(filepath_or_buffer=fosof_lineshape_param_file_name, delimiter=',', comment='#', header=[0, 1, 2, 3, 4], skip_blank_lines=True, index_col=[0, 1, 2, 3, 4, 5, 6, 7])
#%%
grouped_phase = pd.read_csv(filepath_or_buffer=fosof_grouped_phase_file_name, delimiter=',', comment='#', skip_blank_lines=True, header=[0, 1, 2, 3, 4], index_col=[0, 1, 2, 3])
#%%
grouped_phase
#%%
exp_id_df = grouped_phase.loc[(slice(None), -1), slice(None)].reset_index(['Experiment ID', 'Waveguide Carrier Frequency [MHz]', 'Beam RMS Radius [mm]'])[['Experiment ID']].drop_duplicates()
#%%
exp_id_df.loc[[94]]
#%%
val_df = grouped_exp_fosof_lineshape_param_df.loc[('Waveguide Carrier Frequency Sweep', 1.6, 4, 16.27, slice(None), 14, 1), slice(None)]['RF Combiner R Reference', 'First Harmonic', 'Phase Averaging', 'Phase RMS Repeat STD']
val_df
#%%
np.average(a=val_df['Zero-crossing Frequency [MHz]'].values, weights=val_df['Zero-crossing Frequency STD [MHz]'].values)
#%%
np.mean([909.8374103, 909.8407717])
#%%
909.8374103 - 909.8361333199504
#%%
909.8407717 - 909.8394466900108
#%%
grouped_exp_fosof_lineshape_param_df.loc[('Waveguide Carrier Frequency Sweep', -1, 4, 16.27, slice(None), slice(None), 1), ('RF Combiner I Reference', 'First Harmonic', 'Phase Averaging', slice(None), ['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD [MHz]', 'Reduced Chi-Squared'])]
#%%
travis_data_path = r'E:\Google Drive\Research\Lamb shift measurement\Data\FOSOF analyzed data sets\Travis_data_and_code'

raw_data_file_name = 'data_TV_112305.csv'

os.chdir(travis_data_path)

raw_data_df = pd.read_csv(filepath_or_buffer=raw_data_file_name, delimiter=',', comment='#', skip_blank_lines=True, index_col=[0, 1, 2, 3, 4])

raw_data_df['Time'] = raw_data_df['Time'] - raw_data_df['Time'].min()

raw_data_df = raw_data_df.reset_index().set_index(['Repeat', 'Waveguide Carrier Frequency [MHz]', 'Sub-Configuration', 'Average']).sort_index()
#%%

freq_to_use = raw_data_df.index.get_level_values('Waveguide Carrier Frequency [MHz]')[0]

df_to_use = raw_data_df.loc[(slice(None), freq_to_use), (['Phase Difference (Detector - Mixer 1) [rad]'])]
df_to_use
#%%
r1_a = df_to_use.loc[(1, slice(None), 'A')].values
r1_a[0] = r1_a[0] - 2 * np.pi
r1_a[1] = r1_a[1] - 2 * np.pi
r1_a[2] = r1_a[2] - 2 * np.pi

r1_b = df_to_use.loc[(1, slice(None), 'B')].values

r2_a = df_to_use.loc[(2, slice(None), 'A')].values
r2_a[0] = r2_a[0] - 2 * np.pi
r2_a[1] = r2_a[1] - 2 * np.pi
r2_a[3] = r2_a[3] - 2 * np.pi
r2_b = df_to_use.loc[(2, slice(None), 'B')].values
#%%
r1_a
#%%
r2_a
#%%
r1_a = np.array([6.228298-2*np.pi, 6.189607-2*np.pi, 6.268252-2*np.pi, 0.006640])
r1_b = np.array([0.519081, 0.575405, 0.440383, 0.480988])
r2_a = np.array([6.236370-2*np.pi, 6.242705-2*np.pi, 0.004205, 6.258861-2*np.pi])
r2_b = np.array([0.557991, 0.482051, 0.591937, 0.486569])
#%%
np.average(a=[np.mean(r1_a-r1_b),np.mean(r2_a-r2_b)], weights=[1/(np.std((r1_a-r1_b),ddof=1)/np.sqrt(4))**2, 1/(np.std((r2_a-r2_b),ddof=1)/np.sqrt(4))**2])/2
#%%
np.mean([np.mean(r1_a-r1_b), np.mean(r2_a-r2_b)])/2
#%%
np.mean([np.mean(r1_a) - np.mean(r1_b), np.mean(r2_a) - np.mean(r2_b)])/2
#%%
phase_data_file_name = 'phasedata_112305.csv'
phase_data_df =  pd.read_csv(filepath_or_buffer=phase_data_file_name, delimiter=',', comment='#', skip_blank_lines=True, index_col=[0, 1])
#%%
phase_data_df
#%%
phase_data_df.loc[('Unique Uncertainty'), slice(None)]
#%%
exp_folder_name = '180528-112305 - FOSOF Acquisition - 0 config, 14 V per cm PD ON 37.7 V, 16.27 kV'

beam_rms_rad = None
# List of beam  rms radius values for correcting the FOSOF phases using the simulations. Value of None corresponds to not applying the correction.
data_set = fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=False, beam_rms_rad_to_load=beam_rms_rad)

# The power correction is performed only for the simple FOSOF data sets.
# if data_set.get_exp_parameters()['Experiment Type'] != 'Waveguide Carrier Frequency Sweep':
#     beam_rms_rad = None
#     data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

data_set.select_portion_of_data()
#%%
fc_df = data_set.get_fc_data()
quenching_df = data_set.get_quenching_cav_data()
rf_pow_df, rf_system_power_outlier_df = data_set.get_rf_sys_pwr_det_data()

digi_df = data_set.get_digitizers_data()

comb_phase_diff_df = data_set.get_combiners_phase_diff_data()
digi_delay_df = data_set.get_inter_digi_delay_data()

if beam_rms_rad is not None:
    data_set.correct_phase_diff_for_RF_power(beam_rms_rad)

phase_diff_df = data_set.get_phase_diff_data()
phase_av_set_averaged_df = data_set.average_av_sets()
phase_A_minus_B_df, phase_freq_response_df = data_set.cancel_out_freq_response()
fosof_ampl_df, fosof_phase_df = data_set.average_FOSOF_over_repeats()
#%%
#%%
np.mean([0.054887, 0.093578, 0.014934, 6.276546-2*np.pi]) + 2*np.pi
np.std([0.054887, 0.093578, 0.014934, 6.276546-2*np.pi], ddof=1)
#%%
np.sqrt(np.sum(([0.054887, 0.093578, 0.014934, 6.276546-2*np.pi] - np.mean([0.054887, 0.093578, 0.014934, 6.276546-2*np.pi]))**2)/(3))
#%%
np.mean([5.764104, 5.707780, 5.842802, 5.802197])
np.std([5.764104, 5.707780, 5.842802, 5.802197], ddof=1)
#%%
np.sqrt((np.std([5.764104, 5.707780, 5.842802, 5.802197])**2*4 + np.std([0.054887, 0.093578, 0.014934, 6.276546-2*np.pi])**2*4)/6)/2
#%%
#%%
phase_A_minus_B_df['RF Combiner I Reference', 'First Harmonic', 'Phase Averaging']/2
