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
import matplotlib.pyplot as plt
import math

import threading
from Queue import Queue

# Package for wrapping long string to a paragraph
import textwrap

from Tkinter import *
import ttk
import tkMessageBox

#%%
# Folder containing binary traces in .npy format
binary_traces_folder = "//LAMBSHIFT-PC/Users/Public/Documents/binary traces"
experiment_folder = '180608-091743 - FOSOF Acquisition - 0 config, 18 V per cm PD ON 120 V, 49.86 kV, 909.8-910.2 MHz'

os.chdir(binary_traces_folder)
os.chdir(experiment_folder)

#%%
digi_ch_range = 4
sampling_rate = 10000
offset_freq = 800
digi_trace_file_name = 'r001a001f0909.9076chA_02.digi.npy'

# Import the trace and get its parameters (from the file name)
trace_params, trace_V_array = import_digi_trace(digi_trace_file_name, digi_ch_range)
n_samples = trace_V_array.size # Number of samples in the trace
#%%

n_period = 100
t_duration = 1/offset_freq*n_period
dt = 1/sampling_rate
n_samples = t_duration/dt

if n_samples != int(n_samples):
    raise Exception
n_samples = int(n_samples)

#%%
# Split the array into chunks of n_samples in each
trace_V_split_array = np.reshape(trace_V_array, (int(trace_V_array.shape[0]/n_samples), n_samples))

# Dataframe for storing the information about the Fourier harmonic(s) for the trace chunks
f_harmonic_collection_df = pd.DataFrame([])

# Find Fourier harmonic at the frequencies of interest for all trace chunks
for trace_chunk_index in range(trace_V_split_array.shape[0]):
    trace_spectrum_data = get_Fourier_spectrum(trace_V_split_array[trace_chunk_index], sampling_rate)
    f_harmonic_df = get_Fourier_harmonic(trace_spectrum_data, [offset_freq], ac_harm=False)
    f_harmonic_df.index = pd.Index([trace_chunk_index])
    f_harmonic_collection_df = f_harmonic_collection_df.append(f_harmonic_df)

#%%
f_harmonic_collection_df
