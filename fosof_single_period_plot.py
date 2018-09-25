from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

sys.path.insert(0,"C:/Users/Helium1/Google Drive/Code/Python/Testing/Blah") #
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


import threading
from Queue import Queue

# Package for wrapping long string to a paragraph
import textwrap

from Tkinter import *
import ttk
import tkMessageBox

experiment_folder = '180321-150600 - FOSOF Acquisition - 0 config, 18 V per cm PD ON 120 V'
#%%
# Folder containing binary traces in .npy format
binary_traces_folder = "//LAMBSHIFT-PC/Users/Public/Documents/binary traces"
# Folder containing acquired data table
data_folder = "//LAMBSHIFT-PC/Google Drive/data"
# Experiment data file name
data_file = 'data.txt'

# Prepare for the data analysis. This includes getting the experiment parameters and checking if the analysis has been started/finished before.

os.chdir(data_folder)
os.chdir(experiment_folder)

# Get header names from the data file
exp_data = pd.read_csv(filepath_or_buffer=data_file, delimiter=',', comment='#', header=0, skip_blank_lines=True, skipinitialspace=True)
exp_column_names = exp_data.columns.values

# Get the experiment parameters from the data file
exp_params_dict, comment_string_arr = get_experiment_params(data_file)

# Important experiment parameters
n_Bx_steps = exp_params_dict['Number of B_x Steps']
n_By_steps = exp_params_dict['Number of B_y Steps']
n_averages = exp_params_dict['Number of Averages']
sampling_rate = exp_params_dict['Digitizer Sampling Rate [S/s]']
n_digi_samples = exp_params_dict['Number of Digitizer Samples']
n_freq_steps = exp_params_dict['Number of Frequency Steps']
n_repeats = exp_params_dict['Number of Repeats']
digi_ch_range = exp_params_dict['Digitizer Channel Range [V]']
offset_freq_array = exp_params_dict['Offset Frequency [Hz]']
pre_910_on_n_digi_samples = exp_params_dict['Pre-Quench 910 On Number of Digitizer Samples']
gain = exp_params_dict['Current Amplifier Gain (log10)']

exp_params_dict
#%%

#%%
def get_av_trace_data(trace_name, ch_label, n_periods, n_periods_to_remove):

    digi_trace_file_name =trace_name + '_0' + str(ch_label) + '.digi.npy'

    os.chdir(binary_traces_folder)
    os.chdir(experiment_folder)

    # Import the trace and get its parameters (from the file name)
    trace_params, trace_V_array = import_digi_trace(digi_trace_file_name, digi_ch_range)

    n_samples = trace_V_array.size # Number of samples in the trace

    # Get the offset frequency. If there is only one offset frequency in the experiment parameters, then this offset frequency is used for all traces.
    if len(offset_freq_array) > 1:
        offset_freq = trace_params['Offset frequency [Hz]']
    else:
        offset_freq = offset_freq_array[0]
    if trace_params['Pre-910 state [On/Off]'] == 'on':
        n_samples_expected = pre_910_on_n_digi_samples
    else:
        n_samples_expected = n_digi_samples

    if n_samples != n_samples_expected:
        # Close the handle to the analyzed data file
        data_analyzed_file_object.close()

        raise FosofAnalysisError("Number of samples in the digitizer trace does not match the expected number of samples from the experiment parameters")

    f0 = float(sampling_rate)/n_samples # Minimum Fourier frequency [Hz]

    # Find FFT of the trace and extract DC, SNR, phase, amplitude, average noise at the first two harmonics of the offset frequency
    trace_spectrum_data = get_Fourier_spectrum(trace_V_array, sampling_rate)
    dc = trace_spectrum_data.iloc[0].loc['Fourier Amplitude [Arb]']

    # Find the Fourier parameters at the offset frequency and its harmonic. Make sure to keep the order the same - offset frequency is first, second harmonic is the second. This is important for the following combination of the results.
    fourier_params = get_Fourier_harmonic(trace_spectrum_data, [offset_freq, 2*offset_freq], trace_length=n_samples/float(sampling_rate))

    # Combine the Fourier parameters in a way that can be combined with the rest of the experiment data for the given row.
    first_harmonic_fourier_params = fourier_params.iloc[0]

    samples_per_period = int(sampling_rate/offset_freq)
    if samples_per_period != sampling_rate/offset_freq:
        print('Not integer number of cycles per period')

    n_samples_to_use = n_samples - n_periods_to_remove*samples_per_period

    trace_V_array_to_use = trace_V_array[:n_samples_to_use]

    tot_n_periods = n_samples_to_use/samples_per_period/n_periods

    trace_V_per_period_arr = trace_V_array_to_use.reshape(int(tot_n_periods), n_periods*samples_per_period)

    trace_V_av_arr = np.mean(trace_V_per_period_arr, axis=0)
    trace_V_av_std_arr = np.std(trace_V_per_period_arr, axis=0, ddof=1)/np.sqrt(trace_V_av_arr.shape[0])

    t_arr = np.linspace(0, n_periods*samples_per_period/sampling_rate, trace_V_av_arr.shape[0])

    return t_arr, trace_V_av_arr, trace_V_av_std_arr, first_harmonic_fourier_params
#%%
n_periods = 2
n_periods_to_remove = 1
offset_freq = offset_freq_array[0]
samples_per_period = int(sampling_rate/offset_freq)
n_samples_to_use = n_digi_samples - n_periods_to_remove*samples_per_period
tot_n_periods = n_samples_to_use/samples_per_period/n_periods
print(samples_per_period)
print(n_samples_to_use)
print(tot_n_periods)

if tot_n_periods != int(tot_n_periods):
    print('Wrong number of periods. Pick a different number of periods or reduce the size of the array.')
else:
    print('Combination is good')
#%%
# Phase shift of the combiners' signals that is needed to be shown
phi_needed = -np.pi/2

#-----------------------
# RF CHANNEL A
#-----------------------
trace_name = 'r002a002f0911.2435chA'
ch_label = 1
t_arr, x_det_arr_A, x_det_std_arr_A, det_harm_arr_A = get_av_trace_data(trace_name, ch_label, n_periods, n_periods_to_remove)

x_det_arr_A = x_det_arr_A * 10**(-gain) * 10**(9)
x_det_std_arr_A = x_det_std_arr_A * 10**(-gain) * 10**(9)
det_harm_arr_A['Fourier Amplitude [Arb]'] = det_harm_arr_A['Fourier Amplitude [Arb]'] * 10**(-gain) * 10**(9)

ch_label = 4
t_arr, x_comb_arr_A, x_comb_std_arr_A, comb_harm_arr_A = get_av_trace_data(trace_name, ch_label, n_periods, n_periods_to_remove)

dc_det_A = np.mean(x_det_arr_A)
x_det_cos_arr_A = det_harm_arr_A['Fourier Amplitude [Arb]'] * np.cos(2*np.pi*det_harm_arr_A['Fourier Frequency [Hz]']*t_arr+det_harm_arr_A['Fourier Phase [Rad]'])

x_det_sin_arr_A = det_harm_arr_A['Fourier Amplitude [Arb]'] * np.sin(2*np.pi*det_harm_arr_A['Fourier Frequency [Hz]']*t_arr+det_harm_arr_A['Fourier Phase [Rad]'])

delta_phi_A = comb_harm_arr_A['Fourier Phase [Rad]'] - phi_needed

x_det_shifted_cos_arr_A = np.abs(dc_det_A) + x_det_cos_arr_A * np.cos(delta_phi_A) + x_det_sin_arr_A * np.sin(delta_phi_A)

x_det_shifted_arr_A = np.abs(dc_det_A) + (x_det_arr_A-dc_det_A) * np.cos(delta_phi_A) + x_det_sin_arr_A * np.sin(delta_phi_A)

x_comb_cos_arr_A = comb_harm_arr_A['Fourier Amplitude [Arb]'] * np.cos(2*np.pi*comb_harm_arr_A['Fourier Frequency [Hz]']*t_arr+(comb_harm_arr_A['Fourier Phase [Rad]']-delta_phi_A))

#-----------------------
# RF CHANNEL B
#-----------------------

trace_name = 'r002a002f0911.2435chB'
ch_label = 1
t_arr, x_det_arr_B, x_det_std_arr_B, det_harm_arr_B = get_av_trace_data(trace_name, ch_label, n_periods, n_periods_to_remove)

x_det_arr_B = x_det_arr_B * 10**(-gain) * 10**(9)
x_det_std_arr_B = x_det_std_arr_B * 10**(-gain) * 10**(9)
det_harm_arr_B['Fourier Amplitude [Arb]'] = det_harm_arr_B['Fourier Amplitude [Arb]'] * 10**(-gain) * 10**(9)

ch_label = 4
t_arr, x_comb_arr_B, x_comb_std_arr_B, comb_harm_arr_B = get_av_trace_data(trace_name, ch_label, n_periods, n_periods_to_remove)

dc_det_B = np.mean(x_det_arr_B)
x_det_cos_arr_B = det_harm_arr_B['Fourier Amplitude [Arb]'] * np.cos(2*np.pi*det_harm_arr_B['Fourier Frequency [Hz]']*t_arr+det_harm_arr_B['Fourier Phase [Rad]'])

x_det_sin_arr_B = det_harm_arr_B['Fourier Amplitude [Arb]'] * np.sin(2*np.pi*det_harm_arr_B['Fourier Frequency [Hz]']*t_arr+det_harm_arr_B['Fourier Phase [Rad]'])

delta_phi_B = comb_harm_arr_B['Fourier Phase [Rad]'] - phi_needed

x_det_shifted_cos_arr_B = np.abs(dc_det_B) + x_det_cos_arr_B * np.cos(delta_phi_B) + x_det_sin_arr_B * np.sin(delta_phi_B)

x_det_shifted_arr_B = np.abs(dc_det_B) + (x_det_arr_B-dc_det_B) * np.cos(delta_phi_B) + x_det_sin_arr_B * np.sin(delta_phi_B)

x_comb_cos_arr_B = comb_harm_arr_B['Fourier Amplitude [Arb]'] * np.cos(2*np.pi*comb_harm_arr_B['Fourier Frequency [Hz]']*t_arr+(comb_harm_arr_B['Fourier Phase [Rad]']-delta_phi_B))

t_arr = t_arr * 1E3

t_comb_A_cross = (np.pi*(1+1/2) - phi_needed)/(2*np.pi*comb_harm_arr_A['Fourier Frequency [Hz]']) * 1E3

t_det_A_cross = (np.pi*(1+1/2) - (det_harm_arr_A['Fourier Phase [Rad]'] - delta_phi_A))/(2*np.pi*det_harm_arr_A['Fourier Frequency [Hz]']) * 1E3

t_comb_B_cross = (np.pi*(1+1/2) - phi_needed)/(2*np.pi*comb_harm_arr_B['Fourier Frequency [Hz]']) * 1E3

t_det_B_cross = (np.pi*(1+1/2) - (det_harm_arr_B['Fourier Phase [Rad]'] - delta_phi_B))/(2*np.pi*det_harm_arr_B['Fourier Frequency [Hz]']) * 1E3
#%%
from matplotlib.ticker import MaxNLocator

y_range_mult = 1.2
label_font_size = 20*1.2

fig, axes = plt.subplots(nrows=1, ncols=2)

fig.set_size_inches(16, 6)

fract_points_plot = 8
axes[0].errorbar(x=t_arr[::fract_points_plot], y=x_det_shifted_arr_A[::fract_points_plot], marker='.', yerr=x_det_std_arr_A[::fract_points_plot], linestyle='None', color='red', label='"A" Configuration')


axes[0].plot(t_arr, x_det_shifted_cos_arr_A, color='red', linewidth=1, linestyle='dashed')
axes[0].set_xlim(np.min(t_arr), np.max(t_arr))
axes[0].set_ylim(np.abs(dc_det_A) - y_range_mult*det_harm_arr_A['Fourier Amplitude [Arb]'], np.abs(dc_det_A) + y_range_mult*det_harm_arr_A['Fourier Amplitude [Arb]'])
#axes[0].grid()
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Detector current (nA)', color='red')
axes[0].tick_params(axis='y', colors='red')
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[0].axvline(x=t_det_A_cross, color='green', linewidth=0.5)
axes[0].axvline(x=t_comb_A_cross, color='green', linewidth=0.5)

for item in ([axes[0].xaxis.label, axes[0].yaxis.label] +
             axes[0].get_xticklabels() + axes[0].get_yticklabels()):
    item.set_fontsize(label_font_size)

ax_0 = axes[0].twinx()
ax_0.plot(t_arr, x_comb_cos_arr_A, color='purple')
ax_0.set_ylabel('Combiner signal (Arb. Units)', color='purple')
ax_0.set_yticklabels([])
ax_0.set_yticks([])
ax_0.set_ylim(-y_range_mult*comb_harm_arr_A['Fourier Amplitude [Arb]'], y_range_mult*comb_harm_arr_A['Fourier Amplitude [Arb]'])
#axes[0].legend()

for item in ([ax_0.xaxis.label, ax_0.yaxis.label] +
             ax_0.get_xticklabels() + ax_0.get_yticklabels()):
    item.set_fontsize(label_font_size)

axes[1].errorbar(x=t_arr[::fract_points_plot], y=x_det_shifted_arr_B[::fract_points_plot], marker='.', yerr=x_det_std_arr_B[::fract_points_plot], linestyle='None', color='blue', label='"B" Configuration')

axes[1].plot(t_arr, x_det_shifted_cos_arr_B, color='blue', linewidth=1, linestyle='dashed')

axes[1].set_xlim(np.min(t_arr), np.max(t_arr))
axes[1].set_ylabel('Detector current (nA)', color='blue')
axes[1].set_xlim(np.min(t_arr), np.max(t_arr))
axes[1].set_ylim(np.abs(dc_det_B) - y_range_mult*det_harm_arr_B['Fourier Amplitude [Arb]'], np.abs(dc_det_B) + y_range_mult*det_harm_arr_B['Fourier Amplitude [Arb]'])
#axes[1].grid()
axes[1].set_xlabel('Time (ms)')
axes[1].tick_params(axis='y', colors='blue')
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[1].axvline(x=t_det_B_cross, color='green', linewidth=0.5)
axes[1].axvline(x=t_comb_B_cross, color='green', linewidth=0.5)

for item in ([axes[1].xaxis.label, axes[1].yaxis.label] +
             axes[1].get_xticklabels() + axes[1].get_yticklabels()):
    item.set_fontsize(label_font_size)

ax_1 = axes[1].twinx()
ax_1.plot(t_arr, x_comb_cos_arr_B, color='purple')
#axes[1].legend()
ax_1.set_ylabel('Combiner signal (Arb. Units)', color='purple')
ax_1.set_yticklabels([])
ax_1.set_yticks([])
ax_1.set_ylim(-y_range_mult*comb_harm_arr_B['Fourier Amplitude [Arb]'], y_range_mult*comb_harm_arr_B['Fourier Amplitude [Arb]'])

for item in ([ax_1.xaxis.label, ax_1.yaxis.label] +
             ax_1.get_xticklabels() + ax_1.get_yticklabels()):
    item.set_fontsize(label_font_size)

axes[0].arrow(x=t_det_A_cross+0.2, y=79.85, dx=-0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')
axes[0].arrow(x=t_comb_A_cross-0.2, y=79.85, dx=0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')

axes[0].text(x=t_comb_B_cross-0.4, y=79.81, s=r'$\Delta\theta$', fontsize=20, color='green')

axes[1].arrow(x=t_det_B_cross-0.2, y=79.9, dx=0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')
axes[1].arrow(x=t_comb_B_cross+0.2, y=79.9, dx=-0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')

axes[1].text(x=t_det_A_cross-0.55, y=79.85, s=r'$\Delta\theta$', fontsize=20, color='green')

ax_0.axhline(y=0, color='black', linewidth=0.5)
ax_1.axhline(y=0, color='black', linewidth=0.5)

ax_0.margins(x=0)
ax_1.margins(x=0)

fig.tight_layout()

#os.chdir('C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Pics')
#plt_name = 'RF_CHA_and_B.svg'
#plt.savefig(plt_name)

plt.show()
#%%
os.chdir('C:\Users\Helium1\Google Drive\Python\Test')
#%%
data1=get_av_trace_data(trace_name, ch_label, n_periods, n_periods_to_remove)
#%%
str(data1)
#%%
f=open('x_det_arr_A.txt', 'w')
f.write(x_det_arr_A)
f.close()
#%%
f1 = open('x_det_arr_A.txt', 'r')
data = f1.read()
#%%
import pickle
#%%
f = open('data.dat', 'w')
#%%
pickle.dump([t_arr, x_det_arr_A, x_det_std_arr_A, det_harm_arr_A, x_comb_arr_A, x_comb_std_arr_A, comb_harm_arr_A, x_det_arr_B, x_det_std_arr_B, det_harm_arr_B, x_comb_arr_B, x_comb_std_arr_B, comb_harm_arr_B], f)
f.close()
#%%
os.getcwd()
#%%
with open('data.dat') as f:
    t_arr, x_det_arr_A, x_det_std_arr_A, det_harm_arr_A, x_comb_arr_A, x_comb_std_arr_A, comb_harm_arr_A, x_det_arr_B, x_det_std_arr_B, det_harm_arr_B, x_comb_arr_B, x_comb_std_arr_B, comb_harm_arr_B = pickle.load(f)
