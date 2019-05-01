from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

sys.path.insert(0,"E:/Google Drive/Research/Lamb shift measurement/Code")

from exp_data_analysis import *
from fosof_data_set_analysis import *
from quenching_curve_analysis import *
from fosof_data_zero_cross_analysis import *

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

#%%
# Folder containing binary traces in .npy format
binary_traces_folder = r"E:\Google Drive\Research\Lamb shift measurement\Data\Binary_traces_thesis"

experiment_folder_name = '180321-150600 - FOSOF Acquisition - 0 config, 18 V per cm PD ON 120 V'

beam_rms_rad = None
# List of beam  rms radius values for correcting the FOSOF phases using the simulations. Value of None corresponds to not applying the correction.
data_set = DataSetFOSOF(exp_folder_name=experiment_folder_name, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

exp_params_dict = dict(data_set.get_exp_parameters())

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

#%%
def get_av_trace_data(trace_name, ch_label, n_periods, n_periods_to_remove):

    digi_trace_file_name = trace_name + '_0' + str(ch_label) + '.digi.npy'

    os.chdir(binary_traces_folder)

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
label_font_size = 17

fig, axes = plt.subplots(nrows=1, ncols=2)

fig.set_size_inches(12, 4)

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
#ax_0.set_ylabel('Combiner signal (Arb. Units)', color='purple')
ax_0.set_yticklabels([])
ax_0.set_yticks([])
ax_0.set_ylim(-y_range_mult*comb_harm_arr_A['Fourier Amplitude [Arb]'], y_range_mult*comb_harm_arr_A['Fourier Amplitude [Arb]'])
#axes[0].legend()

for item in ([ax_0.xaxis.label, ax_0.yaxis.label] +
             ax_0.get_xticklabels() + ax_0.get_yticklabels()):
    item.set_fontsize(label_font_size)

axes[1].plot(t_arr, x_comb_cos_arr_B, color='purple')
#axes[1].legend()
axes[1].set_ylabel('Combiner signal (arb. units)', color='purple')
axes[1].set_yticklabels([])
axes[1].set_yticks([])
axes[1].set_ylim(-y_range_mult*comb_harm_arr_B['Fourier Amplitude [Arb]'], y_range_mult*comb_harm_arr_B['Fourier Amplitude [Arb]'])

for item in ([axes[1].xaxis.label, axes[1].yaxis.label] +
             axes[1].get_xticklabels() + axes[1].get_yticklabels()):
    item.set_fontsize(label_font_size)

ax_1 = axes[1].twinx()

ax_1.errorbar(x=t_arr[::fract_points_plot], y=x_det_shifted_arr_B[::fract_points_plot], marker='.', yerr=x_det_std_arr_B[::fract_points_plot], linestyle='None', color='blue', label='"B" Configuration')

ax_1.plot(t_arr, x_det_shifted_cos_arr_B, color='blue', linewidth=1, linestyle='dashed')

ax_1.set_xlim(np.min(t_arr), np.max(t_arr))
ax_1.set_ylabel('Detector current (nA)', color='blue')
ax_1.set_xlim(np.min(t_arr), np.max(t_arr))
ax_1.set_ylim(np.abs(dc_det_B) - y_range_mult*det_harm_arr_B['Fourier Amplitude [Arb]'], np.abs(dc_det_B) + y_range_mult*det_harm_arr_B['Fourier Amplitude [Arb]'])
#axes[1].grid()
axes[1].set_xlabel('Time (ms)')
ax_1.tick_params(axis='y', colors='blue')
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax_1.yaxis.set_major_locator(MaxNLocator(integer=True))

ax_1.axvline(x=t_det_B_cross, color='green', linewidth=0.5)
ax_1.axvline(x=t_comb_B_cross, color='green', linewidth=0.5)


for item in ([ax_1.xaxis.label, ax_1.yaxis.label] +
             ax_1.get_xticklabels() + ax_1.get_yticklabels()):
    item.set_fontsize(label_font_size)

axes[0].arrow(x=t_det_A_cross+0.2, y=79.85, dx=-0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')
axes[0].arrow(x=t_comb_A_cross-0.2, y=79.85, dx=0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')

axes[0].text(x=t_comb_B_cross-0.47, y=79.81, s=r'$\Delta\theta^{A}$', fontsize=label_font_size, color='green')

ax_1.arrow(x=t_det_B_cross-0.2, y=79.9, dx=0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')
ax_1.arrow(x=t_comb_B_cross+0.2, y=79.9, dx=-0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')

ax_1.text(x=t_det_A_cross-0.6, y=79.86, s=r'$\Delta\theta^{B}$', fontsize=label_font_size, color='green')

ax_0.axhline(y=0, color='black', linewidth=0.5)
axes[1].axhline(y=0, color='black', linewidth=0.5)

ax_0.margins(x=0)
ax_1.margins(x=0)

fig.tight_layout()

plt.subplots_adjust(wspace = 0.2)

os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\FOSOF_phase_canc')
plt_name = 'RF_CHA_and_B.svg'
plt.savefig(plt_name)

plt.show()
#%%
#%%
''' '0'- and 'pi'-configuration FOSOF plots.
'''

exp_folder_name_0 = '180321-150600 - FOSOF Acquisition - 0 config, 18 V per cm PD ON 120 V'
exp_folder_name_pi = '180321-152522 - FOSOF Acquisition - pi config, 18 V per cm PD ON 120 V'

av_type = 'Phasor Averaging'
av_data_std = 'Phase RMS Repeat STD'
#%%
# 0-configuration data

beam_rms_rad = None
# List of beam  rms radius values for correcting the FOSOF phases using the simulations. Value of None corresponds to not applying the correction.
data_set = DataSetFOSOF(exp_folder_name=exp_folder_name_0, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

# The power correction is performed only for the simple FOSOF data sets.
if data_set.get_exp_parameters()['Experiment Type'] != 'Waveguide Carrier Frequency Sweep':
    beam_rms_rad = None
    data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

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

data_df = fosof_phase_df['RF Combiner I Reference', 'First Harmonic', av_type, av_data_std]

x_0_arr = data_df.index.values
y_0_arr = data_df['Weighted Mean'].values
y_0_std_arr = data_df['Weighted STD'].values

y_0_plot_arr = correct_FOSOF_phases_zero_crossing(x_0_arr, y_0_arr, y_0_std_arr) / 2
y_0_std_plot_arr = data_df['Weighted STD'].values / 2

fosof_phase_0_df = data_df
#%%
# pi-configuration data

beam_rms_rad = None
# List of beam  rms radius values for correcting the FOSOF phases using the simulations. Value of None corresponds to not applying the correction.
data_set = DataSetFOSOF(exp_folder_name=exp_folder_name_pi, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

# The power correction is performed only for the simple FOSOF data sets.
if data_set.get_exp_parameters()['Experiment Type'] != 'Waveguide Carrier Frequency Sweep':
    beam_rms_rad = None
    data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

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

data_df = fosof_phase_df['RF Combiner I Reference', 'First Harmonic', av_type, av_data_std]

x_pi_arr = data_df.index.values
y_pi_arr = data_df['Weighted Mean'].values
y_pi_std_arr = data_df['Weighted STD'].values

y_pi_plot_arr = correct_FOSOF_phases_zero_crossing(x_pi_arr, y_pi_arr, y_pi_std_arr) / 2
y_pi_std_plot_arr = data_df['Weighted STD'].values / 2

fosof_phase_pi_df = data_df
#%%
# 0-pi data

# Calculate fosof phases + their uncertainties
fosof_phase_df = (fosof_phase_0_df[['Weighted Mean']] - fosof_phase_pi_df[['Weighted Mean']]).join(np.sqrt(fosof_phase_0_df[['Weighted STD']]**2 + fosof_phase_pi_df[['Weighted STD']]**2)).sort_index(axis='columns')

# Convert the phases to the 0-2pi range
fosof_phase_df.loc[slice(None), 'Weighted Mean'] = fosof_phase_df['Weighted Mean'].transform(convert_phase_to_2pi_range)

phase_data_df = fosof_phase_df

x_data_arr = phase_data_df.index.get_level_values('Waveguide Carrier Frequency [MHz]').values
y_data_arr = phase_data_df['Weighted Mean'].values
y_sigma_arr = phase_data_df['Weighted STD'].values

# Correct for 0-2*np.pi jumps

y_data_arr = correct_FOSOF_phases_zero_crossing(x_data_arr, y_data_arr, y_sigma_arr)

#Important division by a factor of 4 that was explained before
y_data_arr = y_data_arr / 4
y_sigma_arr = y_sigma_arr / 4

fit_param_dict = calc_fosof_lineshape_param(x_data_arr, y_data_arr, y_sigma_arr)
#%%
# Plotting
fig, ax = plt.subplots()
fig.set_size_inches(12,8)
#data_set.exp_data_averaged_df.reset_index().plot(x='RF System Power [W]', y='Weighted Mean', kind='scatter', yerr='Weighted STD', ax=ax, color='blue')

ax.errorbar(x_0_arr, y_0_plot_arr, y_0_std_plot_arr, linestyle='', marker='.', color='red')

ax.errorbar(x_pi_arr, y_pi_plot_arr, y_pi_std_plot_arr, linestyle='', marker='.', color='blue')

ax.set_xlabel(r'rf frequency (MHz)')
ax.set_ylabel(r'$\theta^{AB}$ (rad)')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(17)

plt.show()
#%%
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
import mpl_toolkits.axes_grid1.inset_locator as inset_locator
#%%
x_plot_arr = np.linspace(np.min(x_data_arr), np.max(x_data_arr), 100)
y_plot_arr = fit_param_dict['Slope [Rad/MHz]'] * x_plot_arr + fit_param_dict['Offset [MHz]']
y_res_arr = fit_param_dict['Slope [Rad/MHz]'] * x_data_arr + fit_param_dict['Offset [MHz]'] - y_data_arr

fig = plt.figure()
fig.set_size_inches(13,15)

gs = gridspec.GridSpec(nrows=6, ncols=2, figure=fig, hspace=1.5)

ax0 = plt.subplot(gs[2:5, :])

ax0.errorbar(x_data_arr, y_data_arr, y_sigma_arr, linestyle='', marker='.', color='green')
ax0.plot(x_plot_arr, y_plot_arr, color='green')

ax1 = plt.subplot(gs[-1, :])

ax1.errorbar(x_data_arr, y_res_arr*1E3, y_sigma_arr*1E3, linestyle='', marker='.', color='green')

ax1.set_xticklabels([])

for item in ([ax0.title, ax0.xaxis.label, ax0.yaxis.label] +
             ax0.get_xticklabels() + ax0.get_yticklabels()):
    item.set_fontsize(17)

for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(17)


x_lim = ax0.get_xlim()
y_lim = ax0.get_ylim()

ax0.plot([ax0.get_xlim()[0], fit_param_dict['Zero-crossing Frequency [MHz]']], [0, 0], color='black', linestyle='dashed')

ax0.plot([fit_param_dict['Zero-crossing Frequency [MHz]'], fit_param_dict['Zero-crossing Frequency [MHz]']], [0, ax0.get_ylim()[0]], color='black', linestyle='dashed')

start_x = 908.5 + 0.2
start_y = 0
end_x = 908.7 + 0.2
end_y = 0
arrow_zero_cross_1 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='-|>', mutation_scale=30, color='black', linewidth=2)

ax0.add_patch(arrow_zero_cross_1)

start_x = fit_param_dict['Zero-crossing Frequency [MHz]']
start_y = -0.1 - 0.02
end_x = fit_param_dict['Zero-crossing Frequency [MHz]']
end_y = -0.1 - 0.02 - 0.02
arrow_zero_cross_2 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='-|>', mutation_scale=30, color='black', linewidth=2)

ax0.add_patch(arrow_zero_cross_2)

ax0.set_xlim(x_lim)
ax0.set_ylim(y_lim)

ax0.set_xlabel('rf frequency (MHz)')
ax0.set_ylabel(r'$\theta$ (rad)')

ax1.set_ylabel('residuals'+'\n'+'(mrad)')

inset_axes = inset_locator.inset_axes(ax0,
                        width="40%", # width = 30% of parent_bbox
                        height="30%", # height : 1 inch
                        loc='upper right',
                        borderpad=2)
inset_axes.errorbar(x_0_arr, y_0_plot_arr, y_0_std_plot_arr, linestyle='', marker='.', color='red')

inset_axes.errorbar(x_pi_arr, y_pi_plot_arr, y_pi_std_plot_arr, linestyle='', marker='.', color='blue')

inset_axes.set_xlabel(r'rf frequency (MHz)')
inset_axes.set_ylabel(r'$\theta^{AB}$ (rad)')
for item in ([inset_axes.title, inset_axes.xaxis.label, inset_axes.yaxis.label] + inset_axes.get_xticklabels() + inset_axes.get_yticklabels()):
    item.set_fontsize(17*0.7)

from matplotlib.ticker import MaxNLocator

y_range_mult = 1.2
label_font_size = 17

fract_points_plot = 8

ax_3 = plt.subplot(gs[0:2, 0])

ax_3.errorbar(x=t_arr[::fract_points_plot], y=x_det_shifted_arr_A[::fract_points_plot], marker='.', yerr=x_det_std_arr_A[::fract_points_plot], linestyle='None', color='red', label='"A" Configuration')

ax_3.plot(t_arr, x_det_shifted_cos_arr_A, color='red', linewidth=1, linestyle='dashed')
ax_3.set_xlim(np.min(t_arr), np.max(t_arr))
ax_3.set_ylim(np.abs(dc_det_A) - y_range_mult*det_harm_arr_A['Fourier Amplitude [Arb]'], np.abs(dc_det_A) + y_range_mult*det_harm_arr_A['Fourier Amplitude [Arb]'])
#axes[0].grid()
ax_3.set_xlabel('Time (ms)')
ax_3.set_ylabel('Detector current (nA)', color='red')
ax_3.tick_params(axis='y', colors='red')
ax_3.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_3.yaxis.set_major_locator(MaxNLocator(integer=True))

ax_3.axvline(x=t_det_A_cross, color='green', linewidth=0.5)
ax_3.axvline(x=t_comb_A_cross, color='green', linewidth=0.5)

for item in ([ax_3.xaxis.label, ax_3.yaxis.label] +
             ax_3.get_xticklabels() + ax_3.get_yticklabels()):
    item.set_fontsize(label_font_size)

ax_0 = ax_3.twinx()
ax_0.plot(t_arr, x_comb_cos_arr_A, color='purple')
#ax_0.set_ylabel('Combiner signal (Arb. Units)', color='purple')
ax_0.set_yticklabels([])
ax_0.set_yticks([])
ax_0.set_ylim(-y_range_mult*comb_harm_arr_A['Fourier Amplitude [Arb]'], y_range_mult*comb_harm_arr_A['Fourier Amplitude [Arb]'])
#axes[0].legend()

for item in ([ax_0.xaxis.label, ax_0.yaxis.label] +
             ax_0.get_xticklabels() + ax_0.get_yticklabels()):
    item.set_fontsize(label_font_size)

ax_4 = plt.subplot(gs[0:2, 1])

ax_4.plot(t_arr, x_comb_cos_arr_B, color='purple')
#axes[1].legend()
ax_4.set_ylabel('Combiner signal (arb. units)', color='purple')
ax_4.set_yticklabels([])
ax_4.set_yticks([])
ax_4.set_ylim(-y_range_mult*comb_harm_arr_B['Fourier Amplitude [Arb]'], y_range_mult*comb_harm_arr_B['Fourier Amplitude [Arb]'])

for item in ([ax_4.xaxis.label, ax_4.yaxis.label] +
             ax_4.get_xticklabels() + ax_4.get_yticklabels()):
    item.set_fontsize(label_font_size)

ax_1 = ax_4.twinx()

ax_1.errorbar(x=t_arr[::fract_points_plot], y=x_det_shifted_arr_B[::fract_points_plot], marker='.', yerr=x_det_std_arr_B[::fract_points_plot], linestyle='None', color='blue', label='"B" Configuration')

ax_1.plot(t_arr, x_det_shifted_cos_arr_B, color='blue', linewidth=1, linestyle='dashed')

ax_1.set_xlim(np.min(t_arr), np.max(t_arr))
ax_1.set_ylabel('Detector current (nA)', color='blue')
ax_1.set_xlim(np.min(t_arr), np.max(t_arr))
ax_1.set_ylim(np.abs(dc_det_B) - y_range_mult*det_harm_arr_B['Fourier Amplitude [Arb]'], np.abs(dc_det_B) + y_range_mult*det_harm_arr_B['Fourier Amplitude [Arb]'])
#axes[1].grid()
ax_4.set_xlabel('Time (ms)')
ax_1.tick_params(axis='y', colors='blue')
ax_4.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_1.yaxis.set_major_locator(MaxNLocator(integer=True))

ax_1.axvline(x=t_det_B_cross, color='green', linewidth=0.5)
ax_1.axvline(x=t_comb_B_cross, color='green', linewidth=0.5)


for item in ([ax_1.xaxis.label, ax_1.yaxis.label] +
             ax_1.get_xticklabels() + ax_1.get_yticklabels()):
    item.set_fontsize(label_font_size)

ax_3.arrow(x=t_det_A_cross+0.2, y=79.85, dx=-0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')
ax_3.arrow(x=t_comb_A_cross-0.2, y=79.85, dx=0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')

ax_3.text(x=t_comb_B_cross-0.47, y=79.81, s=r'$\Delta\theta^{A}$', fontsize=label_font_size , color='green')

ax_1.arrow(x=t_det_B_cross-0.2, y=79.9, dx=0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')
ax_1.arrow(x=t_comb_B_cross+0.2, y=79.9, dx=-0.1, dy=0, head_width=0.05, head_length=2*0.05, fc='green', linestyle='solid', color='green')

ax_1.text(x=t_det_A_cross-0.6, y=79.86, s=r'$\Delta\theta^{B}$', fontsize=label_font_size , color='green')

ax_0.axhline(y=0, color='black', linewidth=0.5)
ax_4.axhline(y=0, color='black', linewidth=0.5)

ax_0.margins(x=0)
ax_1.margins(x=0)

fig.tight_layout()

plt.subplots_adjust(wspace = 0.2)

#os.chdir('C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Pics')
#plt_name = 'RF_CHA_and_B.svg'
#plt.savefig(plt_name)


plt.show()
#%%
np.array([0,1,2,3])[0:2]
#%%
gs[5:6]
