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

import matplotlib.pyplot as plt

import pickle


experiment_folder = '180321-150600 - FOSOF Acquisition - 0 config, 18 V per cm PD ON 120 V'

#%%
# Pickled object path
os.chdir('C:\Users\Helium1\Google Drive\Python\Test')

with open('data.dat') as f:
    t_arr, x_det_arr_A, x_det_std_arr_A, det_harm_arr_A, x_comb_arr_A, x_comb_std_arr_A, comb_harm_arr_A, x_det_arr_B, x_det_std_arr_B, det_harm_arr_B, x_comb_arr_B, x_comb_std_arr_B, comb_harm_arr_B = pickle.load(f)
#%%
t_arr = t_arr * 1E-3

# Phase shift of the combiners' signals that is needed to be shown
phi_needed = -np.pi/2

#-----------------------
# RF CHANNEL A
#-----------------------
trace_name = 'r002a002f0911.2435chA'
ch_label = 1
#t_arr, x_det_arr_A, x_det_std_arr_A, det_harm_arr_A = get_av_trace_data(trace_name, ch_label, n_periods, n_periods_to_remove)

ch_label = 4
#t_arr, x_comb_arr_A, x_comb_std_arr_A, comb_harm_arr_A = get_av_trace_data(trace_name, ch_label, n_periods, n_periods_to_remove)

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
#t_arr, x_det_arr_B, x_det_std_arr_B, det_harm_arr_B = get_av_trace_data(trace_name, ch_label, n_periods, n_periods_to_remove)


ch_label = 4
#t_arr, x_comb_arr_B, x_comb_std_arr_B, comb_harm_arr_B = get_av_trace_data(trace_name, ch_label, n_periods, n_periods_to_remove)

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
#
