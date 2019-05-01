''' Calculation of the beam speed, given the averaged data output by the Mathematica worksheet.
'''

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
import scipy.optimize

import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

path_data_df = pd.read_csv(filepath_or_buffer='path_data.csv', delimiter=',', comment='#', header=[0], skip_blank_lines=True, index_col=[0])

code_folder_path = path_data_df.loc['Code Folder'].values[0].replace('\\', '/')
fosof_analyzed_data_folder_path = path_data_df.loc['FOSOF Analyzed Data Folder'].values[0].replace('\\', '/')
wvg_calib_data_folder_path = path_data_df.loc['Waveguide Calibration Folder'].values[0].replace('\\', '/')
krytar109B_pwr_det_calib_folder_path = path_data_df.loc['KRYTAR 109 B Power Detector Calibration Data Folder'].values[0].replace('\\', '/')
beam_speed_folder_path = path_data_df.loc['Beam Speed Data Folder'].values[0].replace('\\', '/')

sys.path.insert(0, code_folder_path)
#%%
beam_speed_path = beam_speed_folder_path

def gauss_fit_func(x, a, b, sigma, x0):
    return a * np.exp(-(x-x0)**2/(2*sigma**2)) + b

class BeamSpeed():
    def __init__(self, acc_volt, wvg_sep):
        self.wvg_sep = wvg_sep
        self.acc_volt = acc_volt

        self.config_0_name = str(self.acc_volt) + '_' + str(self.wvg_sep) + '_data_0.CSV'
        self.config_pi_name = str(self.acc_volt) + '_' + str(self.wvg_sep) + '_data_pi.CSV'

        os.chdir(beam_speed_path)

        self.beam_speed_raw_0_data = np.loadtxt(self.config_0_name, delimiter=',', dtype=np.float64)

        self.beam_speed_raw_pi_data = np.loadtxt(self.config_pi_name, delimiter=',', dtype=np.float64)

        self.fit_0, self.sigma_0, self.x_fit_0_arr, self.fit_0_arr = self.get_fit(self.beam_speed_raw_0_data, (0.175, 1, 20, 30))
        self.fit_pi, self.sigma_pi, self.x_fit_pi_arr, self.fit_pi_arr = self.get_fit(self.beam_speed_raw_pi_data, (0.175, 1, 20, -30))

    def get_fit(self, beam_speed_raw_data, p0_set):

        t_delay_raw_arr = beam_speed_raw_data[:, 0]
        fract_change_raw_arr = beam_speed_raw_data[:, 1]
        fract_change_std_raw_arr = beam_speed_raw_data[:, 2]

        # Transform the data
        fract_change_arr = -1*(1/fract_change_raw_arr-1)
        fract_change_std_arr = fract_change_std_raw_arr/fract_change_raw_arr**2

        # Fit function for the nonlinear least-squares fitting routine
        fit_raw, cov_raw = scipy.optimize.curve_fit(f=gauss_fit_func, xdata=t_delay_raw_arr, ydata=fract_change_arr, p0=p0_set, sigma=fract_change_std_arr, absolute_sigma=False)

        sigma_raw = np.sqrt(np.diag(cov_raw))

        x_fit_raw_arr = np.linspace(np.min(t_delay_raw_arr), np.max(t_delay_raw_arr), t_delay_raw_arr.shape[0]*10)
        fit_raw_arr = gauss_fit_func(x_fit_raw_arr, *fit_raw)

        return fit_raw, sigma_raw, x_fit_raw_arr, fit_raw_arr

    def calc_speed(self):
        delta_dist = 5/1000 * 2.54
        wvg_width = 3

        v_speed = (self.wvg_sep+wvg_width)/((self.fit_0[3]-self.fit_pi[3])/2)
        v_fract_unc = np.sqrt((delta_dist/self.wvg_sep)**2 + (delta_dist/wvg_width)**2 + (self.sigma_0[3]/self.fit_0[3])**2 + (self.sigma_pi[3]/self.fit_pi[3])**2)

        return [self.wvg_sep, self.acc_volt, v_speed, v_speed*v_fract_unc]

    def get_plot(self, ax):

        t_delay_raw_0_arr = self.beam_speed_raw_0_data[:, 0]
        fract_change_raw_0_arr = self.beam_speed_raw_0_data[:, 1]
        fract_change_std_raw_0_arr = self.beam_speed_raw_0_data[:, 2]

        # Transform the data
        fract_change_0_arr = -1*(1/fract_change_raw_0_arr-1)
        fract_change_std_0_arr = fract_change_std_raw_0_arr/fract_change_raw_0_arr**2

        t_delay_raw_pi_arr = self.beam_speed_raw_pi_data[:, 0]
        fract_change_raw_pi_arr = self.beam_speed_raw_pi_data[:, 1]
        fract_change_std_raw_pi_arr = self.beam_speed_raw_pi_data[:, 2]

        # Transform the data
        fract_change_pi_arr = -1*(1/fract_change_raw_pi_arr-1)
        fract_change_std_pi_arr = fract_change_std_raw_pi_arr/fract_change_raw_pi_arr**2

        ax.errorbar(t_delay_raw_0_arr, fract_change_0_arr, fract_change_std_0_arr, linestyle='', marker='.', color='red')

        ax.errorbar(t_delay_raw_pi_arr, fract_change_pi_arr, fract_change_std_pi_arr, linestyle='', marker='.', color='blue')

        ax.plot(self.x_fit_0_arr, self.fit_0_arr, color='red')
        ax.plot(self.x_fit_pi_arr, self.fit_pi_arr, color='blue')

        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        rect_0 = Rectangle((self.fit_0[3] - self.sigma_0[3], y_lim[0]), 2*self.sigma_0[3], y_lim[1]-y_lim[0], color='red', fill=True, alpha=1)
        ax.add_patch(rect_0)

        rect_pi = Rectangle((self.fit_pi[3] - self.sigma_pi[3], y_lim[0]), 2*self.sigma_pi[3], y_lim[1]-y_lim[0], color='blue', fill=True, alpha=1)
        ax.add_patch(rect_pi)

        arrow_dt_0 = mpatches.FancyArrowPatch((0, 0.10), (self.fit_0[3], 0.10), arrowstyle='<|-|>', mutation_scale=30, color='black', linewidth=1)

        ax.add_patch(arrow_dt_0)

        arrow_dt_pi = mpatches.FancyArrowPatch((0, 0.10), (self.fit_pi[3], 0.10), arrowstyle='<|-|>', mutation_scale=30, color='black', linewidth=1)

        ax.plot([0, 0], [y_lim[0], y_lim[1]], linestyle='dashed', color='black')

        ax.add_patch(arrow_dt_pi)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        ax.set_xlabel(r'$\Delta t_{\mathrm{set}}$ (ns)')
        ax.set_ylabel(r'$-(R_2/R_1 - 1)$')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)

        return ax

#%%
# bs_4_22 = BeamSpeed(acc_volt=22.17, wvg_sep=4)
# bs_4_16 = BeamSpeed(acc_volt=16.27, wvg_sep=4)
# bs_4_50 = BeamSpeed(acc_volt=49.86, wvg_sep=4)
# bs_5_50 = BeamSpeed(acc_volt=49.86, wvg_sep=5)
# bs_7_50 = BeamSpeed(acc_volt=49.86, wvg_sep=7)
# #%%
# fig = plt.figure()
# fig.set_size_inches(10, 7)
# ax = fig.add_subplot(111)
#
# ax = bs_7_50.get_plot(ax)
# fig.tight_layout()
# plt.show()
# #%%
# # These are the speeds that were experimentally measured
# beam_speed_data_df = pd.DataFrame(np.array([bs_4_22.calc_speed(), bs_4_16.calc_speed(), bs_4_50.calc_speed(), bs_5_50.calc_speed(), bs_7_50.calc_speed()]), columns=['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Beam Speed [cm/ns]', 'Beam Speed STD [cm/ns]']).set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]'])
#
# # While taking data for different separation, for the same accelerating voltages, it is true that we are not keeping all of the Source parameters (=voltages) the same all the time. The spread in the values that we got for the beam speeds is the good indicator of the variability of the Source parameters that were used for the experiment. Thus the average of these values gives us the best estimate for the speed. The STDOM of the spread is added with quadruate to the RMS uncertainty in the speed values to give us the average uncertainty in the average beam speed.
# def get_av_speed(df):
#
#     if df.shape[0] > 1:
#         av_s = df[['Beam Speed [cm/ns]']].aggregate(lambda x: np.mean(x))
#         av_s['Beam Speed STD [cm/ns]'] = np.sqrt((np.std(df['Beam Speed [cm/ns]'], ddof=1)/np.sqrt(df['Beam Speed STD [cm/ns]'].shape[0]))**2 + np.sum(df['Beam Speed STD [cm/ns]']**2)/df['Beam Speed STD [cm/ns]'].shape[0]**2)
#     else:
#         av_s = df.iloc[0]
#     return av_s
#
# beam_speed_df = beam_speed_data_df.groupby('Accelerating Voltage [kV]').apply(get_av_speed)
#
# beam_speed_df = beam_speed_data_df.reset_index('Waveguide Separation [cm]').join(beam_speed_df, lsuffix='_Delete').drop(columns=['Beam Speed [cm/ns]_Delete', 'Beam Speed STD [cm/ns]_Delete'])
#
# # We add the 6 cm and 49.86 kV point to the dataframe of beam speeds.
#
# wvg_sep_6_s = beam_speed_df.loc[49.86].iloc[0].copy()
# wvg_sep_6_s['Waveguide Separation [cm]'] = 6
#
# beam_speed_df = beam_speed_df.append(wvg_sep_6_s)
#
# # Calculate the SOD
# # Assumed resonant frequency [MHz]
# freq_diff = 909.894
# # Speed of light [m/s]
# c_light = 299792458
#
# sod_shift_df = beam_speed_df.copy()
# sod_shift_df['SOD Shift [MHz]'] = (1/np.sqrt(1-(beam_speed_df['Beam Speed [cm/ns]'] * 1E-2 * 1E9/c_light)**2) - 1) * freq_diff
#
# sod_shift_df['SOD Shift STD [MHz]'] = freq_diff * beam_speed_df['Beam Speed [cm/ns]'] * 1E-2 * 1E9 * beam_speed_df['Beam Speed STD [cm/ns]'] * 1E-2 * 1E9 / ((1 - (beam_speed_df['Beam Speed [cm/ns]']/c_light)**2)**(1.5) * c_light**2)
#
# sod_shift_df = sod_shift_df.set_index('Waveguide Separation [cm]', append=True).swaplevel(0, 1)
# #%%
# beam_speed_data_df
# #%%
# sod_shift_df
# #%%
