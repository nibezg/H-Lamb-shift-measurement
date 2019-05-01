''' Analysis of the 910 on-off data sets.
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
from quenching_curve_analysis import *
from ZX47_Calibration_analysis import *

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
# Power detector calibration curve for the pre-910 quench cavity
data_calib_set = ZX4755LNCalibration(910)
spl_smoothing_inverse, fract_unc = data_calib_set.get_calib_curve()

# Quench curve for pre-910 cavity. We assume that for all of the data sets these quench curves were similar enough.
pre_910_data_set_name = '180629-171507 - Quench Cavity Calibration - pre-910 PD ON'
data_set_910 = DataSetQuenchCurveCavity(pre_910_data_set_name)

exp_params_s_910 = data_set_910.get_exp_parameters()

rf_pwr_det_calib_df_910 = data_set_910.get_quench_cav_rf_pwr_det_calib()
quenching_cavities_df_910 = data_set_910.get_quenching_cav_data()
quenching_cavities_av_df_910 = data_set_910.get_av_quenching_cav_data()
quenching_curve_df_910 = data_set_910.get_quenching_curve()
quenching_curve_results_s_910 = data_set_910.get_quenching_curve_results()

# This is the amount by which the detected power is different from the power that goes into the quenching cavities.
attenuation = 10 + 30 # [dBm]

#%%
# Perform the analysis

saving_folder_location = 'E:/2017-10-17 Lamb Shift Measurement/Data/FOSOF analyzed data sets'
os.chdir(saving_folder_location)

exp_folder_name_s = pd.read_csv('910_on_off_fosof_list.csv', header=None, index_col=0)[1]

on_off_data_df = pd.DataFrame()
for index, exp_folder_name in exp_folder_name_s.iteritems():
    print('================')
    print(exp_folder_name)
    data_set = fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=False, beam_rms_rad_to_load=None)

    # Get average power detector reading (in Volts) for the pre-910 cavity
    mean_pwr_det_volt = data_set.get_quenching_cav_data().loc[(slice(None), slice(None), 'on'), slice(None)]['Pre-Quench', '910', 'Power Detector Reading [V]'].mean()

    # Convert it to Watts
    mean_pwr_det_dBm = spl_smoothing_inverse(mean_pwr_det_volt) + attenuation
    mean_pwr_det_Watt = 10**(mean_pwr_det_dBm/10) * 10**(-3)

    # Determine the corresponding surviving fraction from the pre-910 quench curve
    surv_frac = data_set_910.quenching_rf_power_func(mean_pwr_det_Watt)

    # Maximum surviving fraction at 0 RF power going into the quench cavity
    surf_frac_max = data_set_910.quenching_rf_power_func(0)

    # Fractional offset = surviving fraction at the first pi-pulse.
    frac_offset = quenching_curve_results_s_910['Quenching Offset']

    # Fraction of the quenched metastable atoms (in the f=0 hyperfine state)
    frac_quenched = 1-(surv_frac-frac_offset)/(surf_frac_max-frac_offset)

    pre_910_states_averaged_df, pre_910_av_difference_df = data_set.analyze_pre_910_switching()

    # Get mean amplitude-to-DC ratio (when pre-910 cavity is OFF) and also maximum fractional deviation from the mean, which is needed to make sure that there is no significant variation of the amplitude with frequency (not much is expected, since the frequency scan range is relatively narrow compared to the SOF linewidth)
    ampl_df, phase_df = data_set.average_FOSOF_over_repeats()
    mean_ampl_off = ampl_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC RMS Repeat STD', 'Weighted Mean'].loc['off'].mean()

    ampl_min_dev = mean_ampl_off - ampl_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC RMS Repeat STD', 'Weighted Mean'].loc['off'].min()
    ampl_max_dev = ampl_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC RMS Repeat STD', 'Weighted Mean'].loc['off'].max() - mean_ampl_off

    max_fract_ampl_dev = np.max([ampl_min_dev, ampl_max_dev]) / mean_ampl_off

    #data_s = pre_910_av_difference_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Averaging Set STD']
    data_s = pre_910_av_difference_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD']


    data_df = pd.DataFrame([[data_set.get_exp_parameters()['Waveguide Separation [cm]'], data_set.get_exp_parameters()['Accelerating Voltage [kV]'], data_set.get_exp_parameters()['Waveguide Electric Field [V/cm]'], data_set.get_exp_parameters()['Configuration'], frac_quenched, mean_ampl_off, max_fract_ampl_dev, data_s['Weighted Mean'], data_s['Weighted STD'], data_s['Reduced Chi Squared'], data_s['P(>Chi Squared)']]], columns=['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Configuration', 'F=0 Fraction Quenched', '<A>', 'Max(|A-<A>|)/<A>', 'Weighted Mean', 'Weighted STD', 'Reduced Chi Squared', 'P(>Chi Squared)'], index=[exp_folder_name])
    on_off_data_df = on_off_data_df.append(data_df)
#%%
os.chdir(saving_folder_location)
on_off_data_df.to_csv(path_or_buf='910_on_off_fosof_data_2.csv')
#%%
'''
========================================
Analysis of the 910 on-off data with the changing Mass Flow Rate.
========================================
'''
# Perform the analysis

saving_folder_location = r'E:\Google Drive\Research\Lamb shift measurement\Data\FOSOF analyzed data sets'
os.chdir(saving_folder_location)

exp_folder_name_s = pd.read_csv('910_on_off_m_flow_fosof_list.csv', header=None, index_col=0)[1]

on_off_data_df = pd.DataFrame()
for index, exp_folder_name in exp_folder_name_s.iteritems():
    print('================')
    print(exp_folder_name)
    data_set = fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=False, beam_rms_rad_to_load=None)

    data_set.select_portion_of_data()

    # Get average power detector reading (in Volts) for the pre-910 cavity
    mean_pwr_det_volt = data_set.get_quenching_cav_data().loc[(slice(None), slice(None), slice(None), 'on'), slice(None)]['Pre-Quench', '910', 'Power Detector Reading [V]'].mean()

    # Convert it to Watts
    mean_pwr_det_dBm = spl_smoothing_inverse(mean_pwr_det_volt) + attenuation
    mean_pwr_det_Watt = 10**(mean_pwr_det_dBm/10) * 10**(-3)

    # Determine the corresponding surviving fraction from the pre-910 quench curve
    surv_frac = data_set_910.quenching_rf_power_func(mean_pwr_det_Watt)

    # Maximum surviving fraction at 0 RF power going into the quench cavity
    surf_frac_max = data_set_910.quenching_rf_power_func(0)

    # Fractional offset = surviving fraction at the first pi-pulse.
    frac_offset = quenching_curve_results_s_910['Quenching Offset']

    # Fraction of the quenched metastable atoms (in the f=0 hyperfine state)
    frac_quenched = 1-(surv_frac-frac_offset)/(surf_frac_max-frac_offset)

    pre_910_states_averaged_df, pre_910_av_difference_df = data_set.analyze_pre_910_switching()

    # Get mean amplitude-to-DC ratio (when pre-910 cavity is OFF) and also maximum fractional deviation from the mean, which is needed to make sure that there is no significant variation of the amplitude with frequency (not much is expected, since the frequency scan range is relatively narrow compared to the SOF linewidth)
    ampl_df, phase_df = data_set.average_FOSOF_over_repeats()
    mean_ampl_off = ampl_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC RMS Repeat STD', 'Weighted Mean'].loc[(slice(None), 'off')].mean()

    ampl_min_dev = mean_ampl_off - ampl_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC RMS Repeat STD', 'Weighted Mean'].loc[(slice(None), 'off')].min()
    ampl_max_dev = ampl_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC RMS Repeat STD', 'Weighted Mean'].loc[(slice(None), 'off')].max() - mean_ampl_off

    max_fract_ampl_dev = np.max([ampl_min_dev, ampl_max_dev]) / mean_ampl_off

    mass_flow_rate_list = list(pre_910_av_difference_df.index.get_level_values('Mass Flow Rate [sccm]').drop_duplicates())

    cgx_press_df = data_set.exp_data_frame[['Charge Exchange Pressure [torr]']].groupby('Mass Flow Rate [sccm]').aggregate(lambda x: np.mean(x))

    data_df_list = []
    for mass_flow_rate in mass_flow_rate_list:

        data_s = pre_910_av_difference_df.loc[mass_flow_rate]['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD']

        data_df = pd.DataFrame([[mass_flow_rate, data_set.get_exp_parameters()['Waveguide Separation [cm]'], data_set.get_exp_parameters()['Accelerating Voltage [kV]'], data_set.get_exp_parameters()['Waveguide Electric Field [V/cm]'], data_set.get_exp_parameters()['Configuration'], frac_quenched, mean_ampl_off, max_fract_ampl_dev, data_s['Weighted Mean'], data_s['Weighted STD'], data_s['Reduced Chi Squared'], data_s['P(>Chi Squared)'], cgx_press_df.loc[mass_flow_rate]['Charge Exchange Pressure [torr]']]], columns=['Mass Flow Rate [sccm]', 'Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Configuration', 'F=0 Fraction Quenched', '<A>', 'Max(|A-<A>|)/<A>', 'Weighted Mean', 'Weighted STD', 'Reduced Chi Squared', 'P(>Chi Squared)', 'Charge Exchange Pressure [Torr]'], index=[exp_folder_name])

        data_df_list.append(data_df)

    data_df = pd.concat(data_df_list).set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Configuration', 'F=0 Fraction Quenched', '<A>', 'Max(|A-<A>|)/<A>', 'Mass Flow Rate [sccm]'], append=True).sort_index()

    on_off_data_df = on_off_data_df.append(data_df)
#%%
os.chdir(saving_folder_location)
on_off_data_df.to_csv(path_or_buf='910_on_off_m_flow_fosof_data.csv')
#%%
on_off_data_df = pd.read_csv('910_on_off_m_flow_fosof_data.csv', index_col=[0])
#%%
on_off_data_df
#%%
#on_off_data_df = on_off_data_df.reset_index().rename(columns={'level_0': 'Experiment Name'}).set_index(['Experiment Name']).sort_index()

# pi-configuration data has opposite slope. It should have opposite frequency shift. To make the frequency shift of the same sign, the phase differences are multiplied by -1.
pi_config_index = on_off_data_df[on_off_data_df ['Configuration'] == 'pi'].index
on_off_data_df.loc[pi_config_index, 'Weighted Mean'] = on_off_data_df.loc[pi_config_index, 'Weighted Mean'] * (-1)

# Expanding the error bars for the data sets with larger than 1 reduced chi squared

chi_squared_large_index = on_off_data_df[on_off_data_df['Reduced Chi Squared'] > 1].index

on_off_data_df.loc[chi_squared_large_index, 'Weighted STD'] = on_off_data_df.loc[chi_squared_large_index, 'Weighted STD'] * np.sqrt(on_off_data_df.loc[chi_squared_large_index, 'Reduced Chi Squared'])

on_off_data_df = on_off_data_df.reset_index().set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])
#%%
on_off_data_df
#%%
# Join the averaged slopes. The pre-910 on/off switching data was acquired only for 49.86 kV accelerating voltage at 120 V proton deflector voltage with +-2MHz range around 910MHz.
on_off_data_df = on_off_data_df.join(slope_df.loc[(slice(None), 49.86, 120, slice(None), 1), slice(None)].reset_index().set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD'].loc[on_off_data_df.index.drop_duplicates()]['Normalized']['Slope [Rad/MHz]'])

# Calculate the frequency shift
on_off_data_df['Frequency Shift [MHz]'] = -on_off_data_df['Weighted Mean'] / on_off_data_df['Slope [Rad/MHz]'] * (1-on_off_data_df['F=0 Fraction Quenched'])

on_off_data_df['Frequency Shift STD [MHz]'] = -on_off_data_df['Weighted STD'] / on_off_data_df['Slope [Rad/MHz]'] * (1-on_off_data_df['F=0 Fraction Quenched'])

# Determine average frequency shift for each combination of the waveguide separation, accelerating voltage and the electric field amplitude.
on_off_data_grouped = on_off_data_df.groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]'])

on_off_av_df = on_off_data_grouped.apply(lambda df: straight_line_fit_params(df['Frequency Shift [MHz]'], df['Frequency Shift STD [MHz]']))

# Normalize the averaged data to reduced chi-squared of 1.

on_off_av_large_chi_squared_index = on_off_av_df[on_off_av_df['Reduced Chi Squared'] > 1].index
on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Weighted STD'] = on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Weighted STD'] * np.sqrt(on_off_av_df.loc[on_off_av_large_chi_squared_index, 'Reduced Chi Squared'])

# Average shift in kHz
on_off_av_df = on_off_av_df[['Weighted Mean', 'Weighted STD']] * 1E3
on_off_av_df.rename(columns={'Weighted Mean': 'Frequency Shift [kHz]', 'Weighted STD': 'Frequency Shift STD [kHz]'}, inplace=True)
