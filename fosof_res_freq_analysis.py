'''
2018-11-10

In this script I am correcting zero-crossings from the FOSOF data for systematic effects. It is possible that this is the final analysis that is performed on the data.

Each zero-crossing needs to be corrected for several systematic effects:

1. Second-order Doppler shift (SOD). Depends on the accelerating voltage only.

2. AC Stark shift. Depends on the waveguide separation, accelerating voltage, waveguide RF power, beam rms radius, quench offset.

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

#
from exp_data_analysis import *
#import fosof_data_set_analysis
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

#import wvg_power_calib_analysis
#%%
# Load the zero-crossing frequencies
# For the lab
saving_folder_location = 'D:/2017-10-17 Lamb Shift Measurement/Data/FOSOF analyzed data sets'

#saving_folder_location = 'E:/2017-10-17 Lamb Shift Measurement/Data/FOSOF analyzed data sets'

fosof_lineshape_param_file_name = 'fosof_lineshape_param.csv'

os.chdir(saving_folder_location)

grouped_exp_fosof_lineshape_param_df = pd.read_csv(filepath_or_buffer=fosof_lineshape_param_file_name, delimiter=',', comment='#', header=[0, 1, 2, 3, 4], skip_blank_lines=True, index_col=[0, 1, 2, 3, 4, 5, 6, 7])

#%%
''' Correction for AC shift, Fractional Offset, and Beam profile.

We want to determine the effect of the fractional offset on the zero-crossing frequencies. For this, to be independent of the calibration data analysis I determine the effect of the fractional offset on the simulated quench curves for given beam speed, and the beam profile. However, it is so that the beam profile has no significant effect on the simulated quench curve, and thus it can be ignored. I am simply using the simulation obtained for the off-axis distance that is the closest to the beam rms radius determined using Monte Carlo simulation.

The reason we need to look at the quench curves is to understand by how much the RF power that we thought we had for given surviving fraction changes when the fractional offset is different.

Now, this effect on the quench curve, in principle, should be determined for every RF frequency, such that one can later determine the effect on each corresponding FOSOF phase for every data set and correct the data accordingly. However, it turns out that all of the frequencies, for which the FOSOF data was acquired, get affected by the same amount. I.e., at each RF frequency for given RF power, the shift in the power for given fractional offset is essentially the same for all frequencies. This allows us to simply construct the FOSOF lineshape that has the same power for all of the frequencies. We do not have to worry about having different power change at different frequencies, which would complicate the analysis.

Notice that the simulated FOSOF lineshapes are sensitive to different beam profiles. Since we do not know the beam profile well, we use the beam RMS radius determined from the Monte Carlo simulation, and assign 50% uncertainty to it. For generality, for each beam RMS radius, I construct the interpolated FOSOF lineshapes, using the simulated data, and determine the AC shift for each of the fractional offsets.
'''

# This is the fractional offset that is our best estimate
fract_offset_best_est = 0.03

# Array of different possible quench offsets. We assign 50% uncertainty to our best estimate. I also include the fractional offset of 0 (for no real reason).
fract_quench_offset_arr = np.array([0, 0.03, fract_offset_best_est*1.5, fract_offset_best_est*0.5])

# The trickiest correction to apply is the AC shift correction. We first create an index that has all of the experiment parameters that are important for this shift
exp_params_AC_shift_df = grouped_exp_fosof_lineshape_param_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD'].reset_index()[['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Beam RMS Radius [mm]']].drop_duplicates()

exp_params_AC_shift_df = exp_params_AC_shift_df[exp_params_AC_shift_df['Beam RMS Radius [mm]'] != -1].drop_duplicates()

# Beam speed to use for the simulation [cm/ns]. All of the waveguide calibration data that is used for the experiments was acquired at 49.87 kV of accelerating voltage. This was later measured to correspond to roughly 0.3223 cm/ns.
v_speed = 0.3223

# Load power scan simulation data that corresponds to the particular beam speed.
old_sim_info_df = hydrogen_sim_data.OldSimInfo().get_info()

old_quench_sim_info_df = old_sim_info_df[old_sim_info_df['Waveguide Electric Field [V/cm]'] == 'Power Scan']

old_quench_sim_info_df = old_quench_sim_info_df[old_quench_sim_info_df['Speed [cm/ns]'] == v_speed]

# In general there are several available simulations. Pick one of them.
# Waveguide Quenching simulation to use
old_quench_sim_info_s = old_quench_sim_info_df.iloc[3]

# Use the simulation file to obtain the analyzed simulation class instance.
old_quench_sim_set = hydrogen_sim_data.WaveguideOldQuenchCurveSimulationSet(old_quench_sim_info_s)
sim_name_list = old_quench_sim_set.get_list_of_simulations()

# We use 1.8 mm off-axis simulation, because this is the RMS beam radius that is closes to the one determined by using Monte Carlo simulation using distances between and diameters of apertures.
quench_sim_vs_freq_df = old_quench_sim_set.get_simulation_data(sim_name_list[2])

# The quench simulations assume 910 MHz to be the resonant frequency. We pick this frequency for the quench curve for which we determine the effect of the fractional offset
rf_freq = 910

# Obtaining the simulated quench curves corrected for different fractional offsets.
quench_data_df_copy = quench_sim_vs_freq_df.loc[(rf_freq), (slice(None))].copy()

quench_data_vs_offset_dict = {}

for fract_quench_offset in fract_quench_offset_arr:

    # Transformation of the simulated quench data with no offset to the quench curve with the specified offset.
    quench_data_fract_offset_df = quench_data_df_copy*(1-fract_quench_offset) + fract_quench_offset

    quench_sim_data_set_offset = hydrogen_sim_data.OldWaveguideQuenchCurveSimulation(quench_data_fract_offset_df)
    quench_sim_data_set_offset.analyze_data()

    quench_data_vs_offset_dict[fract_quench_offset] = quench_sim_data_set_offset

# These are the electric field values for which that FOSOF data was acquired for the whole Lamb Shift experiment
rf_e_field_arr = exp_params_AC_shift_df['Waveguide Electric Field [V/cm]'].drop_duplicates().values

# To calculate the effect of the fractional offset on the RF power that we were suppose to use for the data sets, we first use the quench curve for our best estimate fractional offset and determine the surviving fractions at the field powers that were used in the experiments.
surv_frac_0_arr = quench_data_vs_offset_dict[0.03].get_surv_frac_with_poly_fit(rf_e_field_arr**2)

# For each of the fractional offsets we now want to find to what RF power this surviving fraction corresponds to. (we apply inverse operation here, in a sense)
e_field_vs_offset_dict = {}
for fract_quench_offset, quench_sim_data_set in quench_data_vs_offset_dict.items():
    e_field_vs_offset_dict[fract_quench_offset] = np.sqrt(quench_sim_data_set.get_RF_power_with_poly_fit(surv_frac_0_arr))

# We construct the dataframe that has the fractional offsets, and the electric fields that we (hopefully) had in the waveguides obtained by assuming having the best estimate fractional offset. For each fractional offset we have the related electric field that we actually had (if we indeed had this particular fractional offset)
rf_pow_vs_offset_df = pd.DataFrame(np.array(list(e_field_vs_offset_dict.values())), columns=rf_e_field_arr, index=list(e_field_vs_offset_dict.keys())).T

rf_pow_vs_offset_df.index.names = ['Electric Field [V/cm]']
rf_pow_vs_offset_df.columns.names = ['Fractional Offset']

# We have the set of sets of the experiment parameters used for the FOSOF experiments. For each of the set of the parameters we obtain the deviation of the zero-crossing frequency from the resonant frequency. The parameters include different beam rms radii. For each of the set of the parameters the frequency deviation is determined for the specified set of the fractional offsets.
fosof_lineshape_vs_offset_set_df = pd.DataFrame()

# These are the frequencies used to construct interpolated FOSOF lineshape for given RF power, accelerating voltage, waveguide separation, and off-axis distance, which is equivalent to the beam RMS radius. We do not need to specify many frequencies here. We should limit the range of frequencies to 908-912, so that we can use the NEW simulations with the proper phase averaging. Also, for the accelerations of around 16.67 kV, we do not have OLD simulation data at all.
freq_arr = np.array([908, 910, 912])

for index, data_s in exp_params_AC_shift_df.iterrows():
    print(index)
    data_s['Waveguide Separation [cm]']

    # Load the simulation data

    fosof_sim_data = hydrogen_sim_data.FOSOFSimulation(load_Q=True)

    # Use the appropriate simulation parameters and find the interpolated phase values

    sim_params_dict = { 'Frequency Array [MHz]': freq_arr,
                        'Waveguide Separation [cm]': data_s['Waveguide Separation [cm]'],
                        'Accelerating Voltage [kV]': data_s['Accelerating Voltage [kV]'],
                        'Off-axis Distance [mm]': data_s['Beam RMS Radius [mm]']}

    zero_crossing_vs_off_axis_dist_chosen_df, fosof_sim_info_chosen_df = fosof_sim_data.filter_fosof_sim_set(sim_params_dict, blind_freq_Q=False)

    phase_vs_e_field_poly_fit_df = fosof_sim_data.get_e_field_func()

    e_field_needed = data_s['Waveguide Electric Field [V/cm]']

    fosof_lineshape_vs_offset_dict = {}
    for fract_offset, e_field_to_use in rf_pow_vs_offset_df.loc[e_field_needed].items():

        # Determine interpolated simulation FOSOF lineshape parameters.
        zero_cross_params_s = fosof_sim_data.calc_interp_FOSOF_lineshape_params(e_field_to_use)
        fosof_lineshape_vs_offset_dict[fract_offset] = zero_cross_params_s

    fosof_lineshape_vs_offset_df = pd.DataFrame(fosof_lineshape_vs_offset_dict)
    fosof_lineshape_vs_offset_df.index.names = ['FOSOF Lineshape Fit Parameters']

    fosof_lineshape_vs_offset_df['Waveguide Separation [cm]'] = data_s['Waveguide Separation [cm]']
    fosof_lineshape_vs_offset_df['Accelerating Voltage [kV]'] = data_s['Accelerating Voltage [kV]']
    fosof_lineshape_vs_offset_df['Beam RMS Radius [mm]'] = data_s['Beam RMS Radius [mm]']
    fosof_lineshape_vs_offset_df['Waveguide Electric Field [V/cm]'] = data_s['Waveguide Electric Field [V/cm]']

    fosof_lineshape_vs_offset_set_df = fosof_lineshape_vs_offset_set_df.append(fosof_lineshape_vs_offset_df)

fosof_lineshape_vs_offset_set_df.set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Beam RMS Radius [mm]'], append=True, inplace=True)

fosof_lineshape_vs_offset_set_df.columns.names = ['Fractional Offset']

fosof_lineshape_vs_offset_set_df = fosof_lineshape_vs_offset_set_df.unstack(level='FOSOF Lineshape Fit Parameters').stack(level='Fractional Offset')

fosof_lineshape_vs_offset_set_df = fosof_lineshape_vs_offset_set_df.reset_index().set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Beam RMS Radius [mm]', 'Fractional Offset'])

fosof_lineshape_vs_offset_set_df = fosof_lineshape_vs_offset_set_df.astype(dtype={'Fractional Slope Deviation [ppt]': np.float64, 'Largest Absolute Residual [mrad]': np.float64, 'Resonant Frequency Offset [kHz]': np.float64, 'Slope [Rad/MHz]': np.float64, 'Resonant Frequency Deviation [kHz]': np.float64})
#%%
# We now want to calculate the deviation the resonant frequency determined for each fractional offset from that of the best estimate for the fractional offset. For each beam RMS radius this is done separately.

offset_shift_df = fosof_lineshape_vs_offset_set_df.reset_index('Fractional Offset')

# This is the frequency shift that we get due to different fractional offsets from the best estimate fractional offset.
offset_shift_df['Frequency Shift From Best Estimate Offset [kHz]'] = offset_shift_df['Resonant Frequency Offset [kHz]'] - offset_shift_df[offset_shift_df['Fractional Offset'] == fract_offset_best_est]['Resonant Frequency Offset [kHz]']

offset_shift_df.set_index('Fractional Offset', append=True, inplace=True)

# This calculated deviation from low and high limits for the fractional offset are taken as the + and - uncertainty in the resonant frequency due to the offset. These limits are not equal. However, we assume that both are equally likely. Thus we take the RMS uncertainty by combining the two deviations.

low_fract_offset = fract_offset_best_est * 0.5

high_fract_offset = fract_offset_best_est * 1.5

rms_fract_offset_unc_s = np.sqrt(((offset_shift_df.loc[(slice(None), slice(None), slice(None), slice(None), low_fract_offset), ('Frequency Shift From Best Estimate Offset [kHz]')].reset_index('Fractional Offset', drop=True))**2 + (offset_shift_df.loc[(slice(None), slice(None), slice(None), slice(None), high_fract_offset), ('Frequency Shift From Best Estimate Offset [kHz]')].reset_index('Fractional Offset', drop=True))**2) / 2)

# This is calculated for every beam RMS radius. We combine all of these RMS uncertainties for all of the beam radii and find their RMS to give the final uncertainty due to the fractional offset.

rms_fract_offset_unc_s = rms_fract_offset_unc_s.groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).aggregate(lambda x: np.sqrt(np.sum(x**2)/x.shape[0]))

rms_fract_offset_unc_s.name = 'Fractional Offset Uncertainty [kHz]'

rms_fract_offset_unc_df = pd.DataFrame(rms_fract_offset_unc_s)

#%%
# We now calculate the effect on the resonant frequency due to the beam RMS radius.

# Firstly, the resonant frequency deviations are calculated for each fractional offset. The deviation calculated for different beam rms radii for each fractional offset. The analysis is similar to the one for the fractional offset.

# Best estimate of the beam rms radius that was determined from the Monte Carlo simulation.
beam_rms_rad_best_est = 1.6

rms_rad_shift_df = fosof_lineshape_vs_offset_set_df.reset_index('Beam RMS Radius [mm]').sort_index()

# This is the frequency shift that we get due to different beam rms radii from the best estimate beam rms radius.
rms_rad_shift_df['Frequency Shift From Best Estimate Beam RMS Radius [kHz]'] = rms_rad_shift_df['Resonant Frequency Offset [kHz]'] - rms_rad_shift_df[rms_rad_shift_df['Beam RMS Radius [mm]'] == beam_rms_rad_best_est]['Resonant Frequency Offset [kHz]']

rms_rad_shift_df = rms_rad_shift_df.set_index(['Beam RMS Radius [mm]'], append=True).sort_index()

small_beam_rms_rad = beam_rms_rad_best_est * 0.5

large_beam_rms_rad = round(beam_rms_rad_best_est * 1.5, 3)

# RMS uncertainty due to the max and min limits in the beam RMS radii for each fractional offset.

rms_beam_rms_rad_unc_s = np.sqrt(((rms_rad_shift_df.loc[(slice(None), slice(None), slice(None), slice(None), small_beam_rms_rad), ('Frequency Shift From Best Estimate Beam RMS Radius [kHz]')].reset_index('Beam RMS Radius [mm]', drop=True))**2 + (rms_rad_shift_df.loc[(slice(None), slice(None), slice(None), slice(None), large_beam_rms_rad), ('Frequency Shift From Best Estimate Beam RMS Radius [kHz]')].reset_index('Beam RMS Radius [mm]', drop=True))**2) / 2)

rms_beam_rms_rad_unc_s = rms_beam_rms_rad_unc_s.groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]']).aggregate(lambda x: np.sqrt(np.sum(x**2)/x.shape[0]))

rms_beam_rms_rad_unc_s.name = 'Beam RMS Radius Uncertainty [kHz]'
rms_beam_rms_rad_unc_df = pd.DataFrame(rms_beam_rms_rad_unc_s)
#%%
# Calculation of the resonant frequency shift due to RF power. This calculation is performed for the best estimate for the beam rms radius and the fractional offset

# Uncertainty in the simulation
simul_unc = 0.05
add_unc = 0.00

field_power_shift_unc = simul_unc + add_unc

# The Field power shift
field_power_shift_s = fosof_lineshape_vs_offset_set_df.loc[(slice(None), slice(None), slice(None), beam_rms_rad_best_est, fract_offset_best_est), ('Resonant Frequency Offset [kHz]')].reset_index(['Beam RMS Radius [mm]', 'Fractional Offset'], drop=True)

field_power_shift_s.name = 'Field Power Shift [kHz]'

# The uncertainty in the shift
field_power_shift_unc_s = field_power_shift_s * field_power_shift_unc

field_power_shift_unc_s.name = 'Field Power Shift Uncertainty [kHz]'

field_power_shift_df = pd.DataFrame(field_power_shift_s)
field_power_shift_unc_df = pd.DataFrame(field_power_shift_unc_s)

ac_shift_df = field_power_shift_unc_df.join([field_power_shift_df, rms_beam_rms_rad_unc_df, rms_fract_offset_unc_df])

ac_shift_df = ac_shift_df * 1E-3

ac_shift_df.rename(columns={'Field Power Shift Uncertainty [kHz]': 'Field Power Shift Uncertainty [MHz]', 'Field Power Shift [kHz]': 'Field Power Shift [MHz]', 'Beam RMS Radius Uncertainty [kHz]': 'Beam RMS Radius Uncertainty [MHz]', 'Fractional Offset Uncertainty [kHz]': 'Fractional Offset Uncertainty [MHz]'}, inplace=True)

# The interpolation routine for extracting the FOSOF lineshape is not perfect. We have an RMS error of 0.41 kHz (and the maximum error of 0.85 kHz) in determining the resonant frequency by using this technique as opposed to calculating the resonant frequency of the simulated data directly. This error gets added to the AC shift list of errors. The reason the interpolation does not give the perfect agreement is probably, because, the field power shift is not dependent on the 2-order polynomial exactly, but it has some local maxima and minima. It might be also due to numerical uncertainty in the simulation, but since I do not know, I will simply assume that this is the independent uncertainty.

rms_sim_interp_unc = 0.41 * 1E-3

ac_shift_df['Interpolation Uncertainty [MHz]'] = rms_sim_interp_unc

# Calculating the total uncertainty in the AC shift.
ac_shift_df['AC Shift Uncertainty [MHz]'] = np.sqrt(ac_shift_df['Field Power Shift Uncertainty [MHz]']**2 + ac_shift_df['Beam RMS Radius Uncertainty [MHz]']**2 + ac_shift_df['Fractional Offset Uncertainty [MHz]']**2 + ac_shift_df['Interpolation Uncertainty [MHz]']**2)

#%%
''' Correction for the Second-order Doppler Shift (SOD)
'''

# These are the speeds that were experimentally measured
beam_speed_data_df = pd.DataFrame(np.array([[4, 22.17, 0.225397, 0.00227273], [4, 16.27, 0.195149, 0.00285072], [4, 49.86, 0.326999, 0.00344062], [5, 49.86, 0.319996, 0.00332496], [7, 49.86, 0.322847, 0.00332746]]), columns=['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Beam Speed [cm/ns]', 'Beam Speed STD [cm/ns]']).set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]'])

# While taking data for different separation, for the same accelerating voltages, it is true that we are not keeping all of the Source parameters (=voltages) the same all the time. The spread in the values that we got for the beam speeds is the good indicator of the variability of the Source parameters that were used for the experiment. Thus the average of these values gives us the best estimate for the speed. The STDOM of the spread is added with quadruate to the RMS uncertainty in the speed values to give us the average uncertainty in the average beam speed.
def get_av_speed(df):

    if df.shape[0] > 1:
        av_s = df[['Beam Speed [cm/ns]']].aggregate(lambda x: np.mean(x))
        av_s['Beam Speed STD [cm/ns]'] = np.sqrt((np.std(df['Beam Speed [cm/ns]'], ddof=1)/np.sqrt(df['Beam Speed STD [cm/ns]'].shape[0]))**2 + np.sqrt(np.sum(df['Beam Speed STD [cm/ns]']**2/df['Beam Speed STD [cm/ns]'].shape[0]))**2)
    else:
        av_s = df.iloc[0]
    return av_s

beam_speed_df = beam_speed_data_df.groupby('Accelerating Voltage [kV]').apply(get_av_speed)

beam_speed_df = beam_speed_data_df.reset_index('Waveguide Separation [cm]').join(beam_speed_df, lsuffix='_Delete').drop(columns=['Beam Speed [cm/ns]_Delete', 'Beam Speed STD [cm/ns]_Delete'])

# We add the 6 cm and 49.86 kV point to the dataframe of beam speeds.

wvg_sep_6_s = beam_speed_df.loc[49.86].iloc[0].copy()
wvg_sep_6_s['Waveguide Separation [cm]'] = 6

beam_speed_df = beam_speed_df.append(wvg_sep_6_s)

# Calculate the SOD
# Assumed resonant frequency [MHz]
freq_diff = 909.894
# Speed of light [m/s]
c_light = 299792458

sod_shift_df = beam_speed_df.copy()
sod_shift_df['SOD Shift [MHz]'] = (1/np.sqrt(1-(beam_speed_df['Beam Speed [cm/ns]'] * 1E-2 * 1E9/c_light)**2) - 1) * freq_diff

sod_shift_df['SOD Shift STD [MHz]'] = freq_diff * beam_speed_df['Beam Speed [cm/ns]'] * 1E-2 * 1E9 * beam_speed_df['Beam Speed STD [cm/ns]'] * 1E-2 * 1E9 / ((1 - (beam_speed_df['Beam Speed [cm/ns]']/c_light)**2)**(1.5) * c_light**2)

sod_shift_df = sod_shift_df.set_index('Waveguide Separation [cm]', append=True).swaplevel(0, 1)
#%%
''' Statistical analysis of the lineshape data
'''

# In case if a lineshape has reduced chi-squared of larger than 1 then we expand the uncertainty in the fit parameters to make the chi-squared to be one.

# Addidng the columns of normalized = corrected for large reduced chi-squared, uncertainties in the fit parameters.
fosof_lineshape_param_df = grouped_exp_fosof_lineshape_param_df.join(grouped_exp_fosof_lineshape_param_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Zero-crossing Frequency STD [MHz]', 'Slope STD [Rad/MHz]', 'Offset STD [MHz]'])].rename(columns={'Zero-crossing Frequency STD [MHz]': 'Zero-crossing Frequency STD (Normalized) [MHz]', 'Slope STD [Rad/MHz]': 'Slope STD (Normalized) [Rad/MHz]', 'Offset STD [MHz]': 'Offset STD (Normalized) [MHz]'})).sort_index(axis='columns')

# We now adjust the uncertainties for getting the reduced chi-squared of at least 1.
def normalize_std(df):

    ref_type = df.columns.get_level_values('Phase Reference Type')[0]
    harm_type = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]

    large_chi_squared_index = df[df.loc[slice(None), (ref_type, harm_type, av_type, std_type, 'Reduced Chi-Squared')] > 1].index

    df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Zero-crossing Frequency STD (Normalized) [MHz]')] = df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Zero-crossing Frequency STD [MHz]')] * np.sqrt(df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Reduced Chi-Squared')])

    df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Slope STD (Normalized) [Rad/MHz]')] = df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Slope STD [Rad/MHz]')] * np.sqrt(df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Reduced Chi-Squared')])

    df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Offset STD (Normalized) [MHz]')] = df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Offset STD [MHz]')] * np.sqrt(df.loc[large_chi_squared_index, (ref_type, harm_type, av_type, std_type, 'Reduced Chi-Squared')])

    return df

fosof_lineshape_param_norm_df = fosof_lineshape_param_df.groupby(level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns').apply(normalize_std)
#%%
# Statistaical averaging of the FOSOF data.

def calc_av_freq_for_analysis_type(df):

    ref_type = df.columns.get_level_values('Phase Reference Type')[0]
    harm_type = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]

    df = df[ref_type, harm_type, av_type, std_type]

    data_arr = df['Zero-crossing Frequency [MHz]']
    data_std_arr = df['Zero-crossing Frequency STD [MHz]']

    av_s = straight_line_fit_params(data_arr, data_std_arr)

    av_s['Weighted STD (Normalized)'] = av_s['Weighted STD']
    if av_s['Reduced Chi Squared'] > 1:
        av_s['Weighted STD (Normalized)'] = av_s['Weighted STD'] * np.sqrt(av_s['Reduced Chi Squared'])

    av_s.name = 'Not Normalized'

    data_arr = df['Zero-crossing Frequency [MHz]']
    data_std_arr = df['Zero-crossing Frequency STD (Normalized) [MHz]']

    av_norm_s = straight_line_fit_params(data_arr, data_std_arr)

    av_norm_s['Weighted STD (Normalized)'] = av_norm_s['Weighted STD']
    if av_norm_s['Reduced Chi Squared'] > 1:
        av_norm_s['Weighted STD (Normalized)'] = av_norm_s['Weighted STD'] * np.sqrt(av_norm_s['Reduced Chi Squared'])

    av_norm_s.name = 'Normalized'

    return pd.DataFrame(av_s).join(pd.DataFrame(av_norm_s))

def calc_av_zero_cross_freq(group_df):

    return group_df.groupby(level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns').apply(calc_av_freq_for_analysis_type)


def calc_av_slope_for_analysis_type(df):

    ref_type = df.columns.get_level_values('Phase Reference Type')[0]
    harm_type = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]

    df = df[ref_type, harm_type, av_type, std_type]

    data_arr = df['Slope [Rad/MHz]']
    data_std_arr = df['Slope STD [Rad/MHz]']

    av_s = straight_line_fit_params(data_arr, data_std_arr)

    av_s['Weighted STD (Normalized)'] = av_s['Weighted STD']
    if av_s['Reduced Chi Squared'] > 1:
        av_s['Weighted STD (Normalized)'] = av_s['Weighted STD'] * np.sqrt(av_s['Reduced Chi Squared'])

    av_s.name = 'Not Normalized'

    data_arr = df['Slope [Rad/MHz]']
    data_std_arr = df['Slope STD (Normalized) [Rad/MHz]']

    av_norm_s = straight_line_fit_params(data_arr, data_std_arr)

    av_norm_s['Weighted STD (Normalized)'] = av_norm_s['Weighted STD']
    if av_norm_s['Reduced Chi Squared'] > 1:
        av_norm_s['Weighted STD (Normalized)'] = av_norm_s['Weighted STD'] * np.sqrt(av_norm_s['Reduced Chi Squared'])

    av_norm_s.name = 'Normalized'

    return pd.DataFrame(av_s).join(pd.DataFrame(av_norm_s))

def calc_av_slope(group_df):

    return group_df.groupby(level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type'], axis='columns').apply(calc_av_slope_for_analysis_type)

# First we average the data for given waveguide separation, accelerating voltage, Proton deflector voltage, RF E field amplitude, and the frequency range multiple. We only look at the data for the best estimate of the beam RMS radius, because the uncertainty due to this parameter is already included in the AC shift. We also look at the waveguide carrier frequency sweep-type experiments only.

zero_cross_av_df = fosof_lineshape_param_norm_df.loc[('Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est), (slice(None))].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple']).apply(calc_av_zero_cross_freq).unstack(level=-1)

zero_cross_av_df.columns.names = ['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

zero_cross_av_df.rename(columns={'Weighted Mean': 'Zero-crossing Frequency [MHz]', 'Weighted STD': 'Zero-crossing Frequency STD [MHz]', 'Weighted STD (Normalized)': 'Zero-crossing Frequency STD (Normalized) [MHz]'}, level='Data Field', inplace=True)

slope_av_df = fosof_lineshape_param_norm_df.loc[('Waveguide Carrier Frequency Sweep', beam_rms_rad_best_est), (slice(None))].groupby(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple']).apply(calc_av_slope).unstack(level=-1)

slope_av_df.columns.names = ['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State', 'Data Field']

slope_av_df.rename(columns={'Weighted Mean': 'Slope [Rad/MHz]', 'Weighted STD': 'Slope STD [Rad/MHz]', 'Weighted STD (Normalized)': 'Slope STD (Normalized) [Rad/MHz]'}, level='Data Field', inplace=True)

# There are some data sets that were acquired with the PD turned off. We do not want to include them into the analysis.
zero_cross_av_no_pd_df = zero_cross_av_df.loc[zero_cross_av_df.loc[(slice(None), slice(None), 0), (slice(None))].index]

zero_cross_av_df = zero_cross_av_df.loc[zero_cross_av_df.index.difference(zero_cross_av_no_pd_df.index)]

slope_av_no_pd_df = slope_av_df.loc[slope_av_df.loc[(slice(None), slice(None), 0), (slice(None))].index]

slope_av_df = slope_av_df.loc[slope_av_df.index.difference(slope_av_no_pd_df.index)]
#%%
''' Determining the average zero-crossing frequency using both of the RF Combiners, the RMS STD, and the systematic uncertainty due to the difference in the frequencies obtained with the two combiners.

In principle, it is better to perform this averaging not on the zero-crossings, but on the FOSOF phases for each of the 0-pi data sets. This is explained on p 36-53 in Lab Notes #4 written on August 31, 2018. However, our method, of analyzing the lineshapes for different combiners separately, should be sufficient. In this case, the only downside, is that the systematic shift due to combiners will be somewhat larger.
'''
rf_combiner_I_zero_cross_df = zero_cross_av_df['RF Combiner I Reference'].loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]'])]

rf_combiner_R_zero_cross_df = zero_cross_av_df['RF Combiner R Reference'].loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Zero-crossing Frequency [MHz]', 'Zero-crossing Frequency STD [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]'])]

zero_cross_freq_df = zero_cross_av_df['RF Combiner I Reference'].copy()

zero_cross_freq_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')] = (rf_combiner_I_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')]  + rf_combiner_R_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')] ) / 2

zero_cross_freq_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD [MHz]')] = np.sqrt((rf_combiner_I_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD [MHz]')]**2  + rf_combiner_R_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD [MHz]')]**2 ) / 2)

zero_cross_freq_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD (Normalized) [MHz]')] = np.sqrt((rf_combiner_I_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD (Normalized) [MHz]')]**2  + rf_combiner_R_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency STD (Normalized) [MHz]')]**2 ) / 2)

zero_cross_freq_df = zero_cross_freq_df.join(np.abs(rf_combiner_I_zero_cross_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')] - zero_cross_freq_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')]).rename(columns={'Zero-crossing Frequency [MHz]': 'Combiner Uncertainty [MHz]'}, level='Data Field')).sort_index(axis='columns')

#%%
# Performing similar steps for the FOSOF slopes.
rf_combiner_I_slope_df = slope_av_df['RF Combiner I Reference'].loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Slope [Rad/MHz]', 'Slope STD [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]'])]

rf_combiner_R_slope_df = slope_av_df['RF Combiner R Reference'].loc[slice(None), (slice(None), slice(None), slice(None), slice(None), ['Slope [Rad/MHz]', 'Slope STD [Rad/MHz]', 'Slope STD (Normalized) [Rad/MHz]'])]

slope_df = slope_av_df['RF Combiner I Reference'].copy()

slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')] = (rf_combiner_I_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')]  + rf_combiner_R_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')] ) / 2

slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD [Rad/MHz]')] = np.sqrt((rf_combiner_I_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD [Rad/MHz]')]**2  + rf_combiner_R_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD [Rad/MHz]')]**2 ) / 2)

slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD (Normalized) [Rad/MHz]')] = np.sqrt((rf_combiner_I_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD (Normalized) [Rad/MHz]')]**2  + rf_combiner_R_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope STD (Normalized) [Rad/MHz]')]**2 ) / 2)

slope_df = slope_df.join(np.abs(rf_combiner_I_slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')] - slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')]).rename(columns={'Slope [Rad/MHz]': 'Combiner Uncertainty [Rad/MHz]'}, level='Data Field')).sort_index(axis='columns')
#%%
''' Shift due to imperfect phase control upon waveguide reversal. This shift was measured to be about delta_phi = 0.2 mrad. In frequency units it corresponds to the frequency shift of delta_phi / slope. Instead of correcting the data, we add this as the additional type of the uncertainty.
'''

# Shift due to imperfect phase control [Rad]
delta_phi = 0.2 * 1E-3

phase_control_unc_df = np.abs(delta_phi / slope_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Slope [Rad/MHz]')]).rename(columns={'Slope [Rad/MHz]': 'Phase Control Uncertainty [MHz]'}).copy()

#%%
# We now want to determine the (blinded) resonant frequencies = zero-crossing frequencies corrected for the systematic shifts and also include all of the systematic uncertainties for each of the determined resonant frequencies.

# Adding the columns to the specific level that will store the systematic shifts
col_name_list = list(ac_shift_df.columns.union(sod_shift_df.columns))
col_name_list.append('Resonant Frequency (Blinded) [MHz]')
for col_name in col_name_list:

    zero_cross_freq_df = zero_cross_freq_df.join(zero_cross_freq_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Zero-crossing Frequency [MHz]')].rename(columns={'Zero-crossing Frequency [MHz]': col_name}, level='Data Field')).sort_index(axis='columns')

# Addidng the uncertainty due to imperfect phase control
zero_cross_freq_df = zero_cross_freq_df.join(phase_control_unc_df).sort_index(axis='columns')

def correct_for_sys_shift(df):
    ''' Corrects the zero-crossing frequencies for the systematic shifts, and assign the respective systematic shift uncertainties.
    '''
    harm_type = df.columns.get_level_values('Fourier Harmonic')[0]
    av_type = df.columns.get_level_values('Averaging Type')[0]
    std_type = df.columns.get_level_values('STD Type')[0]
    std_state = df.columns.get_level_values('STD State')[0]

    df = df.reset_index(['Frequency Range Multiple', 'Proton Deflector Voltage [V]'])

    df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] = df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] - ac_shift_df['Field Power Shift [MHz]']

    for col_name in ac_shift_df.columns:
        df[harm_type, av_type, std_type, std_state, col_name] = ac_shift_df[col_name]

    df = df.reset_index('Waveguide Electric Field [V/cm]')

    df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] = df[harm_type, av_type, std_type, std_state, 'Resonant Frequency (Blinded) [MHz]'] + sod_shift_df['SOD Shift [MHz]']

    for col_name in sod_shift_df.columns:
        df[harm_type, av_type, std_type, std_state, col_name] = sod_shift_df[col_name]

    df.set_index(['Proton Deflector Voltage [V]', 'Waveguide Electric Field [V/cm]', 'Frequency Range Multiple'], append=True, inplace=True)

    return df

# List of corrected frequencies (blinded) with the respective systematic and statistical uncertainties included.
res_freq_df = zero_cross_freq_df.groupby(level=['Fourier Harmonic', 'Averaging Type', 'STD Type', 'STD State'], axis='columns').apply(correct_for_sys_shift)
#%%
res_freq_df
#%%
df = res_freq_df['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized']

df = df.loc[(slice(None), slice(None), slice(None), slice(None), 1), (['AC Shift Uncertainty [MHz]', 'Combiner Uncertainty [MHz]', 'SOD Shift STD [MHz]', 'Phase Control Uncertainty [MHz]', 'Zero-crossing Frequency STD (Normalized) [MHz]', 'Resonant Frequency (Blinded) [MHz]'])]

df['Total Uncertainty [MHz]'] = np.sqrt(df['AC Shift Uncertainty [MHz]']**2 + df['Combiner Uncertainty [MHz]']**2 + df['SOD Shift STD [MHz]']**2 + df['Zero-crossing Frequency STD (Normalized) [MHz]']**2 + df['Phase Control Uncertainty [MHz]']**2)

df['Systematic Uncertainty [MHz]'] = np.sqrt(df['AC Shift Uncertainty [MHz]']**2 + df['Combiner Uncertainty [MHz]']**2 + df['SOD Shift STD [MHz]']**2 + df['Phase Control Uncertainty [MHz]']**2)
#%%
s4 = straight_line_fit_params(df.loc[(4) ,('Resonant Frequency (Blinded) [MHz]')], df.loc[(4) ,('Total Uncertainty [MHz]')])
#%%
s5 = straight_line_fit_params(df.loc[(5) ,('Resonant Frequency (Blinded) [MHz]')], df.loc[(5) ,('Total Uncertainty [MHz]')])
#%%
s6 = straight_line_fit_params(df.loc[(6) ,('Resonant Frequency (Blinded) [MHz]')], df.loc[(6) ,('Total Uncertainty [MHz]')])
#%%
s7 = straight_line_fit_params(df.loc[(7) ,('Resonant Frequency (Blinded) [MHz]')], df.loc[(7) ,('Total Uncertainty [MHz]')])
#%%
final_freq = (1/s4['Weighted STD']**2 * s4['Weighted Mean'] + 1/s5['Weighted STD']**2 * s5['Weighted Mean'] + 1/s6['Weighted STD']**2 * s6['Weighted Mean'] + 1/s7['Weighted STD']**2 * s7['Weighted Mean']) / (1/s4['Weighted STD']**2 + 1/s5['Weighted STD']**2 + 1/s6['Weighted STD']**2 + 1/s7['Weighted STD']**2)
#%%
final_freq_std = np.sqrt((s4['Weighted STD']**2 + s5['Weighted STD']**2 + s6['Weighted STD']**2 + s7['Weighted STD']**2)/4)
#%%
f0_2 = final_freq + 0.03174024731
#%%
final_freq_std
#%%
f0_2
#%%
df = res_freq_df['First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD', 'Normalized']

def minimize_unc(w_arr, df):
    w_sum = np.sum(w_arr)
    beam_rms_rad_unc_tot = np.sum(df['Beam RMS Radius Uncertainty [MHz]'] * w_arr) / w_sum
    field_power_shift_unc_tot = np.sum(df['Field Power Shift Uncertainty [MHz]'] * w_arr) / w_sum
    fract_offset_unc_tot = np.sum(df['Fractional Offset Uncertainty [MHz]'] * w_arr) / w_sum
    interp_unc_tot = np.sum(df['Interpolation Uncertainty [MHz]'] * w_arr) / w_sum
    comb_unc_tot = np.sum(df['Combiner Uncertainty [MHz]'] * w_arr) / w_sum
    sod_unc_tot = np.sum(df['SOD Shift STD [MHz]'] * w_arr) / w_sum
    phase_control_unc_tot = np.sum(df['Phase Control Uncertainty [MHz]'] * w_arr) / w_sum

    zero_cross_stat_unc_tot = np.sqrt(np.sum((w_arr*df['Zero-crossing Frequency STD (Normalized) [MHz]'])**2) / w_sum**2)

    tot_unc = np.sqrt(beam_rms_rad_unc_tot**2 + field_power_shift_unc_tot**2 + fract_offset_unc_tot**2 + interp_unc_tot**2 + comb_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)

    return tot_unc

def w_tot_constraint(w_arr):
    return np.sum(w_arr) - 1

def find_unc_weights(df):
    w_arr = np.linspace(1, 10, df.shape[0])
    w_arr = w_arr/np.sum(w_arr)

    bounds_to_use_list = [(0, 1) for i in range(w_arr.shape[0])]

    w_min_arr = scipy.optimize.minimize(fun=minimize_unc, x0=w_arr, args=(df), bounds=bounds_to_use_list, constraints={'type': 'eq', 'fun': w_tot_constraint}, tol=1E-5)

    df['Weight'] = w_min_arr['x']

    return df

df_with_weights = df.loc[(slice(None), slice(None), slice(None), slice(None), 1), (slice(None))].groupby('Waveguide Separation [cm]').apply(find_unc_weights)

df_with_weights['Weight'] = df_with_weights['Weight'] / df_with_weights.index.get_level_values('Waveguide Separation [cm]').drop_duplicates().shape[0]

weight_arr_sum = df_with_weights['Weight'].sum()

beam_rms_rad_unc_tot = np.sum(df_with_weights['Beam RMS Radius Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

field_power_shift_unc_tot = np.sum(df_with_weights['Field Power Shift Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

fract_offset_unc_tot = np.sum(df_with_weights['Fractional Offset Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

interp_unc_tot = np.sum(df_with_weights['Interpolation Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

comb_unc_tot = np.sum(df_with_weights['Combiner Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

sod_unc_tot = np.sum(df_with_weights['SOD Shift STD [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

phase_control_unc_tot = np.sum(df_with_weights['Phase Control Uncertainty [MHz]'] * df_with_weights['Weight']) / weight_arr_sum

zero_cross_stat_unc_tot = np.sqrt(np.sum((df_with_weights['Weight']*df_with_weights['Zero-crossing Frequency STD (Normalized) [MHz]'])**2) / weight_arr_sum**2)

tot_unc = np.sqrt(beam_rms_rad_unc_tot**2 + field_power_shift_unc_tot**2 + fract_offset_unc_tot**2 + interp_unc_tot**2 + comb_unc_tot**2 + sod_unc_tot**2 + phase_control_unc_tot**2 + zero_cross_stat_unc_tot**2)
#%%
phase_control_unc_tot
#%%
zero_cross_stat_unc_tot
#%%
freq = np.sum(df_with_weights['Resonant Frequency (Blinded) [MHz]'] * df_with_weights['Weight']) / weight_arr_sum
#%%
tot_unc
#%%
f0 = freq + 0.03174024731
f0
#%%
f0-909.8717
#%%
f0_2
#%%
#%%
zero_cross_av_2_df['Resonant Frequency [MHz]'] = zero_cross_av_2_df['Zero-crossing Frequency [MHz]'] - zero_cross_av_2_df['Field Power Shift [MHz]'] + zero_cross_av_2_df['SOD Shift [MHz]']

zero_cross_av_2_df['Resonant Frequency STD [MHz]'] = np.sqrt(zero_cross_av_2_df['Zero-crossing Frequency STD (Normalized) [MHz]']**2 + zero_cross_av_2_df['Field Power Shift Uncertainty [MHz]']**2 + zero_cross_av_2_df['Beam RMS Radius Uncertainty [MHz]']**2 + zero_cross_av_2_df['Fractional Offset Uncertainty [MHz]']**2 + zero_cross_av_2_df['SOD Shift STD [MHz]']**2)
#%%
zero_cross_av_2_df.reset_index().plot(style='.', y='Resonant Frequency [MHz]', yerr='Resonant Frequency STD [MHz]')
#%%
def calc_line_fit_params(x_data_arr, y_data_arr, y_sigma_arr):
    ''' Fits the data to the first-order polynomial. Extracts the slope, offset, and the associated uncertainties. Gives the reduced chi-squared parameter.
    '''
    w_arr = 1/y_sigma_arr**2

    delta_arr = np.sum(w_arr) * np.sum(w_arr*x_data_arr**2) - (np.sum(w_arr*x_data_arr))**2

    offset = (np.sum(w_arr*y_data_arr) * np.sum(w_arr*x_data_arr**2) - np.sum(w_arr*x_data_arr*y_data_arr) * np.sum(w_arr*x_data_arr)) / delta_arr

    offset_sigma = np.sqrt(np.sum(w_arr*x_data_arr**2) / delta_arr)

    slope = (np.sum(w_arr*x_data_arr*y_data_arr) * np.sum(w_arr) - np.sum(w_arr*x_data_arr) * np.sum(w_arr*y_data_arr)) / delta_arr

    slope_sigma = np.sqrt(np.sum(w_arr) / delta_arr)

    fit_param_dict = {'Slope [Rad/MHz]': slope, 'Slope STD [Rad/MHz]': slope_sigma, 'Offset [MHz]': offset, 'Offset STD [MHz]': offset_sigma}

    # For the chi-squared determination.
    fit_data_arr = slope * x_data_arr + offset
    n_constraints = 2
    fit_param_dict = {**fit_param_dict, **get_chi_squared(y_data_arr, y_sigma_arr, fit_data_arr, n_constraints)}

    return fit_param_dict
#%%
data_df = slope_av_2_df[['Slope [Rad/MHz]','Slope STD (Normalized) [Rad/MHz]']].join(zero_cross_av_2_df[['Resonant Frequency [MHz]', 'Resonant Frequency STD [MHz]']])
#%%
data_df['Inverse Slope [MHz/Rad]'] = 1/data_df['Slope [Rad/MHz]']

fit_param_dict = calc_line_fit_params(data_df['Inverse Slope [MHz/Rad]'], data_df['Resonant Frequency [MHz]'], data_df['Resonant Frequency STD [MHz]'])

x_plt_arr = np.linspace(np.min(data_df['Inverse Slope [MHz/Rad]']), 0, data_df['Inverse Slope [MHz/Rad]'].shape[0] * 2)

fit_data_arr = fit_param_dict['Offset [MHz]'] + fit_param_dict['Slope [Rad/MHz]'] * x_plt_arr

av_freq_s = straight_line_fit_params(data_df['Resonant Frequency [MHz]'], data_df['Resonant Frequency STD [MHz]'])

av_freq_plot_arr = np.ones(x_plt_arr.shape[0]) * av_freq_s['Weighted Mean']
#%%
fig, ax = plt.subplots()
ax.plot(x_plt_arr, fit_data_arr)
ax.plot(x_plt_arr, av_freq_plot_arr)
data_df.plot(kind='scatter', x='Inverse Slope [MHz/Rad]', y='Resonant Frequency [MHz]', yerr='Resonant Frequency STD [MHz]', ax=ax)

ax.set_xlim(right=0)
#%%
fit_param_dict
#%%

fit_func = np.poly1d(np.polyfit(data_df['Inverse Slope [MHz/Rad]'], data_df['Resonant Frequency [MHz]'], deg=1, w=1/data_df['Resonant Frequency STD [MHz]']**2))

x_plt_arr = np.linspace(np.min(data_df['Inverse Slope [MHz/Rad]']), np.max(data_df['Inverse Slope [MHz/Rad]']), data_df['Inverse Slope [MHz/Rad]'].shape[0] * 10)

fit_data_arr = fit_func(data_df['Inverse Slope [MHz/Rad]'])

get_chi_squared(data_df['Resonant Frequency [MHz]'], data_df['Resonant Frequency STD [MHz]'], fit_data_arr, 2)
#%%

fig, ax = plt.subplots()
ax.plot(x_plt_arr, fit_func(x_plt_arr))

data_df.plot(kind='scatter', x='Inverse Slope [MHz/Rad]', y='Resonant Frequency [MHz]', yerr='Resonant Frequency STD [MHz]', ax=ax)
#%%
slope_av_2_df[['Slope [Rad/MHz]','Slope STD (Normalized) [Rad/MHz]']].join(zero_cross_av_2_df[['Resonant Frequency [MHz]', 'Resonant Frequency STD [MHz]']])
#%%
data_arr = zero_cross_av_2_df['Resonant Frequency [MHz]']
data_std_arr = zero_cross_av_2_df['Resonant Frequency STD [MHz]']

av_freq_s = straight_line_fit_params(data_arr, data_std_arr)
#%%
909.8274-909.841244
#%%
av_freq_s
#%%
df.reset_index().plot(style='-.', y='Zero-crossing Frequency [MHz]', yerr='Zero-crossing Frequency STD [MHz]')
#%%
df.reset_index().plot(style='-.', y='Zero-crossing Frequency [MHz]', yerr='Zero-crossing Frequency STD (Normalized) [MHz]')
#%%
#%%
data_arr = df['Zero-crossing Frequency [MHz]']
data_std_arr = df['Zero-crossing Frequency STD [MHz]']


av_s
#%%
av_s['Weighted Mean'] - ac_correc['Field Power Shift [kHz]']*1E-3 + 52.58*1E-3 + 0.03174
#%%
data_arr = df['Zero-crossing Frequency [MHz]']
data_std_arr = df['Zero-crossing Frequency STD (Normalized) [MHz]']

av_s = straight_line_fit_params(data_arr, data_std_arr)
av_s
#%%

#%%
av_s['Weighted Mean'] - ac_correc['Field Power Shift [kHz]']*1E-3 + 52.58*1E-3 + 0.03174
#%%
# Phasor Averaging
# The assumption is that DC of the signal is the same for the duration of the averaging set. This way we can simply take the amplitudes and phases of the phasors and average them together to obtain a single averaged phasor.
def phasor_av(df):
    df.columns = df.columns.droplevel(['Source', 'Data Type'])
    col_list = ['Fourier Amplitude [V]', 'Fourier Phase [Rad]']
    phasor_av_df = group_agg_mult_col_dict(df, col_list, index_list=averaging_set_grouping_list, func=mean_phasor_aggregate)

    return phasor_av_df

def mean_phasor_aggregate(x, col_list):
    ''' Aggregating function that is used for 'group_agg_mult_col_dict' function.

    The function must have the following line it it:
    data_dict = get_column_data_dict(x, col_list)

    This basically creates a dictionary of values that are used as the input to the function.

    Inputs:
    :x: data columns (from pandas)
    :col_list: list of columns used for combining the data into a single column.
    '''
    data_dict = get_column_data_dict(x, col_list)
    return mean_phasor(amp_arr=data_dict['Fourier Amplitude [V]'], phase_arr=data_dict['Fourier Phase [Rad]'])

phasor_av_df = phase_diff_shifted_data_df.loc[slice(None), (slice(None), slice(None), ['Fourier Phase [Rad]','Fourier Amplitude [V]'])].groupby(level=['Source', 'Data Type'], axis='columns').apply(phasor_av)

#%%
0.03174
#%%
hydrogen_sim_data.FOSOFSimulation.__dict__
#%%

#%%
fosof_sim_data.__dict__
#%%
fosof_sim_data.__dict__
#%%
#%%
# Now it is the time to correct for the SOD.
#%%
fosof_lineshape_vs_offset_set_df.loc[(slice(None), slice(None), slice(None), 1.6, 0.03), ('Resonant Frequency Deviation [kHz]')].reset_index(['Fractional Offset', 'Beam RMS Radius [mm]'], drop=True)
#%%
# Waveguide RF power calibration analyzed data folder path
wvg_power_calib_data_folder_path = 'E:/Google Drive/Research/Lamb shift measurement/Data/Waveguide calibration'

wvg_power_calib_info_file_name = 'Waveguide_calibration_info.csv'

os.chdir(wvg_power_calib_data_folder_path)

# Waveguide power calibration info
wvg_calib_info_df = pd.read_csv(wvg_power_calib_info_file_name, header=2)

wvg_calib_info_df['Start Date'] = pd.to_datetime(wvg_calib_info_df['Start Date'])
wvg_calib_info_df['End Date'] = pd.to_datetime(wvg_calib_info_df['End Date'])
#%%
wvg_calib_info_df
#%%
''' Due to the systematic uncertainty in the offset of 50% we have a systematic error in the power calibration for each waveguide calibration data set. I choose to average all of these calibration errors together for both of the waveguides and all of the frequencies to obtain one averaged fractional power error for each RF power setting.

Now there is also a possibility to obtain this error from the simulation. This can be done in the following way: we can use the simulation with artificially added offset. This way we can see to what power given suriviving fraction corresponds to at different quench offsets. In a sense, this method is better than what I am doing, because the systematic shift can be determined very quickly. However, with the first method above, I am calculating the calibration that I would actually use - it incorporates any imperfections in the analysis techniques, such as a variety of fits to the data and interpolations. This brings us to the question of what kind of uncertainties our RF power calibration has. These are the following:
1. Statistical uncertainty due to random errors in the DC ON/OFF ratios.
2. Systematic uncertainty due to the quench offset. I assume that the quench offset is independent of RF power, which is not true. This is the main reason for adding a relatively large fractional error to the quench offset. Another complication, is that it seems that the quench offset goes down with the decreasing pressure in the Box. Thus, in principle, after taking the calibration data, the quench offset could be smaller the longer the time interval is from the time the calibration data was taken and the given FOSOF data set is acquired.
3. Uncertainty due to the analysis technique. This is a systematic uncertainty.
4. Systematic uncertainty due to simulation itself. We do not know the profile of the fields exactly, and we are not sure if the simulation takes all of the possible effects into account + we are not sure about any numerical inaccuracies of the simulations.

For #1: Since I do not know the exact functional form of the dependence of various RF power-related parameters on the DC ON/OFF ratios and the extracted E fields, I can use the fit parameters, but I cannot use their uncertainties. Thus, in a sense, I do not have the statistical error for the power calibration.

For #3: This uncertainty is estimated by performing various methods of analysis and determining the averaged value = what we use for the power calibration + its STD (not STDOM). This serves as the analysis uncertainty. We have this uncertainty for each RF frequency and RF power value. And I am not sure how to incorporate this uncertainty into the one overall uncertainty value for each RF power. I can simply find the RMS uncertainty, assuming that the uncertainty is the same for all frequencies, but it is not the case: the data did not have the same noise level for all of the frequencies.

Another possibility of dealing with this issue is to assume that each type of analysis is the correct one. Thus we will have several possible calibration curves. In a sense, this is still exactly what I do: I average them and find the standard deviation. This gives me the averaged calibration curve, which is the best estimate in a sense.

I actually think that I am being somewhat incorrect here about the statistical error. I think I have to separate the issues of the analysis uncertainty and the statistical uncertainty. If I assume that the analysis is perfect, then I can use the uncertainty in the fit parameters to give me the uncertainty in the determined calibration. Notice that the fitting with the smoothing spline does not give any uncertainty associated with its parameters. I guess since the spline and the polynomial look very similar, I can assume that they have similar uncertainty. At first I thought that by assuming so I overestimate the uncertainty in the smoothing spline, because it is meant to follow the data points closely, and thus is, necessary, a better approximation of the data. But then I realized that by the same token I can use non-smoothing spline and get a perfect fit through all of the data points and then state that this is the perfect fit function for the data. But this is incorrect, because between the data points the spline will not, generally, behave that well. Therefore, I would say that indeed the spline and the polynomial fit both should have similar uncertainty in the interpolated values. In addition, we are, in a sense, forced to have the same statistical uncertainty in the fit parameters, because we are stating that each type of analysis is equally good. I.e., we are decoupling the systematic uncertainty due to the type of analysis and the statistical uncertainty due to data quality.

For #4. This is an important one. We have also seen that sometimes the resulting calibration curve (RF power setting vs RF frequency) has visible discontinuities in it, corresponding to various frequency ranges acquired at different times. It is possible that the lab temperature, beam parameters, line voltages are not necessary the same for all of these data sets, which results in the additional systematic uncertainty. It is also possible that the RF generator does not output the same power for its given power output setting. Because of that I rely on the RF power detector that are in the temperature-stabilized enclosure to provide me with the reliable metric for the RF power in each of the waveguides. Note that while acquiring data we actually used the RF power generator setting as the calibration parameter. Therefore it is quite possible that the data we acquired does not have the perfect calibration. This is the main reason for employing correction of the FOSOF phases for imperfect RF power calibration for the waveguides. These effects, by the way, are quite different from the simulation uncertainty, as for the simulation not being calculated correctly, but it is more about not using the correct simulation for the given calibration data. Thus, in principle, this is the #5 systematic uncertainty. One can, in principle, correct for this by having many-many simulations done for the beam having slightly different parameters (shape, speed, angle at which the beam is travelling, distribution of velocities), and then using these simulations for each calibration data set to perform the individual analysis on each of these. This results in a very large number of possible calibrations. One can average all of these together and obtain the best estimate calibration curve + the uncertainty in it.

Thus the steps are the following:

1. We need to run Monte Carlo simulation for various possible beam parameters (including the beam speed + spread in the speed) to obtain the beam profile inside the waveguides (at different locations inside the waveguides).
2. These beam profiles are used to obtain corresponding simulated quench curves.
3. For each simulated quench curve and the respective calibration data we perform several types of analysis for several different quench offset values.
4. For each quench offset value we take the respective calibration curves average them together to give us the averaged calibration curve + the statistical and the systematic uncertainty in the calibration. Here, the statistical uncertainty is the uncertainty that comes from the analysis that has the polynomial fit parameters. The systematic uncertainty is due to not knowing exactly the simulation to use is the spread in the power calibration curves obtained from different simulation curves, but using the same analysis method.
5. Now we have a set of different quench curves, each having 2 systematic uncertainties and 1 statistical uncertainty. The average of the average calibration curves for each quench offset gives the final average calibration curve, and the RMS uncertainty of the respective types of uncertainties gives us 2 types of the systematic uncertainties, and 1 type of the statistical uncertainty.
6. Each simulation has some systematic uncertainty due to the accuracy of the model used for the simulation. We need to agree on some number here. Something like +-2% of the quenched fraction = (1-surviving fraction), for instance. Notice, that this means that the larger field power corresponds to larger uncertainty in the quenched fraction. This is another systematic uncertainty that gets added (in quadrature) to other systematic uncertainties for the final averaged calibration curve. Notice that this uncertainty is not that simple: there is some uncertainty in the knowledge of the exact E and B field profiles inside the waveguides + there is the uncertainty in the theoretical model + uncertainty in the numerical accuracy. I assume that all of these uncertainties essentially result in the 2% error (for instance) in the determination of the quenched fraction. Now, to incorporate this uncertainty, for every simulation curve we need to construct 2 more curves having 2% more or less of quenching at every frequency + field power. This is, in a sense, a worst case scenario of the estimate of the effect of the imperfect simulation on the calibration. Obviously, all of the other steps have to be repeated for each of these simulation curves again.
7. At the end, what we get is the averaged calibration curve that has 3 types of systematic uncertainties + 1 statistical uncertainty. The uncertainty is for the RF-power-related parameter vs RF frequency for given RF power. Notice that the total systematic uncertainties for different frequencies and RF powers are not independent from each other, but are very correlated.

8. Now, the reason for knowing the RF power in the waveguides very well for each RF frequency is to be able to properly correct the FOSOF zero-crossing frequency for the AC shift. Thus, in a sense, we do not care about the uncertainty in the power at each RF frequency, but we care about the resulting uncertainty in the AC shift correction. Let us see now how the uncertainty in the calibration curve can be used to obtain the uncertainty in the AC shift correction.

9. Even after performing this complicated analysis, we still, unfortunately, cannot use this averaged calibration to correct the FOSOF phases, because the FOSOF lineshape depends on the beam parameters. Thus for each simulated quench curve we need to have the corresponding simulated FOSOF lineshape. Therefore for each simulated quench curve we use the respective calibration curves obtained with different types of analysis. This way the calibration curve has only 2 types of uncertainties: the systematic uncertainty due to the uncertainty in the accuracy of the model + the statistical uncertainty.

10. The simulated FOSOF lineshape has the uncertainty in it as well that is correlated with the systematic uncertainty in the corresponding simulated quench curve. But it is not just, for instance, 2%, because one expects the FOSOF lineshape to be much more sensitive to any imperfections in the model. Thus, one would use a larger uncertainty. This uncertainty is in the phases, and, certaintly depends on the RF power, and the RF frequency. Now, simularly to the quecnh curves as the RF power is decreased, the zero-crossing frequency of the FOSOF lineshape becomes less and less dependent on the RF power, exact field profile, and the beam profile. In other words, at zero power one expects to obtain the zero-crossing frequency that is equal to from the resonant frequency. As the RF power increases, the uncertainty in the difference between the zero-crossing frequency and the resonant frequency increases. We can assign something like 5% uncertainty for this difference that is proportional to the RF power. It is not easy to come up with the way to assign the systematic uncertainties to each of the FOSOF phase at every RF frequency. However, we can simply assume that all of the phases get shifted by the same amount for given RF power to result in the requested 5% systematic uncertainty in the AC shift correction. Notice, however, that I do not assume that this 5% uncertainty can be explained as the same 5% effect the AC shift for all of the powers. I can assume that the uncertainty can be treated independently at different RF powers. The reason for this is that I am claiming that the possible effect on the zero-crossing is complicated, and that it is hard to know what happens from one RF power to another. Notice that this is different from the systematic uncertainty of 2% in the simulated quench curve. I can also assume that the systematic uncertainties at different powers are totally correlated, as for the simulated quench curves. Thus there are 2 different ways of looking at this uncertainty. I will assume that the 1st method is the valid one. This method essentially allows me to think of this uncertainty as the statistical one. The second method, on the other hand, requires me to perform different analysis depending on what AC curve I pick - 0%, -5% or +5%.

11. Now, for each element of the set of simulated quench curves obtained for various beam parameters (uncertainty #5) we can construct the respective FOSOF lineshape. We first pick a calibration curve that was obtained from one of the analyses methods. This way the calibration curve has only the statistical uncertainty. We want now to use the calibration curve to correct the FOSOF lineshape for imperfect power, given the measured RF power detector voltages. This can be easily done by first assuming that the each calibration value of the RF power detector voltage for each RF frequency and the requested RF power is normally distributed. We can run Monte-Carlo simulation and obtain many-many corrected FOSOF lineshapes by using normally sampled values from the calibration curve. For each corrected FOSOF lineshape we determine the zero-crossing + its uncertainty, which is related to the uncertainty in the FOSOF phases data = #6 uncertainty. These zero-crossing frequencies are averaged together and we obtain the mean zero-crossing frequency + its statistical uncertainty. Thus we get a resonant frequency with 2 types of statistical uncertainties. One is the RMS uncertainty of all of the statistical uncertainties due to the noise in the FOSOF phases data, and another is due to the statistical uncertainty in the calibration data - uncertainty #1.

12. For each simulation curve obtained for various beam parameters we perform step 11. Then all of zero-crossing frequencies from the step 11 are averaged together. The standard deviation is the systematic uncertainty in the zero-crossing due to uncertainty #5 = related to not knowing the beam parameters exactly. The average zero-crossing frequency also has the 2 statistical uncertainties: the RMS uncertainties of the statistical uncertainties of the zero-crossing frequencies in the set in the step 11 (they must be almost identical).

13. We perform the step 11 and 12 for all other 3 types of calibration data analysis. All 4 obtained zero-crossing frequencies are averaged together. The standard deviation in this case is the systematic uncertainty due to different types of analysis (uncertainty #3). The obtained number now has 4 types of systematic uncertainties (#3, #5), and two statistical uncertainties (#1, #6).

14. We perform the step 13 for each of the quench offset. Average of all of the resulting zero-crossing frequencies (3 of them) + the uncertainty is related to the uncertainty #2

15. Now we have the last uncertainty to take care of: #4. Assuming that the AC shift uncertainties are independent from each other at different powers we can do the following. We use the +-2% simulated quench curves and the respective +-5% FOSOF lineshapes, and repeat step 14. The resulting zero-crossing frequencies are averaged together, which is our best estimate of the zero-crossing frequency obtained from single FOSOF data set. The uncertainty of this average is related to the systematic uncertainty #2. Now we have 6 types of uncertainties: 4 systematic ones and 2 statistical. Notice that only one statistical uncertainty can be reduced by taking more data. Other uncertainties stay the same, because they are correlated. We can then correct the averaged zero-crossing frequency for the AC shift.

16. Now the problem with the analysis is that the uncertaintes related to #2, #3, #5 have large correlations between the data acquired for different waveguide separations, powers, and accelerating voltages. #1 is not a part of this (however it for given RF frequency it is actually correlated to the calibration values for different RF powers), because we use different calibration data for different separations. But we use the same analysis code + same code for obtaining simulation data for all of the powers, speeds, and the separations. Hence, the #2, #3, #5 are respectively correlated for different experiment parameters. This means that the steps 11-15 are not entirely correct.

17. We actually need to do the following. For each simulation quench curve we select type of the calibration data analysis, we also select the set of RF powers and set of waveguide separations, with the respective simulated FOSOF lineshapes, and pick the respective calibration curves. For each waveguide separation and beam speed, we pick one set of normally chosen values from the respective calibration curve and then correct the FOSOF data with the these values. Note that we use the same calibration curve values for all RF powers for the given waveguide separation and beam speed, because of the correlation of the uncertainty #1. For each corrected FOSOF lineshape we perform the step similar to step 15. As the result we add the systematic uncertainty to each zero-crossing frequency that is related to the uncertainty #4. Thus each zero-crossing frequency will have 2 types of uncertainties: 1 statistical, and 1 systematic. Each of these zero-crossing frequencies is corrected for its AC shift and we get the set of the resonant frequencies corrected for the AC shift.

We now have a set of the resonant frequencies corrected for their AC shift for given RF power, waveguide separation, accelerating voltage, quench offset, and the set of normally chosen values from the calibration curves. Just to be clear: after this step we have the following data:

Separation:

4 cm: {p_4_1: f_4_1, p_4_2: f_4_2, p_4_3: f_4_3, ...}
5 cm: {p_5_1: f_5_1, p_5_2: f_5_2, p_5_3: f_5_3, ...}
...

, where p_i_j = power for the separation i, power index j, and f_i_j = AC-shift-corrected resonant frequency for the separation i, power index j

For each beam speed and separation we calculate the weighted average of the AC-shift-corrected resonant frequencies.

We now perform these steps for many-many other sets of normally chosen values for the respective calibration curves and obtain a set of weighted averages of the AC-shift-corrected resonant frequencies for given separation and beam speed. These are, again, averaged together and we obtain the number that has two uncertainties in it. One is the uncertainty that is the combination of the statistical and systematic uncertainty due to noise in the FOSOF phase data and AC shift uncertainty, and another is the systematic uncertainty related to the #1 uncertainty.

Notice that this procedure is repeated for every waveguide separation + beam speed. After we take the weighted average of the data to obtain the final AC-shift-corrected resonant frequency obtained by using a set of simulation curves for the same beam parameters, and the same calibration data analysis type. Note that the weights for the averaging can be chosen by us.

19. Step 18 is performed for all of other simulated quench curves for different beam parameters. This is similar to step 12.
20. We do the step similar to the step 13.
21. We do the step similar to the step 14.
22. We now have the set of the AC-shift corrected frequencies with 5 types of uncertainties: 1 is the combination of statistical (#6) + systematic (#4), the second one is related to #1, the third is related to #5, the fourth on is #3, and the fifth one is #2.

2018-11-11

After some thinking I have realized that the step 18, where I assume that the set of simulations for the same beam parameters can be used for all of the data sets, is not entirely correct. The reason is that I was adjusting the beam from data set to data set (usually). Or, even if I did not adjust anything, I cannot assume that the beam stays the same from the data set to data set. Therefore I think I can treat the systematic uncertainty #5 as the independent uncertainty. I.e., each data set should be corrected by a set of simulation curves with different beam parameters.

Therefore, the step 18 should be modified to be the following. I choose a set of simulated quench curves for different beam parameters + their respective simulated FOSOF lineshapes, but the same uncertainty level of type #4 + I pick the calibration data analysis type. For each waveguide separation and beam speed I do the following. I randomly pick the beam parameters for each FOSOF data set for this particular waveguide separation and beam speed, and thus I get a set of respective simulation curves used for the set of the FOSOF data sets. Each of these simulated quench curves have their own calibration curves. I want now to pick the same normally chosen values from each calibration curve. But it does not sound right, because the calibration curves will not be the same, since different simulations are used to obtain them. It seems that for each calibration curve for the same waveguide separation and beam speed, I need to randomly pick the same fractional deviation in terms of the multiple of the standard deviation for each calibration curve's point. I believe this is identical to performing the following. The calibration data consists of the DC ON/OFF surviving ratios with the corresponding uncertainties. For the given set of simulated quench curves obtained for different beam parameters, we can normally pick the DC ON/OFF ratio values from the calibration data, and use these values to find the set of calibration curves using the chosen calibration data analysis type. We now use this set of the calibration quench curves and the respective simulated FOSOF lineshapes to correct the FOSOF data for imperfect RF power. This is done for the FOSOF data sets having the same waveguide separation, beam speed, and waveguide power calibration data. For each of these corrected data sets we determine the zero-crossing frequency, which contains only the FOSOF data-related statistical uncertainty. For the given set of randomly chosen values from the DC ON/OFF ratios, we perform the average of the zero-crossing frequencies obtained for the same waveguide separation, beam speed, RF power, and the waveguide power calibration data to obtain a single zero-crossing frequency that has 2 uncertainties: 1 is the statistical FOSOF uncertainty (type #6) and 1 is the systematic uncertainty due to not knowing the beam parameters exactly for each data set (type #5). Thus at this moment we obtain for the given waveguide separation and the beam speed the following:

{(p_1, f_1), (p_2, f_2), (p_3, f_3), ...} - set of ordered pairs of powers and zero-crossing frequencies. Note that we have not applied any correction for the AC shift, and there is no uncertainty added due to this shift yet.

Now, how do I apply the AC shift correction to this? Do I simply add the #4-type uncertainty to each AC-shift-corrected resonant frequency? The problem is that I do not know how to go from 2% uncertainty in the simulated quench curves to 5% uncertainty in the FOSOF lineshape, because in one case the uncertainty is correlated, and in another case I claim that it is independent. Also, I would have to do this for each type of the calibration data analysis, but then I need to make sure that I assume that this uncertainty is totally correlated for all of the types.

I think to deal with this problem I need to do the similar type of the analysis as for the type #1 uncertainty. For each zero-crossing frequency determined above, that has 2 uncertainties, I normally pick the values of the shift.

We then perform this procedure for many-many choices of the DC ON/OFF ratio values for the chosen waveguide separation, beam speed, and the calibration data, and obtain the set of the zero-crossing frequencies.





It is quite difficult to perform this type of analysis. Let's see if there is anything I can do to simplify the analysis.

I will first assume that the statistical uncertainty #1 is too small to be relevant.

For step 19: The simulated quench curves for different off-axis distance are almost identical. I can assume that the change in the simulated quench curve with different beam parameters is too small to be important. Here I am talking about possible small changes in the beam speed, beam profile, beam velocity profile. Thus I can use the same simulation curve for all FOSOF data sets that have the same accelerating voltage. Now, the FOSOF lineshape is sensitive to the beam diameter. Because of that, just to be on a safe side I am assigning large fractional uncertainty to be beam rms radius. This should take care of the small changes in beam speed, beam shape, beam velocity profile. This takes care of the uncertainty #5.

For step 20. I assume that the different analysis methods do not result in appreciably large uncertainty in the final resonant frequency. In a sense, I am saying that the contribution to the total uncertainty in the resonant frequency is not appreciable, and thus can be ignored. This is the uncertainty #3.

For step 21. I do have calibration curves for different quench offsets. However, instead of correcting the data for each calibration curve with different offset, I can simply change the RF powers by the fractional amount calculated from comparing these calibration curves with different offsets. This is for the uncertainty #2.


'''

# Loading the calibration curves
av_rf_power_calib_error_set_df = pd.DataFrame()

for index, row in wvg_calib_info_df.iterrows():
    os.chdir(wvg_power_calib_data_folder_path)

    wvg_calib_analysis = wvg_power_calib_analysis.WaveguideCalibrationAnalysis(load_Q=True, calib_folder_name=row['Calibration Folder Name'], calib_file_name=row['Calibration File Name'])

    av_rf_power_calib_error_df = wvg_calib_analysis.get_av_rf_power_calib_error().copy()

    av_rf_power_calib_error_df['Calibration Folder Name'] = row['Calibration Folder Name']

    av_rf_power_calib_error_df.set_index('Calibration Folder Name', append=True, inplace=True)

    av_rf_power_calib_error_set_df = av_rf_power_calib_error_set_df.append(av_rf_power_calib_error_df)

mean_frac_calib_error_df = av_rf_power_calib_error_set_df[['Mean Fractional Error [%]']].sort_index().groupby('Proportional To RF Power [V^2/cm^2]').aggregate(lambda x: np.mean(x)).rename(columns={'Mean Fractional Error [%]': 'RF Power Calibration Fractional Error [%]'})

mean_frac_calib_error_df = mean_frac_calib_error_df.reset_index()

mean_frac_calib_error_df['Proportional To RF Power [V^2/cm^2]'] = np.sqrt(mean_frac_calib_error_df['Proportional To RF Power [V^2/cm^2]'])

mean_frac_calib_error_df = mean_frac_calib_error_df.rename(columns={'Proportional To RF Power [V^2/cm^2]': 'Waveguide Electric Field [V/cm]'}).set_index('Waveguide Electric Field [V/cm]')
#%%
mean_frac_calib_error_df
#%%
#

wvg_calib_analysis.get_av_calib_data()
