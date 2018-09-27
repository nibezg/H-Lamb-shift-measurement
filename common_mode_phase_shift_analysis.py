from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string

# For the lab
sys.path.insert(0,"C:/Users/Helium1/Google Drive/Research/Lamb shift measurement/Code") #
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
#%%
exp_folder_name_list = [
'180831-155219 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms',
'180831-163511 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms',
'180831-170509 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms',
'180831-173731 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms',
'180831-182541 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms',
'180831-185613 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms',
'180905-170042 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Pumped down',
'180905-174735 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Pumped down',
'180906-113110 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Pumped down',
'180906-120706 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Pumped down',
'180906-132043 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Pumped down',
'180906-135139 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Pumped down',
'180906-142141 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Pumped down',
'180906-145249 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Pumped down',
'180906-170643 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms',
'180906-174353 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms',
'180911-121721 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms',
'180911-124821 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms',
'180911-132424 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms',
'180911-135705 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms',
'180911-142738 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms',
'180911-154930 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms',
'180912-113743 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180912-120834 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180912-123849 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180912-133238 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180912-140628 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180912-150535 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180912-153959 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180912-161221 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180914-132327 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180914-135544 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180914-142851 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180914-145924 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180914-153241 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180914-160305 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180914-165055 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180914-172153 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180918-141158 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180918-144441 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180918-152949 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180918-160332 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180918-170805 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180918-174146 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180919-113415 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180919-120821 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180919-123733 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180919-131111 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180919-141307 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180919-144527 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180919-151546 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180919-155358 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180919-162415 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180919-165749 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180921-151123 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180921-154352 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180921-173315 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180921-181640 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-112856 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-120804 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-123828 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-131141 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-134923 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-142021 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-145221 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-152340 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-155631 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-163149 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-170134 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180926-173412 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum',
'180927-131104 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum'

]

phase_drift_data_set_df = None

for exp_folder_name in exp_folder_name_list:

    data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_data_Q=True)
    data_set.save_analysis_data()

    digi_df = data_set.get_digitizers_data()
    comb_phase_diff_df = data_set.get_combiners_phase_diff_data()
    digi_delay_df = data_set.get_inter_digi_delay_data()
    phase_diff_df = data_set.get_phase_diff_data()
    phase_av_set_averaged_df = data_set.average_av_sets()

    phase_A_minus_B_df, phase_freq_response_df = data_set.cancel_out_freq_response()
    #fosof_ampl_df, fosof_phase_df = data_set.average_FOSOF_over_repeats()

    inter_comb_phase_diff_df = phase_A_minus_B_df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging', ['Phase [Rad]', 'Phase STD [Rad]'])]

    inter_comb_phase_diff_grouped_df = inter_comb_phase_diff_df.reorder_levels(axis='columns', order=['Averaging Type', 'Fourier Harmonic', 'Phase Reference Type', 'Data Field']).sort_index(axis='columns').groupby(level=['Fourier Harmonic'], axis='columns')

    df = inter_comb_phase_diff_grouped_df.get_group(('First Harmonic'))
    df.columns = df.columns.droplevel(level=['Averaging Type', 'Fourier Harmonic'])

    # Calculate the phase difference between inter-waveguide combiner and other combiners + the standard deviation. The logic behind the calculation is shown on p.30 of Lab Notes #4 (August 31, 2018).
    # The standard deviation is not calculated correctly, due to correlations hidden in the phase difference (inter-waveguide phase difference is common to both of the calculations). Probably this is not even that important.

    df.loc[slice(None), 'Phase Difference [Rad]'] = (1/4*df['RF Combiner I Reference', 'Phase [Rad]'].values + 1/4*df['RF Combiner R Reference', 'Phase [Rad]'].values)

    df.loc[slice(None), 'Phase Difference STD [Rad]'] = np.sqrt((1/4*df['RF Combiner I Reference', 'Phase STD [Rad]'].values)**2 + (1/4*df['RF Combiner R Reference', 'Phase STD [Rad]'].values)**2)

    # Notice that the FOSOF phases that we had are from RF CHA - RF CH B calculation, WITHOUT division by 2. Thus we have to divide it here by 2.
    df.loc[slice(None), 'Phase Difference [Rad]'] = 1/2 * df.loc[slice(None), 'Phase Difference [Rad]']

    df.loc[slice(None), 'Phase Difference STD [Rad]'] = 1/2 * df.loc[slice(None), 'Phase Difference STD [Rad]']

    phase_drift_df = pd.DataFrame(pd.Series({
        'Data Set': data_set,
        'Mean Combiner Phase Difference [Rad]': np.mean(comb_phase_diff_df['First Harmonic', 'Fourier Phase [Rad]']),
        'Mean Phase Difference [Rad]': np.mean(df['Phase Difference [Rad]']),
        'Mean Combiner I Phase [Rad]': np.mean(df['RF Combiner I Reference', 'Phase [Rad]']),
        'Mean Combiner R Phase [Rad]': np.mean(df['RF Combiner R Reference', 'Phase [Rad]'])}, name=exp_folder_name))

    if phase_drift_data_set_df is None:
        phase_drift_data_set_df = phase_drift_df
    else:
        phase_drift_data_set_df = phase_drift_data_set_df.join(phase_drift_df)
#%%
phase_drift_data_df = phase_drift_data_set_df.drop(axis='index', labels='Data Set')

test_1_s = phase_drift_data_df['180831-155219 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms'] - phase_drift_data_df['180831-163511 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms']

test_2_s = phase_drift_data_df['180831-173731 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms'] - phase_drift_data_df['180831-170509 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms']

test_3_s = phase_drift_data_df['180831-182541 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms'] - phase_drift_data_df['180831-185613 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms']

test_4_s = phase_drift_data_df['180905-170042 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Pumped down'] - phase_drift_data_df['180905-174735 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Pumped down']

test_5_s = phase_drift_data_df['180906-113110 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Pumped down'] - phase_drift_data_df['180906-120706 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Pumped down']

test_6_s = phase_drift_data_df['180906-135139 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Pumped down'] - phase_drift_data_df['180906-132043 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Pumped down']

test_7_s = phase_drift_data_df['180906-142141 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Pumped down'] - phase_drift_data_df['180906-145249 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Pumped down']

test_8_s = phase_drift_data_df['180906-170643 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms'] - phase_drift_data_df['180906-174353 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms']

test_9_s = phase_drift_data_df['180911-121721 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms'] - phase_drift_data_df['180911-124821 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms']

test_10_s = phase_drift_data_df['180911-135705 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms'] - phase_drift_data_df['180911-132424 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms']

test_11_s = phase_drift_data_df['180911-142738 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms'] - phase_drift_data_df['180911-154930 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms']

test_12_s = phase_drift_data_df['180912-113743 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180912-120834 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_13_s = phase_drift_data_df['180912-133238 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180912-123849 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_14_s = phase_drift_data_df['180912-140628 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180912-150535 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_15_s = phase_drift_data_df['180912-161221 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180912-153959 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_16_s = phase_drift_data_df['180914-135544 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180914-132327 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_17_s = phase_drift_data_df['180914-142851 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180914-145924 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_18_s = phase_drift_data_df['180914-160305 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180914-153241 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_19_s = phase_drift_data_df['180914-165055 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180914-172153 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_20_s = phase_drift_data_df['180918-141158 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180918-144441 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_21_s = phase_drift_data_df['180918-160332 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180918-152949 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_22_s = phase_drift_data_df['180918-170805 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180918-174146 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_23_s = phase_drift_data_df['180919-113415 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180919-120821 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_24_s = phase_drift_data_df['180919-131111 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180919-123733 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_25_s = phase_drift_data_df['180919-141307 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180919-144527 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_26_s = phase_drift_data_df['180919-155358 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180919-151546 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_27_s = phase_drift_data_df['180919-162415 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180919-165749 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_28_s = phase_drift_data_df['180921-154352 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180921-151123 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_29_s = phase_drift_data_df['180921-173315 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180921-181640 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_30_s = phase_drift_data_df['180926-112856 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180926-120804 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_31_s = phase_drift_data_df['180926-131141 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180926-123828 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_32_s = phase_drift_data_df['180926-134923 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180926-142021 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_33_s = phase_drift_data_df['180926-152340 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180926-145221 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_34_s = phase_drift_data_df['180926-155631 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180926-163149 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

test_35_s = phase_drift_data_df['180926-173412 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'] - phase_drift_data_df['180926-170134 - FOSOF Common-mode phase drift systematic test - pi config, 18 V per cm 910 MHz. No atoms. Under vacuum']

phase_shift_systematic_df = pd.DataFrame([test_1_s, test_2_s, test_3_s, test_4_s, test_5_s, test_6_s, test_7_s, test_8_s, test_9_s, test_10_s, test_11_s, test_12_s, test_13_s, test_14_s, test_15_s, test_16_s, test_17_s, test_18_s, test_19_s, test_20_s, test_21_s, test_22_s, test_23_s, test_24_s, test_25_s, test_26_s, test_27_s, test_28_s, test_29_s, test_30_s, test_31_s, test_32_s, test_33_s, test_34_s, test_35_s])

# Here we divide by 2, because the Combiner I and Combiner R difference calculated was including the A and B difference, but this particular difference was not divided by 2 in the code.
phase_shift_systematic_df['Mean Combiner Phase Difference [Rad]'] = phase_shift_systematic_df['Mean Combiner Phase Difference [Rad]']*1E3/2

phase_shift_systematic_df['Mean Phase Difference [Rad]'] = phase_shift_systematic_df['Mean Phase Difference [Rad]']*1E3

phase_shift_systematic_df = phase_shift_systematic_df.rename(columns={'Mean Combiner Phase Difference [Rad]': 'Mean Combiner Phase Difference Change [mrad]', 'Mean Phase Difference [Rad]': 'Mean Phase Difference Change [mrad]'})
#%%
fix, ax = plt.subplots()
phase_shift_systematic_df.plot(y='Mean Phase Difference Change [mrad]', use_index=True, ax=ax)
plt.show()
phase_shift_systematic_df
#%%
# Dropping data when the Box was under atmospheric pressure.
av_phase_shift_systematic_df = phase_shift_systematic_df.drop([0, 1, 2, 7, 8, 9, 10]).aggregate(lambda x: np.mean(x))
av_phase_shift_systematic_df
#%%
# Dropping data when the Box was under atmospheric pressure.
av_phase_shift_systematic_df = phase_shift_systematic_df.aggregate(lambda x: np.mean(x))
av_phase_shift_systematic_df
#%%
shift = (10.5+9+7.5+6.7+8.4)*av_phase_shift_systematic_df['Mean Phase Difference Change [mrad]']/5

shift
#%%
np.sqrt(2.9**2+shift**2)
#%%

#%%
exp_folder_name = '180914-135544 - FOSOF Common-mode phase drift systematic test - 0 config, 18 V per cm 910 MHz. No atoms. Under vacuum'

data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_data_Q=True)
#data_set.save_analysis_data()

digi_df = data_set.get_digitizers_data()
comb_phase_diff_df = data_set.get_combiners_phase_diff_data()
digi_delay_df = data_set.get_inter_digi_delay_data()
phase_diff_df = data_set.get_phase_diff_data()
phase_av_set_averaged_df = data_set.average_av_sets()

phase_A_minus_B_df, phase_freq_response_df = data_set.cancel_out_freq_response()
#fosof_ampl_df, fosof_phase_df = data_set.average_FOSOF_over_repeats()

inter_comb_phase_diff_df = phase_A_minus_B_df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging', ['Phase [Rad]', 'Phase STD [Rad]'])]

inter_comb_phase_diff_grouped_df = inter_comb_phase_diff_df.reorder_levels(axis='columns', order=['Averaging Type', 'Fourier Harmonic', 'Phase Reference Type', 'Data Field']).sort_index(axis='columns').groupby(level=['Fourier Harmonic'], axis='columns')

df = inter_comb_phase_diff_grouped_df.get_group(('First Harmonic'))
df.columns = df.columns.droplevel(level=['Averaging Type', 'Fourier Harmonic'])

# Calculate the phase difference between inter-waveguide combiner and other combiners + the standard deviation. The standard deviation is not calculated correctly, due to correlations hidden in the phase difference (inter-waveguide phase difference is common to both of the calculations). Probably this is not even that important.

df.loc[slice(None), 'Phase Difference [Rad]'] = 1/4*df['RF Combiner I Reference', 'Phase [Rad]'].values + 1/4*df['RF Combiner R Reference', 'Phase [Rad]'].values

df.loc[slice(None), 'Phase Difference STD [Rad]'] = np.sqrt((1/4*df['RF Combiner I Reference', 'Phase STD [Rad]'].values)**2 + (1/4*df['RF Combiner R Reference', 'Phase STD [Rad]'].values)**2)

#%%
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(16,12)

df['RF Combiner I Reference'].reset_index().plot(kind='scatter', x='Repeat', y='Phase [Rad]', yerr='Phase STD [Rad]', ax=axes[0, 0])

data_arr_max = np.max((df['RF Combiner I Reference', 'Phase [Rad]']+df['RF Combiner I Reference', 'Phase STD [Rad]']).values)
data_arr_min = np.min((df['RF Combiner I Reference', 'Phase [Rad]']-df['RF Combiner I Reference', 'Phase STD [Rad]']).values)

axes[0, 0].set_ylim(data_arr_min, data_arr_max)

df['RF Combiner R Reference'].reset_index().plot(kind='scatter', x='Repeat', y='Phase [Rad]', yerr='Phase STD [Rad]', ax=axes[0, 1])

data_arr_max = np.max((df['RF Combiner R Reference', 'Phase [Rad]']+df['RF Combiner I Reference', 'Phase STD [Rad]']).values)
data_arr_min = np.min((df['RF Combiner R Reference', 'Phase [Rad]']-df['RF Combiner I Reference', 'Phase STD [Rad]']).values)

axes[0, 1].set_ylim(data_arr_min, data_arr_max)

df.reset_index().plot(kind='scatter', x='Repeat', y='Phase Difference [Rad]', yerr='Phase Difference STD [Rad]', ax=axes[1, 0])

data_arr_max = np.max((df['Phase Difference [Rad]']+df['Phase Difference STD [Rad]']).values)
data_arr_min = np.min((df['Phase Difference [Rad]']-df['Phase Difference STD [Rad]']).values)
data_arr_max - data_arr_min
axes[1, 0].set_ylim(data_arr_min, data_arr_max)

comb_phase_diff_df['First Harmonic'].reset_index().plot(kind='scatter', x='Elapsed Time [s]', y='Fourier Phase [Rad]', ax=axes[1, 1])

data_arr_max = np.max((comb_phase_diff_df['First Harmonic', 'Fourier Phase [Rad]']).values)
data_arr_min = np.min((comb_phase_diff_df['First Harmonic', 'Fourier Phase [Rad]']).values)

axes[1, 1].set_ylim(data_arr_min, data_arr_max)

plt.show()
