from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

path_data_df = pd.read_csv(filepath_or_buffer='path_data.csv', delimiter=',', comment='#', header=[0], skip_blank_lines=True, index_col=[0])

code_folder_path = path_data_df.loc['Code Folder'].values[0].replace('\\', '/')
fosof_analyzed_data_folder_path = path_data_df.loc['FOSOF Analyzed Data Folder'].values[0].replace('\\', '/')
wvg_calib_data_folder_path = path_data_df.loc['Waveguide Calibration Folder'].values[0].replace('\\', '/')
krytar109B_pwr_det_calib_folder_path = path_data_df.loc['KRYTAR 109 B Power Detector Calibration Data Folder'].values[0].replace('\\', '/')
sim_data_folder_path = path_data_df.loc['Simulation Data Folder'].values[0].replace('\\', '/')

sys.path.insert(0, code_folder_path)

from exp_data_analysis import *
from fosof_data_set_analysis import *
from ZX47_Calibration_analysis import *
from KRYTAR_109_B_Calib_analysis import *
from wvg_power_calib_raw_data_analysis import *
#from hydrogen_sim_data import *

import re
import time
import math
import copy
import pickle

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt

# Package for wrapping long string to a paragraph
import textwrap

#%%
# Folder containing acquired data table
data_folder = "//LAMBSHIFT-PC/Google Drive/data"

saving_folder_location = wvg_calib_data_folder_path

# Experiment data file name
data_file = 'data.txt'

# File containing quench cavities' parameters.
quench_cav_params_file_name = 'filename.quench'

# Rename the frequency column to this particular value
freq_column_name = 'Waveguide Carrier Frequency [MHz]'

class DataSetQuenchCurveWaveguideOld(DataSetQuenchCurveWaveguide):
    ''' General class for analyzing data sets for the Quench curves of the Waveguides derived from the DataSetQuenchCurveWaveguide class. It contains all of the required analysis function.

    This particular class is for analyzing calibration data when the waveguide power gets turned off for every averaging set for every RF power. In addition to that, the RF synthesizers for the quench cavities get turned off as well. This was the older type of acquisition code.

    Inputs:
    :exp_folder_name: (string) name of the experiment folder that contains the acquired data for the waveguides' calibration
    :load_Q: (bool) Loads the previously instantiated class instance with the same calib_folder_name, if it was saved before.
    '''

    def __init__(self, exp_folder_name, load_Q=True):

        # Flag whether the class instance has been loaded from the previously saved file or not.
        self.loaded_Q = False
        self.exp_folder_name = exp_folder_name

        # Location for storing the analysis folders
        self.saving_folder_location = saving_folder_location

        self.saving_file_name = 'class_instance.pckl'

        # Checking if the class instance has been saved before. If it was, then in case the user wants to load the data, it gets loaded. In all other cases the initialization continues.
        os.chdir(self.saving_folder_location)
        if os.path.isdir(self.exp_folder_name):
            print('The analysis instance has been previously saved.')
            if load_Q:
                self.load_instance()
                self.loaded_Q = True
        else:

            # Here we load the experiment data and make small changes to its structure. This is done for convenience for the subsequent analysis.

            os.chdir(data_folder)
            os.chdir(self.exp_folder_name)

            self.exp_data_df = pd.read_csv(filepath_or_buffer=data_file, delimiter=',', comment='#', header=0)

            # '180324-213144 - Waveguide Calibration - 41 Frequencies, Medium Range' data set, for instance, has some columns for the digitizer data not named properly. It uses 'Quenches' word instead of 'Generator'. These columns get renamed to they proper name.
            self.exp_data_df.rename(columns={
                'Digitizer DC (Quenches On) [V]': 'Digitizer DC (Generator On) [V]',
                'Digitizer DC (Quenches Off) [V]': 'Digitizer DC (Generator Off) [V]',
                'Digitizer STD (Quenches On) [V]': 'Digitizer STD (Generator On) [V]',
                'Digitizer STD (Quenches Off) [V]': 'Digitizer STD (Generator Off) [V]'
                }, inplace=True)

            # This same data set has additional space for the following columns that is removed.
            self.exp_data_df.rename(columns={
                'Waveguide A Power Reading  (Generator On) [V]': 'Waveguide A Power Reading (Generator On) [V]',
                'Waveguide A Power Reading  (Generator Off) [V]': 'Waveguide A Power Reading (Generator Off) [V]'
            }, inplace=True)

            # Convert data types of some columns
            self.exp_data_df = self.exp_data_df.astype({'Time': np.float64, 'Repeat': np.int16})

            # Check for abs(DC) values larger than 10 V. These indicate either saturated amplifier or a high voltage that exceedes range setting of a digitizer. This might not even indicate that there is something wrong with the digitizer settings, for instance, or the gain setting on the amplifier, since it could be just a sudden spike in the detector current.

            digi_dc_gen_on_large_df = self.exp_data_df[np.abs(self.exp_data_df['Digitizer DC (Generator On) [V]']) > 10]
            digi_dc_gen_off_large_df = self.exp_data_df[np.abs(self.exp_data_df['Digitizer DC (Generator Off) [V]']) > 10]

            if digi_dc_gen_on_large_df.shape[0] > 0:
                print('Digitizer DC (Generator On) values have been found that are larger than 10 Volts. There are ' + str(digi_dc_gen_on_large_df.shape[0]) + ' rows. Dropping these rows.')

            if digi_dc_gen_off_large_df.shape[0] > 0:
                print('Digitizer DC (Generator Off) values have been found that are larger than 10 Volts. There are ' + str(digi_dc_gen_off_large_df.shape[0]) + ' rows. Dropping these rows.')

            self.exp_data_df.drop(digi_dc_gen_on_large_df.index, inplace=True)
            self.exp_data_df.drop(digi_dc_gen_off_large_df.index, inplace=True)

            # Add a column of elapsed time since the start of the acquisition
            self.exp_data_df['Elapsed Time [s]'] = self.exp_data_df['Time'] - self.exp_data_df['Time'].min()

            # Get the data set parameters
            self.exp_params_dict, self.comment_string_arr = get_experiment_params(data_file)

            self.n_digi_samples = self.exp_params_dict['Number of Digitizer Samples']

            # Adding additional parameters into the Dictionary

            # Acquisition start time given in UNIX format = UTC time
            self.exp_params_dict['Experiment Start Time [s]'] = self.exp_data_df['Time'].min()

            # Acquisition duration [s]
            self.exp_params_dict['Experiment Duration [s]'] = self.exp_data_df['Elapsed Time [s]'].max()

            self.exp_data_df = self.exp_data_df.rename(columns={'Waveguide Frequency Setting [MHz]': freq_column_name})

            # Create pd.Series containing all of the experiment parameters.
            self.exp_params_s = pd.Series(self.exp_params_dict)

            # Waveguide power settings are set to whatever the np.linspace() function gave. However, the RF generator has only 0.1 dB resolution. Thus we need to round the values.

            # Rename the RF power setting column
            self.exp_data_df.rename(columns={'Waveguide Power Setting [dBm]': 'RF Generator Power Setting [dBm]'}, inplace=True)

            self.exp_data_df['RF Generator Power Setting [dBm]'] = self.exp_data_df['RF Generator Power Setting [dBm]'].transform(lambda x: round(x,1))

            # Trace STDs were calculated. We need STDOMs.
            self.exp_data_df['Digitizer STD (Generator Off) [V]'] = self.exp_data_df['Digitizer STD (Generator Off) [V]']/np.sqrt(self.n_digi_samples)

            self.exp_data_df['Digitizer STD (Generator On) [V]'] = self.exp_data_df['Digitizer STD (Generator On) [V]']/np.sqrt(self.n_digi_samples)

            # Calculate STD of On-Off ratios.
            self.exp_data_df.loc[:, 'Digitizer DC On/Off Ratio STD'] = np.sqrt((self.exp_data_df['Digitizer STD (Generator Off) [V]']/self.exp_data_df['Digitizer DC (Generator Off) [V]'])**2 + (self.exp_data_df['Digitizer STD (Generator On) [V]']/self.exp_data_df['Digitizer DC (Generator On) [V]'])**2) * self.exp_data_df['Digitizer DC On/Off Ratio']

            # Rename some columns
            self.exp_data_df.rename(columns={'Digitizer STD (Generator On) [V]': 'Digitizer DC STDOM (Generator On) [V]'}, inplace=True)

            self.exp_data_df.rename(columns={'Digitizer STD (Generator Off) [V]': 'Digitizer DC STDOM (Generator Off) [V]'}, inplace=True)

            self.index_column_list = ['Generator Channel', 'Repeat', freq_column_name, 'RF Generator Power Setting [dBm]', 'Average', 'Elapsed Time [s]']

            # Index used for the pd.DataFrame objects. It contains names of all of the parameters that were varied (intentionally) during the acquisition process.
            self.general_index = self.exp_data_df.reset_index().set_index(self.index_column_list).index

            self.exp_data_df = self.exp_data_df.set_index(self.index_column_list).sort_index()

            # Defining variables that the data analysis functions will assign the analyzed data to. These are needed so that whenever we want to call the function again, the analysis does not have to be redone. Also, most of the functions use data obtained from other function calls. We want to make it automatic for the function that needs other variables to call required functions. If the function has been called before we do not have to again wait for these functions to needlessly rerun the analysis.

            # Data acquired by the digitizer
            self.digitizer_data_df = None

            # Quenching cavities parameters dataframe
            self.quenching_cavities_df = None

            self.rf_system_power_df = None

            self.beam_dc_rf_off_df = None

            self.surv_frac_df = None

            self.surv_frac_av_df = None

    def get_quench_cav_data(self):
        ''' Obtain data that corresponds to both post- and pre-quenching cavity stacks.
        '''
        if self.quenching_cavities_df is None:
            # Grouping Quenching cavities data together
            level_value = 'Post-Quench'

            # Selected dataframe
            post_quench_df = self.exp_data_df[self.exp_data_df.columns.values[match_string_start(self.exp_data_df.columns.values, level_value)]]

            post_quench_df = post_quench_df.rename(columns=remove_matched_string_from_start(post_quench_df.columns.values, level_value))

            post_quench_df = add_level_data_frame(post_quench_df, 'Quenching Cavity', ['910', '1088', '1147'])

            post_quench_df.rename(columns={
            'Power Detector Reading (Generator On) [V]': 'On Power Detector Reading [V]',
            'Power Detector Reading (Generator Off) [V]': 'Off Power Detector Reading [V]',
            'Attenuator Voltage Reading (Generator On) [V]': 'On Attenuator Voltage Reading [V]',
            'Attenuator Voltage Reading (Generator Off) [V]': 'Off Attenuator Voltage Reading [V]'}, level='Data Field', inplace=True)

            post_quench_910_df = pd.concat([add_level_data_frame(post_quench_df['910'], 'RF Generator State', ['On', 'Off'])], keys=['910'], axis='columns', names=['Quenching Cavity'])

            post_quench_1088_df = pd.concat([add_level_data_frame(post_quench_df['1088'], 'RF Generator State', ['On', 'Off'])], keys=['1088'], axis='columns', names=['Quenching Cavity'])

            post_quench_1147_df = pd.concat([add_level_data_frame(post_quench_df['1088'], 'RF Generator State', ['On', 'Off'])], keys=['1147'], axis='columns', names=['Quenching Cavity'])

            post_quench_df = post_quench_910_df.join([post_quench_1088_df, post_quench_1147_df])

            level_value = 'Pre-Quench'
            # Selected dataframe
            pre_quench_df = self.exp_data_df[self.exp_data_df.columns.values[match_string_start(self.exp_data_df.columns.values, level_value)]]

            pre_quench_df = pre_quench_df.rename(columns=remove_matched_string_from_start(pre_quench_df.columns.values, level_value))

            pre_quench_df = add_level_data_frame(pre_quench_df, 'Quenching Cavity', ['910', '1088', '1147'])

            pre_quench_df.rename(columns={
            'Power Detector Reading (Generator On) [V]': 'On Power Detector Reading [V]',
            'Power Detector Reading (Generator Off) [V]': 'Off Power Detector Reading [V]',
            'Attenuator Voltage Reading (Generator On) [V]': 'On Attenuator Voltage Reading [V]',
            'Attenuator Voltage Reading (Generator Off) [V]': 'Off Attenuator Voltage Reading [V]'}, level='Data Field', inplace=True)

            pre_quench_910_df = pd.concat([add_level_data_frame(pre_quench_df['910'], 'RF Generator State', ['On', 'Off'])], keys=['910'], axis='columns', names=['Quenching Cavity'])

            pre_quench_1088_df = pd.concat([add_level_data_frame(pre_quench_df['1088'], 'RF Generator State', ['On', 'Off'])], keys=['1088'], axis='columns', names=['Quenching Cavity'])

            pre_quench_1147_df = pd.concat([add_level_data_frame(pre_quench_df['1088'], 'RF Generator State', ['On', 'Off'])], keys=['1147'], axis='columns', names=['Quenching Cavity'])

            pre_quench_df = pre_quench_910_df.join([pre_quench_1088_df, pre_quench_1147_df])
            #pre_quench_df.columns.names = ['Quenching Cavity', 'Data Field']

            # Combine pre-quench and post-quench data frames together
            quenching_cavities_df = pd.concat([pre_quench_df, post_quench_df], keys=['Pre-Quench','Post-Quench'], names=['Cavity Stack Type'], axis='columns')

            self.quenching_cavities_df = quenching_cavities_df

        return self.quenching_cavities_df

    def get_beam_dc_rf_off(self):
        ''' Convenience function for showing the DC of the beam, when the waveguides have no RF power going into themself.

            With this function one can determine the level of stability of the signal.
        '''
        if self.beam_dc_rf_off_df is None:
            if self.digitizer_data_df is None:
                self.get_digitizer_data()

            self.beam_dc_rf_off_df = self.digitizer_data_df.loc[(slice(None), slice(None), slice(None)), ('Generator Off')]
            self.beam_dc_rf_off_df = self.beam_dc_rf_off_df.sort_values(by='Elapsed Time [s]')

        return self.beam_dc_rf_off_df

    def get_surv_frac_data(self):

        if self.surv_frac_df is None:

            if self.digitizer_data_df is None:
                self.get_digitizer_data()

            if self.rf_system_power_df is None:
                self.get_rf_sys_pwr_det_data()

            rf_system_power_df_grouped = self.rf_system_power_df.groupby('Generator Channel')

            rf_sys_A_pwr_df = pd.DataFrame(rf_system_power_df_grouped.get_group('A')['Generator On']['Waveguide A Power Sensor Reading [V]']).rename(columns={'Waveguide A Power Sensor Reading [V]': 'RF System Power Sensor Reading [V]'})

            # Obtain object for KRYTAR 109B Power detector calibration
            krytar_109B_calib_obj = KRYTAR109BCalibration()
            krytar_109B_calib_obj.get_calib_curve()

            self.rf_sys_A_pwr_df = rf_sys_A_pwr_df
            self.rf_sys_A_pwr_df['RF System Power Sensor Detected Power [mW]'] = self.rf_sys_A_pwr_df['RF System Power Sensor Reading [V]'].transform(lambda x: krytar_109B_calib_obj.get_RF_power_from_voltage(x)[0])

            rf_sys_B_pwr_df = pd.DataFrame(rf_system_power_df_grouped.get_group('B')['Generator On']['Waveguide B Power Sensor Reading [V]']).rename(columns={'Waveguide B Power Sensor Reading [V]': 'RF System Power Sensor Reading [V]'})

            rf_sys_B_pwr_df['RF System Power Sensor Detected Power [mW]'] = rf_sys_B_pwr_df['RF System Power Sensor Reading [V]'].transform(lambda x: krytar_109B_calib_obj.get_RF_power_from_voltage(x)[0])

            rf_sys_pwr_df = pd.concat([rf_sys_A_pwr_df, rf_sys_B_pwr_df])

            #rf_sys_A_pwr_df.columns.names = ['Data Field']

            surv_frac_df = pd.DataFrame(self.digitizer_data_df['Other']['DC On/Off Ratio']).join(rf_sys_pwr_df)

            self.surv_frac_df = surv_frac_df

        return self.surv_frac_df

    def average_surv_frac_data(self):
        # For each repeat and each RF frequency we scan RF power. At each RF power setting we take N averages, which constitues the averaging set. Each averages constitues taking a single trace with the RF generator set to the required RF power setting (+ quench cavities set to their respective pi-pulses (except pre- and post-910 cavities)) and also another trace with the RF generator set to ouput close to -100 dBm (= off) and quench cavities not having any RF power going into them. Thus for every average we determine On/Off ratio.
        # One can imagine calculating weighted average of these ratios for every averaging set, and then averaging all of these values for all of the repeats together. However, since we do not have much data in averaging set, we have high chance of seeing an averaging set with very small uncertainty, which will not represent the true uncertainty in the data. Because of that it is better to simply average all of the data together for given RF frequency and RF power generator setting.
        # Another comment is that we should not trust the DC On/Off ratio STD obtained for every average. The first reason, is that simply looking at the data for given averaging set, I see that the variation in the ratios is larger than the uncertainty for each ratio. The second reason is that looking at the DC for when RF power is off shows that the variation in DC is very large from one value to another compared to uncertainty in DC. This means that the standard deviations are not a good indicator of the uncertainty in surviving fractions. Thus we indeed should look at the spread of surviving fractions for given RF frequency and RF generator power setting and determine the standard deviation from that.
        # There is one very importaint problem, however. As I have said before, when the RF Generator was set to -100 dBm, the RF synthesizers for the quench cavities were not outputting any power as well. Because of that, we do not have proper surviving ratios. One way of dealing with this problem is to assume that the largest surviving fractions, should be set to 1. Is this well justified? The lowest RF generator power setting in, for example, '180327-173323 - Waveguide Calibration - 41 Frequencies, Medium Range' data set is -25 dBm. This means that about -25+45-3 = 17 dBm of power is going into the waveguides, which is about 50 mW = 0.05 Watt. At 5 V/cm we need to put about -22 dBm = 2 times more power. At 5 V/cm, by looking at the simulation, the surviving fraction is about 0.95. Thus at half the power, the fractions is about 0.97-0.99. Not too bad. Because there is no better way for me to correct for this, I will normalize the data to largest surviving fraction.

        # Hopefully the last comment. Data set 180327-173323 - Waveguide Calibration - 41 Frequencies, Medium Range', has only a single repeat (it already took about 10 hours to acquire the data set) and 2 averages in an averaging set. Thus every RF power has only two values. Thus, even after 'averaging' over repeat(s) we still have high chance for observing some point with extremely small uncertainty. The only way to deal with this is to assume that the uncertainty should be the same for all of the RF frequencies. Thus, in general, for data sets like this (which is basically the way all of the older calibration data sets were), I should average all of the data together.
        # Why cannot I assume that the uncertainty is the same for all RF powers as well? It is because, when quenching large fraction of the signal (high power), the DC at that power cannot fluctuate by much w.r.t to the amount by which it can change when there is no RF power. Thus at high power, we expect to see less noise, because only noise in the DC when the RF generator is set to -100 dBm matters at high powers.

        # ================
        # IMPORTANT (2018-09-12)
        #
        # Need to consider offset in the RF power detector reading when the RF power generator is set to output no power. The offset is most likely due to the RF amplifier outputting lots of broadband noise even when no RF power is going in it.  However, it is hard to subtract this offset, because the RF power is not linear with the power detector voltage reading. And I do not know the exact function form of this relationship. It is also not clear, if this offset gets removed completely as soon as one gets RF power supplied into the amplifier. I have checked that at most the offset is 4% of the RF power detector reading (at -25 dBm of RF generator power). This is quite insignificant, because the AC Stark shift changes by the corresponding 4% as well at that low power, which is on the 5 kHz level.
        #
        # ================
        if self.surv_frac_av_df is None:

            if self.surv_frac_df is None:
                self.get_surv_frac_data()

            surv_frac_index_list = list(self.surv_frac_df.index.names)
            surv_frac_index_list.remove(freq_column_name)
            surv_frac_index_list.insert(surv_frac_index_list.index('RF Generator Power Setting [dBm]'), freq_column_name)
            surv_frac_index_list.remove('Repeat')
            surv_frac_index_list.insert(len(surv_frac_index_list), 'Repeat')
            surv_frac_index_list.remove('Average')
            surv_frac_index_list.insert(len(surv_frac_index_list), 'Average')

            surv_frac_to_average_df = self.surv_frac_df.reset_index().set_index(surv_frac_index_list).sort_index()

            # Perform the normalization.
            surv_frac_to_average_df['DC On/Off Ratio'] = surv_frac_to_average_df['DC On/Off Ratio'] / surv_frac_to_average_df['DC On/Off Ratio'].max()

            # Calculate the average surviving fraction
            surv_frac_to_average_index_list = list(surv_frac_to_average_df.index.names)
            surv_frac_to_average_index_list.remove('Elapsed Time [s]')
            surv_frac_to_average_index_list.remove('Repeat')
            surv_frac_to_average_index_list.remove('Average')

            surv_frac_df_grouped = surv_frac_to_average_df.groupby(surv_frac_to_average_index_list)

            mean_df = surv_frac_df_grouped.aggregate(lambda x: np.mean(x))

            # Needed for calculation of the pooled std.
            num_av_points_s = surv_frac_df_grouped['DC On/Off Ratio'].aggregate(lambda x: x.shape[0])
            num_av_points_s.name = 'Number Of Averaged Data Points'

            stdom_df = surv_frac_df_grouped.aggregate(lambda x: np.std(x, ddof=1)/np.sqrt(x.shape[0]))

            surv_frac_av_df = mean_df.join(stdom_df.rename(columns={'DC On/Off Ratio': 'DC On/Off Ratio STDOM', 'RF System Power Sensor Reading [V]': 'RF System Power Sensor Reading STDOM [V]', 'RF System Power Sensor Detected Power [mW]': 'RF System Power Sensor Detected Power STDOM [mW]'}).join(pd.DataFrame(num_av_points_s)))

            surv_frac_av_df['DC On/Off Ratio STD'] = surv_frac_av_df['DC On/Off Ratio STDOM'] * np.sqrt(surv_frac_av_df['Number Of Averaged Data Points'])

            # Calculated the pooled std = RMS std by combining data from all of the repeats and all of the waveguide frequencies, for given RF power and Generator channel, together.

            # Need to swap some levels for convenience.
            surv_frac_av_for_sigma_rms_df = surv_frac_av_df.swaplevel(j='RF Generator Power Setting [dBm]', i='Waveguide Carrier Frequency [MHz]').sort_index()


            # Calculate pooled std.
            surv_frac_av_for_sigma_rms_grouped_df = surv_frac_av_for_sigma_rms_df.groupby(['Generator Channel', 'RF Generator Power Setting [dBm]'])

            surv_frac_av_sigma_rms_df = surv_frac_av_for_sigma_rms_grouped_df.apply(lambda df: pd.Series(pooled_std(df['DC On/Off Ratio STD'].values, df['Number Of Averaged Data Points'].values))).rename(columns={0: 'DC On/Off Ratio RMS STD'})

            # Combined the pooled std (= rms std) with the rest of the data
            surv_frac_av_with_rms_unc_df = surv_frac_av_for_sigma_rms_df.reset_index().set_index(surv_frac_av_sigma_rms_df.index.names).join(surv_frac_av_sigma_rms_df, how='inner').reset_index().set_index(surv_frac_av_for_sigma_rms_df.index.names).sort_index()

            # Calculate RMS STDOM
            surv_frac_av_with_rms_unc_df['DC On/Off Ratio RMS STDOM'] = surv_frac_av_with_rms_unc_df['DC On/Off Ratio RMS STD'] / np.sqrt(surv_frac_av_with_rms_unc_df['Number Of Averaged Data Points'])

            # Form the dataframe to return. We want it to be have the same columns (and names) as we have in the code for analysing newer RF power calibration data sets.
            surv_frac_av_to_return_df = surv_frac_av_df.drop(columns=['DC On/Off Ratio STDOM', 'DC On/Off Ratio STD', 'Number Of Averaged Data Points'])

            surv_frac_av_to_return_df['DC On/Off Ratio STDOM'] = surv_frac_av_with_rms_unc_df.swaplevel(j='RF Generator Power Setting [dBm]', i='Waveguide Carrier Frequency [MHz]').sort_index()['DC On/Off Ratio RMS STDOM']

            # This data is not meant to be directly accessible, but it is useful to have it for making plots and etc.
            self.surv_frac_av_with_rms_unc_df = surv_frac_av_with_rms_unc_df

            self.surv_frac_av_df = surv_frac_av_to_return_df

        return self.surv_frac_av_df
#%%
# data_set = DataSetQuenchCurveWaveguideOld('180327-173323 - Waveguide Calibration - 41 Frequencies, Medium Range')
#
# quench_cav_data_df = data_set.get_quench_cav_data()
# digitizer_data_df = data_set.get_digitizer_data()
# rf_sys_pwr_det_data_df = data_set.get_rf_sys_pwr_det_data()
#
# fig, ax = plt.subplots()
# fig.set_size_inches(15, 10)
#
# ax = data_set.get_beam_dc_rf_off().reset_index().plot(kind='scatter', x='Elapsed Time [s]', y='DC [V]', yerr = 'DC STDOM [V]', ax=ax, color='black')
#
# #x_arr, y_arr = data_set.get_beam_dc_rf_off_plotting_data(ax)
#
# #ax.plot(x_arr, y_arr, color='blue')
#
# plt.show()
# # #%%
# average_surv_fract_data_df = data_set.average_surv_frac_data()
# average_surv_fract_data_df.loc['A', 907.8].plot(y='DC On/Off Ratio', style='.')
# #%%
# # Plot to see that we do indeed need to use RMS uncertainty, because some frequencies have very small uncertaintiy in the DC On/Off ratio.
# fig, ax = plt.subplots()
# fig.set_size_inches(12, 8)
# data_set.surv_frac_av_with_rms_unc_df.loc['A', -25.0].plot(kind='bar', use_index=True, y='DC On/Off Ratio STD', ax=ax)
# plt.show()
