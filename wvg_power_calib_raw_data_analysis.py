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
#from fosof_data_set_analysis import *
from ZX47_Calibration_analysis import *
from KRYTAR_109_B_Calib_analysis import *
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
# Experiment data file name
data_file = 'data.txt'

saving_folder_location = wvg_calib_data_folder_path

# File containing quench cavities' parameters.
quench_cav_params_file_name = 'filename.quench'

# Rename the frequency column to this particular value
freq_column_name = 'Waveguide Carrier Frequency [MHz]'

class DataSetQuenchCurveWaveguide():
    ''' General class for analyzing data sets for the Quench curves of the Waveguides. It contains all of the required analysis functions

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

            # It is also possible to have some DC values that are very off from the bulk of the DC values. This can happen when, for instance, the DC suddenly drops. The problem with this is that if this drop, for instance, is quite short, then at a given repeat and RF power setting all of the frequencies will be affected. It is better to drop these data sets.

            # This type of the issue is observed, for example, in '180522-195929 - Waveguide Calibration - 0 config, PD ON 82.5 V, 22.17 kV, 908-912 MHz'.

            # Maximum multiple of sigma that a DC value can be off from its mean.
            max_sigma_off = 5

            digi_dc_gen_off_diff_values_df = self.exp_data_df[(np.abs(self.exp_data_df['Digitizer DC (Generator Off) [V]']-self.exp_data_df['Digitizer DC (Generator Off) [V]'].mean())/self.exp_data_df['Digitizer DC (Generator Off) [V]'].std()) > max_sigma_off]

            if digi_dc_gen_off_diff_values_df.shape[0] > 0:
                print('Digitizer DC (Generator Off) values have been found that are more than ' + str(max_sigma_off) + ' sigma off from the average. There are ' + str(digi_dc_gen_off_diff_values_df.shape[0]) + ' rows. Dropping these rows.')

            self.exp_data_df.drop(digi_dc_gen_off_diff_values_df.index, inplace=True)

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

            self.index_column_list = ['Generator Channel', 'Repeat', 'RF Generator Power Setting [dBm]', freq_column_name, 'Elapsed Time [s]']

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

    def get_exp_data(self):
        return self.exp_data_df

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
            #post_quench_df.columns.set_names('Data Field', level=1, inplace=True)

            level_value = 'Pre-Quench'
            # Selected dataframe
            pre_quench_df = self.exp_data_df[self.exp_data_df.columns.values[match_string_start(self.exp_data_df.columns.values, level_value)]]

            pre_quench_df = pre_quench_df.rename(columns=remove_matched_string_from_start(pre_quench_df.columns.values, level_value))

            pre_quench_df = add_level_data_frame(pre_quench_df, 'Quenching Cavity', ['910', '1088', '1147'])
            #pre_quench_df.columns.names = ['Quenching Cavity', 'Data Field']

            # Combine pre-quench and post-quench data frames together
            quenching_cavities_df = pd.concat([pre_quench_df, post_quench_df], keys=['Pre-Quench','Post-Quench'], names=['Cavity Stack Type'], axis='columns').set_index(self.general_index)

            self.quenching_cavities_df = quenching_cavities_df

        return self.quenching_cavities_df

    def get_digitizer_data(self):
        if self.digitizer_data_df is None:
            digitizer_data_df = self.exp_data_df[self.exp_data_df.columns.values[match_string_start(self.exp_data_df.columns.values, 'Digitizer')]]

            digitizer_data_df = digitizer_data_df.rename(columns={
                'Digitizer DC (Generator On) [V]': 'Generator On DC [V]',
                'Digitizer DC (Generator Off) [V]': 'Generator Off DC [V]',
                'Digitizer DC STDOM (Generator On) [V]': 'Generator On DC STDOM [V]',
                'Digitizer DC STDOM (Generator Off) [V]': 'Generator Off DC STDOM [V]',
                'Digitizer DC On/Off Ratio': 'DC On/Off Ratio',
                'Digitizer DC On/Off Ratio STD': 'DC On/Off Ratio STD'})

            digitizer_data_df = add_level_data_frame(digitizer_data_df, 'Generator Power Output State', ['Generator On', 'Generator Off'])

            self.digitizer_data_df = digitizer_data_df
        return self.digitizer_data_df

    def get_rf_sys_pwr_det_data(self):

        rf_system_power_df = self.exp_data_df[self.exp_data_df.columns.values[match_string_start(self.exp_data_df.columns.values, 'Waveguide')]]

        rf_system_power_df = rf_system_power_df.rename(columns={
            'Waveguide A Power Reading (Generator On) [V]': 'Generator On Waveguide A Power Sensor Reading [V]',
            'Waveguide A Power Reading (Generator Off) [V]': 'Generator Off Waveguide A Power Sensor Reading [V]',
            'Waveguide B Power Reading (Generator On) [V]': 'Generator On Waveguide B Power Sensor Reading [V]',
            'Waveguide B Power Reading (Generator Off) [V]': 'Generator Off Waveguide B Power Sensor Reading [V]'
            })

        rf_system_power_df = add_level_data_frame(rf_system_power_df, 'Generator Power Output State', ['Generator On', 'Generator Off'])

        self.rf_system_power_df = rf_system_power_df

        return self.rf_system_power_df

    def get_beam_dc_rf_off(self):

        if self.beam_dc_rf_off_df is None:
            if self.digitizer_data_df is None:
                self.get_digitizer_data()

            freq_to_use = self.digitizer_data_df.reset_index()[freq_column_name].iloc[0]

            self.beam_dc_rf_off_df = self.digitizer_data_df.loc[(slice(None), slice(None), slice(None), slice(None)), ('Generator Off')]
            self.beam_dc_rf_off_df = self.beam_dc_rf_off_df.sort_values(by='Elapsed Time [s]')

        return self.beam_dc_rf_off_df

    def get_beam_dc_rf_off_plotting_data(self, ax):

        if self.beam_dc_rf_off_df is None:
            self.get_beam_dc_rf_off()

        # Data for the smoothing spline fit. Notice that the weights are 1/sigma, not 1/sigma**2, as specified in the function docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html#scipy.interpolate.splrep

        x_arr = self.beam_dc_rf_off_df.reset_index()['Elapsed Time [s]'].values
        y_arr = self.beam_dc_rf_off_df.reset_index()['DC [V]'].values
        y_weight_arr = 1/self.beam_dc_rf_off_df.reset_index()['DC STDOM [V]'].values

        # Determining the min and max values for the smoothing factor
        m = self.beam_dc_rf_off_df.reset_index()['DC STDOM [V]'].shape[0]
        s_min = m - np.sqrt(2*m)
        s_max = m + np.sqrt(2*m)

        # We use spline of order 4, not cubic spline, because later, when calculating zeros of the first derivative, we need to have cubic spline, otherwise we cannot calculate the zeros, because scipy has algorithm developed only for the cubic spline.

        smoothing_coeff = 1

        spl_smoothing = scipy.interpolate.UnivariateSpline(x=x_arr, y=y_arr, k=3, s=smoothing_coeff*s_max, w=y_weight_arr)

        x_spline_arr = np.linspace(np.min(x_arr),np.max(x_arr), 5*x_arr.shape[0])

        return x_spline_arr, spl_smoothing(x_spline_arr)

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

            # s_factor_multiple of 1000 seems to give good smooth agreement with the data.
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

        # Now we need to find the average the data. To do so we need first to talk about the way the data was acquired. For each repeat and each RF power we scan the RF frequency and acquire a single trace for each. After acquiring data for all of the RF frequencies the RF generator power gets set to about -140 dBm and a single trace gets acquired. With this information we can now think about analyzing the data.
        # One point that is immediately evident is that for each repeat and RF power setting if we divide the DC values for every RF frequency by the DC value for the case when RF power is off, then the errors in the fractional surviving populations are correlated, because the uncertainty in DC for RF power = off is the same for all of these surviving fractions. If we then average the surviving fractions for all of the repeats for given RF power setting, then the errors are correlated between different frequencies. Thus the final curve of surviving fraction vs RF frequency for each RF power has uncertainties that are correlated between different RF frequencies. Intuitively I believe this way we overestimate the uncertainty for each RF frequency for this final curve. I am not sure how to deal with this problem.
        # However, we can look at the data in a different way. We can look at the data with the same RF frequency. I.e., we can obtain the quenching curve = surviving population vs RF power. In this case every point has different value of DC when the RF power is off. Thus the errors are not correlated. What happens, however is that the quench curves for different frequencies have correlated uncertainties. I.e., if one curve is particularly noisy, because the DC values for RF power = off were noisy, then other quench curves are likely to be noisy.
        # Thus we can perform this second type of analysis to avoid correlations in a single curve.
        # There is another issue, however. It is possible that while taking data the beam might change its mode of operation - its DC value can suddenly change, even though its noise level might stay the same. If this 'mode hop' happens when we are close to finishing the RF frequency scan and then the RF power gets turned off, then the DC value that we will get will be quite different than the value we would have gotten if we measured the DC (RF power = off) before the mode hop. This will cause the surviving fractions to be quite different from the true values for most of the RF frequencies for this particular repeat and RF power. The main problem, however, is that for this case the standard deviation for this type of surviving fraction might not be any larger than the surviving fractions obtained at different times when the 'mode hop' did not happen.
        # Thus for a case like this taking the weighted average of surviving fractions will be a wrong thing to do, because the uncertainty in these fractions does not neceserily represent deviations from the true surviving fraction. In other words, in might be better to ignore the uncertainties in DC values altogether and simply calculate the standard deviation of the set of surviving fractions for given RF frequency and RF power. We can then also apply Chauvenet criterion to the data to determine if there are some obvious outliers.

        # Looking at the DC for when RF power is off shows that the variation in DC = in true value is very large from one value to another compared to uncertainty in DC. This means that the standard deviations are not a good indicator of the uncertainty in surviving fractions. Thus we indeed should look at the spread of surviving fractions for given RF frequency and RF power and determine the standard deviation from that.

        # =======================
        # Added on October 3 2018
        # =======================
        # While writing the analysis code for the older calibration data, namely '180327-173323 - Waveguide Calibration - 41 Frequencies, Medium Range', where we were switching the RF generator power state (On or Off) for every average, I had to use pooled standard deviation: I assumed that for all RF frequencies for given RF generator channel and RF power, the standard deviation should roughly be the same. I realized now that in principle I can use the same idea here. Thus I will combine standard deviations from all of the repeats and all fo the RF frequencies together (same generator channel + same RF power setting) and calculated pooled std = rms std.

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

            surv_frac_to_average_df = self.surv_frac_df.reset_index().set_index(surv_frac_index_list).sort_index()

            # Calculate the average surviving fraction
            surv_frac_to_average_index_list = list(surv_frac_to_average_df.index.names)
            surv_frac_to_average_index_list.remove('Elapsed Time [s]')
            surv_frac_to_average_index_list.remove('Repeat')

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

    def flush_data(self):
        ''' Clears analyzed/calculated data from the object.

            Useful when one has modified the data set in some way and wants to rerun the analysis on it.
        '''
        # Data acquired by the digitizer
        self.digitizer_data_df = None

        # Quenching cavities parameters dataframe
        self.quenching_cavities_df = None

        self.rf_system_power_df = None

        self.beam_dc_rf_off_df = None

        self.surv_frac_df = None

        self.surv_frac_av_df = None

    def load_instance(self):
        ''' This function loads previously pickled class instance.

        The data contained in the pickled file gets loaded into this class instance. The function is very useful if one does not want to reanalyze data again.
        '''
        os.chdir(self.saving_folder_location)
        os.chdir(self.exp_folder_name)

        f = open(self.saving_file_name, 'rb')
        loaded_dict = pickle.load(f)
        f.close()
        self.__dict__.update(loaded_dict)
        print('The class instance has been loaded')

        self.saving_folder_location = saving_folder_location

        os.chdir(self.saving_folder_location)

    def save_instance(self):
        ''' Calling this function pickles the analysis class instance. If the data has been previously saved, the call to this function overwrites previously written pickled file with the same file name.

        If the required function for certain types of data to save has not be called before, then it gets called here.
        '''

        # Created folder that will contain all of the analyzed data
        os.chdir(self.saving_folder_location)

        if os.path.isdir(self.exp_folder_name):
            print('Saving data folder already exists. It will be rewritten.')
            shutil.rmtree(self.exp_folder_name)

        os.mkdir(self.exp_folder_name)
        os.chdir(self.exp_folder_name)

        f = open(self.saving_file_name, 'wb')
        pickle.dump(self.__dict__, f, 2)

        os.chdir(self.saving_folder_location)

        print('The class instance has been saved')
#%%
# data_set = DataSetQuenchCurveWaveguide('180522-195929 - Waveguide Calibration - 0 config, PD ON 82.5 V, 22.17 kV, 908-912 MHz')
# #%%
# beam_dc_rf_off_df = data_set.get_beam_dc_rf_off()
# av_data_df = data_set.average_surv_frac_data()
# #%%
# rf_pwr_setting_list = list(data_set.surv_frac_av_with_rms_unc_df.index.get_level_values('RF Generator Power Setting [dBm]').drop_duplicates())
# print(rf_pwr_setting_list)
# #%%
# # Plot to see that we do indeed need to use RMS uncertainty, because some frequencies might have significantly smaller uncertaintiy in the DC On/Off ratio compared to its neighbours.
# fig, ax = plt.subplots()
# fig.set_size_inches(12, 8)
# data_set.surv_frac_av_with_rms_unc_df.loc['A', -2.8].plot(kind='bar', use_index=True, y='DC On/Off Ratio STDOM', ax=ax)
# plt.show()
# #%%
