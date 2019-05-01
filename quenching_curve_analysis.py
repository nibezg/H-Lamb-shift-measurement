from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string

#sys.path.insert(0,"C:/Users/Helium1/Google Drive/Code/Python/Testing/Blah") #
sys.path.insert(0,"E:/Google Drive/Research/Lamb shift measurement/Code")

from exp_data_analysis import *
from ZX47_Calibration_analysis import *
import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt

#%%
# Folder containing acquired data table
data_folder = "E:/Google Drive/Research/Lamb shift measurement/Data/Quenching curves"
# Experiment data file name
data_file = 'data.txt'

class DataSetQuenchCurveCavity():
    ''' General class for analyzing data sets for the Quenching curves of the Quenching cavities. It contains all of the required analysis functions
    '''

    def __init__(self, exp_folder_name):
        # Here we load the experiment data and make small changes to its structure. This is done for convenience for the subsequent analysis.

        os.chdir(data_folder)
        os.chdir(exp_folder_name)

        self.exp_data_df = pd.read_csv(filepath_or_buffer=data_file, delimiter=',', comment='#', header=0)

        # Convert data types of some columns
        self.exp_data_df = self.exp_data_df.astype({'Time': np.float64, 'Average': np.int16, 'Repeat': np.int16})

        # Add a column of elapsed time since the start of the acquisition
        self.exp_data_df['Elapsed Time [s]'] = self.exp_data_df['Time'] - self.exp_data_df['Time'].min()

        # Rename some columns
        self.exp_data_df.rename(columns={'Attenuator Voltage Setting [V]': 'Attenuator Setting [V]'}, inplace=True)

        # Calculate STD of On-Off ratios.
        self.exp_data_df.loc[:, 'Digitizer DC On/Off Ratio STD'] = np.sqrt((self.exp_data_df['Digitizer STD (Quenches Off) [V]']/self.exp_data_df['Digitizer DC (Quenches Off) [V]'])**2 + (self.exp_data_df['Digitizer STD (Quenches On) [V]']/self.exp_data_df['Digitizer DC (Quenches On) [V]'])**2) * self.exp_data_df['Digitizer DC On/Off Ratio']

        self.index_column_list = ['Repeat', 'Attenuator Setting [V]', 'Average', 'Elapsed Time [s]']

        def get_cavity_settings(cavity_name):
            ''' Determines quenching cavity parameters from its name

            The name is assumed to be given as cavity_stack_type-quench_freq, where cavity_stack_type = pre or post, and freq = cavity resonant frequency [MHz], given as integer.
            '''
            # Regular expression
            reg_exp = re.compile( r"""(?P<pre_or_post>pre|post)  # pre- or post- cavity that is getting scanned
                                -quench_(?P<cavity_frequency>[\d]+) # (MHz) RF frequency of the quenching cavity that is getting scanned. (Float).
                                    """,
                            re.VERBOSE)
            cavity_name_parsed_object = reg_exp.search(cavity_name)
            if cavity_name_parsed_object.group('pre_or_post') == 'pre':
                cavity_stack = 'Pre-Quench'

            if cavity_name_parsed_object.group('pre_or_post') == 'post':
                cavity_stack = 'Post-Quench'

            return {'Cavity Stack': cavity_stack,
            'Cavity RF Frequency [MHz]': int(cavity_name_parsed_object.group('cavity_frequency'))}

        # Get the data set parameters
        self.exp_params_dict, self.comment_string_arr = get_experiment_params(data_file)

        # Determines the stack (pre or post) and the RF frequency supplied to the cavity.
        self.cavity_params_dict = get_cavity_settings(self.exp_params_dict['Quench Cavity to Scan'])


        # Adding additional parameters into the Dictionary

        # Acquisition start time given in UNIX format = UTC time
        self.exp_params_dict['Experiment Start Time [s]'] = self.exp_data_df['Time'].min()

        # Acquisition duration [s]
        self.exp_params_dict['Experiment Duration [s]'] = self.exp_data_df['Elapsed Time [s]'].max()

        # Cavity stack to which the quenching cavity, which power was getting scanned, belongs.
        self.exp_params_dict['Cavity Stack'] = self.exp_data_df['Elapsed Time [s]'].max()

        # Cavity stack to which the quenching cavity, which power was getting scanned, belongs.
        self.exp_params_dict['Cavity Stack'] = self.cavity_params_dict['Cavity Stack']

        # Name of the Quenching cavity.
        self.exp_params_dict['Cavity RF Frequency [MHz]'] = self.cavity_params_dict['Cavity RF Frequency [MHz]']

        # Create pd.Series containing all of the experiment parameters.
        self.exp_params_s = pd.Series(self.exp_params_dict)

        # Index used for the pd.DataFrame objects. It contains names of all of the parameters that were varied (intentionally) during the acquisition process.
        self.general_index = self.exp_data_df.reset_index().set_index(self.index_column_list).index

        self.exp_data_df = self.exp_data_df.set_index(self.index_column_list).sort_index()

        # Defining variables that the data analysis functions will assign the analyzed data to. These are needed so that whenever we want to call the function again, the analysis does not have to be redone. Also, most of the functions use data obtained from other function calls. We want to make it automatic for the function that needs other variables to call required functions. If the function has been called before we do not have to again wait for these functions to needlessly rerun the analysis.

        # pd.DataFrame object containing RF power detector calibration information
        self.power_det_calib_df = None

        # Quenching cavities parameters dataframe
        self.quenching_cavities_df = None

        # Averaged data for the quenching cavities' parameters.
        self.quenching_cavities_av_df = None

        # Averaged data for the quenching curve
        self.exp_data_averaged_df = None

        # RF attenuation voltage corresponding to pi-pulse = first minimum of the quenching curve.
        self.pi_pulse_att_volt = None

    def get_exp_parameters(self):
        ''' Simply returns the pd.Series with the experiment parameters.
        '''
        return self.exp_params_s

    def get_quench_cav_rf_pwr_det_calib(self):
        ''' For the Quench cavities quench curves: In addition to determining the pi-pulse location we want to see if the quenching curve itself agrees with simulations. For this we need to use the readings obtained from the power detectors installed for each cavity. This way we can convert set RF Attenuator voltages to RF powers. For other data sets: We want to monitor power that goes into the quenching cavities.

        The power detectors used are Mini-Circuits ZX47-55LN-S+. They are positioned on the CPL IN output of the bi-directional couplers.

        The RF calibration is performed for all three RF frequencies: 910, 1088, and 1147 MHz.

        Outputs:

        :power_det_calib_df: pd.DataFrame object containing RF power detector calibration for all three RF frequencies.
        '''

        if self.power_det_calib_df is None:

            self.power_det_calib_df = pd.DataFrame([])

            # Three RF frequencies used in the Quenching cavities.
            cavity_rf_freq_list = [910, 1088, 1147]

            # Perform RF power detector calibration analysis for every RF frequency.
            for rf_freq_index in range(len(cavity_rf_freq_list)):

                pow_det_calib = ZX4755LNCalibration(cavity_rf_freq_list[rf_freq_index])

                power_det_calib_func, calib_frac_unc = pow_det_calib.get_calib_curve()

                calib_data_df = pow_det_calib.get_calib_data()

                x_data, y_data = pow_det_calib.get_spline_data_to_plot(n_points=1000)

                pow_det_calib_s = pd.Series(
                                {   'Calibration Function': power_det_calib_func,
                                    'Calibration Fractional Uncertainty': calib_frac_unc,
                                    'Calibration Data': calib_data_df,
                                    'Calibration Function x-Axis Plotting Data': x_data,
                                    'Calibration Function y-Axis Plotting Data': y_data,
                                    'RF Power Detector Model': pow_det_calib.get_detector_model()
                                })

                pow_det_calib_s.name = cavity_rf_freq_list[rf_freq_index]

                if self.power_det_calib_df.size == 0:
                    self.power_det_calib_df = pd.DataFrame([pow_det_calib_s])
                else:
                    self.power_det_calib_df = self.power_det_calib_df.append(pow_det_calib_s)

        return self.power_det_calib_df

    def get_quenching_cav_data(self):
        ''' Obtain data that corresponds to both post- and pre-quench cavity stacks.

        Stability of the various parameters for the quenching cavities is included, as well as the conversion of the RF power detector readings in Volts to corresponding RF Power values in dBm and Watts.
        '''

        if self.quenching_cavities_df is None:

            if self.power_det_calib_df is None:
                self.get_quench_cav_rf_pwr_det_calib()

            # Get Quenching cavities RF parameters, which include the RF attenuators
            level_value = 'Post-Quench'

            post_quench_df = self.exp_data_df[self.exp_data_df.columns.values[match_string_start(self.exp_data_df.columns.values, level_value)]]

            post_quench_df = post_quench_df.rename(columns=remove_matched_string_from_start(post_quench_df.columns.values, level_value))

            post_quench_df = add_level_data_frame(post_quench_df, 'Quenching Cavity', ['910', '1088', '1147'])

            level_value = 'Pre-Quench'
            # Selected dataframe
            pre_quench_df = self.exp_data_df[self.exp_data_df.columns.values[match_string_start(self.exp_data_df.columns.values, level_value)]]

            pre_quench_df = pre_quench_df.rename(columns=remove_matched_string_from_start(pre_quench_df.columns.values, level_value))

            pre_quench_df = add_level_data_frame(pre_quench_df, 'Quenching Cavity', ['910', '1088', '1147'])
            #pre_quench_df.columns.names = ['Quenching Cavity', 'Data Field']

            pre_quench_df = pre_quench_df.rename(columns={'Attenuator Voltage Reading (Quenches On) [V]' : 'On Attenuator Reading [V]', 'Attenuator Voltage Reading (Quenches Off) [V]' : 'Off Attenuator Reading [V]', 'Power Detector Reading (Quenches Off) [V]' : 'Off Power Detector Reading [V]', 'Power Detector Reading (Quenches On) [V]' : 'On Power Detector Reading [V]'}, level='Data Field')

            post_quench_df = post_quench_df.rename(columns={'Attenuator Voltage Reading (Quenches On) [V]' : 'On Attenuator Reading [V]', 'Attenuator Voltage Reading (Quenches Off) [V]' : 'Off Attenuator Reading [V]', 'Power Detector Reading (Quenches Off) [V]' : 'Off Power Detector Reading [V]', 'Power Detector Reading (Quenches On) [V]' : 'On Power Detector Reading [V]'}, level='Data Field')


            pre_910_df = pd.concat([add_level_data_frame(pre_quench_df['910'], 'Cavity State', ['On', 'Off'])], keys=['910'], axis='columns', names=['Cavity Type'])
            pre_1088_df = pd.concat([add_level_data_frame(pre_quench_df['1088'], 'Cavity State', ['On', 'Off'])], keys=['1088'], axis='columns', names=['Cavity Type'])
            pre_1147_df = pd.concat([add_level_data_frame(pre_quench_df['1147'], 'Cavity State', ['On', 'Off'])], keys=['1147'], axis='columns', names=['Cavity Type'])

            post_910_df = pd.concat([add_level_data_frame(post_quench_df['910'], 'Cavity State', ['On', 'Off'])], keys=['910'], axis='columns', names=['Cavity Type'])
            post_1088_df = pd.concat([add_level_data_frame(post_quench_df['1088'], 'Cavity State', ['On', 'Off'])], keys=['1088'], axis='columns', names=['Cavity Type'])
            post_1147_df = pd.concat([add_level_data_frame(post_quench_df['1147'], 'Cavity State', ['On', 'Off'])], keys=['1147'], axis='columns', names=['Cavity Type'])

            pre_quench_df = pd.concat([pre_910_df, pre_1088_df, pre_1147_df], axis='columns').sort_index(axis='columns')

            post_quench_df = pd.concat([post_910_df, post_1088_df, post_1147_df], axis='columns').sort_index(axis='columns')

            pre_quench_df = pd.concat([pre_quench_df], keys=['Pre-Quench'], names=['Cavity Stack Type'], axis='columns')

            post_quench_df = pd.concat([post_quench_df], keys=['Post-Quench'], names=['Cavity Stack Type'], axis='columns')

            # Combine pre-quench and post-quench data frames together
            quenching_cavities_df = pd.concat([pre_quench_df, post_quench_df], axis='columns')

            # This is the amount by which the detected power is different from the power that goes into the quenching cavities.
            attenuation = 10 + 30 # [dBm]

            cavity_rf_freq_list = [910, 1088, 1147]

            converted_power_list = []

            # Apply proper RF detector calibration function for each Quenching cavity
            for rf_freq_index in range(len(cavity_rf_freq_list)):

                rf_freq = cavity_rf_freq_list[rf_freq_index]

                # Convert power detector voltages to RF powers.
                pw_detected_df = quenching_cavities_df.loc[slice(None), (slice(None), str(rf_freq), 'On', 'Power Detector Reading [V]')].transform(self.power_det_calib_df.loc[rf_freq]['Calibration Function']).rename(columns={'Power Detector Reading [V]': 'Detected Power [dBm]'}, level='Data Field')

                pw_system_df = (pw_detected_df + attenuation).rename(columns={'Detected Power [dBm]': 'RF System Power [dBm]'}, level='Data Field')

                pw_df = pd.concat([pw_detected_df, pw_system_df], axis='columns').sort_index(axis='columns')

                pw_watt_df = pw_df.transform(lambda x: 10**(x/10-3)).rename(columns={'Detected Power [dBm]': 'Detected Power [W]', 'RF System Power [dBm]': 'RF System Power [W]'})

                pd_watt_std_df = (pw_watt_df*self.power_det_calib_df.loc[rf_freq]['Calibration Fractional Uncertainty']).rename(columns={'Detected Power [W]': 'Detected Power STD [W]', 'RF System Power [W]': 'RF System Power STD [W]'})

                converted_power_list.append(pd.concat([pw_df, pw_watt_df, pd_watt_std_df], axis='columns').sort_index(axis='columns'))

            quenching_cavities_df = pd.concat([quenching_cavities_df, pd.concat(converted_power_list, axis='columns').sort_index(axis='columns')], axis='columns').sort_index(axis='columns')


            # We want to monitor the stability of the RF power detected.

            quenching_cavities_repeat_1 = quenching_cavities_df.reset_index().set_index('Elapsed Time [s]').sort_index()

            # We first select data that corresponds to the first repeat only. Then we select only first occurences (in time) of RF attenuator voltage.
            quenching_cavities_initial_df = quenching_cavities_repeat_1.loc[quenching_cavities_repeat_1[quenching_cavities_repeat_1['Repeat'] == 1]['Attenuator Setting [V]'].sort_index().drop_duplicates(keep='first').sort_index().index]

            # We need to remove all other index levels, except that of the 'Attenuator Setting [V]', because when subtracting from the list of phase differences, we want to subtract the same phase difference from each phase, independent of other index levels.

            quenching_cavities_initial_column_list = list(self.general_index.names)
            quenching_cavities_initial_column_list.remove('Attenuator Setting [V]')

            quenching_cavities_initial_df = quenching_cavities_initial_df.reset_index().set_index('Attenuator Setting [V]').drop(columns=quenching_cavities_initial_column_list, level='Cavity Stack Type')
            # We now subtract this list of initial measured RF parameters from the rest of the data frame with the measured RF parameters.

            quenching_cavities_change_df = (quenching_cavities_df - quenching_cavities_initial_df)

            # Calculating fractional change in RF parameters.
            quenching_cavities_frac_change_df = quenching_cavities_change_df / quenching_cavities_initial_df * 1E3

            # Dropping unnecessary columns
            quenching_cavities_change_df = quenching_cavities_change_df.drop(labels=['Detected Power STD [W]', 'Detected Power [dBm]', 'RF System Power STD [W]', 'RF System Power [dBm]'], axis='columns', level='Data Field')

            quenching_cavities_frac_change_df = quenching_cavities_change_df.drop(labels=['Detected Power STD [W]', 'Detected Power [dBm]', 'RF System Power STD [W]', 'RF System Power [dBm]'], axis='columns', level='Data Field')

            # Renaming the df's
            quenching_cavities_frac_change_df.rename(columns={'Attenuator Reading [V]': 'Fractional Change In Attenuator Reading [ppt]', 'Power Detector Reading [V]': 'Fractional Change In Power Detector Reading [ppt]', 'Detected Power [W]': 'Fractional Change In Detected Power [ppt]', 'RF System Power [W]': 'Fractional Change In RF System Power [ppt]'}, inplace=True, level='Data Field')

            quenching_cavities_change_df.rename(columns={'Attenuator Reading [V]': 'Change In Attenuator Reading [V]', 'Power Detector Reading [V]': 'Change In Power Detector Reading [V]', 'Detected Power [W]': 'Change In Detected Power [W]', 'RF System Power [W]': 'Change In RF System Power [W]'}, inplace=True, level='Data Field')

            # Combining all the quenching parameters together
            quenching_cavities_df = pd.concat([quenching_cavities_df, quenching_cavities_frac_change_df, quenching_cavities_change_df], axis='columns').sort_index(axis='columns')

            self.quenching_cavities_df = quenching_cavities_df

        return self.quenching_cavities_df

    def get_av_quenching_cav_data(self):
        ''' For every Attenuator voltage setting we want to find average power that was going into each quenching cavity. This power will then be used as our x-axis for plotting the surviving population vs RF power with proper error bars.

        For the cases when the RF syntheziers were not supplying power to the quenching cavities, we cannot determine the RF power from the power detector reading, because our RF power detector calibration does not go that low in power. Thus, we simply calculate the average power detector reading and its STDOM, simply to see if the voltage reading was at least stable.
        '''
        if self.quenching_cavities_av_df is None:

            if self.quenching_cavities_df is None:
                self.get_quenching_cav_data(self)

            # Picking needed columns from the quenching parameters data for averaging across all of the repeats.
            quenching_cavities_off_state_averaging_df = self.quenching_cavities_df.loc[slice(None), (slice(None), slice(None), 'Off', ['Attenuator Reading [V]', 'Power Detector Reading [V]'])]

            # We do not care about averaging for repeats separately here, because it is very reasonable to assume that the devices have the same level of noise for all of the repeats. It is not like with the data taken from the Detector that depends on the beam stability, which is known to change on a several repeats time scale.
            quenching_cavities_off_state_averaging_group = quenching_cavities_off_state_averaging_df.groupby('Attenuator Setting [V]')

            # Calculate mean and STDOM
            quenching_cavities_off_state_mean_df = quenching_cavities_off_state_averaging_group.aggregate(lambda x: np.mean(x))

            quenching_cavities_off_state_std_df = quenching_cavities_off_state_averaging_group.aggregate(lambda x: np.std(x, ddof=1)/np.sqrt(x.shape[0]))

            quenching_cavities_off_state_std_df.rename(columns={'Attenuator Reading [V]': 'Attenuator Reading STDOM [V]', 'Power Detector Reading [V]': 'Power Detector Reading STDOM [V]'}, inplace=True)

            # Combined averaged data frame for the RF parameters
            quenching_cavities_off_state_av_df = pd.concat([quenching_cavities_off_state_mean_df, quenching_cavities_off_state_std_df], axis='columns').sort_index(axis='columns')

            # Now for the case when the power is supplied to the quenching cavities, we can calculate average power and its standard deviation. We already have uncertainties for the detected power obtained from the calibration. However, this uncertainty is assumed to be of systematic nature: not due to insufficient data taken, but due to not knowing the exact calibration function to which the data needs to be fitted. Thus, we should not be taking weighted average of powers here. We should simply find the average power and then use the calibration curve to determine the uncertainty in the power.

            # Picking needed columns from the quenching parameters data for averaging across all of the repeats.
            quenching_cavities_averaging_df = self.quenching_cavities_df.loc[slice(None), (slice(None), slice(None), 'On', ['Attenuator Reading [V]', 'RF System Power [W]'])]

            # We do not care about averaging for repeats separately here, because it is very reasonable to assume that the devices have the same level of noise for all of the repeats. It is not like with the data taken from the Detector that depends on the beam stability, which is known to change on a several repeats time scale.
            quenching_cavities_averaging_group = quenching_cavities_averaging_df.groupby('Attenuator Setting [V]')

            # Calculate mean and STDOM
            quenching_cavities_mean_df = quenching_cavities_averaging_group.aggregate(lambda x: np.mean(x))

            quenching_cavities_std_df = quenching_cavities_averaging_group.aggregate(lambda x: np.std(x, ddof=1)/np.sqrt(x.shape[0]))

            quenching_cavities_std_df.rename(columns={'Attenuator Reading [V]': 'Attenuator Reading STDOM [V]', 'RF System Power [W]': 'RF System Power STDOM [W]'}, inplace=True)

            # Combined averaged data frame for the RF parameters
            quenching_cavities_av_df = pd.concat([quenching_cavities_mean_df, quenching_cavities_std_df], axis='columns').sort_index(axis='columns')

            # STDOM calculated for the RF System power is the statistical uncertainty due to noise in the power detector voltages. In addition to that we incorporate the systematic uncertinaty in the calibration that gets added in quadrature.

            cavity_rf_freq_list = [910, 1088, 1147]

            # Pick proper RF power calibration uncertainty for each quenching cavity
            for rf_freq_index in range(len(cavity_rf_freq_list)):

                rf_freq = cavity_rf_freq_list[rf_freq_index]
                calib_frac_unc = self.power_det_calib_df.loc[rf_freq]['Calibration Fractional Uncertainty']

                quenching_cavities_av_df.loc[slice(None), (slice(None), str(rf_freq), slice(None), 'RF System Power STDOM [W]')] = np.sqrt(quenching_cavities_av_df.loc[slice(None), (slice(None), str(rf_freq), slice(None), 'RF System Power STDOM [W]')]**2 +  (quenching_cavities_av_df.loc[slice(None), (slice(None), str(rf_freq), slice(None), 'RF System Power [W]')].values * calib_frac_unc)**2)

            quenching_cavities_av_df = pd.concat([quenching_cavities_av_df, quenching_cavities_off_state_av_df], axis='columns').sort_index(axis='columns')

            self.quenching_cavities_av_df = quenching_cavities_av_df

        return self.quenching_cavities_av_df

    def get_quenching_curve(self):
        ''' Calculate average quenching curve
        '''

        if self.exp_data_averaged_df is None:

            if self.quenching_cavities_av_df is None:
                self.get_av_quenching_cav_data()

            columns_no_av_list = list(self.general_index.names)

            columns_no_av_list.remove('Average')
            columns_no_av_list.remove('Elapsed Time [s]')

            exp_data_df_group = self.exp_data_df.groupby(columns_no_av_list)

            exp_data_averages_combined_df = exp_data_df_group.apply(lambda df: straight_line_fit_params(data_arr=df['Digitizer DC On/Off Ratio'].values, sigma_arr=df['Digitizer DC On/Off Ratio STD'].values))

            columns_no_repeat_list = list(exp_data_averages_combined_df.index.names)
            columns_no_repeat_list.remove('Repeat')

            exp_data_averages_combined_df.columns.names = ['Data Field']

            # Averaging of repeats, obviously, needs to be performed only if there is more than 1 repeat of data.
            if self.exp_params_dict['Number of Repeats'] > 1:
                exp_data_averages_combined_group = exp_data_averages_combined_df.groupby(columns_no_repeat_list)
                exp_data_averaged_df = exp_data_averages_combined_group.apply(lambda df: straight_line_fit_params(data_arr=df['Weighted Mean'].values, sigma_arr=df['Weighted STD'].values))

            else:
                exp_data_averaged_df = exp_data_averages_combined_df.reset_index('Repeat').drop('Repeat', axis='columns')


            exp_data_averaged_df = pd.concat([exp_data_averaged_df, self.quenching_cavities_av_df[self.exp_params_s['Cavity Stack'], str(self.exp_params_s['Cavity RF Frequency [MHz]']), 'On'][['RF System Power [W]', 'RF System Power STDOM [W]']]], axis='columns').sort_index(axis='columns')

            self.exp_data_averaged_df = exp_data_averaged_df

        return self.exp_data_averaged_df

    def perform_quenching_analysis(self):
        ''' Calculates parameters needed for conversion of quenching surviving ratio to corresponding RF attenuation voltage. This function does not return any parameters.

        It is assumed that observed offset is constant for all attenuation voltages.
        '''

        # Here we simply would want to calculate that location of the pi-pulse. I.e., we want to determine the attenuator voltage that results in the first local minimum in the quenching curve. Of course, the proper way of doing it would be to fit the data (surviving fraction vs RF Power in the quenching cavity) to the expected quenching curve, and extract the pi-pulse location from the fit. There are, however, several issues with this idea. Firstly, we actually do not know the RF power inside the quenching cavity. If we had perfect RF power detectors we could get at least power that is proportional to the RF power inside the quenching cavity. But the power detectors that we have are not that perfect. Secondly, the simulation that of the quenching curve might not be perfect, especially considering the observed offset in the quenching curve that the simulation does not explain.

        # Instead we simply find the first local minimum in the quenching curve and assume that it is the pi-pulse. The danger is that we are not calculating the correct pi-pulse, because the quenching curve might be quite different from the expected one.

        # To determine the first local minimum one can simply fit the data to high-order polynomial. However, this is not desirable, because we will just force the polynomial to pass through every data point, even though the real quenching curve might have different shape. This might give wrong pi-pulse location. What we know about the quenching curve is that it is expected to have more or less quadratic behavior close to its pi-pulse. Especially since we are not trying to see whether the quenching curve matches the expected simulated quenching curve, we should not be worrying about all of the data, but should concentrate on the data around the first local minimum.

        # To determine the data around the first local minimum I could have visually looked at the data and picked the data around the pi-pulse manually. I decided to automate this procedure. First, I fit smoothing spline to the data. This spline could be used to determine pretty accurate location of the pi-pulse. This tentative pi-pulse location is then used to select data range around the pi-pulse. Then I perform weighted least squares fit to a polynomial of relatively low order. This procedure assumes that the data is not overly noisy, otherwise the spline might have spurious local minima, and the whole procedure fails. It also assumes that the density of points is large enough for having > 10 data points around the pi-pulse.

        # Minimum and maximum attenuator voltages for the spline fit.
        x_min = self.exp_data_averaged_df.reset_index()['Attenuator Setting [V]'].min()
        x_max = self.exp_data_averaged_df.reset_index()['Attenuator Setting [V]'].max()

        # x-axis range to use for plotting spline later on.
        x_spline_fit_arr = np.linspace(x_min, x_max, 100)

        # Data for the smoothing spline fit. Notice that the weights are 1/sigma, not 1/sigma**2, as specified in the function docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html#scipy.interpolate.splrep
        x_arr = self.exp_data_averaged_df.reset_index()['Attenuator Setting [V]'].values
        y_arr = self.exp_data_averaged_df.reset_index()['Weighted Mean'].values
        y_weight_arr = 1/self.exp_data_averaged_df.reset_index()['Weighted STD'].values

        # Determining the min and max values for the smoothing factor
        m = self.exp_data_averaged_df.reset_index()['Weighted STD'].shape[0]
        s_min = m - np.sqrt(2*m)
        s_max = m + np.sqrt(2*m)

        # Seems that s_min is still not enough smoothing for average quenching curve observed. We use s = smoothing_coeff*s_min, where smoothing_coeff < 1. About 0.1 seems to give good fit.
        smoothing_coeff = 0.1

        # We use spline of order 4, not cubic spline, because later, when calculating zeros of the first derivative, we need to have cubic spline, otherwise we cannot calculate the zeros, because scipy has algorithm developed only for the cubic spline.

        spl_smoothing = scipy.interpolate.UnivariateSpline(x=x_arr, y=y_arr, k=4, s=smoothing_coeff*s_min, w=1/self.exp_data_averaged_df.reset_index()['Weighted STD'].values)

        # Calculate zeros of the smoothing spline 1st derivative
        spl_smoothing_roots = np.sort(spl_smoothing.derivative(n=1).roots())

        # Quenching surviving fractions at the minima/maxima attenuator voltages
        zero_der_vals_spline = spl_smoothing(spl_smoothing_roots)

        # We need the first point the is the local minimum. For this we can calculate 2nd derivative and find the first occurence of positive second derivate at the locations where the first derivative is zero.
        spline_second_der = spl_smoothing.derivative(n=2)

        pi_pulse_spline_index = np.min(np.argwhere(spline_second_der(spl_smoothing_roots) > 0))

        # Location of the pi-pulse
        pi_pulse_att_volt_spline = spl_smoothing_roots[pi_pulse_spline_index]

        # Quenching surviving fraction at the pi-pulse determined from the spline fit
        quenching_offset_spline = spl_smoothing([pi_pulse_att_volt_spline])[0]

        # We now need to pick data around the pi-pulse. The data to pick is the one between two adjacent minima/maxima. In case, if there are no more minima/maxima to either left or right, we include the data down to minimum or up to maximum attenuator voltage.
        if pi_pulse_spline_index-1 >= 0:
            min_att_V = spl_smoothing_roots[pi_pulse_spline_index-1]
        else:
            min_att_V = x_min

        if pi_pulse_spline_index+1 <= spl_smoothing_roots.shape[0]-1:
            max_att_V = spl_smoothing_roots[pi_pulse_spline_index+1]
        else:
            max_att_V = x_max

        # Close to the adjacent maxima/minima the data states to curve. We are not really interested in this, thus we take only fraction of this range. This might need to get changed.
        frac_range = 0.5

        # Attenuator voltages around the pi-pulse
        x_safe_fit_range_arr = np.sort(x_arr[np.where( (x_arr <= pi_pulse_att_volt_spline+(max_att_V-pi_pulse_att_volt_spline) * frac_range) & (x_arr >=  min_att_V+(pi_pulse_att_volt_spline - min_att_V) * (1-frac_range)) )])

        # Data around the pi-pulse
        exp_data_averaged_fit_range_df = self.exp_data_averaged_df.loc[x_safe_fit_range_arr]

        # Minimum and maximum attenuator voltages for the chosen data range.
        x_fit_arr_min = exp_data_averaged_fit_range_df.reset_index()['Attenuator Setting [V]'].min()
        x_fit_arr_max = exp_data_averaged_fit_range_df.reset_index()['Attenuator Setting [V]'].max()

        # Plotting range for the polynomial fit to the chosen data around the pi-pulse
        x_poly_fit_arr = np.linspace(x_fit_arr_min, x_fit_arr_max, 100)

        # Data for the polynomial fit to the data around the pi-pulse
        x_fit_range_arr = exp_data_averaged_fit_range_df.reset_index()['Attenuator Setting [V]'].values
        y_fit_range_arr = exp_data_averaged_fit_range_df.reset_index()['Weighted Mean'].values
        y_fit_range_weight_arr = (1/exp_data_averaged_fit_range_df.reset_index()['Weighted STD']**2).values

        # Weighted polynomial fit

        # Polynomial fit order. This might need to get changed.
        pol_fit_order = 5

        quenching_polyfit = np.polyfit(x_fit_range_arr, y_fit_range_arr, deg=pol_fit_order, w=y_fit_range_weight_arr)

        quenching_poly_func = np.poly1d(quenching_polyfit)

        # Calculate attenuator voltages that correspond to zero first derivative of the polynomial
        poly_fit_der_roots = np.roots(np.polyder(quenching_poly_func, m=1))

        # Only real roots are sensible
        poly_fit_der_real_roots = np.abs(poly_fit_der_roots[np.isreal(poly_fit_der_roots)])

        poly_fit_der_real_roots = np.sort(poly_fit_der_real_roots)

        # Quenching surviving fractions corresponding to zero derivative
        zero_der_vals = quenching_poly_func(poly_fit_der_real_roots)

        # We need the first point the is the local minimum. For this we can calculate 2nd derivative and find the first occurence of positive second derivate at the locations where the first derivative is zero.
        poly_fit_second_der = np.polyder(quenching_poly_func, m=2)

        # Location of the pi-pulse
        pi_pulse_att_volt = poly_fit_der_real_roots[np.min(np.argwhere(poly_fit_second_der(poly_fit_der_real_roots) > 0))]

        # Quenching surviving fraction at the pi-pulse
        quenching_offset = quenching_poly_func(pi_pulse_att_volt)

        # We also want to find the maximum value of the surviving fraction. This is needed to calculate attenuation voltage needed to quench certain fraction of the given atomic state. The problem is that when x-axis is the Attenuator Voltage, then polynomial and spline fits to the initial portion of the data (low attenuation voltages) do not result in a good fit. However, picking the x-axis as the RF power gives nicely looking curve, especially in the beginning, because the slope is almost constant almost up intil the pi-pulse.

        # Attenuator voltages up to the pi-pulse
        x_up_to_pi_pulse_arr = np.sort(x_arr[np.where(x_arr <= pi_pulse_att_volt_spline)])

        # Data around corresponding to attenuation voltages up to pi-pulse
        exp_data_averaged_up_to_pi_pulse_df = self.exp_data_averaged_df.loc[x_up_to_pi_pulse_arr]

        # Minimum and maximum attenuator voltages for the chosen data range.
        x_up_to_pi_pulse_min = exp_data_averaged_up_to_pi_pulse_df.reset_index()['RF System Power [W]'].min()
        x_up_to_pi_pulse_max = exp_data_averaged_up_to_pi_pulse_df.reset_index()['RF System Power [W]'].max()

        # Plotting range for the polynomial fit to the chosen data up to the pi-pulse
        x_up_to_pi_pulse_poly_fit_arr = np.linspace(x_up_to_pi_pulse_min, x_up_to_pi_pulse_max, 100)
        # Data for the polynomial fit to the data up to the pi-pulse
        x_up_to_pi_pulse_arr = exp_data_averaged_up_to_pi_pulse_df.reset_index()['RF System Power [W]'].values

        y_up_to_pi_pulse_arr = exp_data_averaged_up_to_pi_pulse_df.reset_index()['Weighted Mean'].values
        y_up_to_pi_pulse_weight_arr = (1/exp_data_averaged_up_to_pi_pulse_df.reset_index()['Weighted STD']**2).values

        # Weighted polynomial fit

        # Polynomial fit order
        pol_fit_order = 3

        quenching_rf_power_polyfit = np.polyfit(x_up_to_pi_pulse_arr, y_up_to_pi_pulse_arr, deg=pol_fit_order, w=y_up_to_pi_pulse_weight_arr)

        quenching_rf_power_func = np.poly1d(quenching_rf_power_polyfit)

        # Maximum surviving fraction
        max_surv_frac = quenching_rf_power_func(0)

        # These are needed for plotting the results in another function
        self.x_spline_fit_arr = x_spline_fit_arr
        self.spl_smoothing = spl_smoothing
        self.x_poly_fit_arr = x_poly_fit_arr
        self.quenching_poly_func = quenching_poly_func
        self.x_up_to_pi_pulse_poly_fit_arr = x_up_to_pi_pulse_poly_fit_arr
        self.quenching_rf_power_func = quenching_rf_power_func

        # Needed for calculating required attenuation voltage for given surviving fraction with the subtracted offset.
        self.quenching_polyfit = quenching_polyfit
        self.max_surv_frac = max_surv_frac
        self.quenching_offset  = quenching_offset
        self.pi_pulse_att_volt = pi_pulse_att_volt
        self.quenching_poly_func = quenching_poly_func
        self.x_fit_arr_min = x_fit_arr_min
        self.x_fit_arr_max = x_fit_arr_max

    def plot_quenching_curve(self, axes):
        ''' Gives two separate plots of the quenching curves. One is the surviving fraction vs RF attenuation voltage, and another one is the surviving fraction vs calculated RF power in the system using the RF Power detector voltages.

        The plots also include all of the interpolation curves used for calculating pi-pulse and other related parameters.
        '''

        if self.pi_pulse_att_volt is None:
            self.perform_quenching_analysis()

        self.exp_data_averaged_df.reset_index().plot(x='Attenuator Setting [V]', y='Weighted Mean', kind='scatter', yerr='Weighted STD', ax=axes[0])

        axes[0].plot(self.x_spline_fit_arr, self.spl_smoothing(self.x_spline_fit_arr))

        axes[0].plot(self.x_poly_fit_arr, self.quenching_poly_func(self.x_poly_fit_arr))

        self.exp_data_averaged_df.reset_index().plot(x='RF System Power [W]', y='Weighted Mean', kind='scatter', xerr='RF System Power STDOM [W]', yerr='Weighted STD', ax=axes[1])

        axes[1].plot(self.x_up_to_pi_pulse_poly_fit_arr, self.quenching_rf_power_func(self.x_up_to_pi_pulse_poly_fit_arr), color='C1')

        return axes

    def get_att_volt(self, quenched_frac):
        ''' Given quenched fraction calculates the respective RF attenuation voltage needed.

        The RF attenuation voltages returned are only for the quenching data before the first pi-pulse. Notice, that one has to be careful with blindly believing the voltages given by this function. Make sure to always compare them with the plot of the quenching curve. This is imporatant for very small values of the quenching fraction (large surviving fraction), the polynomial fit starts to be extrapolating, not interpolating. This might give erroneous results. One might need to change the order of the fitting polynomial used as the function for extracting the RF attenuation voltages or range of data taken into account for this polynomial fit.

        Inputs:
        :quenched_frac: positive float <=1.

        Returns:
        :att_v: Respective attenuation voltage
        :surv_frac: Surviving fraction corresponsing to this attenuation voltage. It includes the quenching offset as well.
        '''

        if self.pi_pulse_att_volt is None:
            self.perform_quenching_analysis()

        # No quenching, therefore we get the maximal surviving fraction
        if quenched_frac == 0:
            att_v = 0
            surv_frac = self.max_surv_frac

        # When we are asking for the rf attenuator voltage corresponding to pi-pulse (quenched_frac is 1), then the np.roots function does not work well for determining the attenuation voltage. Anything less then 1 (even 0.999999999) work fine.
        elif quenched_frac == 1:
            att_v = self.pi_pulse_att_volt
            surv_frac = self.quenching_poly_func(att_v)
        else:
            surviving_frac_with_offset = self.quenching_offset + (self.max_surv_frac - self.quenching_offset) * (1-quenched_frac)

            surviving_frac_poly_fit = self.quenching_polyfit.copy()

            surviving_frac_poly_fit[-1] = surviving_frac_poly_fit[-1]-surviving_frac_with_offset

            att_v_arr = np.roots(surviving_frac_poly_fit)

            att_v_real_arr = att_v_arr[np.isreal(att_v_arr)]

            att_v = np.min(np.abs(att_v_real_arr[(att_v_real_arr >= self.x_fit_arr_min) & (att_v_real_arr <= self.x_fit_arr_max)]))

            surv_frac = self.quenching_poly_func(att_v)

        return att_v, surv_frac

    def get_quenching_curve_results(self):
        ''' Calculates important parameters from the quenching curve. Returns a pd.Series object with these parameters.
        '''

        'Pi-pulse (100 percent efficiency)'
        rf_att_pi_pulse, quenching_offset = self.get_att_volt(quenched_frac=1)

        'RF attenuator voltage for 99.9 percent quenching efficiency'
        rf_att_pi_pulse_delta_minus, quenching_offset_1 = self.get_att_volt(quenched_frac=1-0.001)

        'Allowed change in RF attenutoar voltage to still have 99.9 percent quenching efficiency'
        rf_att_pi_pulse_unc = rf_att_pi_pulse - rf_att_pi_pulse_delta_minus

        att_v_0, max_surv_frac = self.get_att_volt(quenched_frac=0.0)

        'Fractional Quenching offset w.r.t. the maximum surviving fraction'
        frac_quenching_offset = quenching_offset/max_surv_frac
        quenching_curve_results_s = pd.Series({
                            'RF Attenuator Setting For Pi-Pulse [V]': rf_att_pi_pulse,
                            'Quenching Offset': quenching_offset,
                            'Fractional Quenching Offset': frac_quenching_offset,
                            'RF Attenuator Setting Change For Quenched Fraction Change Of 1 ppt [V]': rf_att_pi_pulse_unc})

        return quenching_curve_results_s
#%%
# data_set = DataSetQuenchCurveCavity('180629-173226 - Quench Cavity Calibration - post-1088 PD ON')
# #data_set = DataSetQuenchCurveCavity('180412-200106 - Quench Cavity Calibration - pre-1147 PD 120V')
# #data_set = DataSetQuenchCurveCavity('180629-171507 - Quench Cavity Calibration - pre-910 PD ON')
#
# #data_set = DataSetQuenchCurveCavity('180412-200106 - Quench Cavity Calibration - pre-1147 PD 120V')
#
# exp_params_s = data_set.get_exp_parameters()
#
# rf_pwr_det_calib_df = data_set.get_quench_cav_rf_pwr_det_calib()
# #%%
# rf_pwr_det_calib_df
# #%%
# rf_freq = 910
# x_data = rf_pwr_det_calib_df.loc[rf_freq, 'Calibration Function x-Axis Plotting Data']
# y_data = rf_pwr_det_calib_df.loc[rf_freq, 'Calibration Function y-Axis Plotting Data']
# calib_data_df = rf_pwr_det_calib_df.loc[rf_freq, 'Calibration Data']
#
# fig, axes = plt.subplots(nrows=1, ncols=2)
# fig.set_size_inches(20,9)
# axes[0].plot(x_data, y_data, color='C3', label='Cubic Smoothing Spline')
#
# calib_data_df.plot(x='Power Detector [V]', y='RF Power [dBm]', kind='scatter', ax=axes[0], xerr='Power Detector STD [V]', color='C0', s=30, label='Calibration data')
#
# axes[0].set_title(rf_pwr_det_calib_df.loc[910, 'RF Power Detector Model'] + ' RF power detector calibration')
# axes[0].grid()
# axes[0].legend()
#
# #residual_arr = power_det_calib_func(calib_data_df['Power Detector [V]'].values) - calib_data_df['RF Power [dBm]'].values
#
# calib_data_df.plot(x='Power Detector [V]', y='Fit Residual [dB]', kind='scatter', ax=axes[1], xerr='Power Detector STD [V]', color='C2', s=30)
#
# #ax.set_xlim(min_det_v, max_det_v)
# axes[1].set_title('Spline and data residuals with power in dBm')
# axes[1].grid()
# axes[1].set_xlabel('Power Detector [V]')
# axes[1].set_ylabel('Best fit curve - RF Power [dB] residual')
# #axes[0].set_xlim(0.6,0.8)
# #axes[0].set_ylim(-10,5)
# plt.show()
# #%%
# quenching_cavities_df = data_set.get_quenching_cav_data()
# #%%
# fig, ax = plt.subplots()
# data = quenching_cavities_df['Pre-Quench', '1088', 'On'].reset_index()
# data.plot.scatter(x='Elapsed Time [s]', y='Fractional Change In Power Detector Reading [ppt]', ax=ax)
#
# plt.show()
# #%%
# quenching_cavities_av_df = data_set.get_av_quenching_cav_data()
# quenching_curve_df = data_set.get_quenching_curve()
#
# fig, axes = plt.subplots(nrows=2, ncols=1)
# fig.set_size_inches(10, 16)
# axes = data_set.plot_quenching_curve(axes)
# axes[0].set_xlim((2.0,3.2))
# axes[1].set_ylim((0, 0.010))
# plt.show()
# #%%
# data_set.get_att_volt(1)
# #%%
# 1-0.9995
# #%%
# data_set.get_att_volt(1)
# #%%
# data_set.get_att_volt(1)
# #%%
# data_set.get_att_volt(0)
# #%%
# frac_state_surv = 0.005
# att_param = data_set.get_att_volt(1-frac_state_surv)
# data_set.get_att_volt(1)[0] - att_param[0]
# #%%
# quenching_curve_results_s = data_set.get_quenching_curve_results()
# quenching_curve_results_s
#
# #%%
# y_data_arr = quenching_curve_df['Weighted Mean'].values
# x_data_arr = quenching_curve_df['RF System Power [W]'].values
# x_data_std_arr = quenching_curve_df['RF System Power STDOM [W]'].values
# y_data_std_arr = quenching_curve_df['Weighted STD'].values
#
# fig, ax = plt.subplots()
# ax.scatter(x_data_arr, y_data_arr)
# plt.show()
# #%%
# data_set_910 = DataSetQuenchCurveCavity('180629-171507 - Quench Cavity Calibration - pre-910 PD ON')
#
# exp_params_s_910 = data_set_910.get_exp_parameters()
#
# rf_pwr_det_calib_df_910 = data_set_910.get_quench_cav_rf_pwr_det_calib()
# quenching_cavities_df_910 = data_set_910.get_quenching_cav_data()
# quenching_cavities_av_df_910 = data_set_910.get_av_quenching_cav_data()
# quenching_curve_df_910 = data_set_910.get_quenching_curve()
# quenching_curve_results_s_910 = data_set_910.get_quenching_curve_results()
#
# poly_test_910 = copy.deepcopy(data_set_910.quenching_rf_power_func)
# poly_test_910[0] = poly_test_910[0] - quenching_curve_results_s_910['Quenching Offset']
# p_pi_910_arr = np.roots(poly_test_910)
# p_pi_910 = np.abs(p_pi_910_arr[np.isreal(p_pi_910_arr)])
#
# y_data_910_arr = quenching_curve_df_910['Weighted Mean'].values
# x_data_910_arr = quenching_curve_df_910['RF System Power [W]'].values / p_pi_910
# x_data_910_std_arr = quenching_curve_df_910['RF System Power STDOM [W]'].values / p_pi_910
# y_data_910_std_arr = quenching_curve_df_910['Weighted STD'].values
#
# data_set_1088 = DataSetQuenchCurveCavity('180629-173226 - Quench Cavity Calibration - post-1088 PD ON')
#
# exp_params_s_1088 = data_set_1088.get_exp_parameters()
#
# rf_pwr_det_calib_df_1088 = data_set_1088.get_quench_cav_rf_pwr_det_calib()
# quenching_cavities_df_1088 = data_set_1088.get_quenching_cav_data()
# quenching_cavities_av_df_1088 = data_set_1088.get_av_quenching_cav_data()
# quenching_curve_df_1088 = data_set_1088.get_quenching_curve()
# quenching_curve_results_s_1088 = data_set_1088.get_quenching_curve_results()
#
# poly_test_1088 = copy.deepcopy(data_set_1088.quenching_rf_power_func)
# poly_test_1088[0] = poly_test_1088[0] - quenching_curve_results_s_1088['Quenching Offset']
# p_pi_1088_arr = np.roots(poly_test_1088)
# p_pi_1088 = np.abs(p_pi_1088_arr[np.isreal(p_pi_1088_arr)])
#
# y_data_1088_arr = quenching_curve_df_1088['Weighted Mean'].values
# x_data_1088_arr = quenching_curve_df_1088['RF System Power [W]'].values / p_pi_1088
# x_data_1088_std_arr = quenching_curve_df_1088['RF System Power STDOM [W]'].values / p_pi_1088
# y_data_1088_std_arr = quenching_curve_df_1088['Weighted STD'].values
# #%%
# fig, ax = plt.subplots()
# fig.set_size_inches(12,8)
# #data_set.exp_data_averaged_df.reset_index().plot(x='RF System Power [W]', y='Weighted Mean', kind='scatter', yerr='Weighted STD', ax=ax, color='blue')
#
# ax.errorbar(x_data_910_arr, y_data_910_arr, y_data_910_std_arr, linestyle='', marker='.', color='blue')
#
# ax.errorbar(x_data_1088_arr, y_data_1088_arr, y_data_1088_std_arr, linestyle='', marker='.', color='red')
#
#
# ax.plot([1, 1], [-1, quenching_curve_results_s_910['Quenching Offset']], color='black', linestyle='dashed')
#
# ax.plot([-1, 5], [quenching_curve_results_s_910['Quenching Offset'], quenching_curve_results_s_910['Quenching Offset']], color='black', linestyle='dashed')
#
# ax.set_xlim([0.0, 5])
# ax.set_ylim([0, 0.02])
# ax.set_ylabel('Surviving fraction')
# ax.set_xlabel('Input power relative to the first $pi$-pulse')
# plt.show()
# #%%
