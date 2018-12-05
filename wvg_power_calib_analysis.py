from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

# For lab
#sys.path.insert(0,"C:/Users/Helium1/Google Drive/Research/Lamb shift measurement/Code")
# For home
sys.path.insert(0,"E:/Google Drive/Research/Lamb shift measurement/Code")

saving_folder_location = 'E:/Google Drive/Research/Lamb shift measurement/Data/Waveguide calibration'

from exp_data_analysis import *
#from fosof_data_set_analysis import *
from ZX47_Calibration_analysis import *
from KRYTAR_109_B_Calib_analysis import *
from hydrogen_sim_data import *
from wvg_power_calib_raw_data_analysis import *

import re
import time
import math
import copy
import pickle
import datetime

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt

# Package for wrapping long string to a paragraph
import textwrap

#%%
# Rename the frequency column to this particular value
freq_column_name = 'Waveguide Carrier Frequency [MHz]'

class WaveguideCalibrationAnalysis():

    ''' This class performs different types of analysis of the waveguide quench data to calibrate the waveguides.

    Inputs:
    :quench_sim_vs_freq_df: pd.DataFrame containing simulation files for each RF frequency and RF E field amplitude. It has to be properly formatted.
    :surf_frac_av_df: pd.DataFrame with the averaged quench data.
    :wvg_calib_params_dict: dictionary containing the following settings:
        'Date [date object]' - datetime object containing year, month and day that one wants to associate with the power calibration. Usually one puts the date of the start of the data acquisition for the power calibration.
        'Waveguide Separation [cm]' - (int or float) Physical separation between the waveguides.
        'Accelerating Voltage [kV]' - (float) High voltage setting on the accelerating power supply.
        RF Frequency Scan Range [MHz]' - a string specifying the RF frequencies used in the calibration.
        'Atom Off-Axis Distance (Simulation) [mm]' - float that specifies what was the off-axis distance for the atoms in the quench simulation. This is not very important for this simulation, actually.
        'Fractional DC Offset' - float (<1), which is the fractional DC offset that is seen when the waveguide is set to its respective pi-pulse. I.e., this is the ratio of the DC signal we get, when 1088 and 1147 quench cavities are set to their pi-pulses and the waveguide is set to its pi-pulse, to the case, when the waveguide is off, while the quench cavities are still set to their respective pi-pulses.
        'Minimum RF E Field Amplitude [V/cm]' - Minimum E field amplitude for which the calibration is performed. We do not take data at RF E field amplitudes lower than 5 V/cm.
        'Maximum RF E Field Amplitude [V/cm]' - (float) used to determine minimum surviving fraction used for the analysis. The reason for doing this is that for low powers the quench cuve is quite linear with power, which makes it possible to assume that low-order polynomial should be enough to fit the data well. This setting also puts a limit on the maximum E field for which the calibration procedure is performed. The rule is that the calibration is performed up to E_field_ampl_max - 2. Thus, if E_field_ampl_max = 27 (V/cm), then the maximum E field amplitude for which the calibration is performed is 25 V/cm.
        'Use Boundary Conditions' - boolean. We can assume that, for instance, when RF generator is not outputting any power, then the surviving fraction is 1, which is known very reliably. In principle this type of boundary condition should make the calibration more reliable, especially at low power settings.
        'Polynomial Fit Order' - (integer) order of the polynomial fit used to extract E fields. Usually I would use the value 3 for this. However, it seems that sometimes 4th-order polynomial results in much better fits. Of course it is important not to overfit the data.
    :load_Q: (bool) Loads the previously instantiated class instance with the same calib_folder_name, if it was saved before.
    :calib_folder_name: Should be used only when the analysis data has been previously saved. Name of the calibration folder. If given, then the wvg_calib_param_dict gets bypassed (thus one does not need to specify it) and the data gets loaded. Notice that load_Q has to be TRUE for this parameter to be taken into account.
    :calib_file_name: Part of the calib_folder_name specification. It is needed to load the proper file.
    '''

    def __init__(self, wvg_calib_param_dict=None, load_Q=True, quench_sim_vs_freq_df=None, surv_frac_av_df=None, calib_folder_name=None, calib_file_name=None):

        # Location for storing the analysis folders
        #self.saving_folder_location = 'C:/Research/Lamb shift measurement/Data/Waveguide calibration'
        # For home
        self.saving_folder_location = saving_folder_location

        # Version of the data analysis. Notice that this version number is 'engraved' in the analysis data file name. Thus one has to use the appropriate data analysis version.
        self.version_number = 0.1

        if (wvg_calib_param_dict is None) and (calib_folder_name is not None) and (calib_file_name is not None) and (load_Q):
            self.calib_folder_name = calib_folder_name
            self.analysis_data_file_name = calib_file_name

        else:

            def construct_calib_file_name(wvg_calib_param_dict):
                ''' Converts the important parameters in the calibration parameters dictionary into a file name for storage.
                '''

                # Calibration data analysis folder name.

                calib_folder_name = wvg_calib_param_dict['Date [date object]'].strftime('%Y%m%d') + 'T' + str(wvg_calib_param_dict['Waveguide Separation [cm]']) + 'E' + str(wvg_calib_param_dict['Accelerating Voltage [kV]']) + 'f' + wvg_calib_param_dict['RF Frequency Scan Range [MHz]']

                # Analysis data file name

                analysis_data_file_name = calib_folder_name + 'r' + str(wvg_calib_param_dict['Atom Off-Axis Distance (Simulation) [mm]']) + 'fo' + str(round(wvg_calib_param_dict['Fractional DC Offset'],3)) + 'v' + str(self.version_number) + '.pckl'

                return calib_folder_name, analysis_data_file_name

            self.wvg_calib_param_s = pd.Series(wvg_calib_param_dict)
            self.calib_folder_name, self.analysis_data_file_name = construct_calib_file_name(wvg_calib_param_dict)

        # Checking if the class instance has been saved before. If it was, then in case the user wants to load the data, it gets loaded. In all other cases the initialization continues.

        self.perform_analysis_Q = False

        os.chdir(self.saving_folder_location)
        if os.path.isdir(self.calib_folder_name):

            print('The analysis folder exists. Checking whether the analysis file exists...')
            os.chdir(self.calib_folder_name)

            if os.path.isfile(self.analysis_data_file_name):
                print('The analysis instance has been previously saved.')
                if load_Q:
                    print('Loading the analysis data...')
                    self.load_instance()
                    # Interesting effect here: if the data has been saved before with the perform_analysis_Q flag set to True, then after loading the data it sets is flag to True, making the analysis code believe that the analysis has not been performed before. That is why I have to set it to False here.
                    self.perform_analysis_Q = False
                else:
                    self.perform_analysis_Q = True
            else:
                print('No analysis instance of version ' + str(self.version_number) + ' has been found.')
                self.perform_analysis_Q = True
        else:
            self.perform_analysis_Q = True

        # The analysis gets performed if the perform_analysis_Q flag is True.
        if self.perform_analysis_Q:

            self.use_boundary_condition_bool = self.wvg_calib_param_s['Use Boundary Conditions']
            self.fract_DC_offset = self.wvg_calib_param_s['Fractional DC Offset']

            self.poly_fit_order = self.wvg_calib_param_s['Polynomial Fit Order']

            E_field_ampl_min = self.wvg_calib_param_s['Minimum RF E Field Amplitude [V/cm]']
            E_field_ampl_max = self.wvg_calib_param_s['Maximum RF E Field Amplitude [V/cm]']
            # The spacing between E fields is 1 V/cm.
            self.rf_e_field_ampl_arr = np.linspace(E_field_ampl_min, E_field_ampl_max-2, E_field_ampl_max-2-E_field_ampl_min+1)

            self.surv_frac_av_df = surv_frac_av_df.copy()
            self.quench_sim_vs_freq_df = quench_sim_vs_freq_df

            # Calculate On-Off ratios corrected for by the fractional DC offset that is independent of RF power, which is assumed to be the signal that is not due to the atoms in 2S1/2 state.

            self.surv_frac_av_df.loc[:, 'DC On/Off Ratio'] = (self.surv_frac_av_df.loc[:, 'DC On/Off Ratio'] - self.fract_DC_offset) / (1-self.fract_DC_offset)

            self.surv_frac_av_df.loc[:, 'DC On/Off Ratio STDOM'] = self.surv_frac_av_df.loc[:, 'DC On/Off Ratio STDOM'] / (1-self.fract_DC_offset)

            # Check if the simulation data contains all of the required RF frequencies.
            self.data_freq_index = self.surv_frac_av_df.index.get_level_values('Waveguide Carrier Frequency [MHz]').drop_duplicates()

            if self.data_freq_index.difference(self.quench_sim_vs_freq_df.index.get_level_values(freq_column_name)).shape[0] > 0:
                print('Some RF frequencies are missing in the simulation file.')

            # Add additional columns to the surviving fractions data. This is used later for calculating uncertainties in the corresponding extracted E field values.

            self.surv_frac_av_df['DC On/Off Ratio Max Limit'] = self.surv_frac_av_df['DC On/Off Ratio']+self.surv_frac_av_df['DC On/Off Ratio STDOM']
            self.surv_frac_av_df['DC On/Off Ratio Min Limit'] = self.surv_frac_av_df['DC On/Off Ratio']-self.surv_frac_av_df['DC On/Off Ratio STDOM']

            # Additional columns for getting uvalues and uncertainties in the square root of DC On/Off Ratios
            self.surv_frac_av_df['Square Root Of DC On/Off Ratio'] = np.sqrt(self.surv_frac_av_df['DC On/Off Ratio'])

            self.surv_frac_av_df['Square Root Of DC On/Off Ratio Max Limit'] = np.sqrt(self.surv_frac_av_df['DC On/Off Ratio Max Limit'])

            self.surv_frac_av_df['Square Root Of DC On/Off Ratio Min Limit'] = np.sqrt(self.surv_frac_av_df['DC On/Off Ratio Min Limit'])

            self.surv_frac_av_df['Square Root Of DC On/Off Ratio STDOM'] = self.surv_frac_av_df[['Square Root Of DC On/Off Ratio Max Limit', 'Square Root Of DC On/Off Ratio Min Limit']].subtract(self.surv_frac_av_df['Square Root Of DC On/Off Ratio'], axis='index').T.aggregate(lambda x: np.max(np.abs(x)))

            # Converting RF power settings in dBm to mW
            surf_frac_index_name_list = self.surv_frac_av_df.index.names
            self.surv_frac_av_df = self.surv_frac_av_df.reset_index()
            self.surv_frac_av_df['RF Generator Power Setting [mW]'] = 10**(self.surv_frac_av_df['RF Generator Power Setting [dBm]']/10)

            self.surv_frac_av_df = self.surv_frac_av_df.set_index(surf_frac_index_name_list).sort_index()

            # We now want to select the data that is larger than some minimum allowed surviving fraction. Why do this? Because we want to have good fit to the data. For large surviving fractions the relationship between the surviving population and the RF power is almost linear. Thus low-order polynomial is a good fit for the data. We also avoid getting into the saturation mode of the RF amplifier, which causes the surviving population vs RF generator setting to start to deviate from the straight line even more at higher powers.

            # RF frequency [MHz] to use to for extracting the minimum allowed surviving population. We want to use the frequency closest to the transition resonance, because at this frequency the pi-pulse is attained.
            rf_freq = 910.0

            quench_sim_object = OldWaveguideQuenchCurveSimulation(self.quench_sim_vs_freq_df.loc[(rf_freq), (slice(None))])

            quench_sim_object.analyze_data()

            # Minimum surviving fraction.
            self.surviving_frac_min = quench_sim_object.get_surv_frac([E_field_ampl_max**2])[0]

            # Select the data that is larger than or equal to the minimum allowed surviving fraction
            self.surv_frac_av_df = self.surv_frac_av_df[self.surv_frac_av_df['DC On/Off Ratio'] >= self.surviving_frac_min]

            # It is possible for low RF generator powers that the ON/OFF DC ratio is larger than 1 (but still close to 1). These values will not give correct calibration RF powers. The safest way to deal with this problem is to simply remove these points.
            self.surv_frac_av_modified_df = self.surv_frac_av_df.copy()

            # Points that have their DC On/Off Ratio > 1.

            # if self.surv_frac_av_modified_df[(self.surv_frac_av_modified_df['DC On/Off Ratio'] > 1)].shape[0] > 0:
            #     print('PRESENT')

            # Remove these points
            self.surv_frac_av_modified_df = self.surv_frac_av_modified_df[~(self.surv_frac_av_modified_df['DC On/Off Ratio'] > 1)]

            #self.surv_frac_av_modified_df.loc[;, 'DC On/Off Ratio'] = self.surv_frac_av_modified_df[self.surv_frac_av_modified_df['DC On/Off Ratio'] > 1] = 1

            # Now the points that have their upper limit larger than 1 are simply set to 1. This reduces the deviation from the average surviving fraction, however it means that most likely the lower limit will be the largest deviation, and, as described below, this deviation then will be used as 1 sigma uncertainty for the calibrated powers.

            self.surv_frac_av_modified_df['DC On/Off Ratio Max Limit'][self.surv_frac_av_modified_df['DC On/Off Ratio Max Limit'] > 1] = 1#-np.abs(np.random.rand())/1E5

            # pd.DataFrame object to store the required waveguide quench simulations.
            self.quench_sim_data_sets_df = None

            # pd.DataFrame for storing the DC On/Off ratios converted to RF E field
            self.surv_frac_converted_df = None

            # pd.DataFrame for holding the fit functions and parameters for the extracted E field curves vs various x-axis data columns
            self.extracted_E_field_vs_RF_power_fits_set_df = None

            # pd.DataFrame for holding the fit functions and parameters for the quench curves vs various x-axis data columns
            self.surv_frac_vs_RF_power_fits_set_df = None

            # pd.DataFrame containing the RF E field amplitude calibration. It allows one to know the RF power setting that needs to be set on the RF generator, voltage that needs to be observed on the RF power detector, and also its corresponding RF power (computed from calibration function)
            self.rf_e_field_calib_df = None

            # Averaged rf E field calibration pd.DataFrame object that is obtained by combining all of the different method for performing the power calibration.
            self.calib_av_df = None

            # This is a critically important pd.DataFrame object. It stores the mean uncertainty in RF power calibration for given E field amplitude in the waveguide. The uncertainty comes from the 50% uncertainty assigned to the fractional DC offset. Therefore to calculated this error we need to calculate power calibration for three different types fractional DC offsets. This particular class calculates power calibration only for a single fractional DC offset. That is why one needs to calculate this pd.DataFrame object outside this class and then simply assign this variable to the calculated pd.DataFrame.
            self.av_RF_power_calib_error_df = None

    def get_wvg_calib_parameters(self):
        return self.wvg_calib_params

    def analyze_simulation_quench_curves(self):
        # Perform analysis of the simulation quench curves for all of the required RF frequencies.

        # pd.DataFrame object to store the simulation quench curve objects.

        if self.quench_sim_data_sets_df is None:

            quench_sim_data_sets_df = None

            for rf_freq in self.data_freq_index:

                quench_sim_data_set = OldWaveguideQuenchCurveSimulation(self.quench_sim_vs_freq_df.loc[(rf_freq), (slice(None))])

                quench_sim_data_set.analyze_data()

                df = pd.DataFrame(pd.Series({rf_freq: quench_sim_data_set}, name='Quench Curve Object'))

                if quench_sim_data_sets_df is None:
                    quench_sim_data_sets_df = df
                else:
                    quench_sim_data_sets_df = quench_sim_data_sets_df.append(df)

                quench_sim_data_sets_df.index.names = [freq_column_name]

                self.quench_sim_data_sets_df = quench_sim_data_sets_df

        return self.quench_sim_data_sets_df

    def extract_E_fields(self):

        if self.surv_frac_converted_df is None:

            if self.quench_sim_data_sets_df is None:
                self.analyze_simulation_quench_curves()

            # Group the data for performing the calibration procedure.
            surv_frac_av_grouped_df = self.surv_frac_av_modified_df[['DC On/Off Ratio', 'DC On/Off Ratio Min Limit', 'DC On/Off Ratio Max Limit']].groupby(['Generator Channel', 'Waveguide Carrier Frequency [MHz]'])

            # Apply surviving fraction -> RF power conversion function to each RF frequency separately.
            def calculate_RF_power(df):
                rf_freq = df.index.get_level_values(freq_column_name).drop_duplicates()[0]

                quench_curve_object = self.quench_sim_data_sets_df.loc[rf_freq]['Quench Curve Object']

                df_transformed = df.transform(lambda x: quench_curve_object.get_RF_power(x)).rename(columns={'DC On/Off Ratio': 'E Field Amplitude Squared [V^2/cm^2]', 'DC On/Off Ratio Min Limit': 'E Field Amplitude Squared Max Limit [V^2/cm^2]', 'DC On/Off Ratio Max Limit': 'E Field Amplitude Squared Min Limit [V^2/cm^2]'})
                return df_transformed

            surv_frac_converted_df = surv_frac_av_grouped_df.apply(calculate_RF_power)

            # Calculate uncertainty in the determined RF power. We actually calculate the quantity that is proportional to RF power (E^2 in V^/cm^2) and also calculate the RF E field amplitude. The uncertainties are caclulated in the following way: we calculate surviving fraction +- its uncertainty. These are the 1 sigma limits for the surviving fraction. Then we calculate three RF powers, which correspond to the average surviving fraction + its 1 sigma bounds. From this we pick the largest |deviation| of the limit values from the average power and define it as the 1 sigma for the RF power. Same is done for the E field.

            surv_frac_converted_df['E Field Amplitude Squared Smaller Side STD [V^2/cm^2]'] = surv_frac_converted_df['E Field Amplitude Squared [V^2/cm^2]'] - surv_frac_converted_df['E Field Amplitude Squared Min Limit [V^2/cm^2]']

            surv_frac_converted_df['E Field Amplitude Squared Larger Side STD [V^2/cm^2]'] = surv_frac_converted_df['E Field Amplitude Squared Max Limit [V^2/cm^2]'] - surv_frac_converted_df['E Field Amplitude Squared [V^2/cm^2]']

            surv_frac_converted_df['E Field Amplitude Squared STD [V^2/cm^2]'] = surv_frac_converted_df[['E Field Amplitude Squared Smaller Side STD [V^2/cm^2]', 'E Field Amplitude Squared Larger Side STD [V^2/cm^2]']].T.aggregate(lambda x: np.max(x))

            surv_frac_converted_df['E Field Amplitude [V/cm]'] = np.sqrt(surv_frac_converted_df['E Field Amplitude Squared [V^2/cm^2]'])

            surv_frac_converted_df['E Field Amplitude Max Limit [V/cm]'] = np.sqrt(surv_frac_converted_df['E Field Amplitude Squared Max Limit [V^2/cm^2]'])

            surv_frac_converted_df['E Field Amplitude Min Limit [V/cm]'] = np.sqrt(surv_frac_converted_df['E Field Amplitude Squared Min Limit [V^2/cm^2]'])

            surv_frac_converted_df['E Field Amplitude Smaller Side STD [V/cm]'] = surv_frac_converted_df['E Field Amplitude [V/cm]'] - surv_frac_converted_df['E Field Amplitude Min Limit [V/cm]']

            surv_frac_converted_df['E Field Amplitude Larger Side STD [V/cm]'] = surv_frac_converted_df['E Field Amplitude Max Limit [V/cm]'] - surv_frac_converted_df['E Field Amplitude [V/cm]']

            surv_frac_converted_df['E Field Amplitude STD [V/cm]'] = surv_frac_converted_df[['E Field Amplitude Smaller Side STD [V/cm]', 'E Field Amplitude Larger Side STD [V/cm]']].T.aggregate(lambda x: np.max(x))

            surv_frac_converted_df = surv_frac_converted_df.drop(labels=['E Field Amplitude Max Limit [V/cm]', 'E Field Amplitude Min Limit [V/cm]', 'E Field Amplitude Larger Side STD [V/cm]', 'E Field Amplitude Smaller Side STD [V/cm]', 'E Field Amplitude Squared Max Limit [V^2/cm^2]', 'E Field Amplitude Squared Min Limit [V^2/cm^2]', 'E Field Amplitude Squared Larger Side STD [V^2/cm^2]', 'E Field Amplitude Squared Smaller Side STD [V^2/cm^2]'], axis='columns')

            # Append the waveguide power sensors' data.
            surv_frac_converted_df = surv_frac_converted_df.join(self.surv_frac_av_modified_df[['RF System Power Sensor Reading [V]', 'RF System Power Sensor Reading STDOM [V]', 'RF System Power Sensor Detected Power [mW]', 'RF System Power Sensor Detected Power STDOM [mW]', 'RF Generator Power Setting [mW]']])

            # Convert RF generator power settings from dBm to mW and to sqrt[mW].
            surv_frac_converted_index_name_list = list(surv_frac_converted_df.index.names)

            surv_frac_converted_df = surv_frac_converted_df.reset_index()

            surv_frac_converted_df['RF Generator Proportional To E Field Setting [V/cm]'] = surv_frac_converted_df['RF Generator Power Setting [mW]'].transform(lambda x: np.sqrt(x))

            surv_frac_converted_df = surv_frac_converted_df.set_index(surv_frac_converted_index_name_list).sort_index()

            self.surv_frac_converted_df = surv_frac_converted_df

        return self.surv_frac_converted_df


    def get_converted_E_field_curve_fits(self):
        ''' Calculates the calibration functions for the converted E fields for every RF frequency and generator channel.
        '''
        # We are first forced to find the smoothed spline fit for the E field vs RF generator power, because we need to use the uncertainty in the E field values. After we can use this smoothed spline to find the E field values ()= some sort of the best estimate) for each RF generator power setting. These can then be used to to invert the axes and now fit the inverted spline (without any smoothing) to the data. The inverted spline is the calibration function that gives us RF generator power setting for given E field.

        # We can also fit a polynomial fit function to the extracted square of the E field vs RF generator power (or any other power-related parameters). We expect the relationship between the extracted E field and the experiment parameter that is related to the detected/set E field to be quite linear. This is true for as long as we are not looking at the RF generator set power when the RF amplifiers start to saturate. As for the signal read form the RF power detectors, it is actually the opposite: at larger powers the voltage is quite linear with the square root of RF power, but at the lower powers the relationship is not quite as linear. In any case, we expect that the low-order polynomial should quite well describe the data.

        if self.extracted_E_field_vs_RF_power_fits_set_df is None:

            if self.surv_frac_converted_df is None:
                self.extract_E_fields()

            def parameterize_quench_curve(df):
                ''' Fit a spline and a polynomial to the curves which are taken vs various x-axis data types, as explained below.

                Outputs:
                :e_field_vs_RF_power_fits_set_df: pd.DataFrame object that for every x-axis data column contains the following parameters:
                Polynomial (least-squares) fit function, Smoothing spline fit, Inverted non-smoothing spline fit, and the chi-squared of the polynomial fit with some additional information pertained to it.
                '''
                # We use various data columns as the x-axis for the fits.

                # x-axis = RF generator set power. The hope is that the RF generator has good enough power calibration, so that its power setting represents the set power reasonably well.

                # x-axis = Detected System RF power - uses RF power detector calibration.

                # x-axis = RF power detector voltage reading. The KRYTAR 109B power detector has signal voltage that is proportional to the square root of power (for large enough powers), which is in turn proportional to the RF electric field amplitude. Thus if we use Power detector voltages as the x-axis, we need to use the extracted E fields as the y-axis.

                x_data_column_name_list = ['RF Generator Power Setting [mW]', 'RF System Power Sensor Reading [V]', 'RF System Power Sensor Detected Power [mW]']
                y_data_column_name_list = ['E Field Amplitude Squared [V^2/cm^2]', 'E Field Amplitude [V/cm]', 'E Field Amplitude Squared [V^2/cm^2]']
                y_data_std_column_name_list = ['E Field Amplitude Squared STD [V^2/cm^2]', 'E Field Amplitude STD [V/cm]', 'E Field Amplitude Squared STD [V^2/cm^2]']
                data_s_name_list = ['RF Generator Power Setting', 'RF System Power Sensor Reading', 'RF System Power Sensor Detected Power']

                analysis_data_info_df = pd.DataFrame(np.array([x_data_column_name_list, y_data_column_name_list, y_data_std_column_name_list]).T, index=data_s_name_list, columns=['X-Axis Data', 'Y-Axis Data', 'Y-Axis STD Data'])

                e_field_vs_RF_power_fits_set_df = None

                for analysis_type in analysis_data_info_df.index:

                    x_data_column_name = analysis_data_info_df.loc[analysis_type]['X-Axis Data']
                    y_data_column_name = analysis_data_info_df.loc[analysis_type]['Y-Axis Data']
                    y_data_std_column_name = analysis_data_info_df.loc[analysis_type]['Y-Axis STD Data']

                    df = df.sort_values(x_data_column_name)

                    x_data_arr = df[x_data_column_name].values
                    y_data_arr = df[y_data_column_name].values
                    y_data_std_arr = df[y_data_std_column_name].values

                    if self.use_boundary_condition_bool:

                        # We know for sure that when no we have no reading on the RF power detector, then the E field inside the waveguide is zero. We should not set the uncertainty to zero, however, since the spline gives an error in this case. Interestingly, setting the error to some very low value (about 10^-8 or lower) gives very bad polynomial fit, thus I am setting the uncertainty in this zero point to factor of 100 smaller than the smallest uncertainty in this data set. This should still force the curve to pass through the (0, 0) point.
                        x_data_arr = np.insert(x_data_arr, 0, [0])
                        y_data_arr = np.insert(y_data_arr, 0, [0])
                        y_data_std_arr = np.insert(y_data_std_arr, 0, y_data_std_arr.min()/100)

                        data_set = np.array([x_data_arr, y_data_arr, y_data_std_arr]).T
                        data_set = data_set[data_set[:,0].argsort()]

                        x_data_arr = data_set[:,0]
                        y_data_arr = data_set[:,1]
                        y_data_std_arr = data_set[:,2]

                    spline_order = 3

                    # Determining the min and max values for the smoothing factor
                    m = y_data_std_arr.shape[0]
                    s_min = m - np.sqrt(2*m)
                    s_max = m + np.sqrt(2*m)

                    smoothing_coeff = 1

                    spl_smoothing = scipy.interpolate.UnivariateSpline(x=x_data_arr, y=y_data_arr, k=spline_order, s=smoothing_coeff*s_max, w=1/y_data_std_arr)

                    # Now we calculate the inverted spline (no smoothing this time). We use the values determined by the smoothing spline as the x-axis now.

                    # Sorting the data by its E field value in the increasing order.
                    x_arr = np.linspace(np.min(x_data_arr), np.max(x_data_arr), 10*x_data_arr.shape[0])

                    data_array_inverted = np.array([spl_smoothing(x_arr), x_arr]).T
                    data_array_inverted = data_array_inverted[data_array_inverted[:,0].argsort()]

                    x_data_inverted_arr = data_array_inverted[:, 0]
                    y_data_inverted_arr = data_array_inverted[:, 1]

                    spl_inverted = scipy.interpolate.UnivariateSpline(x=x_data_inverted_arr, y=y_data_inverted_arr, k=spline_order, s=0)

                    # Polynomial fit order
                    pol_fit_order = self.poly_fit_order

                    polyfit = np.polyfit(x_data_arr, y_data_arr, deg=pol_fit_order, w=1/y_data_std_arr**2)

                    poly_func = np.poly1d(polyfit)

                    # Calculate chi-squared for the polynomial fit.
                    chi_squared_s = get_chi_squared(data_arr=y_data_arr, data_std_arr=y_data_std_arr, fit_data_arr=poly_func(x_data_arr), n_constraints=pol_fit_order+1)

                    # Form Series object with the analysis data
                    e_field_vs_RF_power_fits_s = chi_squared_s
                    e_field_vs_RF_power_fits_s = e_field_vs_RF_power_fits_s.append(pd.Series({
                        'Polynomial Fit': poly_func,
                        'Spline Fit': spl_smoothing,
                        'Inverted Spline Fit': spl_inverted
                        }))
                    e_field_vs_RF_power_fits_s.name = analysis_type
                    # Form DataFrame out of the Series
                    e_field_vs_RF_power_fits_df = pd.DataFrame(e_field_vs_RF_power_fits_s)

                    if e_field_vs_RF_power_fits_set_df is None:
                        e_field_vs_RF_power_fits_set_df = e_field_vs_RF_power_fits_df
                    else:
                        e_field_vs_RF_power_fits_set_df = e_field_vs_RF_power_fits_set_df.join(e_field_vs_RF_power_fits_df)

                e_field_vs_RF_power_fits_set_df = e_field_vs_RF_power_fits_set_df.T
                e_field_vs_RF_power_fits_set_df.index.names = ['X-Axis Data']
                e_field_vs_RF_power_fits_set_df.columns.names = ['Data Field']

                return e_field_vs_RF_power_fits_set_df

            self.surv_frac_converted_grouped_df = self.surv_frac_converted_df[['RF System Power Sensor Reading [V]', 'RF System Power Sensor Detected Power [mW]', 'RF Generator Power Setting [mW]', 'E Field Amplitude Squared [V^2/cm^2]', 'E Field Amplitude Squared STD [V^2/cm^2]', 'E Field Amplitude [V/cm]', 'E Field Amplitude STD [V/cm]']].groupby(['Generator Channel', 'Waveguide Carrier Frequency [MHz]'])

            self.extracted_E_field_vs_RF_power_fits_set_df = self.surv_frac_converted_grouped_df.apply(parameterize_quench_curve)

        return self.extracted_E_field_vs_RF_power_fits_set_df


    def get_quench_curve_fits(self):

        # There is a different way to perform the waveguide calibration. In the first method for each RF frequency and RF power and for each averaged fractional surviving population I determine corresponding E field amplitude, based on the spline fit done to the E field amplitude vs Fractional surviving population simulation data. This way for each RF frequency I obtain data sets of E field amplitude vs RF power. Each data set is then interpolated with a spline.
        # Group the data for performing the calibration procedure. Given required E field and using N of these interpolating spline for respective N RF frequencies, I can obtain a plot of RF power vs RF frequency. Now, the 'RF power' here means either the RF generator settng or either voltage or extracted RF power from the voltage detected on the RF power detectors set up in the RF systems A and B.

        # Now the second method is somewhat different. For each RF frequency we have RF power vs Fractional surviving population data sets. These data sets can be interpolated with a spline. We also expect that for large enough surviving fractions the relationship between the square of the E field and the surviving fractions is smooth without any local extrema - this can be seen from the simulation curves. Thus we can also use a low-order polynomial to fit the data in additional to a spline.
        # The first advantage of using the polynomial fit compared to the spline, is that we know how many local extrema it will result in, which is not the case with the smoothing spline (but it possibly can be controlled in the settings of this spline function), and we also do not have to worry about the smoothing parameter. The second advantage is the ease with which we can calculate the reduced chi-squared of the fit - we know the number of degrees of freedom. I am currently not sure how to determine this number of dofs for a smoothing spline.
        # We will have N of these polynomial and spline fits for N RF frequencies. Now, given RF E field amplitude, we can determine the surviving fraction from the simulation data and use this surviving fraction to solve for the corresponding RF power from the polynomial fit. As for the spline interpolation, we can invert the data first to interpolate the data for RF power vs Surviving fraction and then simply use this spline to extract the respective RF power.

        if self.surv_frac_vs_RF_power_fits_set_df is None:

            def parameterize_quench_curve(df):
                ''' Fit a spline and a polynomial to the quench curves which are taken vs various x-axis data types, as explained below.

                Outputs:
                :surf_frac_vs_RF_power_fits_set_df: pd.DataFrame object that for every x-axis data column contains the following parameters:
                Polynomial (least-squares) fit function, Smoothing spline fit, Inverted non-smoothing spline fit, and the chi-squared of the polynomial fit with some additional information pertained to it.
                '''
                # We use various data columns as the x-axis for the fits.

                # x-axis = RF generator set power. The hope is that the RF generator has good enough power calibration, so that its power setting represents the set power reasonably well.

                # x-axis = Detected System RF power - uses RF power detector calibration.

                # x-axis = RF power detector voltage reading. The KRYTAR 109B power detector has signal voltage that is proportional to the square root of power (for large enough powers), which is in turn proportional to the surviving fraction. Thus if we use Power detector voltages as the x-axis, we need to use the square root of the surviving fraction as the y-axis.

                x_data_column_name_list = ['RF Generator Power Setting [mW]', 'RF System Power Sensor Reading [V]', 'RF System Power Sensor Detected Power [mW]']
                y_data_column_name_list = ['DC On/Off Ratio', 'Square Root Of DC On/Off Ratio', 'DC On/Off Ratio']
                y_data_std_column_name_list = ['DC On/Off Ratio STDOM', 'Square Root Of DC On/Off Ratio STDOM', 'DC On/Off Ratio STDOM']
                data_s_name_list = ['RF Generator Power Setting', 'RF System Power Sensor Reading', 'RF System Power Sensor Detected Power']

                analysis_data_info_df = pd.DataFrame(np.array([x_data_column_name_list, y_data_column_name_list, y_data_std_column_name_list]).T, index=data_s_name_list, columns=['X-Axis Data', 'Y-Axis Data', 'Y-Axis STD Data'])

                surf_frac_vs_RF_power_fits_set_df = None

                for analysis_type in analysis_data_info_df.index:

                    x_data_column_name = analysis_data_info_df.loc[analysis_type]['X-Axis Data']
                    y_data_column_name = analysis_data_info_df.loc[analysis_type]['Y-Axis Data']
                    y_data_std_column_name = analysis_data_info_df.loc[analysis_type]['Y-Axis STD Data']

                    df = df.sort_values(x_data_column_name)

                    x_data_arr = df[x_data_column_name].values
                    y_data_arr = df[y_data_column_name].values
                    y_data_std_arr = df[y_data_std_column_name].values

                    if self.use_boundary_condition_bool:

                        # We know for sure that when no we have no reading on the RF power detector, then the E field inside the waveguide is zero. We should not set the uncertainty to zero, however, since the spline gives an error in this case. Interestingly, setting the error to some very low value (about 10^-8 or lower) gives very bad polynomial fit, thus I am setting the uncertainty in this zero point to factor of 100 smaller than the smallest uncertainty in this data set. This should still force the curve to pass through the (0, 1) point.
                        x_data_arr = np.insert(x_data_arr, 0, [0])
                        y_data_arr = np.insert(y_data_arr, 0, [1])
                        y_data_std_arr = np.insert(y_data_std_arr, 0, y_data_std_arr.min()/100)

                        data_set = np.array([x_data_arr, y_data_arr, y_data_std_arr]).T
                        data_set = data_set[data_set[:,0].argsort()]

                        x_data_arr = data_set[:,0]
                        y_data_arr = data_set[:,1]
                        y_data_std_arr = data_set[:,2]

                    spline_order = 3

                    # Determining the min and max values for the smoothing factor
                    m = y_data_std_arr.shape[0]
                    s_min = m - np.sqrt(2*m)
                    s_max = m + np.sqrt(2*m)

                    smoothing_coeff = 1

                    spl_smoothing = scipy.interpolate.UnivariateSpline(x=x_data_arr, y=y_data_arr, k=spline_order, s=smoothing_coeff*s_max, w=1/y_data_std_arr)

                    # Now we calculate the inverted spline (no smoothing this time). We use the values determined by the smoothing spline as the x-axis now.

                    # Sorting the data by its E field value in the increasing order.
                    x_arr = np.linspace(np.min(x_data_arr), np.max(x_data_arr), 10*x_data_arr.shape[0])

                    data_array_inverted = np.array([spl_smoothing(x_arr), x_arr]).T
                    data_array_inverted = data_array_inverted[data_array_inverted[:,0].argsort()]

                    x_data_inverted_arr = data_array_inverted[:, 0]
                    y_data_inverted_arr = data_array_inverted[:, 1]

                    spl_inverted = scipy.interpolate.UnivariateSpline(x=x_data_inverted_arr, y=y_data_inverted_arr, k=spline_order, s=0)

                    # Polynomial fit order
                    pol_fit_order = self.poly_fit_order

                    polyfit = np.polyfit(x_data_arr, y_data_arr, deg=pol_fit_order, w=1/y_data_std_arr**2)

                    poly_func = np.poly1d(polyfit)

                    # Calculate chi-squared for the polynomial fit.
                    chi_squared_s = get_chi_squared(data_arr=y_data_arr, data_std_arr=y_data_std_arr, fit_data_arr=poly_func(x_data_arr), n_constraints=pol_fit_order+1)

                    surf_frac_vs_RF_power_fits_s = chi_squared_s
                    surf_frac_vs_RF_power_fits_s = surf_frac_vs_RF_power_fits_s.append(pd.Series({
                        'Polynomial Fit': poly_func,
                        'Spline Fit': spl_smoothing,
                        'Inverted Spline Fit': spl_inverted
                        }))
                    surf_frac_vs_RF_power_fits_s.name = analysis_type
                    surf_frac_vs_RF_power_fits_df = pd.DataFrame(surf_frac_vs_RF_power_fits_s)

                    if surf_frac_vs_RF_power_fits_set_df is None:
                        surf_frac_vs_RF_power_fits_set_df = surf_frac_vs_RF_power_fits_df
                    else:
                        surf_frac_vs_RF_power_fits_set_df = surf_frac_vs_RF_power_fits_set_df.join(surf_frac_vs_RF_power_fits_df)

                surf_frac_vs_RF_power_fits_set_df = surf_frac_vs_RF_power_fits_set_df.T
                surf_frac_vs_RF_power_fits_set_df.index.names = ['X-Axis Data']
                surf_frac_vs_RF_power_fits_set_df.columns.names = ['Data Field']

                return surf_frac_vs_RF_power_fits_set_df

            self.surv_frac_av_grouped_df = self.surv_frac_av_df[['DC On/Off Ratio', 'DC On/Off Ratio STDOM', 'DC On/Off Ratio Max Limit', 'DC On/Off Ratio Min Limit', 'RF System Power Sensor Reading [V]', 'RF System Power Sensor Detected Power [mW]', 'Square Root Of DC On/Off Ratio', 'Square Root Of DC On/Off Ratio STDOM', 'RF Generator Power Setting [mW]']].groupby(['Generator Channel', 'Waveguide Carrier Frequency [MHz]'])

            self.surv_frac_vs_RF_power_fits_set_df = self.surv_frac_av_grouped_df.apply(parameterize_quench_curve)

        return self.surv_frac_vs_RF_power_fits_set_df


    def produce_plots(self, analysis_data_info_df, data_df, fit_func_df, axes):
        ''' Convenience function that gives the plotting resuts.
        '''
        axes_index = 0

        for analysis_type in analysis_data_info_df.index:

            x_data_column_name = analysis_data_info_df.loc[analysis_type]['X-Axis Data']
            y_data_column_name = analysis_data_info_df.loc[analysis_type]['Y-Axis Data']
            y_data_std_column_name = analysis_data_info_df.loc[analysis_type]['Y-Axis STD Data']

            x_data_arr = data_df[x_data_column_name].values
            y_data_arr = data_df[y_data_column_name].values
            y_data_std_arr = data_df[y_data_std_column_name].values

            data_df.plot(kind='scatter', x=x_data_column_name, y=y_data_column_name, yerr=y_data_std_column_name, ax=axes[axes_index, 0], color='black')


            x_data_arr_min = np.min(x_data_arr)
            x_data_arr_max = np.max(x_data_arr)

            # We want the fit line to show the (0, 0) point.
            # if x_data_arr_min > 0:
            #     x_min = 0
            # else:
            #     x_min = x_data_arr_min
            #
            # if x_data_arr_max < 0:
            #     x_max = 0
            # else:
            #     x_max = x_data_arr_max

            x_min = x_data_arr_min
            x_max = x_data_arr_max
            x_arr = np.linspace(x_min, x_max, x_data_arr.shape[0]*10)

            poly_func = fit_func_df.loc[analysis_type]['Polynomial Fit']
            spl_smoothing = fit_func_df.loc[analysis_type]['Spline Fit']
            spl_inverted = fit_func_df.loc[analysis_type]['Inverted Spline Fit']

            axes[axes_index, 0].plot(x_arr, poly_func(x_arr), label='Polynomial Fit', color='blue')
            axes[axes_index, 0].plot(x_arr, spl_smoothing(x_arr), label='Spline Fit', color='red')
            axes[axes_index, 0].legend()

            axes[axes_index, 1].errorbar(x_data_arr, y_data_arr-poly_func(x_data_arr), yerr=y_data_std_arr, marker='o', linestyle='', label='Polynomial Fit', color='blue')
            axes[axes_index, 1].errorbar(x_data_arr, y_data_arr-spl_smoothing(x_data_arr), yerr=y_data_std_arr, marker='o', linestyle='', label='Spline Fit', color='red')

            axes[axes_index, 1].set_xlabel(x_data_column_name)
            axes[axes_index, 1].set_ylabel(y_data_column_name + ' redisual')
            axes[axes_index, 1].set_title('Fit Residuals')
            axes[axes_index, 1].legend()

            y_data_arr_min = np.min(y_data_arr)
            y_data_arr_max = np.max(y_data_arr)

            # if y_data_arr_min > 0:
            #     y_min = 0
            # else:
            #     y_min = x_data_arr_min
            #
            # if y_data_arr_max < 0:
            #     y_max = 0
            # else:
            #     y_max = y_data_arr_max

            y_min = y_data_arr_min
            y_max = y_data_arr_max
            y_arr = np.linspace(y_min, y_max, y_data_arr.shape[0]*10)

            data_df.plot(kind='scatter', x=y_data_column_name, y=x_data_column_name, xerr=y_data_std_column_name, ax=axes[axes_index, 2], color='black')

            axes[axes_index, 2].plot(y_arr, spl_inverted(y_arr), label='Inverted Spline Fit', color='red')
            axes[axes_index, 2].legend()

            axes_index = axes_index + 1
        return axes

    def plot_quench_curve(self, rf_channel, rf_freq, axes):
        ''' Plots the fits and residuals associated with the different methods of analysing the data.
        '''
        if self.surv_frac_vs_RF_power_fits_set_df is None:
            self.get_quench_curve_fits()

        x_data_column_name_list = ['RF Generator Power Setting [mW]', 'RF System Power Sensor Reading [V]', 'RF System Power Sensor Detected Power [mW]']
        y_data_column_name_list = ['DC On/Off Ratio', 'Square Root Of DC On/Off Ratio', 'DC On/Off Ratio']
        y_data_std_column_name_list = ['DC On/Off Ratio STDOM', 'Square Root Of DC On/Off Ratio STDOM', 'DC On/Off Ratio STDOM']
        data_s_name_list = ['RF Generator Power Setting', 'RF System Power Sensor Reading', 'RF System Power Sensor Detected Power']

        analysis_data_info_df = pd.DataFrame(np.array([x_data_column_name_list, y_data_column_name_list, y_data_std_column_name_list]).T, index=data_s_name_list, columns=['X-Axis Data', 'Y-Axis Data', 'Y-Axis STD Data'])

        surviving_frac_vs_rf_power_df = self.surv_frac_av_grouped_df.get_group((rf_channel, rf_freq))

        fit_func_df = self.surv_frac_vs_RF_power_fits_set_df.loc[rf_channel, rf_freq]

        axes = self.produce_plots(analysis_data_info_df, surviving_frac_vs_rf_power_df, fit_func_df, axes)

        axes[0, 0].set_xlim(xmin=0)
        axes[0, 0].set_ylim(ymax=1)

        axes[0, 1].set_xlim(xmin=0)

        axes[0, 2].set_xlim(xmax=1)
        axes[0, 2].set_ylim(ymin=0)

        axes[1, 0].set_xlim(xmax=0)
        axes[1, 0].set_ylim(ymax=1)

        axes[1, 1].set_xlim(xmax=0)

        axes[1, 2].set_xlim(xmax=1)
        axes[1, 2].set_ylim(ymax=0)

        axes[2, 0].set_xlim(xmin=0)
        axes[2, 0].set_ylim(ymax=1)

        axes[2, 1].set_xlim(xmin=0)

        axes[2, 2].set_xlim(xmax=1)
        axes[2, 2].set_ylim(ymin=0)

        return axes

    def plot_extracted_E_field_curves(self, rf_channel, rf_freq, axes):
        ''' Plots the fits and residuals associated with the different methods of analysing the data.
        '''
        if self.extracted_E_field_vs_RF_power_fits_set_df is None:
            self.get_converted_E_field_curve_fits()

        x_data_column_name_list = ['RF Generator Power Setting [mW]', 'RF System Power Sensor Reading [V]', 'RF System Power Sensor Detected Power [mW]']
        y_data_column_name_list = ['E Field Amplitude Squared [V^2/cm^2]', 'E Field Amplitude [V/cm]', 'E Field Amplitude Squared [V^2/cm^2]']
        y_data_std_column_name_list = ['E Field Amplitude Squared STD [V^2/cm^2]', 'E Field Amplitude STD [V/cm]', 'E Field Amplitude Squared STD [V^2/cm^2]']
        data_s_name_list = ['RF Generator Power Setting', 'RF System Power Sensor Reading', 'RF System Power Sensor Detected Power']

        analysis_data_info_df = pd.DataFrame(np.array([x_data_column_name_list, y_data_column_name_list, y_data_std_column_name_list]).T, index=data_s_name_list, columns=['X-Axis Data', 'Y-Axis Data', 'Y-Axis STD Data'])

        surviving_frac_vs_rf_power_df = self.surv_frac_converted_grouped_df.get_group((rf_channel, rf_freq))

        fit_func_df = self.extracted_E_field_vs_RF_power_fits_set_df.loc[rf_channel, rf_freq]

        axes = self.produce_plots(analysis_data_info_df, surviving_frac_vs_rf_power_df, fit_func_df, axes)

        axes[0, 0].set_xlim(xmin=0)
        axes[0, 0].set_ylim(ymin=0)

        axes[0, 1].set_xlim(xmin=0)

        axes[0, 2].set_xlim(xmin=0)
        axes[0, 2].set_ylim(ymin=0)

        axes[1, 0].set_xlim(xmax=0)
        axes[1, 0].set_ylim(ymin=0)

        axes[1, 1].set_xlim(xmax=0)

        axes[1, 2].set_xlim(xmin=0)
        axes[1, 2].set_ylim(ymax=0)

        axes[2, 0].set_xlim(xmin=0)
        axes[2, 0].set_ylim(ymin=0)

        axes[2, 1].set_xlim(xmin=0)

        axes[2, 2].set_xlim(xmin=0)
        axes[2, 2].set_ylim(ymin=0)

        return axes

    def perform_power_calib(self):

        if self.extracted_E_field_vs_RF_power_fits_set_df is None:
            self.get_converted_E_field_curve_fits()

        if self.surv_frac_vs_RF_power_fits_set_df is None:
            self.get_quench_curve_fits()

        if self.rf_e_field_calib_df is None:

            x_data_column_name_list = ['RF Generator Power Setting [mW]', 'RF System Power Sensor Reading [V]', 'RF System Power Sensor Detected Power [mW]']
            y_data_column_name_list = ['DC On/Off Ratio', 'Square Root Of DC On/Off Ratio', 'DC On/Off Ratio']
            y_data_std_column_name_list = ['DC On/Off Ratio STDOM', 'Square Root Of DC On/Off Ratio STDOM', 'DC On/Off Ratio STDOM']
            data_s_name_list = ['RF Generator Power Setting', 'RF System Power Sensor Reading', 'RF System Power Sensor Detected Power']

            analysis_data_info_df = pd.DataFrame(np.array([x_data_column_name_list, y_data_column_name_list, y_data_std_column_name_list]).T, index=data_s_name_list, columns=['X-Axis Data', 'Y-Axis Data', 'Y-Axis STD Data'])

            def calculate_calib_vs_E_field(df):
                rf_channel = df.index.get_level_values('Generator Channel').drop_duplicates()[0]
                rf_freq = df.index.get_level_values(freq_column_name).drop_duplicates()[0]

                quench_sim_data_object = self.quench_sim_data_sets_df.loc[rf_freq]['Quench Curve Object']

                calib_data_vs_e_field_ampl_df = None

                for rf_e_field_ampl in self.rf_e_field_ampl_arr:

                    calib_data_set_df = None

                    expected_surviving_frac = quench_sim_data_object.get_surv_frac([rf_e_field_ampl**2])[0]

                    for x_axis_data_type in analysis_data_info_df.index:

                        x_axis_column_name = analysis_data_info_df.loc[x_axis_data_type]['X-Axis Data']
                        poly_fit = self.surv_frac_vs_RF_power_fits_set_df.loc[rf_channel, rf_freq, x_axis_data_type]['Polynomial Fit']

                        inverted_spline_fit = self.surv_frac_vs_RF_power_fits_set_df.loc[rf_channel, rf_freq, x_axis_data_type]['Inverted Spline Fit']

                        x_data_arr = self.surv_frac_av_grouped_df.get_group((rf_channel, rf_freq)).reset_index()[x_axis_column_name].values

                        poly_fit_coeff = poly_fit.c

                        if x_axis_data_type == 'RF System Power Sensor Reading':
                            poly_fit_coeff[-1] = poly_fit_coeff[-1] - np.sqrt(expected_surviving_frac)
                            spline_result = inverted_spline_fit(np.sqrt([expected_surviving_frac]))[0]
                        else:
                            poly_fit_coeff[-1] = poly_fit_coeff[-1] - expected_surviving_frac
                            spline_result = inverted_spline_fit([expected_surviving_frac])[0]

                        poly_fit_roots = np.roots(poly_fit_coeff)
                        poly_fit_roots = np.real(poly_fit_roots[np.isreal(poly_fit_roots)])
                        poly_fit_roots = poly_fit_roots[(poly_fit_roots >= np.min(x_data_arr)) & (poly_fit_roots <= np.max(x_data_arr))]

                        poly_fit_root = np.nan

                        if poly_fit_roots.shape[0] > 1:
                            print('---------------')
                            print(rf_e_field_ampl)
                            print(x_axis_data_type)
                            print(rf_freq)
                            print('WARNING!!! More than one solution has been found for the polynomial')
                        else:
                            if poly_fit_roots.shape[0] == 0:
                                print('---------------')
                                print(rf_e_field_ampl)
                                print(x_axis_data_type)
                                print(rf_freq)
                                print('WARNING!! No solution has been found for the polynomial')
                            else:
                                poly_fit_root = poly_fit_roots[0]

                        calib_data_df = pd.DataFrame(pd.Series({'Quench Curve Polynomial Fit': poly_fit_root, 'Quench Curve Inverted Spline Fit': spline_result}, name=x_axis_column_name))

                        if calib_data_set_df is None:
                            calib_data_set_df = calib_data_df
                        else:
                            calib_data_set_df = calib_data_set_df.join(calib_data_df)

                    calib_data_set_df.index.names = ['Method']
                    calib_data_set_df.columns.names = ['Data Field']

                    calib_data_set_df['E Field [V/cm]'] = rf_e_field_ampl
                    calib_data_set_df = calib_data_set_df.reset_index().set_index(['E Field [V/cm]', 'Method'])

                    if calib_data_vs_e_field_ampl_df is None:
                        calib_data_vs_e_field_ampl_df = calib_data_set_df
                    else:
                        calib_data_vs_e_field_ampl_df = calib_data_vs_e_field_ampl_df.append(calib_data_set_df)

                return calib_data_vs_e_field_ampl_df

            surv_frac_vs_RF_power_fits_set_grouped_df = self.surv_frac_vs_RF_power_fits_set_df.groupby(['Generator Channel', freq_column_name])

            quench_curve_fits_calib_df = surv_frac_vs_RF_power_fits_set_grouped_df.apply(calculate_calib_vs_E_field)

            quench_curve_fits_calib_df = quench_curve_fits_calib_df.reset_index().set_index(['Generator Channel', 'E Field [V/cm]', 'Method', 'Waveguide Carrier Frequency [MHz]']).sort_index()

            # ======================
            # Now we use RF E field amplitude vs RF power parameter fits to extract the RF power paramater required vs RF E Field amplitude.
            # ======================
            x_data_column_name_list = ['RF Generator Power Setting [mW]', 'RF System Power Sensor Reading [V]', 'RF System Power Sensor Detected Power [mW]']
            y_data_column_name_list = ['DC On/Off Ratio', 'Square Root Of DC On/Off Ratio', 'DC On/Off Ratio']
            y_data_std_column_name_list = ['DC On/Off Ratio STDOM', 'Square Root Of DC On/Off Ratio STDOM', 'DC On/Off Ratio STDOM']
            data_s_name_list = ['RF Generator Power Setting', 'RF System Power Sensor Reading', 'RF System Power Sensor Detected Power']

            analysis_data_info_df = pd.DataFrame(np.array([x_data_column_name_list, y_data_column_name_list, y_data_std_column_name_list]).T, index=data_s_name_list, columns=['X-Axis Data', 'Y-Axis Data', 'Y-Axis STD Data'])

            def calculate_calib_vs_E_field(df):
                rf_channel = df.index.get_level_values('Generator Channel')[0]
                rf_freq = df.index.get_level_values(freq_column_name)[0]

                calib_data_vs_e_field_ampl_df = None

                for rf_e_field_ampl in self.rf_e_field_ampl_arr:

                    calib_data_set_df = None

                    for x_axis_data_type in analysis_data_info_df.index:

                        x_axis_column_name = analysis_data_info_df.loc[x_axis_data_type]['X-Axis Data']

                        poly_fit = self.extracted_E_field_vs_RF_power_fits_set_df.loc[rf_channel, rf_freq, x_axis_data_type]['Polynomial Fit']

                        inverted_spline_fit = self.extracted_E_field_vs_RF_power_fits_set_df.loc[rf_channel, rf_freq, x_axis_data_type]['Inverted Spline Fit']

                        x_data_arr = self.surv_frac_converted_grouped_df.get_group((rf_channel, rf_freq)).reset_index()[x_axis_column_name].values

                        poly_fit_coeff = poly_fit.c

                        # For the case when we look at the RF power sensor reading, then our fit curves are given as the Electric field vs the reading. For other cases the fit curves have their x-axis as the RF power = E field squared.

                        if x_axis_data_type == 'RF System Power Sensor Reading':
                            poly_fit_coeff[-1] = poly_fit_coeff[-1] - rf_e_field_ampl
                            spline_result = inverted_spline_fit([rf_e_field_ampl])[0]
                        else:
                            poly_fit_coeff[-1] = poly_fit_coeff[-1] - rf_e_field_ampl**2
                            spline_result = inverted_spline_fit([rf_e_field_ampl**2])[0]

                        poly_fit_roots = np.roots(poly_fit_coeff)
                        poly_fit_roots = np.real(poly_fit_roots[np.isreal(poly_fit_roots)])

                        poly_fit_filtered_roots = poly_fit_roots[(poly_fit_roots >= np.min(x_data_arr)) & (poly_fit_roots <= np.max(x_data_arr))]


                        poly_fit_root = np.nan

                        if poly_fit_filtered_roots.shape[0] > 1:
                            print('---------------')
                            print(rf_e_field_ampl)
                            print(x_axis_data_type)
                            print(rf_freq)
                            print('WARNING!!! More than one solution has been found for the polynomial')
                        else:
                            if poly_fit_filtered_roots.shape[0] == 0:
                                print('---------------')
                                print(rf_e_field_ampl)
                                print(x_axis_data_type)
                                print(rf_freq)
                                print('WARNING!! No solution has been found for the polynomial. It is possible that filtering the solution is unnecessary. Including the unfiltered result')

                                if poly_fit_roots.shape[0] > 1:
                                    print('But there are more than one solution available. Looking for the closest solution to the available data range')
                                    poly_fit_root = poly_fit_roots[np.argsort(np.abs(poly_fit_roots - np.min(x_data_arr)))][0]
                                else:
                                    poly_fit_root = poly_fit_roots[0]

                            else:
                                poly_fit_root = poly_fit_filtered_roots[0]

                        calib_data_df = pd.DataFrame(pd.Series({'E Field Conversion Polynomial Fit': poly_fit_root, 'E Field Conversion Inverted Spline Fit': spline_result}, name=x_axis_column_name))

                        if calib_data_set_df is None:
                            calib_data_set_df = calib_data_df
                        else:
                            calib_data_set_df = calib_data_set_df.join(calib_data_df)

                    calib_data_set_df.index.names = ['Method']
                    calib_data_set_df.columns.names = ['Data Field']

                    calib_data_set_df['E Field [V/cm]'] = rf_e_field_ampl
                    calib_data_set_df = calib_data_set_df.reset_index().set_index(['E Field [V/cm]', 'Method'])

                    if calib_data_vs_e_field_ampl_df is None:
                        calib_data_vs_e_field_ampl_df = calib_data_set_df
                    else:
                        calib_data_vs_e_field_ampl_df = calib_data_vs_e_field_ampl_df.append(calib_data_set_df)

                return calib_data_vs_e_field_ampl_df

            extracted_E_field_vs_RF_power_fits_set_grouped_df = self.extracted_E_field_vs_RF_power_fits_set_df.groupby(['Generator Channel', freq_column_name])

            extracted_E_field_fits_calib_df = extracted_E_field_vs_RF_power_fits_set_grouped_df.apply(calculate_calib_vs_E_field)

            extracted_E_field_fits_calib_df = extracted_E_field_fits_calib_df.reset_index().set_index(['Generator Channel', 'E Field [V/cm]', 'Method', 'Waveguide Carrier Frequency [MHz]']).sort_index()

            rf_e_field_calib_df = quench_curve_fits_calib_df.append(extracted_E_field_fits_calib_df)

            rf_e_field_calib_df['RF Generator Power Setting [dBm]'] = 10*np.log10(rf_e_field_calib_df['RF Generator Power Setting [mW]'])

            self.rf_e_field_calib_df = rf_e_field_calib_df

        return self.rf_e_field_calib_df

    def get_av_calib_data(self):
        ''' Averages all of the calibration data obtained from different methods together to obtain one pd.DataFrame object with the averaged calibration.
        '''
        if self.calib_av_df is None:
            calib_av_df = self.rf_e_field_calib_df.groupby(['Generator Channel', 'E Field [V/cm]', freq_column_name]).aggregate(['mean', lambda x: np.std(x, ddof=1)])

            calib_av_df.columns.names = ['Calibration Type', 'Data Field']
            calib_av_df = calib_av_df.rename(columns={'mean': 'Mean Value', '<lambda>': 'Data STD'}, level='Data Field')
            self.calib_av_df = calib_av_df
        return self.calib_av_df

    def get_calibration_plot(self, rf_channel, rf_e_field_ampl, axes):
        ''' Plots RF E field calibration vs various RF power parameters obtained via different calibration method.

        Inputs:
        :rf_channel: string of 'A' or 'B'. Label of the RF waveguide.
        :rf_e_field_ampl: float of the E field for which to show the calibration.
        '''

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'Quench Curve Polynomial Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF Generator Power Setting [dBm]', ax=axes[0], color='blue', label='Quench Curve Polynomial Fit')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'Quench Curve Inverted Spline Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF Generator Power Setting [dBm]', ax=axes[0], color='black', label='Quench Curve Inverted Spline Fit')

        self.calib_av_df.loc[rf_channel, rf_e_field_ampl]['RF Generator Power Setting [dBm]'].reset_index().plot(kind='scatter', x=freq_column_name, y='Mean Value', ax=axes[0], color='purple', label='Average')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'E Field Conversion Inverted Spline Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF Generator Power Setting [dBm]', ax=axes[0], color='green', label='E Field Conversion Inverted Spline Fit')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'E Field Conversion Polynomial Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF Generator Power Setting [dBm]', ax=axes[0], color='gray', label='E Field Conversion Polynomial Fit')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'Quench Curve Polynomial Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF System Power Sensor Reading [V]', ax=axes[1], color='blue', label='Quench Curve Polynomial Fit')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'Quench Curve Inverted Spline Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF System Power Sensor Reading [V]', ax=axes[1], color='black', label='Quench Curve Inverted Spline Fit')

        self.calib_av_df.loc[rf_channel, rf_e_field_ampl]['RF System Power Sensor Reading [V]'].reset_index().plot(kind='scatter', x=freq_column_name, y='Mean Value', ax=axes[1], color='purple', label='Average')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'E Field Conversion Inverted Spline Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF System Power Sensor Reading [V]', ax=axes[1], color='green', label='E Field Conversion Inverted Spline Fit')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'E Field Conversion Polynomial Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF System Power Sensor Reading [V]', ax=axes[1], color='gray', label='E Field Conversion Polynomial Fit')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'Quench Curve Polynomial Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF System Power Sensor Detected Power [mW]', ax=axes[2], color='blue', label='Quench Curve Polynomial Fit')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'Quench Curve Inverted Spline Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF System Power Sensor Detected Power [mW]', ax=axes[2], color='black', label='Quench Curve Inverted Spline Fit')

        self.calib_av_df.loc[rf_channel, rf_e_field_ampl]['RF System Power Sensor Detected Power [mW]'].reset_index().plot(kind='scatter', x=freq_column_name, y='Mean Value', ax=axes[2], color='purple', label='Average')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'E Field Conversion Inverted Spline Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF System Power Sensor Detected Power [mW]', ax=axes[2], color='green', label='E Field Conversion Inverted Spline Fit')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'E Field Conversion Polynomial Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF System Power Sensor Detected Power [mW]', ax=axes[2], color='gray', label='E Field Conversion Polynomial Fit')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'Quench Curve Polynomial Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF Generator Power Setting [mW]', ax=axes[3], color='blue', label='Quench Curve Polynomial Fit')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'Quench Curve Inverted Spline Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF Generator Power Setting [mW]', ax=axes[3], color='black', label='Quench Curve Inverted Spline Fit')

        self.calib_av_df.loc[rf_channel, rf_e_field_ampl]['RF Generator Power Setting [mW]'].reset_index().plot(kind='scatter', x=freq_column_name, y='Mean Value', ax=axes[3], color='purple', label='Average')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'E Field Conversion Inverted Spline Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF Generator Power Setting [mW]', ax=axes[3], color='green', label='E Field Conversion Inverted Spline Fit')

        self.rf_e_field_calib_df.loc[rf_channel, rf_e_field_ampl, 'E Field Conversion Polynomial Fit'].reset_index().plot(kind='scatter', x=freq_column_name, y='RF Generator Power Setting [mW]', ax=axes[3], color='gray', label='E Field Conversion Polynomial Fit')

        for ax in axes:
            ax.legend()

        return axes

    def set_av_rf_power_calib_error(self, av_RF_power_calib_error_df):
        ''' Stores that average RF power error in the class instance that can later be used to access the information about the RF power calibration uncertainty at each required RF E field amplitude in the waveguide.

        Notice that the uncertainty is assumed to be the same for all RF frequencies = systematic uncertainty. The uncertainties given are not random. It is not true that if we have 1% uncertainty for one RF power and 2% uncertainty for another power, then the first power could be 0.5% off, whereas the second power could be -1% off. No, both of them are off by the same amount. This way we are looking at the maximum possible error in the RF power calibration and also the resulting AC shift.

        Inputs:
        :av_RF_power_calib_error_df: pd.DataFrame object of mean RF power calibration uncertainty vs quantity proportional to the RF power inside each of the waveguides.
        '''
        self.av_RF_power_calib_error_df = av_RF_power_calib_error_df


    def get_av_rf_power_calib_error(self):
        return self.av_RF_power_calib_error_df

    def load_instance(self):
        ''' This function loads previously pickled class instance.

        The data contained in the pickled file gets loaded into this class instance. The function is very useful if one does not want to reanalyze data again.
        '''
        os.chdir(self.saving_folder_location)
        os.chdir(self.calib_folder_name)

        f = open(self.analysis_data_file_name, 'rb')
        loaded_dict = pickle.load(f)
        f.close()
        self.__dict__.update(loaded_dict)
        print('The class instance has been loaded')

        # Interesting side effect of the loading function. In case the data stored from a different computer, the path to the saved data can be different. Thus what happens is that the loaded data overwrites the self.saving_folder_location attribute. I need to set it to the current saving folder location in order to avoid errors.

        self.saving_folder_location = saving_folder_location


        os.chdir(self.saving_folder_location)

    def save_instance(self):
        ''' Calling this function pickles the analysis class instance. If the data has been previously saved, the call to this function overwrites previously written pickled file with the same file name.

        If the required function for certain types of data to save has not be called before, then it gets called here.
        '''

        # Created folder that will contain all of the analyzed data
        os.chdir(self.saving_folder_location)

        if os.path.isdir(self.calib_folder_name) == False:
            os.mkdir(self.calib_folder_name)
        else:
            print('Saving data folder already exists.')

        os.chdir(self.calib_folder_name)

        f = open(self.analysis_data_file_name, 'wb')
        pickle.dump(self.__dict__, f, 2)

        os.chdir(self.saving_folder_location)

        print('The class instance has been saved')
