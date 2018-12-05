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
from exp_data_analysis import *
from KRYTAR_109_B_Calib_analysis import *
import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt


import threading
from queue import Queue

# Package for wrapping long string to a paragraph
import textwrap

import pytz

import hydrogen_sim_data
import wvg_power_calib_analysis

import pickle
#%%
# Location where the analyzed experiment is saved
# Home location
saving_folder_location = 'E:/2017-10-17 Lamb Shift Measurement/Data/FOSOF analyzed data sets'
# Lab location
#saving_folder_location = 'C:/Research/Lamb shift measurement/Data/FOSOF analyzed data sets'

# File containing parameters and comments about all of the data sets.
exp_info_file_name = 'fosof_data_sets_info.csv'
exp_info_index_name = 'Experiment Folder Name'

# Analysis version of the analyzed raw data file to use
raw_data_analyzed_version_number = 0.1

# FOSOF data set analysis version_number
version_number = 0.2

# Analyzed data file name
data_analyzed_file_name = 'data_analyzed v' + str(raw_data_analyzed_version_number) + '.txt'

analyzed_data_folder = 'Data Analysis ' + str(version_number)

# Waveguide RF power calibration analyzed data folder path
wvg_power_calib_data_folder_path = 'E:/Google Drive/Research/Lamb shift measurement/Data/Waveguide calibration'

wvg_power_calib_info_file_name = 'Waveguide_calibration_info.csv'

os.chdir(wvg_power_calib_data_folder_path)

# Waveguide power calibration info
wvg_calib_info_df = pd.read_csv(wvg_power_calib_info_file_name, header=2)

wvg_calib_info_df['Start Date'] = pd.to_datetime(wvg_calib_info_df['Start Date'])
wvg_calib_info_df['End Date'] = pd.to_datetime(wvg_calib_info_df['End Date'])

# File for storing externally written Proton Deflector measured voltages
exp_pd_volt_file_name = 'fosof_exp_list_PD_voltage.csv'

os.chdir(saving_folder_location)
exp_pd_volt_df = pd.read_csv(filepath_or_buffer=exp_pd_volt_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True, index_col=0)

pd_volt_col_name = 'Proton Deflector Voltage [V]'

class DataSetFOSOF():
    ''' General class for FOSOF experiments. It contains all of the required analysis functions
    '''

    def __init__(self, exp_folder_name, load_Q=True, beam_rms_rad_to_load=None):
        ''' Here we open the data file and extract the run parameters and also the type of the data set that was acquired (B field scan or pre-quench 910 switching, etc).

        Inputs:
        :exp_folder_name: Folder name of the experiment that needs to be analyzed
        :load_data_Q: Flag whether to load saved analysis data, if it exists
        :beam_rms_rad_to_load: float [mm]. Only used when load_data_Q is True. This is the RMS radius of the beam assumed for the FOSOF simulations used to correct the FOSOF data for imperfect RF power. If this value is set to None, then uncorrected object gets loaded (if exists)
        '''

        # Version of the data analysis. Notice that this version number is 'engraved' in the analysis data file name. Thus one has to use the appropriate data analysis version.
        self.version_number = 0.1

        self.exp_folder_name = exp_folder_name

        self.beam_rms_rad_to_load = beam_rms_rad_to_load

        if load_Q:
            self.analysis_data_file_name_to_load = self.get_analysis_data_object_file_name(beam_rms_rad_to_load)

            # Checking if the class instance has been saved before. If it was, then in case the user wants to load the data, it gets loaded. In all other cases the initialization continues.

            self.perform_analysis_Q = False

            os.chdir(saving_folder_location)
            os.chdir(self.exp_folder_name)

            if os.path.isfile(self.analysis_data_file_name_to_load):
                print('The analysis instance has been previously saved.')
                print('Loading the analysis data...')
                self.load_instance()
                # Interesting effect here: if the data has been saved before with the perform_analysis_Q flag set to True, then after loading the data it sets is flag to True, making the analysis code believe that the analysis has not been performed before. That is why I have to set it to False here.
                self.perform_analysis_Q = False
            else:
                print('No analysis instance of version ' + str(self.version_number) + ' has been found.')
                self.perform_analysis_Q = True
        else:
            self.perform_analysis_Q = True

        # The analysis gets performed if the perform_analysis_Q flag is True.
        if self.perform_analysis_Q:

            # Reading the csv table containing information about data sets.
            os.chdir(saving_folder_location)
            self.exp_info_df = pd.read_csv(filepath_or_buffer=exp_info_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True)

            # Remove possible tabs and whitespaces from the end of the experiment folder names
            self.exp_info_df[exp_info_index_name] = self.exp_info_df[exp_info_index_name].transform(lambda x: x.strip())

            self.exp_info_df = self.exp_info_df.set_index(exp_info_index_name)

            # Open the file containing the analyzed traces
            os.chdir(self.exp_folder_name)

            self.exp_data_frame = pd.read_csv(filepath_or_buffer=data_analyzed_file_name, delimiter=',', comment='#', header=0)

            # Add a column of elapsed time since the start of the acquisition
            self.exp_data_frame['Elapsed Time [s]'] = self.exp_data_frame['Time'].transform(lambda x: x-x[0])

            # There is additional space before [V] in the column below. Fixing it here
            self.exp_data_frame = self.exp_data_frame.rename(columns={'Waveguide A Power Reading  [V]': 'Waveguide A Power Reading [V]'})

            # Get the data set parameters
            self.exp_params_dict, self.comment_string_arr = get_experiment_params(data_analyzed_file_name)

            # Adding additional parameters into the Dictionary

            # Comments field from the info file of FOSOF data sets.
            if exp_folder_name in self.exp_info_df.index:
                self.exp_params_dict['Comments'] = self.exp_info_df.loc[exp_folder_name]['Comments']

            # Acquisition start time given in UNIX format = UTC time
            self.exp_params_dict['Experiment Start Time [s]'] = self.exp_data_frame['Time'].min()

            # Acquisition duration [s]
            self.exp_params_dict['Experiment Duration [s]'] = self.exp_data_frame['Elapsed Time [s]'].max()

            # Series containing flags to determine what kind of data set has been acquired. It is possible for several flags to be True, however, not for all of the combinations of flags there exists analysis at the moment.
            self.data_set_type_s = pd.Series(
                        {'B Field Scan': False,
                        'Pre-910 Switching': False,
                        'Waveguide Carrier Frequency Sweep': False,
                        'Offset Frequency Switching': False,
                        'Charge Exchange Flow Rate Scan': False})

            # Important experiment parameters
            self.n_Bx_steps = self.exp_params_dict['Number of B_x Steps']
            self.n_By_steps = self.exp_params_dict['Number of B_y Steps']
            self.n_averages = self.exp_params_dict['Number of Averages']
            self.sampling_rate = self.exp_params_dict['Digitizer Sampling Rate [S/s]']
            self.n_digi_samples = self.exp_params_dict['Number of Digitizer Samples']
            self.n_freq_steps = self.exp_params_dict['Number of Frequency Steps']
            self.n_repeats = self.exp_params_dict['Number of Repeats']
            self.digi_ch_range = self.exp_params_dict['Digitizer Channel Range [V]']
            self.offset_freq_array = np.array(self.exp_params_dict['Offset Frequency [Hz]'])
            self.pre_910_on_n_digi_samples = self.exp_params_dict['Pre-Quench 910 On Number of Digitizer Samples']

            # Expected number of rows in the data file
            self.rows_expected = self.n_repeats * self.n_freq_steps * len(self.offset_freq_array) * self.n_averages * 2

            if self.n_freq_steps >= 1:
                self.data_set_type_s['Waveguide Carrier Frequency Sweep'] = True

            # B field values for different axes are assumed to be part of the same parameter: B field in such a way that when the B field along one axis is getting scanned, B field along another axis is set to 0. Thus the total number of B field scan steps is the sum of the scan steps along each axis.  When both the Bx and By are set to zero and are not getting scanned though, the B field parameter is not getting changed, thus the total number of B field steps is 1 (one).
            if self.exp_params_dict['B_x Min [Gauss]'] == 0 and self.exp_params_dict['B_x Max [Gauss]'] == 0 and self.exp_params_dict['B_y Min [Gauss]'] == 0 and self.exp_params_dict['B_y Max [Gauss]'] == 0:
                self.n_B_field_steps = 1

            else:
                self.data_set_type_s['B Field Scan'] = True
                if self.n_Bx_steps == 1 and self.n_By_steps > 1:
                    self.n_B_field_steps = self.n_By_steps
                if self.n_By_steps == 1 and self.n_Bx_steps > 1:
                    self.n_B_field_steps = self.n_Bx_steps
                if self.n_Bx_steps > 1 and self.n_By_steps > 1:
                    self.n_B_field_steps = self.n_Bx_steps + self.n_By_steps

            self.rows_expected = self.rows_expected * self.n_B_field_steps

            if self.exp_params_dict['Pre-Quench 910 On/Off']:
                self.data_set_type_s['Pre-910 Switching'] = True
                self.rows_expected = self.rows_expected * 2

            if 'Number of Mass Flow Rate Steps' in self.exp_params_dict:
                self.data_set_type_s['Charge Exchange Flow Rate Scan'] = True
                self.rows_expected = self.rows_expected * self.exp_params_dict['Number of Mass Flow Rate Steps']

            if self.offset_freq_array.shape[0] > 1:
                self.data_set_type_s['Offset Frequency Switching'] = True

            self.data_set_type = None

            if self.data_set_type_s['Waveguide Carrier Frequency Sweep'] == False:
                print('There is no frequency scan for the data set. No analysis will be performed.')
            else:
                self.data_set_type_no_freq_s = self.data_set_type_s.drop(labels=['Waveguide Carrier Frequency Sweep', 'Charge Exchange Flow Rate Scan'])
                if self.data_set_type_no_freq_s[self.data_set_type_no_freq_s==True].size > 1:
                    print('In addition to the Waveguide Carrier Frequency and Charge Exchange flow rate scan more than one parameter was scanned. The analysis will not be performed')
                else:
                    self.data_set_type = self.data_set_type_s[self.data_set_type_s==True].index.values[0]

            if self.exp_data_frame.shape[0] > self.rows_expected:
                raise FosofAnalysisError("Number of rows in the analyzed data file is larger than the expected number of rows")

            if self.exp_data_frame.shape[0] < self.rows_expected:
                print('Seems that the data set has not been fully acquired/analyzed')

            # Fourier harmonics of the offset frequency that are present in the data set. This list will be used throughout the analysis. Modify if needed.
            self.harmonic_name_list = ['First Harmonic', 'Second Harmonic']

            # The data set contains many columns. Some columns are related to each other by sharing one or several common properties. It is convenient to group these columns together. Each of the groups might have its unique structure for subsequent analysis. Thus it is inconvenient to have single data frame. Thus I now categorize the data set into several smaller subsets. Each subset corresponds to data for specific category. The categories at the moment are digitizers, quenching cavities, beam end faraday cup, and RF power in the waveguides.

            # Each subset contains the same general index that depends on the type of the experiment.
            # 'index' and 'Elapsed Time [s]' are position at the end of the list to make sure that the resulting multiindex can be sorted.
            self.index_column_list = ['Repeat', 'Configuration', 'Average', 'Elapsed Time [s]']

            if self.data_set_type_s['Waveguide Carrier Frequency Sweep']:
                self.index_column_list.insert(self.index_column_list.index('Configuration'), 'Waveguide Carrier Frequency [MHz]')

            if self.data_set_type_s['Offset Frequency Switching']:
                self.index_column_list.insert(self.index_column_list.index('Configuration'), 'Offset Frequency [Hz]')

            if self.data_set_type_s['B Field Scan']:
                if self.n_Bx_steps > 1:
                    self.b_field_column_name = 'B_x [Gauss]'
                    self.index_column_list.insert(self.index_column_list.index('Configuration'), self.b_field_column_name)

                if self.n_By_steps > 1:
                    self.b_field_column_name = 'B_y [Gauss]'
                    self.index_column_list.insert(self.index_column_list.index('Configuration'), self.b_field_column_name)

                # The analysis of the B field data for now is designed only for the B field along one axis to change. If the B field was changing along multiple axes during the data analysis, then analysis of the B field data cannot be performed.

                # Boolean to check whether the B field analysis is allowed or not.
                self.b_field_analysis_allowed_Q = True

                if (self.n_Bx_steps > 1) and (self.n_By_steps > 1):
                    print('B field was changing along both x- and y-axes. The B field data analysis cannot be performed.')
                    self.b_field_analysis_allowed_Q = False

            if self.data_set_type_s['Charge Exchange Flow Rate Scan']:
                self.index_column_list.insert(self.index_column_list.index('Configuration'), 'Mass Flow Rate [sccm]')

            if self.data_set_type_s['Pre-910 Switching']:
                self.index_column_list.insert(self.index_column_list.index('Configuration'), 'Pre-Quench 910 State')


            # Add experiment type into the experiment parameters Dictionary
            self.exp_params_dict['Experiment Type'] = self.data_set_type

            # Create pd.Series containing all of the experiment parameters.
            self.exp_params_s = pd.Series(self.exp_params_dict)

            # Some earlier data sets did not have the accelerating voltage recorded. They were all acquired at 49.86 kV.
            if 'Accelerating Voltage [kV]' not in list(self.exp_params_s.keys()):
                self.exp_params_s['Accelerating Voltage [kV]'] = 49.86

            # If data is available from the external source, record the Proton Deflector measured voltage for the data set

            if self.exp_folder_name in list(exp_pd_volt_df.index.values):

                self.exp_params_s[pd_volt_col_name] = exp_pd_volt_df.loc[self.exp_folder_name][pd_volt_col_name]

            # Changing the index columns of the data dataframe
            self.exp_data_frame.set_index(self.index_column_list, inplace=True)
            # Index used for the pd.DataFrame objects. It contains names of all of the parameters that were varied (intentionally) during the acquisition process.
            self.general_index = self.exp_data_frame.index
            # Renaming 'index' name to 'Index' for enforcing first letter = capital letter rule
            #self.general_index.set_names('Index',level=list(self.general_index.names).index('index'), inplace=True)

            # Defining variables that the data analysis functions will assign the analyzed data to. These are needed so that whenever we want to call the function again, the analysis does not have to be redone. Also, most of the functions use data obtained from other function calls. We want to make it automatic for the function that needs other variables to call required functions. If the function has been called before we do not have to again wait for these functions to needlessly rerun the analysis.

            # Beam-end faraday cup dataframe
            self.beam_end_dc_df = None
            # Quenching cavities dataframe
            self.quenching_cavities_df = None
            # RF power detector data (with analysis) for both RF systems (A and B)
            self.rf_system_power_df = None

            # Dataframe for the data from the digitizers
            self.digitizers_data_df = None
            # Phase difference between the RF Combiners DataFrame
            self.combiner_difference_df = None

            # Dataframe of delays between the digitizers
            self.digi_2_from_digi_1_delay_df = None
            self.digi_2_from_digi_1_mean_delay_df = None

            # Detector FOSOF phasors dataframe relative to RF combiners. Most important data.
            self.phase_diff_data_df = None
            # FOSOF phasors averaged across all averaging sets using different types of averaging technique
            self.phase_av_set_averaged_df = None
            # FOSOF phases with subtracted out frequency response at the offset frequency
            self.phase_A_minus_B_df = None
            # FOSOF data (amplitude and phase) averaged across the repeats. This is final analyzed data set that is used for combining it with the same FOSOF data set with the same parameters but different configuration
            self.fosof_ampl_df = None
            self.fosof_phases_df = None

            # pd.DataFrame for storing determine E field values using the appropriate waveguide calibration file
            self.avg_e_field_df = None

            # Flag for whether the FOSOF phases were corrected for the improper wavegude power or not
            self.wvg_pwr_fosof_correc_Q = False

            # RMS radius of the beam assumed for the FOSOF simulations used to correct the FOSOF phases for imperfect RF power. When is set to NONE, it means that no correction has been applied to the data.
            self.beam_rms_rad = None

            # Boolean for whether the phase difference data has been filtered for RF power measurements' outliers, when these outliers correspond exactly to very low FOSOF signal SNR at the offset frequency. This variable is needed for the correction of the FOSOF phases due to imperfect waveguide power.
            self.phase_diff_data_filtered_Q = False

            # pd.DataFrame object for storing any errors/warnings in the analysis. This file gets later saved along with the rest of the analysis data.
            self.err_warn_df = pd.DataFrame()
            self.err_warn_df.index.name = 'Warning/Error Name'

            # B field scan analysis data
            if self.data_set_type_s['B Field Scan'] and self.b_field_analysis_allowed_Q:
                self.fosof_av_phase_B_field_df = None

    def get_analysis_data_object_file_name(self, beam_rms_rad):
        ''' Get name of the analysed data object
        '''
        if beam_rms_rad is None:
            analysis_data_file_name = 'r' + 'NA' + 'v' + str(self.version_number) + '.pckl'
        else:
            analysis_data_file_name = 'r' + str(beam_rms_rad) + 'v' + str(self.version_number) + '.pckl'
        return analysis_data_file_name

    def get_exp_parameters(self):
        ''' Simply returns the pd.Series with the experiment parameters.
        '''
        return self.exp_params_s

    def get_fc_data(self):
        ''' Obtain data for the beam end faraday cup.
        '''
        if self.beam_end_dc_df is None:
            self.beam_end_dc_df = self.exp_data_frame[self.exp_data_frame.columns[match_string_start(self.exp_data_frame.columns, 'fc')]]
        return self.beam_end_dc_df

    def get_quenching_cav_data(self):
        ''' Obtain data that corresponds to both post- and pre-quenching cavity stacks.
        '''
        if self.quenching_cavities_df is None:
            # Grouping Quenching cavities data together
            level_value = 'Post-Quench'

            # Selected dataframe
            post_quench_df = self.exp_data_frame[self.exp_data_frame.columns.values[match_string_start(self.exp_data_frame.columns.values, level_value)]]

            post_quench_df = post_quench_df.rename(columns=remove_matched_string_from_start(post_quench_df.columns.values, level_value))

            post_quench_df = add_level_data_frame(post_quench_df, 'Quenching Cavity', ['910', '1088', '1147'])
            #post_quench_df.columns.set_names('Data Field', level=1, inplace=True)

            level_value = 'Pre-Quench'
            # Selected dataframe
            pre_quench_df = self.exp_data_frame[self.exp_data_frame.columns.values[match_string_start(self.exp_data_frame.columns.values, level_value)]]

            pre_quench_df = pre_quench_df.rename(columns=remove_matched_string_from_start(pre_quench_df.columns.values, level_value))

            pre_quench_df = add_level_data_frame(pre_quench_df, 'Quenching Cavity', ['910', '1088', '1147'])
            #pre_quench_df.columns.names = ['Quenching Cavity', 'Data Field']

            if self.exp_params_dict['Experiment Type'] != 'Pre-910 Switching':
                # No need to have the state of pre-910 quenching cavity, since the state is the same, if we do not have 'pre-910 Switching' experiment
                pre_quench_df.drop(columns=[('910','State')], inplace=True)

            # Combine pre-quench and post-quench data frames together
            quenching_cavities_df = pd.concat([pre_quench_df, post_quench_df], keys=['Pre-Quench','Post-Quench'], names=['Cavity Stack Type'], axis='columns')

            self.quenching_cavities_df = quenching_cavities_df

        return self.quenching_cavities_df

    def get_rf_sys_pwr_det_data(self):
        ''' Obtain data for RF power detectors. Also returns the outliers in the RF power detector readings. The outliers can be used to filter the FOSOF phases.
        '''

        if self.rf_system_power_df is None:

            # Notice that here I am not allowing for bypassing the whole function, if the function has been called for before. The reason for this is the ax object for storing the plot of the RF power detector calibration.

            # Data from the RF Power detectors for each RF system (A and B).
            rf_system_power_df = self.exp_data_frame[['Waveguide A Power Reading [V]','Waveguide B Power Reading [V]']]

            # Restructure the dataframe a bit
            rf_system_power_df = add_level_data_frame(rf_system_power_df, 'RF System', ['Waveguide A', 'Waveguide B']).rename(columns={'Waveguide A': 'RF System A', 'Waveguide B': 'RF System B'}).rename(columns={'Power Reading [V]': 'RF Power Detector Reading [V]'}, level=1).sort_index(level='Elapsed Time [s]')

            rf_system_power_df.columns.names = ['RF System', 'Data Field']

            # It is possible to get positive voltage values as the reading on the power detectors (I have encountered this), which can only, in principle, output numbers that are negative or zero. The reason for this is unknown: possibly the multimeter that reads the voltage from the power detector might give erroneous readings from time to time. In any case, these positive readings are discarded from the rf_syste_power_df, but not from the data set itself. If I do not discard them, then it gives me issue with the conversion of the voltages to RF power. The indeces corresponding to these get also recorded.

            rf_pow_det_faulty_df = rf_system_power_df[(rf_system_power_df['RF System A', 'RF Power Detector Reading [V]'] > 0) | (rf_system_power_df['RF System B', 'RF Power Detector Reading [V]'] > 0)]

            # Dataframe to store the outliers in the power detector data.
            rf_system_power_outlier_df = rf_pow_det_faulty_df

            if rf_pow_det_faulty_df.shape[0] > 0:
                rf_system_power_df = rf_system_power_df.drop(rf_pow_det_faulty_df.index)

                warning_mssg = 'WARNING! Positive KRYTAR 109B voltages detected! These readings are discarded from the list of RF power detector readings. Notice, that they are not removed from the overall experiment data, but only from the dataframe of the RF power readings. Indeces corresponding to these positive readings:\n'+str(rf_pow_det_faulty_df.index.values)

                # Store the warning in the errors/warnings log
                self.err_warn_df = self.err_warn_df.append(pd.Series({'Message': warning_mssg}, name='Positive KRYTAR 109B Voltage'))
                print(warning_mssg)


            # Get the RF power calibration
            self.wvg_pwr_det_calib = KRYTAR109BCalibration()
            self.wvg_pwr_det_calib.get_calib_curve()

            # Convert Detector Voltages to RF power in dBm. We use only the first element of the conversion function output. The second element is the uncertainty in the RF power.
            rf_system_power_df = rf_system_power_df.join(rf_system_power_df.transform(lambda x: self.wvg_pwr_det_calib.get_RF_power_dBm_from_voltage(x)[:,0]).rename(columns={'RF Power Detector Reading [V]':'Detected RF Power [dBm]'}, level='Data Field')).sort_index(axis='columns')

            # Convert dBm to mW
            rf_system_power_df = rf_system_power_df.join(rf_system_power_df.loc[slice(None), (slice(None), ['Detected RF Power [dBm]'])].transform(lambda x: 10**(x/10)).rename(columns={'Detected RF Power [dBm]':'Detected RF Power [mW]'}, level='Data Field')).sort_index(axis='columns')

            # We now want to see by how much the power in each system has changed/drifted while acquiring data. The field with power in dBm is not needed for this calculation

            # We first select data that corresponds to the first repeat only. Then we select only first occurences (in time) of each RF frequency.
            rf_system_power_repeat_1 = rf_system_power_df.loc[slice(None),(slice(None), ['Detected RF Power [mW]', 'RF Power Detector Reading [V]'])].reset_index().set_index('Elapsed Time [s]').sort_index()

            rf_system_power_initial_df = rf_system_power_repeat_1.loc[rf_system_power_repeat_1[rf_system_power_repeat_1['Repeat'] == 1]['Waveguide Carrier Frequency [MHz]'].sort_index().drop_duplicates(keep='first').sort_index().index]

            # We need to remove all other index levels, except that of the 'Waveguide Carrier Frequency [MHz]'.

            rf_system_power_initial_column_list = list(self.general_index.names)
            rf_system_power_initial_column_list.remove('Waveguide Carrier Frequency [MHz]')

            rf_system_power_initial_df = rf_system_power_initial_df.reset_index().set_index('Waveguide Carrier Frequency [MHz]').sort_index().drop(columns=rf_system_power_initial_column_list, level='RF System')

            # We now calculate the fractional change in the RF power detector voltage and detector power given in parts per thousand [ppt]

            rf_system_power_fract_change_df = (rf_system_power_df.loc[slice(None),(slice(None), ['Detected RF Power [mW]', 'RF Power Detector Reading [V]'])].reset_index().set_index(list(self.general_index.names)) - rf_system_power_initial_df)/rf_system_power_initial_df * 1E3

            # Renaming columns and joining with the main rf power dataframe.
            rf_system_power_df = rf_system_power_df.join(rf_system_power_fract_change_df.rename(columns={'Detected RF Power [mW]': 'Fractional Change In RF Power [ppt]', 'RF Power Detector Reading [V]': 'Fractional Change In RF Detector Voltage [ppt]'}, level='Data Field')).sort_index(axis='columns')

            ''' In addition to a possibility of having positive RF power detector voltages, due to some glitch/malfunction in the Keithley logger for recording the measurements or the RF power detector(s), it is also possible to have sometimes values that are negative, but very close to zero, as seen in '180506-123908 - FOSOF Acquisition - 0 config, 14 V per cm, PD ON 120 V,  914-918 MHz'. For this particular data set the index values corresponding to these RF power detector voltage readings also matched with the case when SNR for the first harmonic was NAN. Thus is seems that in this case this was the real effect - maybe RF generator somehow did not supply properly for these index labels. But mainly the reason for being able to see this, was that when trying to find the corresponding E field, I got NAN values for these, which later on gave me an error. In any case, I decided that I need to make sure to detect any obvious outliers in the RF power detector readings. I tried finding the average for each RF System (A or B) and then by how many standard deviation each value was off from the mean (tried to do the same for each averaging set, but there are not enough points to detect an outlier with this method), but it did not give me good results. Later, using histograms, I realized that for that same data set, the RF power detector voltages are peaked around 2 values + a tiny third peak = outliers. This gave me the realization that it is possible for the different RF frequencies to have quite different RF power detector voltages, in general. Thus I should look at the fractional deviation in RF power detector voltages, given for each RF frequency separately, where the deviation is w.r.t. the first occurence of RF power detector reading at the particular frequency (in time). This way I expect to have the values peaking about the same number + some small outliers.

            Later on while analyzing data for the data set '180327-122234 - FOSOF Acquisition - pi config, 5 V per cm PD ON 120 V' I had one outlier, but it actually had the Fractional Change in RF Power Detector Voltage of only -3 ppt. It was that other values were much smaller. To make sure that these cases will not be counter as the outliers, I added an additional condition that the absolute difference between the mean and the given value needs to be larger than certain maximum allowed ppt.
            '''
            # Maximum allowed deviation in the multiples of the standard deviation.
            max_dev_std_mult = 10

            # In addition to the maximum multiple of stadard deviations, the absolute difference of the outlier and the mean needs to be larger than the specified amount in ppt.
            max_ppt = 10

            # RF System A outliers
            mean_val = np.mean(rf_system_power_df.loc[slice(None), ('RF System A', 'Fractional Change In RF Detector Voltage [ppt]')])
            std_val = np.std(rf_system_power_df.loc[slice(None), ('RF System A', 'Fractional Change In RF Detector Voltage [ppt]')])

            rf_sys_A_diff_df = np.abs(rf_system_power_df.loc[slice(None), ('RF System A', ['Fractional Change In RF Detector Voltage [ppt]'])] - mean_val)

            rf_sys_A_outlier_df = rf_sys_A_diff_df / std_val
            rf_sys_A_outlier_df = rf_sys_A_outlier_df[(rf_sys_A_outlier_df['RF System A', 'Fractional Change In RF Detector Voltage [ppt]'] > max_dev_std_mult) & (rf_sys_A_diff_df['RF System A', 'Fractional Change In RF Detector Voltage [ppt]'] > max_ppt)]

            # RF System B outliers
            mean_val = np.mean(rf_system_power_df.loc[slice(None), ('RF System B', 'Fractional Change In RF Detector Voltage [ppt]')])
            std_val = np.std(rf_system_power_df.loc[slice(None), ('RF System B', 'Fractional Change In RF Detector Voltage [ppt]')])

            rf_sys_B_diff_df = np.abs(rf_system_power_df.loc[slice(None), ('RF System B', ['Fractional Change In RF Detector Voltage [ppt]'])] - mean_val)

            rf_sys_B_outlier_df = rf_sys_B_diff_df / std_val
            rf_sys_B_outlier_df = rf_sys_B_outlier_df[(rf_sys_B_outlier_df['RF System B', 'Fractional Change In RF Detector Voltage [ppt]'] > max_dev_std_mult) & (rf_sys_B_diff_df['RF System B', 'Fractional Change In RF Detector Voltage [ppt]'] > max_ppt)]

            # Index corresponding to the outliers
            rf_pow_outlier_index = rf_sys_A_outlier_df.index.union(rf_sys_B_outlier_df.index)

            rf_system_power_outlier_df = rf_system_power_outlier_df.append(rf_system_power_df.loc[rf_pow_outlier_index])

            if rf_pow_outlier_index.shape[0] > 0:

                rf_system_power_df = rf_system_power_df.drop(rf_pow_outlier_index)

                warning_mssg = 'WARNING! Negative outliers in the KRYTAR 109B voltages were detected! These readings are discarded from the list of RF power detector readings. Notice, that they are not removed from the overall experiment data, but only from the dataframe of the RF power readings. Indeces corresponding to these outlying readings:\n'+str(rf_pow_outlier_index.values)

                # Store the warning in the errors/warnings log
                self.err_warn_df = self.err_warn_df.append(pd.Series({'Message': warning_mssg}, name='Negative outliers in the KRYTAR 109B readings'))
                print(warning_mssg)

            self.rf_system_power_outlier_df = rf_system_power_outlier_df
            self.rf_system_power_df = rf_system_power_df

        return self.rf_system_power_df, self.rf_system_power_outlier_df

    def get_digitizers_data(self):
        ''' Main data containing the extracted data from the digitizers.
        '''

        if self.digitizers_data_df is None:
            # Now we group the data from the digitizers together.
            # Pick all the columns that have the 'level_value' as the first characters.
            level_value = 'Detector'

            # Selected dataframe
            detector_df = self.exp_data_frame[self.exp_data_frame.columns.values[match_string_start(self.exp_data_frame.columns.values, level_value)]]

            detector_df = detector_df.rename(columns=remove_matched_string_from_start(detector_df.columns.values, level_value))

            detector_df = add_level_data_frame(detector_df, 'Data Type', self.harmonic_name_list)

            # Pick all the columns that have the 'level_value' as the first characters.
            level_value = 'RF Power Combiner I Digi 1'

            # Selected dataframe
            combiner_I_digi_1_df = self.exp_data_frame[self.exp_data_frame.columns.values[match_string_start(self.exp_data_frame.columns.values, level_value)]]

            combiner_I_digi_1_df = combiner_I_digi_1_df.rename(columns=remove_matched_string_from_start(combiner_I_digi_1_df.columns.values, level_value))

            combiner_I_digi_1_df = add_level_data_frame(combiner_I_digi_1_df, 'Data Type', self.harmonic_name_list)

            # Pick all the columns that have the 'level_value' as the first characters.
            level_value = 'RF Power Combiner I Digi 2'

            # Selected dataframe
            combiner_I_digi_2_df = self.exp_data_frame[self.exp_data_frame.columns.values[match_string_start(self.exp_data_frame.columns.values, level_value)]]

            combiner_I_digi_2_df = combiner_I_digi_2_df.rename(columns=remove_matched_string_from_start(combiner_I_digi_2_df.columns.values, level_value))

            combiner_I_digi_2_df = add_level_data_frame(combiner_I_digi_2_df, 'Data Type', self.harmonic_name_list)

            # Pick all the columns that have the 'level_value' as the first characters.
            level_value = 'RF Power Combiner R'

            # Selected dataframe
            combiner_R_df = self.exp_data_frame[self.exp_data_frame.columns.values[match_string_start(self.exp_data_frame.columns.values, level_value)]]

            combiner_R_df = combiner_R_df.rename(columns=remove_matched_string_from_start(combiner_R_df.columns.values, level_value))

            combiner_R_df = add_level_data_frame(combiner_R_df, 'Data Type', self.harmonic_name_list)

            # Combine the phasor data together

            # This is the wrong way of doing this, because when transposing, the dtype of the columns is lost - gets turned into 'object'
            #phasor_data_df = pd.concat([detector_df.T, combiner_I_digi_1_df.T, combiner_I_digi_2_df.T, combiner_R_df.T], keys = ['Detector', 'RF Combiner I Digi 1', 'RF Combiner I Digi 2', 'RF Combiner R']).T

            phasor_data_df = pd.concat([detector_df, combiner_I_digi_1_df, combiner_I_digi_2_df, combiner_R_df], keys = ['Detector', 'RF Combiner I Digi 1', 'RF Combiner I Digi 2', 'RF Combiner R'], axis='columns')

            phasor_data_df = phasor_data_df.sort_index()
            phasor_data_df.columns.rename(['Source','Data Type','Data Field'], inplace=True)

            self.digitizers_data_df = phasor_data_df

        return self.digitizers_data_df

    def get_combiners_phase_diff_data(self):
        ''' Obtain phase difference between combiners corrected for the possible phase shift due to frequency response of the RF combiners at the offset frequency. This phase shift is also calculated.

        It is critical to realize that the corrected phase differences are not divided by 2, as they are supposed to, for not having to keep track of the maximum range of phases allowed (division by 2 will change if from [0, 2pi) to [0, pi)). However, the frequency response = phase shift is divided by two.

        Outputs:
        :combiner_difference_df: pd.DataFrame of phase differences between the combiners with its change compared to initial phase difference in time.
        :combiner_phase_diff_freq_response_df: pd.DataFrame of the phase shifts, which are due to the average frequency response of both the Combiner I and Combiner R.
        '''
        if self.combiner_difference_df is None:
            # Forming the data frame with the combiners' phase difference stability vs time.
            # In principle the power detectors (and the digitizer) that we use on the combiners (ZX47-**LN+ from Mini Circuits) have phase shift due to its frequency response at the offset frequency. This phase shift should be cancelled out in a similar way as for the Detector signal. The difference is that we cannot average phases for each combiner for the given averaging set separately, because these phases cannot be referenced to anything. Thus we need to calculate phase difference between RF Combiner I and RF Combiner R for every average and then find the difference between these phase differences for the same average of configurations 'A' and 'B'.
            # This quantity divided by two should cancel out the phase shift due to the frequency response of the RF power detectors at the offset frequency. Overall, we do not really expect the frequency response of the power detectors to change by any appreciable amount, because they are in a temperature stabilized enclosure, but nevertheless it the correct way of looking at the phase difference between the combiners.

            if self.digitizers_data_df is None:
                self.get_digitizers_data()

            # We first select the combiner phase data and sort it by the elapsed time
            combiner_phase_df = self.digitizers_data_df.loc[slice(None), (['RF Combiner I Digi 2', 'RF Combiner R'], self.harmonic_name_list, ['Fourier Phase [Rad]'])].sort_index(level='Elapsed Time [s]')

            # Consider having Combiner I phase of 0.1 rad and Combiner R phase of 6.2 rad. One can calculate the phase difference to be 0.1-6.2 = -6.1 = roughly 0.1 rad. How do I know that this is actually not 0.1+2*pi*(k+1) rad for Combiner I and 6.2 + 2*pi*k rad for combiner R? Well, even with this assumption the phase difference is still 0.1 rad. Thus it seems that it does not matter.
            # What about the next set of combiner traces? They will start getting acquired at a random phase. If for the next trace we get Combiner I phase of 1 rad, then we will see Combiner R phase of 1 - 0.1 = 0.9 rad. In this case the phase difference is 1-0.9 = 0.1, which is the same answer from the one above. Thus the conclusion is that we do not have to worry about this. We just need to make sure that every calculated phase is in the 0 to 2*pi range.

            # Calculate phase difference between the combiners.
            combiner_phase_diff_df = convert_phase_to_2pi_range((combiner_phase_df.loc[slice(None), ('RF Combiner I Digi 2', self.harmonic_name_list, 'Fourier Phase [Rad]')] - combiner_phase_df.loc[slice(None), ('RF Combiner R', self.harmonic_name_list, 'Fourier Phase [Rad]')].values)['RF Combiner I Digi 2'])

            A_df = combiner_phase_diff_df.xs('A', level='Configuration')
            B_df = combiner_phase_diff_df.xs('B', level='Configuration')

            # Cancel out Power detector frequency response.
            # We perform phase subtraction of RF CH A from RF CH B. This way we cancel out the phase shift due to frequency response of the combiners at the offset frequency.

            # We remove the 'Elapsed Time [s]' column, because these times are different for the same averages for A and B RF channel configurations. Notice that I am NOT dividing this by 2 here. The division by 2 should be performed later, when combining data sets from '0' and 'pi' configurations together.
            diff_df = (A_df.reset_index(level=['Elapsed Time [s]'], drop=True) - B_df.reset_index(level=['Elapsed Time [s]'], drop=True)).transform(convert_phase_to_2pi_range)

            # Here we calculate the mean elapsed time between when the same average was taken for different RF channel configurations. This is better than simply taking one of the RF channel configurations as the indicator of the elapsed time, because the time difference between the phase differences is not uniform (RF channel configurations are randomized for each new iteration).

            mean_elapsed_time_df = (A_df.reset_index().set_index(diff_df.index.names)[['Elapsed Time [s]']] + B_df.reset_index().set_index(diff_df.index.names)[['Elapsed Time [s]']])/2

            combiner_phase_diff_no_freq_response_df = diff_df.join(mean_elapsed_time_df).rename(columns={'Elapsed Time [s]': 'Mean Elapsed Time [s]'}).set_index(['Mean Elapsed Time [s]'], append=True)

            # # Calculate the phase shift due to the frequency response.
            # We perform phase addition of RF CH A from RF CH B that gets divided by two after. This way extract the phase shift due to frequency response of the combiners at the offset frequency.

            ##Find the phase shift. Notice the division by 2. This phase shift is the 1/2*(phase_shift_combiner_I + phase_shift_combiner_R) = average phase shift due to frequency response of Combiner I and Combiner R.
            sum_df = (A_df.reset_index(level=['Elapsed Time [s]'], drop=True) + B_df.reset_index(level=['Elapsed Time [s]'], drop=True)).transform(convert_phase_to_2pi_range).transform(divide_and_minimize_phase, div_number=2)

            combiner_phase_diff_freq_response_df = sum_df.join(mean_elapsed_time_df).rename(columns={'Elapsed Time [s]': 'Mean Elapsed Time [s]'}).set_index(['Mean Elapsed Time [s]'], append=True)

            # Changing the scale to mrad from rad.
            combiner_phase_diff_freq_response_df
            combiner_phase_diff_freq_response_df.rename(columns={'Fourier Phase [Rad]': 'RC-Type Phase Shift [mrad]'}, level='Data Field', inplace=True)

            combiner_phase_diff_freq_response_df.loc[slice(None), (slice(None), 'RC-Type Phase Shift [mrad]')] = combiner_phase_diff_freq_response_df.loc[slice(None), (slice(None), 'RC-Type Phase Shift [mrad]')] * 1E3


            # Different waveguide frequencies have different phase difference between the combiners. For typical data set we do not have enough data for each waveguide frequency to see smooth variation of the combiners' phase difference vs time. However, we can look at the deviation of the Combiners' phase difference from its initial phase difference for each RF frequency.
            phase_diff_repeat_1 = combiner_phase_diff_no_freq_response_df.reset_index().set_index('Mean Elapsed Time [s]').sort_index()

            # We first select data that corresponds to the first repeat only. Then we select only first occurences (in time) of each RF frequency.
            phase_diff_initial_df = phase_diff_repeat_1.loc[phase_diff_repeat_1[phase_diff_repeat_1['Repeat'] == 1]['Waveguide Carrier Frequency [MHz]'].sort_index().drop_duplicates(keep='first').sort_index().index]

            # We need to remove all other index levels, except that of the 'Waveguide Carrier Frequency [MHz]', because when subtracting from the list of phase differences, we want to subtract the same phase difference from each phase, independent of other index levels.

            index_level_name_remove_list = remove_sublist(ini_list=combiner_phase_diff_no_freq_response_df.index.names, remove_list=['Waveguide Carrier Frequency [MHz]'])

            phase_diff_initial_df = phase_diff_initial_df.drop(columns=index_level_name_remove_list, level='Data Type').set_index('Waveguide Carrier Frequency [MHz]')

            # We now subtract the initial combiners' phase differences from the rest of the phase differences.

            # The difference is given in mrad. This is also where division by 2 is happening to obtain Phase difference between the combiner unaffected by their individual frequency responses.
            self.combiner_phase_diff_no_freq_response_df = combiner_phase_diff_no_freq_response_df
            self.phase_diff_initial_df = phase_diff_initial_df
            combiner_phase_diff_variation_df = combiner_phase_diff_no_freq_response_df.subtract(phase_diff_initial_df, level='Waveguide Carrier Frequency [MHz]').transform(convert_phase_to_2pi_range).transform(divide_and_minimize_phase, div_number=2) * 1E3

            combiner_phase_diff_variation_df.rename(columns={'Fourier Phase [Rad]': 'Phase Change [mrad]'}, level='Data Field', inplace=True)

            combiner_difference_df = combiner_phase_diff_no_freq_response_df.join(combiner_phase_diff_variation_df)

            combiner_difference_df = pd.concat([combiner_difference_df, combiner_phase_diff_freq_response_df], axis='columns').sort_index(axis='columns')

            self.combiner_difference_df = combiner_difference_df

        return self.combiner_difference_df

    def get_inter_digi_delay_data(self):
        ''' Obtain data for the delay between the digitizers.
        '''

        if self.digi_2_from_digi_1_mean_delay_df is None:
            # Another group with the delay between the Digitizers

            # Calculate the delay between triggering the digitizers. We are calculating Digitizer 2 - Digitizer 1 phase delay. The reason for this is that we expect the phase delay to be positive: Digitizer 1 should trigger faster than Digitizer 2, thus the initial phase of Digitizer 2 should be larger than of Digitizer 1, thus Digitizer 2 - Digitizer 1 has to be positive.

            if self.digitizers_data_df is None:
                self.get_digitizers_data()

            digi_2_from_digi_1_delay_df = self.digitizers_data_df.loc[slice(None), ('RF Combiner I Digi 2', self.harmonic_name_list, ['Fourier Frequency [Hz]', 'Fourier Phase [Rad]'])]['RF Combiner I Digi 2']

            # Shift the phases into proper multiples of 2pi.

            # This is a fast way (by about a factor of 3) of shifting the phases. Here I simply act on the values of the dataframe, without use of grouping or aggregation functions, since in principle I simply need to act with the phase shifting function on each row of the dataset.

            phasor_data_for_delay_df = self.digitizers_data_df.loc[slice(None), (['RF Combiner I Digi 1','RF Combiner I Digi 2'], self.harmonic_name_list, 'Fourier Phase [Rad]')].copy()

            for harmonic in self.harmonic_name_list:
                data_arr = phasor_data_for_delay_df.loc[slice(None), (['RF Combiner I Digi 1','RF Combiner I Digi 2'], harmonic, 'Fourier Phase [Rad]')].values
                phasor_data_for_delay_df.loc[slice(None), (['RF Combiner I Digi 1','RF Combiner I Digi 2'], harmonic, 'Fourier Phase [Rad]')] = np.array(list(map(lambda x: phases_shift(x)[0], list(data_arr))))

            # Calculate the phase difference between the Digitizers at the offset frequency harmonics
            digi_2_from_digi_1_delay_df.loc[slice(None), (self.harmonic_name_list, 'Fourier Phase [Rad]')] = (phasor_data_for_delay_df.loc[slice(None), ('RF Combiner I Digi 2', self.harmonic_name_list, 'Fourier Phase [Rad]')] - phasor_data_for_delay_df.loc[slice(None), ('RF Combiner I Digi 1', self.harmonic_name_list, 'Fourier Phase [Rad]')].values)['RF Combiner I Digi 2']


            # Calculate the Delay in microseconds and samples between the digitizers
            for harmonic_value in self.harmonic_name_list:
                digi_2_from_digi_1_delay_df[harmonic_value,'Delay [Sample]'] = digi_2_from_digi_1_delay_df[harmonic_value, 'Fourier Phase [Rad]'] / (2*np.pi*digi_2_from_digi_1_delay_df[harmonic_value, 'Fourier Frequency [Hz]']) * self.sampling_rate

                digi_2_from_digi_1_delay_df[harmonic_value,'Delay [us]'] = digi_2_from_digi_1_delay_df[harmonic_value, 'Fourier Phase [Rad]'] / (2*np.pi*digi_2_from_digi_1_delay_df[harmonic_value, 'Fourier Frequency [Hz]'])*1E6

            digi_2_from_digi_1_delay_df = digi_2_from_digi_1_delay_df.sort_index(axis='columns')

            self.digi_2_from_digi_1_delay_df = digi_2_from_digi_1_delay_df

            # The averaging is done for both the Delay in digitizer samples and the Delay in microseconds.

            # Here we are using a little bit different method for calculating mean and std. I could simply use pd.DataFrame.aggregate([np.mean, lambda x: np.std(x, ddof=1)/np.sqrt(x.shape[0])]), but this, for whatever reason, takes much more time (about 3 times longer). My suspicion is that in this case the aggregate function produces pd.DataFrame object, whereas is we do the aggregation separately, as done below, the output object is pd.Series. It is natural to assume that operations with pd.Series are faster than with pd.DataFrame objects. It is possible to make this operation even faster, by not using aggragate function alltogether, but instead acting with np.mean and np.std on pd.DataFrame.values arrays. This is what I am doing below

            mean_df = pd.DataFrame(np.mean(digi_2_from_digi_1_delay_df.loc[slice(None), (self.harmonic_name_list, 'Delay [Sample]')], axis='columns')).rename(columns={0: 'Mean Inter Digitizer Delay [Sample]'})

            std_df = pd.DataFrame(np.std(digi_2_from_digi_1_delay_df.loc[slice(None), (self.harmonic_name_list, 'Delay [Sample]')], axis='columns')/np.sqrt(len(self.harmonic_name_list)-1)).rename(columns={0: 'Mean Inter Digitizer Delay STD [Sample]'})

            mean_inter_digi_delay_sample_df = mean_df.join(std_df)

            mean_df = pd.DataFrame(np.mean(digi_2_from_digi_1_delay_df.loc[slice(None), (self.harmonic_name_list, 'Delay [us]')], axis='columns')).rename(columns={0: 'Mean Inter Digitizer Delay [us]'})

            std_df = pd.DataFrame(np.std(digi_2_from_digi_1_delay_df.loc[slice(None), (self.harmonic_name_list, 'Delay [us]')], axis='columns')/np.sqrt(len(self.harmonic_name_list)-1)).rename(columns={0: 'Mean Inter Digitizer Delay STD [us]'})

            mean_inter_digi_delay_time_df = mean_df.join(std_df)

            digi_2_from_digi_1_mean_delay_df = mean_inter_digi_delay_sample_df.join(mean_inter_digi_delay_time_df)

            digi_2_from_digi_1_mean_delay_df.columns.set_names('Data Field', inplace=True)


            self.digi_2_from_digi_1_mean_delay_df = digi_2_from_digi_1_mean_delay_df

        return self.digi_2_from_digi_1_mean_delay_df

    def get_phase_diff_data(self):
        ''' Main FOSOF data frame. Here we calculate Detector phases relative to various phase references and also perform the filtering on the phase difference data, by checking whether there are any very low SNR values at the offset frequency represented by NaN.
        '''

        if self.phase_diff_data_df is None:

            if self.digi_2_from_digi_1_delay_df is None:
                self.get_inter_digi_delay_data()

            phase_diff_data_df = self.digitizers_data_df[['Detector']].rename(columns={'Detector': 'RF Combiner I Reference'},level=0)

            phase_diff_data_df = phase_diff_data_df.join(self.digitizers_data_df[['Detector']].rename(columns={'Detector': 'RF Combiner R Reference'},level=0))

            # Trace Filename is not needed here.
            phase_diff_data_df.drop('Trace Filename', axis='columns', level=2, inplace=True)

            # Calculate Detector in RF Combiner I and RF Combiner R phase differences.
            phase_diff_data_df.loc[slice(None), ('RF Combiner I Reference', self.harmonic_name_list, 'Fourier Phase [Rad]')] = (self.digitizers_data_df.loc[slice(None), ('Detector', self.harmonic_name_list, 'Fourier Phase [Rad]')] - self.digitizers_data_df.loc[slice(None), ('RF Combiner I Digi 1', self.harmonic_name_list, 'Fourier Phase [Rad]')].values).rename(columns={'Detector': 'RF Combiner I Reference'}).transform(lambda x: convert_phase_to_2pi_range(x))

            phase_diff_data_df.loc[slice(None), ('RF Combiner R Reference', self.harmonic_name_list, 'Fourier Phase [Rad]')] = (
            self.digitizers_data_df.loc[slice(None), ('Detector', self.harmonic_name_list, 'Fourier Phase [Rad]')] - \
            self.digitizers_data_df.loc[slice(None), ('RF Combiner R', self.harmonic_name_list, 'Fourier Phase [Rad]')].values +\
            self.digi_2_from_digi_1_delay_df.loc[slice(None), (self.harmonic_name_list, 'Fourier Phase [Rad]')].values
            ).rename(columns={'Detector': 'RF Combiner R Reference'}).transform(lambda x: convert_phase_to_2pi_range(x))

            # We dropped the "Trace Filename" column, however, for some reason it is still contained in the index of columns. This function gets rid of it.
            phase_diff_data_df.columns = phase_diff_data_df.columns.remove_unused_levels()

            ''' We now look for any Detector traces with undefined = very low SNR at the offset frequency, and then determines the action depending on the presence of any previously detected RF power measurement outliers for both the RF system A and B. If no issues are found, then no actions are taken.
            '''
            snr_null_df = phase_diff_data_df.loc[phase_diff_data_df.loc[slice(None), ('RF Combiner I Reference', 'First Harmonic', 'SNR')].isnull()]

            rf_pow_df, rf_system_power_outlier_df = self.get_rf_sys_pwr_det_data()

            if snr_null_df.shape[0] > 0:

                warning_mssg = 'WARNING!!! Traces with unacceptably low SNR are present. Indeces corresponding to these outlying readings:\n'+str(snr_null_df.index.values) + '\n'

                rf_pwr_outlier_and_snr_null_diff_index = rf_system_power_outlier_df.index.difference(snr_null_df.index)

                if rf_pwr_outlier_and_snr_null_diff_index.shape[0] == 0:

                    if rf_system_power_outlier_df.index.shape[0] == snr_null_df.index.shape[0]:

                        warning_mssg = warning_mssg + 'These traces match with the traces that have outlying RF power detector readings. Therefore the very low SNR can be explained by the RF generator not supplying the requested amount of power. These indeces are removed from the dataframe containing phase differences.'

                        phase_diff_data_df = phase_diff_data_df.drop(rf_system_power_outlier_df.index)
                        self.phase_diff_data_filtered_Q = True

                    else:

                        if rf_system_power_outlier_df.index.shape[0] == 0:
                            warning_mssg = warning_mssg + '\nPossibly for some traces the noise in the signal was too large during the data acqusition. There is no compelling reason to remove these indeces from the dataframe containing phase differences, except that Phase Averaging will not work properly.'

                        else:
                            warning_mssg = warning_mssg + 'There seems to be something wrong with the data acquisition. For some traces we might have an unlikely combination of large signal noise, RF generator not outputting correct power, or Keithley multimeter not functioning well. One needs to closely examine the data set.'

                else:
                    warning_mssg = warning_mssg + 'There seems to be something wrong with the data acquisition. For some traces we might have an unlikely combination of large signal noise, RF generator not outputting correct power, or Keithley multimeter not functioning well. One needs to closely examine the data set.'

                self.err_warn_df = self.err_warn_df.append(pd.Series({'Message': warning_mssg}, name='Low SNR'))

                print(warning_mssg)

            else:
                if rf_system_power_outlier_df.index.shape[0] > 0:
                    warning_mssg = "The SNR is high enough for all of the traces, but there are some outliers in the RF power detectors' readings. Most likely there was some issue with the RF power measuring equipment: multimeter, RF power detectors, interconnections, etc."

                    self.err_warn_df = self.err_warn_df.append(pd.Series({'Message': warning_mssg}, name='Good SNR with RF power measurement outliers.'))

                    print(warning_mssg)

            self.phase_diff_data_df = phase_diff_data_df

        return self.phase_diff_data_df

    def average_av_sets(self):
        ''' Phases containing FOSOF phase are averaged using various methods for every averaging set.
        '''

        if self.phase_av_set_averaged_df is None:

            if self.phase_diff_data_df is None:
                self.get_phase_diff_data()

            # List of columns by which to group the phase data for averaging set averaging
            averaging_set_grouping_list = remove_sublist(ini_list=self.general_index.names, remove_list=['Average', 'Elapsed Time [s]'])

            phase_diff_group = self.phase_diff_data_df.groupby(averaging_set_grouping_list)
            # For the subsequent analysis we assume that the phases of the phasors are normally distributed (for the Phase Averaging method), as well as A*cos(phi) and A*sin(phi) of the phasors - its x and y components, where A = amplitude of the given phasor and phi is its phase (for the Phasor Averaging and Phasor Averaging Relative To DC ). We also assume that the average amplitude of the phasors relative to DC are normally distributed (For Phasor Averaging Relative To DC, but not for calculation of the phase, but for estimation of FOSOF relative amplitude, when averaging amplitude relative to dc obtained from each averaging set for the given Waveguide carrier frequency).
            # These assumptions is how we can use the formulas for the best estimate of the mean and standard deviation (with N-1 factor instead of N). If this is not true, then we might need to use different type of analysis.
            # Notice that it seems that NONE of these quantities exactly normally distributed. But it seems to be intuitive, that if the signal is not too weak, then the errors have to be small and in that case the quantities are approximately normally distributed.

            # Shift all of the phase subsets, corresponding to their respective averaging set, in proper quadrants.

            phase_shifted_df = self.phase_diff_data_df.loc[slice(None), (slice(None),slice(None), 'Fourier Phase [Rad]')].groupby(averaging_set_grouping_list).transform(lambda x: phases_shift(x)[0])

            #phase_shifted_df = group_apply_transform(self.phase_diff_data_df.loc[slice(None), (slice(None),slice(None), 'Fourier Phase [Rad]')].groupby(averaging_set_grouping_list), lambda x: phases_shift(x)[0])

            phase_diff_shifted_data_df = self.phase_diff_data_df.copy()
            phase_diff_shifted_data_df.loc[slice(None), (slice(None), slice(None), 'Fourier Phase [Rad]')] = phase_shifted_df

            # Different types of averaging are performed below
            # The methods presented are not perfect: the Lyman-alpha detector frequency response at the offset frequency seems to have some DC dependence and thus the average phase changes with DC, thus making it quite impossible to perform any averaging if we have large DC changes (what is large? I do not know) between traces of the averaging set. Thus all of the methods here assume that this issue does not exist.

            # Phase averaging
            # Here we simply average the phases together. Amplitudes are not used. We can, however, calculate average Amplitude-to-DC ratio. This, when SNR is high, should, in principle, give better results, compared to phasor averaging, when DC is not stable during the averaging set, but changes from one trace to another.
            def phase_angle_range(x):
                return (x.max()-x.min())*180/np.pi

            def phase_av(df):
                df.columns = df.columns.droplevel(['Source', 'Data Type'])
                df_grouped = df.groupby(averaging_set_grouping_list)

                phases_averaged_df = df_grouped.aggregate({np.mean, np.std, lambda x: x.shape[0], phase_angle_range})

                phases_averaged_df.columns = phases_averaged_df.columns.droplevel(0)
                return phases_averaged_df


            phase_av_df = phase_diff_shifted_data_df.loc[slice(None), (slice(None), slice(None), 'Fourier Phase [Rad]')].groupby(level=['Source', 'Data Type'], axis='columns').apply(phase_av)

            phase_av_df.rename(columns={'mean': 'Phase [Rad]', 'std': 'Phase STD [Rad]', '<lambda>': 'Number Of Averaged Data Points', 'phase_angle_range': 'Range Of Phases [Deg]'}, inplace=True)

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

            phasor_av_df.rename(columns={'Number Of Averaged Phasors': 'Number Of Averaged Data Points'}, inplace=True)

            # Adding SNR of the average phasor. This is just an estimate. We assume that the true SNR is the same for each phasor in the averaging set. Thus, we can calculate the mean of the SNR. However, averaging N phasors results in the SNR going up by a factor of N**0.5. The reason for this is that the resulting phasor has, real and imaginary components as the mean of the set {A_n_i * cos(phi_n_i), A_n_i * sin(phi_n_i)}, where phi_n_i is random phase, and A_n_i is the noise amplitude. In this case the amplitude of the resulting averaged phasor is smaller by a factor of N**0.5, assuming that A_n_i is a constant. I have tested this in Mathematica and it seems indeed to be the case, except that it seems to go down as about 1.1*N**0.5.

            # Calculate SNR of the averaging set.
            def get_av_set_SNR(x):
                ''' Calculate SNR of the averaging set.
                '''
                # We are testing if there are any SNR values that are NaN. In this case it is assumed that the averaging set has large noise (or the signal of interest is not there)
                if x[x.isnull()].size == 0:
                    mean_SNR_val = x.mean() * np.sqrt(x.size)
                else:
                    mean_SNR_val = np.nan
                return mean_SNR_val

            av_set_SNR_df = phase_diff_shifted_data_df.loc[slice(None), (slice(None), slice(None), 'SNR')].groupby(averaging_set_grouping_list).aggregate(get_av_set_SNR)
            # Combine the averaging set SNR with the averaged phasor data
            phasor_av_df = phasor_av_df.join(av_set_SNR_df).sort_index(axis='columns')


            # Phasor averaging relative to DC signal level.
            # Same as phasor averaging, however instead of the phasor amplitudes we use amplitude-to-DC ratios for individual phasors. The reason for doing this is to eliminate the possibility of DC changing between traces for the given averaging set and skewing our average, because it would mean that the amplitude of the average phasor is not the same for all the phasors of the averaging set\. Now we should be insensitive to this, assuming that the mean phase (true phase) does not depend on DC. We still have the issue that the signal can be more or less noisy for different DC levels, therefore changing the standard deviation from trace to trace, hence making our formula for the best estimate of mean and standard deviation not entirely correct. (We assume Gaussian distribution here).

            phase_rel_to_dc_df = phase_diff_shifted_data_df.loc[slice(None), (slice(None), slice(None), ['Fourier Phase [Rad]','Fourier Amplitude [V]'])].copy()

            # Normalize to DC values
            phase_rel_to_dc_df.loc[slice(None), (slice(None), slice(None), 'Fourier Amplitude [V]')] = phase_rel_to_dc_df.loc[slice(None), (slice(None), slice(None), 'Fourier Amplitude [V]')].transform(lambda x: x/phase_diff_shifted_data_df.loc[slice(None), ('RF Combiner I Reference', 'Other', 'DC [V]')].values)


            phasor_rel_to_dc_av_df = phase_rel_to_dc_df.loc[slice(None), (slice(None), slice(None), ['Fourier Phase [Rad]','Fourier Amplitude [V]'])].groupby(level=['Source', 'Data Type'], axis='columns').apply(phasor_av)

            phasor_rel_to_dc_av_df.rename(columns={'Number Of Averaged Phasors': 'Number Of Averaged Data Points', 'Amplitude': 'Amplitude Relative To DC', 'Amplitude STD': 'Amplitude Relative To DC STD'}, inplace=True)

            phase_av_set_df = pd.concat([phase_av_df, phasor_av_df, phasor_rel_to_dc_av_df], axis='columns', keys=['Phase Averaging', 'Phasor Averaging', 'Phasor Averaging Relative To DC'], names=['Averaging Type'])

            phase_av_set_df = phase_av_set_df.swaplevel(axis='columns', i='Averaging Type', j='Source').swaplevel(axis='columns', i='Averaging Type', j='Data Type')

            # Change the names of the index levels. We basically just want to add the 'Data Field' level name (instead of None)
            phase_av_set_df.columns.names = ['Source', 'Data Type', 'Averaging Type', 'Data Field']

            # We assume that for the given repeat, offset frequency, B Field, and pre-910 state all of the phasors obtained should have the same average SNR and thus the same true standard deviation for phase and amplitude at the given offset frequency. That is why we calculate the RMS quantities below. Now, this assumption might be wrong, especially when we have the beam that deteriorates over time or when the beam is such that it abruptly changes its mode of operation (we see it sometimes). When we have scan through offset frequencies and other additional parameters, then it takes even more time to acquire single repeat, thus the chance for the standard deviation to change is even larger. The additional assumption is that the RF scan range is small enough for there to be no appreciable variation in SNR with the RF frequency.

            rms_repeat_grouping_list = remove_sublist(ini_list=self.general_index.names, remove_list=['Elapsed Time [s]', 'Waveguide Carrier Frequency [MHz]', 'Average', 'Configuration'])

            phase_av_set_df_index_names_list = list(phase_av_set_df.index.names)

            phase_av_set_group = phase_av_set_df.groupby(rms_repeat_grouping_list)

            data_rms_std_df = phase_av_set_group.apply(lambda df: pd.Series(pooled_std(df.loc[slice(None), (slice(None), slice(None), slice(None), 'Phase STD [Rad]')].values
            , df.loc[slice(None), (slice(None), slice(None), slice(None), 'Number Of Averaged Data Points')].values), index=df.loc[slice(None), (slice(None), slice(None), slice(None), 'Phase STD [Rad]')].columns))

            data_rms_std_df.rename(columns={'Phase STD [Rad]': 'Phase RMS Repeat STD [Rad]'}, level='Data Field', inplace=True)

            # We have to make sure that the data frames we are trying to join have the same index columns
            phase_av_set_next_df = phase_av_set_df.reset_index().set_index(rms_repeat_grouping_list).join(data_rms_std_df, how='inner').sort_index(axis='columns').reset_index().set_index(phase_av_set_df_index_names_list)

            phase_av_set_next_df = phase_av_set_next_df.sort_index(axis='columns')
            phase_av_set_next_df.columns = phase_av_set_next_df.columns.remove_unused_levels()

            # Another way to calculate the STD of the phases is to assume that the standard deviation is the same for the data for the same averaging set, which includes phases obtained for the same B field, pre-quench 910 state, repeat and waveguide carrier frequency. The configuration can be either A and B, of course.

            # Note that later on, when calculating A-B phases, the standard deviation that we get from combining this types of STD and simply STD determining for each configuration and averaging set, are exactly the same. Thus in a sense we are not getting any advantage of performing this type of calculation.
            rms_av_set_grouping_list = remove_sublist(ini_list=self.general_index.names, remove_list=['Elapsed Time [s]', 'Average', 'Configuration'])

            phase_av_set_group = phase_av_set_next_df.groupby(rms_av_set_grouping_list)

            data_rms_std_df = phase_av_set_group.apply(
                        lambda df: pd.Series(pooled_std(df.loc[slice(None), (slice(None), slice(None), slice(None), 'Phase STD [Rad]')].values
                        , df.loc[slice(None), (slice(None), slice(None), slice(None), 'Number Of Averaged Data Points')].values), index=df.loc[slice(None), (slice(None), slice(None), slice(None), 'Phase STD [Rad]')].columns)
                        )

            data_rms_std_df.rename(columns={'Phase STD [Rad]': 'Phase RMS Averaging Set STD [Rad]'}, level='Data Field', inplace=True)

            phase_av_set_next_df = phase_av_set_next_df.reset_index().set_index(rms_av_set_grouping_list).join(data_rms_std_df, how='inner').reset_index().set_index(phase_av_set_df_index_names_list).sort_index(axis='columns')

            phase_av_set_next_df.columns = phase_av_set_next_df.columns.remove_unused_levels()


            # These calculations (determination of RMS STD) are also performed for the Amplitude Relative To DC. Here we expect that this quantity should have uncertainty independent of DC level, thus it makes sense to use it as relative FOSOF amplitude.

            # RMS Repeat STD
            phase_av_set_group = phase_av_set_next_df.groupby(rms_repeat_grouping_list)

            data_rms_std_df = phase_av_set_group.apply(lambda df: pd.Series(pooled_std(df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC STD')].values
            , df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC', 'Number Of Averaged Data Points')].values), index=df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC','Amplitude Relative To DC STD')].columns))

            data_rms_std_df.rename(columns={'Amplitude Relative To DC STD': 'Amplitude Relative To DC RMS Repeat STD'}, level='Data Field', inplace=True)

            phase_av_set_next_df = phase_av_set_next_df.reset_index().set_index(rms_repeat_grouping_list).join(data_rms_std_df, how='inner').reset_index().set_index(phase_av_set_df_index_names_list).sort_index(axis='columns')

            # RMS Averaging Set STD
            phase_av_set_group = phase_av_set_df.groupby(rms_av_set_grouping_list)

            data_rms_std_df = phase_av_set_group.apply(lambda df: pd.Series(pooled_std(df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC STD')].values
            , df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC', 'Number Of Averaged Data Points')].values), index=df.loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC','Amplitude Relative To DC STD')].columns)
                        )

            data_rms_std_df.rename(columns={'Amplitude Relative To DC STD': 'Amplitude Relative To DC RMS Averaging Set STD'}, level='Data Field', inplace=True)

            phase_av_set_next_df = phase_av_set_next_df.reset_index().set_index(rms_av_set_grouping_list).join(data_rms_std_df, how='inner').reset_index().set_index(phase_av_set_df_index_names_list).sort_index(axis='columns')

            phase_av_set_next_df.columns = phase_av_set_next_df.columns.remove_unused_levels()

            # Calculate phase STD of the mean.

            column_sigma_list = [
                    'Phase RMS Repeat STD [Rad]',
                    #'Phase RMS STD With Covariance [Rad]',
                    'Phase STD [Rad]',
                    'Phase RMS Averaging Set STD [Rad]']#,
                    #'Phase STD With Covariance [Rad]']

            column_mean_sigma_list = [
                    'Phase RMS Repeat STDOM [Rad]',
                    #'Phase Mean RMS STD With Covariance [Rad]',
                    'Phase STDOM [Rad]',
                    'Phase RMS Averaging Set STDOM [Rad]']#,
                    #'Phase Mean STD With Covariance [Rad]']

            # List of STD's for Phasor Averaging Relative to DC method. We include Ampltude To DC STD's here.
            column_sigma_rel_to_dc_list = [
                    'Phase RMS Repeat STD [Rad]',
                    #'Phase RMS STD With Covariance [Rad]',
                    'Phase STD [Rad]',
                    'Phase RMS Averaging Set STD [Rad]',
                    'Amplitude Relative To DC RMS Repeat STD',
                    #'Phase RMS STD With Covariance [Rad]',
                    'Amplitude Relative To DC STD',
                    'Amplitude Relative To DC RMS Averaging Set STD']#,
                    #'Phase STD With Covariance [Rad]']

            column_mean_sigma_rel_to_dc_list = [
                    'Phase RMS Repeat STDOM [Rad]',
                    #'Phase Mean RMS STD With Covariance [Rad]',
                    'Phase STDOM [Rad]',
                    'Phase RMS Averaging Set STDOM [Rad]',
                    'Amplitude Relative To DC RMS Repeat STDOM',
                    #'Phase Mean RMS STD With Covariance [Rad]',
                    'Amplitude Relative To DC STDOM',
                    'Amplitude Relative To DC RMS Averaging Set STDOM']#,
                    #'Phase Mean STD With Covariance [Rad]']

            averaging_type_list = ['Phase Averaging', 'Phasor Averaging', 'Phasor Averaging Relative To DC']

            n_averages_col_name = 'Number Of Averaged Data Points'

            for reference_type in phase_av_set_next_df.columns.levels[0].values:
                for harmonic_value in self.harmonic_name_list:
                    for averaging_type in averaging_type_list:

                        if averaging_type == 'Phasor Averaging Relative To DC':
                            column_sigma_to_use_list = column_sigma_rel_to_dc_list
                            column_mean_sigma_to_use_list = column_mean_sigma_rel_to_dc_list
                        else:
                            column_sigma_to_use_list = column_sigma_list
                            column_mean_sigma_to_use_list = column_mean_sigma_list

                        for column_sigma_index in range(len(column_sigma_to_use_list)):
                            phase_av_set_next_df[reference_type, harmonic_value, averaging_type, column_mean_sigma_to_use_list[column_sigma_index]] = phase_av_set_next_df[reference_type, harmonic_value, averaging_type, column_sigma_to_use_list[column_sigma_index]] / np.sqrt(phase_av_set_next_df[reference_type, harmonic_value, averaging_type, n_averages_col_name])

            phase_av_set_next_df = phase_av_set_next_df.sort_index(axis='columns')

            # Rename the levels representing RF Combiner as the phase reference and  Fourier harmonics
            phase_av_set_next_df.columns.rename(names='Phase Reference Type', level='Source', inplace=True)
            phase_av_set_next_df.columns.rename(names='Fourier Harmonic', level='Data Type', inplace=True)

            self.phase_av_set_averaged_df = phase_av_set_next_df
        return self.phase_av_set_averaged_df

    def cancel_out_freq_response(self):
        ''' Subtracts phases obtained for RF Systems A and B (we perform RF System A phase - RF System B phase) to eliminate the frequency response of the detection system. Also, as a bonus, we can calculate the phase shift due to the frequency response of the detection system at the offset frequency. This phase shift is simply 1/2*(RF System A phase + RF System B phase).

        The detection system is comprised of Detector + Transimpedance amplifier + RF power detectors for the combiners + cables. Note that we are not dividing the RF System A and B phase difference by 2 in this function.

        Outputs:
        :phase_A_minus_B_df: pd.DataFrame of 2 x FOSOF phases with frequency response at the offset frequency cancelled out.
        :phase_freq_response_df: pd.DataFrame of phase shift due to the frequency response at the offset frequency.
        '''

        if self.phase_A_minus_B_df is None:

            if self.phase_av_set_averaged_df is None:
                self.average_av_sets()

            # We now want to eliminate the frequency response of the detection system at the offset frequency and its harmonics.

            def eliminate_freq_response(df, columns_phase_std_dict, column_phase):
                '''Perform phase subtraction of RF CH A from RF CH B. The result is 2*(phi_FOSOF + phi_RF) with the phase shift due to frequency response of the detection system at the offset frequency eliminated.

                Analysis is performed for the specified types of averaging set averaging.
                '''
                A_df = df.xs('A', level='Configuration')
                B_df = df.xs('B', level='Configuration')

                phase_diff_df = (A_df.loc[slice(None),(slice(None), slice(None), slice(None), column_phase)] - B_df.loc[slice(None), (slice(None), slice(None), slice(None), column_phase)]).transform(convert_phase_to_2pi_range)

                std_df = np.sqrt(A_df.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2 + B_df.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2)

                std_df = std_df.rename(columns=columns_phase_std_dict, level='Data Field')

                return phase_diff_df.join(std_df).sort_index(axis='columns')

            def calculate_freq_response(df, columns_phase_std_dict, column_phase):
                '''Perform phase addition of RF CH A from RF CH B. The result is (phi_FOSOF + phi_RF)/2 = delta_phi, which is the phase shift due to the frequency response of the detection system at the offset frequency.

                Analysis is performed for the specified types of averaging set averaging.
                '''
                A_df = df.xs('A', level='Configuration')
                B_df = df.xs('B', level='Configuration')

                phase_sum_df = (A_df.loc[slice(None),(slice(None), slice(None), slice(None), column_phase)] + B_df.loc[slice(None), (slice(None), slice(None), slice(None), column_phase)]).transform(convert_phase_to_2pi_range).transform(divide_and_minimize_phase, div_number=2)

                std_df = np.sqrt(A_df.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2 + B_df.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2)/2

                std_df = std_df.rename(columns=columns_phase_std_dict, level='Data Field')

                return phase_sum_df.join(std_df).sort_index(axis='columns')

            columns_phase_std_dict = {
                    'Phase STDOM [Rad]': 'Phase STD [Rad]',
                    'Phase RMS Repeat STDOM [Rad]': 'Phase RMS Repeat STD [Rad]',
                    'Phase RMS Averaging Set STDOM [Rad]': 'Phase RMS Averaging Set STD [Rad]'}
            column_phase = 'Phase [Rad]'

            rms_no_config_grouping_list = remove_sublist(ini_list=self.phase_av_set_averaged_df.index.names, remove_list=['Waveguide Carrier Frequency [MHz]', 'Configuration'])
            rms_no_config_grouping_list.insert(0, 'Waveguide Carrier Frequency [MHz]')

            # Perform RF CH A and RF CH B phase subtraction to get rid of the detection system frequency response at the offset frequency.
            # We also reverse the order of some of the indeces for convenience.
            phase_A_minus_B_df = eliminate_freq_response(self.phase_av_set_averaged_df, columns_phase_std_dict, column_phase).reset_index().set_index(rms_no_config_grouping_list).sort_index()

            phase_freq_response_df = calculate_freq_response(self.phase_av_set_averaged_df, columns_phase_std_dict, column_phase).reset_index().set_index(rms_no_config_grouping_list).sort_index()

            self.phase_A_minus_B_df = phase_A_minus_B_df
            self.phase_freq_response_df = phase_freq_response_df

        return self.phase_A_minus_B_df, self.phase_freq_response_df

    def average_data_field(self, df, reference_type_list, harmonics_names_list, averaging_type_list, data_column, columns_dict, data_is_phases=False):
        ''' Perform averaging of the given data field.

        The averaging of data field is performed by performing a weighted least-squares straight line (slope = 0) fit to the data. This is equivalent to weighted average. We also extract Reduced Chi-Squared that tells us if our estimates for standard deviation are reasonable.

        It is assumed that the data set has the following structure: first level of columns is the reference_type_list = phase reference type for FOSOF. Second level consists of the Fourier Harmonics used, specified in the harmonics_names_list. Third level is the list of different types of averaging that was performed on the data frame before, specified in the averaging_type_list. The variable data_column is the column name of the variable that needs to get averaged. The columns_dict is the dictionary of what standard deviations to use from 4th level. Key = how this type of standard deviation will be named in the output series, value = current name of that standard deviation type in the data frame. If we are trying to average phases, then they need to be properly shifted: set the data_is_phases boolean to True.
        '''

        average_reference_type_list = []
        for reference_type in reference_type_list:
            reference_data_df = df[reference_type]

            average_harmonic_list = []
            for harmonic_name in harmonics_names_list:
                harmonic_data_df = reference_data_df[harmonic_name]

                averaging_type_data_list = []
                for averaging_type in averaging_type_list:
                    data_df = harmonic_data_df[averaging_type]
                    std_type_list = []
                    if data_is_phases:
                        data_arr = phases_shift(data_df[data_column])[0]
                    else:
                        data_arr = data_df[data_column]

                    for std_output_type, std_input_type in columns_dict.items():

                        std_arr = data_df[std_input_type]
                        av_s = straight_line_fit_params(data_arr, std_arr)
                        #av_s.rename({'Weighted Mean':'RF CH A - RF CH B Weighted Averaged Phase [Rad]','Weighted STD':'RF CH A - RF CH B Weighted Averaged Phase STD [Rad]'}, inplace=True)
                        av_s.index.name = 'Data Field'

                        av_s = pd.concat([av_s], keys=[std_output_type], names=['STD Type'])
                        std_type_list.append(av_s)

                    averaging_type_data_list.append(pd.concat([pd.concat(std_type_list)], keys=[averaging_type], names=['Averaging Type']))

                average_harmonic_list.append(pd.concat([pd.concat(averaging_type_data_list)], keys=[harmonic_name], names=['Fourier Harmonic']))

            average_reference_type_list.append(pd.concat([pd.concat(average_harmonic_list)], keys=[reference_type], names=['Phase Reference Type']))

        average_reference_type_s = pd.concat(average_reference_type_list)

        return average_reference_type_s


    def average_FOSOF_over_repeats(self):
        ''' Average FOSOF phases and amplitudes accross all of the repeats

            Outputs:
            :fosof_ampl_df: pd.DataFrame of values of Amplitude-to-DC ratio averaged across all of the repeats, where both 'A' and 'B' configurations are taken into account
            :fosof_phases_df: pd.DataFrame of values of RF system A - RF system B phases averaged across all of the repeats.
        '''

        if self.fosof_phases_df is None:

            if self.phase_A_minus_B_df is None:
                self.cancel_out_freq_response()

            # Perform averaging of phases across all of the Repeats for every RF Carrier Frequency.
            # First we group the data frame by the carrier frequency to collect the phases from all of the repeats together for every carrier frequency.
            phase_A_minus_B_df = copy.copy(self.phase_A_minus_B_df)
            rms_av_over_repeat_grouping_list = list(phase_A_minus_B_df.index.names)
            rms_av_over_repeat_grouping_list.remove('Repeat')

            rms_final_averaged_phasor_data_column_list = copy.copy(rms_av_over_repeat_grouping_list)

            rms_final_averaged_phasor_data_column_list.remove('Waveguide Carrier Frequency [MHz]')
            rms_final_averaged_phasor_data_column_list.append('Waveguide Carrier Frequency [MHz]')

            phase_A_minus_B_df.columns = phase_A_minus_B_df.columns.remove_unused_levels()

            # Shift the phases in a proper quadrant
            phase_A_minus_B_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Phase [Rad]')] = phase_A_minus_B_df.loc[slice(None), (slice(None), slice(None), slice(None), 'Phase [Rad]')].groupby(rms_av_over_repeat_grouping_list).transform(lambda x: phases_shift(x)[0])

            # Average phases for each type of averaging over the required groups of indeces
            def find_av_phase(x, col_list):
                ''' Aggregating function that is used for 'group_agg_mult_col_dict' function.

                The function must have the following line it it:
                data_dict = get_column_data_dict(x, col_list)

                This basically creates a dictionary of values that are used as the input to the function.

                Inputs:
                :x: data columns (from pandas)
                :col_list: list of columns used for combining the data into a single column.
                '''
                data_dict = get_column_data_dict(x, col_list)

                std_col = list(set(data_dict.keys()) - {'Phase [Rad]'})[0]

                return dict(straight_line_fit_params(data_arr=data_dict['Phase [Rad]'], sigma_arr=data_dict[std_col]))

            def phase_av_std_type(df, std_type_col):
                col_list = ['Phase [Rad]', std_type_col]
                df.columns = df.columns.droplevel(['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type'])
                return group_agg_mult_col_dict(df[col_list], col_list, index_list=rms_av_over_repeat_grouping_list, func=find_av_phase)

            # Level name for different types of averaging in final dataframe of averaged phases over repeats
            std_type_level_name = 'STD Type'

            # Phase data used for phase averaging
            phase_data_grouped_df = phase_A_minus_B_df.groupby(axis='columns', level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type'])

            # Phase averaging with using uncertainties from simple averaging of averaging sets
            std_type_col = 'Phase STD [Rad]'
            std_type_level = 'Phase STD'

            phase_std_df = phase_data_grouped_df.apply(lambda df: phase_av_std_type(df, std_type_col))

            phase_std_df = pd.concat([phase_std_df], names=[std_type_level_name], keys=[std_type_level], axis='columns').reorder_levels(axis='columns', order=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', None])

            # Phase averaging with using uncertainties from RMS uncertainty for each averaging set
            std_type_col = 'Phase RMS Averaging Set STD [Rad]'
            std_type_level = 'Phase RMS Averaging Set STD'
            phase_av_set_std_df = phase_data_grouped_df.apply(lambda df: phase_av_std_type(df, std_type_col))

            phase_av_set_std_df = pd.concat([phase_av_set_std_df], names=[std_type_level_name], keys=[std_type_level], axis='columns').reorder_levels(axis='columns', order=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', None])

            # Phase averaging with using uncertainties from RMS uncertainty for each repeat
            std_type_col = 'Phase RMS Repeat STD [Rad]'
            std_type_level = 'Phase RMS Repeat STD'
            phase_repeat_std_df = phase_data_grouped_df.apply(lambda df: phase_av_std_type(df, std_type_col))

            phase_repeat_std_df = pd.concat([phase_repeat_std_df], names=[std_type_level_name], keys=[std_type_level], axis='columns').reorder_levels(axis='columns', order=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', None])

            fosof_phases_df = phase_std_df.join([phase_av_set_std_df, phase_repeat_std_df]).reset_index().set_index(rms_final_averaged_phasor_data_column_list).sort_index().sort_index(axis='columns')

            fosof_phases_df.columns.names = ['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'Data Field']

            # We now want to average set of amplitude relative to dc values obtained for every averaging set (both 'A' and 'B' configurations) for all repeats for given RF frequency.
            rms_no_config_grouping_list = remove_sublist(ini_list=self.phase_av_set_averaged_df.index.names, remove_list=['Waveguide Carrier Frequency [MHz]', 'Configuration'])

            rms_no_config_grouping_list.insert(0, 'Waveguide Carrier Frequency [MHz]')

            ampl_av_set_df = self.phase_av_set_averaged_df.reset_index().set_index(rms_no_config_grouping_list).sort_index(axis='index').loc[slice(None), (slice(None), slice(None), 'Phasor Averaging Relative To DC')]
            ampl_av_set_df.columns = ampl_av_set_df.columns.remove_unused_levels()

            # Average amplitude-to-dc ratios for each type of averaging over the required groups of indeces
            def find_av_ampl(x, col_list):
                ''' Aggregating function that is used for 'group_agg_mult_col_dict' function.

                The function must have the following line it it:
                data_dict = get_column_data_dict(x, col_list)

                This basically creates a dictionary of values that are used as the input to the function.

                Inputs:
                :x: data columns (from pandas)
                :col_list: list of columns used for combining the data into a single column.
                '''
                data_dict = get_column_data_dict(x, col_list)

                std_col = list(set(data_dict.keys()) - {'Amplitude Relative To DC'})[0]

                return dict(straight_line_fit_params(data_arr=data_dict['Amplitude Relative To DC'], sigma_arr=data_dict[std_col]))

            def ampl_av_std_type(df, std_type_col):
                col_list = ['Amplitude Relative To DC', std_type_col]
                df.columns = df.columns.droplevel(['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type'])
                return group_agg_mult_col_dict(df[col_list], col_list, index_list=rms_av_over_repeat_grouping_list, func=find_av_ampl)

            # Level name for different types of averaging in final dataframe of averaged amplitude-to-dc ratios over repeats
            std_type_level_name = 'STD Type'

            # Amplitude-to-dc ratio data used for phase averaging
            ampl_data_grouped_df = ampl_av_set_df.groupby(axis='columns', level=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type'])

            # Amplitude-to-dc ratio averaging with using uncertainties from simple averaging of averaging sets
            std_type_col = 'Amplitude Relative To DC STDOM'
            std_type_level = 'Amplitude Relative To DC STD'

            ampl_std_df = ampl_data_grouped_df.apply(lambda df: ampl_av_std_type(df, std_type_col))

            ampl_std_df = pd.concat([ampl_std_df], names=[std_type_level_name], keys=[std_type_level], axis='columns').reorder_levels(axis='columns', order=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', None])

            # Amplitude-to-dc ratio averaging with using uncertainties from RMS uncertainty for each averaging set
            std_type_col = 'Amplitude Relative To DC RMS Averaging Set STDOM'
            std_type_level = 'Amplitude Relative To DC RMS Averaging Set STD'
            ampl_av_set_std_df = ampl_data_grouped_df.apply(lambda df: ampl_av_std_type(df, std_type_col))

            ampl_av_set_std_df = pd.concat([ampl_av_set_std_df], names=[std_type_level_name], keys=[std_type_level], axis='columns').reorder_levels(axis='columns', order=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', None])

            # Amplitude-to-dc ratio averaging with using uncertainties from RMS uncertainty for each repeat
            std_type_col = 'Amplitude Relative To DC RMS Repeat STDOM'
            std_type_level = 'Amplitude Relative To DC RMS Repeat STD'
            ampl_repeat_std_df = ampl_data_grouped_df.apply(lambda df: ampl_av_std_type(df, std_type_col))

            ampl_repeat_std_df = pd.concat([ampl_repeat_std_df], names=[std_type_level_name], keys=[std_type_level], axis='columns').reorder_levels(axis='columns', order=['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', None])

            fosof_ampl_df = ampl_std_df.join([ampl_av_set_std_df, ampl_repeat_std_df]).reset_index().set_index(rms_final_averaged_phasor_data_column_list).sort_index().sort_index(axis='columns')

            fosof_ampl_df.columns.names = ['Phase Reference Type', 'Fourier Harmonic', 'Averaging Type', 'STD Type', 'Data Field']

            self.fosof_phases_df = fosof_phases_df
            self.fosof_ampl_df = fosof_ampl_df

        return self.fosof_ampl_df, self.fosof_phases_df

    def analyze_pre_910_switching(self):
        ''' Averaging phase data for pre-quenching 910 cavity toggling ON and OFF.

        For B field scan and pre-quench 910 ON/OFF data set we are not taking 0 and pi configurations, since we are not interested in the absolute resonant frequencies, but their difference for different values of B field and pre-quench 910 state.
        Outputs:
        :pre_910_states_averaged_df: pd.DataFrame for pre-quench 910 ON phase - pre-quench 910 OFF phase for every RF frequency
        :pre_910_av_difference_df: pd.DataFrame for the phase difference for all RF carrier frequencies averaged together for all types of analysis method.
        '''
        if self.pre_910_av_difference_df is None:

            if self.phase_A_minus_B_df is None:
                self.cancel_out_freq_response()

            # For pre-910 state switching data set.

            # We want to calculate for every averaging set the difference in obtained phases for the case when pre-quench 910 state is ON and OFF.

            pre_910_state_index_list = list(self.phase_A_minus_B_df.index.names)
            pre_910_state_index_list.remove('Pre-Quench 910 State')

            phase_A_minus_B_for_pre_910_state_df = self.phase_A_minus_B_df.reset_index().set_index(pre_910_state_index_list).sort_index()

            phase_A_minus_B_for_pre_910_state_df.columns = phase_A_minus_B_for_pre_910_state_df.columns.remove_unused_levels()

            pre_910_state_switching_df_group = phase_A_minus_B_for_pre_910_state_df.groupby(pre_910_state_index_list)

            def pre_910_state_subtract(df, columns_phase_std_dict, column_phase):
                '''Perform phase subtraction of phases for pre-quench 910 state being 'on' and 'off' for each averaging set of data.

                Analysis is performed for the specified types of averaging set averaging.
                '''

                df_on = df[df['Pre-Quench 910 State'] == 'on']
                df_off = df[df['Pre-Quench 910 State'] == 'off']

                phase_diff_df = (df_on.loc[slice(None), (slice(None), slice(None), slice(None), column_phase)] - df_off.loc[slice(None),(slice(None), slice(None), slice(None), column_phase)]).transform(convert_phase_to_2pi_range)

                std_df = np.sqrt(df_on.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2 + df_off.loc[slice(None),(slice(None), slice(None), slice(None), columns_phase_std_dict.keys())]**2)

                std_df = std_df.rename(columns=columns_phase_std_dict, level='Data Field')

                return phase_diff_df.join(std_df).sort_index(axis='columns').iloc[0]

            columns_phase_std_dict = {
                    'Phase STD [Rad]': 'Phase STD [Rad]',
                    'Phase RMS Repeat STD [Rad]': 'Phase RMS Repeat STD [Rad]',
                    'Phase RMS Averaging Set STD [Rad]': 'Phase RMS Averaging Set STD [Rad]'}
            column_phase = 'Phase [Rad]'

            pre_910_states_subtracted_df = pre_910_state_switching_df_group.apply(self.pre_910_state_subtract, columns_phase_std_dict, column_phase)

            # We now have obtained for every RF frequency and repeat (and possibly other experiment parameters) phase difference between two states of pre-quench 910 cavity. We want now want to average for every RF frequency all of this data for all of the repeats.

            pre_910_states_subtracted_index_list = list(pre_910_states_subtracted_df.index.names)
            pre_910_states_subtracted_index_list.remove('Repeat')
            pre_910_states_subtracted_df_group = pre_910_states_subtracted_df.groupby(pre_910_states_subtracted_index_list)

            reference_type_list = ['RF Combiner I Reference', 'RF Combiner R Reference']
            averaging_type_list = ['Phase Averaging', 'Phasor Averaging', 'Phasor Averaging Relative To DC']

            data_column = 'Phase [Rad]'

            columns_dict = {
                'Phase RMS Repeat STD': 'Phase RMS Repeat STD [Rad]',
                'Phase RMS Averaging Set STD': 'Phase RMS Averaging Set STD [Rad]',
                'Phase STD': 'Phase STD [Rad]'}

            # Average the phases over all repeats for every Waveguide Carrier Frequency.

            pre_910_states_averaged_df = pre_910_states_subtracted_df_group.apply(self.average_data_field, reference_type_list, self.harmonic_name_list, averaging_type_list, data_column, columns_dict, True)

            # This difference now needs to get divided by two, because while calculating RF CH A - RF CH B we left it without the division by 2.

            pre_910_states_averaged_df.loc[slice(None),(slice(None), slice(None), slice(None), slice(None), ['Weighted Mean'])] = pre_910_states_averaged_df.loc[slice(None),(slice(None), slice(None), slice(None), slice(None), ['Weighted Mean'])].transform(divide_and_minimize_phase, div_number=2)

            # Of course we have to divide the error by 2 as well.
            pre_910_states_averaged_df.loc[slice(None),(slice(None), slice(None), slice(None), slice(None), ['Weighted STD'])] = pre_910_states_averaged_df.loc[slice(None),(slice(None), slice(None), slice(None), slice(None), ['Weighted STD'])]/2

            # We now finally average the deviations for all RF frequencies for each analysis method.
            pre910_grouping_columns_list = list(pre_910_states_averaged_df.columns.names)
            pre910_grouping_columns_list.remove('Data Field')

            pre_910_av_difference_df = pre_910_states_averaged_df.groupby(level=pre910_grouping_columns_list, axis='columns').apply(lambda x: straight_line_fit_params(data_arr=x.xs(axis='columns',key='Weighted Mean', level='Data Field').values[:,0], sigma_arr=x.xs(axis='columns',key='Weighted STD', level='Data Field').values[:,0]))

            self.pre_910_states_averaged_df = pre_910_states_averaged_df
            self.pre_910_av_difference_df = pre_910_av_difference_df

        return self.pre_910_states_averaged_df, self.pre_910_av_difference_df

    def analyze_B_field_scan(self):
        ''' Analyze FOSOF data with B field scan
        '''

        # I assume that even if the RF phase difference for both of the combiners is changing, this change is independent of the magnetic field. This was checked previously by bringing 0.1 Tesla magnet close to the RF enclosure that contained both of the RF combiners. During this test no shift in the phase difference between the combiners was observed, thus I conclude that the phase difference between the combiners' signals is independent of the B field. I also conclude from this that the RF system itself behaves the same way regardless of the external B field.

        # To analyze the data I average all of the FOSOF phases together (for all repeats and frequencies) for each B field separately. Then I subtract the averaged FOSOF phase at the B field = 0 from all of the other B fields. The reason for this analysis is explained on p.18 Lab notes 4 (2018-08-10)

        # The averaged phase vs B field curve can be converted into resonant frequency vs B field by using the FOSOF slope for the situation when B field is set to 0 Gauss.

        # ===============================
        # Previous data analysis method. This method was deemed to be not satisfactory, because of the error analysis - I do not know how to incorporate common-mode error due to B field = 0 phases.
        # ===============================

        # I assume that even if the RF phase difference for both of the combiners is changing, this change is independent of the magnetic field. This was checked previously by bringing 0.1 Tesla magnet close to the RF enclosure that contained both of the RF combiners. During this test no shift in the phase difference between the combiners was observed, thus I conclude that the phase difference between the combiners' signals is independent of the B field. I also conclude from this that the RF system itself behaves the same way regardless of the external B field.
        # With this we have that after averaging the phases for all repeats for each B field and RF frequency separately, we get the averaged phases that have the same phase offset due to RF system + its phase drifts for given RF frequency. This means that if we subtract phase for one of the B fields from all of the B fields for given RF frequency, then the RF phase shifts will be cancelled out. After performing this operation we will have phase vs B field for every RF frequency. The assumption is then that this curve is identical for all of the RF frequencies, which in other words states that the FOSOF lineshape is independent of the B field, except of the B field dependent phase offset.
        # The averaged phase vs B field curve can be converted into resonant frequency vs B field by using the FOSOF slope for the situation when B field is set to 0 Gauss.

        # b_field_scan_index_names_list = list(fosof_phase_df.index.names)
        # b_field_scan_index_names_list.remove('Waveguide Carrier Frequency [MHz]')
        # b_field_scan_index_names_list.insert(0,'Waveguide Carrier Frequency [MHz]')
        #
        # b_field_fosof_phase_df = fosof_phase_df.reset_index().set_index(b_field_scan_index_names_list).sort_index(axis='index')
        #
        # b_field_fosof_phase_df.columns = b_field_fosof_phase_df.columns.remove_unused_levels()
        #
        # # Subtract the corresponding B field = 0 Gauss phase from every RF carrier frequency. Notice that here we use the pd.DataFrame.subtract() function, which is the same as using the '-' operation, but it also allows to specify the level at which to broadcast the values = ignore index at that given level - needed for MultiIndex dataframes.
        # b_field_fosof_phase_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')] = b_field_fosof_phase_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')].subtract(b_field_fosof_phase_df.loc[(slice(None), 0), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')].reset_index(level=1, drop=True), level=0)
        #
        # # Uncertainty in the phase shift. Notice that for given RF frequency all of the B fields have the same uncertainty contribution due to subtracting the zero field phase.
        # b_field_fosof_phase_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted STD')] = np.sqrt((b_field_fosof_phase_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted STD')]**2).add((b_field_fosof_phase_df.loc[(slice(None), 0), (slice(None), slice(None), slice(None), slice(None), 'Weighted STD')]**2).reset_index(level=1, drop=True), level=0))
        #
        #
        # # Now for every B field we want to average the phase shifts for all RF frequencies.
        # b_field_phase_shift_df = b_field_fosof_phase_df.reset_index().set_index(fosof_phase_df.index.names).sort_index(axis='index')
        #
        # b_field_phase_shift_df.columns = b_field_phase_shift_df.columns.remove_unused_levels()
        #
        # b_field_column_name = 'B_x [Gauss]'
        #
        # # Prepare the data for averaging: convert the phase shifts into [0, 2*np.pi) range and shift the phases in proper quadrants for each B field.
        # b_field_phase_shift_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')] = b_field_phase_shift_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')].transform(convert_phase_to_2pi_range).groupby(b_field_column_name).transform(lambda x: phases_shift(x)[0])
        #
        # fig, ax = plt.subplots()
        # b_field_phase_shift_df.loc[b_field_phase_shift_df.index.levels[0][0]].loc[slice(None), ('RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Averaging Set STD')].reset_index().plot(x='Waveguide Carrier Frequency [MHz]', y='Weighted Mean', kind='scatter', yerr='Weighted STD', ax=ax)
        # plt.show()
        #
        # fosof_phase_df.columns = fosof_phase_df.columns.remove_unused_levels()
        # fosof_phase_grouped_df = fosof_phase_df.groupby(b_field_column_name)
        #
        # fig, ax = plt.subplots()
        # fosof_phase_grouped_df.get_group(b_field_phase_shift_df.index.levels[0][2]).loc[slice(None), ('RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Averaging Set STD')].reset_index().plot(x='Waveguide Carrier Frequency [MHz]', y='Weighted Mean', kind='scatter', yerr='Weighted STD', ax=ax)
        # plt.show()
        #
        # b_field_column_name_list = list(b_field_phase_shift_df.columns.names)
        # b_field_column_name_list.remove('Data Field')
        #
        # b_field_av_phase_shift_df = b_field_phase_shift_df.groupby(level=b_field_column_name_list, axis='columns').apply(lambda y: y.groupby(b_field_column_name).apply(lambda x: straight_line_fit_params(data_arr=x.xs(axis='columns',key='Weighted Mean', level='Data Field').values[:,0], sigma_arr=x.xs(axis='columns',key='Weighted STD', level='Data Field').values[:,0])))
        #
        # b_field_av_phase_shift_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')] = b_field_av_phase_shift_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')]-2*np.pi
        #
        # fig, ax = plt.subplots()
        # b_field_av_phase_shift_df.drop(axis='index', labels=0).loc[slice(None), ('RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Averaging Set STD')].reset_index().plot(x='B_x [Gauss]', y='Weighted Mean', kind='scatter', yerr='Weighted STD', ax=ax)
        # plt.show()



        if self.fosof_av_phase_B_field_df is None:

            if self.fosof_phases_df is None:
                self.average_FOSOF_over_repeats()

            fosof_phase_df = self.fosof_phases_df.copy()

            fosof_phase_column_name_list = list(fosof_phase_df.columns.names)
            fosof_phase_column_name_list.remove('Data Field')

            fosof_phase_df.columns = fosof_phase_df.columns.remove_unused_levels()

            # We want to correct the FOSOF data for every B field for 2pi crossing. It is suprisingly tricky to do so in pandas. First, I am selecting only the phases themselves. Then I group this selected dataframe by B field, to which I apply the following function (acts on the whole dataframe). For every B field value we want to select the index = waveguide carrier frequencies. We then want to transform each dataframe by acting on the data column with the zero-crossing-correcting function that needs the respective list of frequencies = index as one of its inputs. We return the transformed data frame.

            def apply_correct_zero_crossing(df):
                ''' Corrects pd.DataFrame FOSOF phases for zero-crossings
                '''
                df_index = df.index.get_level_values('Waveguide Carrier Frequency [MHz]').values
                df_transformed=df.transform(lambda x: correct_FOSOF_phases_zero_crossing(df_index, x.values))
                return df_transformed

            fosof_phase_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')] = fosof_phase_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Weighted Mean')].groupby(self.b_field_column_name).apply(lambda df: apply_correct_zero_crossing(df))

            # We now want to average the FOSOF data for every B field.

            def get_av_FOSOF_phase(df):

                data_arr = df.xs(axis='columns',key='Weighted Mean', level='Data Field').values[:,0]
                data_std_arr = df.xs(axis='columns',key='Weighted STD', level='Data Field').values[:,0]
                data_length = data_arr.shape[0]
                return pd.Series({  'Phase [Rad]': np.mean(data_arr),
                                    'Phase STD [Rad]': np.sqrt(np.sum(data_std_arr**2))/data_length
                                    })

            fosof_av_phase_B_field_df = fosof_phase_df.groupby(level=fosof_phase_column_name_list, axis='columns').apply(lambda df: df.groupby(self.b_field_column_name).apply(get_av_FOSOF_phase))\

            # Subtacting B field = 0 phase = phase offset from other B field phases.

            fosof_av_phase_0_B_field_df = fosof_av_phase_B_field_df.loc[0, (slice(None), slice(None), slice(None), slice(None), 'Phase [Rad]')]

            fosof_av_phase_B_field_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Phase [Rad]')] = fosof_av_phase_df.loc[slice(None), (slice(None), slice(None), slice(None), slice(None), 'Phase [Rad]')] - fosof_av_phase_0_B_field_df

            self.fosof_av_phase_B_field_df = fosof_av_phase_B_field_df

        return self.fosof_av_phase_B_field_df

    def get_rf_pwr_calib_folder_name(self):
        # Time zone of where the data was acquired. This is needed to properly convert the UNIX time
        eastern_tz = pytz.timezone('US/Eastern')

        exp_start_datetime = datetime.datetime.fromtimestamp(self.get_exp_parameters()['Experiment Start Time [s]'], tz=eastern_tz)

        # Converting back to the string and then converting to the pandas datetime object. This is needed to remove any location information from datetime object, so that I can compare the times in the calibration info file with this converted time object.

        exp_start_datetime = pd.to_datetime(pd.to_datetime(exp_start_datetime.strftime("%Y-%m-%d %H:%M:%S")))

        calibration_s = wvg_calib_info_df[(wvg_calib_info_df['Start Date'] < exp_start_datetime) & (wvg_calib_info_df['End Date'] > exp_start_datetime)].iloc[0]

        calibration_folder_name = calibration_s['Calibration Folder Name']
        analysis_data_file_name = calibration_s['Calibration File Name']

        return calibration_folder_name, analysis_data_file_name

    def calc_wvg_E_field(self):
        ''' Determines average of the RF powers inside the waveguides corresponding to each RF power sensor reading

        Read comments in the method's code for additional details
        '''
        if self.avg_e_field_df is None:
            # Determine waveguide power calibration analysis folder and analysis file name that corresponds to this particular data set
            calibration_folder_name, analysis_data_file_name = self.get_rf_pwr_calib_folder_name()

            # Load the respective waveguide power calibration analysis. It is assumed that the analysis has been performed before.
            wvg_calib_analysis = wvg_power_calib_analysis.WaveguideCalibrationAnalysis(load_Q=True, calib_folder_name=calibration_folder_name, calib_file_name=analysis_data_file_name)

            # I will use only the calibration polynomial fits for E field vs RF power detector signal (Voltage) or its respective detected power. This is because the relationship is quite linear. This is not the case for the surviving fraction vs RF power detector signal + power.

            # RF E Field amplitude vs RF power parameters fit curves.
            extracted_E_field_vs_RF_power_fits_set_df = wvg_calib_analysis.get_converted_E_field_curve_fits()

            # Calibration frequencies and the frequencies used in the experiment are not the same. I am determining calibration frequencies that are closest to the frequencies in the experiment

            # Unique frequency values used for the experiment
            freq_arr = self.exp_data_frame.index.get_level_values('Waveguide Carrier Frequency [MHz]').drop_duplicates().values

            closest_freq_arr = freq_arr.copy()

            # Unique frequencie values used for the calibration
            calib_freq_arr = extracted_E_field_vs_RF_power_fits_set_df.index.get_level_values('Waveguide Carrier Frequency [MHz]').drop_duplicates().values

            for i in range(freq_arr.shape[0]):
                freq_diff_arr = np.abs(calib_freq_arr - freq_arr[i])
                min_freq_diff = np.min(freq_diff_arr)
                closest_freq_arr[i] = calib_freq_arr[np.where(freq_diff_arr == min_freq_diff)][0]

            # Dictionary of the experiment frequencies and the calibration frequencies (key and value respectively)
            closest_freq_dict = dict(np.array([freq_arr, closest_freq_arr]).T)

            # We need to make sure that the closest calibration frequencies are not very off from the experiment frequncies. Here I am allowing only 0.2 MHz difference
            max_diff = 0.2
            if freq_arr[np.abs(freq_arr - closest_freq_arr) > max_diff].shape[0] > 0:
                print('WARNING!!! There is/are frequencies in the experiment that are more than ' +str(max_diff) + ' MHz off from the closest calibration frequencies')

            # Dataframe containing readings from the RF power detectors
            rf_pow_df, rf_system_power_outlier_df = self.get_rf_sys_pwr_det_data()

            def get_system_E_field(df):
                ''' Calculates average electric field amplitude using the polynomial calibration fit functions for E^2 vs RF power sensor detected power [mW] and E vs RF power sensor reading [V]. The average of the two results is taken as the best estimate for the power used.

                I am not using the spline fits. Also I am not using the calibration fit curves obtained with the DC ON/OFF ratio values on the y-axis. The reason is that the curves with the E field are much more linear, and thus I believe they should give better calibration.
                '''
                freq = df.index.get_level_values('Waveguide Carrier Frequency [MHz]')[0]
                calib_freq = closest_freq_dict[freq]
                rf_channel = df.index.get_level_values('Configuration')[0]

                fits_df = extracted_E_field_vs_RF_power_fits_set_df.loc[rf_channel].loc[calib_freq].loc[['RF System Power Sensor Reading', 'RF System Power Sensor Detected Power']]

                sensor_pwr_poly_fit_func = fits_df['Polynomial Fit']['RF System Power Sensor Detected Power']
                sensor_volt_poly_fit_func = fits_df['Polynomial Fit']['RF System Power Sensor Reading']

                # Determine the electric field value using two different polynomial fit functions
                e_field_df = df[['Detected RF Power [mW]']].transform(lambda x: np.sqrt(sensor_pwr_poly_fit_func(x))).join(df[['RF Power Detector Reading [V]']].transform(lambda x: sensor_volt_poly_fit_func(x)))

                e_field_df['E Field [V/cm]'] = 1/2 * (e_field_df['Detected RF Power [mW]'] + e_field_df['RF Power Detector Reading [V]'])

                return e_field_df[['E Field [V/cm]']]

            # The conversion to the E field is performed for each waveguide separately
            rf_pow_A_df_grouped = rf_pow_df['RF System A'].groupby(['Waveguide Carrier Frequency [MHz]', 'Configuration'])
            rf_e_field_A_df = rf_pow_A_df_grouped.apply(get_system_E_field)

            rf_pow_B_df_grouped = rf_pow_df['RF System B'].groupby(['Waveguide Carrier Frequency [MHz]', 'Configuration'])
            rf_e_field_B_df = rf_pow_B_df_grouped.apply(get_system_E_field)

            # The waveguides do not have the same power in them. However, the correction that we can apply is only for the case when both the waveguides are off from the expected power by the same amount. One way to deal with this (as an approximation) is to simply take the average of the RF powers in both of the waveguides. I can also take the average of the E field amplitudes. I guess, since ultimately I am interested in power, I should take the average of two powers, not the fields.
            avg_e_field_df = np.sqrt(1/2 * (rf_e_field_A_df**2 + rf_e_field_B_df**2))

            self.avg_e_field_df = avg_e_field_df

        return self.avg_e_field_df

    def correct_phase_diff_for_RF_power(self, beam_rms_rad=1.60):
        ''' Correct FOSOF phases for the required RF power in the waveguides, given the assumed beam rms radius [mm].
        '''
        if self.beam_rms_rad != beam_rms_rad:

            self.wvg_pwr_fosof_correc_Q = True
            self.beam_rms_rad = beam_rms_rad


            avg_e_field_df = self.calc_wvg_E_field()

            # Non-corrected phases

            # Here I am setting the dataframe to None, so that the class will recalculate the phase difference data from scratch. This is important, since I am setting the corrected phase difference data to be of the same name as the original phase difference data. This way I can easily use all of the other analysis methods without any modification. If I would not set this data frame to None, then if I decide to correct the phases again (for different beam rms radius, for instance) then it will apply the correction to already corrected phase differences, which is a wrong thing to do.
            self.phase_diff_data_df = None

            phase_diff_df = self.get_phase_diff_data()

            # We have sometimes that there are one or two traces that have outlying RF power detector voltages, even though there are no issues with the SNR. Because of that we cannot perform correction for imperfect power. In this case it does not harm us to simply disregard these traces. Notice that that we can perform this filtering only if we had no filtering done before by the get_phase_diff_data() method, since it is possible to have the exact correspondence between the very low SNR and the RF power detector voltage outliers. In this case we get these traces removed in the get_phase_diff_data() method

            rf_pow_df, rf_system_power_outlier_df = self.get_rf_sys_pwr_det_data()

            # Check if the filtering was done before in the get_rf_sys_pwr_det_data() method and then if there are any power measurement outliers
            if not(self.phase_diff_data_filtered_Q) and (rf_system_power_outlier_df.shape[0] > 0):
                snr_null_df = phase_diff_df.loc[phase_diff_df.loc[slice(None), ('RF Combiner I Reference', 'First Harmonic', 'SNR')].isnull()]
                # Check if there are no low SNR values
                if snr_null_df.shape[0] == 0:

                    phase_diff_df = phase_diff_df.drop(rf_system_power_outlier_df.index)

                    warning_mssg = "To perform the correction of FOSOF phases for imperfect waveguides' power the traces corresponding to RF power detectors' outlying readings are removed from the phase difference dataframe."

                    self.err_warn_df = self.err_warn_df.append(pd.Series({'Message': warning_mssg}, name='Removal traces corresponding to positive KRYTAR power detector readings'))

                    print(warning_mssg)

            # We now want to determine the electric field amplitude that was in each of the waveguides for each experiment data point.

            # To apply the correction to FOSOF phases we need to know the FOSOF lineshape slope for each RF channel configuration. This is what I am determining in the code below
            index_name_list = list(phase_diff_df.index.names)
            index_name_list.remove('Configuration')
            index_name_list.remove('Average')
            index_name_list.remove('Repeat')
            index_name_list.remove('Elapsed Time [s]')
            index_name_list.insert(0, 'Configuration')

            # Calculating average phases for each RF frequency. We need to make sure to shift the phases in their proper qudrants before performing the average
            phase_diff_av_df = phase_diff_df.reset_index().set_index(index_name_list).sort_index()['RF Combiner I Reference', 'First Harmonic'][['Fourier Phase [Rad]']].groupby(index_name_list).aggregate(lambda x: np.mean(phases_shift(x)[0]))

            def get_fosof_slope_sign(df):
                ''' Outputs flag of whether the slope of the first-order polynomial fit to the FOSOF data is positive or not.
                '''
                x_data_arr = df['Waveguide Carrier Frequency [MHz]'].values
                phase_data_arr = df['Fourier Phase [Rad]'].values

                # Important, to make sure that the FOSOF phases do not have 0-2pi discontinuities vs RF frequency
                phase_data_arr = correct_FOSOF_phases_zero_crossing(x_data_arr, phase_data_arr)

                return np.polyfit(x_data_arr, phase_data_arr, 1)[0] > 0

            index_name_list = list(phase_diff_av_df.index.names)
            index_name_list.remove('Waveguide Carrier Frequency [MHz]')

            fosof_slope_df = phase_diff_av_df.reset_index().set_index(index_name_list).sort_index().groupby(index_name_list).apply(get_fosof_slope_sign)

            fosof_slope_df.name = 'Slope Positive'

            # Unique frequency values used for the experiment
            freq_arr = self.exp_data_frame.index.get_level_values('Waveguide Carrier Frequency [MHz]').drop_duplicates().values

            # Load the simulation data
            fosof_sim_data = hydrogen_sim_data.FOSOFSimulation(load_Q=True)

            # Use the appropriate simulation parameters and find the interpolated phase values
            e_field_needed = self.get_exp_parameters()['Waveguide Electric Field [V/cm]']
            wvg_sep = self.get_exp_parameters()['Waveguide Separation [cm]']
            acc_volt = self.get_exp_parameters()['Accelerating Voltage [kV]']

            # Important note: the simulation contains absolute frequencies, whereas we have blinded frequencies. To ensure the correct phase correction, we need to add the blind to our frequencies in the simulation.

            sim_params_dict = { 'Frequency Array [MHz]': freq_arr,
                                'Waveguide Separation [cm]': wvg_sep,
                                'Accelerating Voltage [kV]': acc_volt,
                                'Off-axis Distance [mm]': beam_rms_rad}

            zero_crossing_vs_off_axis_dist_chosen_df, fosof_sim_info_chosen_df = fosof_sim_data.filter_fosof_sim_set(sim_params_dict, blind_freq_Q=True)

            phase_vs_e_field_poly_fit_df = fosof_sim_data.get_e_field_func()

            # Determine interpolated simulation FOSOF lineshape parameters. Mainly we need to know the slope sign at each RF frequency
            zero_cross_params_s = fosof_sim_data.calc_interp_FOSOF_lineshape_params(e_field_needed)
            zero_cross_params_s

            def correct_phase_for_RF_power(df):
                ''' Determine the phase correction that should be ADDED to the experiment FOSOF phases in order to correct them for incorrect power supplied to the waveguides
                '''
                freq_value = df.index.get_level_values('Waveguide Carrier Frequency [MHz]')[0]
                slope_pos_Q = df.index.get_level_values('Slope Positive')[0]

                df_corr = df.transform(lambda col: list(map(lambda x: fosof_sim_data.interp_FOSOF_phi_for_E_field(freq_value, 0, x, e_field_needed, slope_pos_Q), col))).rename(columns={'E Field [V/cm]': 'FOSOF Phase Correction [Rad]'})

                return df_corr

            fosof_phase_corr_df = avg_e_field_df.join(fosof_slope_df).set_index('Slope Positive', append=True).groupby(['Slope Positive', 'Waveguide Carrier Frequency [MHz]']).apply(correct_phase_for_RF_power).reset_index().set_index(avg_e_field_df.index.names).join(avg_e_field_df)

            # Determine the corrected FOSOF phases. Notice that the correction is applied ONLY to the first harmonic. We do not have any reason to believe that the second harmonic should even need a correction - there should be no signal there.
            phase_diff_orig_df = phase_diff_df.copy()

            # phase_diff_df has the same id as the self.phase_diff_data_df. Thus setting phase_diff_df sets the self.phase_diff_data_df as well.
            for source in phase_diff_df.columns.get_level_values('Source').drop_duplicates().values:

                phase_diff_df.loc[slice(None), (source, 'First Harmonic', 'Fourier Phase [Rad]')] = (phase_diff_orig_df.loc[slice(None), (source, 'First Harmonic', 'Fourier Phase [Rad]')] + fosof_phase_corr_df['FOSOF Phase Correction [Rad]']).transform(convert_phase_to_2pi_range)

    def load_instance(self):
        ''' This function loads previously pickled class instance.

        The data contained in the pickled file gets loaded into this class instance. The function is very useful if one does not want to reanalyze data again.
        '''
        os.chdir(saving_folder_location)
        os.chdir(self.exp_folder_name)

        f = open(self.analysis_data_file_name_to_load, 'rb')
        loaded_dict = pickle.load(f)
        f.close()
        self.__dict__.update(loaded_dict)
        print('The class instance has been loaded')

        os.chdir(saving_folder_location)
        os.chdir(self.exp_folder_name)

    def save_instance(self, rewrite_Q=False):
        ''' Calling this function pickles the analysis class instance. If the data has been previously saved, the call to this function overwrites previously written pickled file with the same file name.
        '''

        already_exist_Q = False

        analysis_data_file_name = self.get_analysis_data_object_file_name(self.beam_rms_rad)

        os.chdir(saving_folder_location)
        os.chdir(self.exp_folder_name)

        if os.path.isfile(analysis_data_file_name):
            print('Data object file already exists.')
            already_exist_Q = True

        if (already_exist_Q and rewrite_Q) or (already_exist_Q == False):

            f = open(analysis_data_file_name, 'wb')
            pickle.dump(self.__dict__, f, 2)

            print('The class instance has been saved')

        os.chdir(saving_folder_location)
        os.chdir(self.exp_folder_name)

#%%
# #exp_folder_name='180506-123908 - FOSOF Acquisition - 0 config, 14 V per cm, PD ON 120 V,  914-918 MHz'
# #exp_folder_name='180508-170626 - FOSOF Acquisition - 0 config, 14 V per cm, PD 120 V, 914-918 MHz'
# exp_folder_name='180416-141245 - FOSOF Acquisition - 0 config, 14 V per cm PD ON 120 V'
#exp_folder_name = '180706-124508 - FOSOF Acquisition - pi config, 18 V per cm PD 120 V, 49.86 kV, 908-912 MHz'
# exp_folder_name = '180319-161930 - FOSOF Acquisition - 0 config, 18 V per cm PD OFF'
# exp_folder_name = '180327-122234 - FOSOF Acquisition - pi config, 5 V per cm PD ON 120 V'

#exp_folder_name = '180405-220057 - FOSOF Acquisition 910 onoff CGX pressure change - pi config, 18 V per cm PD 120V, P_CGX change'

# exp_folder_name = '180706-124508 - FOSOF Acquisition - pi config, 18 V per cm PD 120 V, 49.86 kV, 908-912 MHz'

# exp_folder_name = '180327-101626 - FOSOF Acquisition - pi config, 8 V per cm PD ON 120 V'
#
# exp_folder_name = '180327-103559 - FOSOF Acquisition - 0 config, 8 V per cm PD ON 120 V'
#
# exp_folder_name = '180430-123222 - FOSOF Acquisition - pi config, 18 V per cm, PD 120 V'
#
# exp_folder_name = '180430-130724 - FOSOF Acquisition - 0 config, 18 V per cm, PD 120 V'
#
# exp_folder_name = '180430-140520 - FOSOF Acquisition - 0 config, 18 V per cm, PD 120 V'
#
# exp_folder_name = '180430-144824 - FOSOF Acquisition - pi config, 18 V per cm, PD 120 V'

# exp_folder_name = '180430-154020 - FOSOF Acquisition - pi config, 18 V per cm, PD 120 V'
#
# exp_folder_name = '180625-150533 - FOSOF Acquisition - 0 config, 18 V per cm PD OFF, 49.86 kV, 908-912 MHz'

#exp_folder_name = '180625-172031 - FOSOF Acquisition - pi config, 18 V per cm PD OFF, 49.86 kV, 908-912 MHz'

#exp_folder_name = '180625-140037 - FOSOF Acquisition - 0 config, 18 V per cm PD OFF, 49.86 kV, 908-912 MHz'

#exp_folder_name = '180409-183122 - FOSOF Acquisition - pi config, 24 V per cm PD ON 120 V'

# exp_folder_name = '180416-110628 - FOSOF Acquisition - pi config, 14 V per cm PD ON 120 V'
#
# beam_rms_rad = 2.4
# # List of beam  rms radius values for correcting the FOSOF phases using the simulations. Value of None corresponds to not applying the correction.
# data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=False, beam_rms_rad_to_load=beam_rms_rad)
#
# # The power correction is performed only for the simple FOSOF data sets.
# if data_set.get_exp_parameters()['Experiment Type'] != 'Waveguide Carrier Frequency Sweep':
#     beam_rms_rad = None
#     data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)
#
# fc_df = data_set.get_fc_data()
# quenching_df = data_set.get_quenching_cav_data()
# rf_pow_df, rf_system_power_outlier_df = data_set.get_rf_sys_pwr_det_data()
#
# digi_df = data_set.get_digitizers_data()
#
# comb_phase_diff_df = data_set.get_combiners_phase_diff_data()
# digi_delay_df = data_set.get_inter_digi_delay_data()
#
# if beam_rms_rad is not None:
#     data_set.correct_phase_diff_for_RF_power(beam_rms_rad)
#
# phase_diff_df = data_set.get_phase_diff_data()
# phase_av_set_averaged_df = data_set.average_av_sets()
# phase_A_minus_B_df, phase_freq_response_df = data_set.cancel_out_freq_response()
# fosof_ampl_df, fosof_phase_df = data_set.average_FOSOF_over_repeats()
#
# data_set.save_instance(rewrite_Q=True)
# #%%
# data_set.get_exp_parameters()
