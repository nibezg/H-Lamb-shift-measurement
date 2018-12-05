# version 2:
# - Employs experiment information file to check if the analysis needs to be performed. This will save a lot of time if only several data sets need to be analyzed, since other data sets will not have to be opened. The experiment information file also stores data set parameters for easy access. This can later be used for easy sorting/filtering of the data sets.
from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
#sys.path.insert(0,"C:/Users/Helium1/Google Drive/Code/Python/Testing/Blah") #
# For home
sys.path.insert(0,"E:/Google Drive/Research/Lamb shift measurement/Code")

from exp_data_analysis import *

import re
import numpy.fft
import time
import scipy.fftpack
import matplotlib.pyplot as plt
import math

import threading
from queue import Queue

# Package for wrapping long string to a paragraph
import textwrap

from tkinter import *
from tkinter import ttk

from tkinter import messagebox

def update_exp_info_file(exp_time_stamp, analysis_done_Q=True, fully_acquired_Q=False, errors_Q=True):
    ''' Use this function if  during analysis one needs to update the analysis parameters for the given data set and write all of the experiment information to back to the file.
    '''

    global exp_analysis_done_col, exp_errors_col, exp_fully_acquired_col, exp_info_file_loc, exp_info_file_name, exp_info_df, exp_info_df_comments_string

    exp_info_df.loc[experiment_time_stamp, exp_analysis_done_col] = analysis_done_Q
    exp_info_df.loc[experiment_time_stamp, exp_fully_acquired_col] = fully_acquired_Q
    exp_info_df.loc[experiment_time_stamp, exp_errors_col] = errors_Q

    os.chdir(exp_info_file_loc)
    exp_info_file_object = open(exp_info_file_name, 'w')
    exp_info_file_object.write(exp_info_df_comments_string)
    exp_info_file_object.close()

    exp_info_df.to_csv(path=exp_info_file_name, mode='a', header=True)


def fosof_analysis():

    global analysis_interrupted_Q, analysis_in_process_Q
    analysis_in_process_Q = True
    start_button_text.set('Stop the analysis')

    # Analysis version
    version_number = 0.1

    # Folder containing binary traces in .npy format
    binary_traces_folder = "//LAMBSHIFT-PC/Users/Public/Documents/binary traces"
    # Folder containing acquired data table
    data_folder = "//LAMBSHIFT-PC/Google Drive/data"
    # Experiment data file name
    data_file = 'data.txt'

    # Location where the analyzed experiment is saved
    saving_folder_location = 'E:/2017-10-17 Lamb Shift Measurement/Data/FOSOF analyzed data sets'
    # Analyzed data file name
    data_analyzed_file = 'data_analyzed v' + str(version_number) + '.txt'

    # File containing information on experiments. Stores all of the experiment parameters data and help to decide whether the data needs to be analyzed
    exp_info_file_loc = saving_folder_location

    exp_info_file_name = 'fosof_data_sets_info.csv'

    # A subset of columns in the data sets information file
    exp_analysis_done_col = 'Analysis Finished'
    exp_errors_col = 'Error(s) During Analysis'
    exp_fully_acquired_col = 'Data Set Fully Acquired'
    exp_folder_col = 'Experiment Folder Name'
    exp_comments_col = 'Comments'

    exp_index_name = 'Acquisition Start Date'

    # Comments that go with the file
    general_comments = {'General information': 'Table of FOSOF data sets with the experiment parameters. With this table one can determine if the data set has been analyzed before and whether there were some problems during acquisition or analysis.',
    exp_analysis_done_col: '[Boolean] This column indicates whether the raw data analysis has been run on the data set and finished without errors. This means that all of the available data has been analyzed. It is possible, however, that the experiment has been interrupted, so that not all of the data has been acquired, however, it did not result in the error during the analysis.',
    exp_errors_col: '[Boolean] The column indicates if there were any errors during the analysis. This usually means that the data set should not be used for any further data analysis, due to some errors during the data acquisition or, possibly, low signal quality, or that there are some issues with the raw data analysis code itself.',
    exp_fully_acquired_col: '[Boolean] Shows whether the data set has been fully acquired. Which is determined by the match between the amount of data expected from the data set and the amount of data available from the data set. Obviously it also checks if during any of these checks, there was more data available than expected (exception is raised in this case).'}

    # List of folder names of FOSOF experiments to analyze
    exp_name_list_folder = saving_folder_location

    exp_name_list_file = 'fosof_data_sets_list.csv'

    os.chdir(exp_name_list_folder)
    exp_name_list = pd.read_csv(filepath_or_buffer=exp_name_list_file, delimiter=',', comment='#', header=0)

    # -------------------------------------
    # Added on 2018-06-22
    # For some reason some experiment names have tabs or spaces after them. This removes them.
    exp_name_list[exp_folder_col] = exp_name_list[exp_folder_col].transform(lambda x: x.strip())

    # Check whether there are duplicates in the list of experiment names. This is important later on when the comments get updated for the file of the experiment info.
    if exp_name_list[exp_folder_col][exp_name_list[exp_folder_col].duplicated(keep=False)].shape[0] != 0:
        raise FosofAnalysisError('The file with FOSOF data set names has duplicated experiment names.')
    # -------------------------------------

    exp_info_df_comments_string = '\n'.join(list(map(lambda x: '# ' + x[0] + ' = ' + x[1], general_comments.items())))


    # Perform data analysis for the list of the experiments. Stop the analysis, if the analysis has been interrupted.
    exp_counter = 0;
    while exp_counter < exp_name_list.shape[0] and analysis_interrupted_Q == False:
        #print(exp_counter)
        # Current experiment folder name
        experiment_folder = exp_name_list.loc[exp_counter, exp_folder_col].strip()

        experiment_current_name_tk_var.set('Experiment: (' + str(exp_counter+1) + str('/') + str(exp_name_list.shape[0]) + ') ' + experiment_folder)

        # Analyzed experiment folder name
        data_analyzed_folder = experiment_folder
        # Time for the start of the data acquisition
        exp_timestamp = get_experiment_start_time(experiment_folder)

        # Open the file containing the information about previously analyzed experiments. This file makes it much easier to see if the data set had been analyzed before.

        # Flag to decide whether the data analysis has to be performed, which means that the experiment data has to be accessed.
        perform_analysis_Q = True

        os.chdir(exp_info_file_loc)

        # Of course, before opening the file we should check if it exists.
        if not(os.path.isfile(exp_info_file_name)):
            print('The table for storing information about the FOSOF experiments has not been created yet.')
            # The Data Frame containing information about the FOSOF experiments.
            exp_info_df = pd.DataFrame([])
        else:
            exp_info_df = pd.read_csv(filepath_or_buffer=exp_info_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True)
            exp_info_df = exp_info_df.set_index(exp_index_name)

            # Convert index to timestamps, otherwise they are read as strings
            exp_info_df.index = pd.to_datetime(exp_info_df.index, infer_datetime_format=True)

        # We check if there is any data in the file first, if there is, then we can check if we have the timestamp (as the index) that matched the start time of the current experiment and (if this is true), then check if at that timestamp the experiment folder matches the current one, and if this is true as well, then we check whether the data analysis has been finished. Only if these all four conditions are satisfied, then we skip this data set.

        # This flag tells whether the experiment folder parameters are already in the experiment information file
        exp_folder_in_info_file_Q = False
        if  (exp_info_df.size > 0):
            if (exp_timestamp in exp_info_df.index):
                exp_folder_in_info_file_Q = True
                if (exp_info_df.loc[exp_timestamp, exp_folder_col] == experiment_folder) and \
                (exp_info_df.loc[exp_timestamp, exp_analysis_done_col] == True):
                    perform_analysis_Q = False
        if perform_analysis_Q:
            # Prepare for the data analysis. This includes getting the experiment parameters and checking if the analysis has been started/finished before.

            os.chdir(data_folder)
            os.chdir(experiment_folder)

            # Get header names from the data file
            exp_data = pd.read_csv(filepath_or_buffer=data_file, delimiter=',', comment='#', header=0, skip_blank_lines=True, skipinitialspace=True)
            exp_column_names = exp_data.columns.values

            # Get the experiment parameters from the data file
            exp_params_dict, comment_string_arr = get_experiment_params(data_file)

            # Create pd.Series with the experiment parameters and append it to the experiment information Data Frame, if needed.
            exp_info_s = pd.Series(exp_params_dict, name=exp_timestamp)
            if not(exp_folder_in_info_file_Q):
                exp_info_s[exp_folder_col] = experiment_folder
                exp_info_s[exp_analysis_done_col] = False
                exp_info_s[exp_fully_acquired_col] = False
                exp_info_s[exp_errors_col] = False
                exp_info_s[exp_comments_col] = exp_name_list.loc[exp_counter, exp_comments_col]
                exp_info_df = exp_info_df.append(exp_info_s)

            # Important experiment parameters
            n_Bx_steps = exp_params_dict['Number of B_x Steps']
            n_By_steps = exp_params_dict['Number of B_y Steps']
            n_averages = exp_params_dict['Number of Averages']
            sampling_rate = exp_params_dict['Digitizer Sampling Rate [S/s]']
            n_digi_samples = exp_params_dict['Number of Digitizer Samples']
            n_freq_steps = exp_params_dict['Number of Frequency Steps']
            n_repeats = exp_params_dict['Number of Repeats']
            digi_ch_range = exp_params_dict['Digitizer Channel Range [V]']
            offset_freq_array = exp_params_dict['Offset Frequency [Hz]']
            pre_910_on_n_digi_samples = exp_params_dict['Pre-Quench 910 On Number of Digitizer Samples']

            # Expected number of rows in the data file
            rows_expected = n_repeats * n_freq_steps * len(offset_freq_array) * n_averages * 2

            # B field values for different axes are assumed to be part of the same parameter: B field in such a way that when the B field along one axis is getting scanned, B field along another axis is set to 0. Thus the total number of B field scan steps is the sum of the scan steps along each axis.  When both the Bx and By are set to zero and are not getting scanned though, the B field parameter is not getting changed, thus the total number of B field steps is 1 (one).
            if exp_params_dict['B_x Min [Gauss]'] == 0 and exp_params_dict['B_x Max [Gauss]'] == 0 and exp_params_dict['B_y Min [Gauss]'] == 0 and exp_params_dict['B_y Max [Gauss]'] == 0:
                n_B_field_steps = 1
            else:
                if n_Bx_steps == 1 and n_By_steps > 1:
                    n_B_field_steps = n_By_steps
                if n_By_steps == 1 and n_Bx_steps > 1:
                    n_B_field_steps = n_Bx_steps
                if n_Bx_steps > 1 and n_By_steps > 1:
                    n_B_field_steps = n_Bx_steps + n_By_steps

            rows_expected = rows_expected * n_B_field_steps

            if exp_params_dict['Pre-Quench 910 On/Off']:
                rows_expected = rows_expected * 2

            if 'Number of Mass Flow Rate Steps' in exp_params_dict:
                rows_expected = rows_expected * exp_params_dict['Number of Mass Flow Rate Steps']

            # Boolean that determines if the analysis of the data set has been already performed
            analysis_finished_Q = False

            # Check whether the analyzed data folder exists.
            os.chdir(saving_folder_location)
            if os.path.isdir(data_analyzed_folder):
                print("Data folder exists.")
            else:
                print("Data set has not been analyzed yet. Creating the experiment analysis folder")
                os.mkdir(data_analyzed_folder)
            os.chdir(data_analyzed_folder)

            # Check whether the analyzed data file exists. If it does, then determine number of already analyzed rows. Else, create the analyzed data file and write experiment parameters = comment lines + column names to it.
            if os.path.isfile(data_analyzed_file):
                print("Analyzed data file exists.")
                print(data_analyzed_folder)
                data_analyzed = pd.read_csv(filepath_or_buffer=data_analyzed_file, delimiter=',', comment='#', header=0)
                # Set the number of analyzed rows
                rows_analyzed_index = data_analyzed.shape[0]

                # If all of the data has been analyzed already, then there is no need to rerun the analysis
                if rows_analyzed_index == rows_expected:

                    analysis_finished_Q = True

                    # Update the corresponding parameters in the experiment information Data Frame
                    exp_info_df.loc[exp_timestamp, exp_analysis_done_col] = True
                    exp_info_df.loc[exp_timestamp, exp_fully_acquired_col] = True

                    print('The data set has been already fully analyzed.')
            else:
                print("Data set has not been analyzed yet. Creating the analysis data file")
                data_analyzed_file_object = open(data_analyzed_file, 'w')

                # Write list of comments and a row of column names to the newly created data analysis file
                data_analyzed_file_object.write(''.join(comment_string_arr))

                # column_names_row_string = ','.join([column_name for column_name in exp_column_names])+'\n'

                #data_analyzed_file_object.write(column_names_row_string)
                data_analyzed_file_object.close()

                # Set the number of analyzed rows to zero
                rows_analyzed_index = 0

            # Open the analyzed data file for appending
            data_analyzed_file_object = open(data_analyzed_file, 'a')

            # Start data analysis. We go from one row to another. In every row we get the trace names for both of the digitizers and analyze each trace. However, before that we check if the given trace file name exist, because it is possible that the file might not appear in the binary folder for some reason. If the file does not exist we wait for t_wait_trace_appear seconds and then try again. If no file is found after that then the exception will be raised.

            # in seconds
            t_wait_trace_appear = 60

            # It is also possible that the acquisition gets interrupted. This way the number of rows in the experiment data file will not change after reading it several times. Of course, it is possible that the analysis code is analyzing data faster than it is getting acquired. Thus we need to have some time delay to allow for that.

            # Per row we acquire 4 rows. To allow for other delays we multiply this minimum delay time by a factor of 3. This time is in seconds
            if exp_params_dict['Pre-Quench 910 On/Off']:
                t_wait_rows_appear = 3 * (4 * pre_910_on_n_digi_samples/sampling_rate) + 30
            else:
                t_wait_rows_appear = 3 * (4 * n_digi_samples/sampling_rate)

            t_wait_rows_appear = t_wait_rows_appear * 2

            # Column names for the digitizer trace names
            digi_traces_names_list = ['Detector Trace Filename', 'RF Power Combiner I Digi 1 Trace Filename', 'RF Power Combiner R Trace Filename', 'RF Power Combiner I Digi 2 Trace Filename']

            # Channel names to prepend to analyzed data columns for each channel
            digi_channel_names = ['Detector', 'RF Power Combiner I Digi 1', 'RF Power Combiner R', 'RF Power Combiner I Digi 2']

            acquisition_interrupted_Q = False
            exp_data_rows_previous = 0
            exp_data_rows = 0

            while analysis_finished_Q == False and acquisition_interrupted_Q == False and analysis_interrupted_Q == False:

                os.chdir(data_folder)
                os.chdir(experiment_folder)

                # Get the experiment data
                exp_data = pd.read_csv(filepath_or_buffer=data_file, delimiter=',', comment='#', header=0)

                # Number of rows currently in the experiment data file
                exp_data_rows_previous = exp_data_rows
                exp_data_rows = exp_data.shape[0]

                # Check if for some reason number of rows in the experimet file is larger than the expected number of rows. If true, then raise the exception, because it might mean that the data set is not of the expected type. Or something is wrong in the analysis code, which requires attention.
                if exp_data_rows > rows_expected:
                    # Close the handle to the analyzed data file
                    data_analyzed_file_object.close()
                    raise FosofAnalysisError("The experiment data file has "+str(exp_data_rows)+"rows. We expect only "+str(rows_expected)+" rows.")
                    os.chdir(data_folder)

                # If the number of rows in the experiment file did not change, since checked last time, then wait for some time
                if exp_data_rows == exp_data_rows_previous:
                    print('No new data in the experiment file has been found. Waiting for ' + str(t_wait_rows_appear) + ' seconds for the data to appear...')
                    time.sleep(t_wait_rows_appear)
                    # Read in the data again and check its numbers of rows
                    exp_data = pd.read_csv(filepath_or_buffer=data_file, delimiter=',', comment='#', header=0)
                    exp_data_rows = exp_data.shape[0]

                    # If still the number of rows did not change, then most likely the acquisition has been interrupted.
                    if exp_data_rows == exp_data_rows_previous:
                        print('Most likely the acquisition of the data set has been interrupted. Exiting the analysis loop...')

                        acquisition_interrupted_Q = True;

                        # Update the corresponding parameters in the experiment information Data Frame
                        exp_info_df.loc[exp_timestamp, exp_analysis_done_col] = True
                        exp_info_df.loc[exp_timestamp, exp_fully_acquired_col] = False


                # Since, while analyzing the data, the experiment might not be fully finished, then the number of rows in the experiment data file can be less than the expected one. Thus we will first analyze all of the available rows and after that, if everything goes well, the experiment data file gets read again to check its number of rows.
                while rows_analyzed_index < exp_data_rows and analysis_interrupted_Q == False:
                    # trace file names in the current row
                    digi_traces_arr = exp_data.loc[rows_analyzed_index,digi_traces_names_list].values

                    # Perform analysis of each trace in the row and combine the analysis information with the rest of the experiment data

                    os.chdir(binary_traces_folder)
                    os.chdir(experiment_folder)

                    # Series to store analysis information from each trace of the row
                    analyzed_data = pd.Series([])
                    trace_exists_Q = True;

                    for trace_index in range(len(digi_traces_arr)):

                        digi_trace_file_name = digi_traces_arr[trace_index]
                        # Add extension name to the file
                        digi_trace_file_name = digi_trace_file_name + '.digi.npy'

                        # Check if the trace file name exist
                        if not(os.path.isfile(digi_trace_file_name)):
                            print('Trace file <' + digi_trace_file_name +'> has not been found. Waiting for ' + str(t_wait_trace_appear) + ' seconds...')
                            time.sleep(t_wait_trace_appear)
                            # Even after waiting the required amount of time, the trace is still not available, then there is something wrong with the acquisition.
                            if not(os.path.isfile(digi_trace_file_name)):
                                # Close the handle to the analyzed data file
                                data_analyzed_file_object.close()
                                trace_exists_Q = False

                                raise FosofAnalysisError('The trace is still not available. There seems to be an error in saving or accessing the traces. Or the data set acquisition has been interrupted.')


                        # Import the trace and get its parameters (from the file name)
                        trace_params, trace_V_array = import_digi_trace(digi_trace_file_name, digi_ch_range)
                        n_samples = trace_V_array.size # Number of samples in the trace

                        # Get the offset frequency. If there is only one offset frequency in the experiment parameters, then this offset frequency is used for all traces.
                        if len(offset_freq_array) > 1:
                            offset_freq = trace_params['Offset frequency [Hz]']
                        else:
                            offset_freq = offset_freq_array[0]
                        if trace_params['Pre-910 state [On/Off]'] == 'on':
                            n_samples_expected = pre_910_on_n_digi_samples
                        else:
                            n_samples_expected = n_digi_samples

                        if n_samples != n_samples_expected:
                            # Close the handle to the analyzed data file
                            data_analyzed_file_object.close()

                            raise FosofAnalysisError("Number of samples in the digitizer trace does not match the expected number of samples from the experiment parameters")

                        f0 = float(sampling_rate)/n_samples # Minimum Fourier frequency [Hz]

                        # Find FFT of the trace and extract DC, SNR, phase, amplitude, average noise at the first two harmonics of the offset frequency
                        trace_spectrum_data = get_Fourier_spectrum(trace_V_array, sampling_rate)
                        dc = trace_spectrum_data.iloc[0].loc['Fourier Amplitude [Arb]']

                        # Find the Fourier parameters at the offset frequency and its harmonic. Make sure to keep the order the same - offset frequency is first, second harmonic is the second. This is important for the following combination of the results.
                        fourier_params = get_Fourier_harmonic(trace_spectrum_data, [offset_freq, 2*offset_freq], trace_length=n_samples/float(sampling_rate))

                        # Combine the Fourier parameters in a way that can be combined with the rest of the experiment data for the given row.
                        first_harmonic_fourier_params = fourier_params.iloc[0]

                        # Renaming the index
                        first_harmonic_fourier_params = first_harmonic_fourier_params.rename(index=
                        {'Average Total Noise Fourier Amplitude [Arb]' : 'Average Total Noise Fourier Amplitude [V]',
                        'Fourier Amplitude [Arb]' : 'Fourier Amplitude [V]'})
                        first_harmonic_fourier_params = first_harmonic_fourier_params.rename(index={i: digi_channel_names[trace_index] +' '+ 'First Harmonic ' + i for i in first_harmonic_fourier_params.index.values})

                        second_harmonic_fourier_params = fourier_params.iloc[1]
                        second_harmonic_fourier_params = second_harmonic_fourier_params.rename(index=
                        {'Average Total Noise Fourier Amplitude [Arb]' : 'Average Total Noise Fourier Amplitude [V]',
                        'Fourier Amplitude [Arb]' : 'Fourier Amplitude [V]'})
                        second_harmonic_fourier_params = second_harmonic_fourier_params.rename(index={i: digi_channel_names[trace_index] +' '+ 'Second Harmonic ' + i for i in second_harmonic_fourier_params.index.values})

                        # Combining the data for the first two harmonics with the DC value
                        analyzed_data = pd.concat([analyzed_data, first_harmonic_fourier_params,second_harmonic_fourier_params, pd.Series([dc], index={digi_channel_names[trace_index] + ' DC [V]'})])

                    # Experiment data row combined with the Fourier analysis data. We first convert the Series into 2D DataFrame and then transpose it to turn the rows into columns
                    combined_experiment_row = pd.DataFrame(pd.concat([exp_data.loc[rows_analyzed_index], analyzed_data])).transpose()

                    #os.chdir(saving_folder_location)
                    #os.chdir(data_analyzed_folder)

                    # Appending the experiment data to the analyzed data file

                    # In case if no analysis has been done previously on the data set, then write column names to the analyzed data file in addition to the first row analyzed data.
                    if rows_analyzed_index == 0:

                        data_analyzed_string = combined_experiment_row.to_csv(mode='a', header=True, index=False)

                    else:
                        data_analyzed_string = combined_experiment_row.to_csv(mode='a', header=False, index=False)

                    data_analyzed_file_object.write(data_analyzed_string)

                    # Incrementing the analyzed row index counter
                    rows_analyzed_index = rows_analyzed_index + 1

                    # Update GUI variables for the progress indicator

                    fract_rows_analyzed = float(rows_analyzed_index)/rows_expected

                    num_rows_analyzed.set('Rows analyzed: ' + str(round(100*fract_rows_analyzed,1)) + '% =' + str(rows_analyzed_index) + '/' + str(rows_expected))
                    progress_indicator['value'] = progress_indicator['maximum'] * fract_rows_analyzed

                    if rows_analyzed_index == rows_expected:
                        analysis_finished_Q = True

                        # Update the corresponding parameters in the experiment information Data Frame
                        exp_info_df.loc[exp_timestamp, exp_analysis_done_col] = True
                        exp_info_df.loc[exp_timestamp, exp_fully_acquired_col] = True

                        print('Data analysis has been completed.')

            # Write the experiment information DataFrame to the file
            os.chdir(exp_info_file_loc)
            exp_info_file_object = open(exp_info_file_name, 'w')
            exp_info_file_object.write(exp_info_df_comments_string+'\n'+'\n')
            exp_info_file_object.close()

            exp_info_df.index.name = exp_index_name
            exp_info_df.to_csv(path_or_buf=exp_info_file_name, mode='a', header=True)

            # Close the handle to the analyzed data file
            data_analyzed_file_object.close()
            os.chdir(data_folder)

        exp_counter += 1

    # -------------------------------------
    # Added on 2018-06-22

    # It is possible that the csv file with the list of experiments had its comments column updated. It happens that after already acquiring a data set I put some additional comments into for it. Thus we would want to make sure that the comments get updated even after the data set has been analyzed and recorded into the csv file containing information about the data sets.


    # Read in the experiment info file
    os.chdir(exp_info_file_loc)
    exp_info_df = pd.read_csv(filepath_or_buffer=exp_info_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True)
    exp_info_df = exp_info_df.set_index(exp_index_name)

    # Convert index to timestamps, otherwise they are read as strings
    exp_info_df.index = pd.to_datetime(exp_info_df.index, infer_datetime_format=True)

    # Make sure there are not tabs or whitespaces at the end of the experiment folder names
    exp_info_df[exp_folder_col] = exp_info_df[exp_folder_col].transform(lambda x: x.strip())

    exp_info_df = exp_info_df.reset_index().set_index(exp_folder_col)

    # Update the comments in the experiment info file with the comments in the list of the experiments file.
    exp_info_df[exp_comments_col] = exp_name_list.set_index(exp_folder_col).loc[exp_info_df.index][exp_comments_col]

    # Set the index back to initial column
    exp_info_df = exp_info_df.reset_index().set_index(exp_index_name)

    # Write the experiment information DataFrame to the file
    os.chdir(exp_info_file_loc)
    exp_info_file_object = open(exp_info_file_name, 'w')
    exp_info_file_object.write(exp_info_df_comments_string+'\n'+'\n')
    exp_info_file_object.close()

    exp_info_df.index.name = exp_index_name
    exp_info_df.to_csv(path_or_buf=exp_info_file_name, mode='a', header=True)
    # -------------------------------------

    # In case the analysis has been interrupted, then after stopping the analysis, set this boolean back to False, so that the analysis could continued again if needed.
    if analysis_interrupted_Q == True:
        analysis_interrupted_Q = False

    # Set the button name back to its initial text value.
    start_button_text.set('Analyze data sets')
    analysis_in_process_Q = False

def fosof_analysis_thread_start():
    global analysis_interrupted_Q, analysis_in_process_Q
    if analysis_in_process_Q == False:
        fosof_analysis_thread = threading.Thread(target=fosof_analysis)
        fosof_analysis_thread.start()
    else:
        analysis_interrupted_Q = True

def ask_quit():
    if analysis_in_process_Q == False:
        root.destroy()
    else:
        if messagebox.askokcancel("Quit", "Do you really want to quit without stopping the analysis first?"):
            root.destroy()

if __name__ == '__main__':

    analysis_in_process_Q = False
    analysis_interrupted_Q = False

    root = Tk()
    root.title("FOSOF analysis")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    mainframe = ttk.Frame(root, padding=(3,3,12,12))
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    experiment_current_name_tk_var = StringVar()
    experiment_current_name_tk_var.set('Experiment: ')

    exp_label = ttk.Label(mainframe, textvariable=experiment_current_name_tk_var)
    exp_label.grid(column=0, row=0, sticky=(W))

    num_rows_analyzed = StringVar()
    num_rows_analyzed.set('Rows analyzed:')

    progress_label = ttk.Label(mainframe, textvariable=num_rows_analyzed)
    progress_label.grid(column=0, row=1, sticky=(W,E))

    progress_indicator = ttk.Progressbar(mainframe, orient=HORIZONTAL, length=300, mode='determinate')
    progress_indicator.grid(column=0, row=2, sticky=(W, E))


    start_button_text = StringVar()
    start_button_text.set('Analyze data sets')

    start_button = ttk.Button(mainframe, textvariable=start_button_text, command=fosof_analysis_thread_start)
    start_button.grid(column=0, row=3)

    for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)


    mainframe.columnconfigure(1, weight=1)
    mainframe.rowconfigure(1, weight=1)
    mainframe.columnconfigure(2, weight=1)
    mainframe.rowconfigure(2, weight=1)
    mainframe.columnconfigure(3, weight=1)
    mainframe.rowconfigure(3, weight=1)

    root.protocol("WM_DELETE_WINDOW", ask_quit)
    root.mainloop()
#%%
