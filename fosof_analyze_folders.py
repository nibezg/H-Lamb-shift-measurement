from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

sys.path.insert(0,"C:/Users/Helium1/Google Drive/Code/Python/Testing/Blah") #
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


import threading
from Queue import Queue

# Package for wrapping long string to a paragraph
import textwrap

from Tkinter import *
import ttk
import tkMessageBox
#%%

def fosof_data_sets_analyze():
    global analysis_interrupted_Q, analysis_in_process_Q, stop_progress_thread, expected_analysis_duration

    analysis_in_process_Q = True
    start_button_text.set('Stop the analysis')

    # Location where the analyzed experiment is saved
    saving_folder_location = 'C:/Research/Lamb shift measurement/Data/FOSOF analyzed data sets'

    # Analysis version. Needed for checking if the data set has been analyzed before.
    version_number = 0.1

    # Name of the folder that stores experiment analysis data
    analyzed_data_folder = 'Data Analysis ' + str(version_number)

    # File containing parameters and comments about all of the data sets.
    exp_info_file_name = 'fosof_data_sets_info.csv'
    exp_info_index_name = 'Experiment Folder Name'

    av_time_per_row_file_name = 'analysis_time_per_row.txt'

    os.chdir(saving_folder_location)
    exp_info_df = pd.read_csv(filepath_or_buffer=exp_info_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True, dtype={'Error(s) During Analysis': np.bool})

    exp_info_df[exp_info_index_name] = exp_info_df[exp_info_index_name].transform(lambda x: x.strip())

    exp_info_df = exp_info_df.set_index(exp_info_index_name)

    # Pick only fully analyzed data sets that had no errors during the analysis/acquisition.
    exp_info_chosen_df = exp_info_df[exp_info_df['Data Set Fully Acquired'] & ~(exp_info_df['Error(s) During Analysis']) & (exp_info_df['Analysis Finished'])]

    exp_name_list = exp_info_chosen_df.index

    os.chdir(saving_folder_location)

    if os.path.isfile(av_time_per_row_file_name):
        av_duration_per_row_df = pd.read_csv(filepath_or_buffer=av_time_per_row_file_name, delimiter=',', comment='#', header=0, skip_blank_lines=True)
        av_duration_per_row_df = av_duration_per_row_df.set_index(exp_info_index_name)
    else:
        av_duration_per_row_df = None

    # Perform data analysis for the list of the experiments. Stop the analysis, if the analysis has been interrupted.
    exp_counter = 0

    while exp_counter < exp_name_list.shape[0] and analysis_interrupted_Q == False:

        exp_folder_name = exp_name_list[exp_counter]

        experiment_current_name_tk_var.set('Experiment: (' + str(exp_counter+1) + str('/') + str(exp_name_list.shape[0]) + ') ' + exp_folder_name)

        os.chdir(saving_folder_location)
        os.chdir(exp_folder_name)

        if not(os.path.isdir(analyzed_data_folder)):
            print(exp_folder_name)

            time_start = time.time()

            if av_duration_per_row_df is None:
                av_duration_average = 0
            else:
                av_duration_average = np.mean(av_duration_per_row_df['Average Analysis Time Per Row [ms]'].values) / 1E3

            data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_data_Q=False)
            n_rows = data_set.exp_data_frame.shape[0]

            expected_analysis_duration = av_duration_average * n_rows

            stop_progress_thread = False

            progress_bar_thread = threading.Thread(target=progress_bar_update)
            progress_bar_thread.start()

            data_set.save_analysis_data()

            stop_progress_thread = True
            progress_bar_thread.join()

            time_end = time.time()
            analysis_duration = time_end - time_start
            av_duration_per_row = analysis_duration / n_rows

            av_duration_per_row_append_df = pd.DataFrame({'Experiment Folder Name': pd.Series(exp_folder_name), 'Average Analysis Time Per Row [ms]':pd.Series([av_duration_per_row*1E3])}).set_index('Experiment Folder Name')

            if av_duration_per_row_df is None:
                av_duration_per_row_df = av_duration_per_row_append_df

            else:
                av_duration_per_row_df = av_duration_per_row_df.append(av_duration_per_row_append_df)

            # Saving the averaging time per row data back to the file. This writing is done inefficiently: after every new data set we rewrite the whole file instead of just appending the new row. I am too lazy to fix this.
            av_duration_per_row_df.drop_duplicates().to_csv(path_or_buf=av_time_per_row_file_name, mode='w', header=True)

        exp_counter = exp_counter + 1

    os.chdir(saving_folder_location)



    # In case the analysis has been interrupted, then after stopping the analysis, set this boolean back to False, so that the analysis could continued again if needed.
    if analysis_interrupted_Q == True:
        analysis_interrupted_Q = False

    # Set the button name back to its initial text value.
    start_button_text.set('Analyze data sets')
    analysis_in_process_Q = False

#%%

def progress_bar_update():
    global expected_analysis_duration, stop_progress_thread, expected_analysis_duration
    sec_elapsed = 0
    sleep_time = 1

    while not(stop_progress_thread):
        time.sleep(sleep_time)
        sec_elapsed = sec_elapsed + sleep_time

        # Update GUI variables for the progress indicator
        if expected_analysis_duration == 0:
            fract_time_elapsed = 0
        else:
            fract_time_elapsed = float(sec_elapsed)/expected_analysis_duration

        time_elapsed.set('Time elapsed: ' + str(round(100*fract_time_elapsed,1)) + '% =' + str(sec_elapsed) + '/' + str(round(expected_analysis_duration,1)))
        progress_indicator['value'] = progress_indicator['maximum'] * fract_time_elapsed


def fosof_analysis_thread_start():
    global analysis_interrupted_Q, analysis_in_process_Q
    if analysis_in_process_Q == False:
        fosof_analysis_thread = threading.Thread(target=fosof_data_sets_analyze)
        fosof_analysis_thread.start()
    else:
        analysis_interrupted_Q = True

def ask_quit():
    if analysis_in_process_Q == False:
        root.destroy()
    else:
        if tkMessageBox.askokcancel("Quit", "Do you really want to quit without stopping the analysis first?"):
            root.destroy()
#%%
if __name__ == '__main__':

    analysis_in_process_Q = False
    analysis_interrupted_Q = False
    stop_progress_thread = True
    expected_analysis_duration = 0

    root = Tk()
    root.title("FOSOF data analysis")
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

    time_elapsed = StringVar()
    time_elapsed.set('Time elapsed:')

    progress_label = ttk.Label(mainframe, textvariable=time_elapsed)
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

# #%%
# os.chdir('C:\Users\Helium1\Downloads')
# f = open('fosofdatasetsinanalysis.txt', 'r')
# file_list = f.read()
# f.close()
# file_list = file_list.split('\n')
# file_list.remove('LOW SPEED STUFF')
# file_list.remove('')
# file_list.remove('')
# #%%
# len(file_list)
# #%%
# np.setdiff1d(file_list, exp_info_df.index.values)
#%%
av_duration_per_row_s = pd.Series({
    'Experiment Folder Name': 'Blah',
    'Average Analysis Time Per Row [ms]': 1})
#%%
av_duration_per_row_s
#%%

av_duration_per_row_df = pd.DataFrame({'Experiment Folder Name': pd.Series('Blah'), 'Average Analysis Time Per Row [ms]':pd.Series([1])}).set_index('Experiment Folder Name')
#%%
av_duration_per_row_df
#%%
