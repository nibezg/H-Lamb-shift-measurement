# Functions for helping to perform the data analysis for the Lamb shift experiment.
import numpy as np
import pandas as pd
import scipy
import sys
import os
import re
import numpy.fft
import time
import scipy.fftpack
import scipy.special
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import math
import datetime
# Package for wrapping long string to a paragraph
import textwrap
# For arbitrary precision arithmetic
import decimal
import fractions
#%%
class FosofAnalysisError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return textwrap.fill(str(self.value), break_long_words=False)

def get_trace_params(digi_trace_name):
    """
    Parses the digitizer trace binary name and returns a dictionary of the parameters recorded in it
    """
    # Regular expression
    reg_exp = re.compile( r"""r(?P<repeat>\d+)  # repeat number (positive integer)
                    a(?P<average>\d+)      # average number (positive integer)
                    f(?P<rf_frequency_MHz>[\d\.]+)  # frequency in MHz (positive float)
                    (of(?P<offset_frequency_Hz>[\d]+))* # Optional offset frequency in Hz (positive integer)
                    (Bx(?P<bx_Gauss>[\d\.-]+))* # Optional Bx value in Gauss (float)
                    (By(?P<by_Gauss>[\d\.-]+))* # Optional By value in Gauss (float)
                    ch(?P<rf_channel>A|B) # To which RF system the offset frequency is applied (A or B)
                    (_910(?P<pre_910_state>on|off))*  # Optional state of pre-910 cavity (on or off)
                    (_mf(?P<mf_setting>[\d]+))*  # Optional mass flow controller set point for the Charge Exchange (integer, which is the index of the sorted list of the setpoints used for the given experiment.)
                    _(?P<digi_channel>\d+).digi.npy""", # Digitizer channel (positive integer)
                    re.VERBOSE)
    digi_trace_parsed_object = reg_exp.search(digi_trace_name)

    # If the match has been found
    if digi_trace_parsed_object:
        trace_params_dict = {
            'repeat': int(digi_trace_parsed_object.group('repeat')),
            'average': int(digi_trace_parsed_object.group('average')),
            'RF frequency [MHz]': float(digi_trace_parsed_object.group('rf_frequency_MHz')),
            'RF channel': digi_trace_parsed_object.group('rf_channel'),
            'Digitizer channel': int(digi_trace_parsed_object.group('digi_channel')),
            'Pre-910 state [On/Off]': digi_trace_parsed_object.group('pre_910_state'),
            'Mass Flow Controller Set Point [int]': digi_trace_parsed_object.group('mf_setting')
            }

        # Type conversion of optional parameters

        if digi_trace_parsed_object.group('offset_frequency_Hz') != None:
            trace_params_dict["Offset frequency [Hz]"] = int(digi_trace_parsed_object.group('offset_frequency_Hz'))

        if digi_trace_parsed_object.group('bx_Gauss') != None:
            trace_params_dict["Bx [Gauss]"] = float(digi_trace_parsed_object.group('bx_Gauss'))

        if digi_trace_parsed_object.group('by_Gauss') != None:
            trace_params_dict["By [Gauss]"] = float(digi_trace_parsed_object.group('by_Gauss'))

        if digi_trace_parsed_object.group('mf_setting') != None:
            trace_params_dict['Mass Flow Controller Set Point [int]'] = int(digi_trace_parsed_object.group('mf_setting'))

    else:
        trace_params_dict = np.nan
    return trace_params_dict

def trace_in16_to_float(trace_array, channel_range):
    ''' Converts digitizer trace in int16 representaion to voltage values

        Inputs:
        :trace_array: numpy array in signed int16 representation
        :channel_range: +- range of the digitizer channel

        Outputs:
        :trace_V_array: numpy array in np.float64 representation containing trace votlages
    '''
    n_digitizer_bits = 16 # number of bits
    n_digitizer_val = 2**(n_digitizer_bits) # number of possible digitized samples from the digitizer

    # Convert to np.float64 type. This is very important.
    trace_array = trace_array.astype(np.float64)

    # Perform the conversion
    # Digitizer channel range is between + and - values.
    trace_V_array = 2*channel_range * trace_array / n_digitizer_val

    return trace_V_array

def import_digi_trace(digi_trace_name, channel_range):
    ''' Imports int16 digitizer trace and returns its voltage values with the trace parameters contained in the trace file name

    Inputs:
    :digi_trace_name: file name of the binary file in .npy format. Has to contain numpy array in signed int16 representation.
    :channel_range: +- range of the digitizer channel

    Outputs:
    :trace_params: dictionary of the trace parameters contained in the trace file name
    :trace_V_array: numpy array in np.float64 representation containing trace votlages
    '''

    # Load the ditizer trace in .npy format
    trace_array = np.load(digi_trace_name)
    trace_params = get_trace_params(digi_trace_name)
    trace_V_array = trace_in16_to_float(trace_array, channel_range)

    return trace_params, trace_V_array

def get_Fourier_spectrum(data_arr, sampling_rate):
    ''' Calculates spectrum of data array in np.float64 format and outputs amplitude and phase information of the Fourier coefficients

    Phase of the Fourier coefficients is given, assuming that it should represent nth Fourier coefficient given by
        ampl_n * cos(omega_n*t+phase)
    The spectrum is computed by using scipy.fftpack module

    Inputs:
    :data_arr: numpy array of samples in np.float64 format
    :sampling_rate: sampling rate of the data [Samples/s]

    Outputs:
    :spectrum_data: pandas DataFrame containing Fourier frequencies [Hz] and respective Fourier amplitudes and phases. The phases are given in radians and coverted to [0, 2pi) range.
    '''
    # Seems that people recommend using scipy.fftpack, due to scipy using more efficient algorithms in general than numpy.fft. Also I noticed that these two modules give slightly DIFFERENT spectra.

    # Calculate FFT of the real data.
    data_fft = scipy.fftpack.rfft(data_arr)

    # For some reason scipy.fftpack.rfft returns spectrum by separately listing real and imaginary components of the Fourier coefficients. To make it easier to work with the array we reconstruct it in such a way that it has the same form as the output array of numpy.fft.rfft function.

    if data_fft.size % 2 == 0:
        # If there is even number of elements in the spectrum array, then since the 0th element is the zeroth Fourier component = DC value, then it means that last Fourier component has no imaginary part, i.e., it is zero. For making it easier to reconstruct the array, we prepend this zero imaginary component.
        data_fft = np.append(data_fft,0)

    # Reshape the array to have real and imaginary components of the respective Fourier coefficient in one row. We, of course, discregard the first element on the spectrum (DC component)
    reshaped_data_fft = data_fft[1:].reshape((data_fft.size-1)//2, 2)

    # Combine real and imaginary components to construct single complex number. We also prepend the DC component.
    complex_data_fft = np.insert(
        reshaped_data_fft[:,0] + 1j * reshaped_data_fft[:,1],
         0, data_fft[0])

    # Number of samples in the data
    n_samples = data_arr.size

    # Step size [s]
    dt = 1.0 / sampling_rate

    phase_fft = convert_phase_to_2pi_range(np.angle(complex_data_fft))

    # This function outputs floating point numbers. It can be a problem later on for the following reason. Let's say we have 6 second long trace. 1/6 = 0.1666666666... Hz. Thus there is not enough precision to properly represent this number. Let's say the offset frequency is 800 Hz. If we want to find if 1/6 is multiple integer of 800 Hz, then symbolically 800/(1/6) = 4800, thus it is the integer multiple of df. However, using floating point numbers: 800/(1/6) = 800/0.1666666666.. = some non integer number.
    freq_fft = numpy.fft.rfftfreq(n_samples, dt)

    ampl_fft = 2 * np.abs(complex_data_fft) / n_samples

    # DC component.
    ampl_fft[0] = ampl_fft[0] / 2
    phase_fft[0] = np.nan

    spectrum_data = pd.DataFrame({  'Fourier Frequency [Hz]': freq_fft,
                                    'Fourier Amplitude [Arb]': ampl_fft,
                                    'Fourier Phase [Rad]': phase_fft},
                                )
    return spectrum_data

def get_file_n_lines(file_name):
    ''' Obtain number of lines in the filename

        Inputs:
        :file_name: name of the file (must exist in the current working directory)

        Returns:
        :num_lines: number of lines (integer) in the file
    '''
    # Open the data file in the read only mode to prevent accidental writes to the file
    file_object = open(file_name, mode='r')

    # Number of lines in the file. This is monitor if the end of the file is reached
    num_lines = sum(1 for line in file_object)
    file_object.close()
    return num_lines

def get_experiment_params(data_file_name):
    ''' Extract experiment parameters from the data file

        The parameters are converted to their respective type (int, float, or string) automatically.

        Inputs:
        :data_file_name: name of the data file (txt)

        Outputs:
        :exp_params_dict: dictionary with the experiment parameters
        :comment_string_arr: array of comment strings.
    '''
    # Regular expression for the comment lines
    # Notice the [^=] = we want to match the comment_name to any character, except '='. This is to ensure that in case of several '=' symbols, we are not including longer than needed string into the comment_name, since the first '=' separates the comment_name from the comment_value
    reg_exp = re.compile(r"""
                            [#]
                            (?P<comment_name>[\S\s][^=]+)
                            =
                            (?P<comment_value>[\S\s]+)$ # Match up to the newline character
                        """
                            , re.VERBOSE)

    # Number of lines in the file. This is monitor if the end of the file is reached
    num_lines = get_file_n_lines(data_file_name)

    # Open the data file in the read only mode to prevent accidental writes to the file
    data_file_object = open(data_file_name, mode='r')

    # Read lines from the file from the beginning of the file until the line does not start with the '#' character

    # Dictionary to store experiment parameters
    exp_params_dict = {}
    line_index = 0
    comments_end_bool = False
    comment_string_arr = []
    while comments_end_bool == False or line_index < num_lines:
        read_line = data_file_object.readline()
        line_index = line_index + 1

        # If we have empty line = '\n', then it is simply ignored. If we did not include this 'if' statement, then trying to perform .lstrip()[0] will result in the error, since '\n'.lstrip() = '' string has zero size = no elements.
        if read_line != '\n':
            if read_line.lstrip()[0] == '#':
                comment_string_arr.append(read_line)
                read_line = read_line.lstrip()
                # Get experiment parameter and name from the file line. Whitespace characters are stripped from left and right ends of the string
                comment_line_parsed_object = reg_exp.search(read_line)
                comment_name = comment_line_parsed_object.group('comment_name').strip()
                comment_value = comment_line_parsed_object.group('comment_value').strip()

                # Converting parameter values to proper type. First we try to convert the value to integer. If it fails, then we try to convert to float. If this also fails, then the value is a string. However, we can also have booleans - we check for that as well.
                # If the parameter name is 'Configuration', then at '0' configuration we do not want '0' to turn into integer 0.
                # If the parameter name is 'Offset Frequency [Hz]', then it is possible for having comma separated integer offset frequencies. Separate analysis
                if comment_name != 'Configuration':
                    if comment_name == 'Offset Frequency [Hz]':
                        comment_value = [int(i) for i in comment_value.split(',')]
                    else:
                        try:
                            comment_value = int(comment_value)
                        except ValueError:
                            try:
                                comment_value = float(comment_value)
                            except ValueError:
                                comment_value = str(comment_value)
                                if comment_value == 'True':
                                    comment_value = True
                                if comment_value == 'False':
                                    comment_value = False
                # Appending the experiment parameter to the dictionary
                exp_params_dict[comment_name] = comment_value
            else:
                # It is possible that the comments are not necesserily in the beginning of the file, thus we do not want to stop the search for the comments right away.
                if len(exp_params_dict) > 0:
                    comments_end_bool = True

    # Close the handle to the file
    data_file_object.flush()
    data_file_object.close()
    return exp_params_dict, comment_string_arr


def get_Fourier_harmonic(spectrum_data, f_arr, ac_harm=True, trace_length=None):
    ''' Gives amplitude and phase of the requested Fourier frequency and ratio of the Fourier amplitude due to the signal to the noise Fourier amplitude.

    The calculation of noise is performed between two adjacent AC harmonics, value of which is specified in the function. To determine noise amplitude, we make several assumptions:
    1. Total noise is composed of many noise types in such a way that the amplitude of each noise type is constant in time and in Fourier frequency around the frequency of interest, but phase, of course, is completely random. This way the total average noise amplitude is the RMS sum of the total noise amplitudes at the Fourier frequencies around the frequency of interest

    2. Total ampltide at the frequency of interest is taken as the average total amplitude at that frequency. This way it is equal to root-squared-sum of noise and signal amplitudes (which are assumed to be constant)

    With these two assumptions we can calculate the SNR. Of course, for proper calculation we need to obtain many traces to get the statistical distribution of the total amplitude at the frequency of interest. However, it is better than nothing.

    Inputs:
    :spectrum_data: pandas DataFrame containing the trace spectrum from  get_Fourier_spectrum functions
    :f_arr: Array of Fourier frequencies for which the spectrum parameters are to be returned
    :ac_harm: Whether to worry about possible presence of AC harmonics. It will basically not raise an exception if the AC harmonic is not an integer multiple of 1/T=delta_f = f0. This flag can be set to False whenever one knows that there are not significant AC harmonics that can pollute the Fourier phasor at the frequency of interest.

    Returns:
    :fourier_params_frame: pandas.DataFrame object containing spectrum parameters at the for all of the frequencies
    '''

    # delta f = f0 = first harmonic of the Fourier spectrum
    f0 = spectrum_data.iloc[1]['Fourier Frequency [Hz]']-spectrum_data.iloc[0]['Fourier Frequency [Hz]']

    # It is possible for f0 to be a rational number. However, if f0 = 1/9 exactly, then the decimal package below does not represent it exactly as a rational. We need additional package: 'fractions' for that. It becomes quite cumbersome at the end to make sure that everything is represented exactly.

    # Here we sort of do a dumb trick: I first represent use 1/f0 to get back to the length of the data in time domain. After I represent it as a decimal just in case. After I convert this number to a fraction and then switch denominator with numerator to get at the end rational representation of the f0.

    # Unfortunately it is still possible that f0 itself is not represented correctly. For instance, f0 gets first calculated using numpy.rfft function that takes dt = time spacing between samples as one of the arguments. The problem is that dt itself cannot be represented exactly. For instance, if dt = 100000/600000 (100000 samples/s with 600000 samples), then dt is not 0.1 exactly, but some other number. Because of this, 1/f0 is not what one would think. There is nothing else I can come up with to deal with this problem. The only way to resolve this issue is to have the length of the trace as one of the parameters

    if trace_length is None:
        f0_fract = 1/fractions.Fraction(decimal.Decimal(str(1/f0)))
    else:
        # fractions.Fraction(trace_length) is for the case when trace_length is not an integer, but a float.
        f0_fract = fractions.Fraction(1, fractions.Fraction(trace_length))

    # First AC harmonic frequency [Hz]. Note that strictly AC line frequency is not exactly 60 Hz. It is, however, within +- 0.05 Hz from it - was determine experimentally by taking 100 second long traces.y
    ac_f0 = 60
    ac_f0_fract = fractions.Fraction(decimal.Decimal(str(ac_f0)))

    fourier_params_frame = pd.DataFrame([])
    for f in f_arr:
        # f % f0 can give incorrect answers, when, for example f0=0.1, because in floating point representation it is not exactly equal to 0.1. Because of that It is possible that, for instance, 625 % 0.1 is not 0, but some number a bit less than 1. To avoid problems like that we use the decimal package that can represent numbers exactly. Notice that we convert the floats to strings first - so that we are not putting the float representation of the number into the Decimal class.
        #print(decimal.Decimal(str(f)) % decimal.Decimal(str(f0)) != 0)

        f_fract = fractions.Fraction(decimal.Decimal(str(f)))

        if (f_fract % f0_fract).numerator != 0:

            raise FosofAnalysisError("Requested frequency, " + str(f) + " Hz, is not a Fourier harmonics of the signal. Amplitude and phase cannot be reliably determined, since the power at this frequency gets distributed between nearby Fourier harmonics.")

        if ac_harm:
            if (ac_f0_fract % f0_fract).numerator != 0:
                raise FosofAnalysisError("AC line frequency is not a Fourier harmonic of the signal. This may cause spectral leakage into the frequency of interest")

        if (f_fract % ac_f0_fract).numerator == 0:
            raise FosofAnalysisError("Requested frequency, " + str(f) + " Hz, is the harmonic of the AC line frequency. The phase at the offset frequency cannot be reliably determined")

        if f > (spectrum_data.shape[0]-1)*f0:
            raise FosofAnalysisError("Requested frequency, " + str(f) + " Hz, is larger than the Nyquist frequency.")

        # Calculate averaged total noise amplitude

        # Calculate closest AC harmonics to the left and to the right of the offset frequency.

        # // = remainder. Note that when f < ac_f0, then f // ac_f0 is zero. Therefore f_ac_min = f0. This is exactly what we want, because in that case we skip the DC component of the Fourier spectrum and get all the noise components between f0 and ac_f0-f0. However, if f = f0, then this fails, since we get that f_ac_min = 1. This special case, therefore, has to be handled separately (done below)
        f_ac_min = ac_f0 * (f // ac_f0) + f0
        f_ac_max = (f_ac_min - f0) + ac_f0 - f0

        # Calculate corresponding indeces to access the required Fourier components
        f_index = int(f/f0)
        ac_min_ind = np.ceil(f_ac_min/f0).astype(int)
        ac_max_ind = np.floor(f_ac_max/f0).astype(int)

        # Noise Fourier amplitudes to the left and right of the frequency of interest.
        noise_ampl_left_arr = spectrum_data.loc[ac_min_ind:f_index-1, 'Fourier Amplitude [Arb]'].values

        # Making sure the the next AC harmonic index is not larger than the Nyquist frequency = maximum Fourier frequency. If it is, then the noise amplitudes are selected to the maximum Fourier frequency.
        if ac_max_ind <= spectrum_data.shape[0]:
            noise_ampl_right_arr = spectrum_data.loc[f_index+1:ac_max_ind, 'Fourier Amplitude [Arb]'].values
        else:
            noise_ampl_right_arr = spectrum_data.loc[f_index+1:, 'Fourier Amplitude [Arb]'].values

        # Combine left and right noise amplitude arrays

        # Taking care of the case when f = f0. I.e., when there are no noise frequencies to the left of the frequency of interest
        if f != f0:
            noise_amplitude_arr = np.concatenate((noise_ampl_left_arr, noise_ampl_right_arr), axis=0)
        else:
            noise_amplitude_arr = noise_ampl_right_arr

        # Average total noise amplitude (that can be a combination of many noise types (with flat spectrum in the window of interest (about 60 Hz)) is the RMS of the observed noise amplitudes at a range of Fourier frequencies )
        noise_ampl = np.sqrt(np.mean(noise_amplitude_arr**2))
        fourier_params = spectrum_data.iloc[f_index]

        # Calculate SNR. Sometimes, especially when looking at the Fourier harmonics where there is only noise, the resulting total averaged noise amplitude can be larger than the total amplitude at this Fourier frequency. In this case, SNR cannot be computed, because it becomes complex. We use np.nan to represent these cases.

        snr_squared = (fourier_params['Fourier Amplitude [Arb]']/noise_ampl)**2 - 1
        if snr_squared < 0:
            snr = np.nan
        else:
            snr = np.sqrt(snr_squared)

        snr_series = pd.Series([snr, noise_ampl], index=['SNR', 'Average Total Noise Fourier Amplitude [Arb]'])

        fourier_params = fourier_params.append(snr_series)
        fourier_params_frame = fourier_params_frame.append(fourier_params, ignore_index=True)
    return fourier_params_frame

def get_experiment_start_time(exp_folder_name):
    ''' Extracts the time of the acquisition initialization from the folder name.

    Inputs:
    :exp_folder_name: [string] It is assumed that the input string (folder name) has the following format: %y%m%d-%H%M%S

    Returns:
    pandas Timestamp object
    '''
    # Regular expression for extracting time of the data acquisition start from the folder name. This can be changed later: we might want to record the start time as additional field in the comments of the file. Or, possibly, we can access the creation date of the file (possibly dangerous, if the file has been copied after)

    date_reg_exp = re.compile( r"""(?P<date1>\d{6,6})    # first time portion
                            -(?P<date2>\d{6,6})             # second time portion
                            """,
                            re.VERBOSE)
    date_string = date_reg_exp.search(exp_folder_name).group()

    experiment_start_time_object = datetime.datetime.strptime(date_string,'%y%m%d-%H%M%S')

    return pd.Timestamp(experiment_start_time_object)

def convert_phase_to_2pi_range(phase_arr):
    ''' Converts np.array of phases, possibly in [-2pi, 2pi] range (or any other range) to [0, 2pi) range
    '''
    return (phase_arr + 2*np.pi) % (2*np.pi)

def mean_phasor(amp_arr, phase_arr):
    '''Calculates mean phasor, given list of amplitudes and corresponding list of phases for the respective phasor.

    It is assumed that the list of phasors is represented as A*np.cos(phi), where A is the phasor amplitude and phi is the phase. The output is the average phasor with average amplitude, A_av, and average phase, phi_av, such that the phasor has the form A_av*np.cos(phi_av). Standard deviations are also calculated. However, we are NOT calculating standard deviation of the mean. If one wants to calculate this quantity, then the size of the set of the arrays is included in the output dictionary.

    Inputs:
    :amp_arr: np.array of phasor amlitudes.
    :phase_arr: np.array of phasor phases.

    Outputs:
    :av_phasor: dictionary with the mean phasor parameters.
    '''

    x_arr = amp_arr * np.cos(phase_arr)
    y_arr = amp_arr * np.sin(phase_arr)
    y = np.mean(y_arr)
    x = np.mean(x_arr)
    sigma_y = np.std(y_arr, ddof=1)
    sigma_x = np.std(x_arr, ddof=1)
    av_amp = (x**2 + y**2)**0.5
    av_amp_std = 1/av_amp*np.sqrt((x*sigma_x)**2+(y*sigma_y)**2)

    phase_av = convert_phase_to_2pi_range(np.arctan2(y,x))
    phase_av_std =  np.abs(x*y)/(av_amp**2)*np.sqrt((sigma_x/x)**2+(sigma_y/y)**2)
    # COVARIANCE CANNOT BE CALCULATED, SINCE WE HAVE ONLY SINGLE MEASUREMENTS, AND THUS THE COVARIANCE IS AUTOMATICALLY ZERO. SEE P. 41 OF ERROR ANALYSIS NOTES.
    ## Calculating the covariance of the phases. This is mainly done, because I am not sure if the sine and cosine coefficients of the Fourier series are correlated.
    #phase_av_cov = 1/x_arr.size*np.sum((x_arr-x)*(y_arr-y))

    ## This expression was obtained with the help of Mathematica.
    #phase_av_std_with_cov = phase_av_std - 2*y/((x**3)*(1+(y/x)**2)**2)*phase_av_cov

    #av_phasor = {'Amplitude':av_amp, 'Amplitude STD':av_amp_std, 'Phase [Rad]':phase_av, 'Phase STD [Rad]':phase_av_std, 'Phase STD With Covariance [Rad]':phase_av_std_with_cov, 'Number Of Averaged Phasors':x_arr.size}

    av_phasor = {'Amplitude':av_amp, 'Amplitude STD':av_amp_std, 'Phase [Rad]':phase_av, 'Phase STD [Rad]':phase_av_std, 'Number Of Averaged Phasors':x_arr.size}
    return av_phasor

def phases_shift(phase_arr):
    ''' Useful for averaging phases. It attempts to convert phases given in [0, 2*np.pi) range to a continuous quantity that has a range of less than 2*np.pi, but with each phase not anymore restricted to the abovementioned range.

    The assumption is that firstly the set of phases given is in the [0, 2*np.pi) range. I.e., the phases are modular. Another assumption is that the real phase, which is the quantity that does not suffer from modularity (and hence ranging from -infinity to +infinity), that due to imperfect/improper phase measurement technique is mapped only to [0, 2*np) range is covering the range of < 2*np.pi. I.e., max{phase} - min{phase} < 2*np.pi. Third assumption is that to find the set of real phases = shifted phases (up to some common factor of 2*np.pi*k, where k is integer) we assume that the spread (standard deviation) of the phases has to be minimal. Thus, shifted phases that result in the smallest spread are assumed to be the correct choice.

    Inputs:
    :phase_arr: numpy array of phases in the [0, 2*np.pi) range [Rad] in np.float64 type.

    Returns:
    :phase_arr_shifted_original_order: numpy array of shifted phases in the original order.
    '''
    # The algorithm is described in Lab notes #3 p.151, written on April 24-26, 2018.

    # This is the array to store the initial ORDER of the array of phases. This is very important, because later we want to output the shifted phases in the same order as the initial phases. This is needed for cases when we care about the order of the returned phases. For instance, for the pd.DataFrame.transform operation. In my case, when I was not caring about the order of the output phases, it caused subtle bugs.

    phase_arr_initial_order = np.argsort(phase_arr)
    # We first sort the phases in the increasing order
    phase_arr = np.sort(phase_arr)
    N = phase_arr.size

    phase_arr_mean = np.mean(phase_arr)
    phase_arr_std = np.std(phase_arr)

    # This was used before for testing purposes. It constructs all possible choices of dk vector (without taking care of phases covering < 2*np.pi range.)
    #dk_combination_array = np.array(list(itertools.product([0,1], repeat=phase_arr.size)))

    # All VALID possible choices of to which index of the phase array dk=-1 can to be applied.
    dk_combination_array = np.tril(np.ones([N, N]))

    # Instead of making the last choice as dk applied to every elements of the phase array, we simply make it that no dk is applied to the array, which is the same thing, but results in the shifted phases still in the [0, 2*np.pi) range, if this choice turns out to have the smallest standard deviation.
    dk_combination_array[N-1] = dk_combination_array[N-1]*0

    # Array to keep the standard deviations calculated for different choices of dk vector
    std_array = np.zeros(dk_combination_array.shape[0])

    # For every vector dk we calculated the standard deviation that would result.
    for dk_index in range(dk_combination_array.shape[0]):
        k_arr = dk_combination_array[dk_index]
        phase_selected_arr = phase_arr[np.array(k_arr, dtype=np.bool)]
        n = phase_selected_arr.size
        if n == 0:
            phase_selected_arr_mean = 0
        else:
            phase_selected_arr_mean = np.mean(phase_selected_arr)

        # The formula is derived on page 156
        std_array[dk_index] = np.sqrt(phase_arr_std**2 + 4*np.pi*n/N*(phase_selected_arr_mean-phase_arr_mean+np.pi*(1-n/N)))

    # Find the dk vector resulting in smallest standard deviation and construct the shifted phases.

    # Sometimes it is possible for the set of phases to be such that there are multiple ways to pick vector dk to minimize the standard deviation. In this case all of the possible choices are listed.
    phase_arr_shifted = phase_arr + 2*np.pi*dk_combination_array[std_array == np.min(std_array)]

    phase_arr_shifted_original_order = np.zeros(phase_arr_shifted.shape)
    phase_arr_shifted_original_order[:,phase_arr_initial_order] = phase_arr_shifted

    return phase_arr_shifted_original_order

def straight_line_fit_params(data_arr, sigma_arr):
    ''' Calculates the weighted average of the data, given its uncertainty. The reduced chi-squared is calculated, where we are testing how well the data to average fits a straight line, which is the weighted average.

    Inputs:
    :data_arr: np.array of the data set
    :sigma_arr: np.array of uncertainties of the data set.

    Returns:
    pd.Series object containing the statistical infromation
    '''
    # Only one constraint: the data is used to calculate the weighted mean of the values.

    n_constraints = 1
    # Calculate the weighted average of the data

    weights_arr = 1/sigma_arr**2

    weighted_sigma = 1/np.sqrt(np.sum(weights_arr))

    expected_arr = np.average(data_arr, weights=weights_arr)*np.ones(data_arr.shape)

    # Degrees of freedom for the chi squared calculation
    dof = data_arr.shape[0] - n_constraints

    # Chi-squared calculated makes sense only if dof > 0:
    if dof >= 1:
        # Reduced chi-squared. I tried using the scipy.stats function for calculating this, however it seems to give me a different answer for some reason. Not sure what is the issue with it.
        chi_squared_reduced = np.sum(((data_arr-expected_arr)/sigma_arr)**2)/dof

        # Probability of obtaining chi-squared larger than the calculated value = Survival function = 1 - cdf for the chi-squared distribution, where x is the non-reduced chi squared and df = degrees of freedom
        prob_large_chi_squared = scipy.stats.chi2.sf(x=chi_squared_reduced*dof, df=dof)
    else:

        chi_squared_reduced = np.nan
        prob_large_chi_squared = np.nan

    return pd.Series({'Weighted Mean': expected_arr[0], 'Weighted STD': weighted_sigma, 'Reduced Chi Squared': chi_squared_reduced, 'P(>Chi Squared)': prob_large_chi_squared})

def get_krytar_109B_calib(min_det_v, max_det_v, rf_frequency_MHz = 910):
    '''Given RF frequency cubic spline is given for the corresponding calibration data for KRYTAR 109B RF power detector. This cubic spline will allow to determine detected power given voltage reading.

    Note that the power calibration curve was obtained with the IFR 2026B RF generator. Its power output is not perfect (check the specs). However it is useful to use this calibration to monitor relative power stability of the RF system.

    Inputs:
    :min_det_v, max_det_v: range of voltages from the RF power detector. The cubic spline will work only for this particular range.
    :rf_frequency_MHz: RF frequency for which to give the calibration. Given in MHz. Note that this should always be set to 910 (MHz) for now, since the calibration data exists only for this particular RF frequency.
    '''

    # We suspect that either the generator or amplifiers might not output the same RF power all the time. Thus we want to monitor the change in the output of the power detectors vs time. The power detectors are in the temperature stabilized enclosure and thus we assert that the change in power read from the power detectors corresponds to the same fractional power change of the power inside the respective waveguides. We are not, however, reading the power from the power detectors but voltage. However, we have calibrated this voltage to the power using the RF generator (IFR2026B) that we use to run the experiment.

    # For every RF frequency we expect the power reading to be different, because in general we put different powers at different RF frequencies so that the RF Power inside the waveguide stays the same for every frequency. Thus one can assume that we need to have calibration curves for every RF frequency.
    # Power is measured by KRYTAR 109B power detectors. They can be used for up to 18 GHz. We suspect that if we stay at about +-100 MHz about 910 MHz, then the power detector will show the same voltage for each frequency for the same RF power level. This way we can use the calibration obtained for 910 MHz for all other RF frequencies.

    # Path to the power detector calibration
    # In the lab
    krytar109B_pwr_det_calib_folder_path = 'C:/Users/Helium1/Google Drive/Research/Lamb shift measurement/Data'
    # At home
    #krytar109B_pwr_det_calib_folder_path = 'E:/Google Drive/Research/Lamb shift measurement/Data'
    # KRYTAR 109B Power detector calibration folder name
    krytar109B_pwr_det_calib_folder = '170822-130101 - RF power detector calibration'
    krytar109B_pwr_det_calib_filename = 'KRYTAR 109B power calibration.CSV'

    os.chdir(krytar109B_pwr_det_calib_folder_path)
    os.chdir(krytar109B_pwr_det_calib_folder)

    # Import the CSV file with the detector power calibration taken at 910 MHz.
    pwr_det_calib_df = pd.read_csv(filepath_or_buffer=krytar109B_pwr_det_calib_filename, delimiter=',', comment='#', header=0)

    # The detector is useful only to measuring powers above about -30 dBm. The calibration that we performed went all the way down to -50 dBm, which gave extremely small voltages that are almost the same for -50 to -40 dBm - they seem to be simply the offset of the Keithley multimeter. I consider this data as unreliable and throw it away. In any case we do not measure powers that small in the experiment.
    pwr_det_calib_df = pwr_det_calib_df[pwr_det_calib_df['RF power [dBm]'] >= -30]

    # We select only the calibration data that corresponds to the range of voltage measured from the RF power detectors in the actual experiment.
    pwr_det_calib_df_plot = pwr_det_calib_df[(pwr_det_calib_df['RF power [dBm]'] <= max_det_v) & (pwr_det_calib_df['Power detector signal [V]'] >= (min_det_v))]

    pwr_det_calib_to_use_df = pwr_det_calib_df[(pwr_det_calib_df['Power detector signal [V]'] <= max_det_v) & (pwr_det_calib_df['Power detector signal [V]'] >= (min_det_v))]

    # Just to be on a safe side we also add data that at the powers that are extending the range of powers by +- 0.4 dBm
    max_gen_power = pwr_det_calib_to_use_df['RF power [dBm]'].max()
    min_gen_power = pwr_det_calib_to_use_df['RF power [dBm]'].min()

    pwr_det_calib_to_use_df = pwr_det_calib_df[(pwr_det_calib_df['RF power [dBm]'] <= max_gen_power+0.4) & (pwr_det_calib_df['RF power [dBm]'] >= min_gen_power-0.4)]

    # We use cubic spline to interpolate between the data point so that we can convert power detector voltages to corresponding RF powers. The calibration itself has several visible discontinuities in it. This happens when the RF generator switches to a different attenuator at certain output power levels. In a way this tells us that the RF generator does not output exactly correct power or that its power output is not linear across its dynamic range. It is not a problem for us, because we are not relying on the power detector calibration to determine the needed RF generator power output setting for required RF power inside the waveguides. It will still give us an idea of how stable the power in each waveguide is.
    det_calib_cspline_func = scipy.interpolate.interp1d(x=pwr_det_calib_to_use_df['Power detector signal [V]'].values, y=pwr_det_calib_to_use_df['RF power [dBm]'].values, kind='cubic')

    # Plotting the cubic spline and the calibration data. This is done to be able to visually see if there are any issues in the cubic spline and/or calibration data in the region of interest

    # Number of points to use for the cubic spline plot
    n_plot_points = 1000

    x_spline_data = np.linspace(min_det_v,
    max_det_v, n_plot_points)
    y_spline_data = det_calib_cspline_func(x_spline_data)

    return det_calib_cspline_func, [x_spline_data, y_spline_data], pwr_det_calib_to_use_df

def pooled_std(std_arr, n_av_points_arr):
    ''' Calculates unbiased pooled standard deviation, given array of standard deviations (can be 2D) and corresponding array of number of points used to find each standard deviation.

    Check https://en.wikipedia.org/wiki/Pooled_variance for more details.
    '''

    # Unbiased list of sums of squared deviations from mean.
    var_product_arr = (std_arr**2)*(n_av_points_arr-1)

    # Number of degrees of freedom
    dof = var_product_arr.shape[0]

    # Total number of standard deviations used minus the number of dof's used to calculate these standard deviations
    n_points_tot_corrected = np.sum(n_av_points_arr, axis=0) - dof
    return np.sqrt(np.sum(var_product_arr, axis=0)/n_points_tot_corrected)

def divide_and_minimize_phase(phase_arr, div_number):
    ''' Simple function that divides phases by positive div_number and represents phases in a format that minimizes the absolute value of the phase.

    Notice an interesting example: If we had phase difference of 2*pi-0.2 rad. Then dividing it by two results in pi-0.1 rad phase difference. Now. 2*pi - 0.2 = -0.2 rad. Dividing it by two gives -0.1 rad.

    For this phase difference I would want to keep it in this format, when the phase difference is such that its absolute value is minimized - easier for plotting and tracking the phase difference stability.

    But now we get two answers. Which one is the correct one? It seems that the second answer should be the right one. Where is the mistake then? The mistake is the following: for the first case: after dividing by two we have limited the range of possible phases from [0, 2pi) to [0, pi). Thus we cannot then shift the phases by 2*pi, but we should shift only by pi. Thus for the pi-0.1 rad we can represent it as pi-0.1-pi = -0.1 rad - same answer as the second case.

    We can easily generalize this to division by any positive real number. Interestingly, one can see that for modular arithmetic one should also have allowed range of values for every modular number and perform the arithmetic operations accordingly taking the range into account.
    '''
    phase_arr = phase_arr / div_number
    two_pi_fract = 2*np.pi/div_number
    phase_arr = phase_arr % two_pi_fract
    phase_arr[phase_arr > two_pi_fract/2] = phase_arr[phase_arr > two_pi_fract/2] - two_pi_fract

    return phase_arr

def correct_FOSOF_phases_zero_crossing(x_data_arr, phase_data_arr, phase_sigma_arr, zero_cross_freq=910, slope=-0.1*4):
    ''' Special function for FOSOF lineshape. Corrects for any 0 phase crossing discontinuities in the FOSOF lineshape. This is done by first specifying the approximate fit parameters for the data and then checking for the deviations of the data from the approximate fit and correcting the data accordingly.

    Inputs:

    :x_data: np.array for x-axis (Usually this is the array of RF frequencies used)
    :phase_data_arr: corresponding list of FOSOF phases in [0, 2*np.pi range) that has not been divided by 4 yet.
    :phase_sigma_arr: array of respective uncertainties in the phases.
    :zero_cross_freq: frequency of zero-crossing for the approximate fit [MHz].
    :slope: slope of the approximate fit line. Notice that I assume that the phase_data_arr has not been divided by 4 yet.

    Outputs:
    :phase_shifted_arr: np.array in the original order as the phase_data_arr with properly shifted phases
    '''

    # Correct for 0-2*np.pi jumps

    # Before I used this function to do so, however, it fails whenever the data is close to the 0-2pi jump. Because of the noise in the data, sometimes we might get several 0-2pi transitions at frequencies close to each other. The function would consider them as real jumps. This is problematic. Because of that I am using the phase_shift function. It will not work for all possible cases, unfortunately - whenever the scan range is such that the phases do actually go through more than 2pi. But we, as far as I know, do not have scan ranges that large here.

    # Later on I realized that the phases_shift will not work as well. This is due to scanning over large range, where now there might be actually no apparent 0-2pi jump in the data itself, but this is because the 0-2pi jump happens somewhere in the middle of the frequency range for which the data was not acquired. At the end I think the best way of doing this is to have some approximate fit parameters for the line and then correct the data based on these parameters. One problem is the fact that the slope might be not what I expect. For this I can try to fit with both the negative and positive slope and check which one gives smaller overall residuals from the fit.


    # # Sorting the data array by x-axis in the increasing order
    # data_arr = np.array([x_data_arr, phase_data_arr]).T
    # data_arr = data_arr[x_data_arr.argsort()]
    #
    # diff_arr = np.diff(phase_data_arr)
    #
    # # Counter for 2pi-0 jumps
    # n_rotation = 0
    #
    # # First phase element is never shifted
    # phase_shifted_arr = np.zeros(phase_data_arr.shape[0])
    # phase_shifted_arr[0] = phase_data_arr[0]
    #
    # # We now look for any jumps of more than pi radians in phase. Whenever a jump like that is detected, the counter gets incremented in the respective direction, which corresponds to adding 2*np.pi*n_counter to the current phase
    #
    # # We start with the second element in the array of phases
    # for i in range(phase_data_arr.shape[0])[1:]:
    #
    #     if diff_arr[i-1] > np.pi:
    #         # Whenever the jump is positive (current phase is larger than the previous one by more than pi, then it means that at this x-axis value the FOSOF slope is negative, and thus we should subtract 2*np.pi from this point relative to the previous to make the lineshape continuous)
    #         n_rotation = n_rotation - 1
    #     if diff_arr[i-1] < -3*np.pi/2:
    #         n_rotation = n_rotation + 1
    #     phase_shifted_arr[i] = phase_data_arr[i] + 2*np.pi*n_rotation
    #
    # phase_shifted_arr = phase_shifted_arr[x_data_arr.argsort()]

    # Approximate fit parameters for the data
    offset = -zero_cross_freq * slope

    approx_fit_data_arr = slope * x_data_arr + offset

    phase_data_shifted_arr = phase_data_arr - 2 * np.pi * np.round((phase_data_arr - approx_fit_data_arr) / (2*np.pi))

    fit_data_arr = np.poly1d(np.polyfit(x_data_arr, phase_data_shifted_arr, deg=1, w=1/phase_sigma_arr**2))(x_data_arr)

    slope_1_tot_res = np.sum(np.abs(fit_data_arr - phase_data_shifted_arr))
    phase_data_shifted_1_arr = phase_data_shifted_arr

    slope = -1 * slope
    offset = -zero_cross_freq * slope

    approx_fit_data_arr = slope * x_data_arr + offset

    phase_data_shifted_arr = phase_data_arr - 2 * np.pi * np.round((phase_data_arr - approx_fit_data_arr) / (2*np.pi))

    fit_data_arr = np.poly1d(np.polyfit(x_data_arr, phase_data_shifted_arr, deg=1, w=1/phase_sigma_arr**2))(x_data_arr)

    slope_2_tot_res = np.sum(np.abs(fit_data_arr - phase_data_shifted_arr))
    phase_data_shifted_2_arr = phase_data_shifted_arr

    if slope_2_tot_res <= slope_1_tot_res:
        phase_shifted_arr = phase_data_shifted_2_arr
    else:
        phase_shifted_arr = phase_data_shifted_1_arr

    return phase_shifted_arr

# Convenience functions for working with pandas
def match_string_start(array_str, match_str):
    ''' Returns boolean index for the array of strings, array_str, of whether the string match_str was a subset of the given string at its start.
    '''
    reg_exp = re.compile(match_str)
    matched_arr = np.array(list(map(lambda x: reg_exp.match(x), array_str))) != None

    return matched_arr

def remove_matched_string_from_start(array_str, match_str):
    ''' Removes first encounter (most left) of 'match_str' in the strings of the array_str. Each member of the array of strings, array_str, must contain at least one occurence of match_str.
    '''

    reg_exp = re.compile(match_str)

    # We split the column names to get the part that has no 'match_str' characters in it. We should only allow for single splitting to happen, to allow for the match_str string to be contained somewhere in the bulk of the string.

    stripped_column_arr = np.array(list(map(lambda x: reg_exp.split(x, maxsplit=1)[1], array_str)))

    # The leftover string has whitespace (most likely) as the first character for every split column name. We remove it.
    stripped_column_arr = list(map(lambda x: x.strip(), stripped_column_arr))

    return dict(np.array([array_str, stripped_column_arr]).T)


#-----------------------------
# Added on 2018-07-05. Completely rewrote the function to take care of the fact that some of the columns would change their dtype to generic object.
#-----------------------------

# def add_level_data_frame(df, level_name, level_value_arr):
#     ''' Adds a new level to otherwise flat (non MultiIndex) dataframe.
#
#     The level can have column names specified in an array. For each member of the array the columns are determined that have match in the beginning of the name with that particular array element. Column names that do not match any of the array elements are assigned value of 'Other'.
#
#     Note that this function works only on flat non multiIndex dataframe. Also, it changes the level name to 'index'. Rename it to needed one later.
#
#     Inputs:
#     :df: pandas DataFrame
#     :level_name: Name of the level to add
#     :level_value_arr: Sublevel column names
#     '''
#     df_T = df.T
#     df_T[level_name] = np.nan
#     for level_value in level_value_arr:
#         df_T.loc[df_T.index.values[match_string_start(df_T.index.values, level_value)], level_name] = level_value
#
#     # There are also indeces that do not correspond to any level_value.
#     df_T.loc[df_T[level_name].isnull(), level_name] = 'Other'
#     level_name_to_use = df_T.index.names[0]
#     df_T.index.names = ['index']
#
#     updated_df = df_T.reset_index().set_index([level_name, df_T.index.names[0]]).sort_index().T
#
#     for level_value in level_value_arr:
#         updated_df.rename(columns=remove_matched_string_from_start(updated_df[level_value].columns.values, level_value), level=1, inplace=True)
#
#     updated_df.columns.rename(level_name_to_use, level=1, inplace=True)
#     return updated_df

def add_level_data_frame(df, level_name, level_value_arr):
    ''' Adds a new level to otherwise flat (non MultiIndex) dataframe.

    The level can have column names specified in an array. For each member of the array the columns are determined that have match in the beginning of the name with that particular array element. Column names that do not match any of the array elements are assigned value of 'Other'.

    Inputs:
    :df: pandas DataFrame
    :level_name: Name of the level to add
    :level_value_arr: Sublevel column names
    '''
    partial_df_list = []
    partial_df_columns_list = []

    for level_value_index in range(len(level_value_arr)):

        level_value = level_value_arr[level_value_index]

        partial_df_columns = df.columns.values[match_string_start(df.columns.values, level_value)]

        partial_df = df[partial_df_columns]

        partial_df = partial_df.rename(columns=remove_matched_string_from_start(partial_df.columns.values, level_value))
        partial_df.columns.names = ['Data Field']

        partial_df = pd.concat([partial_df], keys=[level_value], names=[level_name], axis='columns')

        partial_df_columns_list.append(partial_df_columns)
        partial_df_list.append(partial_df)

    # Function to flatten a list
    flatten = lambda l: [item for sublist in l for item in sublist]

    # Find the columns that were not matched. It is possible for the partial_df_columns_list to have np.array with difference number of elements as the members of the list. Because of this, converting this list to np.array will not work. That is why I am first flattening the list itself and only after convert it to the np.array.

    non_matched_column_arr = np.setdiff1d(df.columns.values, np.array(flatten(partial_df_columns_list)).flatten())

    if non_matched_column_arr.shape[0] > 0:
        level_value = 'Other'

        partial_df_columns = non_matched_column_arr

        partial_df = df[partial_df_columns]

        partial_df.columns.names = ['Data Field']

        partial_df = pd.concat([partial_df], keys=[level_value], names=[level_name], axis='columns')

        partial_df_columns_list.append(partial_df_columns)
        partial_df_list.append(partial_df)
    return pd.concat(partial_df_list, axis='columns')


def group_apply_transform(grouped, func):
    ''' I either do not understand how the grouped.transform function works or there is a glitch in its implementation. I want to apply transformation operating to every group. When I do it thought using grouped.transform then it takes forever - the editor freezes. However, if I just iterate through groups and join the resulting transformed data frames together, then it takes little time. It is possible that the transform function somehow operated on each group (= a data frame), instead of operating on the columns of the group.

    Inputs:
    :grouped: pd.grouped object
    :func: lambda function to use as the transformation

    Outputs:
    :joined_df: combined dataframe with its elements transformed
    '''
    group_list = []
    joined_df = pd.DataFrame()
    for name, group in grouped:
        group_trasformed_df = group.transform(func).T
        if joined_df.size == 0:
            joined_df = group_trasformed_df
        else:
            joined_df = joined_df.join(group_trasformed_df)
    return joined_df

def get_chi_squared(data_arr, data_std_arr, fit_data_arr, n_constraints):
    ''' Convenience function. Calculates the Reduced Chi-Squared of the deviation between the original data and the fit function for the data and various parameters related to it.

    Inputs:
    :data_arr: np.array of original data
    :data_std_arr: one sigma uncertainty in the data_arr
    :fit_data_arr: np.array of data from some fit function.
    :n_constraints: number of constraints = parameters used in the fit function

    Outputs:
    :pd.Series object: Contains the value of the reduced Chi-Squared, its standard deviation, number of degrees of freedom (can be used to interpret the standard deviation in terms of probability), and the probability of having in statistical sense larger value of Chi-Squared.
    '''
    # Degrees of freedom for the chi squared calculation
    dof = data_arr.shape[0] - n_constraints

    # Reduced chi-squared. I tried using the scipy.stats function for calculating this, however it seems to give me a different answer for some reason. Not sure what is the issue with it.
    chi_squared_reduced = np.sum(((data_arr-fit_data_arr)/data_std_arr)**2)/dof

    # Probability of obtaining chi-squared larger than the calculated value = Survival function = 1 - cdf for the chi-squared distribution, where x is the non-reduced chi squared and df = degrees of freedom
    prob_large_chi_squared = scipy.stats.chi2.sf(x=chi_squared_reduced*dof, df=dof)

    return pd.Series({
        'Number Of Degrees Of Freedom' : dof,
        'Reduced Chi-Squared': chi_squared_reduced,
        'P(>Chi Squared)': prob_large_chi_squared,
        'Reduced Chi-Squared STD': 2*np.sqrt(dof)/dof
    })

# This function to extends the range of application of the aggregate method in pandas. What usually is the issue is that one wants to aggregate a dataframe, but not by using data from each column separately, but combining the data from columns together as the input to the aggregating function. I did not find a way to do this, using the aggregate function in pandas. A way around this is to use the apply function. But the function is quite slow. To get around the slowness of the function I made my own function that can take several columns as the input for the 'aggregate' method. Another difficulty is that sometimes the result of the aggregation is not only a single number, but a dictionary with the results of the aggregation. The function takes care of this as well.

def get_column_data_dict(x, col_list):
    ''' Forms a dictionary of the values belonging to each column. Simply for convenience, to make it easier to define the aggregating function.
    '''
    data = np.array(list(x.values))
    data_dict = {}
    for col_index in range(data.shape[1]):
        data_dict[col_list[col_index]] = data[:, col_index]
    return data_dict

def group_agg_mult_col_dict(df, col_list, index_list, func):
    ''' Grouped the data and aggregates it by using a function that takes data from several columns of the input dataframe and outputs a dictionary.

    Inputs:
    :df: pd.Dataframe as the input
    :col_list: list of columns used for the aggregate function
    :index_list: list of index name by which to group the dataframe.
    :func: function to use as the aggregating function. It must return a dictionary. The function must have the following line it it (as, probably, the first line):
    data_dict = get_column_data_dict(x)

    This basically creates a dictionary of values that can be used as the input to the function that return a dictionary.
    '''

    def to_single_col(df, col_list):
        ''' Combines data from several columns into a single column, with the elmement in each row of this column containing np.array of elements of all of the columns listed in col_list. Columns, except the first one, are removed form the dataframe.
        '''
        df[col_list[0]] = [np.array(x) for x in df[col_list].values]
        df.drop(columns=col_list[1:], inplace=True)
        return df

    df = to_single_col(df, col_list)

    df_grouped = df.groupby(index_list)

    agg_df = df_grouped.aggregate(lambda x: func(x, col_list))

    data = agg_df.values.flatten()
    data_list = np.array([list(data[i].values()) for i in range(data.shape[0])])

    return pd.DataFrame(data_list, columns=data[0].keys(), index=agg_df.index)

def remove_sublist(ini_list, remove_list):
    ''' Simple convenience function for removing sublists from a list.

    Inputs:
    :ini_lst: initial list from which one want to remove a sublist
    :remove_list: list of elements to remove
    '''
    return [x for x in list(ini_list) if x not in remove_list]

def get_range(data_arr, fract_range):
    ''' Useful function for setting the range of axis/axes for plotting.

        Sets the range to the range covered by the data + fract_range * range
    '''
    data_min = np.min(data_arr)
    data_max = np.max(data_arr)

    data_range = data_max-data_min
    return [data_min - fract_range*data_range, data_max + fract_range*data_range]

def reshape_axes(fig, axes, ncols, nrows):
    ''' Reshapes the otherwise flat array of axes.

    Very useful when one does not know beforehand the shape of the axes array needed. Notice, that the axes need to contain at least two elements
    '''
    gs = gridspec.GridSpec(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            k = i+j*nrows
            if k < len(axes):
                axes[k].set_position(gs[k].get_position(fig))
    return fig, axes
