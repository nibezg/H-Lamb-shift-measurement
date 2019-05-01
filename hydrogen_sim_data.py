from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

import pickle

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt

import copy

path_data_df = pd.read_csv(filepath_or_buffer='path_data.csv', delimiter=',', comment='#', header=[0], skip_blank_lines=True, index_col=[0])

code_folder_path = path_data_df.loc['Code Folder'].values[0].replace('\\', '/')
fosof_analyzed_data_folder_path = path_data_df.loc['FOSOF Analyzed Data Folder'].values[0].replace('\\', '/')
wvg_calib_data_folder_path = path_data_df.loc['Waveguide Calibration Folder'].values[0].replace('\\', '/')
krytar109B_pwr_det_calib_folder_path = path_data_df.loc['KRYTAR 109 B Power Detector Calibration Data Folder'].values[0].replace('\\', '/')
sim_data_folder_path = path_data_df.loc['Simulation Data Folder'].values[0].replace('\\', '/')

sys.path.insert(0, code_folder_path)

from exp_data_analysis import *
#%%
# Path to the folder that stores FOSOF simulation files
sim_data_path = sim_data_folder_path
#sim_data_path = 'E:/Google Drive/Research/Lamb shift measurement/Data/Simulation data'

# Old Simulation path that did not average phases between the quenching cavities and the SOF waveguides.
old_sim_folder_path = 'Old Lamb Shift Data (No Phase Averaging)'
# File that stores information about all of the old simulations
old_sim_info_file_name = 'FOSOF Simulations for Hydrogen.xlsx'

# New Simulation path with properly averaged phases between the quenching cavities and the SOF waveguides.
new_sim_folder_path = 'Lamb Shift Data (WITH Phase Averaging)'
# File that stores information about all of the old simulations
new_sim_info_file_name = 'New FOSOF Simulations for Hydrogen.xlsx'

class OldSimInfo():
    ''' Class for accessing information about the Old Hydrogen Simulation data. These simulations are 'Old', because there was no proper phase averaging between the Fields in the Waveguides and the Quench cavities.
    '''
    def __init__(self):

        os.chdir(sim_data_path)
        os.chdir(old_sim_folder_path)

        self.old_sim_xlsx = pd.ExcelFile(old_sim_info_file_name)
        self.old_sim_info_df = pd.read_excel(pd.ExcelFile(self.old_sim_xlsx), sheet_name='Sheet1', header=[0], index_col=[0])

    def get_info(self):
        return self.old_sim_info_df

class NewSimInfo():
    ''' Class for accessing information about the New Hydrogen Simulation data. These simulations are 'New', because there was proper phase averaging between the Fields in the Waveguides and the Quench cavities, as opposed to the 'Old' simulations.
    '''
    def __init__(self):

        os.chdir(sim_data_path)
        os.chdir(new_sim_folder_path)

        self.new_sim_xlsx = pd.ExcelFile(new_sim_info_file_name)
        self.new_sim_info_df = pd.read_excel(pd.ExcelFile(self.new_sim_xlsx), sheet_name='Sheet1', header=[0], index_col=[0])

    def get_info(self):
        return self.new_sim_info_df


quench_sim_file = 'SOFQuenchEfficiency.xlsx'

# After converting the delta f's to frequency we need to rename this column
freq_col_change_name = 'Waveguide Carrier Frequency [MHz]'

class WaveguideOldQuenchCurveSimulationSet():
    ''' Class containing all the functions required to work with the waveguide quench curve data.
    '''
    def __init__(self, old_quench_sim_info_s):

        self.old_quench_sim_info_s = old_quench_sim_info_s

        os.chdir(sim_data_path)
        os.chdir(old_sim_folder_path)

        # Folder path to the quenching simulation data
        self.quenching_sim_data_path = 'SOFQ/' + self.old_quench_sim_info_s['Simulation Key']

        os.chdir(self.quenching_sim_data_path)

        # Import the quenching data. The file contains several sheets that correspond to having atoms at a different distance away from the experiment axis. All of these sheets = different off-axis parameters, are stored in a dictionary.
        self.quench_sim_xlsx = pd.ExcelFile(quench_sim_file)

        self.quench_sim_dict = pd.read_excel(self.quench_sim_xlsx, sheet_name=None, header=[1], index_col=None)

        # Base frequency used = frequency offset
        self.base_rf_freq = self.old_quench_sim_info_s['Base Frequency [MHz]']

        if np.isnan(self.base_rf_freq):
            print('No base frequency has been found. Setting the base frequency to 910 MHz')
            self.base_rf_freq = 910

        # Excel sheet has the frequency column in unicode format
        self.freq_col_name = u'\u0394'+'f (MHz)'

    def get_list_of_simulations(self):
        # Pick the required DataFrame

        return list(self.quench_sim_dict.keys())

    def get_simulation_data(self, sim_name):
        quench_sim_vs_freq_df = self.quench_sim_dict[sim_name].copy()

        freq_col_name = self.freq_col_name
        # Some files have square brackets for 'MHz' instead of round ones
        if not(freq_col_name in quench_sim_vs_freq_df.columns):
            freq_col_name = u'\u0394'+'f [MHz]'

        # Shift the delta f's by their base frequency
        quench_sim_vs_freq_df[freq_col_name] = quench_sim_vs_freq_df[freq_col_name] + self.base_rf_freq

        quench_sim_vs_freq_df = quench_sim_vs_freq_df.set_index(freq_col_name)

        quench_sim_vs_freq_df.index.names=[freq_col_change_name]

        # Drop this column. It has nothing to do with the data analysis.

        if 'ALL OFF' in quench_sim_vs_freq_df.columns:
            quench_sim_vs_freq_df.drop(columns=['ALL OFF'], inplace=True)

        elif 'OFF' in quench_sim_vs_freq_df.columns:
            quench_sim_vs_freq_df.drop(columns=['OFF'], inplace=True)

        # We want to look at the data as a function of E**2, not as E.
        quench_sim_vs_freq_df.columns.names=['E Field Amplitude [V/cm]']

        quench_sim_vs_freq_df = quench_sim_vs_freq_df.T.transform(lambda x: x/x[0])

        quench_sim_vs_freq_df = pd.concat([quench_sim_vs_freq_df], keys=['Fractional Surviving Population'], names=['Data Field'], axis='columns').swaplevel(axis='columns')

        def transform_df(df):
            rf_freq = df.columns.get_level_values(0)[0]
            df = pd.concat([df[rf_freq]], axis='index', keys=[rf_freq], names=['Waveguide Carrier Frequency [MHz]'])
            return df

        quench_sim_vs_freq_transformed_df = None

        for rf_freq, df in quench_sim_vs_freq_df.groupby(level='Waveguide Carrier Frequency [MHz]', axis='columns'):
            if quench_sim_vs_freq_transformed_df is None:
                quench_sim_vs_freq_transformed_df = transform_df(df)
            else:
                quench_sim_vs_freq_transformed_df = quench_sim_vs_freq_transformed_df.append(transform_df(df))

        return quench_sim_vs_freq_transformed_df


class OldWaveguideQuenchCurveSimulation():
    ''' Class for managing a quenching curve for given RF frequency from the Old hydrogen simulations (No phase averaging).
    '''
    def __init__(self, quench_sim_df):

        self.quench_sim_df = quench_sim_df

        # We want to look at the data as a function of E**2, not as E. When we look at the quenching curve vs power, then it has linear trend for small power values, whereas if we look at it vs E field, then the relationship is more complicated: the curve is quite flat at small E fields. At the end, I would assume that it should not matter vs what we plot the data, because we should be able to convert between various representations without any information loss.
        #self.quench_sim_df.columns.names=['E Field Amplitude Squared [V^2/cm^2]']

        self.quench_sim_df.index.names = ['E Field Amplitude Squared [V^2/cm^2]']

        # Convert E to E**2
        self.quench_sim_df = quench_sim_df.reset_index()
        self.quench_sim_df['E Field Amplitude Squared [V^2/cm^2]'] = self.quench_sim_df['E Field Amplitude Squared [V^2/cm^2]']**2
        self.quench_sim_df = self.quench_sim_df.set_index('E Field Amplitude Squared [V^2/cm^2]')

        # Final calibration fit functions for extracting RF power from given surviving fraction
        self.spl_inverted = None

    def get_quench_curve_sim_data(self):
        return self.quench_sim_df

    def analyze_data(self):
        ''' Performs the analysis of the quenching curve.

        The analysis is explained in the comments in the body of the function
        '''

        # For each column = each RF frequency we want to look at the quench curve as a function of (proportional to) RF Power. Each of these quench curves need to be parameterized, so that when analyzing experiment quench data from the waveguides, where for each waveguide the RF generator power for the respective RF System was scanned, we can determine the respective RF Power in the waveguides. This can be done in the following way.
        # We pick a surviving fraction data point from the experiment data. We then determine to what RF power this surviving fraction value corresponds to on the quench curve from the simulation. Thus it means that we need to parameterize the simulation quench curve, that is not in the form of 'Surviving fraction vs RF Power', but in the form of 'RF Power vs Surviving Fraction'. This sounds quite easy to do. However, there is a problem. In general a quenching curve has several pi-pulses. Thus for some Surviving fraction values there are multiple corresponding RF powers. In other words, if we look a the simulation quench curve as 'RF power vs Suriving fraction' then the parameterizing function will not be single valued.
        # However, we are never interested in the simulation data beyond the first pi-pulse. Thus if we take only the first pi-pulse into account then there will be no issue with non-single-valuedness. Therefore to perform the analysis we need to do the following.
        # We follow similar steps as was done for the analysis of the quench curves from the quench cavities. We first fit the simulation data to a 4th-order spline. Then we find the derivate of the spline (turns it into a third-order spline) and from that we find the RF power that corresponds to the first pi-pulse. We then disregard all of the simulation data after the first pi-pulse. The data before the pi-pulse is now inverted and fit to a spline again, so that we now have 'RF Power vs Surviving Fraction' parameterization. This function can now be used to calibration the RF power in the waveguiges.

        # We want to compare parameterization obtained with the spline method and the polynomial method as the means of showing the that the spline fit gives better results.

        if self.spl_inverted is None:

            quench_sim_for_analysis_df = self.quench_sim_df.reset_index()

            x_data_arr = quench_sim_for_analysis_df['E Field Amplitude Squared [V^2/cm^2]'].values
            y_data_arr = quench_sim_for_analysis_df['Fractional Surviving Population'].values

            # Spline fit of 4th order to the simulated quench data.
            spl = scipy.interpolate.UnivariateSpline(x=x_data_arr, y=y_data_arr, k=4, s=0)

            # This code is almost entirely copied from the quenching_curve_anaylsis.py

            # Calculate zeros of the smoothing spline 1st derivative
            spl_roots = np.sort(spl.derivative(n=1).roots())

            # It is possible that the simulation quench curve does not reach even the first pi-pulse. In this case the analysis needs not to be performed.
            if spl_roots.shape[0] > 0:
                # Quenching surviving fractions at the minima/maxima attenuator voltages
                zero_der_vals_spl = spl(spl_roots)

                # We need the first point the is the local minimum. For this we can calculate 2nd derivative and find the first occurence of positive second derivate at the locations where the first derivative is zero.
                spl_second_der = spl.derivative(n=2)

                pi_pulse_spl_index = np.min(np.argwhere(spl_second_der(spl_roots) > 0))

                # Location of the pi-pulse
                pi_pulse_rf_power_spl = spl_roots[pi_pulse_spl_index]

                # Quenching surviving fraction at the pi-pulse determined from the spline fit
                quenching_offset_spl = spl([pi_pulse_rf_power_spl])[0]

                quench_sim_chosen_data_df = quench_sim_for_analysis_df[quench_sim_for_analysis_df['E Field Amplitude Squared [V^2/cm^2]'] < pi_pulse_rf_power_spl]

            else:
                quench_sim_chosen_data_df = quench_sim_for_analysis_df

            # Sort the data by the fractional surviving population. This is needed for the following spline fit - it requires the x-axis to be in the strictly increasing order.
            quench_sim_chosen_data_df = quench_sim_chosen_data_df.sort_values(by='Fractional Surviving Population')

            # We now perform the polynomial fit. The main reason for using the polynomial fitting is that I noticed that if I determine the spline fit and then the inverted spline fit, then there is a problem, By putting some RF power into the spline fit I get the surviving fraction. This surviving fraction is then input into the inverted spline fit. In general I get about 0.4% error in the determination RF power that I get back. I expect that this problem should not exist for the polynomial fit. What possible, however, is that the general fit quality might be unacceptable. This can be tested, however.

            # Notice also that the polynomial fitting is done on the data much earlier than the occurrence of the first pi-pulse. This is, of, course for fit quality purposes.

            # Maximum E field value [V/cm] used for the polynomial fitting
            max_E_field_poly_fit = 31

            poly_fit_degree = 4

            quench_sim_poly_fit_chosen_data_df = quench_sim_chosen_data_df[quench_sim_chosen_data_df['E Field Amplitude Squared [V^2/cm^2]'] <= max_E_field_poly_fit**2]

            x_data_arr = quench_sim_poly_fit_chosen_data_df['E Field Amplitude Squared [V^2/cm^2]'].values
            y_data_arr = quench_sim_poly_fit_chosen_data_df['Fractional Surviving Population'].values

            # Polynomial fit
            poly_fit_coeff_arr = numpy.polyfit(x=x_data_arr, y=y_data_arr, deg=poly_fit_degree)

            self.poly_fit_coeff_arr = poly_fit_coeff_arr
            self.poly_fit_func = np.poly1d(poly_fit_coeff_arr)

            # Values for the RF power used for the polynomial fit + the surviving fractions. This is needed for plotting later on
            self.x_data_poly_fit_arr = x_data_arr
            self.y_data_poly_fit_arr = y_data_arr

            x_data_arr = quench_sim_chosen_data_df['Fractional Surviving Population'].values

            y_data_arr = quench_sim_chosen_data_df['E Field Amplitude Squared [V^2/cm^2]'].values

            # Inverted spline.
            spl_inverted = scipy.interpolate.UnivariateSpline(x=x_data_arr, y=y_data_arr, k=3, s=0)

            self.spl = spl
            self.pi_pulse_rf_power_spl = pi_pulse_rf_power_spl
            self.quenching_offset_spl = quenching_offset_spl
            self.quench_sim_chosen_data_df = quench_sim_chosen_data_df
            self.spl_inverted = spl_inverted

    def get_surv_frac(self, RF_power_arr):
        ''' Given the array of values proportional to the RF power calculates corresponding surviving fractions.

        This function uses the spline fit
        '''
        if self.spl_inverted is None:
            self.analyze_data()

        return self.spl(RF_power_arr)

    def get_surv_frac_with_poly_fit(self, RF_power_arr):
        ''' Given the array of values proportional to the RF power calculates corresponding surviving fractions.

        This function uses the polynomial fit
        '''
        if self.spl_inverted is None:
            self.analyze_data()

        return self.poly_fit_func(RF_power_arr)

    def get_RF_power(self, surv_frac_arr):
        ''' Calculates value proportional to RF power in units of V^2/cm^2, given surviving fraction.

        This function uses the inverted spline fit.
        '''
        if self.spl_inverted is None:
            self.analyze_data()

        return self.spl_inverted(surv_frac_arr)

    def get_RF_power_with_poly_fit(self, surv_frac_arr):
        ''' Calculates value proportional to RF power in units of V^2/cm^2, given surviving fraction.
        This function uses the polynomial fit
        '''
        if self.spl_inverted is None:
            self.analyze_data()

        rf_pow_arr = np.zeros(surv_frac_arr.shape[0])

        index = 0
        for surv_frac in surv_frac_arr:
            poly_fit_coeff = copy.copy(self.poly_fit_coeff_arr)
            poly_fit_coeff[-1] = poly_fit_coeff[-1] - surv_frac

            poly_fit_roots = np.roots(poly_fit_coeff)
            poly_fit_roots = np.real(poly_fit_roots[np.isreal(poly_fit_roots)])
            poly_fit_roots = poly_fit_roots[(poly_fit_roots >= np.min(self.x_data_poly_fit_arr)) & (poly_fit_roots <= np.max(self.x_data_poly_fit_arr))]

            rf_pow_arr[index] = poly_fit_roots[0]

            index = index + 1
        return rf_pow_arr

    def plot_data(self, axes, poly_fit_degree=7):
        ''' Convenience function, that ouputs various plots. Quite important for testing the validity of the data analysis.

        Plots the quenching curve with the polynomial fit and spline fit, and also the residulas between the fits and the data, and also between the two fit curves. Also shows similar plots, but now for the inverted quenching curve
        '''
        quench_sim_for_analysis_df = self.quench_sim_df.reset_index()

        # Quench curve plots
        quench_sim_for_analysis_df.plot(kind='scatter', x='E Field Amplitude Squared [V^2/cm^2]', y='Fractional Surviving Population', color='blue', label='Data', ax=axes[0, 0])

        x_data_arr = quench_sim_for_analysis_df['E Field Amplitude Squared [V^2/cm^2]'].values
        y_data_arr = quench_sim_for_analysis_df['Fractional Surviving Population'].values

        # Polynomial fit
        poly_fit_coeff_arr = numpy.polyfit(x=x_data_arr, y=y_data_arr, deg=poly_fit_degree)
        poly_fit_func = np.poly1d(poly_fit_coeff_arr)

        # Plotting the fits
        x_fit_arr = np.linspace(np.min(x_data_arr), np.max(x_data_arr), x_data_arr.shape[0]*5)

        axes[0, 0].plot(x_fit_arr, poly_fit_func(x_fit_arr), color='red', label='Polynomial Fit')

        x_poly_fit_arr = np.linspace(np.min(self.x_data_poly_fit_arr), np.max(self.x_data_poly_fit_arr), self.x_data_poly_fit_arr.shape[0]*5)

        axes[0, 0].plot(x_poly_fit_arr, self.poly_fit_func(x_poly_fit_arr), color='black', label='Polynomial Fit (up to the pi-pulse)')

        # Spline fit of 4th order to the simulated quench data.
        axes[0, 0].plot(x_fit_arr, self.spl(x_fit_arr), color='green', label='Spline Fit')

        axes[0, 0].set_title('Quench curve')
        axes[0, 0].legend()

        axes[0, 1].plot(x_data_arr, (poly_fit_func(x_data_arr)-y_data_arr)/y_data_arr*1E3, marker='.', linestyle='-', color='red', label='Polynomial Fit')

        axes[0, 1].plot(self.x_data_poly_fit_arr, (self.poly_fit_func(self.x_data_poly_fit_arr)-self.y_data_poly_fit_arr)/self.y_data_poly_fit_arr*1E3, marker='.', linestyle='-', color='black', label='Polynomial Fit (up to the pi-pulse)')

        axes[0, 1].plot(x_data_arr, (self.spl(x_data_arr)-y_data_arr)/y_data_arr*1E3, marker='.', linestyle='-', color='green', label='Spline Fit')

        axes[0, 1].set_xlabel('E Field Amplitude Squared [V^2/cm^2]')
        axes[0, 1].set_ylabel('Fractional deviation [ppt]')
        axes[0, 1].set_title('Fractional deviation of the fit from the data')

        axes[0, 1].legend()

        axes[0, 2].plot(x_fit_arr, (poly_fit_func(x_fit_arr)-self.spl(x_fit_arr))/self.spl(x_fit_arr)*1E3, marker='.', linestyle='-', color='blue')

        axes[0, 2].set_xlabel('E Field Amplitude Squared [V^2/cm^2]')
        axes[0, 2].set_ylabel('Fractional deviation [ppt]')
        axes[0, 2].set_title('Fractional deviation of the polynomail fit from the spline')

        # Inverted quench curve plots

        self.quench_sim_chosen_data_df.plot(kind='scatter', x='Fractional Surviving Population', y='E Field Amplitude Squared [V^2/cm^2]', ax=axes[1, 0], color='blue', label='Data')

        x_data_arr = self.quench_sim_chosen_data_df['Fractional Surviving Population'].values

        y_data_arr = self.quench_sim_chosen_data_df['E Field Amplitude Squared [V^2/cm^2]'].values

        poly_fit_coeff_arr = numpy.polyfit(x=x_data_arr, y=y_data_arr, deg=poly_fit_degree)
        poly_fit_func = np.poly1d(poly_fit_coeff_arr)

        x_fit_arr = np.linspace(np.min(x_data_arr), np.max(x_data_arr), x_data_arr.shape[0]*5)
        axes[1, 0].plot(x_fit_arr, poly_fit_func(x_fit_arr), color='red', label='Polynomial Fit')

        # Inverted spline.

        axes[1, 0].plot(x_fit_arr, self.spl_inverted(x_fit_arr), color='green', label='Spline Fit')

        axes[1, 0].set_title('Inverted Quench Curve')
        axes[1, 0].legend()

        axes[1, 1].plot(x_data_arr, (poly_fit_func(x_data_arr)-y_data_arr), marker='.', linestyle='-', color='red', label='Polynomial Fit')
        axes[1, 1].plot(x_data_arr, (self.spl_inverted(x_data_arr)-y_data_arr), marker='.', linestyle='-', color='green', label='Spline Fit')

        axes[1, 1].set_xlabel('Fractional Surviving Population')
        axes[1, 1].set_ylabel('Deviation [V^2/cm^2]')
        axes[1, 1].set_title('Deviation of the fit from the data')

        axes[1, 1].legend()

        axes[1, 2].plot(x_fit_arr, (poly_fit_func(x_fit_arr)-self.spl_inverted(x_fit_arr)), marker='.', linestyle='-', color='blue')

        axes[1, 2].set_xlabel('Fractional Surviving Population')
        axes[1, 2].set_ylabel('Deviation [V^2/cm^2]')
        axes[1, 2].set_title('Deviation of the polynomial fit from the spline')

        return axes

class FOSOFSimulation():
    ''' Convenience class to work with the FOSOF simulation data.

        One of the important uses of this class is that we fundamentally want to be able to determine by how many radians phase at given RF frequency and expected RF power should be shifted if the RF power is different from the expected. To implement this we can do the following sequential steps.

        1. The simulation data (FOSOF phase vs RF frequency) is given for a number of different off-axis distances, which is the distance, away from the axis of the experiment, at which the atom in the simulation was placed. We, in general, can have different off-axis distance for our beam than the ones available in the simulation. Thus we can use the available simulation data, and for each beam speed, waveguide separation, RF electric field amplitude, and RF frequency we can take the list of off-axis distances and the respective FOSOF phases. This list can be used in polynomial fitting. The expectation is that the phase is linearly dependent on the square of the off-axis distance (when this distance is, of course, not too large). Thus a simple linear fit will suffice. Using these fits we can now determine the FOSOF phase for the required off-axis distance. To make sure that the fits are reasonable I also calculate the residuals of the fit. In addition, I use the interpolated FOSOF phases for the same off-axis distances as already given by the simulations, and then fit a first-order polynomial to the given set of FOSOF phases vs frequency for the given simulation key and off-axis distance to determine the zero-crossing frequency and compare it with the zero-crossing frequency determined by doing the first-order polynomial fit to the original simulation data for the same simulation key and off-axis distance. I then search for the simulation key and the off-axis distance that have the largest difference in the zero-crossing frequency between the interpolated data and the original simulation data. This largest difference can act as some type of the systematic uncertainty for our off-axis distance interpolation.

        2. We now pick a set of simulation keys + off-axis distance that correspond to the range of frequencies of interest + beam speed + waveguide separation + off-axis distance. In general, the frequencies of interest can be different that the ones present in the simulation. In other words, the RF frequencies used in the experiment are different from the frequencies used in the simulation. We can easily overcome this difficulty by using the previously determined first-order polynomial fit for the FOSOF phase vs RF frequency data for each simulation key in the set of chosen simulation + the required off-axis distance. This way we can determine the FOSOF phase at the required RF frequency. The first-order polynomial fit is sufficient for relatively short frequency ranges, which includes any of the frequency ranges that we scanned through in the experiment. Notice that here we should not be worried about the quality of the interpolation for frequencies, because we use the same polynomial as used for finding zero-crossing frequency of the interpolated FOSOF phases vs frequency.

        3. For the set of simulation keys chosen we use their respective waveguide electric field values. For each RF frequency we expect that the FOSOF phase is linearly dependent on the square of the waveguide electric field. Thus it is sufficient to fit a first-order polynomial to the FOSOF phase vs the square of the electric field. However, for larger electric field we need to have the second-order polynomial. At the end we get for every RF frequency a polynomial function that allows us to find the FOSOF phase for given waveguide electric field amplitude.

        The first step can be performed once, and the resulting fits can be stored as an object. And for each FOSOF experiment we can load this saved object and use it to perform the step 2 and 3. The steps 2 and 3 can be performed really quickly. It actually seems to be more time consuming to check the object for whether the steps 2 and 3 have been previously performed for each required frequency. Thus it is easier to simply recalculate steps 2 and 3 for each acquired FOSOF data set separately then to have it stored in the object and then perform the search that checks whether the required data is present in the object for each FOSOF data set.
    '''
    # path to the blind offset file
    blind_offset_path = sim_data_folder_path

    blind_offset_file_name = 'blind-20150113.npy'

    def __init__(self, load_Q=True):

        # Version of the data analysis. Notice that this version number is 'engraved' in the analysis data file name. Thus one has to use the appropriate data analysis version.
        self.version_number = 0.1

        # Location for storing the analysis folders
        self.saving_folder_location = sim_data_folder_path

        self.fosof_sim_data_folder_name = 'FOSOF_sim_data_analysis'

        self.analysis_data_file_name = 'fosof_sim_data_v' + str(self.version_number) + '.pckl'

        # Checking if the class instance has been saved before. If it was, then in case the user wants to load the data, it gets loaded. In all other cases the initialization continues.

        self.perform_analysis_Q = False

        os.chdir(self.saving_folder_location)
        if os.path.isdir(self.fosof_sim_data_folder_name):

            print('The analysis folder exists. Checking whether the analysis file exists...')
            os.chdir(self.fosof_sim_data_folder_name)

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

            # Extract the OLD and the NEW simulation data.
            # IMPORTANT!!!
            # The new simulation data has RF frequencies that we used in our experiment. They were given to Alain with the blind offset as well. The frequencies stored in the resulting simulations do not have the blind offset added to them. The resonant frequency is still assumed to be 909.874 MHz. What needs to be done is that the list of frequencies from the NEW simulation needs to have the blind offset added to them to have the correct frequencies stored.

            os.chdir(FOSOFSimulation.blind_offset_path)

            # Blind in MHz. It gets added to all of the RF frequencies. Thus the RF frequencies used in the experiment need to have the blind offset subtracted from them.
            self.__BLIND_OFFSET_VALUE = np.load(FOSOFSimulation.blind_offset_file_name).flatten()[0]

            old_sim_info_df = OldSimInfo().get_info().reset_index().set_index('Simulation Key').sort_index()
            new_sim_info_df = NewSimInfo().get_info().reset_index().set_index('Simulation Key').sort_index()

            # Some of the simulations are not for studying the FOSOF lineshape. We include only the FOSOF-related simulation keys.
            reg_exp = re.compile('FOSOF')
            matched_arr = np.array(list(map(lambda x: reg_exp.match(x), old_sim_info_df.index.values))) != None
            fosof_sim_info_old_df = old_sim_info_df.loc[old_sim_info_df.index.values[matched_arr]].copy()

            fosof_sim_info_old_df.loc[:, 'Old Simulation'] = True

            matched_arr = np.array(list(map(lambda x: reg_exp.match(x), new_sim_info_df.index.values))) != None
            fosof_sim_info_new_df = new_sim_info_df.loc[new_sim_info_df.index.values[matched_arr]].copy()

            fosof_sim_info_new_df.loc[:, 'Old Simulation'] = False

            # There are some duplicate simulation keys in new and old simulation dataframes. In case of a duplicate, we keep the new on: I assume that the folder to which the key is pointing has the new simulation data.

            fosof_sim_info_old_reindexed_df = fosof_sim_info_old_df.loc[fosof_sim_info_old_df.index.difference(fosof_sim_info_new_df.index)]

            self.fosof_sim_info_df = pd.concat([fosof_sim_info_new_df, fosof_sim_info_old_reindexed_df], sort=True)

            # The OLD simulation files have some additional columns that are not present in the NEW simulation files. I use only the columns that are present in the NEW simulation files
            self.fosof_sim_info_df = self.fosof_sim_info_df.loc[:, fosof_sim_info_new_df.columns]

            # Resonant frequency [MHz] used for the simulations.
            self.f0 = 909.874

            self.fosof_sim_data_df = None
            self.off_axis_poly_fit_df = None
            self.phase_vs_off_axis_dist_df = None
            self.zero_crossing_vs_off_axis_dist_df = None
            self.zero_crossing_sim_df = None
            self.zero_crossing_diff_df = None
            self.interp_fosof_lineshape_param_s = None

            # Loading the simulation data
            self.load_fosof_sim_data()

            # Perform interpolation for the FOSOF data vs off-axis distance for each simulation key
            self.get_off_axis_func()

            # Array of the off-axis distances used in the simulations.
            off_axis_dist_arr = self.fosof_sim_data_df['Off-axis Distance [mm]'].drop_duplicates().values

            # For the off-axis distances for which the simulation data is present I want to check the quality of the interpolation to other off-axis distances. For this I use the polynomial fits determined for the FOSOF phase vs off-axis distance and find the difference in the zero-crossing frequencies determined by a linear fit to the simulation data and to the interpolated data.
            self.calc_fosof_phase_for_off_axis_dist(off_axis_dist_arr)

            # Calculate zero-crossing frequencies for all interpolated off-axis distances
            self.get_off_axis_dist_fosof_res_freq()
            self.get_max_zero_cross_diff_vs_off_axis_dist()

            # Similar idea for the data vs E field amplitude. (see function description)
            self.calc_fosof_interpolation_unc()

            self.save_instance(rewrite_Q=True)

    def load_fosof_sim_data(self):
        ''' Loads all of the FOSOF simulation data
        '''

        # Firstly I want to load of all the FOSOF simulation data (frequency and the respective FOSOF phase) into one dataframe)

        # Folder where the old simulation data is stored for FOSOF
        fosof_sim_data_folder_name_old = 'FOSOF'

        # Name of the old simulation file in each simulation folder
        fosof_sim_data_file_name_old = 'FOSOFLineshapeData.xlsx'

        if self.fosof_sim_data_df is None:
            self.fosof_sim_data_df = pd.DataFrame()

        fosof_sim_data_sets_df = pd.DataFrame()

        fosof_sim_data_keys_arr = self.fosof_sim_data_df.index.drop_duplicates().values

        for fosof_sim_name in self.fosof_sim_info_df.index.values:
            # Only new data gets loaded. Therefore, if one has changed some of the simulation data, then it is safer to reinitialize the object so that all of the changes can be properly loaded
            if (fosof_sim_name is not None) and (fosof_sim_name not in fosof_sim_data_keys_arr):

                os.chdir(sim_data_path)

                # The new and old simulations have a bit different format for the FOSOF data. Thus we need to use somewhat different code to load these different types of simulations.

                if self.fosof_sim_info_df.loc[fosof_sim_name, 'Old Simulation'] == True:

                    os.chdir(old_sim_folder_path)
                    os.chdir(fosof_sim_data_folder_name_old)

                    if os.path.isdir(fosof_sim_name):

                        os.chdir(fosof_sim_name)

                        fosof_sim_data_xlsx = pd.ExcelFile(fosof_sim_data_file_name_old)
                        fosof_sim_data_df = pd.read_excel(pd.ExcelFile(fosof_sim_data_xlsx), sheet_name='Sheet1', header=[1, 2], index_col=[0]).sort_index(axis='columns')

                        # Changing some of the column names and adding the resonant frequency to the delta frequency data to change it to absolute frequency.
                        freq_col_name = 'Frequency [MHz]'
                        fosof_sim_data_df.index.names = [freq_col_name]
                        fosof_sim_data_df.columns.names = ['Off-axis Distance [mm]', 'Data Field']
                        fosof_sim_data_df = fosof_sim_data_df.rename(columns={'Amp.': 'Amplitude', 'Phase': 'Phase [Rad]', 'Avg.': 'Average'}).drop(['Shift (kHz):', 'Slope (rad/MHz):']).reset_index()

                        # Sometimes there is an additional line present. This gets removed. If a simply use the 'in' method, then the interpreter gives me warning. It seems that it happens when there is no additional line present and I am trying to see if there is a string object in the np.array of numbers. That is why I am first checking the type of the numpy array, and then convert this array to a list, since numpy and python disagree on what to do in the aforementioned case. This is somewhat described in  (https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur?rq=1)
                        if (fosof_sim_data_df['Frequency [MHz]'].dtype == np.object) and ('NOTE: base freq.=909.874 MHz' in list(fosof_sim_data_df['Frequency [MHz]'].values)):
                            fosof_sim_data_df = fosof_sim_data_df.drop(fosof_sim_data_df[fosof_sim_data_df['Frequency [MHz]'] == 'NOTE: base freq.=909.874 MHz'].index.values, axis='index')

                        fosof_sim_data_df[freq_col_name] = fosof_sim_data_df[freq_col_name] + self.f0

                        fosof_sim_data_df = fosof_sim_data_df.set_index(freq_col_name).sort_index()

                        fosof_sim_data_reshaped_df = fosof_sim_data_df.stack(level='Off-axis Distance [mm]')

                        # Convert data types of some columns
                        fosof_sim_data_reshaped_df = fosof_sim_data_reshaped_df.reset_index().astype({'Off-axis Distance [mm]': np.float64}).set_index(['Frequency [MHz]'])

                        fosof_sim_data_reshaped_df['Simulation Key'] = fosof_sim_name

                        fosof_sim_data_reshaped_df = fosof_sim_data_reshaped_df.reset_index().set_index('Simulation Key')

                        fosof_sim_data_sets_df = fosof_sim_data_sets_df.append(fosof_sim_data_reshaped_df)

                    else:
                        print('Old simulation folder: ' + fosof_sim_name + ' is not found.')

                # If the simulation type is 'New'...
                else:

                    fosof_sim_file_name_new = 'Lineshapes_' + fosof_sim_name + '.csv'

                    os.chdir(new_sim_folder_path)

                    if os.path.isfile(fosof_sim_file_name_new):

                        fosof_sim_data_df = pd.read_csv(filepath_or_buffer=fosof_sim_file_name_new, delimiter=',', comment='#', header=None, skip_blank_lines=True, index_col=0)

                        fosof_sim_data_df.columns = pd.MultiIndex.from_tuples(list(zip(*[fosof_sim_data_df.iloc[0].values/1E3, ['Phase [Rad]', 'Amplitude', 'Average'] * int(fosof_sim_data_df.columns.shape[0]/3)])))

                        fosof_sim_data_df.columns.names = ['Off-axis Distance [mm]', 'Data Field']
                        fosof_sim_data_df.index.names = ['Frequency [MHz]']

                        # Remove the first row and the two lost rows.
                        fosof_sim_data_df.drop(fosof_sim_data_df.index[[0, -1, -2]], inplace=True)

                        # Subtract the blind offset from the frequencies.
                        fosof_sim_data_df = fosof_sim_data_df.reset_index()
                        fosof_sim_data_df['Frequency [MHz]'] = fosof_sim_data_df['Frequency [MHz]'] + self.__BLIND_OFFSET_VALUE
                        fosof_sim_data_df = fosof_sim_data_df.set_index('Frequency [MHz]')

                        fosof_sim_data_reshaped_df = fosof_sim_data_df.stack(level='Off-axis Distance [mm]')

                        fosof_sim_data_reshaped_df['Simulation Key'] = fosof_sim_name

                        fosof_sim_data_reshaped_df = fosof_sim_data_reshaped_df.reset_index().set_index('Simulation Key')

                        # Change the type of the Off-axis distance column to float
                        fosof_sim_data_reshaped_df.loc[slice(None), ('Off-axis Distance [mm]')] = fosof_sim_data_reshaped_df['Off-axis Distance [mm]'].astype(np.float64)

                        fosof_sim_data_sets_df = fosof_sim_data_sets_df.append(fosof_sim_data_reshaped_df)

                    else:
                        print('New simulation file: ' + fosof_sim_name + ' is not found.')

        # Appending should be only doine when the dataframe to be appended has at least 1 row. Otherwise pandas throws a warning that the columns are not aligned (makes sense, since there is nothing to align)

        if fosof_sim_data_sets_df.shape[0] > 0:
            self.fosof_sim_data_df = self.fosof_sim_data_df.append(fosof_sim_data_sets_df)

        return self.fosof_sim_data_df

    def get_off_axis_func(self):
        ''' Performs least-squares fit to obtain fit functions of FOSOF phase vs off-axis distance for each simulation key and RF frequency
        '''
        if self.off_axis_poly_fit_df is None:
            def calc_off_axis_poly_fit(df):
                ''' Peforming polynomial fit to the FOSOF phase vs off-axis distance.
                '''
                poly_fit_order = 1

                # We expect linear dependence of the FOSOF phase with the square of the off-axis distance (as long as the off-axis distance is small enough)
                x_data_arr = df['Off-axis Distance [mm]'].values**2

                phase_vs_offset_dist_poly_fit_params = np.polyfit(x=x_data_arr, y=df['Phase [Rad]'].values, deg=poly_fit_order)

                poly_fit_func = np.poly1d(phase_vs_offset_dist_poly_fit_params)

                return pd.Series({
                    'Polynomial Fit Function': poly_fit_func,
                    'Largest Absolute Residual [mrad]': np.max(np.abs(poly_fit_func(x_data_arr)-df['Phase [Rad]'].values)) * 1E3
                    })

            off_axis_poly_fit_df = self.fosof_sim_data_df.groupby(['Simulation Key', 'Frequency [MHz]']).apply(calc_off_axis_poly_fit)

            self.off_axis_poly_fit_df = off_axis_poly_fit_df

        return self.off_axis_poly_fit_df

    def select_off_axis_large_residuals(self, min_residual, max_residual):
        ''' Selects fosof simulation data that corresponds to the off-axis polynomial fits resulting in the residual values within the specified allowed range

        Useful for checking if there are some outliers in the simulation data, because even though one expects that the FOSOF phase is linearly dependent on the square of the off-axis distance, it is not perfectly the case. For some separations and frequencies the relationship is actually not linear, albeit the residuals are still small.
        '''
        self.residuals_off_axis_df = self.off_axis_poly_fit_df[(self.off_axis_poly_fit_df['Largest Absolute Residual [mrad]'] < max_residual) & (self.off_axis_poly_fit_df['Largest Absolute Residual [mrad]'] > min_residual)]

        if self.residuals_off_axis_df.shape[0] > 0:
            self.residuals_off_axis_data_df = self.fosof_sim_data_df.set_index('Frequency [MHz]', append=True).loc[self.residuals_off_axis_df.index]

        else:
            self.residuals_off_axis_data_df = pd.DataFrame()

        return self.residuals_off_axis_df, self.residuals_off_axis_data_df

    def get_off_axis_data_plots(self, axes):
        ''' Auxiliary plotting function to visualize the fosof simulation data sets having the residuals within the specified range specified off-axis fit residuals.
        '''
        for i in range(self.residuals_off_axis_df.index.values.shape[0]):

            index = self.residuals_off_axis_df.index.values[i]

            df = self.residuals_off_axis_data_df.loc[index]

            x_data_arr = df['Off-axis Distance [mm]'].values**2
            y_data_arr = df['Phase [Rad]'].values*1E3

            x_arr = np.linspace(np.min(x_data_arr), np.max(x_data_arr), x_data_arr.shape[0]*10)

            poly_fit_func = self.off_axis_poly_fit_df.loc[index]['Polynomial Fit Function']

            y_arr = poly_fit_func(x_arr)*1E3

            axes[i,0].scatter(x=x_data_arr, y=y_data_arr, label='Simulation Data', color='blue')
            axes[i,0].plot(x_arr, y_arr, color='red', label='Polynomial Fit')

            axes[i,0].set_ylim(get_range(y_data_arr, fract_range=0.1))

            axes[i,0].set_xlabel('Off-axis value$^2$ [mm$^2$]')
            axes[i,0].set_ylabel('FOSOF Phase [mrad]')
            axes[i,0].legend()
            axes[i,0].set_title('Simulation Key: ' + self.residuals_off_axis_df.index.values[i][0] + '\n' + 'Frequency [MHz]: ' + str(self.residuals_off_axis_df.index.values[i][1]))

            residual_data_arr = y_data_arr-poly_fit_func(x_data_arr)*1E3

            axes[i,1].bar(x=x_data_arr, height=residual_data_arr)

            axes[i,1].set_ylim(get_range(residual_data_arr, fract_range=0.1))

            axes[i,1].set_xlabel('Off-axis value$^2$ [mm$^2$]')
            axes[i,1].set_ylabel('Residual [mrad]')
            axes[i,1].set_title('Simulation Key: ' + self.residuals_off_axis_df.index.values[i][0] + '\n' + 'Frequency [MHz]: ' + str(self.residuals_off_axis_df.index.values[i][1]))

            axes[i,1].grid()

        return axes

    def calc_fosof_phase_for_off_axis_dist(self, off_axis_dist_arr):
        ''' Given the array of the off-axis distances, this method calculates the respective FOSOF phases vs frequency for each simulation key.

        The calculation is done once for each off-axis distance. Thus if the method is called again, it will append new off-axis distances to the dataframe
        '''
        for off_axis_dist in off_axis_dist_arr:

            if self.phase_vs_off_axis_dist_df is None:
                self.phase_vs_off_axis_dist_df = pd.DataFrame()

            if (self.phase_vs_off_axis_dist_df.shape[0] == 0) or (off_axis_dist not in list(self.phase_vs_off_axis_dist_df.index.get_level_values('Off-axis Distance [mm]').drop_duplicates().values)):

                fosof_phase_df = self.off_axis_poly_fit_df[['Polynomial Fit Function']].transform(lambda x: list(map(lambda y: y(off_axis_dist**2), x))).rename(columns={'Polynomial Fit Function': 'Phase [Rad]'})

                fosof_phase_df['Off-axis Distance [mm]'] = off_axis_dist
                fosof_phase_df = fosof_phase_df.reset_index().set_index(['Simulation Key', 'Off-axis Distance [mm]'])

                self.phase_vs_off_axis_dist_df = self.phase_vs_off_axis_dist_df.append(fosof_phase_df)
        return self.phase_vs_off_axis_dist_df

    def calc_fosof_res_freq(self, df):
        ''' Peforming polynomial fit to the FOSOF phase vs frequency, and extracting the zero-crossing frequency as well as the difference of the zero-crossing frequency from the resonant frequency defined by the simulation
        '''
        poly_fit_order = 1

        x_data_arr = df['Frequency [MHz]'].values

        phase_vs_freq_poly_fit_params = np.polyfit(x=x_data_arr, y=df['Phase [Rad]'].values, deg=poly_fit_order)

        poly_fit_func = np.poly1d(phase_vs_freq_poly_fit_params)

        zero_crossing = -phase_vs_freq_poly_fit_params[1]/phase_vs_freq_poly_fit_params[0]
        return pd.Series({
            'Polynomial Fit Function': poly_fit_func,
            'Largest Absolute Residual [mrad]': np.max(np.abs(poly_fit_func(x_data_arr)-df['Phase [Rad]'].values)) * 1E3,
            'Resonant Frequency Offset [kHz]': (zero_crossing - self.f0) * 1E3,
            'Slope [Rad/MHz]': phase_vs_freq_poly_fit_params[0]
            })

    def get_off_axis_dist_fosof_res_freq(self):
        ''' Calculates zero-crossing frequency for the interpolated FOSOF data for the given off-axis distances.
        '''
        if self.phase_vs_off_axis_dist_df is None:

            off_axis_dist_arr = self.fosof_sim_data_df['Off-axis Distance [mm]'].drop_duplicates().values

            self.calc_fosof_phase_for_off_axis_dist(off_axis_dist_arr)

        self.zero_crossing_vs_off_axis_dist_df = self.phase_vs_off_axis_dist_df.groupby(['Simulation Key', 'Off-axis Distance [mm]']).apply(self.calc_fosof_res_freq)

        return self.zero_crossing_vs_off_axis_dist_df

    def get_max_zero_cross_diff_vs_off_axis_dist(self):
        ''' For the off-axis distances for which the simulation data is present I want to check the quality of the interpolation to other off-axis distances. For this I use the polynomial fits determined for the FOSOF phase vs off-axis distance and find the difference in the zero-crossing frequencies determined by a linear fit to the simulation data and to the interpolated data. The maximum zero crossing frequency difference is returned.

        This maximum zero crossing frequency difference can be used as the maximum systematic uncertainty that is assigned to the resonant frequencies extracted from the interpolated lineshapes
        '''
        if self.zero_crossing_sim_df is None:
            self.zero_crossing_sim_df = self.fosof_sim_data_df.set_index('Off-axis Distance [mm]', append=True).groupby(['Simulation Key', 'Off-axis Distance [mm]']).apply(self.calc_fosof_res_freq)

        if self.zero_crossing_vs_off_axis_dist_df is None:
            self.get_off_axis_dist_fosof_res_freq()

        # Index common to the simulation and the interpolated data.
        common_index = self.zero_crossing_vs_off_axis_dist_df.index.intersection(self.zero_crossing_sim_df.index)

        # Maximum absolute deviation from the simulation zero-crossing
        self.max_zero_cross_diff_vs_off_axis_dist = np.max(np.abs(self.zero_crossing_sim_df.loc[common_index]['Resonant Frequency Offset [kHz]'] - self.zero_crossing_vs_off_axis_dist_df.loc[common_index]['Resonant Frequency Offset [kHz]']))

        return self.max_zero_cross_diff_vs_off_axis_dist

    def filter_fosof_sim_set(self, sim_params_dict, blind_freq_Q=False):
        ''' Selects FOSOF simulations for given parameters.

        The parameters are the following:
        'Frequency array [MHz]': Array of frequencies that determines whether the NEW or the OLD simulations are used. Read the comments in the code below to understand better.
        'Waveguide Separation [cm]': physical separation between the waveguides [cm].
        'Accelerating Voltage [kV]': Accelerating voltage that gets converted to the beam speed. These are the accelerating voltages used during actual data acquisition.
        'Speed [cm/ns]': Alternatively one can specify directly the beam speed of the atoms. In case if this is the parameter that gets specified, then even if the 'Accelerating Voltage [kV]' is present, the speed parameter takes over.
        'Off-axis distance [mm]': The rms size of the beam. If the particular chosen size is not present in the simulation, then the interpolation is used to calculate the FOSOF phases for this particular size,
        'Old Simulation': True or False. If this field is present, it forces the code to pick the simulation keys that are corresponding to the chosen simulation type (Old or New). If not present, then the code choses the best available simulation type, depending on the frequency range. Read the comments in the code below for additional information on this.

        :blind_freq_Q: bool. Flag of whether the list of frequencies supplied is blinded. This is important, when we use the experiment frequencies that have the blind added to them.
        '''

        if 'Speed [cm/ns]' not in list(sim_params_dict.keys()):

            if sim_params_dict['Accelerating Voltage [kV]'] > 49 and sim_params_dict['Accelerating Voltage [kV]'] < 50:
                beam_speed_to_use = 0.3223

            if sim_params_dict['Accelerating Voltage [kV]'] > 22 and sim_params_dict['Accelerating Voltage [kV]'] < 23:
                beam_speed_to_use = 0.2254

            if sim_params_dict['Accelerating Voltage [kV]'] > 16 and sim_params_dict['Accelerating Voltage [kV]'] < 17:
                beam_speed_to_use = 0.199

        else:
            beam_speed_to_use = sim_params_dict['Speed [cm/ns]']

        self.freq_arr = sim_params_dict['Frequency Array [MHz]']

        self.blind_freq_Q = blind_freq_Q

        if self.blind_freq_Q:
            self.freq_arr = self.freq_arr + self.__BLIND_OFFSET_VALUE

        f_min = np.min(self.freq_arr)
        f_max = np.max(self.freq_arr)

        if 'Old Simulation' not in list(sim_params_dict.keys()):
            # In case the 'Old simulation' field is not specified: the simulations are such that when the range of the frequencies is between around +- 2 MHz about the resonant frequency, then we should use the new simulation. And whenever we are interested in larger range of frequencies, we should use the old simulations. This is mostly because we did not ask Alain to perform new simulations (with proper phase averaging between the RF fields in the waveguides and the RF fields in the quench cavities) for larger than +-2 MHz ranges of frequencies, since it was quite time consuming.
            # Range of frequencies for which we can use the NEW simulations.
            f_sim_range_min_new = 907.8
            f_sim_range_max_new = 912.2


            # Flag for whether to use new or old simulations
            sim_old_bool = False

            if f_min < f_sim_range_min_new or f_min > f_sim_range_max_new or f_max < f_sim_range_min_new or f_max > f_sim_range_max_new:
                sim_old_bool = True
        else:
            sim_old_bool = sim_params_dict['Old Simulation']

        # Choose the FOSOF simulations to use
        fosof_sim_info_chosen_df = self.fosof_sim_info_df[(self.fosof_sim_info_df['Old Simulation'] == sim_old_bool) & (self.fosof_sim_info_df['Waveguide Separation [cm]'] == sim_params_dict['Waveguide Separation [cm]']) & (self.fosof_sim_info_df['Speed [cm/ns]'] == beam_speed_to_use)]

        # There are two simulations that were done for having unequal power in the waveguides. These should not be combined with other simulations, and thus are removed.
        if sim_params_dict['Waveguide Separation [cm]'] == 4 and sim_old_bool and beam_speed_to_use == 0.3223:
            fosof_sim_info_chosen_df = fosof_sim_info_chosen_df.drop(['FOSOF-04-08-16-3223-003', 'FOSOF-04-08-16-3223-004'])

        fosof_sim_info_chosen_df.loc[slice(None), ('Waveguide Electric Field [V/cm]')] = fosof_sim_info_chosen_df['Waveguide Electric Field [V/cm]'].astype(np.float64)

        # Making sure that the requested off-axis distance is present in the simulation object. If it is not present, then the respective FOSOF phases and linear fit to these phases = FOSOF lineshape gets calculated for this off-axis value and the class instance gets saved to store the new information.
        if sim_params_dict['Off-axis Distance [mm]'] not in list(self.phase_vs_off_axis_dist_df.index.get_level_values('Off-axis Distance [mm]').drop_duplicates().values):
            print('There is no previsouly generated data for the requested off-axis distance. Generating it and saving the object..')
            self.calc_fosof_phase_for_off_axis_dist([sim_params_dict['Off-axis Distance [mm]']])
            zero_crossing_vs_off_axis_dist_df = self.get_off_axis_dist_fosof_res_freq()
            self.save_instance()

        zero_crossing_vs_off_axis_dist_chosen_df = self.zero_crossing_vs_off_axis_dist_df.loc[(fosof_sim_info_chosen_df.index.values, sim_params_dict['Off-axis Distance [mm]']), (slice(None))].reset_index('Off-axis Distance [mm]', drop=True)

        self.zero_crossing_vs_off_axis_dist_chosen_df = zero_crossing_vs_off_axis_dist_chosen_df

        self.fosof_sim_info_chosen_df = fosof_sim_info_chosen_df

        return self.zero_crossing_vs_off_axis_dist_chosen_df, self.fosof_sim_info_chosen_df

    def calc_e_field_poly_fit(self, df):
        ''' Peforming polynomial fit to the FOSOF phase vs the electric field amplitude.
        '''

        # We expect that second-order polynomial should be enough to describe the AC stark shift. For low electric field amplitudes the phase shift should be linear with the power contained in the electric field. With larger field strength we start seeing that square of the power term is also needed.

        x_data_arr = df['Waveguide Electric Field [V/cm]']**2
        y_data_arr = df['Phase [Rad]']

        if x_data_arr.shape[0] >= 3:
            poly_fit_order = 2
        else:
            #print('WARNING!!! Given dataframe contains less than 3 electric field values. Setting the polynomial fit order to 1 instead of 2.')
            #print('Electric field values [V/cm]:' + str(df['Waveguide Electric Field [V/cm]']))
            poly_fit_order = 1

        phase_vs_freq_poly_fit_params = np.polyfit(x=x_data_arr, y=y_data_arr, deg=poly_fit_order)

        poly_fit_func = np.poly1d(phase_vs_freq_poly_fit_params)

        return pd.Series({
            'Polynomial Fit Function': poly_fit_func,
            'Largest Absolute Residual [mrad]': np.max(np.abs(poly_fit_func(x_data_arr)-y_data_arr)) * 1E3,
            'Number Of Points': x_data_arr.shape[0]
            })

    def get_e_field_func(self):
        ''' Performs least-squares fit to obtain fit functions of FOSOF phase vs the electric field amplitude for each simulation key and given array of frequencies
        '''

        def calc_fosof_phase_vs_freq(df, freq_arr):
            ''' Calculates FOSOF phase for given list of frequencies.

            '''
            poly_fit_func = df['Polynomial Fit Function'][0]

            s = pd.Series(dict(np.array([freq_arr, poly_fit_func(freq_arr)]).T))

            s.index.name = 'Frequency [MHz]'
            return s

        # Calculate FOSOF phases for the given list of frequencies.

        phase_vs_freq_df = pd.DataFrame(self.zero_crossing_vs_off_axis_dist_chosen_df.groupby(['Simulation Key']).apply(lambda df: calc_fosof_phase_vs_freq(df, self.freq_arr)).stack(level='Frequency [MHz]'), columns=['Phase [Rad]']).reset_index().set_index(['Simulation Key'])

        # Combine the FOSOF phases with the waveguide E field amplitudes
        phase_vs_e_field_df = phase_vs_freq_df.join(self.fosof_sim_info_chosen_df['Waveguide Electric Field [V/cm]']).reset_index().set_index('Frequency [MHz]').drop(columns=['Simulation Key']).sort_index()

        phase_vs_e_field_poly_fit_df = phase_vs_e_field_df.groupby('Frequency [MHz]').apply(self.calc_e_field_poly_fit)

        self.phase_vs_e_field_df = phase_vs_e_field_df
        self.phase_vs_e_field_poly_fit_df = phase_vs_e_field_poly_fit_df

        return self.phase_vs_e_field_poly_fit_df

    def select_e_field_large_residuals(self, min_residual, max_residual):
        ''' Selects fosof simulation data that corresponds to the FOSOF phase vs waveguide Electric field amplitude polynomial fits resulting in the residual values within the specified allowed range

        Useful for checking if there are some outliers in the simulation data.
        '''
        residuals_e_field_df = self.phase_vs_e_field_poly_fit_df[(self.phase_vs_e_field_poly_fit_df['Largest Absolute Residual [mrad]'] < max_residual) & (self.phase_vs_e_field_poly_fit_df['Largest Absolute Residual [mrad]'] > min_residual)]

        if residuals_df.shape[0] > 0:
            residuals_e_field_data_df = self.phase_vs_e_field_df.loc[residuals_e_field_df.index]

        else:
            residuals_e_field_data_df = pd.DataFrame()

        self.residuals_e_field_df = residuals_e_field_df
        self.residuals_e_field_data_df = residuals_e_field_data_df

        return self.residuals_e_field_df, self.residuals_e_field_data_df

    def get_e_field_data_plots(self, axes):
        ''' Auxiliary plotting function to visualize the fosof simulation data sets having the residuals within the specified range of FOSOF phase vs E field fro given RF frequency fit residuals.
        '''
        for i in range(self.residuals_e_field_df.index.values.shape[0]):

            index = self.residuals_e_field_df.index.values[i]

            df = self.residuals_e_field_data_df.loc[index]

            x_data_arr = df['Waveguide Electric Field [V/cm]'].values**2
            y_data_arr = df['Phase [Rad]'].values*1E3

            x_arr = np.linspace(np.min(x_data_arr), np.max(x_data_arr), x_data_arr.shape[0]*10)

            poly_fit_func = self.phase_vs_e_field_poly_fit_df.loc[index]['Polynomial Fit Function']

            y_arr = poly_fit_func(x_arr)*1E3

            axes[i,0].scatter(x=x_data_arr, y=y_data_arr, label='Simulation Data', color='blue')
            axes[i,0].plot(x_arr, y_arr, color='red', label='Polynomial Fit')

            axes[i,0].set_ylim(get_range(y_data_arr, fract_range=0.1))

            axes[i,0].set_xlabel('E Field Amplitude$^2$ [(V/cm)$^2$]')
            axes[i,0].set_ylabel('FOSOF Phase [mrad]')
            axes[i,0].legend()
            axes[i,0].set_title('Frequency [MHz]: ' + str(self.residuals_e_field_df.index.values[i]))

            residual_data_arr = y_data_arr-poly_fit_func(x_data_arr)*1E3

            axes[i,1].bar(x=x_data_arr, height=residual_data_arr, width=20)

            axes[i,1].set_ylim(get_range(residual_data_arr, fract_range=0.1))

            axes[i,1].set_xlabel('E Field Amplitude$^2$ [(V/cm)$^2$]')
            axes[i,1].set_ylabel('Residual [mrad]')
            axes[i,1].set_title('Frequency [MHz]: ' + str(self.residuals_e_field_df.index.values[i]))

            axes[i,1].grid()

        return axes

    def calc_fosof_interpolation_unc(self):
        ''' We want to understand how well our second-order polynomial fit to FOSOF phase vs E field amplitude fits the simulation data. The devised test is the following. We first group the simulation data by whether it is the new simulation or the old simulation, by its Speed, Waveguide Separation, and off-axis distance. Now for each group we select the distinct frequencies. The parameters of the group, including the list of frequencies are used as the dictionary for the function that selects the respective FOSOF data sets.

        After we apply our interpolating function that determines the FOSOF phase, given off-axis distance, frequency, and the electric field amplitude. Now, for each simulation key we use this interpolating function to calculate the FOSOF phase for each off-axis distance, corresponding waveguide electric field amplitude, and every frequency associated with that particular simulation key. All of this is used to constuct the FOSOF lineshape for each off-axis distance and waveguide electric field amplitude value. We can then calculate the zero-crossing frequency, which is compared to the zero-crossing frequency determined from the simulation directly. For given chosen set of simulation keys, the maximum deviation from the simulation zero-crossing can be used as the systematic uncertainty in our interpolating method.

        Note that this is very important to call this function BEFORE performing interpolation to the requested set of FOSOF frequencies. Otherwise the interpolation data for the simulation data will overwrite the interpolation for the previously requested set of FOSOF frequencies.
        '''
        # Combine all of the simulation data into a single dataframe.
        fosof_sim_full_data_df = self.fosof_sim_data_df.join(self.fosof_sim_info_df).reset_index().set_index(['Old Simulation', 'Speed [cm/ns]', 'Waveguide Separation [cm]', 'Off-axis Distance [mm]'])

        # Group the simulation data by its distinct simulation type (old or new), speed, waveguide separation, and off-axis distance.
        fosof_sim_full_data_grouped_df = fosof_sim_full_data_df.groupby(fosof_sim_full_data_df.index.names)

        def calc_interp_fosof_phase(df, phase_vs_e_field_poly_fit_df):
            ''' Calculates FOSOF phase using the interpolating function that depends on the off-axis distance value, frequency and the electric field amplitude.

            This function is used for calculating FOSOF phases that are for the specific off-axis distance, frequencies and the electric field amplitude values given in the simulation data. This way we can compare how well the interpolating function matches the simulation data for the particular off-axis distance.
            '''
            sim_name = df.iloc[[0]].index.get_level_values('Simulation Key')[0]
            e_field = self.fosof_sim_info_df.loc[sim_name]['Waveguide Electric Field [V/cm]']

            df = df[['Frequency [MHz]']]
            initial_index_name_arr = df.index.names

            sim_freq_arr = df['Frequency [MHz]'].values

            fosof_phase_df = phase_vs_e_field_poly_fit_df.loc[sim_freq_arr, ['Polynomial Fit Function']].transform(lambda col: list(map(lambda x: x(e_field**2), col))).rename(columns={'Polynomial Fit Function': 'Phase [Rad]'})

            return df.reset_index().set_index('Frequency [MHz]').join(fosof_phase_df).reset_index().set_index(initial_index_name_arr)

        # We determine the FOSOF phases corresponding to the simulation type (old or new), electric field value, off-axis distance, and frequencies of the simulation keys contained in the group.

        fosof_interpolated_data_df = pd.DataFrame()

        for name, df in fosof_sim_full_data_grouped_df:
            # Array of distinct frequencies associated with the given set of simulations
            freq_arr = df['Frequency [MHz]'].drop_duplicates().values

            # Very complicated-looking expression that simply returns a dictionary of index level names and corresponding values
            group_param_dict = dict(list(zip(*[list(df.iloc[[0]].index.names), list(list(df.iloc[[0]].index.values)[0])])))

            # Constructing the dictionary for using in the function that selects the appropriate simulations.
            group_param_dict['Frequency Array [MHz]'] = freq_arr

            sim_params_dict = group_param_dict
            # Determining the interpolating function (Phase vs E Field amplitude) for each specified frequency.
            zero_crossing_vs_off_axis_dist_chosen_df, fosof_sim_info_chosen_df = self.filter_fosof_sim_set(sim_params_dict)

            # Performe interpolating fits to the FOSOF phase vs electric field amplitude data
            phase_vs_e_field_poly_fit_df = self.get_e_field_func()

            # Select the FOSOF simulation data that is related to the chosen simulation keys for given simulation parameters
            fosof_sim_data_to_interpolate_df = self.fosof_sim_data_df.loc[fosof_sim_info_chosen_df.index]

            # Select the off-axis distance for which we have performed the FOSOF phase vs electric field amplitude fit
            fosof_sim_data_to_interpolate_df = fosof_sim_data_to_interpolate_df[fosof_sim_data_to_interpolate_df['Off-axis Distance [mm]'] == sim_params_dict['Off-axis Distance [mm]']]

            # Group the selected fosof data by their simulation keys and off-axis distance
            fosof_sim_data_grouped_df = fosof_sim_data_to_interpolate_df.set_index(keys=['Off-axis Distance [mm]'] , append=True).groupby(['Simulation Key', 'Off-axis Distance [mm]'])

            fosof_interpolated_df = fosof_sim_data_grouped_df.apply(lambda df: calc_interp_fosof_phase(df, phase_vs_e_field_poly_fit_df))
            fosof_interpolated_data_df = fosof_interpolated_data_df.append(fosof_interpolated_df)

        # Calculate the interpolated zero-crossing frequencies.
        fosof_interpolated_zero_crossing_df = fosof_interpolated_data_df.groupby(['Simulation Key', 'Off-axis Distance [mm]']).apply(self.calc_fosof_res_freq)

        # Determine the difference between the zero crossing frequencies determined from the simulations and from the interpolations.
        zero_crossing_diff_df = self.zero_crossing_sim_df.loc[self.zero_crossing_sim_df.index.intersection(fosof_interpolated_zero_crossing_df.index)]

        zero_crossing_diff_df['Resonant Frequency Deviation [kHz]'] = zero_crossing_diff_df['Resonant Frequency Offset [kHz]'] - fosof_interpolated_zero_crossing_df['Resonant Frequency Offset [kHz]']

        zero_crossing_diff_df['Fractional Slope Deviation [ppt]'] = (1 - fosof_interpolated_zero_crossing_df['Slope [Rad/MHz]']/zero_crossing_diff_df['Slope [Rad/MHz]']) * 1E3

        zero_crossing_diff_df = zero_crossing_diff_df[['Resonant Frequency Deviation [kHz]', 'Fractional Slope Deviation [ppt]']]

        self.zero_crossing_diff_df = zero_crossing_diff_df

        return self.zero_crossing_diff_df

    def get_fosof_interpolation_unc(self):
        return self.zero_crossing_diff_df

    def calc_interp_FOSOF_lineshape_params(self, e_field_ampl):
        ''' Provides maximum deviation from the simulation data of the interpolated zero-crossing frequency detetermined from the previously chosen set of simulation keys, and the parameters of the FOSOF lineshapes fit to this interpolated data for the requested E field amplitude.
        '''

        fosof_phase_df = self.phase_vs_e_field_poly_fit_df.loc[self.freq_arr, ['Polynomial Fit Function']].transform(lambda col: list(map(lambda x: x(e_field_ampl**2), col))).rename(columns={'Polynomial Fit Function': 'Phase [Rad]'})

        chosen_sim_data_diff_df = self.get_fosof_interpolation_unc().reset_index().set_index('Simulation Key').loc[self.fosof_sim_info_chosen_df.index].set_index('Off-axis Distance [mm]', append=True).sort_index()

        self.interp_fosof_lineshape_param_s = self.calc_fosof_res_freq(fosof_phase_df.reset_index()).append(np.max(np.abs(chosen_sim_data_diff_df)))

        return self.interp_fosof_lineshape_param_s

    def get_interp_FOSOF_lineshape_params(self):
        return self.interp_fosof_lineshape_param_s


    def calc_FOSOF_phase(self, freq_value, e_field_ampl):
        ''' Auxiliary function for calculating the FOSOF phase [Rad] at the requested frequency [MHz] and E field amplitude [V/cm].

        The frequency has to be a member of the input array of frequencies for the FOSOF phase vs E field amplitude interpolation.
        '''
        return self.phase_vs_e_field_poly_fit_df.loc[freq_value, 'Polynomial Fit Function'](e_field_ampl**2)

    def interp_FOSOF_phi_for_E_field(self, freq_value, fosof_phase, e_field_ampl_det, e_field_ampl_needed, slope_pos_Q):
        ''' Interpolates FOSOF phase at given frequency, and electric field value to the FOSOF phase that one would expect at another electric field amplitude.

        This function is important for the experiment data, where it is possible for the detected E field amplitude (e_field_ampl_det) in the waveguides to be different from the expected value (e_field_ampl_needed). And thus one can correct the FOSOF data for its imperfect RF power in the waveguides. The function is using interpolated functions for FOSOF phase vs E field valu determined previously for the given set of frequencies, beam radius or off-axis value, nominal accelerating voltage or beam speed, and the waveguide separation.

        Inputs:
        :freq_value: FOSOF frequency [MHz]. Needs to be the one previously specified in the list of interpolating frequencies.
        :fosof_phase: current value of the fosof phase [Rad] that needs to be interpolated to the required power
        :e_field_ampl_det: E field amplitude [V/cm] for which the FOSOF phase is given
        :e:field_ampl_needed: E field amplitude [V/cm] for which one wants to finds the needed FOSOF phase.
        :slope_pos_Q: boolean for whether the FOSOF lineshape slope is positive. This is needed for applying correct sign to the correction, since the interpolated lineshape slope sign might be different from the lineshape slope sign of the input FOSOF phase.
        '''

        # In case of blinded frequency we unblind it first
        if self.blind_freq_Q:
            freq_value = freq_value + self.__BLIND_OFFSET_VALUE

        delta_phi = self.calc_FOSOF_phase(freq_value, e_field_ampl_det) - self.calc_FOSOF_phase(freq_value, e_field_ampl_needed)

        fosof_phase_corrected = fosof_phase - delta_phi

        fosof_lineshape_params = self.get_interp_FOSOF_lineshape_params()

        if ((slope_pos_Q == True) and (fosof_lineshape_params['Slope [Rad/MHz]'] <= 0)) or ((slope_pos_Q == False) and (fosof_lineshape_params['Slope [Rad/MHz]'] >= 0)):
            fosof_phase_corrected = -1 * fosof_phase_corrected

        return fosof_phase_corrected

    def load_instance(self):
        ''' This function loads previously pickled class instance.

        The data contained in the pickled file gets loaded into this class instance. The function is very useful if one does not want to reanalyze data again.
        '''
        os.chdir(self.saving_folder_location)
        os.chdir(self.fosof_sim_data_folder_name)

        f = open(self.analysis_data_file_name, 'rb')
        loaded_dict = pickle.load(f)
        f.close()
        self.__dict__.update(loaded_dict)
        print('The class instance has been loaded')

        os.chdir(self.saving_folder_location)

    def save_instance(self, rewrite_Q=True):
        ''' Calling this function pickles the analysis class instance. If the data has been previously saved, the call to this function overwrites previously written pickled file with the same file name.
        '''

        already_exist_Q = False
        # Created folder that will contain all of the analyzed data
        os.chdir(self.saving_folder_location)

        if os.path.isdir(self.fosof_sim_data_folder_name) == False:
            os.mkdir(self.fosof_sim_data_folder_name)
        else:
            print('Saving data folder already exists.')
            already_exist_Q = True

        if (already_exist_Q and rewrite_Q) or (already_exist_Q == False):
            os.chdir(self.fosof_sim_data_folder_name)

            f = open(self.analysis_data_file_name, 'wb')
            pickle.dump(self.__dict__, f, 2)

            os.chdir(self.saving_folder_location)

            print('The class instance has been saved')
#%%
# fosof_sim_data = FOSOFSimulation(load_Q=True)
# #%%
# fosof_sim_data_df = fosof_sim_data.load_fosof_sim_data()
# #%%
# sim_name = 'FOSOF-04-08-16-3223-003'
# freq_arr = fosof_sim_data_df.loc[sim_name].set_index('Off-axis Distance [mm]', append=True).loc[(sim_name, 1.414)]['Frequency [MHz]'].values
# phase_arr = fosof_sim_data_df.loc[sim_name].set_index('Off-axis Distance [mm]', append=True).loc[(sim_name, 1.414)]['Phase [Rad]'].values
# fit_coeff = np.polyfit(freq_arr, phase_arr, 1)
# -fit_coeff[1]/fit_coeff[0]
# #%%
# sim_name = 'FOSOF-04-08-16-3223-004'
# freq_arr = fosof_sim_data_df.loc[sim_name].set_index('Off-axis Distance [mm]', append=True).loc[(sim_name, 1.414)]['Frequency [MHz]'].values
# phase_arr = fosof_sim_data_df.loc[sim_name].set_index('Off-axis Distance [mm]', append=True).loc[(sim_name, 1.414)]['Phase [Rad]'].values
# fit_coeff = np.polyfit(freq_arr, phase_arr, 1)
# -fit_coeff[1]/fit_coeff[0]
# #%%
# sim_name = 'FOSOF-04-08-16-3223-001'
# freq_arr = fosof_sim_data_df.loc[sim_name].set_index('Off-axis Distance [mm]', append=True).loc[(sim_name, 1.414)]['Frequency [MHz]'].values
# phase_arr = fosof_sim_data_df.loc[sim_name].set_index('Off-axis Distance [mm]', append=True).loc[(sim_name, 1.414)]['Phase [Rad]'].values
# fit_coeff = np.polyfit(freq_arr, phase_arr, 1)
# -fit_coeff[1]/fit_coeff[0]
#
# #%%
# off_axis_poly_fit_df = fosof_sim_data.get_off_axis_func()
#
# residuals_df, residuals_fosof_sim_data_df = fosof_sim_data.select_off_axis_large_residuals(min_residual=0.135, max_residual=0.3)
# #%%
# fosof_sim_data.residuals_off_axis_df.shape[0]
# #%%
# fig, axes = plt.subplots(nrows=fosof_sim_data.residuals_off_axis_df.index.values.shape[0], ncols=2)
#
# axes = fosof_sim_data.get_off_axis_data_plots(axes)
#
# fig.set_size_inches(8*2, 8*fosof_sim_data.residuals_off_axis_df.index.values.shape[0])
#
# plt.show()
# #%%
# # Array of the off-axis distances used in the simulations.
# off_axis_dist_arr = fosof_sim_data_df['Off-axis Distance [mm]'].drop_duplicates().values
#
# # For the off-axis distances for which the simulation data is present I want to check the quality of the interpolation to other off-axis distances. For this I use the polynomial fits determined for the FOSOF phase vs off-axis distance and find the difference in the zero-crossing frequencies determined by a linear fit to the simulation data and to the interpolated data.
# fosof_sim_data.calc_fosof_phase_for_off_axis_dist(off_axis_dist_arr)
#
# # Best estimate of the off-axis distance based on the Monte Carlo simulation of the experiment [mm]
# #off_axis_dist_best_guess = 1.6
#
# #phase_vs_off_axis_dist_df = fosof_sim_data.calc_fosof_phase_for_off_axis_dist([off_axis_dist_best_guess])
#
# zero_crossing_vs_off_axis_dist_df = fosof_sim_data.get_off_axis_dist_fosof_res_freq()
#
# max_zero_cross_diff_vs_off_axis_dist = fosof_sim_data.get_max_zero_cross_diff_vs_off_axis_dist()
# #%%
# off_axis_dist_arr
# #%%
# zero_crossing_vs_off_axis_dist_df
# #%%
# # Here we determine the combined quality of the interpolation used for the off-axis distance, the RF E Field dependence, and RF frequency. For each combination of the off-axis distance, E field amplitude, speed, and waveguide separation, the FOSOF phases are calculated using the interpolated functions and the zero-crossing is determined. The difference between the zero-crossings determined from the simulation data and the interpolated data is reported.
# zero_crossing_diff_df = fosof_sim_data.calc_fosof_interpolation_unc()
#
# #%%
# # Resonant frequency deviations for simulations
# df = zero_crossing_diff_df[['Resonant Frequency Deviation [kHz]']].join(fosof_sim_data.fosof_sim_info_df[['Old Simulation']])
# #%%
# df[df['Old Simulation'] == False][['Resonant Frequency Deviation [kHz]']].iloc[0:60]
# #%%
# df[df['Old Simulation'] == False][['Resonant Frequency Deviation [kHz]']].aggregate(lambda x: np.sqrt(np.sum(x**2)/x.shape[0]))
# #%%
# residuals_df, residuals_fosof_sim_data_df = fosof_sim_data.select_e_field_large_residuals(min_residual=0.17, max_residual=0.5)
# #%%
# residuals_df.shape[0]
# #%%
# fig, axes = plt.subplots(nrows=fosof_sim_data.residuals_e_field_df.index.values.shape[0], ncols=2)
#
# axes = fosof_sim_data.get_e_field_data_plots(axes)
#
# fig.set_size_inches(8*2,8*fosof_sim_data.residuals_e_field_df.index.values.shape[0])
#
# plt.show()
# #%%
#
# #fosof_sim_data.save_instance(rewrite_Q=True)
# #%%
# freq_arr = np.linspace(908, 912, 3)
# sim_params_dict = { 'Frequency Array [MHz]': freq_arr,
#                     'Waveguide Separation [cm]': 4,
#                     'Accelerating Voltage [kV]': 49.86,
#                     'Off-axis Distance [mm]': 0.8}
#
# zero_crossing_vs_off_axis_dist_chosen_df, fosof_sim_info_chosen_df = fosof_sim_data.filter_fosof_sim_set(sim_params_dict)
#
# phase_vs_e_field_poly_fit_df = fosof_sim_data.get_e_field_func()
# #%%
#
# 5.35
# 3.4
# #%%
# e_field_ampl = 5
# zero_cross_params_s = fosof_sim_data.calc_interp_FOSOF_lineshape_params(e_field_ampl)
# zero_cross_params_s
# #%%
#
# #%%
# freq_value = 906.0
# e_field_ampl_det = 18.2
# e_field_ampl_needed = 18
# fosof_phase = 0
#
# fosof_sim_data.interp_FOSOF_phi_for_E_field(freq_value, fosof_phase, e_field_ampl_det, e_field_ampl_needed, slope_pos_Q=True)
# #%%
# old_sim_info_df.reset_index()[old_sim_info_df.reset_index()[('Waveguide Electric Field [V/cm]' != 'Power Scan') & ('Waveguide Separation [cm]' == 4)]]
# #%%
# old_sim_info_df
# #%%
# old_quench_sim_info_df = old_sim_info_df[old_sim_info_df['Waveguide Electric Field [V/cm]'] == 'Power Scan']
# #%%
# old_sim_info_df
# #%%
# # Simulation for the beam speed [cm/ns]
# v_speed = 0.3223
# #%%
# old_quench_sim_info_df = old_quench_sim_info_df[old_quench_sim_info_df['Speed [cm/ns]'] == v_speed]
#
# # Waveguide Quenching simulation to use
# old_quench_sim_info_s = old_quench_sim_info_df.iloc[3]
# #%%
# old_quench_sim_info_s
# #%%
# old_quench_sim_set = WaveguideOldQuenchCurveSimulationSet(old_quench_sim_info_s)
# #%%
# sim_name_list = old_quench_sim_set.get_list_of_simulations()
# sim_name_list
# quench_sim_vs_freq_df = old_quench_sim_set.get_simulation_data(sim_name_list[0])
# #%%
# rf_freq = 910.0
#
# quench_sim_data_set = OldWaveguideQuenchCurveSimulation(quench_sim_vs_freq_df.loc[(rf_freq), (slice(None))])
#
# quench_sim_data_set.analyze_data()
# #%%
# fig, axes = plt.subplots(nrows=2, ncols=3)
# fig.set_size_inches(20,12)
# axes = quench_sim_data_set.plot_data(axes=axes, poly_fit_degree=5)
# axes[0, 0].set_xlim(0,1200)
# #axes[0, 0].set_ylim(0.2,0.3)
# plt.show()
# #%%
# fosof_sim_data_info_df = fosof_sim_data.fosof_sim_info_df.copy()
# #%%
# fosof_sim_data_info_new_df = fosof_sim_data_info_df[fosof_sim_data_info_df['Old Simulation'] == False]
# #%%
# fosof_sim_data_info_new_df
# # #%%
# # fosof_sim_data.zero_crossing_sim_df.reset_index().set_index('Simulation Key').loc[fosof_sim_data_info_new_df.index].set_index('Off-axis Distance [mm]', append=True)
# # #%%
# # fosof_sim_data.zero_crossing_sim_df
# #%%
# fosof_sim_data.get_fosof_interpolation_unc().sort_index().reset_index().set_index('Simulation Key').loc[fosof_sim_data_info_new_df.index].set_index('Off-axis Distance [mm]', append=True)
