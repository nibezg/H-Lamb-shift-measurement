from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt

sys.path.insert(0,"C:/Users/Helium1/Google Drive/Research/Lamb shift measurement/Code")

#%%
# Path to the folder that stores FOSOF simulation files
sim_data_path = 'C:/Research/Lamb shift measurement/Data/Simulation data'
# Old Simulation path that did not average phases between the quenching cavities and the SOF waveguides.
old_sim_folder_path = 'Old Lamb Shift Data (No Phase Averaging)'
# File that stores information about all of the old simulations
old_sim_info_file_name = 'FOSOF Simulations for Hydrogen.xlsx'


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
        # We pick a surviving fraction data point from the experiment data. We then determine to what RF power this surviving fraction value correspond to on the quench curve from the simulation. Thus it means that we need to parameterize the simulation quench curve, that is not in the form of 'Surviving fraction vs RF Power', but in the form of 'RF Power vs Surviving Fraction'. This sounds quite easy to do. However, there is a problem. In general a quenching curve has several pi-pulses. Thus for some Surviving fraction values there are multiple corresponding RF powers. In other words, if we look a the simulation quench curve as 'RF power vs Suriving fraction' then the parameterizing function will not be single valued.
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
        '''
        if self.spl_inverted is None:
            self.analyze_data()

        return self.spl(RF_power_arr)

    def get_RF_power(self, surv_frac_arr):
        ''' Calculates value proportional to RF power in units of V^2/cm^2, given surviving fraction
        '''
        if self.spl_inverted is None:
            self.analyze_data()

        return self.spl_inverted(surv_frac_arr)

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

        # Spline fit of 4th order to the simulated quench data.
        axes[0, 0].plot(x_fit_arr, self.spl(x_fit_arr), color='green', label='Spline Fit')

        axes[0, 0].set_title('Quench curve')
        axes[0, 0].legend()

        axes[0, 1].plot(x_data_arr, (poly_fit_func(x_data_arr)-y_data_arr)/y_data_arr*1E3, marker='.', linestyle='-', color='red', label='Polynomial Fit')
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
#%%
# old_sim_info_df = OldSimInfo().get_info()
#
# old_quench_sim_info_df = old_sim_info_df[old_sim_info_df['Waveguide Electric Field [V/cm]'] == 'Power Scan']
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
# axes[0]
# #%%
# quench_sim_vs_freq_df.loc[910]
