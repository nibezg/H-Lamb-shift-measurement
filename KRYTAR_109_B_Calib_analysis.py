from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string

# For lab
sys.path.insert(0,"C:/Users/Helium1/Google Drive/Research/Lamb shift measurement/Code")

import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt

from ZX47_Calibration_analysis import *
#%%
# Path to the power detector calibration
calib_folder_path = 'C:/Users/Helium1/Google Drive/Research/Lamb shift measurement/Data'

# KRYTAR 109B Power detector calibration folder name
calib_folder = '170822-130101 - RF power detector calibration'

calib_filename = 'KRYTAR 109B power calibration.CSV'

class KRYTAR109BCalibration(ZX4755LNCalibration):
    ''' Class for power calibration of the KRYTAR 109B RF power detector.

    The class is used to obtain calibration data and use it to determine what RF power was incident on the detector for given voltage reading. It uses the same analysis method as the ZX4755LNCalibration class.
    '''
    def __init__(self):
        '''Reads the calibration corresponding to RF calibration frequency.

        Inputs:
        :calib_freq: can be of one of 910, 1088, 1147 - calibration frequencies in MHz.
        '''

        self.detector_model = 'KRYTAR 109B'

        os.chdir(calib_folder_path)
        os.chdir(calib_folder)

        # Import the CSV file with the detector power calibration taken at 910 MHz.
        self.calib_data_df = pd.read_csv(filepath_or_buffer=calib_filename, delimiter=',', comment='#', header=0)

        self.calib_data_df.rename(columns={
            'RF power [dBm]': 'RF Power [dBm]',
            'STD the mean of power detector signal [V]':'Power Detector STD [V]',
            'Power detector signal [V]': 'Power Detector [V]'
            }, inplace=True)

        # The detector is useful only to measuring powers above about -30 dBm. The calibration that we performed went all the way down to -50 dBm, which gave extremely small voltages that are almost the same for -50 to -40 dBm - they seem to be simply the offset of the Keithley multimeter. I consider this data as unreliable and throw it away. In any case we do not measure powers that small in the experiment.
        self.calib_data_df = self.calib_data_df[self.calib_data_df['RF Power [dBm]'] >= -30]

        self.calib_data_df['RF Power [mW]'] = self.calib_data_df['RF Power [dBm]'].transform(lambda x: (10**(x/10)))

        self.calib_data_df['RF rt-Power [rt-mW]'] = self.calib_data_df['RF Power [mW]'].transform(lambda x: np.sqrt(x))

        self.max_det_v = self.calib_data_df['Power Detector [V]'].max()
        self.min_det_v = self.calib_data_df['Power Detector [V]'].min()

        self.spl_smoothing_inverse = None


    def get_calib_curve(self, s_factor_multiple=50):
        '''Obtain smoothed spline fit for the data that gives RF power going into the Power Detector for given voltage reading on the detector

        Inputs:
        :s_factor_multiple: smoothing factor for the B-spline. Larger value means more data smoothing.

        Outputs:
        :spl_smoothing_inverse: B-spline that gives RF power in mW for given RF power detector voltage in Volts.
        :fract_unc: Estimated fractional uncertainty for each power value obtained from the spline function. The uncertainty is assumed to be not systematic, but random. That is, it is not the same systematic shift for all the powers obtained from the spline function.

        Read comments in the body of the function to understand the logic used for the fitting and uncertainty extraction.
        '''
        # Read the comments in the corresponding function in the ZX4755LNCalibration class to understand the analysis method. The analysis for the KRYTAR 109B power detector is identical with one important difference: we are fitting to the RF power in mW, NOT dBm. The reason for this is that, for example, the ZX47-55LN+ power detector is designed to give voltage linear with the input RF power in dBm scale. However, the Schottky diode (KRYTAR 109B) gives voltage that is proportional to the current that flows through it. This current is proportional to the square root of the RF power.

        if self.spl_smoothing_inverse is None:

            # Smoothing cubic spline, where y-axis = RF power detector voltages, and x-axis = RF rt-power in rt-mW. Notice that the weights are 1/sigma, not 1/sigma**2, as specified in the function docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html#scipy.interpolate.splrep

            # Determining the min and max values for the smoothing factor
            m = self.calib_data_df['Power Detector STD [V]'].shape[0]
            s_min = m - np.sqrt(2*m)
            s_max = m + np.sqrt(2*m)
            # Notice that we are using s_factor_multiple*s_max for smoothing factor.

            x_data_arr = self.calib_data_df['RF rt-Power [rt-mW]'].values
            y_data_arr = self.calib_data_df['Power Detector [V]'].values
            y_data_std_arr = self.calib_data_df['Power Detector STD [V]'].values

            self.spl_smoothing_best_estimate = scipy.interpolate.UnivariateSpline(x=x_data_arr, y=y_data_arr, w=1/y_data_std_arr, k=3, s=s_factor_multiple*s_max)

            # Now the obtained smoothing spline is not really what we need, because the uncertainties are given for Power detector voltages that we want to be the x-axis values. Because at the end we want to determine the RF power going into the RF power detector from the RF power detector voltage. I can do, however, the following. I can use this weighted smoothed spline to calculate power detector voltages at all RF power settings of the calibration generator. These values will be the best estimate RF power detector voltages for the smoothing spline. I can now reverse the axes, where for the detector voltages I will use these values from the smoothing spline. For this reversed axis situation I will now fit a smoothing spline again. However, this time no smoothing will be applied. The idea is that this spline now will simply allow me to get RF power for given power detector voltage.

            x_arr = np.linspace(np.min(x_data_arr), np.max(x_data_arr), 10*x_data_arr.shape[0])

            det_pwr_volt_best_estimate_arr = self.spl_smoothing_best_estimate(x_arr)

            # Sorting indeces
            pwr_det_volt_best_est_arr_sort_index = np.argsort(det_pwr_volt_best_estimate_arr)

            # Non smoothing spline for the inverse situation: RF power vs RF power detector reading
            self.spl_smoothing_inverse = scipy.interpolate.UnivariateSpline(x=det_pwr_volt_best_estimate_arr[pwr_det_volt_best_est_arr_sort_index], y=x_arr[pwr_det_volt_best_est_arr_sort_index], k=3, s=0)

            # We now need to understand how well do we know RF power from RF Power detector voltages. The issue is that the RF generator itself is not a good RF power reference. To deal with this we assume that on average the generator is outputting power that maybe does not have good absolute accuracy, but the power linearity is good on average. We also assume that the power detector has true calibration with its the voltage reading strictly smoothly decreasing with the RF power in dBm. Thus a smooth curve through the Power detector voltage vs RF power should be the best estimate of the calibration curve.
            # We then invert the spline = make it as power detector voltage vs RF power. On average we assume that each RF power determined from the Power detector voltage reading has the same uncertainty in dB. This uncertainty can be estimated from looking at the residuals of the spline from the actual data. We can then determine the best estimate for the uncertainty, by calculating the pooled standard deviation.
            # What is the justification for calculating the uncertainty in this way? Well, from our discussion we expect the reduced chi-squared of the spline to be about 1. Assuming that at every RF power reading the standard deviation in RF power is the same, then we see right away that our expression is correct.
            # Notice that we calculate the deviation: spline - RF rt-power [rt-mW]

            self.calib_data_df['Calibration RF rt-Power [rt-mW]'] = self.calib_data_df['Power Detector [V]'].transform(lambda x: self.spl_smoothing_inverse(x))

            self.calib_data_df['Calibration RF Power [mW]'] = self.calib_data_df['Calibration RF rt-Power [rt-mW]']**2

            residual_arr = self.spl_smoothing_inverse(self.calib_data_df['Power Detector [V]'].values) - self.calib_data_df['RF rt-Power [rt-mW]'].values
            self.calib_data_df['Fit Residual [rt-mW]'] = residual_arr

            self.calib_data_df['Fit Residual [mW]'] = self.calib_data_df['Calibration RF Power [mW]'] - self.calib_data_df['RF Power [mW]']

            # Here we have dof's = N-1, where 1 degree of freedom goes for calculating N.
            pooled_calib_std = np.sqrt(np.sum(residual_arr**2)/(residual_arr.shape[0]-1))

            # This uncertainty is given in rt-mW. We can convert it to fractional uncertainty in power.
            self.calib_data_df['Calibration RF Power Max Limit [mW]'] = (self.calib_data_df['Calibration RF rt-Power [rt-mW]'] + pooled_calib_std)**2
            self.calib_data_df['Calibration RF Power Min Limit [mW]'] = (self.calib_data_df['Calibration RF rt-Power [rt-mW]'] - pooled_calib_std)**2

            self.calib_data_df['Calibration RF Power Max Deviation [mW]'] = self.calib_data_df['Calibration RF Power Max Limit [mW]'] - self.calib_data_df['Calibration RF Power [mW]']

            self.calib_data_df['Calibration RF Power Min Deviation [mW]'] = self.calib_data_df['Calibration RF Power [mW]'] - self.calib_data_df['Calibration RF Power Min Limit [mW]']

            self.calib_data_df['Calibration RF Power STD [mW]'] = self.calib_data_df[['Calibration RF Power Min Deviation [mW]', 'Calibration RF Power Max Deviation [mW]']].T.aggregate(lambda x: np.max(x))

            self.calib_data_df['Calibration RF Power Fractional STD [mW]'] = self.calib_data_df['Calibration RF Power STD [mW]'] / self.calib_data_df['Calibration RF Power [mW]']

            self.calib_data_df = self.calib_data_df.sort_values(by='Power Detector [V]')

            # Non-smoothing spline for determining uncertainty in the RF powers from the calibration function

            self.spl_calib_std = scipy.interpolate.UnivariateSpline(x=self.calib_data_df['Power Detector [V]'].values, y=self.calib_data_df['Calibration RF Power STD [mW]'].values, k=3, s=0)

        # IMPORTANT!!! It seems that the smoothing spline can give extrapolated value outside its interpolation range.
        return self.spl_smoothing_inverse, self.spl_calib_std

    def get_RF_power_from_voltage(self, v_array):
        ''' Get RF power in mW and its uncertainty for an array of voltage readings
        '''

        return np.array([self.spl_smoothing_inverse(v_array)**2, self.spl_calib_std(v_array)]).T

    def get_RF_power_dBm_from_voltage(self, v_array):
        ''' Get RF power in dBm and its uncertainty for an array of voltage readings
        '''
        return 10*np.log10(self.get_RF_power_from_voltage(v_array))

    def get_spline_data_to_plot(self, n_points):
        ''' Convenience function that gives x and y-axes data for plotting the spline

        Inputs:
        :n_points: integer specifying number of plotting points between maximum and minimum Power detector voltage readings for the calibration data.

        Outputs:
        :x_data: array of generated Power detector voltage
        :y_data: array of corresponding calculated RF powers in [mW]
        '''

        x_data = np.linspace(self.min_det_v, self.max_det_v, n_points)

        calib_data = self.get_RF_power_from_voltage(x_data)

        y_data = calib_data[:, 0]
        y_std_data = calib_data[:, 1]

        return x_data, y_data, y_std_data
#%%
# data_set = KRYTAR109BCalibration()
# 
# # s_factor_multiple of 50 seems to give good smooth agreement with the data.
# spl_smoothing_inverse, spl_calib_std = data_set.get_calib_curve()
# x_data, y_data, y_std_data = data_set.get_spline_data_to_plot(100)
# calib_data_df = data_set.get_calib_data()
#
# fig, axes = plt.subplots(nrows=2, ncols=2)
# fig.set_size_inches(20,20)
#
# axes[0, 0].plot(x_data, np.sqrt(y_data), color='C3', label='Cubic Smoothing Spline')
#
# calib_data_df.plot(x='Power Detector [V]', y='RF rt-Power [rt-mW]', kind='scatter', ax=axes[0, 0], xerr='Power Detector STD [V]', color='C0', s=30, label='Calibration data')
#
# axes[0, 0].set_title(data_set.get_detector_model() + ' RF power detector calibration')
# axes[0, 0].grid()
# axes[0, 0].legend()
#
# #residual_arr = power_det_calib_func(calib_data_df['Power Detector [V]'].values) - calib_data_df['RF Power [dBm]'].values
#
# calib_data_df.plot(x='Power Detector [V]', y='Fit Residual [rt-mW]', kind='scatter', ax=axes[0, 1], xerr='Power Detector STD [V]', color='C2', s=30)
#
# #ax.set_xlim(min_det_v, max_det_v)
# axes[0, 1].set_title('Spline and data residuals with power in rt-mW')
# axes[0, 1].grid()
# axes[0, 1].set_xlabel('Power Detector [V]')
# axes[0, 1].set_ylabel('Best fit curve - RF rt-Power [rt-mW] residual')
#
# calib_data_df.plot(x='Power Detector [V]', y='RF Power [mW]', kind='scatter', ax=axes[1, 0], xerr='Power Detector STD [V]', color='C0', s=30, label='Calibration data')
#
# axes[1, 0].set_title(data_set.get_detector_model() + ' RF power detector calibration')
# axes[1, 0].grid()
# axes[1, 0].legend()
#
# axes[1, 0].errorbar(x_data, y_data, yerr=y_std_data, color='C3', label='Cubic Smoothing Spline')
#
# calib_data_df.plot(x='Power Detector [V]', y='Fit Residual [mW]', kind='scatter', ax=axes[1, 1], xerr='Power Detector STD [V]', yerr='Calibration RF Power STD [mW]', color='C2', s=30)
#
# #ax.set_xlim(min_det_v, max_det_v)
# axes[1, 1].set_title('Spline and data residuals with power in mW')
# axes[1, 1].grid()
# axes[1, 1].set_xlabel('Power Detector [V]')
# axes[1, 1].set_ylabel('Best fit curve - RF Power [mW] residual')
#
# #axes[0, 0].set_xlim(-0.5, -0.4)
# #axes[0, 0].set_ylim(1.75,2.25)
# #axes[1, 0].set_xlim(-0.08, -0.06)
# #axes[1, 0].set_ylim(0, 0.5)
#
# plt.show()
# #%%
# data_set.get_RF_power_from_voltage([-0.1])
