from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string

path_data_df = pd.read_csv(filepath_or_buffer='path_data.csv', delimiter=',', comment='#', header=[0], skip_blank_lines=True, index_col=[0])

code_folder_path = path_data_df.loc['Code Folder'].values[0].replace('\\', '/')
fosof_analyzed_data_folder_path = path_data_df.loc['FOSOF Analyzed Data Folder'].values[0].replace('\\', '/')
wvg_calib_data_folder_path = path_data_df.loc['Waveguide Calibration Folder'].values[0].replace('\\', '/')
krytar109B_pwr_det_calib_folder_path = path_data_df.loc['KRYTAR 109 B Power Detector Calibration Data Folder'].values[0].replace('\\', '/')
zx47_pd_calib_folder_path = path_data_df.loc['ZX47 Power Sensor Calibration Data Folder'].values[0].replace('\\', '/')


sys.path.insert(0, code_folder_path)

import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt

#%%
# The power calibration is for Mini-Circuits ZX47-55LN-S+ RF Power detectors. Power calibration was obtained with the IFR2026B RF generator. There are visible steps in the calibration, which are due to different attenuators clicking inside the RF generator at certain power levels. This tells us that the absolute power calibration of the RF generator itself is not perfect. The calibration was performed for three different frequencies: 910 MHz, 1088 MHz, and 1147 MHz.

# pd.Series of folder names for power calibration files for the power detectors for various RF frequencies, given in MHz.

pow_det_calib_folder_s = pd.Series(  {
            910: '150928-112257-ZX47-55LN-S_Plus RF sensor calibration for 910MHz analysis',
            1088: '150928-123416-ZX47-55LN-S_Plus RF sensor calibration for 1088MHz analysis',
            1147: '150928-142818-ZX47-55LN-S_Plus RF sensor calibration for 1147MHz analysis'
            })

# Calibration file name
calib_data_file = 'rf_power_sensor_calibration.txt'

calib_data_folder = zx47_pd_calib_folder_path

class ZX4755LNCalibration():
    ''' Class for power calibration of the Mini-Circuits ZX47-55LN-S+ RF power detector.

    The class is used to obtain calibration data and use it to determine what RF power was incident on the detector for given voltage reading.
    '''
    def __init__(self, calib_freq):
        '''Reads the calibration corresponding to RF calibration frequency.

        Inputs:
        :calib_freq: can be of one of 910, 1088, 1147 - calibration frequencies in MHz.
        '''

        self.detector_model = 'Mini-Circuits ZX47-55LN-S+'
        self.quench_cav_calib_folder = pow_det_calib_folder_s[calib_freq]

        os.chdir(calib_data_folder)
        os.chdir(self.quench_cav_calib_folder)

        self.calib_data_df = pd.read_csv(filepath_or_buffer=calib_data_file, sep='\t')

        # Renaming the calibration column names for convenience.
        self.calib_data_df = self.calib_data_df.rename(columns={'RF_power_[dBm]': 'RF Power [dBm]', 'Power_sensor_voltage_[V]': 'Power Detector [V]', 'Power_sensor_voltage_standard_deviation_[V]': 'Power Detector STD [V]'})

        # Sort by Detector voltage. This is needed for the B-spline fit performed later on.
        self.calib_data_df = self.calib_data_df.sort_values(by='RF Power [dBm]')

        # Maximum and minimum values of Detector voltage. Needed for plotting of the 'best fit' spline.

        self.max_det_v = self.calib_data_df['Power Detector [V]'].max()
        self.min_det_v = self.calib_data_df['Power Detector [V]'].min()

    def get_detector_model(self):
        return self.detector_model

    def get_calib_data(self):
        return self.calib_data_df

    def get_calib_curve(self, s_factor_multiple=70):
        '''Obtain smoothed spline fit for the data that gives RF power going into the Power Detector for given voltage reading on the detector

        Inputs:
        :s_factor_multiple: smoothing factor for the B-spline. Larger value means more data smoothing.

        Outputs:
        :spl_smoothing_inverse: B-spline that gives RF power in dBm for given RF power detector voltage in Volts.
        :fract_unc: Estimated fractional uncertainty for each power value obtained from the spline function. The uncertainty is assumed to be not systematic, but random. That is, it is not the same systematic shift for all the powers obtained from the spline function.

        Read comments in the body of the function to understand the logic used for the fitting and uncertainty extraction.
        '''
        # Before I want to use linear spline to interpolate between the data point so that we can convert power detector voltages to corresponding RF powers. The calibration itself has several visible discontinuities in it. This happens when the RF generator switches to a different attenuator at certain output power levels. In a way this tells us that the RF generator does not output exactly correct power or that its power output is not linear across its dynamic range. These discontinuities is the reason for using the linear, not cubic spline; with the cubic and even quadratic splines I noticed some wiggles in the interpolation at these discontinuities. I do not think linear interpolation is a bad choice here, because the calibration data looks quite linear. Also, the density of points is very high: I doubt there are any appreciable second or higher order derivatives between any two consecutive samples.
        # Main problem with using the splines is that in order to use the spline we have to know each of the x-axis and y-axis points exactly. This is, of course, not true, since our power detector voltages obtained during calibration have some uncertainty associated with them. We assume, however, that the uncertainty is so small that it does not have any appreciable effect on the cubic spline. But if we still want to be careful and include the uncertainties, then we cannot really use the usual spline anymore, since we now need to supply the weights. I also noticed that even though uncertainties are small, they are still significant enough so that there were still some points that were visibly off from the general trend, but the spline still passed through them, which is clearly not the right thing to do here.

        # Another problem for this specific calibration data is the set of discontinuities. They clearly tell us that the generator does not output proportional RF power for its whole range. Before I thought that maybe stitching the data was the right thing to do, as described below.
        # -------------------
        # We are fixing the RF power calibration of the ZX47-55LN-S+ RF power sensor. Main issue is the RF generator (IFR2026B) used as the RF power reference. When going from one power setting to another there are cases when the RF generator has to switch to a different configuration of attenuators, because I can hear a click in the RF generator body. Now, these attenuators are not perfectly matched, unfortunately, and as the result we see steps in the calibration.

        # One can check that in order to make the curve continuous, then, for instance, we should shift the second half of the data by about 0.6 dBm. Which is the difference of 15%. This is quite significant, since it means that all of the powers below are systematically 15% off. But it gets even worse, since we see those discontnuities again at higher detector voltages. There are in total 5 visible discontinuities, which roughly turns into almost 100% error for the lowest powers. This is very severe, especially when we are trying to use this calibration to check if the quenching curves obtain from the experiment are matching simulations.

        # This discussion would be identical if we would shift the higher half of the data by 0.6 dBm up. It does not matter as long as we care only about linearity of power, not its absolute value.

        # Another note is that we assume that the discontinuities are not due to the power detectors, but due to the RF generator itself, because there is no physical reason for the power detector to have these sudden jumps.

        # The calibration data strongly shows that for the most part the relationship between the power detector voltages and the RF powers (in dBm) is linear. We can employ this fact to correct the calibration for discontinuities. This can be done in the following way: we can fit small continuous portion of data (y =axis = power detector readings, x-axis = RF power in dBm) to the left of the continuity to a line Same can be done for the data to the right of the jump. Now, the data to the left can be shifted along x-axis until we get a continuous line (more on it later).
        # -------------------
        # I have later realized that this whole idea with stiching would not work, because maybe the reason for seeing the discontinuities is that the RF generator is trying to reduce its output error (the error is such that the data still looks smooth - i.e., we are talking about systematic error). And by not fixing the discontinuities it might make the systematic error in the power output smaller. Thus by removing the discontinuities I might actually make the RF power output linearity even worse.

        # The specs for the generator power output error is +- 1 dBm for the frequency range that we care about.

        # This discussion about data stiching tells us that it seems that the curve that passes in the middle of the calibration data smoothing the discontinuities should give us the best estimate for the correct RF power curve. One can say that it is very similar to stiching and thus the RF power errors are still present, but no, it should be better, assuming that the RF generator designers knew about this issue and they introduced some corrections to these discontinuities that are not of the 'sudden jump' type, but are smoothly applied to every power level setting. Thus the smoothing curve should take these assumed corrections, employed by the designers, into account. Of course, this is not rigorously justified.

        # I will use smoothing spline = B-spline, that also accepts weights. I am not sure exactly how it works, but this kind of spline will, in addition of trying to pass through all the points, will also try to minimize the curvature of the resulting curve. This is exactly what I need for smoothing the data.

        # Smoothing cubic spline, where y-axis = RF power detector voltages, and x-axis = RF power in dBm. Notice that the weights are 1/sigma, not 1/sigma**2, as specified in the function docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html#scipy.interpolate.splrep

        # Determining the min and max values for the smoothing factor
        m = self.calib_data_df['Power Detector STD [V]'].shape[0]
        s_min = m - np.sqrt(2*m)
        s_max = m + np.sqrt(2*m)

        # Notice that we are using s_factor_multiple*s_max for smoothing factor.
        self.spl_smoothing_best_estimate = scipy.interpolate.UnivariateSpline(x=self.calib_data_df['RF Power [dBm]'].values, y=self.calib_data_df['Power Detector [V]'].values, w=1/self.calib_data_df['Power Detector STD [V]'].values, k=3, s=s_factor_multiple*s_max)

        # Now the obtained smoothing spline is not really what we need, because the uncertainties are given for Power detector voltages that we want to be the x-axis values. Because at the end we want to determine the RF power going into the RF power detector from the RF power detector voltage. I can do, however, the following. I can use this weighted smoothed spline to calculate power detector voltages at all RF power settings of the calibration generator. These values will be the best estimate RF power detector voltages for the smoothing spline. I can now reverse the axes, where for the detector voltages I will use these values from the smoothing spline. For this reversed axis situation I will now fit a smoothing spline again. However, this time no smoothing will be applied. The idea is that this spline now will simply allow me to get RF power for given power detector voltage.

        det_pwr_volt_best_estimate_arr = self.spl_smoothing_best_estimate(self.calib_data_df['RF Power [dBm]'])

        # Sorting indeces
        pwr_det_volt_best_est_arr_sort_index = np.argsort(det_pwr_volt_best_estimate_arr)

        # Smoothing spline for the inverse situation: RF power vs RF power detector reading
        self.spl_smoothing_inverse = scipy.interpolate.UnivariateSpline(x=det_pwr_volt_best_estimate_arr[pwr_det_volt_best_est_arr_sort_index], y=(self.calib_data_df['RF Power [dBm]'].values)[pwr_det_volt_best_est_arr_sort_index], k=3, s=s_factor_multiple*s_max)

        # We now need to understand how well do we know RF power from RF Power detector voltages. The issue is that the RF generator itself is not a good RF power reference. To deal with this we assume that on average the generator is outputting power that maybe does not have good absolute accuracy, but the power linearity is good on average. We also assume that the power detector has true calibration with its the voltage reading strictly smoothly decreasing with the RF power in dBm. Thus a smooth curve through the Power detector voltage vs RF power should be the best estimate of the calibration curve.
        # We then invert the spline = make it as power detector voltage vs RF power. On average we assume that each RF power determined from the Power detector voltage reading has the same uncertainty in dB. This uncertainty can be estimated from looking at the residuals of the spline from the actual data. We can then determine the best estimate for the uncertainty, by calculating the pooled standard deviation.
        # What is the justification for calculating the uncertainty in this way? Well, from our discussion we expect the reduced chi-squared of the spline to be about 1. Assuming that at every RF power reading the standard deviation in RF power is the same, then we see right away that our expression is correct.
        # Notice that we calculate the deviation: spline - RF_power [dBm]


        residual_arr = self.spl_smoothing_inverse(self.calib_data_df['Power Detector [V]'].values) - self.calib_data_df['RF Power [dBm]'].values
        self.calib_data_df['Fit Residual [dB]'] = residual_arr

        # Here we have dof's = N-1, where 1 degree of freedom goes for calculating N.
        pooled_calib_std = np.sqrt(np.sum(residual_arr**2)/(residual_arr.shape[0]-1))

        # This uncertainty is given in dB. We can convert it to fractional uncertainty in power.

        lower_fract_unc = 1-10**(pooled_calib_std/10)
        higher_fract_unc = 1-10**(-pooled_calib_std/10)

        # Since the uncertainty in dB is relatively small, then lower and higher limits are not very different from each other. We can take RMS average of them:
        fract_unc = np.sqrt((lower_fract_unc**2 + higher_fract_unc**2)/2)

        # It seems that the smoothing spline can give extrapolated value outside its interpolation range.
        return self.spl_smoothing_inverse, fract_unc

    def get_spline_data_to_plot(self, n_points):
        ''' Convenience function that gives x and y-axes data for plotting the spline

        Inputs:
        :n_points: integer specifying number of plotting points between maximum and minimum Power detector voltage readings for the calibration data.

        Outputs:
        :x_data: array of generated Power detector voltage
        :y_data: array of corresponding calculated RF powers in [dBm]
        '''

        x_data = np.linspace(self.min_det_v, self.max_det_v, n_points)
        y_data = self.spl_smoothing_inverse(x_data)

        return x_data, y_data
#%%
# data_set = ZX4755LNCalibration(1088)
#
# spl_smoothing_inverse, fract_unc = data_set.get_calib_curve()
# x_data, y_data = data_set.get_spline_data_to_plot(1000)
#
# calib_data_df = data_set.get_calib_data()
#
# fig, axes = plt.subplots(nrows=1, ncols=2)
# fig.set_size_inches(20,9)
# axes[0].plot(x_data, y_data, color='C3', label='Cubic Smoothing Spline')
#
# calib_data_df.plot(x='Power Detector [V]', y='RF Power [dBm]', kind='scatter', ax=axes[0], xerr='Power Detector STD [V]', color='C0', s=30, label='Calibration data')
#
# axes[0].set_title(data_set.get_detector_model() + ' RF power detector calibration')
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
# #axes[0].set_xlim(-0.2, -1)
# #axes[0].set_ylim(0,12)
# plt.show()
