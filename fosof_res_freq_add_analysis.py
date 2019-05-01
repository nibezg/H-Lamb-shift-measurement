
''' Probably unnecessary code taken from fosof_res_freq_analysis.py
    Date: 2019-02-01
    Author: N. Bezginov
'''
#%%
f0-909.8717
#%%
f0_2
#%%
#%%
zero_cross_av_2_df['Resonant Frequency [MHz]'] = zero_cross_av_2_df['Zero-crossing Frequency [MHz]'] - zero_cross_av_2_df['Field Power Shift [MHz]'] + zero_cross_av_2_df['SOD Shift [MHz]']

zero_cross_av_2_df['Resonant Frequency STD [MHz]'] = np.sqrt(zero_cross_av_2_df['Zero-crossing Frequency STD (Normalized) [MHz]']**2 + zero_cross_av_2_df['Field Power Shift Uncertainty [MHz]']**2 + zero_cross_av_2_df['Beam RMS Radius Uncertainty [MHz]']**2 + zero_cross_av_2_df['Fractional Offset Uncertainty [MHz]']**2 + zero_cross_av_2_df['SOD Shift STD [MHz]']**2)
#%%
zero_cross_av_2_df.reset_index().plot(style='.', y='Resonant Frequency [MHz]', yerr='Resonant Frequency STD [MHz]')
#%%
def calc_line_fit_params(x_data_arr, y_data_arr, y_sigma_arr):
    ''' Fits the data to the first-order polynomial. Extracts the slope, offset, and the associated uncertainties. Gives the reduced chi-squared parameter.
    '''
    w_arr = 1/y_sigma_arr**2

    delta_arr = np.sum(w_arr) * np.sum(w_arr*x_data_arr**2) - (np.sum(w_arr*x_data_arr))**2

    offset = (np.sum(w_arr*y_data_arr) * np.sum(w_arr*x_data_arr**2) - np.sum(w_arr*x_data_arr*y_data_arr) * np.sum(w_arr*x_data_arr)) / delta_arr

    offset_sigma = np.sqrt(np.sum(w_arr*x_data_arr**2) / delta_arr)

    slope = (np.sum(w_arr*x_data_arr*y_data_arr) * np.sum(w_arr) - np.sum(w_arr*x_data_arr) * np.sum(w_arr*y_data_arr)) / delta_arr

    slope_sigma = np.sqrt(np.sum(w_arr) / delta_arr)

    fit_param_dict = {'Slope [Rad/MHz]': slope, 'Slope STD [Rad/MHz]': slope_sigma, 'Offset [MHz]': offset, 'Offset STD [MHz]': offset_sigma}

    # For the chi-squared determination.
    fit_data_arr = slope * x_data_arr + offset
    n_constraints = 2
    fit_param_dict = {**fit_param_dict, **get_chi_squared(y_data_arr, y_sigma_arr, fit_data_arr, n_constraints)}

    return fit_param_dict
#%%
data_df = slope_av_2_df[['Slope [Rad/MHz]','Slope STD (Normalized) [Rad/MHz]']].join(zero_cross_av_2_df[['Resonant Frequency [MHz]', 'Resonant Frequency STD [MHz]']])
#%%
data_df['Inverse Slope [MHz/Rad]'] = 1/data_df['Slope [Rad/MHz]']

fit_param_dict = calc_line_fit_params(data_df['Inverse Slope [MHz/Rad]'], data_df['Resonant Frequency [MHz]'], data_df['Resonant Frequency STD [MHz]'])

x_plt_arr = np.linspace(np.min(data_df['Inverse Slope [MHz/Rad]']), 0, data_df['Inverse Slope [MHz/Rad]'].shape[0] * 2)

fit_data_arr = fit_param_dict['Offset [MHz]'] + fit_param_dict['Slope [Rad/MHz]'] * x_plt_arr

av_freq_s = straight_line_fit_params(data_df['Resonant Frequency [MHz]'], data_df['Resonant Frequency STD [MHz]'])

av_freq_plot_arr = np.ones(x_plt_arr.shape[0]) * av_freq_s['Weighted Mean']
#%%
fig, ax = plt.subplots()
ax.plot(x_plt_arr, fit_data_arr)
ax.plot(x_plt_arr, av_freq_plot_arr)
data_df.plot(kind='scatter', x='Inverse Slope [MHz/Rad]', y='Resonant Frequency [MHz]', yerr='Resonant Frequency STD [MHz]', ax=ax)

ax.set_xlim(right=0)
#%%
fit_param_dict
#%%

fit_func = np.poly1d(np.polyfit(data_df['Inverse Slope [MHz/Rad]'], data_df['Resonant Frequency [MHz]'], deg=1, w=1/data_df['Resonant Frequency STD [MHz]']**2))

x_plt_arr = np.linspace(np.min(data_df['Inverse Slope [MHz/Rad]']), np.max(data_df['Inverse Slope [MHz/Rad]']), data_df['Inverse Slope [MHz/Rad]'].shape[0] * 10)

fit_data_arr = fit_func(data_df['Inverse Slope [MHz/Rad]'])

get_chi_squared(data_df['Resonant Frequency [MHz]'], data_df['Resonant Frequency STD [MHz]'], fit_data_arr, 2)
#%%

fig, ax = plt.subplots()
ax.plot(x_plt_arr, fit_func(x_plt_arr))

data_df.plot(kind='scatter', x='Inverse Slope [MHz/Rad]', y='Resonant Frequency [MHz]', yerr='Resonant Frequency STD [MHz]', ax=ax)
#%%
slope_av_2_df[['Slope [Rad/MHz]','Slope STD (Normalized) [Rad/MHz]']].join(zero_cross_av_2_df[['Resonant Frequency [MHz]', 'Resonant Frequency STD [MHz]']])
#%%
data_arr = zero_cross_av_2_df['Resonant Frequency [MHz]']
data_std_arr = zero_cross_av_2_df['Resonant Frequency STD [MHz]']

av_freq_s = straight_line_fit_params(data_arr, data_std_arr)
#%%
909.8274-909.841244
#%%
av_freq_s
#%%
df.reset_index().plot(style='-.', y='Zero-crossing Frequency [MHz]', yerr='Zero-crossing Frequency STD [MHz]')
#%%
df.reset_index().plot(style='-.', y='Zero-crossing Frequency [MHz]', yerr='Zero-crossing Frequency STD (Normalized) [MHz]')
#%%
#%%
data_arr = df['Zero-crossing Frequency [MHz]']
data_std_arr = df['Zero-crossing Frequency STD [MHz]']


av_s
#%%
av_s['Weighted Mean'] - ac_correc['Field Power Shift [kHz]']*1E-3 + 52.58*1E-3 + 0.03174
#%%
data_arr = df['Zero-crossing Frequency [MHz]']
data_std_arr = df['Zero-crossing Frequency STD (Normalized) [MHz]']

av_s = straight_line_fit_params(data_arr, data_std_arr)
av_s
#%%

#%%
av_s['Weighted Mean'] - ac_correc['Field Power Shift [kHz]']*1E-3 + 52.58*1E-3 + 0.03174
#%%
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

#%%
0.03174
#%%
hydrogen_sim_data.FOSOFSimulation.__dict__
#%%

#%%
fosof_sim_data.__dict__
#%%
fosof_sim_data.__dict__
#%%
#%%
# Now it is the time to correct for the SOD.
#%%
fosof_lineshape_vs_offset_set_df.loc[(slice(None), slice(None), slice(None), 1.6, 0.03), ('Resonant Frequency Deviation [kHz]')].reset_index(['Fractional Offset', 'Beam RMS Radius [mm]'], drop=True)
#%%
# Waveguide RF power calibration analyzed data folder path
wvg_power_calib_data_folder_path = 'E:/Google Drive/Research/Lamb shift measurement/Data/Waveguide calibration'

wvg_power_calib_info_file_name = 'Waveguide_calibration_info.csv'

os.chdir(wvg_power_calib_data_folder_path)

# Waveguide power calibration info
wvg_calib_info_df = pd.read_csv(wvg_power_calib_info_file_name, header=2)

wvg_calib_info_df['Start Date'] = pd.to_datetime(wvg_calib_info_df['Start Date'])
wvg_calib_info_df['End Date'] = pd.to_datetime(wvg_calib_info_df['End Date'])
#%%
wvg_calib_info_df
#%%
''' Due to the systematic uncertainty in the offset of 50% we have a systematic error in the power calibration for each waveguide calibration data set. I choose to average all of these calibration errors together for both of the waveguides and all of the frequencies to obtain one averaged fractional power error for each RF power setting.

Now there is also a possibility to obtain this error from the simulation. This can be done in the following way: we can use the simulation with artificially added offset. This way we can see to what power given suriviving fraction corresponds to at different quench offsets. In a sense, this method is better than what I am doing, because the systematic shift can be determined very quickly. However, with the first method above, I am calculating the calibration that I would actually use - it incorporates any imperfections in the analysis techniques, such as a variety of fits to the data and interpolations. This brings us to the question of what kind of uncertainties our RF power calibration has. These are the following:
1. Statistical uncertainty due to random errors in the DC ON/OFF ratios.
2. Systematic uncertainty due to the quench offset. I assume that the quench offset is independent of RF power, which is not true. This is the main reason for adding a relatively large fractional error to the quench offset. Another complication, is that it seems that the quench offset goes down with the decreasing pressure in the Box. Thus, in principle, after taking the calibration data, the quench offset could be smaller the longer the time interval is from the time the calibration data was taken and the given FOSOF data set is acquired.
3. Uncertainty due to the analysis technique. This is a systematic uncertainty.
4. Systematic uncertainty due to simulation itself. We do not know the profile of the fields exactly, and we are not sure if the simulation takes all of the possible effects into account + we are not sure about any numerical inaccuracies of the simulations.

For #1: Since I do not know the exact functional form of the dependence of various RF power-related parameters on the DC ON/OFF ratios and the extracted E fields, I can use the fit parameters, but I cannot use their uncertainties. Thus, in a sense, I do not have the statistical error for the power calibration.

For #3: This uncertainty is estimated by performing various methods of analysis and determining the averaged value = what we use for the power calibration + its STD (not STDOM). This serves as the analysis uncertainty. We have this uncertainty for each RF frequency and RF power value. And I am not sure how to incorporate this uncertainty into the one overall uncertainty value for each RF power. I can simply find the RMS uncertainty, assuming that the uncertainty is the same for all frequencies, but it is not the case: the data did not have the same noise level for all of the frequencies.

Another possibility of dealing with this issue is to assume that each type of analysis is the correct one. Thus we will have several possible calibration curves. In a sense, this is still exactly what I do: I average them and find the standard deviation. This gives me the averaged calibration curve, which is the best estimate in a sense.

I actually think that I am being somewhat incorrect here about the statistical error. I think I have to separate the issues of the analysis uncertainty and the statistical uncertainty. If I assume that the analysis is perfect, then I can use the uncertainty in the fit parameters to give me the uncertainty in the determined calibration. Notice that the fitting with the smoothing spline does not give any uncertainty associated with its parameters. I guess since the spline and the polynomial look very similar, I can assume that they have similar uncertainty. At first I thought that by assuming so I overestimate the uncertainty in the smoothing spline, because it is meant to follow the data points closely, and thus is, necessary, a better approximation of the data. But then I realized that by the same token I can use non-smoothing spline and get a perfect fit through all of the data points and then state that this is the perfect fit function for the data. But this is incorrect, because between the data points the spline will not, generally, behave that well. Therefore, I would say that indeed the spline and the polynomial fit both should have similar uncertainty in the interpolated values. In addition, we are, in a sense, forced to have the same statistical uncertainty in the fit parameters, because we are stating that each type of analysis is equally good. I.e., we are decoupling the systematic uncertainty due to the type of analysis and the statistical uncertainty due to data quality.

For #4. This is an important one. We have also seen that sometimes the resulting calibration curve (RF power setting vs RF frequency) has visible discontinuities in it, corresponding to various frequency ranges acquired at different times. It is possible that the lab temperature, beam parameters, line voltages are not necessary the same for all of these data sets, which results in the additional systematic uncertainty. It is also possible that the RF generator does not output the same power for its given power output setting. Because of that I rely on the RF power detector that are in the temperature-stabilized enclosure to provide me with the reliable metric for the RF power in each of the waveguides. Note that while acquiring data we actually used the RF power generator setting as the calibration parameter. Therefore it is quite possible that the data we acquired does not have the perfect calibration. This is the main reason for employing correction of the FOSOF phases for imperfect RF power calibration for the waveguides. These effects, by the way, are quite different from the simulation uncertainty, as for the simulation not being calculated correctly, but it is more about not using the correct simulation for the given calibration data. Thus, in principle, this is the #5 systematic uncertainty. One can, in principle, correct for this by having many-many simulations done for the beam having slightly different parameters (shape, speed, angle at which the beam is travelling, distribution of velocities), and then using these simulations for each calibration data set to perform the individual analysis on each of these. This results in a very large number of possible calibrations. One can average all of these together and obtain the best estimate calibration curve + the uncertainty in it.

Thus the steps are the following:

1. We need to run Monte Carlo simulation for various possible beam parameters (including the beam speed + spread in the speed) to obtain the beam profile inside the waveguides (at different locations inside the waveguides).
2. These beam profiles are used to obtain corresponding simulated quench curves.
3. For each simulated quench curve and the respective calibration data we perform several types of analysis for several different quench offset values.
4. For each quench offset value we take the respective calibration curves average them together to give us the averaged calibration curve + the statistical and the systematic uncertainty in the calibration. Here, the statistical uncertainty is the uncertainty that comes from the analysis that has the polynomial fit parameters. The systematic uncertainty is due to not knowing exactly the simulation to use is the spread in the power calibration curves obtained from different simulation curves, but using the same analysis method.
5. Now we have a set of different quench curves, each having 2 systematic uncertainties and 1 statistical uncertainty. The average of the average calibration curves for each quench offset gives the final average calibration curve, and the RMS uncertainty of the respective types of uncertainties gives us 2 types of the systematic uncertainties, and 1 type of the statistical uncertainty.
6. Each simulation has some systematic uncertainty due to the accuracy of the model used for the simulation. We need to agree on some number here. Something like +-2% of the quenched fraction = (1-surviving fraction), for instance. Notice, that this means that the larger field power corresponds to larger uncertainty in the quenched fraction. This is another systematic uncertainty that gets added (in quadrature) to other systematic uncertainties for the final averaged calibration curve. Notice that this uncertainty is not that simple: there is some uncertainty in the knowledge of the exact E and B field profiles inside the waveguides + there is the uncertainty in the theoretical model + uncertainty in the numerical accuracy. I assume that all of these uncertainties essentially result in the 2% error (for instance) in the determination of the quenched fraction. Now, to incorporate this uncertainty, for every simulation curve we need to construct 2 more curves having 2% more or less of quenching at every frequency + field power. This is, in a sense, a worst case scenario of the estimate of the effect of the imperfect simulation on the calibration. Obviously, all of the other steps have to be repeated for each of these simulation curves again.
7. At the end, what we get is the averaged calibration curve that has 3 types of systematic uncertainties + 1 statistical uncertainty. The uncertainty is for the RF-power-related parameter vs RF frequency for given RF power. Notice that the total systematic uncertainties for different frequencies and RF powers are not independent from each other, but are very correlated.

8. Now, the reason for knowing the RF power in the waveguides very well for each RF frequency is to be able to properly correct the FOSOF zero-crossing frequency for the AC shift. Thus, in a sense, we do not care about the uncertainty in the power at each RF frequency, but we care about the resulting uncertainty in the AC shift correction. Let us see now how the uncertainty in the calibration curve can be used to obtain the uncertainty in the AC shift correction.

9. Even after performing this complicated analysis, we still, unfortunately, cannot use this averaged calibration to correct the FOSOF phases, because the FOSOF lineshape depends on the beam parameters. Thus for each simulated quench curve we need to have the corresponding simulated FOSOF lineshape. Therefore for each simulated quench curve we use the respective calibration curves obtained with different types of analysis. This way the calibration curve has only 2 types of uncertainties: the systematic uncertainty due to the uncertainty in the accuracy of the model + the statistical uncertainty.

10. The simulated FOSOF lineshape has the uncertainty in it as well that is correlated with the systematic uncertainty in the corresponding simulated quench curve. But it is not just, for instance, 2%, because one expects the FOSOF lineshape to be much more sensitive to any imperfections in the model. Thus, one would use a larger uncertainty. This uncertainty is in the phases, and, certaintly depends on the RF power, and the RF frequency. Now, simularly to the quecnh curves as the RF power is decreased, the zero-crossing frequency of the FOSOF lineshape becomes less and less dependent on the RF power, exact field profile, and the beam profile. In other words, at zero power one expects to obtain the zero-crossing frequency that is equal to from the resonant frequency. As the RF power increases, the uncertainty in the difference between the zero-crossing frequency and the resonant frequency increases. We can assign something like 5% uncertainty for this difference that is proportional to the RF power. It is not easy to come up with the way to assign the systematic uncertainties to each of the FOSOF phase at every RF frequency. However, we can simply assume that all of the phases get shifted by the same amount for given RF power to result in the requested 5% systematic uncertainty in the AC shift correction. Notice, however, that I do not assume that this 5% uncertainty can be explained as the same 5% effect the AC shift for all of the powers. I can assume that the uncertainty can be treated independently at different RF powers. The reason for this is that I am claiming that the possible effect on the zero-crossing is complicated, and that it is hard to know what happens from one RF power to another. Notice that this is different from the systematic uncertainty of 2% in the simulated quench curve. I can also assume that the systematic uncertainties at different powers are totally correlated, as for the simulated quench curves. Thus there are 2 different ways of looking at this uncertainty. I will assume that the 1st method is the valid one. This method essentially allows me to think of this uncertainty as the statistical one. The second method, on the other hand, requires me to perform different analysis depending on what AC curve I pick - 0%, -5% or +5%.

11. Now, for each element of the set of simulated quench curves obtained for various beam parameters (uncertainty #5) we can construct the respective FOSOF lineshape. We first pick a calibration curve that was obtained from one of the analyses methods. This way the calibration curve has only the statistical uncertainty. We want now to use the calibration curve to correct the FOSOF lineshape for imperfect power, given the measured RF power detector voltages. This can be easily done by first assuming that the each calibration value of the RF power detector voltage for each RF frequency and the requested RF power is normally distributed. We can run Monte-Carlo simulation and obtain many-many corrected FOSOF lineshapes by using normally sampled values from the calibration curve. For each corrected FOSOF lineshape we determine the zero-crossing + its uncertainty, which is related to the uncertainty in the FOSOF phases data = #6 uncertainty. These zero-crossing frequencies are averaged together and we obtain the mean zero-crossing frequency + its statistical uncertainty. Thus we get a resonant frequency with 2 types of statistical uncertainties. One is the RMS uncertainty of all of the statistical uncertainties due to the noise in the FOSOF phases data, and another is due to the statistical uncertainty in the calibration data - uncertainty #1.

12. For each simulation curve obtained for various beam parameters we perform step 11. Then all of zero-crossing frequencies from the step 11 are averaged together. The standard deviation is the systematic uncertainty in the zero-crossing due to uncertainty #5 = related to not knowing the beam parameters exactly. The average zero-crossing frequency also has the 2 statistical uncertainties: the RMS uncertainties of the statistical uncertainties of the zero-crossing frequencies in the set in the step 11 (they must be almost identical).

13. We perform the step 11 and 12 for all other 3 types of calibration data analysis. All 4 obtained zero-crossing frequencies are averaged together. The standard deviation in this case is the systematic uncertainty due to different types of analysis (uncertainty #3). The obtained number now has 4 types of systematic uncertainties (#3, #5), and two statistical uncertainties (#1, #6).

14. We perform the step 13 for each of the quench offset. Average of all of the resulting zero-crossing frequencies (3 of them) + the uncertainty is related to the uncertainty #2

15. Now we have the last uncertainty to take care of: #4. Assuming that the AC shift uncertainties are independent from each other at different powers we can do the following. We use the +-2% simulated quench curves and the respective +-5% FOSOF lineshapes, and repeat step 14. The resulting zero-crossing frequencies are averaged together, which is our best estimate of the zero-crossing frequency obtained from single FOSOF data set. The uncertainty of this average is related to the systematic uncertainty #2. Now we have 6 types of uncertainties: 4 systematic ones and 2 statistical. Notice that only one statistical uncertainty can be reduced by taking more data. Other uncertainties stay the same, because they are correlated. We can then correct the averaged zero-crossing frequency for the AC shift.

16. Now the problem with the analysis is that the uncertaintes related to #2, #3, #5 have large correlations between the data acquired for different waveguide separations, powers, and accelerating voltages. #1 is not a part of this (however it for given RF frequency it is actually correlated to the calibration values for different RF powers), because we use different calibration data for different separations. But we use the same analysis code + same code for obtaining simulation data for all of the powers, speeds, and the separations. Hence, the #2, #3, #5 are respectively correlated for different experiment parameters. This means that the steps 11-15 are not entirely correct.

17. We actually need to do the following. For each simulation quench curve we select type of the calibration data analysis, we also select the set of RF powers and set of waveguide separations, with the respective simulated FOSOF lineshapes, and pick the respective calibration curves. For each waveguide separation and beam speed, we pick one set of normally chosen values from the respective calibration curve and then correct the FOSOF data with the these values. Note that we use the same calibration curve values for all RF powers for the given waveguide separation and beam speed, because of the correlation of the uncertainty #1. For each corrected FOSOF lineshape we perform the step similar to step 15. As the result we add the systematic uncertainty to each zero-crossing frequency that is related to the uncertainty #4. Thus each zero-crossing frequency will have 2 types of uncertainties: 1 statistical, and 1 systematic. Each of these zero-crossing frequencies is corrected for its AC shift and we get the set of the resonant frequencies corrected for the AC shift.

We now have a set of the resonant frequencies corrected for their AC shift for given RF power, waveguide separation, accelerating voltage, quench offset, and the set of normally chosen values from the calibration curves. Just to be clear: after this step we have the following data:

Separation:

4 cm: {p_4_1: f_4_1, p_4_2: f_4_2, p_4_3: f_4_3, ...}
5 cm: {p_5_1: f_5_1, p_5_2: f_5_2, p_5_3: f_5_3, ...}
...

, where p_i_j = power for the separation i, power index j, and f_i_j = AC-shift-corrected resonant frequency for the separation i, power index j

For each beam speed and separation we calculate the weighted average of the AC-shift-corrected resonant frequencies.

We now perform these steps for many-many other sets of normally chosen values for the respective calibration curves and obtain a set of weighted averages of the AC-shift-corrected resonant frequencies for given separation and beam speed. These are, again, averaged together and we obtain the number that has two uncertainties in it. One is the uncertainty that is the combination of the statistical and systematic uncertainty due to noise in the FOSOF phase data and AC shift uncertainty, and another is the systematic uncertainty related to the #1 uncertainty.

Notice that this procedure is repeated for every waveguide separation + beam speed. After we take the weighted average of the data to obtain the final AC-shift-corrected resonant frequency obtained by using a set of simulation curves for the same beam parameters, and the same calibration data analysis type. Note that the weights for the averaging can be chosen by us.

19. Step 18 is performed for all of other simulated quench curves for different beam parameters. This is similar to step 12.
20. We do the step similar to the step 13.
21. We do the step similar to the step 14.
22. We now have the set of the AC-shift corrected frequencies with 5 types of uncertainties: 1 is the combination of statistical (#6) + systematic (#4), the second one is related to #1, the third is related to #5, the fourth on is #3, and the fifth one is #2.

2018-11-11

After some thinking I have realized that the step 18, where I assume that the set of simulations for the same beam parameters can be used for all of the data sets, is not entirely correct. The reason is that I was adjusting the beam from data set to data set (usually). Or, even if I did not adjust anything, I cannot assume that the beam stays the same from the data set to data set. Therefore I think I can treat the systematic uncertainty #5 as the independent uncertainty. I.e., each data set should be corrected by a set of simulation curves with different beam parameters.

Therefore, the step 18 should be modified to be the following. I choose a set of simulated quench curves for different beam parameters + their respective simulated FOSOF lineshapes, but the same uncertainty level of type #4 + I pick the calibration data analysis type. For each waveguide separation and beam speed I do the following. I randomly pick the beam parameters for each FOSOF data set for this particular waveguide separation and beam speed, and thus I get a set of respective simulation curves used for the set of the FOSOF data sets. Each of these simulated quench curves have their own calibration curves. I want now to pick the same normally chosen values from each calibration curve. But it does not sound right, because the calibration curves will not be the same, since different simulations are used to obtain them. It seems that for each calibration curve for the same waveguide separation and beam speed, I need to randomly pick the same fractional deviation in terms of the multiple of the standard deviation for each calibration curve's point. I believe this is identical to performing the following. The calibration data consists of the DC ON/OFF surviving ratios with the corresponding uncertainties. For the given set of simulated quench curves obtained for different beam parameters, we can normally pick the DC ON/OFF ratio values from the calibration data, and use these values to find the set of calibration curves using the chosen calibration data analysis type. We now use this set of the calibration quench curves and the respective simulated FOSOF lineshapes to correct the FOSOF data for imperfect RF power. This is done for the FOSOF data sets having the same waveguide separation, beam speed, and waveguide power calibration data. For each of these corrected data sets we determine the zero-crossing frequency, which contains only the FOSOF data-related statistical uncertainty. For the given set of randomly chosen values from the DC ON/OFF ratios, we perform the average of the zero-crossing frequencies obtained for the same waveguide separation, beam speed, RF power, and the waveguide power calibration data to obtain a single zero-crossing frequency that has 2 uncertainties: 1 is the statistical FOSOF uncertainty (type #6) and 1 is the systematic uncertainty due to not knowing the beam parameters exactly for each data set (type #5). Thus at this moment we obtain for the given waveguide separation and the beam speed the following:

{(p_1, f_1), (p_2, f_2), (p_3, f_3), ...} - set of ordered pairs of powers and zero-crossing frequencies. Note that we have not applied any correction for the AC shift, and there is no uncertainty added due to this shift yet.

Now, how do I apply the AC shift correction to this? Do I simply add the #4-type uncertainty to each AC-shift-corrected resonant frequency? The problem is that I do not know how to go from 2% uncertainty in the simulated quench curves to 5% uncertainty in the FOSOF lineshape, because in one case the uncertainty is correlated, and in another case I claim that it is independent. Also, I would have to do this for each type of the calibration data analysis, but then I need to make sure that I assume that this uncertainty is totally correlated for all of the types.

I think to deal with this problem I need to do the similar type of the analysis as for the type #1 uncertainty. For each zero-crossing frequency determined above, that has 2 uncertainties, I normally pick the values of the shift.

We then perform this procedure for many-many choices of the DC ON/OFF ratio values for the chosen waveguide separation, beam speed, and the calibration data, and obtain the set of the zero-crossing frequencies.





It is quite difficult to perform this type of analysis. Let's see if there is anything I can do to simplify the analysis.

I will first assume that the statistical uncertainty #1 is too small to be relevant.

For step 19: The simulated quench curves for different off-axis distance are almost identical. I can assume that the change in the simulated quench curve with different beam parameters is too small to be important. Here I am talking about possible small changes in the beam speed, beam profile, beam velocity profile. Thus I can use the same simulation curve for all FOSOF data sets that have the same accelerating voltage. Now, the FOSOF lineshape is sensitive to the beam diameter. Because of that, just to be on a safe side I am assigning large fractional uncertainty to be beam rms radius. This should take care of the small changes in beam speed, beam shape, beam velocity profile. This takes care of the uncertainty #5.

For step 20. I assume that the different analysis methods do not result in appreciably large uncertainty in the final resonant frequency. In a sense, I am saying that the contribution to the total uncertainty in the resonant frequency is not appreciable, and thus can be ignored. This is the uncertainty #3.

For step 21. I do have calibration curves for different quench offsets. However, instead of correcting the data for each calibration curve with different offset, I can simply change the RF powers by the fractional amount calculated from comparing these calibration curves with different offsets. This is for the uncertainty #2.


'''

# Loading the calibration curves
av_rf_power_calib_error_set_df = pd.DataFrame()

for index, row in wvg_calib_info_df.iterrows():
    os.chdir(wvg_power_calib_data_folder_path)

    wvg_calib_analysis = wvg_power_calib_analysis.WaveguideCalibrationAnalysis(load_Q=True, calib_folder_name=row['Calibration Folder Name'], calib_file_name=row['Calibration File Name'])

    av_rf_power_calib_error_df = wvg_calib_analysis.get_av_rf_power_calib_error().copy()

    av_rf_power_calib_error_df['Calibration Folder Name'] = row['Calibration Folder Name']

    av_rf_power_calib_error_df.set_index('Calibration Folder Name', append=True, inplace=True)

    av_rf_power_calib_error_set_df = av_rf_power_calib_error_set_df.append(av_rf_power_calib_error_df)

mean_frac_calib_error_df = av_rf_power_calib_error_set_df[['Mean Fractional Error [%]']].sort_index().groupby('Proportional To RF Power [V^2/cm^2]').aggregate(lambda x: np.mean(x)).rename(columns={'Mean Fractional Error [%]': 'RF Power Calibration Fractional Error [%]'})

mean_frac_calib_error_df = mean_frac_calib_error_df.reset_index()

mean_frac_calib_error_df['Proportional To RF Power [V^2/cm^2]'] = np.sqrt(mean_frac_calib_error_df['Proportional To RF Power [V^2/cm^2]'])

mean_frac_calib_error_df = mean_frac_calib_error_df.rename(columns={'Proportional To RF Power [V^2/cm^2]': 'Waveguide Electric Field [V/cm]'}).set_index('Waveguide Electric Field [V/cm]')
#%%
mean_frac_calib_error_df
#%%
#

wvg_calib_analysis.get_av_calib_data()
#%%
