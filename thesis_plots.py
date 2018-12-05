from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
#%%
# Proton radius plot
p_rad_data_arr = np.array([0.8751, 0.8764, 0.879, 0.84087, 0.899, 0.877, 0.8335])
p_rad_data_std_arr = np.array([0.0061, 0.0089, 0.011, 0.00039, 0.059, 0.013, 0.0095])
y_data_arr = np.array([1, 3, 2, 2.5, 4, 1.5, 3.5])

color_list = ['red', 'blue', 'green', 'brown', 'purple', 'black', 'magenta']
y_ticks_arr = ['CODATA 2014', 'H spectroscopy', 'e-p scattering', '$\mu \mathrm{H}$ spectroscopy', 'Lundeen & Pipkin', '1S-3S (2018)', '2S-4P (2017)']

p_size_df = pd.DataFrame({'Proton Radius [fm]': p_rad_data_arr, 'Proton Radius STD [fm]': p_rad_data_std_arr, 'Y-axis Position': y_data_arr, 'Color': color_list, 'Label Name': y_ticks_arr}).set_index('Label Name')
#%%
arrow_length = p_size_df.loc['H spectroscopy','Proton Radius [fm]'] - p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]']
std_dev = np.round(arrow_length / p_size_df.loc['H spectroscopy','Proton Radius STD [fm]'], 1)
#%%
fig, ax = plt.subplots()

fig.set_size_inches(8/1.5,5/1.5)

for i in range(p_rad_data_arr.shape[0]):
    ax.errorbar(p_rad_data_arr[i], y_data_arr[i], xerr=p_rad_data_std_arr[i], linestyle='', color=color_list[i], elinewidth=2, capsize=5, capthick=1.25, marker='.', markersize='10')

index = 1
rect = Rectangle((p_rad_data_arr[index] - p_rad_data_std_arr[index], -10), 2*p_rad_data_std_arr[index], 20, color=color_list[index], fill=True, alpha=0.5)
ax.add_patch(rect)

index = 3
rect = Rectangle((p_rad_data_arr[index] - p_rad_data_std_arr[index], -10), 2*p_rad_data_std_arr[index], 20, color=color_list[index], fill=True, alpha=0.5)
ax.add_patch(rect)

plt.yticks(y_data_arr, ['CODATA 2014', 'H spectroscopy', 'e-p scattering', '$\mu \mathrm{H}$ spectroscopy', 'Lundeen & Pipkin', '1S-3S (2018)', '2S-4P (2017)'])

for ytick, color in zip(ax.get_yticklabels(), color_list):
    ytick.set_color(color)

arr_width = 0.01
head_width = 10 * arr_width
head_length = 0.15 * arrow_length

#plt.arrow(x=p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]'], y=2.25, dx=arrow_length-head_length, dy=0, width=arr_width, head_width=head_width, head_length=head_length, shape='full')

#plt.annotate(s=str(std_dev) + '$\sigma$', xy=(1,1), xytext=(0,0), arrowprops=dict(arrowstyle='<->'))

ax.annotate(xy=(p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]'], 2.25), xytext=(p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]']+arrow_length, 2.25), text='', arrowprops=dict(arrowstyle='<|-|>', connectionstyle='arc3', facecolor='black', edgecolor='black', shrinkA=0, shrinkB=0, mutation_scale=20))

ax.text(p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]']+arrow_length/2, 2.45, str(std_dev) + '$\sigma_{\mathrm{H}}$', fontsize=13, horizontalalignment='center')
ax.set_xlabel('Proton RMS charge radius, $r_\mathrm{p}$ (fm)')

os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis')

ax.set_xlim(right=0.92)


#plt.savefig('proton_rad_data.pdf', format='pdf',  bbox_inches='tight')
plt.show()
#%%
p_size_df.loc['H spectroscopy','Proton Radius [fm]'] - p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]']
#%%

#%%
x_arr = [1, 1.01]
y_arr = [1, 1.01]

fig, ax = plt.subplots()
ax.scatter(x_arr, y_arr, s=500, marker='v')
ax.errorbar(x_arr, y_arr, yerr=[0.1, 0.1], linestyle='', elinewidth=5, capsize=50, capthick=10, marker='.', markersize='10')
plt.show()
#%%
p_size_df.loc['H spectroscopy','Proton Radius [fm]']
#%%
arrow_length-head_length
#%%
arrow_length2 = p_size_df.loc['CODATA 2014','Proton Radius [fm]'] - p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]']
std_dev2 = np.round(arrow_length2 / p_size_df.loc['CODATA 2014','Proton Radius STD [fm]'], 1)
std_dev2
#%%
34*0.7
#%
34*0.6
#%%
(46-10)*0.7
#%%
46*0.6
