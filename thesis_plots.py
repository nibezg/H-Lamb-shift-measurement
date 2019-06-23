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
import scipy.optimize

import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

path_data_df = pd.read_csv(filepath_or_buffer='path_data.csv', delimiter=',', comment='#', header=[0], skip_blank_lines=True, index_col=[0])

code_folder_path = path_data_df.loc['Code Folder'].values[0].replace('\\', '/')
fosof_analyzed_data_folder_path = path_data_df.loc['FOSOF Analyzed Data Folder'].values[0].replace('\\', '/')
wvg_calib_data_folder_path = path_data_df.loc['Waveguide Calibration Folder'].values[0].replace('\\', '/')
krytar109B_pwr_det_calib_folder_path = path_data_df.loc['KRYTAR 109 B Power Detector Calibration Data Folder'].values[0].replace('\\', '/')
travis_data_folder_path = path_data_df.loc['Travis Data Folder'].values[0].replace('\\', '/')
these_high_n_folder_path = path_data_df.loc['Thesis High-n Shift Folder'].values[0].replace('\\', '/')
these_ac_folder_path = path_data_df.loc['Thesis AC Shift Folder'].values[0].replace('\\', '/')
these_beam_speed_folder_path = path_data_df.loc['Thesis Speed Measurement Folder'].values[0].replace('\\', '/')
these_phase_control_folder_path = path_data_df.loc['Thesis Phase Control Folder'].values[0].replace('\\', '/')

sys.path.insert(0, code_folder_path)

from exp_data_analysis import *
from fosof_data_set_analysis import *
from quenching_curve_analysis import *

# os.chdir(code_folder_path)
# from fosof_data_zero_cross_analysis import *
# os.chdir(code_folder_path)

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

#%%
'''
Proton radius data plot
'''
p_rad_data_arr = np.array([0.8751, 0.8764, 0.879, 0.84087, 0.899, 0.877, 0.8335])
p_rad_data_std_arr = np.array([0.0061, 0.0089, 0.011, 0.00039, 0.059, 0.013, 0.0095])
y_data_arr = np.array([0, 6, 1, 3, 5, 2, 4])

color_list = ['red', 'blue', 'green', 'brown', 'purple', 'black', 'magenta']
y_ticks_arr = ['CODATA 2014', 'H spectroscopy', 'e-p scattering', '$\mu \mathrm{H}$ spectroscopy', 'Lundeen & Pipkin', '1S-3S (2018)', '2S-4P (2017)']

p_size_df = pd.DataFrame({'Proton Radius [fm]': p_rad_data_arr, 'Proton Radius STD [fm]': p_rad_data_std_arr, 'Y-axis Position': y_data_arr, 'Color': color_list, 'Label Name': y_ticks_arr}).set_index('Label Name')

arrow_length = p_size_df.loc['H spectroscopy','Proton Radius [fm]'] - p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]']
std_dev = np.round(arrow_length / p_size_df.loc['H spectroscopy','Proton Radius STD [fm]'], 1)

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

plt.yticks(y_data_arr, ['CODATA 2014', 'H spectroscopy',  'e-p scattering', '$\mu \mathrm{H}$ spectroscopy', 'Lundeen & Pipkin', '1S-3S (2018)', '2S-4P (2017)'])

for ytick, color in zip(ax.get_yticklabels(), color_list):
    ytick.set_color(color)

arr_width = 0.01
head_width = 10 * arr_width
head_length = 0.15 * arrow_length

#plt.arrow(x=p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]'], y=2.25, dx=arrow_length-head_length, dy=0, width=arr_width, head_width=head_width, head_length=head_length, shape='full')

#plt.annotate(s=str(std_dev) + '$\sigma$', xy=(1,1), xytext=(0,0), arrowprops=dict(arrowstyle='<->'))

ax.annotate(xy=(p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]'], 2.25+0.4), xytext=(p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]']+arrow_length, 2.25+0.4), text='', arrowprops=dict(arrowstyle='<|-|>', connectionstyle='arc3', facecolor='black', edgecolor='black', shrinkA=0, shrinkB=0, mutation_scale=20))

ax.text(p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]']+arrow_length/2, 2.45+0.4, str(std_dev) + '$\sigma_{\mathrm{H}}$', fontsize=13, horizontalalignment='center')
ax.set_xlabel('Proton RMS charge radius, $r_\mathrm{p}$ (fm)')

ax.set_xlim(right=0.92)

# os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Thesis\p_rad')
# plt.savefig('proton_rad_data.pdf', format='pdf',  bbox_inches='tight')
plt.show()
#%%
''' Plots for the qualitative explanation of the SOF technique.
'''

os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis/Spin precession')


#%%
# Data for the plot of the surface of the Bloch sphere = possible orientations of the angular momentum vector.
r_sphere = 1
num_points = 100

phi_arr = np.linspace(0, 2*np.pi, num_points)
theta_arr = np.linspace(0, np.pi, num_points)

phi_s, theta_s = np.meshgrid(phi_arr, theta_arr)

x_s = r_sphere * np.sin(theta_s) * np.cos(phi_s)
y_s = r_sphere * np.sin(theta_s) * np.sin(phi_s)
z_s = r_sphere * np.cos(theta_s)

x_c = r_sphere * np.cos(phi_arr)
y_c = r_sphere * np.sin(phi_arr)

x_c_perp = r_sphere * np.cos(phi_arr)
z_c_perp = r_sphere * np.sin(phi_arr)
#%%
''' First pulse rotating the angular momentum by pi/2 radians.
'''
# Numerical solution for the magnetic dipole under influence of the constant magnetic field and perpendicular rotating magnetic field.

# Magnitude of the constant B field [T]
B0_mag = 1
# Amplitude of the rotating B field [T]
B_mag = 1
# Gyromagnetic ratio [C/kg]
gamma = -1
# Magnitude of the initial angular momentum vector [m^2*kg/s]
L_mag = 1

# Frequency of rotation of the rotating B field [rad/s] in multiples of the larmor frequency for the static field

# Notice the negative sign. It is needed here, so that when the Larmor frequency is equal to the rotation frequency, in the rotating reference frame in the direction of precession of the dipole due to the static field, the rotating field looks stationary.
omega_fact = -1

# Larmor angular frequency for the precession about the static field [rad/s]
omega_0 = gamma * B0_mag

# Frequency of rotation of the rotating field [rad/s]
omega = omega_fact * omega_0

# Period of the one precession about the static field [s]
T_per = 2 * np.pi / (np.abs(omega_0))

# Time [s] it takes for the rotating field, at the same rotating frequency as the Larmor frequency of the static field, to tilt the angular momentum vector by 90 degrees w.r.t. the static field axis, assuming that initialy the angular momentum is colinear with the static field direction.
Thalfpi = np.pi/2 / np.abs(gamma * B_mag)

# Final time for the simulation [s]
t_f = Thalfpi

# Numerical solution

# Time step for the numerical solver [s]
dt = 10**-3

n_steps = int(t_f/(dt))-1
L_arr = np.zeros((n_steps, 3))
L_arr[0] = np.array([0, 0, L_mag])

t_arr = np.linspace(0, t_f, n_steps)

# static field along z-axis
B0_arr = np.zeros((n_steps, 3))
B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

# Rotating field in the xy-plane
B_arr = np.zeros((n_steps, 3))

# Initial phase of the rotating field [rad]
phi_0 = 0

B_arr[:, 0] = B_mag * np.cos(omega * t_arr + phi_0)
B_arr[:, 1] = B_mag * np.sin(omega * t_arr + phi_0)

# Numerical solution
for i in range(1, n_steps):
    L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]+B_arr[i-1]) * dt

#%%
# Plotting the figure
fig = plt.figure()
fig.set_size_inches(10,10)

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

# Plot horizonal dashed circle
ax.plot(x_c, y_c, 0, color='black', linestyle='dashed', alpha=0.25)
#ax.scatter(x_c_perp, y_c_perp, z_c_perp, color='gray')
#ax.plot(x_c_perp, [0]*num_points, z_c_perp, color='black', alpha=0.5)
# Plot the transparent spherical surface
ax.plot_surface(x_s, y_s, z_s, color='gray', alpha=0.02)

# Plot the axes with arrowheads
x_axis_length = 2 * r_sphere
y_axis_length = 1.5 * r_sphere
z_axis_length = 1.4 * r_sphere

ax.scatter([0], [0], [0], color='black', s=10)

x_ax = Arrow3D([0, x_axis_length], [0, 0], [0, 0], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(x_ax)

y_ax = Arrow3D([0, 0], [0, y_axis_length], [0, 0], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(y_ax)

z_ax = Arrow3D([0, 0], [0, 0], [0, z_axis_length], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(z_ax)

ax.text(0, -0.15, z_axis_length-0.1, 'z', fontsize=16, horizontalalignment='center')

ax.text(x_axis_length+0.1, 0.2, -0.1, 'x', fontsize=16, horizontalalignment='center')

ax.text(0.3, y_axis_length+0.1, -0.1, 'y', fontsize=16,
horizontalalignment='center')

# # Initial angular momentum vector
# S_0_ax = Arrow3D([0, L_arr[0, 0]], [0, L_arr[0, 1]], [0, L_arr[0, 2]], mutation_scale=30,
#             lw=3, arrowstyle="-|>", color="blue")
# ax.add_artist(S_0_ax)
#
# # Static magnetic field vector
# B_0_ax = Arrow3D([0, 0.5*B0_arr[0, 0]], [0, 0.5*B0_arr[0, 1]], [0, 0.5*B0_arr[0, 2]], mutation_scale=30,
#             lw=3, arrowstyle="-|>", color="green")
# ax.add_artist(B_0_ax)

# Plotting of the evolution of the angular momentum vector
# Number of arrows to plot
n_arrows_plot = 9

dn = int(n_steps / n_arrows_plot)

x_a_arr = L_arr[0::dn, 0]
y_a_arr = L_arr[0::dn, 1]
z_a_arr = L_arr[0::dn, 2]

x_B_a_arr = B_arr[0::dn, 0]
y_B_a_arr = B_arr[0::dn, 1]
z_B_a_arr = B_arr[0::dn, 2]

# Plotting angular momentum vectors and the rotating magnetic field vectors. These vectors are semi-transparent

dalpha = (0.5-0.1)/(x_a_arr.shape[0]-1)
for arr_index in range(x_a_arr.shape[0]-1):
    alpha_dt = 0.1 + dalpha * arr_index
    S_dt_ax = Arrow3D([0, x_a_arr[arr_index]], [0, y_a_arr[arr_index]], [0, z_a_arr[arr_index]], mutation_scale=30,
                lw=3, arrowstyle="-|>", color="blue", alpha=alpha_dt)
    ax.add_artist(S_dt_ax)

    B_dt_ax = Arrow3D([0, x_B_a_arr[arr_index]], [0, y_B_a_arr[arr_index]], [0, z_B_a_arr[arr_index]], mutation_scale=30,
                lw=3, arrowstyle="-|>", color="red", alpha=alpha_dt)
    ax.add_artist(B_dt_ax)

# Final position of the angular momentum vector
S_1_ax = Arrow3D([0, x_a_arr[-1]], [0, y_a_arr[-1]], [0, z_a_arr[-1]], mutation_scale=30,
            lw=3, arrowstyle="-|>", color="blue", alpha=1)
ax.add_artist(S_1_ax)

# Final position of the rotating B field vector
B_1_ax = Arrow3D([0, x_B_a_arr[-1]], [0, y_B_a_arr[-1]], [0, z_B_a_arr[-1]], mutation_scale=30,
            lw=3, arrowstyle="-|>", color="red", alpha=1)
ax.add_artist(B_1_ax)

# Plot the 'trajectory' of the angular momentum and the rotating B field vector
ax.plot(L_arr[:, 0], L_arr[:, 1], L_arr[:, 2], linestyle='--', color='blue')
ax.plot(B_arr[:, 0], B_arr[:, 1], B_arr[:, 2], linestyle='--', color='red')

# Plot the arrow indicating the direction of the evolution of the angular momentum and the rotating B field vectors
arr_dir_index = int(0.4 * n_steps)
arr_dir_dur = int(0.1 * n_steps)

S_dir_ax = Arrow3D([L_arr[arr_dir_index-arr_dir_dur, 0], L_arr[arr_dir_index, 0]], [L_arr[arr_dir_index-arr_dir_dur, 1], L_arr[arr_dir_index, 1]], [L_arr[arr_dir_index-arr_dir_dur, 2], L_arr[arr_dir_index, 2]], mutation_scale=20,
            lw=2, arrowstyle="->", color="blue", alpha=1)
ax.add_artist(S_dir_ax)

B_dir_ax = Arrow3D([B_arr[arr_dir_index-arr_dir_dur, 0], B_arr[arr_dir_index, 0]], [B_arr[arr_dir_index-arr_dir_dur, 1], B_arr[arr_dir_index, 1]], [B_arr[arr_dir_index-arr_dir_dur, 2], B_arr[arr_dir_index, 2]], mutation_scale=20,
            lw=2, arrowstyle="->", color="red", alpha=1)
ax.add_artist(B_dir_ax)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_axis_off()
ax.view_init(elev = 10.0, azim = 30)

bbox = fig.bbox_inches.from_bounds(2, 2, 6.7, 6.7)

plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)

plt.savefig('bloch_1.pdf', format='pdf', bbox_inches=bbox)

plt.show()

# Final angular momentum and rotating B field values. These are needed as the input for the next stage of the simulation
L_final = L_arr[-1]
B_final = B_arr[-1]
t_final = t_arr[-1]
#%%
''' Precession under the action of the static field only.
'''
# Final time for the simulation [s]
t_f = 0.75 * T_per

# Numerical solution

n_steps = int(t_f/(dt))-1
L_arr = np.zeros((n_steps, 3))
L_arr[0] = L_final

t_arr = np.linspace(t_final, t_f+t_final, n_steps)

# Static field along z-axis
B0_arr = np.zeros((n_steps, 3))
B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

# Rotating field in the xy-plane
B_arr = np.zeros((n_steps, 3))

#B_arr[:, 0] = B_mag * np.cos(omega * t_arr + phi_0)
#B_arr[:, 1] = B_mag * np.sin(omega * t_arr + phi_0)

# Numerical solution
for i in range(1, n_steps):
    L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]) * dt

#%%
# Plotting the figure
fig = plt.figure()
fig.set_size_inches(10,10)

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

# Plot horizonal dashed circle
ax.plot(x_c, y_c, 0, color='black', linestyle='dashed', alpha=0.25)
#ax.scatter(x_c_perp, y_c_perp, z_c_perp, color='gray')
#ax.plot(x_c_perp, [0]*num_points, z_c_perp, color='black', alpha=0.5)
# Plot the transparent spherical surface
ax.plot_surface(x_s, y_s, z_s, color='gray', alpha=0.02)

# Plot the axes with arrowheads
x_axis_length = 2 * r_sphere
y_axis_length = 1.5 * r_sphere
z_axis_length = 1.4 * r_sphere

ax.scatter([0], [0], [0], color='black', s=10)

x_ax = Arrow3D([0, x_axis_length], [0, 0], [0, 0], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(x_ax)

y_ax = Arrow3D([0, 0], [0, y_axis_length], [0, 0], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(y_ax)

z_ax = Arrow3D([0, 0], [0, 0], [0, z_axis_length], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(z_ax)

ax.text(0, -0.15, z_axis_length-0.1, 'z', fontsize=16, horizontalalignment='center')

ax.text(x_axis_length+0.1, 0.2, -0.1, 'x', fontsize=16, horizontalalignment='center')

ax.text(0.3, y_axis_length+0.1, -0.1, 'y', fontsize=16,
horizontalalignment='center')

# # Static magnetic field vector
# B_0_ax = Arrow3D([0, 0.5*B0_arr[0, 0]], [0, 0.5*B0_arr[0, 1]], [0, 0.5*B0_arr[0, 2]], mutation_scale=30,
#             lw=3, arrowstyle="-|>", color="green")
# ax.add_artist(B_0_ax)

# Plotting of the evolution of the angular momentum vector
# Number of arrows to plot
n_arrows_plot = 12

dn = int(n_steps / n_arrows_plot)

x_a_arr = L_arr[0::dn, 0]
y_a_arr = L_arr[0::dn, 1]
z_a_arr = L_arr[0::dn, 2]

x_B_a_arr = B_arr[0::dn, 0]
y_B_a_arr = B_arr[0::dn, 1]
z_B_a_arr = B_arr[0::dn, 2]

# Plotting angular momentum vectors and the rotating magnetic field vectors. These vectors are semi-transparent

dalpha = (0.5-0.1)/(x_a_arr.shape[0]-1)
for arr_index in range(x_a_arr.shape[0]-1):
    alpha_dt = 0.1 + dalpha * arr_index
    #alpha_dt = 0.1
    S_dt_ax = Arrow3D([0, x_a_arr[arr_index]], [0, y_a_arr[arr_index]], [0, z_a_arr[arr_index]], mutation_scale=30, lw=3, arrowstyle="-|>", color="blue", alpha=alpha_dt)
    ax.add_artist(S_dt_ax)

    #B_dt_ax = Arrow3D([0, x_B_a_arr[arr_index]], [0, y_B_a_arr[arr_index]], [0, z_B_a_arr[arr_index]], mutation_scale=30, lw=3, arrowstyle="-|>", color="red", alpha=alpha_dt)
    #ax.add_artist(B_dt_ax)

# Final position of the angular momentum vector
S_1_ax = Arrow3D([0, x_a_arr[-1]], [0, y_a_arr[-1]], [0, z_a_arr[-1]], mutation_scale=30, lw=3, arrowstyle="-|>", color="blue", alpha=1)
ax.add_artist(S_1_ax)

# Final position of the rotating B field vector
#B_1_ax = Arrow3D([0, x_B_a_arr[-1]], [0, y_B_a_arr[-1]], [0, z_B_a_arr[-1]], mutation_scale=30, lw=3, arrowstyle="-|>", color="red", alpha=1)
#ax.add_artist(B_1_ax)

# Plot the 'trajectory' of the angular momentum and the rotating B field vector
ax.plot(L_arr[:, 0], L_arr[:, 1], L_arr[:, 2], linestyle='--', color='blue')
#ax.plot(B_arr[:, 0], B_arr[:, 1], B_arr[:, 2], linestyle='--', color='red')

# Plot the arrow indicating the direction of the evolution of the angular momentum and the rotating B field vectors
arr_dir_index = int(0.75 * n_steps)
arr_dir_dur = int(0.05 * n_steps)

S_dir_ax = Arrow3D([L_arr[arr_dir_index-arr_dir_dur, 0], L_arr[arr_dir_index, 0]], [L_arr[arr_dir_index-arr_dir_dur, 1], L_arr[arr_dir_index, 1]], [L_arr[arr_dir_index-arr_dir_dur, 2], L_arr[arr_dir_index, 2]], mutation_scale=20, lw=2, arrowstyle="->", color="blue", alpha=1)
ax.add_artist(S_dir_ax)

arr_dir_index = int(0.15 * n_steps)
arr_dir_dur = int(0.05 * n_steps)

S_dir_ax = Arrow3D([L_arr[arr_dir_index-arr_dir_dur, 0], L_arr[arr_dir_index, 0]], [L_arr[arr_dir_index-arr_dir_dur, 1], L_arr[arr_dir_index, 1]], [L_arr[arr_dir_index-arr_dir_dur, 2], L_arr[arr_dir_index, 2]], mutation_scale=20, lw=2, arrowstyle="->", color="blue", alpha=1)
ax.add_artist(S_dir_ax)

#B_dir_ax = Arrow3D([B_arr[arr_dir_index-arr_dir_dur, 0], B_arr[arr_dir_index, 0]], [B_arr[arr_dir_index-arr_dir_dur, 1], B_arr[arr_dir_index, 1]], [B_arr[arr_dir_index-arr_dir_dur, 2], B_arr[arr_dir_index, 2]], mutation_scale=20, lw=2, arrowstyle="-|>", color="red", alpha=1)
#ax.add_artist(B_dir_ax)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_axis_off()
ax.view_init(elev = 10.0, azim = 30)

bbox = fig.bbox_inches.from_bounds(2, 2, 6.7, 6.7)

plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)

plt.savefig('bloch_2.pdf', format='pdf', bbox_inches=bbox)

plt.show()

# Final angular momentum and rotating B field values. These are needed as the input for the next stage of the simulation
L_final = L_arr[-1]
#B_final = B_arr[-1]
t_final = t_arr[-1]
#%%
''' Precession of the angular momentum in the field of the second pulse in phase with the first pulse
'''
# Final time for the simulation [s]
t_f = Thalfpi

# Numerical solution

n_steps = int(t_f/(dt))-1
L_arr = np.zeros((n_steps, 3))
L_arr[0] = L_final

t_arr = np.linspace(t_final, t_f+t_final, n_steps)

# Static field along z-axis
B0_arr = np.zeros((n_steps, 3))
B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

# Rotating field in the xy-plane
B_arr = np.zeros((n_steps, 3))

# Initial phase of the rotating field [rad]
phi_0 = 0

B_arr[:, 0] = B_mag * np.cos(omega * t_arr + phi_0)
B_arr[:, 1] = B_mag * np.sin(omega * t_arr + phi_0)

# Numerical solution
for i in range(1, n_steps):
    L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]+B_arr[i-1]) * dt

#%%
# Plotting the figure
fig = plt.figure()
fig.set_size_inches(10,10)

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

# Plot horizonal dashed circle
ax.plot(x_c, y_c, 0, color='black', linestyle='dashed', alpha=0.25)
#ax.scatter(x_c_perp, y_c_perp, z_c_perp, color='gray')
#ax.plot(x_c_perp, [0]*num_points, z_c_perp, color='black', alpha=0.5)
# Plot the transparent spherical surface
ax.plot_surface(x_s, y_s, z_s, color='gray', alpha=0.02)

# Plot the axes with arrowheads
x_axis_length = 2 * r_sphere
y_axis_length = 1.5 * r_sphere
z_axis_length = 1.4 * r_sphere

ax.scatter([0], [0], [0], color='black', s=10)

x_ax = Arrow3D([0, x_axis_length], [0, 0], [0, 0], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(x_ax)

y_ax = Arrow3D([0, 0], [0, y_axis_length], [0, 0], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(y_ax)

z_ax = Arrow3D([0, 0], [0, 0], [0, z_axis_length], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(z_ax)

ax.text(0, -0.15, z_axis_length-0.1, 'z', fontsize=16, horizontalalignment='center')

ax.text(x_axis_length+0.1, 0.2, -0.1, 'x', fontsize=16, horizontalalignment='center')

ax.text(0.3, y_axis_length+0.1, -0.1, 'y', fontsize=16,
horizontalalignment='center')

# # Static magnetic field vector
# B_0_ax = Arrow3D([0, 0.5*B0_arr[0, 0]], [0, 0.5*B0_arr[0, 1]], [0, 0.5*B0_arr[0, 2]], mutation_scale=30,
#             lw=3, arrowstyle="-|>", color="green")
# ax.add_artist(B_0_ax)

# Plotting of the evolution of the angular momentum vector
# Number of arrows to plot
n_arrows_plot = 9

dn = int(n_steps / n_arrows_plot)

x_a_arr = L_arr[0::dn, 0]
y_a_arr = L_arr[0::dn, 1]
z_a_arr = L_arr[0::dn, 2]

x_B_a_arr = B_arr[0::dn, 0]
y_B_a_arr = B_arr[0::dn, 1]
z_B_a_arr = B_arr[0::dn, 2]

# Plotting angular momentum vectors and the rotating magnetic field vectors. These vectors are semi-transparent

dalpha = (0.5-0.1)/(x_a_arr.shape[0]-1)
for arr_index in range(x_a_arr.shape[0]-1):
    alpha_dt = 0.1 + dalpha * arr_index
    #alpha_dt = 0.1
    S_dt_ax = Arrow3D([0, x_a_arr[arr_index]], [0, y_a_arr[arr_index]], [0, z_a_arr[arr_index]], mutation_scale=30, lw=3, arrowstyle="-|>", color="blue", alpha=alpha_dt)
    ax.add_artist(S_dt_ax)

    B_dt_ax = Arrow3D([0, x_B_a_arr[arr_index]], [0, y_B_a_arr[arr_index]], [0, z_B_a_arr[arr_index]], mutation_scale=30, lw=3, arrowstyle="-|>", color="red", alpha=alpha_dt)
    ax.add_artist(B_dt_ax)

# Final position of the angular momentum vector
S_1_ax = Arrow3D([0, x_a_arr[-1]], [0, y_a_arr[-1]], [0, z_a_arr[-1]], mutation_scale=30, lw=3, arrowstyle="-|>", color="blue", alpha=1)
ax.add_artist(S_1_ax)

# Final position of the rotating B field vector
B_1_ax = Arrow3D([0, x_B_a_arr[-1]], [0, y_B_a_arr[-1]], [0, z_B_a_arr[-1]], mutation_scale=30, lw=3, arrowstyle="-|>", color="red", alpha=1)
ax.add_artist(B_1_ax)

# Plot the 'trajectory' of the angular momentum and the rotating B field vector
ax.plot(L_arr[:, 0], L_arr[:, 1], L_arr[:, 2], linestyle='--', color='blue')
ax.plot(B_arr[:, 0], B_arr[:, 1], B_arr[:, 2], linestyle='--', color='red')

# Plot the arrow indicating the direction of the evolution of the angular momentum and the rotating B field vectors
arr_dir_index = int(0.55 * n_steps)
arr_dir_dur = int(0.05 * n_steps)

S_dir_ax = Arrow3D([L_arr[arr_dir_index-arr_dir_dur, 0], L_arr[arr_dir_index, 0]], [L_arr[arr_dir_index-arr_dir_dur, 1], L_arr[arr_dir_index, 1]], [L_arr[arr_dir_index-arr_dir_dur, 2], L_arr[arr_dir_index, 2]], mutation_scale=20, lw=2, arrowstyle="->", color="blue", alpha=1)
ax.add_artist(S_dir_ax)

B_dir_ax = Arrow3D([B_arr[arr_dir_index-arr_dir_dur, 0], B_arr[arr_dir_index, 0]], [B_arr[arr_dir_index-arr_dir_dur, 1], B_arr[arr_dir_index, 1]], [B_arr[arr_dir_index-arr_dir_dur, 2], B_arr[arr_dir_index, 2]], mutation_scale=20, lw=2, arrowstyle="-|>", color="red", alpha=1)
ax.add_artist(B_dir_ax)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_axis_off()
ax.view_init(elev = 10.0, azim = 30)

bbox = fig.bbox_inches.from_bounds(2, 2, 6.7, 6.7)

plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)

plt.savefig('bloch_3.pdf', format='pdf', bbox_inches=bbox)

plt.show()
#%%
''' Precession of the angular momentum in the field of the second pulse out of phase with the first pulse
'''
# Final time for the simulation [s]
t_f = Thalfpi

# Numerical solution

n_steps = int(t_f/(dt))-1
L_arr = np.zeros((n_steps, 3))
L_arr[0] = L_final

t_arr = np.linspace(t_final, t_f+t_final, n_steps)

# Static field along z-axis
B0_arr = np.zeros((n_steps, 3))
B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

# Rotating field in the xy-plane
B_arr = np.zeros((n_steps, 3))

# Initial phase of the rotating field [rad]
phi_0 = np.pi

B_arr[:, 0] = B_mag * np.cos(omega * t_arr + phi_0)
B_arr[:, 1] = B_mag * np.sin(omega * t_arr + phi_0)

# Numerical solution
for i in range(1, n_steps):
    L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]+B_arr[i-1]) * dt

#%%
# Plotting the figure
fig = plt.figure()
fig.set_size_inches(10,10)

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

# Plot horizonal dashed circle
ax.plot(x_c, y_c, 0, color='black', linestyle='dashed', alpha=0.25)
#ax.scatter(x_c_perp, y_c_perp, z_c_perp, color='gray')
#ax.plot(x_c_perp, [0]*num_points, z_c_perp, color='black', alpha=0.5)
# Plot the transparent spherical surface
ax.plot_surface(x_s, y_s, z_s, color='gray', alpha=0.02)

# Plot the axes with arrowheads
x_axis_length = 2 * r_sphere
y_axis_length = 1.5 * r_sphere
z_axis_length = 1.4 * r_sphere

ax.scatter([0], [0], [0], color='black', s=10)

x_ax = Arrow3D([0, x_axis_length], [0, 0], [0, 0], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(x_ax)

y_ax = Arrow3D([0, 0], [0, y_axis_length], [0, 0], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(y_ax)

z_ax = Arrow3D([0, 0], [0, 0], [0, z_axis_length], mutation_scale=30,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(z_ax)

ax.text(0, -0.15, z_axis_length-0.1, 'z', fontsize=16, horizontalalignment='center')

ax.text(x_axis_length+0.1, 0.2, -0.1, 'x', fontsize=16, horizontalalignment='center')

ax.text(0.3, y_axis_length+0.1, -0.1, 'y', fontsize=16,
horizontalalignment='center')

# Plotting of the evolution of the angular momentum vector
# Number of arrows to plot
n_arrows_plot = 9

dn = int(n_steps / n_arrows_plot)

x_a_arr = L_arr[0::dn, 0]
y_a_arr = L_arr[0::dn, 1]
z_a_arr = L_arr[0::dn, 2]

x_B_a_arr = B_arr[0::dn, 0]
y_B_a_arr = B_arr[0::dn, 1]
z_B_a_arr = B_arr[0::dn, 2]

# Plotting angular momentum vectors and the rotating magnetic field vectors. These vectors are semi-transparent

dalpha = (0.5-0.1)/(x_a_arr.shape[0]-1)
for arr_index in range(x_a_arr.shape[0]-1):
    alpha_dt = 0.1 + dalpha * arr_index
    #alpha_dt = 0.1
    S_dt_ax = Arrow3D([0, x_a_arr[arr_index]], [0, y_a_arr[arr_index]], [0, z_a_arr[arr_index]], mutation_scale=30, lw=3, arrowstyle="-|>", color="blue", alpha=alpha_dt)
    ax.add_artist(S_dt_ax)

    B_dt_ax = Arrow3D([0, x_B_a_arr[arr_index]], [0, y_B_a_arr[arr_index]], [0, z_B_a_arr[arr_index]], mutation_scale=30, lw=3, arrowstyle="-|>", color="red", alpha=alpha_dt)
    ax.add_artist(B_dt_ax)

# Final position of the angular momentum vector
S_1_ax = Arrow3D([0, x_a_arr[-1]], [0, y_a_arr[-1]], [0, z_a_arr[-1]], mutation_scale=30, lw=3, arrowstyle="-|>", color="blue", alpha=1)
ax.add_artist(S_1_ax)

# Final position of the rotating B field vector
B_1_ax = Arrow3D([0, x_B_a_arr[-1]], [0, y_B_a_arr[-1]], [0, z_B_a_arr[-1]], mutation_scale=30, lw=3, arrowstyle="-|>", color="red", alpha=1)
ax.add_artist(B_1_ax)

# # Static magnetic field vector
# B_0_ax = Arrow3D([0, 0.5*B0_arr[0, 0]], [0, 0.5*B0_arr[0, 1]], [0, 0.5*B0_arr[0, 2]], mutation_scale=30,
#             lw=3, arrowstyle="-|>", color="green")
# ax.add_artist(B_0_ax)

# Plot the 'trajectory' of the angular momentum and the rotating B field vector
ax.plot(L_arr[:, 0], L_arr[:, 1], L_arr[:, 2], linestyle='--', color='blue')
ax.plot(B_arr[:, 0], B_arr[:, 1], B_arr[:, 2], linestyle='--', color='red')

# Plot the arrow indicating the direction of the evolution of the angular momentum and the rotating B field vectors
arr_dir_index = int(0.55 * n_steps)
arr_dir_dur = int(0.05 * n_steps)

S_dir_ax = Arrow3D([L_arr[arr_dir_index-arr_dir_dur, 0], L_arr[arr_dir_index, 0]], [L_arr[arr_dir_index-arr_dir_dur, 1], L_arr[arr_dir_index, 1]], [L_arr[arr_dir_index-arr_dir_dur, 2], L_arr[arr_dir_index, 2]], mutation_scale=20, lw=2, arrowstyle="->", color="blue", alpha=1)
ax.add_artist(S_dir_ax)

arr_dir_index = int(0.3 * n_steps)
arr_dir_dur = int(0.05 * n_steps)

B_dir_ax = Arrow3D([B_arr[arr_dir_index-arr_dir_dur, 0], B_arr[arr_dir_index, 0]], [B_arr[arr_dir_index-arr_dir_dur, 1], B_arr[arr_dir_index, 1]], [B_arr[arr_dir_index-arr_dir_dur, 2], B_arr[arr_dir_index, 2]], mutation_scale=20, lw=2, arrowstyle="-|>", color="red", alpha=1)
ax.add_artist(B_dir_ax)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_axis_off()
ax.view_init(elev = 10.0, azim = 30)

bbox = fig.bbox_inches.from_bounds(2, 2, 6.7, 6.7)

plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)

plt.savefig('bloch_4.pdf', format='pdf', bbox_inches=bbox)

plt.show()
#%%
''' Performing numerical simulation for a range of rotating angular frequencies.
'''

''' For the phase difference between the pulses of 0 rad.
'''
# Numerical solution for the magnetic dipole under influence of the constant magnetic field and perpendicular rotating magnetic field.

# Magnitude of the constant B field [T]
B0_mag = 1
# Amplitude of the rotating B field [T]
B_mag = 0.1
# Gyromagnetic ratio [C/kg]
gamma = -1
# Magnitude of the initial angular momentum vector [m^2*kg/s]
L_mag = 1

# Multiple of the 2 pi / T_sep factor
freq_mult_max = 2
freq_steps = 201
freq_mult_arr = np.linspace(-freq_mult_max, freq_mult_max, freq_steps)

# Phase difference between the pulses = phase of pulse 2 - phase of pulse 1 [rad]
delta_phi = 0

# Larmor angular frequency for the precession about the static field [rad/s]
omega_0 = gamma * B0_mag

# Period of the one precession about the static field [s]
T_per = 2 * np.pi / (np.abs(omega_0))

# Time [s] it takes for the rotating field, at the same rotating frequency as the Larmor frequency of the static field, to tilt the angular momentum vector by 90 degrees w.r.t. the static field axis, assuming that initialy the angular momentum is colinear with the static field direction.
Thalfpi = np.pi/2 / np.abs(gamma * B_mag)

# Duration of each of the pulses [s]
tau = Thalfpi

# Separation between the pulses [s]
T_sep = 0.75 * T_per

delta_omega_arr = 2 * np.pi / T_sep * freq_mult_arr

# Frequency of rotation of the rotating B field [rad/s] in multiples of the larmor frequency for the static field. Notice the negative sign. It is needed here, so that when the Larmor frequency is equal to the rotation frequency, in the rotating reference frame in the direction of precession of the dipole due to the static field, the rotating field looks stationary.
omega_fact = -1

# Frequency of rotation of the rotating field [rad/s]
omega_arr = omega_fact * (delta_omega_arr + omega_0)

L_final_arr = np.zeros(omega_arr.shape[0])

L_final_index = 0

for omega in omega_arr:

    # Numerical solution

    # Final time for the simulation [s]
    t_f = tau

    # Time step for the numerical solver [s]
    dt = 10**-4

    n_steps = int(t_f/(dt))-1
    L_arr = np.zeros((n_steps, 3))
    L_arr[0] = np.array([0, 0, L_mag])

    t_arr = np.linspace(0, t_f, n_steps)

    # static field along z-axis
    B0_arr = np.zeros((n_steps, 3))
    B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

    # Rotating field in the xy-plane
    B_arr = np.zeros((n_steps, 3))

    # Initial phase of the rotating field [rad]
    phi_0 = 0

    B_arr[:, 0] = B_mag * np.cos(omega * t_arr + phi_0)
    B_arr[:, 1] = B_mag * np.sin(omega * t_arr + phi_0)

    # Numerical solution
    for i in range(1, n_steps):
        L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]+B_arr[i-1]) * dt

    # Final angular momentum value. Needed as the input for the next stage of the simulation
    L_final = L_arr[-1]
    #B_final = B_arr[-1]
    t_final = t_arr[-1]

    # Field-free region

    t_f = T_sep

    n_steps = int(t_f/(dt))-1
    L_arr = np.zeros((n_steps, 3))
    L_arr[0] = L_final

    t_arr = np.linspace(t_final, t_f+t_final, n_steps)

    # Static field along z-axis
    B0_arr = np.zeros((n_steps, 3))
    B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

    # Rotating field in the xy-plane
    B_arr = np.zeros((n_steps, 3))

    # Numerical solution
    for i in range(1, n_steps):
        L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]) * dt

    # Final angular momentum and rotating B field values. These are needed as the input for the next stage of the simulation
    L_final = L_arr[-1]
    #B_final = B_arr[-1]
    t_final = t_arr[-1]

    # Second pulse

    # Final time for the simulation [s]
    t_f = tau

    # Numerical solution

    n_steps = int(t_f/(dt))-1
    L_arr = np.zeros((n_steps, 3))
    L_arr[0] = L_final

    t_arr = np.linspace(t_final, t_f+t_final, n_steps)

    # Static field along z-axis
    B0_arr = np.zeros((n_steps, 3))
    B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

    # Rotating field in the xy-plane
    B_arr = np.zeros((n_steps, 3))

    # Initial phase of the rotating field [rad]
    phi_0 = delta_phi

    B_arr[:, 0] = B_mag * np.cos(omega * t_arr + phi_0)
    B_arr[:, 1] = B_mag * np.sin(omega * t_arr + phi_0)

    # Numerical solution
    for i in range(1, n_steps):
        L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]+B_arr[i-1]) * dt

    L_final_arr[L_final_index] = L_arr[-1, 2]
    L_final_index = L_final_index + 1

L_final_0_arr = L_final_arr
#%%
figure, ax = plt.subplots()

ax.plot(omega_fact*omega_arr-omega_0, L_final_0_arr/L_mag)

plt.show()
#%%
''' For pi phase difference
'''
# Phase difference between the pulses = phase of pulse 2 - phase of pulse 1 [rad]
delta_phi = np.pi

L_final_arr = np.zeros(omega_arr.shape[0])

L_final_index = 0

for omega in omega_arr:

    # Numerical solution

    # Final time for the simulation [s]
    t_f = tau

    # Time step for the numerical solver [s]
    dt = 10**-3

    n_steps = int(t_f/(dt))-1
    L_arr = np.zeros((n_steps, 3))
    L_arr[0] = np.array([0, 0, L_mag])

    t_arr = np.linspace(0, t_f, n_steps)

    # static field along z-axis
    B0_arr = np.zeros((n_steps, 3))
    B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

    # Rotating field in the xy-plane
    B_arr = np.zeros((n_steps, 3))

    # Initial phase of the rotating field [rad]
    phi_0 = 0

    B_arr[:, 0] = B_mag * np.cos(omega * t_arr + phi_0)
    B_arr[:, 1] = B_mag * np.sin(omega * t_arr + phi_0)

    # Numerical solution
    for i in range(1, n_steps):
        L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]+B_arr[i-1]) * dt

    # Final angular momentum value. Needed as the input for the next stage of the simulation
    L_final = L_arr[-1]
    #B_final = B_arr[-1]
    t_final = t_arr[-1]

    # Field-free region

    t_f = T_sep

    n_steps = int(t_f/(dt))-1
    L_arr = np.zeros((n_steps, 3))
    L_arr[0] = L_final

    t_arr = np.linspace(t_final, t_f+t_final, n_steps)

    # Static field along z-axis
    B0_arr = np.zeros((n_steps, 3))
    B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

    # Rotating field in the xy-plane
    B_arr = np.zeros((n_steps, 3))

    # Numerical solution
    for i in range(1, n_steps):
        L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]) * dt

    # Final angular momentum and rotating B field values. These are needed as the input for the next stage of the simulation
    L_final = L_arr[-1]
    #B_final = B_arr[-1]
    t_final = t_arr[-1]

    # Second pulse

    # Final time for the simulation [s]
    t_f = tau

    # Numerical solution

    n_steps = int(t_f/(dt))-1
    L_arr = np.zeros((n_steps, 3))
    L_arr[0] = L_final

    t_arr = np.linspace(t_final, t_f+t_final, n_steps)

    # Static field along z-axis
    B0_arr = np.zeros((n_steps, 3))
    B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

    # Rotating field in the xy-plane
    B_arr = np.zeros((n_steps, 3))

    # Initial phase of the rotating field [rad]
    phi_0 = delta_phi

    B_arr[:, 0] = B_mag * np.cos(omega * t_arr + phi_0)
    B_arr[:, 1] = B_mag * np.sin(omega * t_arr + phi_0)

    # Numerical solution
    for i in range(1, n_steps):
        L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]+B_arr[i-1]) * dt

    L_final_arr[L_final_index] = L_arr[-1, 2]
    L_final_index = L_final_index + 1

L_final_pi_arr = L_final_arr
#%%
figure, ax = plt.subplots()

ax.plot(omega_fact*omega_arr-omega_0, L_final_pi_arr/L_mag)

plt.show()
#%%
''' For pi/2 phase difference
'''
# Phase difference between the pulses = phase of pulse 2 - phase of pulse 1 [rad]
delta_phi = np.pi / 2

L_final_arr = np.zeros(omega_arr.shape[0])

L_final_index = 0

for omega in omega_arr:

    # Numerical solution

    # Final time for the simulation [s]
    t_f = tau

    # Time step for the numerical solver [s]
    dt = 10**-3

    n_steps = int(t_f/(dt))-1
    L_arr = np.zeros((n_steps, 3))
    L_arr[0] = np.array([0, 0, L_mag])

    t_arr = np.linspace(0, t_f, n_steps)

    # static field along z-axis
    B0_arr = np.zeros((n_steps, 3))
    B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

    # Rotating field in the xy-plane
    B_arr = np.zeros((n_steps, 3))

    # Initial phase of the rotating field [rad]
    phi_0 = 0

    B_arr[:, 0] = B_mag * np.cos(omega * t_arr + phi_0)
    B_arr[:, 1] = B_mag * np.sin(omega * t_arr + phi_0)

    # Numerical solution
    for i in range(1, n_steps):
        L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]+B_arr[i-1]) * dt

    # Final angular momentum value. Needed as the input for the next stage of the simulation
    L_final = L_arr[-1]
    #B_final = B_arr[-1]
    t_final = t_arr[-1]

    # Field-free region

    t_f = T_sep

    n_steps = int(t_f/(dt))-1
    L_arr = np.zeros((n_steps, 3))
    L_arr[0] = L_final

    t_arr = np.linspace(t_final, t_f+t_final, n_steps)

    # Static field along z-axis
    B0_arr = np.zeros((n_steps, 3))
    B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

    # Rotating field in the xy-plane
    B_arr = np.zeros((n_steps, 3))

    # Numerical solution
    for i in range(1, n_steps):
        L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]) * dt

    # Final angular momentum and rotating B field values. These are needed as the input for the next stage of the simulation
    L_final = L_arr[-1]
    #B_final = B_arr[-1]
    t_final = t_arr[-1]

    # Second pulse

    # Final time for the simulation [s]
    t_f = tau

    # Numerical solution

    n_steps = int(t_f/(dt))-1
    L_arr = np.zeros((n_steps, 3))
    L_arr[0] = L_final

    t_arr = np.linspace(t_final, t_f+t_final, n_steps)

    # Static field along z-axis
    B0_arr = np.zeros((n_steps, 3))
    B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

    # Rotating field in the xy-plane
    B_arr = np.zeros((n_steps, 3))

    # Initial phase of the rotating field [rad]
    phi_0 = delta_phi

    B_arr[:, 0] = B_mag * np.cos(omega * t_arr + phi_0)
    B_arr[:, 1] = B_mag * np.sin(omega * t_arr + phi_0)

    # Numerical solution
    for i in range(1, n_steps):
        L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]+B_arr[i-1]) * dt

    L_final_arr[L_final_index] = L_arr[-1, 2]
    L_final_index = L_final_index + 1

L_final_pi_half_arr = L_final_arr
#%%
figure, ax = plt.subplots()

ax.plot(omega_fact*omega_arr-omega_0, L_final_pi_half_arr/L_mag)

plt.show()
#%%
figure, ax = plt.subplots()

ax.plot(omega_fact*omega_arr-omega_0, (L_final_pi_arr-L_final_0_arr)/L_mag)

plt.show()
#%%
figure, ax = plt.subplots()

ax.plot(omega_fact*omega_arr-omega_0, (L_final_pi_half_arr-L_final_0_arr)/L_mag)

plt.show()
#%%
figure, ax = plt.subplots()

ax.plot(omega_fact*omega_arr-omega_0, (L_final_pi_half_arr-L_final_0_arr + L_final_pi_arr-L_final_0_arr)/L_mag)

plt.show()
#%%
''' Numerical simulation for single pulse of twice the length
'''

L_final_arr = np.zeros(omega_arr.shape[0])

L_final_index = 0

for omega in omega_arr:

    # Numerical solution

    # Final time for the simulation [s]
    t_f = 2 * tau

    # Time step for the numerical solver [s]
    dt = 10**-3

    n_steps = int(t_f/(dt))-1
    L_arr = np.zeros((n_steps, 3))
    L_arr[0] = np.array([0, 0, L_mag])

    t_arr = np.linspace(0, t_f, n_steps)

    # static field along z-axis
    B0_arr = np.zeros((n_steps, 3))
    B0_arr[:, 2] = B0_arr[:, 2] + B0_mag

    # Rotating field in the xy-plane
    B_arr = np.zeros((n_steps, 3))

    # Initial phase of the rotating field [rad]
    phi_0 = 0

    B_arr[:, 0] = B_mag * np.cos(omega * t_arr + phi_0)
    B_arr[:, 1] = B_mag * np.sin(omega * t_arr + phi_0)

    # Numerical solution
    for i in range(1, n_steps):
        L_arr[i] = L_arr[i-1] + gamma * np.cross(L_arr[i-1], B0_arr[i-1]+B_arr[i-1]) * dt

    L_final_arr[L_final_index] = L_arr[-1, 2]
    L_final_index = L_final_index + 1
#%%
L_final_not_free_arr = L_final_arr
figure, ax = plt.subplots()

ax.plot(omega_fact*omega_arr-omega_0, L_final_not_free_arr/L_mag)

plt.show()
#%%
figure, ax = plt.subplots()

ax.plot(omega_fact*omega_arr-omega_0, L_final_not_free_arr/L_mag)
ax.plot(omega_fact*omega_arr-omega_0, L_final_0_arr/L_mag)

plt.show()
#%%
figure, ax = plt.subplots()

figure.set_size_inches(8,6)

ax.plot(freq_mult_arr, L_final_not_free_arr/L_mag, color='black', linestyle='dashed')
ax.plot(freq_mult_arr, L_final_0_arr/L_mag, color='blue')

ax.set_xlabel('$(\omega-\omega_0)T/2\pi$')
ax.set_ylabel('$\\vec{S_f} \cdot \\hat{z}\;/\; \\vert\\vec{S_i}\\vert$')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()
#%%
''' Extraction of the numerical solution performed in Mathematica.
Mathematica code is much faster, because it uses some built-in solvers.
'''

lineshape_0_data_arr = np.loadtxt('lineshape_0.CSV', delimiter=',', dtype=np.float64)

omega_fract_arr = lineshape_0_data_arr[:, 0]
fract_Lz_arr = lineshape_0_data_arr[:, 1]

fig = plt.figure()
fig.set_size_inches(10, 12)

ax = fig.add_subplot(211)

ax.plot(omega_fract_arr, fract_Lz_arr, color='blue')

lineshape_free_data_arr = np.loadtxt('lineshape_free.CSV', delimiter=',', dtype=np.float64)

omega_fract_arr = lineshape_free_data_arr[:, 0]
fract_Lz_free_arr = lineshape_free_data_arr[:, 1]

ax.plot(omega_fract_arr, fract_Lz_free_arr, color='black', linestyle='dashed')

ax.set_xticklabels([])
ax.set_ylabel('$\\vec{S_f} \\cdot \\hat{z}\;/\; \\vert\\vec{S_i}\\vert$')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

# FOSOF phase
fosof_ph_data_arr = np.loadtxt('FOSOF_phase.CSV', delimiter=',', dtype=np.float64)

omega_fract_arr = fosof_ph_data_arr[:, 0]
fract_Lz_arr = fosof_ph_data_arr[:, 1]

fosof_fit = np.poly1d(np.polyfit(omega_fract_arr, fract_Lz_arr, 1))

ax2 = fig.add_subplot(212)

ax2.plot(omega_fract_arr, fract_Lz_arr, color='blue')

ax2.plot([0, 0], [-35, 0], color='black', linestyle='dashed')
ax2.plot([-5,0], [0, 0], color='black', linestyle='dashed')

ax2.arrow(-1.8, 0, 0.1, 0, shape='full', width=0.6, head_length=0.3, color='black')

ax2.arrow(0, -10, 0, -1, shape='full', width=0.05, head_length=3.3, color='black')

ax2.set_xlabel('$(\omega-\omega_0)T \; / \; 2\pi$')
ax2.set_ylabel('$\\theta$ (rad)')

ax2.set_ylim(np.min(fract_Lz_arr), np.max(fract_Lz_arr))
ax2.set_xlim(np.min(omega_fract_arr), np.max(omega_fract_arr))

for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(15)

ax2_twin = ax2.twinx()

res_phase_arr = (fosof_fit(omega_fract_arr)-fract_Lz_arr)
ax2_twin.plot(omega_fract_arr, res_phase_arr, color='green', linestyle='dashed')
ax2_twin.set_ylim(np.min(res_phase_arr), np.max(res_phase_arr))
ax2_twin.set_ylabel('Residual from the linear trend (rad)')

for item in ([ax2_twin.title, ax2_twin.xaxis.label, ax2_twin.yaxis.label] +
             ax2_twin.get_xticklabels() + ax2_twin.get_yticklabels()):
    item.set_fontsize(15)

ax2.tick_params(axis='y', colors='blue')
ax2.yaxis.label.set_color('blue')
ax2_twin.tick_params(axis='y', colors='green')
ax2_twin.yaxis.label.set_color('green')
ax2_twin.yaxis.labelpad = 10
plt.savefig('u_lineshapes.pdf', format='pdf',  bbox_inches='tight')

plt.show()
#%%
''' FOSOF/SOF sequence diagram
'''

os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis/two-level atom')

E0 = 1
tau = 3
T = 6
dt = tau/400
omega_E = 2 * np.pi * 1.5

time_pulse_1_arr = np.linspace(0, tau, int(tau/dt))
phi_01 = np.pi / 2
E1_arr = E0 * np.cos(omega_E*time_pulse_1_arr+phi_01)

time_T_arr = np.linspace(tau, T+tau, int(T/dt))
ET_arr = 0 * time_T_arr

time_pulse_2_arr = np.linspace(tau+T, T+tau+tau, int(tau/dt))
phi_02 = -np.pi / 2
E2_arr = E0 * np.cos(omega_E*time_pulse_2_arr+phi_02)

fig = plt.figure()
fig.set_size_inches(12, 6)

ax = fig.add_subplot(111)

alpha_to_use = 1

ax.plot(time_pulse_1_arr, E1_arr, color='black')
ax.plot(time_T_arr, ET_arr, color='black', alpha=alpha_to_use)
ax.plot(time_pulse_2_arr, E2_arr, color='black')

ax.set_ylim(-2*E0, 2*E0)

x_min = -0.5
x_max = T + 2 * tau + 0.5

time_pulse_before_arr = np.linspace(-0.5, 0, int(0.5/dt))
E_before_arr = 0 * time_pulse_before_arr

time_pulse_after_arr = np.linspace(T+2*tau, T+2*tau+0.5, int((T+2*tau+0.5)/dt))
E_after_arr = 0 * time_pulse_after_arr

ax.plot(time_pulse_before_arr, E_before_arr, color='black', alpha=alpha_to_use)
ax.plot(time_pulse_after_arr, E_after_arr, color='black', alpha=alpha_to_use)


ax.set_xlim(x_min, x_max)

ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_axis_off()

# Drawing arrows

# Vertical lines
start_x = 0
start_y = 0
end_x = start_x
end_y = 1.3
ax.plot([start_x, end_x], [start_y, end_y], linestyle='dashed', color='black')

start_x = tau
start_y = 0
end_x = start_x
end_y = 1.3
ax.plot([start_x, end_x], [start_y, end_y], linestyle='dashed', color='black')

start_x = tau + T
start_y = 0
end_x = start_x
end_y = 1.3
ax.plot([start_x, end_x], [start_y, end_y], linestyle='dashed', color='black')

start_x = 2 * tau + T
start_y = 0
end_x = start_x
end_y = 1.3
ax.plot([start_x, end_x], [start_y, end_y], linestyle='dashed', color='black')

# Horizontal arrows
start_x = 0
start_y = 1.3
end_x = tau
end_y = 1.3
arrow_1 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='<|-|>', mutation_scale=20, color='black')
ax.add_patch(arrow_1)

start_x = tau
start_y = 1.3
end_x = T + tau
end_y = 1.3
arrow_2 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='<|-|>', mutation_scale=20, color='black')
ax.add_patch(arrow_2)

start_x = T + tau
start_y = 1.3
end_x = T + 2 * tau
end_y = 1.3
arrow_3 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='<|-|>', mutation_scale=20, color='black')
ax.add_patch(arrow_3)

# Text labels
text_x = tau / 2
text_y = 1.3 + 0.2
ax.text(x=text_x, y=text_y, s='$\\tau$', fontsize=15, horizontalalignment='center')

text_x = tau + T / 2
text_y = 1.3 + 0.2
ax.text(x=text_x, y=text_y, s='$T$', fontsize=15, horizontalalignment='center')

text_x = T + tau + tau / 2
text_y = 1.3 + 0.2
ax.text(x=text_x, y=text_y, s='$\\tau$', fontsize=15, horizontalalignment='center')

# Time axis
start_x = tau + 0.2 * tau
start_y = -1.3
end_x = start_x + 0.7 * T
end_y = -1.3
arrow_4 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='-|>', mutation_scale=30, color='black', linewidth=2)
ax.add_patch(arrow_4)

text_x = start_x + (end_x - start_x) / 2
text_y = -1.3 + 0.2
ax.text(x=text_x, y=text_y, s='Time', fontsize=15, horizontalalignment='center')

plt.savefig('pulse_sequence.pdf', format='pdf',  bbox_inches='tight')

plt.show()
#%%
''' Extraction of the FOSOF/SOF lineshapes obtained in Mathematica for the n=2 Lamb shift, since the Mathematica has already symbolic solution typed into it.
'''
os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Thesis\two-level atom')

lineshape_SOF_data_arr = np.loadtxt('lineshape_SOF_atom.CSV', delimiter=',', dtype=np.float64)

delta_freq_arr = lineshape_SOF_data_arr[:, 0]
fract_pop_arr = lineshape_SOF_data_arr[:, 1] * 1E3

fig = plt.figure()
fig.set_size_inches(10, 12)

ax = fig.add_subplot(211)

ax.plot(delta_freq_arr, fract_pop_arr, color='blue')

ax.set_xticklabels([])
ax.set_ylabel('$(\\rho_{11}^{\\pi}-\\rho_{11}^0) \\times 10^{-3}$')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax.tick_params(axis='y', colors='blue')
ax.yaxis.label.set_color('blue')

lineshape_free_data_arr = np.loadtxt('lineshape_free_atom.CSV', delimiter=',', dtype=np.float64)

delta_freq_arr = lineshape_free_data_arr[:, 0]
fract_pop_arr = lineshape_free_data_arr[:, 1]

ax_twin = ax.twinx()
ax_twin.plot(delta_freq_arr, fract_pop_arr, color='black', linestyle='dashed')

ax_twin.set_ylabel('$\\rho_{11}$')

for item in ([ax_twin.title, ax_twin.xaxis.label, ax_twin.yaxis.label] +
             ax_twin.get_xticklabels() + ax_twin.get_yticklabels()):
    item.set_fontsize(15)

# FOSOF phase
fosof_ph_data_arr = np.loadtxt('FOSOF_phase_atom.CSV', delimiter=',', dtype=np.float64)

delta_freq_arr = fosof_ph_data_arr[:, 0]
theta_arr = fosof_ph_data_arr[:, 1]

fosof_fit = np.poly1d(np.polyfit(delta_freq_arr, theta_arr, 1))

ax2 = fig.add_subplot(212)

ax2.plot(delta_freq_arr, theta_arr, color='blue')

ax2.plot([0, 0], [-35, 0], color='black', linestyle='dashed')
ax2.plot([-80,0], [0, 0], color='black', linestyle='dashed')

start_x = -40
start_y = 0
end_x = -35
end_y = 0
arrow_FOSOF_1 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='-|>', mutation_scale=30, color='black', linewidth=2)

ax2.add_patch(arrow_FOSOF_1)

start_x = 0
start_y = -5.5
end_x = 0
end_y = -4
arrow_FOSOF_2 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='-|>', mutation_scale=30, color='black', linewidth=2)

ax2.add_patch(arrow_FOSOF_2)

ax2.set_xlabel('$f-f_0$ (MHz)')
ax2.set_ylabel('$\\theta$ (rad)')

ax2.set_ylim(np.min(theta_arr), np.max(theta_arr))
ax2.set_xlim(np.min(delta_freq_arr), np.max(delta_freq_arr))

for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(15)

ax2_twin = ax2.twinx()

res_phase_arr = (fosof_fit(delta_freq_arr)-theta_arr)
ax2_twin.plot(delta_freq_arr, res_phase_arr, color='green', linestyle='dashed')
ax2_twin.set_ylim(np.min(res_phase_arr), np.max(res_phase_arr))
ax2_twin.set_ylabel('Residual from the linear trend (rad)')

for item in ([ax2_twin.title, ax2_twin.xaxis.label, ax2_twin.yaxis.label] +
             ax2_twin.get_xticklabels() + ax2_twin.get_yticklabels()):
    item.set_fontsize(15)

ax2.tick_params(axis='y', colors='blue')
ax2.yaxis.label.set_color('blue')
ax2_twin.tick_params(axis='y', colors='green')
ax2_twin.yaxis.label.set_color('green')
ax2_twin.yaxis.labelpad = 10
# plt.savefig('atom_lineshapes.pdf', format='pdf',  bbox_inches='tight')

plt.show()
#%%
# ''' Lamb shift energy diagram
# '''
# p_3over2_shift = 9700
# p_1over2_shift = -800
# y_fs_data_arr = np.array([0-p_3over2_shift, -9911.22, -10969.1-p_1over2_shift])
# x_fs_data_arr = np.ones(y_fs_data_arr.shape[0])
# y_fs_ticks_arr = ['$2P_{3/2}$', '$2S_{1/2}$', '$2P_{1/2}$']
#
# y_hfs_data_arr = np.array([8.87-y_fs_data_arr[0], -14.78-y_fs_data_arr[0], -9866.83, -10044.4, -10954.3-p_1over2_shift, -11013.4-p_1over2_shift])
# x_hfs_data_arr = 2 * np.ones(y_hfs_data_arr.shape[0])
# y_hfs_ticks_arr = ['$f=2$', '$f=1$', '$f=1$', '$f=0$', '$f=1$', '$f=0$']
#
# #%%
# fig, ax = plt.subplots()
#
# fig.set_size_inches(16, 10)
#
# ax.scatter(x_fs_data_arr, y_fs_data_arr)
# plt.yticks(y_fs_data_arr, y_fs_ticks_arr)
#
# ax.scatter(x_hfs_data_arr, y_hfs_data_arr)
# #plt.yticks(y_hfs_data_arr, y_hfs_ticks_arr)
#
# ax.plot([1, 2], [y_fs_data_arr[0], y_fs_data_arr[0]])
#
# plt.show()
#%%
''' Plotting of the quench-cavity quench curves.
'''

os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis/rf_cav')

pre_910_data_set_name = '180629-171507 - Quench Cavity Calibration - pre-910 PD ON'
pre_1088_data_set_name = '180629-172109 - Quench Cavity Calibration - pre-1088 PD ON'
pre_1147_data_set_name = '180629-172702 - Quench Cavity Calibration - post-1147 PD ON'

data_set_910 = DataSetQuenchCurveCavity(pre_910_data_set_name)

exp_params_s_910 = data_set_910.get_exp_parameters()

rf_pwr_det_calib_df_910 = data_set_910.get_quench_cav_rf_pwr_det_calib()
quenching_cavities_df_910 = data_set_910.get_quenching_cav_data()
quenching_cavities_av_df_910 = data_set_910.get_av_quenching_cav_data()
quenching_curve_df_910 = data_set_910.get_quenching_curve()
quenching_curve_results_s_910 = data_set_910.get_quenching_curve_results()

poly_test_910 = copy.deepcopy(data_set_910.quenching_rf_power_func)
poly_test_910[0] = poly_test_910[0] - quenching_curve_results_s_910['Quenching Offset']
p_pi_910_arr = np.roots(poly_test_910)
p_pi_910 = np.abs(p_pi_910_arr[np.isreal(p_pi_910_arr)])

y_data_910_arr = quenching_curve_df_910['Weighted Mean'].values * 10**2
x_data_910_arr = quenching_curve_df_910['RF System Power [W]'].values / p_pi_910
x_data_910_std_arr = quenching_curve_df_910['RF System Power STDOM [W]'].values / p_pi_910
y_data_910_std_arr = quenching_curve_df_910['Weighted STD'].values * 10**2

data_set_1088 = DataSetQuenchCurveCavity(pre_1088_data_set_name)

exp_params_s_1088 = data_set_1088.get_exp_parameters()

rf_pwr_det_calib_df_1088 = data_set_1088.get_quench_cav_rf_pwr_det_calib()
quenching_cavities_df_1088 = data_set_1088.get_quenching_cav_data()
quenching_cavities_av_df_1088 = data_set_1088.get_av_quenching_cav_data()
quenching_curve_df_1088 = data_set_1088.get_quenching_curve()
quenching_curve_results_s_1088 = data_set_1088.get_quenching_curve_results()

poly_test_1088 = copy.deepcopy(data_set_1088.quenching_rf_power_func)
poly_test_1088[0] = poly_test_1088[0] - quenching_curve_results_s_1088['Quenching Offset']
p_pi_1088_arr = np.roots(poly_test_1088)
p_pi_1088 = np.abs(p_pi_1088_arr[np.isreal(p_pi_1088_arr)])

y_data_1088_arr = quenching_curve_df_1088['Weighted Mean'].values * 10**2
x_data_1088_arr = quenching_curve_df_1088['RF System Power [W]'].values / p_pi_1088
x_data_1088_std_arr = quenching_curve_df_1088['RF System Power STDOM [W]'].values / p_pi_1088
y_data_1088_std_arr = quenching_curve_df_1088['Weighted STD'].values * 10**2

data_set_1147 = DataSetQuenchCurveCavity(pre_1147_data_set_name)

exp_params_s_1147 = data_set_1147.get_exp_parameters()

rf_pwr_det_calib_df_1147 = data_set_1147.get_quench_cav_rf_pwr_det_calib()
quenching_cavities_df_1147 = data_set_1147.get_quenching_cav_data()
quenching_cavities_av_df_1147 = data_set_1147.get_av_quenching_cav_data()
quenching_curve_df_1147 = data_set_1147.get_quenching_curve()
quenching_curve_results_s_1147 = data_set_1147.get_quenching_curve_results()

poly_test_1147 = copy.deepcopy(data_set_1147.quenching_rf_power_func)
poly_test_1147[0] = poly_test_1147[0] - quenching_curve_results_s_1147['Quenching Offset']
p_pi_1147_arr = np.roots(poly_test_1147)
p_pi_1147 = np.abs(p_pi_1147_arr[np.isreal(p_pi_1147_arr)])

y_data_1147_arr = quenching_curve_df_1147['Weighted Mean'].values * 10**2
x_data_1147_arr = quenching_curve_df_1147['RF System Power [W]'].values / p_pi_1147
x_data_1147_std_arr = quenching_curve_df_1147['RF System Power STDOM [W]'].values / p_pi_1147
y_data_1147_std_arr = quenching_curve_df_1147['Weighted STD'].values * 10**2
#%%
fig, ax = plt.subplots()
fig.set_size_inches(12,8)

#data_set.exp_data_averaged_df.reset_index().plot(x='RF System Power [W]', y='Weighted Mean', kind='scatter', yerr='Weighted STD', ax=ax, color='blue')

ax.errorbar(x_data_910_arr, y_data_910_arr, y_data_910_std_arr, linestyle='', marker='.', color='blue', label='pre-quench 910-MHz')

ax.errorbar(x_data_1088_arr, y_data_1088_arr, y_data_1088_std_arr, linestyle='', marker='.', color='red', label='pre-quench 1088-MHz')

ax.errorbar(x_data_1147_arr, y_data_1147_arr, y_data_1147_std_arr, linestyle='', marker='.', color='green', label='pre-quench 1147-MHz')

ax.plot([1, 1], [-1, 10**2 * quenching_curve_results_s_910['Quenching Offset']], color='black', linestyle='dashed')

ax.plot([-1, 5], [10**2 * quenching_curve_results_s_910['Quenching Offset'], 10**2 * quenching_curve_results_s_910['Quenching Offset']], color='black', linestyle='dashed')

ax.legend(fontsize=17)

ax.set_xlim([0.0, 4])
ax.set_ylim([0, 2])
ax.set_ylabel('Fraction detected')
ax.set_xlabel('Input power relative to the power required to drive a $\pi$ pulse')

tick_spacing = 0.5

ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

# We need to draw the canvas, otherwise the labels won't be positioned and
# won't have values yet.
fig.canvas.draw()

y_tick_label_list = list(ax.get_yticklabels())
y_tick_to_use_arr = np.arange(len(y_tick_label_list)-1)
y_tick_to_use_arr = y_tick_to_use_arr.astype(np.str)

for i in range(len(y_tick_label_list)-1):
    y_tick_to_use_arr[i] = y_tick_label_list[i].get_text() + '%'

ax.set_yticklabels(y_tick_to_use_arr)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(17)

os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis/rf_cav')
plt.savefig('quench_cav_curve.pdf', format='pdf',  bbox_inches='tight')

plt.show()
#%%
y_tick_label_list = list(ax.get_yticklabels())
y_tick_label_list
#%%
''' '0'- and 'pi'-configuration FOSOF plots.
'''

exp_folder_name_0 = '180321-150600 - FOSOF Acquisition - 0 config, 18 V per cm PD ON 120 V'
exp_folder_name_pi = '180321-152522 - FOSOF Acquisition - pi config, 18 V per cm PD ON 120 V'

av_type = 'Phasor Averaging'
av_data_std = 'Phase RMS Repeat STD'
#%%
# 0-configuration data

beam_rms_rad = None
# List of beam  rms radius values for correcting the FOSOF phases using the simulations. Value of None corresponds to not applying the correction.
data_set = DataSetFOSOF(exp_folder_name=exp_folder_name_0, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

# The power correction is performed only for the simple FOSOF data sets.
if data_set.get_exp_parameters()['Experiment Type'] != 'Waveguide Carrier Frequency Sweep':
    beam_rms_rad = None
    data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

fc_df = data_set.get_fc_data()
quenching_df = data_set.get_quenching_cav_data()
rf_pow_df, rf_system_power_outlier_df = data_set.get_rf_sys_pwr_det_data()

digi_df = data_set.get_digitizers_data()

comb_phase_diff_df = data_set.get_combiners_phase_diff_data()
digi_delay_df = data_set.get_inter_digi_delay_data()

if beam_rms_rad is not None:
    data_set.correct_phase_diff_for_RF_power(beam_rms_rad)

phase_diff_df = data_set.get_phase_diff_data()
phase_av_set_averaged_df = data_set.average_av_sets()
phase_A_minus_B_df, phase_freq_response_df = data_set.cancel_out_freq_response()
fosof_ampl_df, fosof_phase_df = data_set.average_FOSOF_over_repeats()

data_df = fosof_phase_df['RF Combiner I Reference', 'First Harmonic', av_type, av_data_std]

x_0_arr = data_df.index.values
y_0_arr = data_df['Weighted Mean'].values
y_0_std_arr = data_df['Weighted STD'].values

y_0_plot_arr = correct_FOSOF_phases_zero_crossing(x_0_arr, y_0_arr, y_0_std_arr) / 2
y_0_std_plot_arr = data_df['Weighted STD'].values / 2

fosof_phase_0_df = data_df
#%%
# pi-configuration data

beam_rms_rad = None
# List of beam  rms radius values for correcting the FOSOF phases using the simulations. Value of None corresponds to not applying the correction.
data_set = DataSetFOSOF(exp_folder_name=exp_folder_name_pi, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

# The power correction is performed only for the simple FOSOF data sets.
if data_set.get_exp_parameters()['Experiment Type'] != 'Waveguide Carrier Frequency Sweep':
    beam_rms_rad = None
    data_set = DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=True, beam_rms_rad_to_load=beam_rms_rad)

fc_df = data_set.get_fc_data()
quenching_df = data_set.get_quenching_cav_data()
rf_pow_df, rf_system_power_outlier_df = data_set.get_rf_sys_pwr_det_data()

digi_df = data_set.get_digitizers_data()

comb_phase_diff_df = data_set.get_combiners_phase_diff_data()
digi_delay_df = data_set.get_inter_digi_delay_data()

if beam_rms_rad is not None:
    data_set.correct_phase_diff_for_RF_power(beam_rms_rad)

phase_diff_df = data_set.get_phase_diff_data()
phase_av_set_averaged_df = data_set.average_av_sets()
phase_A_minus_B_df, phase_freq_response_df = data_set.cancel_out_freq_response()
fosof_ampl_df, fosof_phase_df = data_set.average_FOSOF_over_repeats()

data_df = fosof_phase_df['RF Combiner I Reference', 'First Harmonic', av_type, av_data_std]

x_pi_arr = data_df.index.values
y_pi_arr = data_df['Weighted Mean'].values
y_pi_std_arr = data_df['Weighted STD'].values

y_pi_plot_arr = correct_FOSOF_phases_zero_crossing(x_pi_arr, y_pi_arr, y_pi_std_arr) / 2
y_pi_std_plot_arr = data_df['Weighted STD'].values / 2

fosof_phase_pi_df = data_df
#%%
# 0-pi data

# Calculate fosof phases + their uncertainties
fosof_phase_df = (fosof_phase_0_df[['Weighted Mean']] - fosof_phase_pi_df[['Weighted Mean']]).join(np.sqrt(fosof_phase_0_df[['Weighted STD']]**2 + fosof_phase_pi_df[['Weighted STD']]**2)).sort_index(axis='columns')

# Convert the phases to the 0-2pi range
fosof_phase_df.loc[slice(None), 'Weighted Mean'] = fosof_phase_df['Weighted Mean'].transform(convert_phase_to_2pi_range)

phase_data_df = fosof_phase_df

x_data_arr = phase_data_df.index.get_level_values('Waveguide Carrier Frequency [MHz]').values
y_data_arr = phase_data_df['Weighted Mean'].values
y_sigma_arr = phase_data_df['Weighted STD'].values

# Correct for 0-2*np.pi jumps

y_data_arr = correct_FOSOF_phases_zero_crossing(x_data_arr, y_data_arr, y_sigma_arr)

#Important division by a factor of 4 that was explained before
y_data_arr = y_data_arr / 4
y_sigma_arr = y_sigma_arr / 4

fit_param_dict = calc_fosof_lineshape_param(x_data_arr, y_data_arr, y_sigma_arr)
#%%
# Plotting
fig, ax = plt.subplots()
fig.set_size_inches(12,8)
#data_set.exp_data_averaged_df.reset_index().plot(x='RF System Power [W]', y='Weighted Mean', kind='scatter', yerr='Weighted STD', ax=ax, color='blue')

ax.errorbar(x_0_arr, y_0_plot_arr, y_0_std_plot_arr, linestyle='', marker='.', color='red')

ax.errorbar(x_pi_arr, y_pi_plot_arr, y_pi_std_plot_arr, linestyle='', marker='.', color='blue')

ax.set_xlabel(r'rf frequency (MHz)')
ax.set_ylabel(r'$\theta^{AB}$ (rad)')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(17)

plt.show()
#%%
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
import mpl_toolkits.axes_grid1.inset_locator as inset_locator
#%%
x_plot_arr = np.linspace(np.min(x_data_arr), np.max(x_data_arr), 100)
y_plot_arr = fit_param_dict['Slope [Rad/MHz]'] * x_plot_arr + fit_param_dict['Offset [MHz]']
y_res_arr = fit_param_dict['Slope [Rad/MHz]'] * x_data_arr + fit_param_dict['Offset [MHz]'] - y_data_arr

fig = plt.figure()
fig.set_size_inches(10,10)

gs = gridspec.GridSpec(nrows=4, ncols=1, figure=fig, hspace=0.5)

ax_0 = plt.subplot(gs[0:3])

ax_0.errorbar(x_data_arr, y_data_arr, y_sigma_arr, linestyle='', marker='.', color='green')
ax_0.plot(x_plot_arr, y_plot_arr, color='green')

ax_1 = plt.subplot(gs[-1])

ax_1.errorbar(x_data_arr, y_res_arr*1E3, y_sigma_arr*1E3, linestyle='', marker='.', color='green')

ax_1.set_xticklabels([])

for item in ([ax_0.title, ax_0.xaxis.label, ax_0.yaxis.label] +
             ax_0.get_xticklabels() + ax_0.get_yticklabels()):
    item.set_fontsize(17)

for item in ([ax_1.title, ax_1.xaxis.label, ax_1.yaxis.label] +
             ax_1.get_xticklabels() + ax_1.get_yticklabels()):
    item.set_fontsize(17)


x_lim = ax_0.get_xlim()
y_lim = ax_0.get_ylim()

ax_0.plot([ax_0.get_xlim()[0], fit_param_dict['Zero-crossing Frequency [MHz]']], [0, 0], color='black', linestyle='dashed')

ax_0.plot([fit_param_dict['Zero-crossing Frequency [MHz]'], fit_param_dict['Zero-crossing Frequency [MHz]']], [0, ax_0.get_ylim()[0]], color='black', linestyle='dashed')

start_x = 908.5 + 0.2
start_y = 0
end_x = 908.7 + 0.2
end_y = 0
arrow_zero_cross_1 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='-|>', mutation_scale=30, color='black', linewidth=2)

ax_0.add_patch(arrow_zero_cross_1)

start_x = fit_param_dict['Zero-crossing Frequency [MHz]']
start_y = -0.1 - 0.02
end_x = fit_param_dict['Zero-crossing Frequency [MHz]']
end_y = -0.1 - 0.02 - 0.02
arrow_zero_cross_2 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='-|>', mutation_scale=30, color='black', linewidth=2)

ax_0.add_patch(arrow_zero_cross_2)

ax_0.set_xlim(x_lim)
ax_0.set_ylim(y_lim)

ax_0.set_xlabel('rf frequency (MHz)')
ax_0.set_ylabel(r'$\theta$ (rad)')

ax_1.set_ylabel('residuals'+'\n'+'(mrad)')

inset_axes = inset_locator.inset_axes(ax_0,
                        width="40%", # width = 30% of parent_bbox
                        height="30%", # height : 1 inch
                        loc='upper right',
                        borderpad=2)
inset_axes.errorbar(x_0_arr, y_0_plot_arr, y_0_std_plot_arr, linestyle='', marker='.', color='red')

inset_axes.errorbar(x_pi_arr, y_pi_plot_arr, y_pi_std_plot_arr, linestyle='', marker='.', color='blue')

inset_axes.set_xlabel(r'rf frequency (MHz)')
inset_axes.set_ylabel(r'$\theta^{AB}$ (rad)')

for item in ([inset_axes.title, inset_axes.xaxis.label, inset_axes.yaxis.label] + inset_axes.get_xticklabels() + inset_axes.get_yticklabels()):
    item.set_fontsize(17*0.7)

fig.tight_layout()

os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\FOSOF_phase_canc')
plt_name = '0-pi_config.svg'
plt.savefig(plt_name)
plt.show()
#%%
''' Plotting the changes in the extracted FOSOF frequency vs imperfect power flatness factor
'''
os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\FOSOF_pwr_dep')

# Power dependence
fosof_f0_pwr_dep_8_10_data_arr = np.loadtxt('FOSOF_f0_pwr_dep_8_10.CSV', delimiter=',', dtype=np.float64)

#[V/cm]
p_ampl_arr = (fosof_f0_pwr_dep_8_10_data_arr[:, 0]/100)**2
#[kHz]
delta_f_arr = fosof_f0_pwr_dep_8_10_data_arr[:, 1]


fosof_f0_freq_range_data_arr = np.loadtxt('FOSOF_f0_det_range_14_10.CSV', delimiter=',', dtype=np.float64)

#[V/cm]
freq_range_arr = 2 * fosof_f0_freq_range_data_arr[:, 0]
#[kHz]
delta_f_2_arr = fosof_f0_freq_range_data_arr[:, 1]


fosof_f0_p_change_fract_arr = np.loadtxt('FOSOF_f0_pwr_frac_14_8_MHz.CSV', delimiter=',', dtype=np.float64)

#[V/cm]
alpha_p_arr = fosof_f0_p_change_fract_arr[:, 0]
#[kHz]
delta_f_3_arr = fosof_f0_p_change_fract_arr[:, 1]

fig = plt.figure()
fig.set_size_inches(10, 12)

ax = fig.add_subplot(211)

ax.plot(p_ampl_arr, delta_f_arr, color='blue')

#ax.set_xticklabels([])
ax.set_ylabel(r'$\Delta f \ (\mathrm{kHz})$')
ax.set_xlabel(r'$E_0^2 \ (\mathrm{V/cm})^2$')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax.tick_params(axis='x', colors='blue')
ax.xaxis.label.set_color('blue')

ax_twin_y = ax.twiny()
ax_twin_y.plot(freq_range_arr, delta_f_2_arr, color='purple')

for item in ([ax_twin_y.title, ax_twin_y.xaxis.label, ax_twin_y.yaxis.label] +
             ax_twin_y.get_xticklabels() + ax_twin_y.get_yticklabels()):
    item.set_fontsize(15)

ax_twin_y.set_xlabel(r'$\Delta \Omega / (2\pi) \ (\mathrm{MHz})$')
ax_twin_y.xaxis.labelpad = 15
ax_twin_y.tick_params(axis='x', colors='purple')
ax_twin_y.xaxis.label.set_color('purple')

# ax_twin_y2 = ax.twiny()
# ax_twin_y2.plot(alpha_p_arr, delta_f_3_arr, color='green')
#
# ax_twin_y2.set_xlabel(r'$\alpha_p$')
#
# ax_twin_y2.tick_params(axis='x', colors='green')
# ax_twin_y2.xaxis.label.set_color('green')
#
# ax_twin_y2.spines['top'].set_position(('outward', -380))
#
# ax_twin_y2.xaxis.labelpad = -47
#
# for item in ([ax_twin_y2.title, ax_twin_y2.xaxis.label, ax_twin_y2.yaxis.label] +
#              ax_twin_y2.get_xticklabels() + ax_twin_y2.get_yticklabels()):
#     item.set_fontsize(15)
fig.tight_layout()
plt_name = 'pwr_dep_sim.svg'
plt.savefig(plt_name)
plt.show()
#%%
''' Phase difference between the combiners
'''
os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\c1_c2_phase_diff')

comb_diff_folder_name = '180405-075213 - Combiner phase monitoring - 5 V per cm'

os.chdir(comb_diff_folder_name)

comb_diff_data_df = pd.read_csv(filepath_or_buffer='data.txt', delimiter=',', dtype=np.float64, comment='#')

comb_diff_data_df['Time'] = comb_diff_data_df['Time'] - comb_diff_data_df['Time'].min()

comb_diff_data_df = comb_diff_data_df[comb_diff_data_df['Time'] <= 3600]

comb_diff_data_df = comb_diff_data_df.set_index('Time')

phase_diff_0 = comb_diff_data_df.loc[0, 'Power Combiner Phase Difference (R - I) [rad]']
#%%
t_data_arr = comb_diff_data_df.index.values / 3600
phi_diff_data_arr = (comb_diff_data_df['Power Combiner Phase Difference (R - I) [rad]'].values - phase_diff_0) * 1E3
#%%
fig = plt.figure()
fig.set_size_inches(10, 7)
ax = fig.add_subplot(111)

ax.scatter(t_data_arr, phi_diff_data_arr, color='black')
#ax.set_ylim(0.2262, 0.22725)

ax.set_ylabel('Phase difference change (mrad)')
ax.set_xlabel('Time elapsed (h)')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.tight_layout()

os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\c1_c2_phase_diff')

plt_name = 'comb_phase_diff_e_change.pdf'
plt.savefig(plt_name)

plt.show()
#%%
''' Phase difference between the combiners (for 0- and pi-configuration reversals)
'''
os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\c1_c2_phase_diff')

comb_diff_folder_name = '180405-075213 - Combiner phase monitoring - 5 V per cm'

os.chdir(comb_diff_folder_name)

comb_diff_data_df = pd.read_csv(filepath_or_buffer='data.txt', delimiter=',', dtype=np.float64, comment='#')

comb_diff_data_df['Time'] = comb_diff_data_df['Time'] - comb_diff_data_df['Time'].min()

comb_diff_data_df = comb_diff_data_df[comb_diff_data_df['Time'] >= 14000]

min_Time2 = comb_diff_data_df['Time'].min()

comb_diff_data_df = comb_diff_data_df.set_index('Time')

phase_diff_0 = comb_diff_data_df.loc[min_Time2, 'Power Combiner Phase Difference (R - I) [rad]']
#%%
t_data_arr = comb_diff_data_df.index.values / 3600
phi_diff_data_arr = (comb_diff_data_df['Power Combiner Phase Difference (R - I) [rad]'].values - phase_diff_0) * 1E3
#%%
fig = plt.figure()
fig.set_size_inches(10, 7)
ax = fig.add_subplot(111)

ax.scatter(t_data_arr, phi_diff_data_arr, color='black')
#ax.set_ylim(0.2262, 0.22725)

ax.set_ylabel('Phase difference change (mrad)')
ax.set_xlabel('Time elapsed (h)')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.tight_layout()

#os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\c1_c2_phase_diff')

#plt_name = 'comb_phase_diff_e_change.pdf'
#plt.savefig(plt_name)

plt.show()
#%%
''' Beam speed measurement plot
'''
beam_speed_path = r'E:\Google Drive\Research\Lamb shift measurement\Thesis\speed_meas'

os.chdir(beam_speed_path)

beam_speed_raw_0_data = np.loadtxt('data_raw_0.CSV', delimiter=',', dtype=np.float64)

t_delay_raw_0_arr = beam_speed_raw_0_data[:, 0]
fract_change_raw_0_arr = beam_speed_raw_0_data[:, 1]
fract_change_std_raw_0_arr = beam_speed_raw_0_data[:, 2]

beam_speed_raw_pi_data = np.loadtxt('data_raw_pi.CSV', delimiter=',', dtype=np.float64)

t_delay_raw_pi_arr = beam_speed_raw_pi_data[:, 0]
fract_change_raw_pi_arr = beam_speed_raw_pi_data[:, 1]
fract_change_std_raw_pi_arr = beam_speed_raw_pi_data[:, 2]

# Transform the data
fract_change_0_arr = -1*(1/fract_change_raw_0_arr-1)
fract_change_std_0_arr = fract_change_std_raw_0_arr/fract_change_raw_0_arr**2

fract_change_pi_arr = -1*(1/fract_change_raw_pi_arr-1)
fract_change_std_pi_arr = fract_change_std_raw_pi_arr / fract_change_raw_pi_arr**2

# fract_change_0_arr = fract_change_raw_0_arr - 1
# fract_change_std_0_arr = fract_change_std_raw_0_arr
#
# fract_change_pi_arr = fract_change_raw_pi_arr - 1
# fract_change_std_pi_arr = fract_change_std_raw_pi_arr
#%%
# Perform nonlinear least-squares fit

# Fit function for the nonlinear least-squares fitting routine
def gauss_fit_func(x, a, b, sigma, x0):
    return a * np.exp(-(x-x0)**2/(2*sigma**2)) + b

fit_raw_0, cov_raw_0 = scipy.optimize.curve_fit(f=gauss_fit_func, xdata=t_delay_raw_0_arr, ydata=fract_change_0_arr, p0=(0.175, 1, 20, 30), sigma=fract_change_std_0_arr, absolute_sigma=False)
#%%
fit_raw_0
#%%
sigma_raw_0 = np.sqrt(np.diag(cov_raw_0))
sigma_raw_0
#%%
fit_raw_pi, cov_raw_pi = scipy.optimize.curve_fit(f=gauss_fit_func, xdata=t_delay_raw_pi_arr, ydata=fract_change_pi_arr, p0=(0.175, 1, 20, -30), sigma=fract_change_std_pi_arr, absolute_sigma=False)
#%%
fit_raw_pi
#%%
sigma_raw_pi = np.sqrt(np.diag(cov_raw_pi))
sigma_raw_pi
#%%
x_fit_raw_0_arr = np.linspace(np.min(t_delay_raw_0_arr), np.max(t_delay_raw_0_arr), t_delay_raw_0_arr.shape[0]*10)
fit_raw_0_arr = gauss_fit_func(x_fit_raw_0_arr, *fit_raw_0)

x_fit_raw_pi_arr = np.linspace(np.min(t_delay_raw_pi_arr), np.max(t_delay_raw_pi_arr), t_delay_raw_pi_arr.shape[0]*10)
fit_raw_pi_arr = gauss_fit_func(x_fit_raw_pi_arr, *fit_raw_pi)

#%%
fig = plt.figure()
fig.set_size_inches(10, 7)
ax = fig.add_subplot(111)

ax.errorbar(t_delay_raw_0_arr, fract_change_0_arr, fract_change_std_0_arr, linestyle='', marker='.', color='red')

ax.errorbar(t_delay_raw_pi_arr, fract_change_pi_arr, fract_change_std_pi_arr, linestyle='', marker='.', color='blue')

ax.plot(x_fit_raw_0_arr, fit_raw_0_arr, color='red')
ax.plot(x_fit_raw_pi_arr, fit_raw_pi_arr, color='blue')

x_lim = ax.get_xlim()
y_lim = ax.get_ylim()

rect_0 = Rectangle((fit_raw_0[3] - sigma_raw_0[3], y_lim[0]), 2*sigma_raw_0[3], y_lim[1]-y_lim[0], color='red', fill=True, alpha=1)
ax.add_patch(rect_0)

rect_pi = Rectangle((fit_raw_pi[3] - sigma_raw_pi[3], y_lim[0]), 2*sigma_raw_pi[3], y_lim[1]-y_lim[0], color='blue', fill=True, alpha=1)
ax.add_patch(rect_pi)

arrow_dt_0 = mpatches.FancyArrowPatch((0, 0.10), (fit_raw_0[3], 0.10), arrowstyle='<|-|>', mutation_scale=30, color='black', linewidth=1)

ax.add_patch(arrow_dt_0)

arrow_dt_pi = mpatches.FancyArrowPatch((0, 0.10), (fit_raw_pi[3], 0.10), arrowstyle='<|-|>', mutation_scale=30, color='black', linewidth=1)

ax.plot([0, 0], [y_lim[0], y_lim[1]], linestyle='dashed', color='black')

ax.add_patch(arrow_dt_pi)

ax.set_xlim(x_lim)
ax.set_ylim(y_lim)

ax.set_xlabel(r'$\Delta t_{\mathrm{set}}$ (ns)')
ax.set_ylabel(r'$-(R_2/R_1 - 1)$')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.tight_layout()

os.chdir(beam_speed_path)

#plt_name = 'beam_speed_7cm.svg'
#plt.savefig(plt_name)

plt.show()
#%%
delta_dist = 5/1000 * 2.54
wvg_sep = 7
wvg_width = 3
#%%
sigma_raw_0[3]/fit_raw_0[3]
#%%
sigma_raw_pi[3]/fit_raw_pi[3]
#%%
v_speed = (wvg_sep+wvg_width)/((fit_raw_0[3]-fit_raw_pi[3])/2)
v_fract_unc = np.sqrt((delta_dist/wvg_sep)**2 + (delta_dist/wvg_width)**2 + (sigma_raw_0[3]/fit_raw_0[3])**2 + (sigma_raw_pi[3]/fit_raw_pi[3])**2)
[v_speed, v_speed*v_fract_unc]
#%%
''' Correction for the Second-order Doppler Shift (SOD)
'''

# These are the speeds that were experimentally measured
beam_speed_data_df = pd.DataFrame(np.array([[4, 22.17, 0.225397, 0.00227273], [4, 16.27, 0.195149, 0.00285072], [4, 49.86, 0.326999, 0.00344062], [5, 49.86, 0.319996, 0.00332496], [7, 49.86, 0.32266976, 0.002319942]]), columns=['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Beam Speed [cm/ns]', 'Beam Speed STD [cm/ns]']).set_index(['Waveguide Separation [cm]', 'Accelerating Voltage [kV]'])

# While taking data for different separation, for the same accelerating voltages, it is true that we are not keeping all of the Source parameters (=voltages) the same all the time. The spread in the values that we got for the beam speeds is the good indicator of the variability of the Source parameters that were used for the experiment. Thus the average of these values gives us the best estimate for the speed. The STDOM of the spread is added with quadruate to the RMS uncertainty in the speed values to give us the average uncertainty in the average beam speed.
def get_av_speed(df):

    if df.shape[0] > 1:
        av_s = df[['Beam Speed [cm/ns]']].aggregate(lambda x: np.mean(x))
        av_s['Beam Speed STD [cm/ns]'] = np.sqrt((np.std(df['Beam Speed [cm/ns]'], ddof=1)/np.sqrt(df['Beam Speed STD [cm/ns]'].shape[0]))**2 + np.sqrt(np.sum(df['Beam Speed STD [cm/ns]']**2/df['Beam Speed STD [cm/ns]'].shape[0]))**2)
    else:
        av_s = df.iloc[0]
    return av_s

beam_speed_df = beam_speed_data_df.groupby('Accelerating Voltage [kV]').apply(get_av_speed)

beam_speed_df = beam_speed_data_df.reset_index('Waveguide Separation [cm]').join(beam_speed_df, lsuffix='_Delete').drop(columns=['Beam Speed [cm/ns]_Delete', 'Beam Speed STD [cm/ns]_Delete'])

# We add the 6 cm and 49.86 kV point to the dataframe of beam speeds.

wvg_sep_6_s = beam_speed_df.loc[49.86].iloc[0].copy()
wvg_sep_6_s['Waveguide Separation [cm]'] = 6

beam_speed_df = beam_speed_df.append(wvg_sep_6_s)

# Calculate the SOD
# Assumed resonant frequency [MHz]
freq_diff = 909.894
# Speed of light [m/s]
c_light = 299792458

sod_shift_df = beam_speed_df.copy()
sod_shift_df['SOD Shift [MHz]'] = (1/np.sqrt(1-(beam_speed_df['Beam Speed [cm/ns]'] * 1E-2 * 1E9/c_light)**2) - 1) * freq_diff

sod_shift_df['SOD Shift STD [MHz]'] = freq_diff * beam_speed_df['Beam Speed [cm/ns]'] * 1E-2 * 1E9 * beam_speed_df['Beam Speed STD [cm/ns]'] * 1E-2 * 1E9 / ((1 - (beam_speed_df['Beam Speed [cm/ns]']/c_light)**2)**(1.5) * c_light**2)

sod_shift_df = sod_shift_df.set_index('Waveguide Separation [cm]', append=True).swaplevel(0, 1)
#%%
sod_shift_df
#%%
beam_speed_df
#%%
beam_speed_data_df
#%%
''' 910 On-Off data set example
'''

# Power detector calibration curve for the pre-910 quench cavity
data_calib_set = ZX4755LNCalibration(910)
spl_smoothing_inverse, fract_unc = data_calib_set.get_calib_curve()

# Quench curve for pre-910 cavity. We assume that for all of the data sets these quench curves were similar enough.
pre_910_data_set_name = '180629-171507 - Quench Cavity Calibration - pre-910 PD ON'
data_set_910 = DataSetQuenchCurveCavity(pre_910_data_set_name)

exp_params_s_910 = data_set_910.get_exp_parameters()

rf_pwr_det_calib_df_910 = data_set_910.get_quench_cav_rf_pwr_det_calib()
quenching_cavities_df_910 = data_set_910.get_quenching_cav_data()
quenching_cavities_av_df_910 = data_set_910.get_av_quenching_cav_data()
quenching_curve_df_910 = data_set_910.get_quenching_curve()
quenching_curve_results_s_910 = data_set_910.get_quenching_curve_results()

# This is the amount by which the detected power is different from the power that goes into the quenching cavities.
attenuation = 10 + 30 # [dBm]

saving_folder_location = 'E:/2017-10-17 Lamb Shift Measurement/Data/FOSOF analyzed data sets'
os.chdir(saving_folder_location)

exp_folder_name_s = pd.read_csv('910_on_off_fosof_list.csv', header=None, index_col=0)[1]
#%%
exp_folder_name = exp_folder_name_s[0]
exp_folder_name
#%%

data_set = fosof_data_set_analysis.DataSetFOSOF(exp_folder_name=exp_folder_name, load_Q=True, beam_rms_rad_to_load=None)

# Get average power detector reading (in Volts) for the pre-910 cavity
mean_pwr_det_volt = data_set.get_quenching_cav_data().loc[(slice(None), slice(None), 'on'), slice(None)]['Pre-Quench', '910', 'Power Detector Reading [V]'].mean()

# Convert it to Watts
mean_pwr_det_dBm = spl_smoothing_inverse(mean_pwr_det_volt) + attenuation
mean_pwr_det_Watt = 10**(mean_pwr_det_dBm/10) * 10**(-3)

# Determine the corresponding surviving fraction from the pre-910 quench curve
surv_frac = data_set_910.quenching_rf_power_func(mean_pwr_det_Watt)

# Maximum surviving fraction at 0 RF power going into the quench cavity
surf_frac_max = data_set_910.quenching_rf_power_func(0)

# Fractional offset = surviving fraction at the first pi-pulse.
frac_offset = quenching_curve_results_s_910['Quenching Offset']

# Fraction of the quenched metastable atoms (in the f=0 hyperfine state)
frac_quenched = 1-(surv_frac-frac_offset)/(surf_frac_max-frac_offset)

pre_910_states_averaged_df, pre_910_av_difference_df = data_set.analyze_pre_910_switching()

# Get mean amplitude-to-DC ratio (when pre-910 cavity is OFF) and also maximum fractional deviation from the mean, which is needed to make sure that there is no significant variation of the amplitude with frequency (not much is expected, since the frequency scan range is relatively narrow compared to the SOF linewidth)
ampl_df, phase_df = data_set.average_FOSOF_over_repeats()
mean_ampl_off = ampl_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC RMS Repeat STD', 'Weighted Mean'].loc['off'].mean()

ampl_min_dev = mean_ampl_off - ampl_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC RMS Repeat STD', 'Weighted Mean'].loc['off'].min()
ampl_max_dev = ampl_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging Relative To DC', 'Amplitude Relative To DC RMS Repeat STD', 'Weighted Mean'].loc['off'].max() - mean_ampl_off

max_fract_ampl_dev = np.max([ampl_min_dev, ampl_max_dev]) / mean_ampl_off

data_s = pre_910_av_difference_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Averaging Set STD']

data_df = pd.DataFrame([[data_set.get_exp_parameters()['Waveguide Separation [cm]'], data_set.get_exp_parameters()['Accelerating Voltage [kV]'], data_set.get_exp_parameters()['Waveguide Electric Field [V/cm]'], data_set.get_exp_parameters()['Configuration'], frac_quenched, mean_ampl_off, max_fract_ampl_dev, data_s['Weighted Mean'], data_s['Weighted STD'], data_s['Reduced Chi Squared'], data_s['P(>Chi Squared)']]], columns=['Waveguide Separation [cm]', 'Accelerating Voltage [kV]', 'Waveguide Electric Field [V/cm]', 'Configuration', 'F=0 Fraction Quenched', '<A>', 'Max(|A-<A>|)/<A>', 'Weighted Mean', 'Weighted STD', 'Reduced Chi Squared', 'P(>Chi Squared)'], index=[exp_folder_name])

on_off_data_df = pre_910_states_averaged_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD']
#%%
weighted_av_s = pre_910_av_difference_df['RF Combiner I Reference', 'First Harmonic', 'Phasor Averaging', 'Phase RMS Repeat STD']
weighted_av_s
#%%
fig = plt.figure()
fig.set_size_inches(8, 5)
ax = fig.add_subplot(111)

x_arr = on_off_data_df.reset_index()['Waveguide Carrier Frequency [MHz]']
y_arr = on_off_data_df['Weighted Mean'] * 1E3
y_std_arr = on_off_data_df['Weighted STD'] * 1E3

ax.errorbar(x_arr, y_arr, y_std_arr, linestyle='', marker='.', color='blue')

ax.set_xlabel('rf frequency (MHz)')
ax.set_ylabel(r'$\phi^{(\mathrm{ON})}-\phi^{(\mathrm{OFF})}$ (mrad)')

xlim = ax.get_xlim()

ax.plot([np.min(x_arr)-10, np.max(x_arr)+10], [weighted_av_s['Weighted Mean']*1E3, weighted_av_s['Weighted Mean']*1E3], color='green')

rect = Rectangle((np.min(x_arr)-10, weighted_av_s['Weighted Mean']*1E3-weighted_av_s['Weighted STD']*1E3), np.max(x_arr)+10-np.min(x_arr)+10, 2*weighted_av_s['Weighted STD']*1E3, color='green', fill=True, alpha=0.5)

ax.add_patch(rect)

ax.set_xlim(xlim)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.tight_layout()

os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\high-n_shift')

plt_name = '910_on_off_ex.pdf'
plt.savefig(plt_name)
plt.show()
#%%
data_df
#%%
''' Study of the imperfect quenching of the f=1 hyperfine state. Effect on the resonant frequency extracted from the FOSOF lineshape, assuming perfectly square fields
'''
os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\f=1_hyperfine')

f_shift_4cm_data = np.loadtxt('f_shift_vs_alphaf1_4_18.CSV', delimiter=',', dtype=np.float64)
f_shift_5cm_data = np.loadtxt('f_shift_vs_alphaf1_5_18.CSV', delimiter=',', dtype=np.float64)
f_shift_6cm_data = np.loadtxt('f_shift_vs_alphaf1_6_18.CSV', delimiter=',', dtype=np.float64)
f_shift_7cm_data = np.loadtxt('f_shift_vs_alphaf1_7_18.CSV', delimiter=',', dtype=np.float64)

alpha_f1_arr = f_shift_4cm_data[:, 0]

f_shift_4cm_shift_arr = f_shift_4cm_data[:, 1]
f_shift_5cm_shift_arr = f_shift_5cm_data[:, 1]
f_shift_6cm_shift_arr = f_shift_6cm_data[:, 1]
f_shift_7cm_shift_arr = f_shift_7cm_data[:, 1]

f_shift_E0_data = np.loadtxt('f_shift_vs_E0_0.005_5.CSV', delimiter=',', dtype=np.float64)
E0_arr = f_shift_E0_data[:, 0]/100
f_shift_E0_data_arr = f_shift_E0_data[:, 1]
#%%
fig = plt.figure()
fig.set_size_inches(8, 5)
ax = fig.add_subplot(111)
ax.plot(alpha_f1_arr, np.abs(f_shift_4cm_shift_arr), color='blue')
ax.plot(alpha_f1_arr, np.abs(f_shift_5cm_shift_arr), color='black')
ax.plot(alpha_f1_arr, np.abs(f_shift_6cm_shift_arr), color='red')
ax.plot(alpha_f1_arr, np.abs(f_shift_7cm_shift_arr), color='green')

ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$|\Delta_{f=1}|\,\mathrm{(kHz)}$')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax.tick_params(axis='x', colors='black')
ax.xaxis.label.set_color('black')

# ax_twin_y = ax.twiny()
# ax_twin_y.plot(E0_arr, np.abs(f_shift_E0_data_arr), color='purple')
#
# for item in ([ax_twin_y.title, ax_twin_y.xaxis.label, ax_twin_y.yaxis.label] +
#              ax_twin_y.get_xticklabels() + ax_twin_y.get_yticklabels()):
#     item.set_fontsize(15)
#
# ax_twin_y.set_xlabel(r'$E_0 \mathrm{(V/cm)}$')
# ax_twin_y.xaxis.labelpad = 15
# ax_twin_y.tick_params(axis='x', colors='purple')
# ax_twin_y.xaxis.label.set_color('purple')

fig.tight_layout()

os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\f=1_hyperfine')

plt_name = 'fosof_sim_f=1.pdf'
plt.savefig(plt_name)

plt.show()
#%%
''' Extraction of the Rabi and SOF lineshapes obtained in Mathematica for the n=2 Lamb shift, since the Mathematica has already symbolic solution typed into it.
'''
os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis/two-level atom')

lineshape_SOF_data_arr = np.loadtxt('lineshape_SOF_atom.CSV', delimiter=',', dtype=np.float64)

delta_freq_arr = lineshape_SOF_data_arr[:, 0]
fract_pop_arr = lineshape_SOF_data_arr[:, 1] * 1E3

fig = plt.figure()
fig.set_size_inches(10, 6)

ax = fig.add_subplot(111)

ax.plot(delta_freq_arr, fract_pop_arr, color='blue')

ax.set_ylabel(r'$(P^\mathrm{(SOF)}_{\pi}-P^\mathrm{(SOF)}_{0}) \times 10^{-3}$')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax.tick_params(axis='y', colors='blue')
ax.yaxis.label.set_color('blue')

lineshape_free_data_arr = np.loadtxt('lineshape_free_atom.CSV', delimiter=',', dtype=np.float64)

delta_freq_arr = lineshape_free_data_arr[:, 0]
fract_pop_arr = lineshape_free_data_arr[:, 1]

ax_twin = ax.twinx()
ax_twin.plot(delta_freq_arr, fract_pop_arr, color='black', linestyle='dashed')

ax_twin.set_ylabel(r'$P^{\mathrm{(one\,pulse)}}$')

for item in ([ax_twin.title, ax_twin.xaxis.label, ax_twin.yaxis.label] +
             ax_twin.get_xticklabels() + ax_twin.get_yticklabels()):
    item.set_fontsize(15)

ax.set_xlabel('$f-f_0$ (MHz)')

plt.savefig('sof_rabi_lineshape.pdf', format='pdf',  bbox_inches='tight')

plt.show()
#%%
''' Single E field diagram
'''

os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis/two-level atom')

E0 = 1
tau = 2 * 3

dt = tau/400
omega_E = 2 * np.pi * 1.5

time_pulse_1_arr = np.linspace(0, tau, int(tau/dt))
phi_01 = np.pi / 2
E1_arr = E0 * np.cos(omega_E*time_pulse_1_arr+phi_01)

fig = plt.figure()
fig.set_size_inches(12, 6)

ax = fig.add_subplot(111)

alpha_to_use = 1

ax.plot(time_pulse_1_arr, E1_arr, color='black')

ax.set_ylim(-2*E0, 2*E0)

x_min = -2
x_max = tau + 2

time_pulse_before_arr = np.linspace(-2, 0, int(2/dt))
E_before_arr = 0 * time_pulse_before_arr

time_pulse_after_arr = np.linspace(tau, tau+2, int((tau+2)/dt))
E_after_arr = 0 * time_pulse_after_arr

ax.plot(time_pulse_before_arr, E_before_arr, color='black', alpha=alpha_to_use)
ax.plot(time_pulse_after_arr, E_after_arr, color='black', alpha=alpha_to_use)


ax.set_xlim(x_min, x_max)

ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_axis_off()

# Drawing arrows

# Vertical lines
start_x = 0
start_y = 0
end_x = start_x
end_y = 1.3
ax.plot([start_x, end_x], [start_y, end_y], linestyle='dashed', color='black')

start_x = tau
start_y = 0
end_x = start_x
end_y = 1.3
ax.plot([start_x, end_x], [start_y, end_y], linestyle='dashed', color='black')

# Horizontal arrows
start_x = 0
start_y = 1.3
end_x = tau
end_y = 1.3
arrow_1 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='<|-|>', mutation_scale=20, color='black')
ax.add_patch(arrow_1)

# Text labels
text_x = tau / 2
text_y = 1.3 + 0.2
ax.text(x=text_x, y=text_y, s=r'$2\tau$', fontsize=15, horizontalalignment='center')

# Time axis
start_x = 0.4 * tau
start_y = -1.5
end_x = start_x + 1
end_y = -1.5
arrow_4 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='-|>', mutation_scale=30, color='black', linewidth=2)
ax.add_patch(arrow_4)

text_x = start_x + (end_x - start_x) / 2
text_y = -1.5 + 0.2
ax.text(x=text_x, y=text_y, s='Time', fontsize=15, horizontalalignment='center')

plt.savefig('single_pulse_sequence.pdf', format='pdf',  bbox_inches='tight')

plt.show()
#%%
''' FOSOF sequence diagram
'''

os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis/two-level atom')

E0 = 1
tau = 3
T = 6
dt = tau/400
omega_E1 = 2 * np.pi * 1.5
omega_E2 = 2 * np.pi * 2

time_pulse_1_arr = np.linspace(0, tau, int(tau/dt))
phi_01 = np.pi / 2
E1_arr = E0 * np.cos(omega_E1*time_pulse_1_arr+phi_01)

time_T_arr = np.linspace(tau, T+tau, int(T/dt))
ET_arr = 0 * time_T_arr

time_pulse_2_arr = np.linspace(tau+T, T+tau+tau, int(tau/dt))
phi_02 = -np.pi / 2
E2_arr = E0 * np.cos(omega_E2*time_pulse_2_arr+phi_02)

fig = plt.figure()
fig.set_size_inches(12, 6)

ax = fig.add_subplot(111)

alpha_to_use = 1

ax.plot(time_pulse_1_arr, E1_arr, color='black')
ax.plot(time_T_arr, ET_arr, color='black', alpha=alpha_to_use)
ax.plot(time_pulse_2_arr, E2_arr, color='green')

ax.set_ylim(-2*E0, 2*E0)

x_min = -0.5
x_max = T + 2 * tau + 0.5

time_pulse_before_arr = np.linspace(-0.5, 0, int(0.5/dt))
E_before_arr = 0 * time_pulse_before_arr

time_pulse_after_arr = np.linspace(T+2*tau, T+2*tau+0.5, int((T+2*tau+0.5)/dt))
E_after_arr = 0 * time_pulse_after_arr

ax.plot(time_pulse_before_arr, E_before_arr, color='black', alpha=alpha_to_use)
ax.plot(time_pulse_after_arr, E_after_arr, color='black', alpha=alpha_to_use)


ax.set_xlim(x_min, x_max)

ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_axis_off()

# Drawing arrows

# Vertical lines
start_x = 0
start_y = 0
end_x = start_x
end_y = 1.3
ax.plot([start_x, end_x], [start_y, end_y], linestyle='dashed', color='black')

start_x = tau
start_y = 0
end_x = start_x
end_y = 1.3
ax.plot([start_x, end_x], [start_y, end_y], linestyle='dashed', color='black')

start_x = tau + T
start_y = 0
end_x = start_x
end_y = 1.3
ax.plot([start_x, end_x], [start_y, end_y], linestyle='dashed', color='black')

start_x = 2 * tau + T
start_y = 0
end_x = start_x
end_y = 1.3
ax.plot([start_x, end_x], [start_y, end_y], linestyle='dashed', color='black')

# Horizontal arrows
start_x = 0
start_y = 1.3
end_x = tau
end_y = 1.3
arrow_1 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='<|-|>', mutation_scale=20, color='black')
ax.add_patch(arrow_1)

start_x = tau
start_y = 1.3
end_x = T + tau
end_y = 1.3
arrow_2 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='<|-|>', mutation_scale=20, color='black')
ax.add_patch(arrow_2)

start_x = T + tau
start_y = 1.3
end_x = T + 2 * tau
end_y = 1.3
arrow_3 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='<|-|>', mutation_scale=20, color='black')
ax.add_patch(arrow_3)

# Text labels
text_x = tau / 2
text_y = 1.3 + 0.2
ax.text(x=text_x, y=text_y, s='$\\tau$', fontsize=15, horizontalalignment='center')

text_x = tau + T / 2
text_y = 1.3 + 0.2
ax.text(x=text_x, y=text_y, s='$T$', fontsize=15, horizontalalignment='center')

text_x = T + tau + tau / 2
text_y = 1.3 + 0.2
ax.text(x=text_x, y=text_y, s='$\\tau$', fontsize=15, horizontalalignment='center')

# Time axis
start_x = tau + 0.2 * tau
start_y = -1.3
end_x = start_x + 0.7 * T
end_y = -1.3
arrow_4 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='-|>', mutation_scale=30, color='black', linewidth=2)
ax.add_patch(arrow_4)

text_x = start_x + (end_x - start_x) / 2
text_y = -1.3 + 0.2
ax.text(x=text_x, y=text_y, s='Time', fontsize=15, horizontalalignment='center')

plt.savefig('fosof_pulse_sequence_2.pdf', format='pdf',  bbox_inches='tight')

plt.show()
#%%
''' Probability oscillation from Mathematica'''

os.chdir(r'E:\Google Drive\Research\Lamb shift measurement\Thesis\two-level atom')

prob_osc_data_arr = np.loadtxt('prob_osc_800Hz_2_pulses_14_5.CSV', delimiter=',', dtype=np.float64)

omega_offset = 800 * 2 * np.pi
t_arr = prob_osc_data_arr[:, 0]
pop_arr = prob_osc_data_arr[:, 1]
mixed_arr = np.cos(omega_offset * t_arr)

# Fit function for the nonlinear least-squares fitting routine
def cos_fit_func(t, A, phi, P0):
    return A * np.cos(omega_offset*t + phi) + P0

fit_raw_0, cov_raw_0 = scipy.optimize.curve_fit(f=cos_fit_func, xdata=t_arr, ydata=pop_arr, p0=(0.0002, 1, 0.5))

amp = fit_raw_0[0]
phase = fit_raw_0[1]
P0 = fit_raw_0[2]
#%%
fig = plt.figure()
fig.set_size_inches(10, 6)

ax = fig.add_subplot(111)
ax_twin = ax.twinx()

ax_twin.plot(t_arr, pop_arr-P0, color='blue')
ax_twin.tick_params(axis='y', colors='blue')
ax_twin.yaxis.label.set_color('blue')

ax_twin.set_ylabel(r'$P^{\mathrm{(FOSOF)}}_{\Delta\omega t}-P_0$')
ax.set_xlabel('time')

ax.plot(t_arr, mixed_arr, color='purple', linestyle='dashed')
ax.tick_params(axis='y', colors='purple')
ax.yaxis.label.set_color('purple')

ax.set_ylabel('Beatnote signal (arb. units)')
ax.set_yticklabels([])
ax.set_yticks([])

T_per = 2 * np.pi / omega_offset
x_tick_arr = np.array([0, T_per/2, T_per, 3*T_per/2, 2*T_per])
ax_twin.set_xticks(x_tick_arr)

x_tick_label_arr = np.array([r'$0$', r'$\pi/\Delta\omega$', r'$2\pi/\Delta\omega$', r'$3\pi/\Delta\omega$', r'$4\pi/\Delta\omega$'])
ax.set_xticklabels(x_tick_label_arr)

y_tick_arr = np.array([-amp,-amp/2, 0, amp/2, amp])
ax_twin.set_yticks(y_tick_arr)

y_tick_label_arr = np.array([r'$-A$', r'$-A/2$', r'$0$', r'$A/2$', r'$A$'])
ax_twin.set_yticklabels(y_tick_label_arr)

n = 0.5
t_1 = T_per * (n+1) - phase * T_per / (2*np.pi)
ax_twin.axvline(x=t_1, color='green', linewidth=1)

n = 0
t_2 = T_per * (n+1)
ax.axvline(x=t_2, color='green', linewidth=1)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

for item in ([ax_twin.title, ax_twin.xaxis.label, ax_twin.yaxis.label] +
             ax_twin.get_xticklabels() + ax_twin.get_yticklabels()):
    item.set_fontsize(15)

plt.savefig('fosof_phase_ex.pdf', format='pdf',  bbox_inches='tight')

plt.show()
#%%
# FOSOF phase
fosof_ph_data_arr = np.loadtxt('FOSOF_phase_atom.CSV', delimiter=',', dtype=np.float64)

delta_freq_arr = fosof_ph_data_arr[:, 0]
theta_arr = fosof_ph_data_arr[:, 1]

fosof_fit = np.poly1d(np.polyfit(delta_freq_arr, theta_arr, 1))
#%%
fig, axes = plt.subplots(2, 1)
fig.set_size_inches(10, 10)

#ax = fig.add_subplot(111)
#ax2 = fig.add_subplot(212)

axes[0].plot(delta_freq_arr, theta_arr, color='blue')

x_ax = axes[0].get_xlim()
y_ax = axes[0].get_ylim()

axes[0].plot([0, 0], [-35, 0], color='black', linestyle='dashed')
axes[0].plot([-80,0], [0, 0], color='black', linestyle='dashed')

axes[0].plot([10, 10], [-35, phase], color='green')
axes[0].plot([-80, 10], [phase, phase], color='green')

start_x = -40
start_y = 0
end_x = -35
end_y = 0
arrow_FOSOF_1 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='-|>', mutation_scale=30, color='black', linewidth=2)

axes[0].add_patch(arrow_FOSOF_1)

start_x = 0
start_y = -5.5
end_x = 0
end_y = -4
arrow_FOSOF_2 = mpatches.FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='-|>', mutation_scale=30, color='black', linewidth=2)

axes[0].add_patch(arrow_FOSOF_2)

axes[0].set_xlabel(r'$f-f_0$ (MHz)')
axes[0].set_ylabel(r'$\theta$ (rad)')

#axes[0].set_ylim(np.min(theta_arr), np.max(theta_arr))
#axes[0].set_xlim(np.min(delta_freq_arr), np.max(delta_freq_arr))
axes[0].set_xlim(x_ax)
axes[0].set_ylim(y_ax)

#axes[0].set_xticklabels([])

for item in ([axes[0].title, axes[0].xaxis.label, axes[0].yaxis.label] +
             axes[0].get_xticklabels() + axes[0].get_yticklabels()):
    item.set_fontsize(15)

res_phase_arr = (fosof_fit(delta_freq_arr)-theta_arr)
axes[1].plot(delta_freq_arr, res_phase_arr, color='black', linestyle='dashed')
axes[1].set_ylim(np.min(res_phase_arr), np.max(res_phase_arr))
axes[1].set_ylabel(r'$\Gamma$ (rad)')

axes[1].set_xlabel(r'$f-f_0$ (MHz)')

for item in ([axes[1].title, axes[1].xaxis.label, axes[1].yaxis.label] +
             axes[1].get_xticklabels() + axes[1].get_yticklabels()):
    item.set_fontsize(15)

#ax2.tick_params(axis='y', colors='blue')
#ax2.yaxis.label.set_color('blue')
#ax2.tick_params(axis='y', colors='green')

#ax2n.yaxis.label.set_color('green')
#ax2.yaxis.labelpad = 10

plt.savefig('fosof_lineshape_ex.pdf', format='pdf',  bbox_inches='tight')

plt.show()
#%%
''' Plot of proton radii from Hydrogen spectroscopy
'''

os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis/two-level atom')

p_data_spect_df = pd.read_csv(filepath_or_buffer='p_radius_h_spect.CSV', sep=',', header=None)

y_data_arr = p_data_spect_df.index.values
y_ax_label_arr = p_data_spect_df[0].values
x_data_arr = p_data_spect_df[1].values
x_std_data_arr = p_data_spect_df[2].values

fig = plt.figure()
fig.set_size_inches(10, 6)

ax = fig.add_subplot(111)

ax.errorbar(x=x_data_arr, y=y_data_arr, xerr=x_std_data_arr, linestyle='', marker='.', color='blue')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()
#%%
p_data_spect_df = pd.read_csv(filepath_or_buffer='p_radius_h_spect_real_data.CSV', sep=',', header=None)
#%%
y_data_arr = p_data_spect_df.index.values
y_ax_label_arr = p_data_spect_df[0].values
x_data_arr = p_data_spect_df[1].values
x_std_data_arr = p_data_spect_df[2].values

fig = plt.figure()
fig.set_size_inches(10, 6)

ax = fig.add_subplot(111)

ax.errorbar(x=x_data_arr, y=y_data_arr, xerr=x_std_data_arr, linestyle='', color='black', elinewidth=1, capsize=5, capthick=1, marker='.', markersize='15')

ax.errorbar(x=[0.8764], y=[7.5], xerr=[0.0089],linestyle='', color='blue', elinewidth=2, capsize=7, capthick=1, marker='.', markersize='15')

ax.errorbar(x=[0.84087], y=[7.5], xerr=[0.00039],linestyle='', color='brown', elinewidth=2, capsize=7, capthick=1, marker='.', markersize='10')

ax.set_yticks(y_data_arr)
ax.set_yticklabels(y_ax_label_arr)

ax.set_xlabel('Proton RMS charge radius, $r_p$ (fm)')

ax.set_xlim(0.82, 0.95)

index = 1
rect = Rectangle((0.8764 - 0.0089, -10), 2*0.0089, 25, color='blue', fill=True, alpha=0.5)
ax.add_patch(rect)

rect = Rectangle((0.84087 - 0.00039, -10), 2*0.00039, 25, color='brown', fill=True, alpha=0.5)
ax.add_patch(rect)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis/two-level atom')

plt.savefig('proton_rad_h_data.pdf', format='pdf',  bbox_inches='tight')


plt.show()
#%%
'''
===================
Proton radius data plot with our measurement
===================
'''
p_rad_data_arr = np.array([0.8751, 0.8764, 0.879, 0.84087, 0.899, 0.877, 0.8335, 0.833])
p_rad_data_std_arr = np.array([0.0061, 0.0089, 0.011, 0.00039, 0.059, 0.013, 0.0095, 0.010])
y_data_arr = np.array([0, 7, 1, 3, 6, 2, 5, 4])

color_list = ['red', 'blue', 'green', 'brown', 'purple', 'black', 'magenta', 'black']
y_ticks_arr = ['CODATA 2014', 'H spectroscopy', 'e-p scattering', '$\mu \mathrm{H}$ spectroscopy', 'Lundeen & Pipkin', '1S-3S (2018)', '2S-4P (2017)', '2S-2P (this work)']

p_size_df = pd.DataFrame({'Proton Radius [fm]': p_rad_data_arr, 'Proton Radius STD [fm]': p_rad_data_std_arr, 'Y-axis Position': y_data_arr, 'Color': color_list, 'Label Name': y_ticks_arr}).set_index('Label Name')

arrow_length = p_size_df.loc['H spectroscopy','Proton Radius [fm]'] - p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]']
std_dev = np.round(arrow_length / p_size_df.loc['H spectroscopy','Proton Radius STD [fm]'], 1)

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

plt.yticks(y_data_arr, ['CODATA 2014', 'H spectroscopy',  'e-p scattering', '$\mu \mathrm{H}$ spectroscopy', 'Lundeen & Pipkin', '1S-3S (2018)', '2S-4P (2017)', '2S-2P (this work)'])

for ytick, color in zip(ax.get_yticklabels(), color_list):
    ytick.set_color(color)

arr_width = 0.01
head_width = 10 * arr_width
head_length = 0.15 * arrow_length

#plt.arrow(x=p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]'], y=2.25, dx=arrow_length-head_length, dy=0, width=arr_width, head_width=head_width, head_length=head_length, shape='full')

#plt.annotate(s=str(std_dev) + '$\sigma$', xy=(1,1), xytext=(0,0), arrowprops=dict(arrowstyle='<->'))

ax.annotate(xy=(p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]'], 2.25+0.4), xytext=(p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]']+arrow_length, 2.25+0.4), text='', arrowprops=dict(arrowstyle='<|-|>', connectionstyle='arc3', facecolor='black', edgecolor='black', shrinkA=0, shrinkB=0, mutation_scale=20))

ax.text(p_size_df.loc['$\mu \mathrm{H}$ spectroscopy','Proton Radius [fm]']+arrow_length/2, 2.45+0.4, str(std_dev) + '$\sigma_{\mathrm{H}}$', fontsize=13, horizontalalignment='center')
ax.set_xlabel('Proton RMS charge radius, $r_\mathrm{p}$ (fm)')


index = 7
rect = Rectangle((p_rad_data_arr[index] - p_rad_data_std_arr[index]-0.002, y_data_arr[index]-0.5), 2*p_rad_data_std_arr[index]+2*0.002, 1, color=color_list[index], fill=False, alpha=1)
ax.add_patch(rect)

ax.set_xlim(left=0.82)
ax.set_xlim(right=0.92)

os.chdir(r'C:\Users\Helium1\Google Drive\Research\Lamb shift measurement\Thesis\p_rad')
plt.savefig('proton_rad_data_all.pdf', format='pdf',  bbox_inches='tight')
plt.show()
#%%
p_rad_data_std_arr
