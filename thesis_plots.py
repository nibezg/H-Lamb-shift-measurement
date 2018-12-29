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

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches


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
os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis')

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

ax.set_xlim(right=0.92)

#plt.savefig('proton_rad_data.pdf', format='pdf',  bbox_inches='tight')
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
os.chdir('E:/Google Drive/Research/Lamb shift measurement/Thesis/two-level atom')

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
plt.savefig('atom_lineshapes.pdf', format='pdf',  bbox_inches='tight')

plt.show()
