''' Monte Carlo simulation for determining RMS atomic beam radius

Date: 2018-10-30
Author: Nikita Bezginov
'''
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
import scipy.stats

import matplotlib.pyplot as plt

#%%
# Dimensions used in the simulation

# Pumping restriction radius of the Charge Exchange (CEX) [m]
r_CEX_pump_rest = 4.826/2 * 1E-3

# Effective inner radius of the Proton Deflector (PD) [m]. The PD is not round, but has square cross-section due to four square plates that have electric potential applied to them. The inner radius of this section can be defined as the radius of the circle inscribed into the square cross-section.
r_PD_inner = 50.8/2 * 1E-3

# Length of the PD deflecting plates [m].
PD_plates_length = 749.3 * 1E-3

# Distance from the start or end of the body of the PD to the start or the end of the deflecting plates [m]
PD_body_to_plates_dist = 20.574 * 1E-3

# Inner radius of the PD outer body [m] (6" CF nipple)
r_PD_body_inner = 97.3836/2 * 1E-3

# Length of the 6" CF flange at the end of the PD [m]
cf_6_length = 19.812 * 1E-3

# Radius of the bore in the 6" CF at the end of the PD [m]
r_CF_6_bore = 39.878/2 * 1E-3

z_PD_start = 0
z_PD_plates_start = z_PD_start + PD_body_to_plates_dist
z_PD_plates_end = z_PD_plates_start + PD_plates_length
z_PD_nipple_end = z_PD_plates_end + PD_body_to_plates_dist
z_PD_end = z_PD_nipple_end + cf_6_length

# Distance from the end of the 6" Reducer CF for the PD to the beginning of the middle part of the beam diameter limiting aperture (BDLA) [m].
dist_PD_end_to_BDLA_middle_part = (106.4959 + 7.62) * 1E-3

# Radius of the inner opening for the space between the PD and the middle part of the BDLA [m]
r_PD_to_to_middle = 38.1/2 * 1E-3

# Distance from the middle part of the BDLA to the aperture [m]
dist_BDLA_middle_to_aperture = 27.94 * 1E-3

# Radius of the middle part of the BDLA [m]
r_BDLA_middle = 19.05/2 * 1E-3

# Radius of the BDLA aperture [m]
r_BDLA_aperture = 3.9688/2 * 1E-3

z_BDLA_middle = z_PD_end + dist_PD_end_to_BDLA_middle_part
z_BDLA_aperture = z_BDLA_middle + dist_BDLA_middle_to_aperture

# Distance form the BDLA aperture to the edge of the RF choke tube of the pre-quench cavity stack [m]. For simplicity I assume that in that space the radius is that of the inner radius of the RF choke tubes for the quench cavity stacks.
dist_BDLA_to_pre_quench_cav_stack = 7.3875 * 1E-3

# Inner radius of the quench cavities' RF choke tubes [m].
r_inner_quench_cav = 22.098/2 * 1E-3

# Length of the quench cavity stack (with the RF choke tubes)
quench_cav_stack_length = 226.525 * 1E-3

# I assume that the radius of the quench cavity stack is the same along the whole length of the stack. This is not true in reality, but it should not matter for this simulation.

# Notice that I am assuming that the pre-quench cavity stack starts right after the aperture, and I am including the distance from the end of the BLDA to the beginning of the pre-quench stack to the length of the pre-quench stack. This is done for simplicity.
z_pre_quench_cav_start = z_BDLA_aperture
z_pre_quench_cav_end = z_pre_quench_cav_start + dist_BDLA_to_pre_quench_cav_stack + quench_cav_stack_length

# Inner radius of the waveguides' RF choke tubes [m].
r_inner_wvg = 19.9898/2 * 1E-3

# Distances below are given for the waveguides separated by 7 cm

# Distance from the end of the RF choke at the end of the pre-quench cavity stack to the closest RF choke tube face of the waveguides. [m]. This distance is the same for the post-quench cavity stack to the waveguides' closest RF choke tube face.
dist_quench_cav_stack_to_wvg = 59.7091 * 1E-3

# I assume that the radius of the waveguides is the same along the whole length of the waveguides. This is not true in reality, but it should not matter for this simulation.

# Length of the waveguides [mm]
wvg_length = 224.4099 * 1E-3

# Here I include the distance from the end of the pre-quench cavity stack to the start of the waveguides into the length of the waveguides. And the start of the waveguides is assumed to begin right after the pre-quench cavity stack.
z_wvg_start = z_pre_quench_cav_end
z_wvg_end = z_wvg_start + dist_quench_cav_stack_to_wvg + wvg_length

# Here I include the distance from the end of the waveguides to the start of the post-quench cavity stack as the part of the post-quench cavity stack.
z_post_quench_cav_start = z_wvg_end
z_post_quench_cav_end = z_post_quench_cav_start + dist_quench_cav_stack_to_wvg + quench_cav_stack_length

# Distance from the end of the RF choke tube of the post-quench cavity stack to the beginning of the Lyman-alpha detector (its shielding atom tube) [m].
dist_quench_cav_to_det = 7.8638 * 1E-3

# Inner diameter of the shielding atom tube [m]
r_inner_det_shield_tube = 20.447/2 * 1E-3

# Length of the shielding atom tube for the Lyman-alpha detector [m]
det_shield_tube_length = 89.45 * 1E-3

# Distance from the end of the shielding tube to grounded aperture of the detector [m]. It is assumed that if the atom reaches beyond this point, then it gets detected as our signal.
shield_tube_to_GND_ring = 31.7344 * 1E-3

# Inner radius of the Lyman-alpha detector grounded ring [m]
r_det_ring = 17.0658/2 * 1E-3

# Here I assume that the Lyman-alpha detector starts right after the post-quench cavity stack. The distance between these two components is included into the length of the detector.
z_det_shield_tube_start = z_post_quench_cav_end
z_det_GND_ring = z_det_shield_tube_start + dist_quench_cav_to_det + det_shield_tube_length + shield_tube_to_GND_ring

# Maximum position along the experiment axis that the atoms should cross in order to be considered to be part of the detected signal.
z_max = z_det_GND_ring

def collision_check(r_now, r_prev):
    ''' Checking whether the atom has collided with the cylindrical walls or circular apertures.

    This code works only for the case when all of the cylindrical components are axial w.r.t each other.

    Inputs:
    :r_now: np.array([x,y,z]) (floats). Current posisiton of the particle
    :r_before: np.array([x,y,z]) (floats). Previous posiiton of the particle (1 step before)
    '''
    collision_Q = False

    # z-axis is assumed to be the experiment axis.

    # distance from the experiment axis (xy-plane)
    r_norm_xy = np.linalg.norm(r_now[[0, 1]])

    r_z = r_now[2]
    r_z_before = r_prev[2]

    # The check for collision is very simple. If the particle is in a given region, then we check if its current distance from the axis is larger or equal than the radius of the respective cylinder.

    # If we have an aperture of zero length, then the check is a bit different. I first compare if the particle has crossed the aperture z-location. This is done by comparing the current and the previous z-position of the particle. Then I check if the current distance from the axis is larger or equal than the radius of the aperture.

    # The whole collision check algorithm has many-many nested if-else statements for making the code to go through the conditional statements as fast as I could make it. We start the collision check from the left-most element and more to the right, until we find the range of z-locations in which the particle is currently located.

    # For the PD region
    if r_z >= z_PD_start and r_z <= z_PD_end:
        # From the start of the PD to the start of the deflecting plates
        if r_z <= z_PD_plates_start:
            if r_norm_xy >= r_PD_body_inner:
                collision_Q = True
        else:
            # From the start to the end of the deflecting plates
            if r_z <= z_PD_plates_end:
                if r_norm_xy >= r_PD_inner:
                    collision_Q = True
            else:
                # From the end of the deflecting plates to the start of the 6" CF reducer flange
                if r_z <= z_PD_nipple_end:
                    if r_norm_xy >= r_PD_body_inner:
                        collision_Q = True
                else:
                    # From the start to the end of the reducer 6" CF flange
                    if r_z <= z_PD_end:
                        if r_norm_xy >= r_CF_6_bore:
                            collision_Q = True
    else:
        # For the BDLA:
        if r_z <= z_BDLA_aperture:
            # From the end of the PD to the middle of the BDLA
            if r_z <= z_BDLA_middle:
                if r_norm_xy >= r_PD_to_to_middle:
                    collision_Q = True
            else:
                # From the middle of the BDLA to its aperture
                if r_z <= z_BDLA_aperture:
                    if r_norm_xy >= r_BDLA_middle:
                        collision_Q = True
        else:
            # For the pre-quench cavity stack
            if r_z <= z_pre_quench_cav_end:
                # Check if the atom has transitioned through the BDLA aperture
                if r_z_before < z_BDLA_aperture and r_norm_xy >= r_BDLA_aperture:
                    collision_Q = True
                else:
                    if r_norm_xy >= r_inner_quench_cav:
                        collision_Q = True
            else:
                # For the waveguides
                if r_z >= z_pre_quench_cav_end and r_z <= z_wvg_end:
                    if r_norm_xy > r_inner_wvg:
                        collision_Q = True
                else:
                    # For the post-quench cavity stack
                    if r_z >= z_wvg_end and r_z <= z_post_quench_cav_end:
                        if r_norm_xy > r_inner_quench_cav:
                            collision_Q = True
                    else:
                        # For the Lyman-alpha detector
                        if r_z >= z_post_quench_cav_end:
                            if r_z <= z_det_GND_ring:
                                if r_norm_xy > r_inner_det_shield_tube:
                                    collision_Q = True
                            else:
                                if r_z_before < z_det_GND_ring and r_norm_xy > r_det_ring:
                                    collision_Q = True
    return collision_Q

def construct_cyl(p_i, p_f, cyl_rad, cyl_length, n_points_length=50, n_points_angle=50, angle_max=2*np.pi):
    ''' Construct the cylinder of given radius and length in 3D by specifying its axis vector, defined by the initial and final points of the vector.

    The cylinder is constructed by finding the set of unit vectors (e_1 and e_2) perpendicular to the normalized axis vector (e_3). These three vectors form the orthonormal basis set. Then polar coordinates are used to find the components along e_1 and e_2 axes for each value alond e_3 axis.
    Inputs:
    :p_i: np.array of 3 elements [x, y, z]. Coordinates of the initial point of the axis vector
    :p_f: np.array of 3 elements [x, y, z]. Coordinates of the final point of the axis vector

    The axis vector gets normalized. Thus the magnitude of the axis vector is not important.

    :cyl_rad: (float) radius of the cylinder
    :cyl_length: (float) length of the cylinder
    :n_points_length: (int) number of points in the np.linspace function for constructing the points along the axis of the cylinder
    :n_points_angle: (int) number of angle values used to calculate values parallel to the e_1 and e_2 unit vectors.
    :angle_max: (float) The array of polar angles is formed from 0 to this value (in radians). It is useful if one does not want to form the full cylinder, but only a part of it.

    :Outputs:
    :x_arr, y_arr, z_arr: np.array(float) of coordinates in the original system of cartesian coordinates.
    '''

    # Form the axis vector and normalize it
    ax_vec = p_f - p_i
    ax_vec = ax_vec/np.linalg.norm(ax_vec)

    # Pick some other vector that is not parallel to the axis vector. I choose it to point either along x- or y-axis
    other_vec = [0, -1, 0]
    if np.array_equal(other_vec, ax_vec):
        other_vec = [1, 0, 0]

    # Find the unit vectors e_1 and e_2 perpendicular to each other and e_3 = axis vector
    e1_vec = np.cross(ax_vec, other_vec)
    e1_vec = e1_vec / np.linalg.norm(e1_vec)

    e2_vec = np.cross(e1_vec, ax_vec)
    e2_vec = e2_vec / np.linalg.norm(e2_vec)

    # Form the arrays of coordinates of the cylinder in a type of simplified cylindrical coordinate system.
    ax_arr = np.linspace(0, cyl_length, n_points_length)
    theta_angle_arr = np.linspace(0, angle_max, n_points_angle)

    # Form the array of the coordinates of points on the surface of the cylinder. First we form a grid of points and then for each axis of the grid we perform the calculation below
    ax_arr, theta_angle_arr = np.meshgrid(ax_arr, theta_angle_arr)

    x_arr, y_arr, z_arr = [p_i[i] + ax_vec[i] * ax_arr + cyl_rad * (np.cos(theta_angle_arr) * e1_vec[i] + np.sin(theta_angle_arr) * e2_vec[i]) for i in [0, 1, 2]]

    return x_arr, y_arr, z_arr

def cylinder_2D_plot(ax, z_loc, cyl_rad, cyl_length, color='black'):
    ''' Auxiliary function for making 2D plot along the xz plane of the cross-sections of experiment components modeled as cylinders with their axis centered on the z-axis.
    :Inputs:
    :z_loc: (float) starting position of the cylinder along z-axis
    :cyl_rad: (float) radius of the cylinder
    :cyl_length: (float) length of the cylinder
    :color: color of the cross-section lines in the plot
    :ax: axis for plotting
    '''
    p_i = np.array([0,0,z_loc])
    p_f = np.array([0,0,z_loc+1])

    x_arr, y_arr, z_arr = construct_cyl(p_i, p_f, cyl_rad, cyl_length, n_points_length=50, n_points_angle=2, angle_max=np.pi)

    ax.plot(z_arr[0, :], x_arr[0, :], color=color)
    ax.plot(z_arr[1, :], x_arr[1, :], color=color)

    return ax
#%%
''' Simulation parameters

The atoms are assumed to move with the same speed, v_norm. Each atom has its initial position specified as (x, y, z=0). The x-, and y-components are specified in polar coordinates: r in [0, r_max], r_theta in [0, r_theta_max]. These polar coordinates are then converted into the cartesian coordinates. The velocity is specified independently of the position in spherical coordinates: {(v_norm, v_theta, v_phi)}, where theta in [0, v_theta_max] and v_phi in [0, 2*np.pi]. Again, these coordinates are then converted into the cartesian system. For the given atoms the velocity vector is assumed to be the same throughout its trajectory.

The r, r_theta, v_theta, v_phi are each randomly picked from the uniform distribution bounded by the above mentioned respective ranges.
'''
# Time step [s]
delta_t = 2.5 * 1E-5
# Maximum number of steps
max_steps = 1000
# Magnitude of the velocity vector [m/s]
v_norm = 100
# Number of atoms = number of simulation runs
sim_num = 40000
# Numpy cannot create very large arrays. That is why I need to subdivide the total number of atoms into chunks and run these chunks separately, and then combine the resulting data together.
sim_chunk_size = int(1E3)
sim_num_chunks = int(sim_num / sim_chunk_size)

# Maximum allowed r_phi angle [rad]
r_phi_angle_range = 6.28/1000

# Maximum allowed v_theta angle [rad]
v_range_theta_angle = np.pi/288/4


#This is needed for selecting the positions of the atoms after the simulations within the specified ranges of their z-locations.
#---------------------------
# Maximum displacement of the atom along z-axis.
dz_max = v_norm * delta_t

# z-coordinate of the start, the middle, and the end of the waveguides.
z_start = z_pre_quench_cav_end + dist_quench_cav_stack_to_wvg
z_end = z_wvg_start + wvg_length
z_mid = z_start + (z_end - z_start) / 2
#---------------------------

# Maximum allowed initial distance of the atoms from the axis
r0_max = r_CEX_pump_rest
#%%
# Here the simulation is ran

# Arrays for storing specific ranges of the position of the atoms that made it to the detector
r_reached_set_arr = np.array([])
r_wvg_start_set_arr = np.array([])
r_wvg_mid_set_arr = np.array([])
r_wvg_end_set_arr = np.array([])

# For each simulation chunk
for sim_chunk_set in range(sim_num_chunks):
    print(sim_chunk_set)

    # Initializing various arrays
    r_record_set_arr = np.zeros((sim_chunk_size, max_steps, 3))
    steps_count_arr = np.zeros(sim_chunk_size)
    collision_Q_arr = np.ones(sim_chunk_size)
    reached_end_Q_arr = np.ones(sim_chunk_size)
    collision_Q_arr = collision_Q_arr.astype(dtype=bool)
    reached_end_Q_arr = reached_end_Q_arr.astype(dtype=bool)

    # Initializing initial positions and velocities of the atoms
    r0_norm_arr = r0_max * np.random.rand(sim_chunk_size)

    r_phi_angle_arr = np.random.rand(sim_chunk_size)

    r_phi_angle_arr = r_phi_angle_range * r_phi_angle_arr

    # This is needed for testing - when I was allowing the simulation to run only along xz plane.
    # r_phi_angle_arr[r_phi_angle_arr > 0.5] = np.pi
    # r_phi_angle_arr[r_phi_angle_arr <= 0.5] = 0

    v_phi_angle_arr = np.random.rand(sim_chunk_size)
    v_phi_angle_arr = 2 * np.pi * v_phi_angle_arr

    # v_phi_angle_arr[v_phi_angle_arr > 0.5] = np.pi
    # v_phi_angle_arr[v_phi_angle_arr <= 0.5] = 0

    v_theta_angle_arr = v_range_theta_angle * np.random.rand(sim_chunk_size)

    v_arr = np.zeros((sim_chunk_size, 3))
    r0_arr = np.zeros((sim_chunk_size, 3))

    r0_arr[:, 0] = r0_norm_arr * np.cos(r_phi_angle_arr)
    r0_arr[:, 1] = r0_norm_arr * np.sin(r_phi_angle_arr)
    r0_arr[:, 2] = 0

    v_arr[:, 0] = v_norm * np.sin(v_theta_angle_arr)*np.cos(v_phi_angle_arr)
    v_arr[:, 1] = v_norm * np.sin(v_theta_angle_arr)*np.sin(v_phi_angle_arr)
    v_arr[:, 2] = v_norm * np.cos(v_theta_angle_arr)

    # For each simulation in the simulation chunk
    for sim_counter in range(sim_chunk_size):
        v = v_arr[sim_counter]
        r0 = r0_arr[sim_counter]

        r_record_arr = np.zeros((max_steps,3))

        # Collion flag
        collision_Q = False
        # The atom made it to the detector flag
        reached_end_Q = False

        r_record_arr[0] = r0

        i = 0
        while not(collision_Q) and i < max_steps-1 and not(reached_end_Q):
            i = i + 1
            # Stepping in positon of the atom
            r = r0 + v * delta_t
            r0 = r
            r_record_arr[i] = r
            # Check for collision
            collision_Q = collision_check(r_record_arr[i], r_record_arr[i-1])
            # Check whether the atom reached the maximum needed z-position
            if r[2] > z_max: reached_end_Q = True

        steps_count_arr[sim_counter] = i
        r_record_set_arr[sim_counter] = r_record_arr
        collision_Q_arr[sim_counter] = collision_Q
        reached_end_Q_arr[sim_counter] = reached_end_Q

    # We now select only the atoms that made it to the detector. Out of these atoms we select only the r vectors that are within the specified range of its z-components. Specifically, here we select the position of the atoms at their maximum z-component, i.e., when they reached the detector. We also select the positions of the atoms when they just reached the waveguides, are in the middle of the waveguides, and about to exit the waveguides.

    r_final_arr = np.array([])
    r_wvg_start_s_chunk_arr = np.array([])
    r_wvg_end_s_chunk_arr = np.array([])
    r_wvg_mid_s_chunk_arr = np.array([])

    for sim_counter in range(sim_chunk_size):
        if not collision_Q_arr[sim_counter] and reached_end_Q_arr[sim_counter]:
            r_pos_arr = r_record_set_arr[sim_counter, 0:int(steps_count_arr[sim_counter]+1)]
            r_final_arr = np.append(r_final_arr, r_pos_arr[-1])

            r_wvg_start_arr = r_pos_arr[np.where((r_pos_arr[:,2] >= z_start) & (r_pos_arr[:,2] <= z_start + dz_max))]
            r_wvg_start_s_chunk_arr = np.append(r_wvg_start_s_chunk_arr, r_wvg_start_arr)

            r_wvg_end_arr = r_pos_arr[np.where((r_pos_arr[:,2] >= z_end - dz_max) & (r_pos_arr[:,2] <= z_end))]
            r_wvg_end_s_chunk_arr = np.append(r_wvg_end_s_chunk_arr, r_wvg_end_arr)

            r_wvg_mid_arr = r_pos_arr[np.where((r_pos_arr[:,2] >= z_mid - dz_max/2) & (r_pos_arr[:,2] <= z_mid + dz_max/2))]
            r_wvg_mid_s_chunk_arr = np.append(r_wvg_mid_s_chunk_arr, r_wvg_mid_arr)

            if r_wvg_start_arr.shape[0] == 0:
                print('No elements for the start of the waveguides')
            if r_wvg_mid_arr.shape[0] == 0:
                print('No elements for the middle of the waveguides')
            if r_wvg_end_arr.shape[0] == 0:
                print('No elements for the end of the waveguides')

    r_reached_set_arr = np.append(r_reached_set_arr, r_final_arr)
    r_wvg_start_set_arr = np.append(r_wvg_start_set_arr, r_wvg_start_s_chunk_arr)
    r_wvg_end_set_arr = np.append(r_wvg_end_set_arr, r_wvg_end_s_chunk_arr)
    r_wvg_mid_set_arr = np.append(r_wvg_mid_set_arr, r_wvg_mid_s_chunk_arr)

r_reached_set_arr = r_reached_set_arr.reshape(np.int(r_reached_set_arr.shape[0]/3), 3)
r_wvg_start_set_arr = r_wvg_start_set_arr.reshape(np.int(r_wvg_start_set_arr.shape[0]/3), 3)
r_wvg_end_set_arr = r_wvg_end_set_arr.reshape(np.int(r_wvg_end_set_arr.shape[0]/3), 3)
r_wvg_mid_set_arr = r_wvg_mid_set_arr.reshape(np.int(r_wvg_mid_set_arr.shape[0]/3), 3)
#%%
# Auxiliary code for plotting xz-crossection of the experiment + the trajectories of the atoms from the last simulation chunk that passed though the whole experiment
fig = plt.figure()
ax = fig.add_subplot(111)

for sim_counter in range(sim_chunk_size):
    #if not collision_Q_arr[sim_counter] and reached_end_Q_arr[sim_counter]:
    r_pos_arr = r_record_set_arr[sim_counter, 0:int(steps_count_arr[sim_counter]+1)]
    #ax.plot(r_pos_arr[:,0], r_pos_arr[:,1], r_pos_arr[:,2])
    ax.plot(r_pos_arr[:,2], r_pos_arr[:,0])
    #print(sim_counter)

z_loc = 0

cyl_rad = r_PD_body_inner
cyl_length = PD_body_to_plates_dist
ax = cylinder_2D_plot(ax, z_loc, cyl_rad, cyl_length)
z_loc = z_loc + cyl_length

cyl_rad = r_PD_inner
cyl_length = PD_plates_length
ax = cylinder_2D_plot(ax, z_loc, cyl_rad, cyl_length)
z_loc = z_loc + cyl_length

cyl_rad = r_PD_body_inner
cyl_length = PD_body_to_plates_dist
ax = cylinder_2D_plot(ax, z_loc, cyl_rad, cyl_length)
z_loc = z_loc + cyl_length

cyl_rad = r_CF_6_bore
cyl_length = cf_6_length
ax = cylinder_2D_plot(ax, z_loc, cyl_rad, cyl_length)
z_loc = z_loc + cyl_length

cyl_rad = r_PD_to_to_middle
cyl_length = dist_PD_end_to_BDLA_middle_part
ax = cylinder_2D_plot(ax, z_loc, cyl_rad, cyl_length)
z_loc = z_loc + cyl_length

cyl_rad = r_BDLA_middle
cyl_length = dist_BDLA_middle_to_aperture
ax = cylinder_2D_plot(ax, z_loc, cyl_rad, cyl_length)
z_loc = z_loc + cyl_length

# BDLA aperture
z_arr = np.array([z_loc, z_loc])
x_arr = np.array([-cyl_rad, -r_BDLA_aperture])
ax.plot(z_arr, x_arr, color='black')
z_arr = np.array([z_loc, z_loc])
x_arr = np.array([r_BDLA_aperture, cyl_rad])
ax.plot(z_arr, x_arr, color='black')

cyl_rad = r_inner_quench_cav
cyl_length = quench_cav_stack_length + dist_BDLA_to_pre_quench_cav_stack
ax = cylinder_2D_plot(ax, z_loc, cyl_rad, cyl_length)
z_loc = z_loc + cyl_length

cyl_rad = r_inner_wvg
cyl_length = dist_quench_cav_stack_to_wvg + wvg_length
ax = cylinder_2D_plot(ax, z_loc, cyl_rad, cyl_length)
z_loc = z_loc + cyl_length

cyl_rad = r_inner_quench_cav
cyl_length = dist_quench_cav_stack_to_wvg + quench_cav_stack_length
ax = cylinder_2D_plot(ax, z_loc, cyl_rad, cyl_length)
z_loc = z_loc + cyl_length

cyl_rad = r_inner_det_shield_tube
cyl_length = dist_quench_cav_to_det + det_shield_tube_length + shield_tube_to_GND_ring
ax = cylinder_2D_plot(ax, z_loc, cyl_rad, cyl_length)
z_loc = z_loc + cyl_length

# Detector aperture = grounded ring
z_arr = np.array([z_loc, z_loc])
x_arr = np.array([-cyl_rad, -r_det_ring])
ax.plot(z_arr, x_arr, color='black')
z_arr = np.array([z_loc, z_loc])
x_arr = np.array([r_det_ring, cyl_rad])
ax.plot(z_arr, x_arr, color='black')


#ax.set_ylim(-0.0022, 0.0022)
#ax.set_xlim(0.9, 1.00)
#ax.set_xlim(1.86, 1.9)
fig.set_size_inches(15,12)
plt.show()
#%%
# saving_folder_loc = 'E:/Google Drive/Research/Lamb shift measurement/Data/MC_simulation_beam_shape'
# os.chdir(saving_folder_loc)
#
# sim_file_name = 'beam_shape_n' + str(sim_num_chunks) + 'r' + str(int(r_phi_angle_range*1E6)) + 'v' + str(int(v_range_theta_angle*1E6)) + '.txt'
#
# np.savetxt(sim_file_name, r_reached_set_arr, delimiter=',')
#%%
def propagate_circular_pattern(rot_angle, r_set_arr):
    ''' For this simulation we know that the distribution of positions about z-axis has to cylindrically symmetric. This is true, because all of the experiment components are forced to be axial with each other. Thus even though we might run the simulation only for small range of r_theta angles, we can propagate this pattern along all 2pi radians by rotating the observed pattern in the rot_angle = r_theta_max increments.
    Inputs:
    :rot_angle: This is the angle [rad] by which to incremently rotate the available set of positions in xy-plane. This angle has to be equal to the r_theta_max specified in the simulation.
    :r_set_arr: set of positions of the atoms that had their initial positions specified in the [0, r_theta_max] range.

    Outputs:
    :r_full_arr: Array of positions for the atoms that had their initial positions specified in the [0, 2*np.pi] range.
    '''

    # Number of rotations to perform
    n_rot = int((2*np.pi-rot_angle)/rot_angle)

    # Array of angles through which to rotate the atoms' positions to cover full [0, 2pi] range.
    rot_angle_arr = np.linspace(rot_angle, 2*np.pi-rot_angle, n_rot)

    # Array to store full circulat pattern
    r_full_arr = np.zeros((n_rot+1, r_set_arr.shape[0], 2))
    r_full_arr[0] = r_set_arr[:, [0, 1]]

    i = 1
    for rot_angle_to_use in rot_angle_arr:
        # Rotation matrix in 2D
        rot_matrix = np.array([[np.cos(rot_angle_to_use), -np.sin(rot_angle_to_use)], [np.sin(rot_angle_to_use), np.cos(rot_angle_to_use)]])

        # Rotate the available position vectors
        x_y_arr = r_set_arr[:,[0,1]]
        x_y_rot_arr = np.array([rot_matrix.dot(x_y_vec) for x_y_vec in x_y_arr])

        r_full_arr[i] = x_y_rot_arr
        i = i + 1

    r_full_arr = r_full_arr.reshape(r_full_arr.shape[0]*r_full_arr.shape[1], r_full_arr.shape[2])
    return r_full_arr

def calc_rms_rad(use_every_nth_element, r_arr, bins):
    ''' For the positions of the atoms in the specified range of z-values we calculate the rms radius. For this we need the surface probability density, which can be approximated by a 2d normalized histogram. After that the rms_radius can be calculated as the integral (= sum) of (r^2 * sigma(x, y) dA), where dA = dx*dy, r^2 = x^2 + y^2, sigma(x,y) = normalized surface prob. density. The derivation is shown on p.78 of Lab Notes 4 written on October 26, 2018.

    Inputs:
    :use_every_nth_element: (int) The np.histogram2d complains that it cannot handle large arrays. That is why we will use only every nth element for the histogram.
    :r_arr: np.array() (floats) of positions
    :bins: number of bins along x- and y-directions to use for the histogram.
    '''

    sigma, xedges, yedges = np.histogram2d(r_arr[::use_every_nth_element, 0], r_arr[::use_every_nth_element, 1], bins=bins, density=True)

    # Get the positions of the middle of the bins
    x0_arr = xedges[:-1] + np.diff(xedges)/2
    y0_arr = yedges[:-1] + np.diff(yedges)/2

    xx, yy = np.meshgrid(x0_arr, y0_arr)

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    # RMS radius.
    r_rms = np.sqrt(np.sum(sigma * (yy**2 + xx**2) * dx * dy))

    # Make sure that the total probability density is 1
    prob_dens_integral = np.sum(sigma * dx * dy)

    return r_rms, prob_dens_integral

#%%
use_every_nth_element = 5
bins = 200
#%%
''' At the end of the path
'''

# 2D histogram of the atoms
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist2d(r_reached_set_arr[:,0], r_reached_set_arr[:,1], bins=bins)

ax.set_xlim(-0.005, 0.005)
ax.set_ylim(-0.005, 0.005)

ax.set_aspect('equal')
fig.set_size_inches(15, 10)

plt.show()
#%%
rot_angle = r_phi_angle_range
r_full_arr = propagate_circular_pattern(rot_angle, r_reached_set_arr)
#%%
# 2D full histogram of the atoms
fig = plt.figure()
fig.set_size_inches(15, 10)
ax = fig.add_subplot(111)
ax.hist2d(r_full_arr[::use_every_nth_element, 0], r_full_arr[::use_every_nth_element, 1], bins=bins)

ax.set_xlim(-0.005, 0.005)
ax.set_ylim(-0.005, 0.005)

ax.set_aspect('equal')

plt.show()
#%%
r_rms_end, integ_prob_dens = calc_rms_rad(use_every_nth_element, r_full_arr, bins)
#%%
''' Start of the waveguides
'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist2d(r_wvg_start_set_arr[:,0], r_wvg_start_set_arr[:,1], bins=bins)

ax.set_xlim(-0.005, 0.005)
ax.set_ylim(-0.005, 0.005)

ax.set_aspect('equal')
fig.set_size_inches(15, 10)

plt.show()
#%%
rot_angle = r_phi_angle_range
r_full_arr = propagate_circular_pattern(rot_angle, r_wvg_start_set_arr)
#%%
# 2D full histogram of the atoms
fig = plt.figure()
fig.set_size_inches(15, 10)
ax = fig.add_subplot(111)
ax.hist2d(r_full_arr[::use_every_nth_element, 0], r_full_arr[::use_every_nth_element, 1], bins=bins)

ax.set_xlim(-0.005, 0.005)
ax.set_ylim(-0.005, 0.005)

ax.set_aspect('equal')

plt.show()
#%%
r_rms_start, integ_prob_dens = calc_rms_rad(use_every_nth_element, r_full_arr, bins)
#%%
''' Middle of the waveguides
'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist2d(r_wvg_mid_set_arr[:,0], r_wvg_mid_set_arr[:,1], bins=bins)

ax.set_xlim(-0.005, 0.005)
ax.set_ylim(-0.005, 0.005)

ax.set_aspect('equal')
fig.set_size_inches(15, 10)

plt.show()
#%%
rot_angle = r_phi_angle_range
r_full_arr = propagate_circular_pattern(rot_angle, r_wvg_mid_set_arr)
#%%
# 2D full histogram of the atoms
fig = plt.figure()
fig.set_size_inches(15, 10)
ax = fig.add_subplot(111)
ax.hist2d(r_full_arr[::use_every_nth_element, 0], r_full_arr[::use_every_nth_element, 1], bins=bins)

ax.set_xlim(-0.005, 0.005)
ax.set_ylim(-0.005, 0.005)

ax.set_aspect('equal')

plt.show()
#%%
r_rms_mid, integ_prob_dens = calc_rms_rad(use_every_nth_element, r_full_arr, bins)
#%%
''' End of the waveguides
'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist2d(r_wvg_end_set_arr[:,0], r_wvg_end_set_arr[:,1], bins=bins)

ax.set_xlim(-0.005, 0.005)
ax.set_ylim(-0.005, 0.005)

ax.set_aspect('equal')
fig.set_size_inches(15, 10)

plt.show()
#%%
rot_angle = r_phi_angle_range
r_full_arr = propagate_circular_pattern(rot_angle, r_wvg_end_set_arr)
# 2D full histogram of the atoms
fig = plt.figure()
fig.set_size_inches(15, 10)
ax = fig.add_subplot(111)
ax.hist2d(r_full_arr[::use_every_nth_element, 0], r_full_arr[::use_every_nth_element, 1], bins=bins)

ax.set_xlim(-0.005, 0.005)
ax.set_ylim(-0.005, 0.005)

ax.set_aspect('equal')

plt.show()
#%%
r_rms_end, integ_prob_dens = calc_rms_rad(use_every_nth_element, r_full_arr, bins)
#%%
# RMS radius [mm]
r_rms_avg = (r_rms_start + r_rms_mid + r_rms_end) / 3 * 1E3
#%%
r_rms_avg
