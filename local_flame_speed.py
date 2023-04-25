# -*- coding: utf-8 -*-
'''
Created on Mon Mar 20 23:31:58 2023

@author: luuka
'''

#%% IMPORT PACKAGES
import os
import numpy as np
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
from contour_properties import *
from shared_parameters import parameters
from scipy.stats import gaussian_kde
import statsmodels.api as sm

#%% CLOSE ALL FIGURES
plt.close('all')

#%% SET CONSTANTS
sep = os.path.sep # OS separator

#%% FIGURE SETTINGS
# # Use default parameters in plots
# mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['pcolor.shading'] = 'auto'
# Use Latex font in plots
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 14.0})


fig_scale = 1
default_fig_dim = mpl.rcParams['figure.figsize']

# viridis = mpl.colormaps['viridis']
# viridis_20 = mpl.colormaps.get_cmap('viridis', 20)
# viridis_40 = mpl.colormaps.get_cmap('viridis', 40)

jet = mpl.colormaps['jet']
# jet_20 = mpl.colormaps.get_cmap('jet', 20)
# jet_40 = mpl.colormaps.get_cmap('jet', 40)

# blues = mpl.colormaps['Blues']
# blues_20 = mpl.colormaps.get_cmap('Blues', 20)
# blues_40 = mpl.cm.get_cmap('Blues', 40)

# reds = mpl.colormaps['Reds']
# reds_20 = mpl.colormaps.get_cmap('Reds', 20)
# reds_40 = mpl.colormaps.get_cmap('Reds', 40)

# Google hex-colors
google_red = '#db3236'
google_green = '#3cba54'
google_blue = '#4885ed'
google_yellow = '#f4c20d'


#%% IMPORT FLAME PARAMETERS FROM SHARED PARAMETERS FILE
flame, flame_nr, frame_nr, normalized, D, U_bulk, scale, piv_dir, \
        n_windows_x_raw, n_windows_y_raw, window_size_x_raw, window_size_y_raw, \
        x_left_raw, x_right_raw, y_bottom_raw, y_top_raw, extent_raw, \
        n_windows_x, n_windows_y, extent_piv, toggle_plot = parameters(normalized=False)
        
#%% MAIN FUNCTIONS
def read_piv_data(image_nr):
    
    # Obtain contour number
    contour_nr = image_nr - 1
    
    # Transient file name and scaling parameters from headers of file
    xyuv_file = piv_dir + 'B%.4d' % image_nr + '.txt'
    
    piv_file = open(xyuv_file, 'r')
    scaling_info = piv_file.readline()
    piv_file.close()
    scaling_info_array = scaling_info.split()
    n_windows_x = int(scaling_info_array[7])
    n_windows_y = int(scaling_info_array[6])
    
    # Read velocity data
    xyuv = np.genfromtxt(xyuv_file, delimiter=',')

    x = xyuv[:,0].reshape(n_windows_y, n_windows_x)
    y = xyuv[:,1].reshape(n_windows_y, n_windows_x)
    u = xyuv[:,2].reshape(n_windows_y, n_windows_x) # *-1 because -> inverse x-axis
    v = xyuv[:,3].reshape(n_windows_y, n_windows_x)
    velocity_abs = np.sqrt(u**2 + v**2)
    
    return contour_nr, n_windows_x, n_windows_y, x/D, y/D, u/U_bulk, v/U_bulk, velocity_abs/U_bulk


def contour_correction(contour_nr, frame_nr=0):
    
    segmented_contour =  flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
    
    segmented_contour_x = segmented_contour[:,0,0]
    segmented_contour_y = segmented_contour[:,0,1]
    
    # x and y coordinates of the discretized (segmented) flame front 
    contour_x_correction = segmented_contour_x*window_size_x_raw + x_left_raw
    contour_y_correction = segmented_contour_y*window_size_y_raw + y_top_raw
    
    # Non-dimensionalize coordinates by pipe diameter
    contour_x_correction /= D
    contour_y_correction /= D
    
    contour_x_correction_array = np.array(contour_x_correction)
    contour_y_correction_array = np.array(contour_y_correction)
    
    # Combine the x and y coordinates into a single array of shape (n_coords, 2)
    contour_correction_coords = np.array([contour_x_correction_array, contour_y_correction_array]).T

    # Create a new array of shape (n_coords, 1, 2) and assign the coordinates to it
    contour_correction = np.zeros((len(contour_x_correction_array), 1, 2))
    contour_correction[:, 0, :] = contour_correction_coords
    
    return contour_correction

def determine_local_flame_speed(selected_segments_2, final_segments, colors, ax=None):
    
    # Please note that:
    
    # - final_segment = [selected_segment, x_intersect_ref, y_intersect_ref, x_intersect, y_intersect, V_n, S_d]
    # - selected_segment = [x_A, y_A, x_B, y_B, x_loc[side], y_loc[side], V_nx, V_ny, V_n[side], V_tx, V_ty, V_t[side], V_x, V_y]
    
    n_digits = 5
    
    for t, selected_segments in enumerate(selected_segments_2):
        
        for selected_segment in selected_segments:
            
            V_nx = selected_segment[0][6]
            V_ny = selected_segment[0][7]
            V_n = selected_segment[0][8]
            
            x_intersect_ref = selected_segment[1]
            y_intersect_ref = selected_segment[2]
            x_intersect = selected_segment[3]
            y_intersect = selected_segment[4]
            
            S_d = selected_segment[8]
            S_d_dx, S_d_dy = x_intersect-x_intersect_ref, y_intersect-y_intersect_ref
            S_d_angle = np.arctan2(S_d_dy, S_d_dx)
            
            V_n_angle = np.arctan2(V_ny, V_nx)
            
            S_d = np.round(S_d, n_digits)
            S_d_dx = np.round(S_d_dx, n_digits)
            S_d_dy = np.round(S_d_dy, n_digits)
            S_d_angle = np.round(S_d_angle, n_digits)
            
            V_n = np.round(V_n, n_digits)
            V_n_angle = np.round(V_n_angle, n_digits)
            
            # print(V_n_angle, S_d_angle)
            
            if (V_n_angle >= 0) != (S_d_angle >= 0) or V_n_angle != S_d_angle:
                
                S_f = V_n + S_d
                    
            elif V_n_angle == S_d_angle:
                
                S_f = V_n - S_d
            
            else:
                
                S_f = np.nan
                print("Local flame speed could not be detected, check this image!")
                
                
            selected_segment.append(S_f)
            
            final_segments.append(selected_segment)
            
            if toggle_plot:
                plot_local_flame_speed(ax, selected_segment, colors[t]) 
    
    return final_segments
    
def local_flame_speed_from_double_image_single_frame(image_nr, n_time_steps=1, n_nearest_coords=1, threshold_angle=30):
    
    # Set color map
    colormap = jet(np.linspace(0, 1, 2))
    
    # Create color iterator for plotting the data
    colormap_iter_time = iter(colormap)
    
    image_nr_t0 = image_nr
    image_nr_t1 = image_nr_t0 + n_time_steps
    
    image_nrs = [image_nr_t0, image_nr_t1]
    
    # Time interval between two contours
    dt = n_time_steps*(1/flame.image_rate)
    
    final_segments = []
    
    selected_segments_1 = []
    selected_segments_2 = []
    
    colors = []
    
    for i, image_nr in enumerate(image_nrs):
        
        # Read PIV data
        contour_nr_ref, n_windows_x, n_windows_y, x_ref, y_ref, u_ref, v_ref, velocity_abs_ref = read_piv_data(image_nrs[0])
        contour_nr, n_windows_x, n_windows_y, x, y, u, v, velocity_abs = read_piv_data(image_nrs[1])
        
        # Contour correction RAW --> (non-dimensionalized) WORLD
        contour_corrected_ref = contour_correction(contour_nr_ref)
        contour_corrected = contour_correction(contour_nr)
        
        # First selection round: Checks if a velocity vector crosses a contour segment at the reference time step. 
        # The amount of velocity vectors considered for selection is set with n_nearest_coords.
        selected_segments_1_ref = first_segment_selection_procedure(contour_corrected_ref, n_windows_x, n_windows_y, x_ref, y_ref, u_ref, v_ref, n_nearest_coords)
        selected_segments_1.append(selected_segments_1_ref)
        
        # Second selection round: Checks if the normal component of the velocity vector found at the reference time step crosses 
        # the segment at the reference time step and a segment in the 'other' time step. Another restriction is that between to selected 
        # segments (segments at t0 and t1) may not be greater than threshold_angle (in degrees)
        selected_segments_2_ref = second_segment_selection_procedure(contour_corrected, selected_segments_1_ref, dt, threshold_angle)
        selected_segments_2.append(selected_segments_2_ref)
        
        # print(y[2])
        # print(len(selected_segments_1_ref))
        # print(len(selected_segments_2_ref))
        
        if toggle_plot:
            
            if i == 0:
                
                # Create figure for velocity field + frame front contour
                fig, ax = plt.subplots()
                
                ax.set_title('Flame ' + str(flame_nr) + ': ' + '$\phi$=' + str(flame.phi) + ', $H_{2}\%$=' + str(flame.H2_percentage)+ '\n' +
                             '$D_{in}$=' + str(flame.D_in) + ' mm, $Re_{D_{in}}$=' + str(flame.Re_D) + '\n' + 
                             'Image: ' + str(image_nr_t0) + ' - ' + str(image_nr_t1) + ', Frame:' + str(frame_nr))
                
                # for element in selected_segments_1_ref:
                    # ax.plot(element[4], element[5], marker='s', color='m')
                # print('---------------------')
                
                if normalized:
                    
                    ax.set_xlabel(r'$r/D$')
                    ax.set_ylabel(r'$x/D$')
                    
                else:
                    
                    ax.set_xlabel(r'$r$ [mm]')
                    ax.set_ylabel(r'$x$ [mm]')
                    
                    # x_left, x_right = 5, 14
                    # y_left, y_right = 3, 9
                    # zoom = [x_left, x_right, y_left, y_right]
                    # ax.set_xlim(np.array([zoom[0], zoom[1]]))
                    # ax.set_ylim(np.array([zoom[2], zoom[3]]))
                
                # Plot velocity vectors
                # plot_velocity_vectors(fig, ax, x_ref, y_ref, u_ref, v_ref, scale)
                
                # Plot velocity field
                quantity = velocity_abs_ref
                
                # Choose 'imshow' or 'pcolor' by uncommenting the correct line
                # plot_velocity_field_imshow(fig, ax, x_ref, y_ref, quantity)
                plot_velocity_field_pcolor(fig, ax, x_ref, y_ref, quantity)
                plot_velocity_vectors(fig, ax, x_ref, y_ref, u_ref, v_ref, scale)
    
            # Create timestamp for plot
            timestamp = ((image_nrs[0]-image_nr_t0)/flame.image_rate)*1e3
            
            # Plot flame front contour
            c = next(colormap_iter_time)
            colors.append(c)
            ax = plot_contour_single_color(ax, contour_corrected_ref, timestamp, c)
            
            # Turn on legend
            ax.legend()
        
            # Tighten figure
            fig.tight_layout()
        
        # Important: This operation reverses the image_nrs, so that the reference time step changes from image_t0 to image_t1
        image_nrs.reverse()
    
    selected_segments_2 = [selected_segments_2[0]]
    final_segments = determine_local_flame_speed(selected_segments_2, final_segments, colors, ax=ax if toggle_plot else None)
        
    return final_segments


def local_flame_speed_from_single_image_double_frame(image_nr, n_nearest_coords=1, threshold_angle=30):
    
    # Set color map
    colormap = jet(np.linspace(0, 1, 2))
    
    # Create color iterator for plotting the data
    colormap_iter_time = iter(colormap)
    
    frame0 = 0
    frame1 = 1
    
    frame_nrs = [frame0, frame1]
    
    # Time interval between two contours
    dt = flame.dt*1e-6
    
    final_segments = []
    
    selected_segments_1 = []
    selected_segments_2 = []
    
    colors = []
    
    for i, frame_nr in enumerate(frame_nrs):
        
        # Read PIV data
        contour_nr, n_windows_x, n_windows_y, x, y, u, v, velocity_abs = read_piv_data(image_nr)
        
        # Contour correction RAW --> (non-dimensionalized) WORLD
        contour_corrected_ref = contour_correction(contour_nr, frame_nrs[0])
        contour_corrected = contour_correction(contour_nr, frame_nrs[1])
        
        # First selection round: Checks if a velocity vector crosses a contour segment at the reference time step. 
        # The amount of velocity vectors considered for selection is set with n_nearest_coords.
        selected_segments_1_ref = first_segment_selection_procedure(contour_corrected_ref, n_windows_x, n_windows_y, x, y, u, v, n_nearest_coords)
        selected_segments_1.append(selected_segments_1_ref)
        
        # Second selection round: Checks if the normal component of the velocity vector found at the reference time step crosses 
        # the segment at the reference time step and a segment in the 'other' time step. Another restriction is that between to selected 
        # segments (segments at t0 and t1) may not be greater than threshold_angle (in degrees)
        selected_segments_2_ref= second_segment_selection_procedure(contour_corrected, selected_segments_1_ref, dt, threshold_angle)
        selected_segments_2.append(selected_segments_2_ref)
        
        if toggle_plot:
            
            if i == 0:
                
                # Create figure for velocity field + frame front contour
                fig, ax = plt.subplots()
                
                # ax.set_title('Flame ' + str(flame_nr) + ': ' + '$\phi$=' + str(flame.phi) + ', $H_{2}\%$=' + str(flame.H2_percentage)+ '\n' +
                #              '$D_{in}$=' + str(flame.D_in) + ' mm, $Re_{D_{in}}$=' + str(flame.Re_D) + '\n' + 
                #              'Image:' + str(image_nr) + ', Frame:' + str(frame0) + ' - ' + str(frame1))
                
                if normalized:
                    
                    ax.set_xlabel(r'$r/D$')
                    ax.set_ylabel(r'$x/D$')
                    
                else:
                    
                    ax.set_xlabel(r'$r$ [mm]')
                    ax.set_ylabel(r'$x$ [mm]')
                    
                    # x_left, x_right = 6, 8
                    # y_left, y_right = 6.75, 8.5
                    # zoom = [x_left, x_right, y_left, y_right]
                    # ax.set_xlim(np.array([zoom[0], zoom[1]]))
                    # ax.set_ylim(np.array([zoom[2], zoom[3]]))
                
                # Plot velocity vectors
                # plot_velocity_vectors(fig, ax, x, y, u, v, scale)
                
                # Plot velocity field
                quantity = velocity_abs
                
                # Choose 'imshow' or 'pcolor' by uncommenting the correct line
                # plot_velocity_field_imshow(fig, ax, x, y, quantity)
                plot_velocity_field_pcolor(fig, ax, x, y, quantity)
                # plot_velocity_vectors(fig, ax, x, y, u, v, scale)
    
            # Create timestamp for plot
            timestamp = dt*(frame_nrs[0]-frame0)*1e3
            # timestamp = frame_nrs[0]
            
            
            # Plot flame front contour t0
            c = next(colormap_iter_time)
            colors.append(c)
            ax = plot_contour_single_color(ax, contour_corrected_ref, timestamp, c)
            
            # Turn on legend
            ax.legend(loc='upper right')
                    
            # Tighten figure
            fig.tight_layout()
        
        # Important: This operation reverses the frame_nrs, so that the reference time step changes from frame0 to frame1
        frame_nrs.reverse()
    
    selected_segments_2 = [selected_segments_2[0]]
    final_segments = determine_local_flame_speed(selected_segments_2, final_segments, colors, ax=ax if toggle_plot else None)
    
    return final_segments

    
def local_flame_speed_from_time_resolved_single_frame(image_nr, n_nearest_coords=1, threshold_angle=30):
    
    # Set color map
    colormap = jet(np.linspace(0, 1, 2))
    
    # Create color iterator for plotting the data
    colormap_iter_time = iter(colormap)
    
    image_nr_t0 = image_nr
    image_nr_t1 = image_nr_t0 + 1
    
    # image_nrs = [image_nr_t0, image_nr_t1]
    
    # Time interval between two contours
    dt = (1/flame.image_rate)
    
    final_segments = []
    
    selected_segments_1 = []
    selected_segments_2 = []
    
    colors = []
    
    # Read PIV data
    contour_nr_t0, n_windows_x, n_windows_y, x_ref, y_ref, u_ref, v_ref, velocity_abs_ref = read_piv_data(image_nr_t0)
    contour_nr_t1, n_windows_x, n_windows_y, x, y, u, v, velocity_abs = read_piv_data(image_nr_t1)
    
    contour_nrs = [contour_nr_t0, contour_nr_t1]
    
    for i, contour_nr in enumerate(contour_nrs):
        
        # Contour correction RAW --> (non-dimensionalized) WORLD
        contour_corrected_ref = contour_correction(contour_nrs[0])
        contour_corrected = contour_correction(contour_nrs[1])
        
        # First selection round: Checks if a velocity vector crosses a contour segment at the reference time step. 
        # The amount of velocity vectors considered for selection is set with n_nearest_coords.
        selected_segments_1_ref = first_segment_selection_procedure(contour_corrected_ref, n_windows_x, n_windows_y, x_ref, y_ref, u_ref, v_ref, n_nearest_coords)
        selected_segments_1.append(selected_segments_1_ref)
        
        # Second selection round: Checks if the normal component of the velocity vector found at the reference time step crosses 
        # the segment at the reference time step and a segment in the 'other' time step. Another restriction is that between to selected 
        # segments (segments at t0 and t1) may not be greater than threshold_angle (in degrees)
        selected_segments_2_ref= second_segment_selection_procedure(contour_corrected, selected_segments_1_ref, dt, threshold_angle)
        selected_segments_2.append(selected_segments_2_ref)
        
        if toggle_plot:
            
            if i == 0:
                
                # Create figure for velocity field + frame front contour
                fig, ax = plt.subplots()
                
                ax.set_title('Flame ' + str(flame_nr) + ': ' + '$\phi$=' + str(flame.phi) + ', $H_{2}\%$=' + str(flame.H2_percentage)+ '\n' +
                             '$D_{in}$=' + str(flame.D_in) + ' mm, $Re_{D_{in}}$=' + str(flame.Re_D) + '\n' + 
                             'Image: ' + str(image_nr_t0) + ' - ' + str(image_nr_t1) + ', time-resolved')
                
                if normalized:
                    
                    ax.set_xlabel(r'$r/D$')
                    ax.set_ylabel(r'$x/D$')
                    
                else:
                    
                    ax.set_xlabel(r'$r$ [mm]')
                    ax.set_ylabel(r'$x$ [mm]')
                
                # Plot velocity vectors
                plot_velocity_vectors(fig, ax, x_ref, y_ref, u_ref, v_ref, scale)
                
                # Plot velocity field
                quantity = velocity_abs_ref
                
                # Choose 'imshow' or 'pcolor' by uncommenting the correct line
                plot_velocity_field_imshow(fig, ax, x_ref, y_ref, quantity)
                # plot_velocity_field_pcolor(fig, ax, x_t0, y_t0, quantity)
    
            # Create timestamp for plot
            timestamp = ((contour_nrs[0]-contour_nr_t0)/flame.image_rate)*1e3
            
            # Plot flame front contour
            c = next(colormap_iter_time)
            colors.append(c)
            ax = plot_contour_single_color(ax, contour_corrected_ref, timestamp, c)
            
            # Turn on legend
            ax.legend()
            
            # Tighten figure
            fig.tight_layout()
        
        # Important: This operation reverses the image_nrs, so that the reference time step changes from image_t0 to image_t1
        contour_nrs.reverse()
    
    selected_segments_2 = [selected_segments_2[0]]
    final_segments = determine_local_flame_speed(selected_segments_2, final_segments, colors, ax=ax if toggle_plot else None)
        
    return final_segments

def first_segment_selection_procedure(contour_correction, n_windows_x, n_windows_y, x, y, u, v, n_nearest_coords):
    
    contour_x = contour_correction[:,0,0]
    contour_y = contour_correction[:,0,1]
    
    # Create closed contour to define coordinates (of the interrogation windows) in unburned and burned region 
    contour_open = zip(contour_x, contour_y)
    contour_open = list(contour_open)
    contour_open.append((contour_x[0], contour_y[0]))
    contour_closed_path = mpl.path.Path(np.array(contour_open))
    
    selected_segments_1 = []
    
    for i_segment in range(0, len(contour_x) - 1):
        
        # Initialize velocity data related to a selected segment (index 0: unburnt, index 1: burnt)
        x_loc = [0, 0]
        y_loc = [0, 0]
        V_x = [0, 0]
        V_y = [0, 0]
        V_n = [0, 0]
        V_t = [0, 0]
        
        coords_u = []
        coords_b = []
        
        nearest_coords = []
        
        x_A, y_A, x_B, y_B  = contour_x[i_segment], contour_y[i_segment], contour_x[i_segment+1], contour_y[i_segment+1]
        x_mid, y_mid = (x_A + x_B)/2, (y_A + y_B)/2
        dx, dy = x_B-x_A, y_B-y_A
        
        # Segment length and angle
        # segment_angle = np.arctan2(dy, dx)
        L, segment_angle = segment_properties(dy, dx)
                
        for j in range(n_windows_y):
            
            for i in range(n_windows_x):
                
                x_piv_ij = x[j][i]
                y_piv_ij = y[j][i]
                
                distance_to_segment = np.sqrt((x_mid - x_piv_ij)**2 + (y_mid - y_piv_ij)**2)
                
                distance_u_threshold_lower = 0 # units: mm 
                distance_u_threshold_upper = 5*L # units: mm 
                distance_b_threshold_lower = 0 # units: mm
                distance_b_threshold_upper = 5*L # units: mm 
                    
                if contour_closed_path.contains_point((x_piv_ij, y_piv_ij)):
                    
                    if distance_u_threshold_lower <= distance_to_segment <= distance_u_threshold_upper:
                        
                        coord_u = [j, i, distance_to_segment, x_piv_ij, y_piv_ij, u[j][i], v[j][i]]
                        coords_u.append(coord_u)

                else:
                    
                    if distance_b_threshold_lower/D < distance_to_segment < distance_b_threshold_upper/D:
                        
                        coord_b = [j, i, distance_to_segment, x_piv_ij, y_piv_ij, u[j][i], v[j][i]]
                        coords_b.append(coord_b)
        
        # Sort coordinates to get candidate coordinates based on distance closest to segment
        coords_u.sort(key = lambda i: i[2])
        coords_b.sort(key = lambda i: i[2])
        
        candidate_coords_u = coords_u[0:n_nearest_coords]
        candidate_coords_b = coords_b[0:n_nearest_coords]
        # both_sides_candidate_coords = [candidate_coords_u, candidate_coords_b]
        both_sides_candidate_coords = [candidate_coords_u]
        
        # Check if velocity vector normal of the reference time step intersects the segment itself 
        for side, candidate_coords in enumerate(both_sides_candidate_coords):
            
            nearest_coords = []
            
            for candidate_coord in candidate_coords:
        
                    x_piv = candidate_coord[3]
                    y_piv = candidate_coord[4]
                    Vx = candidate_coord[5]
                    Vy = candidate_coord[6]
                    
                    if (Vy*dx - Vx*dy) == 0:
                        
                        k = 0
                        
                    else:
                        
                        k = (Vx*(y_A-y_piv) - Vy*(x_A-x_piv))/(Vy*dx - Vx*dy)
                    
                    if Vx == 0:
                        
                        l = 0
                        
                    else:
                        
                        l = (k*dx + x_A - x_piv)/Vx
                    
                    x_intersect = x_A + k*dx
                    y_intersect = y_A + k*dy
                    
                    distance_A = np.sqrt((x_intersect - x_A)**2 + (y_intersect - y_A)**2)
                    distance_B = np.sqrt((x_intersect - x_B)**2 + (y_intersect - y_B)**2)
                    
                    # print(k, l, x_intersect, y_intersect, distance_A, distance_B, L)
                    
                    if distance_A <= L and distance_B <= L:
                        
                        if (side==0 and l>=0) or (side==1 and l<=0):
                            
                            nearest_coords.append(candidate_coord)
            
            # print(nearest_coords, x_mid, y_mid, segment_angle)
            x_avg_list = []
            y_avg_list = []
            Vx_avg_list = []
            Vy_avg_list = []
            
            for j, i, distance_dummy, x_dummy, y_dummy, u_dummy, v_dummy in nearest_coords:
                    
                x_avg_list.append(x_dummy)
                y_avg_list.append(y_dummy)
                Vx_avg_list.append(u_dummy)
                Vy_avg_list.append(v_dummy)
                # print(Vy_avg_list)
        
            if nearest_coords:
                
                x_avg = np.nanmean(x_avg_list)
                y_avg = np.nanmean(y_avg_list)
                Vx_avg = np.nanmean(Vx_avg_list)
                Vy_avg = np.nanmean(Vy_avg_list)
                # print(Vy_avg)
                
            else:
                
                x_avg = np.nanmean(0)
                y_avg = np.nanmean(0)
                Vx_avg = np.nanmean(0)
                Vy_avg = np.nanmean(0)
            
            x_loc[side] = x_avg
            y_loc[side] = y_avg
            V_x[side] = Vx_avg
            V_y[side] = Vy_avg
            V_t[side] = Vx_avg*np.cos(segment_angle) + Vy_avg*np.sin(segment_angle)
            V_n[side] = -Vx_avg*np.sin(segment_angle) + Vy_avg*np.cos(segment_angle)
        
        if V_n[side] == 0:

            pass
        
        else:
            
            for side in range(len(both_sides_candidate_coords)): 
                
                if side == 0:
                    
                    V_x_select = V_x[side]
                    V_y_select = V_y[side]
                    
                    V_tx_select = V_t[side]*np.cos(segment_angle)
                    V_ty_select = V_t[side]*np.sin(segment_angle)
                    
                    V_nx_select = -V_n[side]*np.sin(segment_angle)
                    V_ny_select = V_n[side]*np.cos(segment_angle)
                    
                    # Segment may not touch the top boundary of the image
                    if (y_A or y_B) > y[0][0]:
                        pass
                    else:
                        selected_segment_1 = [x_A, y_A, x_B, y_B, x_loc[side], y_loc[side], V_nx_select, V_ny_select, np.abs(V_n[side]), V_tx_select, V_ty_select, np.abs(V_t[side]), V_x_select, V_y_select, i_segment, contour_correction]
                    
                        selected_segments_1.append(selected_segment_1)
                    
    return selected_segments_1


def second_segment_selection_procedure(contour_correction, selected_segments_1, dt, threshold_angle):
    
    contour_x = contour_correction[:,0,0]
    contour_y = contour_correction[:,0,1]
    
    selected_segments_2 = []
    
    for i_selected_segment, selected_segment in enumerate(selected_segments_1):
        
        # We select the segment that is closest to a select segment in selected_segments_1
        counter = 0
        # distance_between_segment_ref = 1000
        flame_front_displacement_ref = 1000
        
        # Check if velocity vector normal to the segment of the reference time step intersects the segment itself 
        x_A, y_A, x_B, y_B  = selected_segment[0], selected_segment[1], selected_segment[2], selected_segment[3]
        x_mid_ref, y_mid_ref = (x_A + x_B)/2, (y_A + y_B)/2
        dx, dy = x_B-x_A, y_B-y_A
        
        # Segment length and angle
        # segment_angle_ref = np.arctan2(dy, dx)
        L_ref, segment_angle_ref = segment_properties(dy, dx)
        
        # Velocity data of reference time step
        x_piv = selected_segment[4]
        y_piv = selected_segment[5]
        V_nx = selected_segment[6]
        V_ny = selected_segment[7]
        V_n = selected_segment[8]
        
        if (V_ny*dx - V_nx*dy) == 0:
            
            k = 0
            
        else:
            
            k = (V_nx*(y_A-y_piv) - V_ny*(x_A-x_piv))/(V_ny*dx - V_nx*dy)
            
        if V_nx == 0:
            
            l = 0
            
        else:
            
            l = (k*dx + x_A - x_piv)/V_nx
        
        x_intersect_ref = x_A + k*dx
        y_intersect_ref = y_A + k*dy
        
        distance_A = np.sqrt((x_intersect_ref - x_A)**2 + (y_intersect_ref - y_A)**2)
        distance_B = np.sqrt((x_intersect_ref - x_B)**2 + (y_intersect_ref - y_B)**2)
        
        # Check if velocity vector normal to the segment intersects with the segment of the reference time step
        if distance_A <= L_ref and distance_B <= L_ref:
            
            # Check if velocity vector normal to the segment of the reference time step 
            # intersects a segment in the chosen time step 
            
            for i_segment in range(0, len(contour_x) - 1):
                
                x_A, y_A, x_B, y_B  = contour_x[i_segment], contour_y[i_segment], contour_x[i_segment+1], contour_y[i_segment+1]
                x_mid, y_mid = (x_A + x_B)/2, (y_A + y_B)/2
                dx, dy = x_B-x_A, y_B-y_A
                
                # Segment length and angle
                # segment_angle = np.arctan2(dy, dx)
                L, segment_angle = segment_properties(dy, dx)
                
                if (V_ny*dx - V_nx*dy) == 0:
                    
                    k = 0
                    
                else:
                    
                    k = (V_nx*(y_A-y_piv) - V_ny*(x_A-x_piv))/(V_ny*dx - V_nx*dy)
                
                if V_nx == 0:
                    
                    l = 0
                    
                else:
                    
                    l = (k*dx + x_A - x_piv)/V_nx
                
                x_intersect = x_A + k*dx
                y_intersect = y_A + k*dy
                
                distance_A = np.sqrt((x_intersect - x_A)**2 + (y_intersect - y_A)**2)
                distance_B = np.sqrt((x_intersect - x_B)**2 + (y_intersect - y_B)**2)
                
                # Check if velocity vector normal to segment intersects with the segment of the chosen time step
                if distance_A <= L and distance_B <= L:
                    
                    flame_front_displacement = np.sqrt((x_intersect_ref - x_intersect)**2 + (y_intersect_ref - y_intersect)**2)
                    # distance_between_segment = np.sqrt((x_mid_ref - x_mid)**2 + (y_mid_ref - y_mid)**2)
                    
                    if (flame_front_displacement < flame_front_displacement_ref and (flame_front_displacement < 2*L)):
                        
                        flame_front_displacement_ref = flame_front_displacement
                        
                        if (np.abs(segment_angle - segment_angle_ref)) < np.deg2rad(threshold_angle):
                            
                            S_d = flame_front_displacement*(D*1e-3)/dt
                            
                            S_d /= U_bulk
                                 
                            selected_segment_2 = [selected_segment, x_intersect_ref, y_intersect_ref, x_intersect, y_intersect, i_segment, contour_correction, flame_front_displacement, S_d]
                            
                            if counter < 1:
                                
                                selected_segments_2.append(selected_segment_2)
                                counter += 1
                                
                            else:

                                selected_segments_2[-1] = selected_segment_2
                                   
    return selected_segments_2           

#%% MAIN PLOT FUNCTIONS

def plot_velocity_vectors(fig, ax, X, Y, U, V, scale):
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=scale, ls='solid', fc='k', ec='k')
    
    
def plot_velocity_field_imshow(fig, ax, X, Y, quantity):
    
    ax.imshow(quantity, extent=extent_piv, cmap='viridis')
    
    # Set aspect ratio of plot
    ax.set_aspect('equal')
    
    
def plot_velocity_field_pcolor(fig, ax, X, Y, quantity):
    
    # plt.imshow(quantity, extent=extent_raw, cmap='viridis')
    quantity_plot = ax.pcolor(X, Y, quantity, cmap='viridis')
    
    quantity_plot.set_clim(0, quantity.max())
    # quantity_plot.set_clim(0, 10)
    
    # Set aspect ratio of plot
    ax.set_aspect('equal')
    
    bar = fig.colorbar(quantity_plot)
    bar.set_label(r'$|V|$ [ms$^-1$]') #('$u$/U$_{b}$ [-]')

def plot_contour_single_color(ax, contour, timestamp=0, c='b'):
    
    contour_x, contour_y =  contour[:,0,0], contour[:,0,1]
    
    # # Plot flame front contour
    # label = r'$t_0$ = ' + str(timestamp) + ' $ms$'
    label = r'$t=' + str(timestamp) + '$' + ' $ms$'
    ax.plot(contour_x, contour_y, color=c, marker='o', lw=1, label=label)
    
    # Set aspect ratio of plot 
    ax.set_aspect('equal')
    
    return ax

def plot_segmented_contour_multi_color(ax, contour, timestamp, colormap):
    
    contour_x, contour_y =  contour[:,0,0], contour[:,0,1]
    
    for i in range(0, len(contour_x) - 1):
        
        c = next(colormap_iter_space)

        x_A, y_A, x_B, y_B  = contour_x[i], contour_y[i], contour_x[i+1], contour_y[i+1]
        
        ax.plot((x_A, x_B), (y_A, y_B), color=c, marker='None', linestyle='-', linewidth=2)
                            
    # Set aspect ratio of plot 
    ax.set_aspect('equal')
        
    return ax

def plot_local_flame_speed(ax, final_segment, color):
    
    segment = final_segment[0]
    
    x_piv = segment[4]
    y_piv = segment[5]
    V_nx = segment[6]
    V_ny = segment[7]
    V_n = segment[8]
    # V_tx = segment[9]
    # V_ty = segment[10]
    V_t = segment[11]
    V_x = segment[12]
    V_y = segment[13]
    
    x_intersect_ref = final_segment[1]
    y_intersect_ref = final_segment[2]
    x_intersect = final_segment[3]
    y_intersect = final_segment[4]
    flame_front_displacement = final_segment[7]
    S_d = final_segment[8]
    S_f = final_segment[9]
    
    
    # ax.text(x_intersect+0.2, y_intersect-0.1, r'$\vec{V_{n}}$' , color=color, fontsize=18)
    # ax.text(x_intersect+0.05, y_intersect+0.35, r'$\vec{V}$' , color='k', fontsize=18)
    # ax.text(x_piv, y_piv, np.round(V_n, 3), c='y', fontweight='bold')    
    # ax.text(x_intersect, y_intersect, str(np.round(S_d, 3)), color='k')
    
    ax.text((x_intersect_ref + x_intersect)/2 , (y_intersect_ref + y_intersect)/2, np.round(S_f, 3), color='w')
    
    markersize = 10
    mew = 1
    # ax.plot(x_intersect_ref, y_intersect_ref, c='r', marker='x', ms=markersize, mew=mew)
    # ax.plot(x_intersect, y_intersect, c='r', marker='x', ms=markersize, mew=mew)
    ax.plot(x_piv, y_piv, c='m', marker='o', ms=8)
    
    ax.quiver(x_piv, y_piv, V_nx, V_ny, angles='xy', scale_units='xy', scale=scale, ls='-', fc=color, ec=color, width=0.015, headwidth=4, headlength=6)
    ax.quiver(x_piv, y_piv, V_x, V_y, angles='xy', scale_units='xy', scale=scale, ls='-', fc='k', ec='k', width=0.015, headwidth=4, headlength=6)
    
    # m = V_ny/V_nx
    # x_working_line = range(4, 10)
    # y_working_line = [m * (x_i - x_piv) + y_piv for x_i in x_working_line]
    
    # ax.plot(x_working_line, y_working_line, color=color, ls='--')
    
    # markersize = 10
    # mew = 2
    # ax.plot(x_intersect_ref, y_intersect_ref, c='r', marker='x', ms=markersize, mew=mew)
    # ax.plot(x_intersect, y_intersect, c='r', marker='x', ms=markersize, mew=mew)
    # ax.plot(x_piv, y_piv, c='k', marker='o', ms=8)
    
    # distance = np.sqrt((x_intersect - x_intersect_ref)**2 + (y_intersect - y_intersect_ref)**2)
    
    # dx, dy = x_intersect_ref - x_intersect, y_intersect_ref - y_intersect 
    # alpha = np.arctan(dy/dx)
    # ax.quiver(x_intersect_ref, y_intersect_ref, -distance*np.cos(alpha), -distance*np.sin(alpha), angles='xy', scale_units='xy', scale=1, ls='-', fc='w', ec='w', width=0.015, headwidth=4, headlength=6)
    # ax.text(x_intersect+0.2, y_intersect-0.1, r'$\vec{S_{d}}$' , color='w', fontsize=18)

#%% AXUILIARY PLOTTING FUNCTIONS
def save_animation(camera, filename, interval=500, repeat_delay=1000):
    
    animation = camera.animate(interval = interval, repeat = True, repeat_delay = repeat_delay)
    animation.save(filename + '.gif', writer='pillow')


def save_figure(fig, fig_name):
    
    fig.savefig(fig_name + '.png', dpi=300) 
    
#%% "START OF CODE"
# Before executing code, Python interpreter reads source file and define few special variables/global variables. 
# If the python interpreter is running that module (the source file) as the main program, it sets the special __name__ variable to have a value “__main__”. If this file is being imported from another module, __name__ will be set to the module’s name. Module’s name is available as value to __name__ global variable. 
if __name__ == '__main__':
    
    #%%% SET PARAMETERS OF INTEREST
    
    # Toggle plot
    toggle_plot = True
    
    # Image nr
    image_nr = 360
    
    # Number of steps between two timesteps
    n_time_steps = 1
    
    # Set amount of vector considered
    n_nearest_coords = 1
    
    # Threshold angle between segments of consecutive timesteps
    threshold_angle = 180
    
    # Flame front velocity determined by frame nr 0 between image X and X + n_time_steps
    final_segments = local_flame_speed_from_double_image_single_frame(image_nr, n_time_steps, n_nearest_coords, threshold_angle)
    
    # Flame front velocity determined by frame nr 0 and frame nr 1 of image X
    # final_segments = local_flame_speed_from_single_image_double_frame(image_nr, n_nearest_coords, threshold_angle)
    
    # Flame front velocity determined from time-resolved PIV (single image, single frame)
    # final_segments = local_flame_speed_from_time_resolved_single_frame(image_nr, n_nearest_coords, threshold_angle)
    
    #%%% PLOT CONTOUR 1
    # Create figure for velocity field + frame front contour
    fig, ax = plt.subplots()
    # ax.set_xlabel('$r/D$')
    # ax.set_ylabel('$x/D$')
    
    ax.set_title('Flame ' + str(flame_nr) + ': ' + '$\phi$=' + str(flame.phi) + ', $H_{2}\%$=' + str(flame.H2_percentage)+ '\n' +
                  '$D_{in}$=' + str(flame.D_in) + ' mm, $Re_{D_{in}}$=' + str(flame.Re_D) + '\n' + 
                  'Image \#: ' + str(image_nr))
    
    # Read PIV data
    contour_nr, n_windows_x, n_windows_y, x, y, u, v, velocity_abs = read_piv_data(image_nr)
    
    # Contour correction RAW --> (non-dimensionalized) WORLD [reference]
    contour_corrected = contour_correction(contour_nr)
    
    contour = flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
    plot_contour_single_color(ax, contour_corrected)
    contour_slope_changes = flame.frames[frame_nr].contour_data.slope_changes_of_segmented_contours[contour_nr]

    # Set color map
    colormap = jet(np.linspace(0, 1, len(contour_slope_changes)))

    # Create color iterator for plotting the data
    colormap_iter = iter(colormap)

    for i, slope_change in enumerate(contour_slope_changes):
        
        # This is because slopes_of_segmented_contours was calculated with the origin of the image,
        # which is top left instead of the conventional bottom left
        slope_change = -slope_change
        
        x_coord_text = contour_corrected[i+1,0,0] 
        y_coord_text = contour_corrected[i+1,0,1]
        color = next(colormap_iter)
        ax.text(x_coord_text, y_coord_text, f"{np.round(slope_change, 3)}", color='k')

        # Calculate tortuosity
        tau = np.sum(np.abs(contour_slope_changes))

        # Calculate Acc
        Acc = np.cumsum(contour_slope_changes)

        # Add tau and Acc to labels
        ax.text(0.05, 0.95, f"Tau: {np.round(tau, 3)}", transform=ax.transAxes)
        ax.text(0.05, 0.9, f"Acc: {np.round(Acc[-1], 3)}", transform=ax.transAxes)
    
    
    
    
    
    
    #         # p1 = [contour_corrected[i,0,0], contour_corrected[i,0,1]]
    #         # p2 = [contour_corrected[i+1,0,0], contour_corrected[i+1,0,1]]
    #         # p3 = [contour_corrected[i+2,0,0], contour_corrected[i+2,0,1]]
    #         # xc, yc, radius, angle = radius_of_curvature(p1, p2, p3)
            
    #         # c = next(colormap_iter)
    #         # # if radius < 3:
    #         # ax.plot(p1[0], p1[1], c=c, marker='o')
    #         # ax.plot(p3[0], p3[1], c=c, marker='o')
    #         # ax.plot(xc, yc, c=c, marker='o')
    #         # draw_radius_of_curvature(ax, c, xc, yc, radius)
    
    #%%% CHECK SLOPE OF CONTOUR
    # contour_slopes = flame.frames[frame_nr].contour_data.slopes_of_segmented_contours[contour_nr]
    # # contour_slopes = slope(contour_corrected)
    
    # corrected_slopes = []
    
    # for i, slope in enumerate(contour_slopes):
        
    #     # This is because slopes_of_segmented_contours was calculated with the origin of the image,
    #     # which is top left instead of the conventional bottom left
    #     # slope = -slope
        
    #     slope = np.abs(slope)
        
    #     if  slope >= 0.5:
    #         slope -= 1
        
        
    #     # slope = -slope
        
    #     corrected_slopes.append(slope)
        
    #     x_coord_text = (contour_corrected[i,0,0] + contour_corrected[i+1,0,0])/2
    #     y_coord_text = (contour_corrected[i,0,1] + contour_corrected[i+1,0,1])/2
    #     ax.text(x_coord_text, y_coord_text, str(np.round(slope, 3)), color='k')
    
    #%%% CHECK CHANGE OF SLOPE OF CONTOUR 2 cem

        
#%%% CHECK CHANGE OF SLOPE OF CONTOUR luuk
    # contour_slope_changes = flame.frames[frame_nr].contour_data.slope_changes_of_segmented_contours[contour_nr]
    
    # # Set color map
    # colormap = jet(np.linspace(0, 1, len(contour_slope_changes)))
    
    # # Create color iterator for plotting the data
    # colormap_iter = iter(colormap)
    
    # for i, slope_change in enumerate(contour_slope_changes):
        
    #     # This is because slopes_of_segmented_contours was calculated with the origin of the image,
    #     # which is top left instead of the conventional bottom left
    #     slope_change = -slope_change
        
    #     x_coord_text = contour_corrected[i+1,0,0] 
    #     y_coord_text = contour_corrected[i+1,0,1]
    #     ax.text(x_coord_text, y_coord_text, str(np.round(slope_change, 3)), color='k')
        
    #     p1 = [contour_corrected[i,0,0], contour_corrected[i,0,1]]
    #     p2 = [contour_corrected[i+1,0,0], contour_corrected[i+1,0,1]]
    #     p3 = [contour_corrected[i+2,0,0], contour_corrected[i+2,0,1]]
    #     xc, yc, radius, angle = radius_of_curvature(p1, p2, p3)
        
    #     c = next(colormap_iter)
    #     # if radius < 3:
    #     ax.plot(p1[0], p1[1], c=c, marker='o')
    #     ax.plot(p3[0], p3[1], c=c, marker='o')
    #     ax.plot(xc, yc, c=c, marker='o')
    #     draw_radius_of_curvature(ax, c, xc, yc, radius)
  
    #%%% CHECK CURVATURE OF CONTOUR
    # contour_curvature = curvature(contour_corrected)
    
    # for i, curv in enumerate(contour_curvature):
        
    #     x_coord_text = contour_corrected[i+1,0,0] 
    #     y_coord_text = contour_corrected[i+1,0,1]
        
    #     # R = 1/curv
    #     # print(R)
        
    #     ax.text(x_coord_text, y_coord_text, str(np.round(curv, 3)), color='k')

    # # # cx, cy, radius = define_circle(p1, p2, p3)
    # # contour_tortuosity = flame.frames[frame_nr].contour_data.tortuosity_of_segmented_contours[contour_nr]
    
    # # tort = -contour_tortuosity[0]
    # # radius = radius_of_curvature(p1, p2, p3)
    
    # draw_radius_of_curvature(ax, center_x2, center_y2, radius)
    # xc, yc, r, k = circle_from_two_segments(np.array(p1), np.array(p2), np.array(p3))
    # draw_radius_of_curvature(ax, xc, yc, r)
    
    #%%% CONTOUR DISTRIBUTION
    # contour_distribution = flame.frames[frame_nr].contour_data.contour_distribution
    
    # fig, ax = plt.subplots()
    # ax.imshow(contour_distribution, extent=extent_raw, cmap='viridis')
    
    # # quantity_plot = ax.pcolor(contour_distribution, cmap='viridis')
    
    # # quantity_plot.set_clim(0, contour_distribution.max())
    # # quantity_plot.set_clim(0, 20)
    
    # # Set aspect ratio of plot
    # ax.set_aspect('equal')
    
#%%% Plotting 3
    
# S_f = [segment[8] for segment in final_segments]
# S_d = [segment[9] for segment in final_segments]
# V_n = [segment[0][8] for segment in final_segments]
# V_t = [segment[0][11] for segment in final_segments]
    
# # Create a new figure with a 2x2 grid of subplots
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
# # # Assuming contour_corrected is a numpy array
# contour_corrected = contour_corrected.squeeze() # Remove the singleton dimension

# x = [(final_segments[i][0][0]+ final_segments[i][0][2])/2 for i in range(len(final_segments))]
# y = [(final_segments[i][0][1]+ final_segments[i][0][3])/2 for i in range(len(final_segments))]

# # Create a list of the variables and their corresponding scatter plot colors
# var_colors = [('S_f (m/s)', S_f, 'Blues'), ('S_d (m/s)', S_d, 'Greens'), ('V_n (m/s)', V_n, 'Reds'), ('V_t (m/s)', V_t, 'Purples')]

# # Plot a scatter plot of the midpoint positions for each variable on a separate subplot
# for i, (var, data, cmap) in enumerate(var_colors):
#     # Determine the subplot location
#     row = i // 2
#     col = i % 2
#     ax = axs[row, col]

#     # Create the scatter plot
#     x = [(final_segments[i][0][0] + final_segments[i][0][2])/2 for i in range(len(final_segments))]
#     y = [(final_segments[i][0][1] + final_segments[i][0][3])/2 for i in range(len(final_segments))]
#     scatter = ax.scatter(x, y, c=data, cmap=cmap, s=75)  # Set the size of the colorbar points with the `s` parameter
#     ax.set_xlabel('r (mm)', fontsize=12)
#     ax.set_ylabel('x (mm)', fontsize=12)
#     ax.set_title(var, fontsize=14)
#     ax.grid(True)
#     ax.plot(contour_corrected[:,0], contour_corrected[:,1], linestyle='--', color='black', alpha=0.8)

#     # Add a colorbar to the subplot
#     cbar = plt.colorbar(scatter, ax=ax, aspect=30)
#     cbar.ax.tick_params(labelsize=10, length=5, width=2)
    
# # Adjust the layout of the subplots and display the figure
# fig.tight_layout()
# plt.show()



#%%% Plotting 4

# # Create a list of the variables and their corresponding colors
# var_colors = [('S_d', S_d, 'Greens'), ('V_n', V_n, 'Reds')]

# # Create a new figure for the comparison plot
# fig, ax = plt.subplots(figsize=(8, 6))

# # Create a list of segment indices
# seg_indices = list(range(len(final_segments)))

# # Plot the comparison plot
# for var, data, cmap in var_colors:
#     ax.plot(seg_indices, data, label=var, alpha=0.7)

# # Add x-axis labels and tick labels
# ax.set_xlabel('Detected Segment')
# ax.set_xticks(seg_indices)
# ax.set_xticklabels(['{}'.format(i) for i in seg_indices])

# # Add y-axis label
# ax.set_ylabel('Velocity (m/s)')

# # Add a legend
# ax.legend()

# # Show the plot
# plt.show()
    
    
#%%% Plotting 5 
# # Create a list of the variables
# vars = [('S_f (m/s)', S_f), ('S_d (m/s)', S_d), ('V_n (m/s)', V_n), ('V_t (m/s)', V_t)]

# # Create a new figure with a 2x2 grid of subplots
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

# # Iterate over each variable and plot its PDF distribution
# for i, (var, data) in enumerate(vars):
#     # Determine the subplot location
#     row = i // 2
#     col = i % 2
#     ax = axs[row, col]

#     # Create a kernel density estimate (KDE) of the variable values and plot it
#     kde = gaussian_kde(data)
#     x_vals = np.linspace(min(data), max(data), 100)
#     ax.plot(x_vals, kde(x_vals), linestyle='--', linewidth=2, alpha=0.7)

#     # Add axis labels and a title to the subplot
#     ax.set_xlabel('Variable Values ')
#     ax.set_ylabel('Probability Density')
#     ax.set_title(var)

#     # Add a grid to the subplot
#     ax.grid(True, alpha=0.2)

# # Adjust the layout of the subplots and display the figure
# fig.tight_layout()
# plt.show()

#%%% Plotting 6 
    
# # Create a scatter plot of V_n vs S_d
# fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
# ax = axs
# ax.scatter(S_d, V_n, alpha=0.5)
# ax.set_xlabel('S_d')
# ax.set_ylabel('V_n')
# ax.set_title('Scatter Plot of V_n vs S_d')

# # Fit a linear regression line to the data
# slope, intercept = np.polyfit(S_d, V_n, 1)
# x_vals = np.array(ax.get_xlim())
# y_vals = intercept + slope * x_vals
# ax.plot(x_vals, y_vals, '--', color='red', label='Linear Regression')

# # Fit a 2nd order polynomial regression line to the data
# p2 = np.polyfit(S_d, V_n, 2)
# x_vals = np.linspace(min(S_d), max(S_d), 100)
# y_vals = p2[0] * x_vals**2 + p2[1] * x_vals + p2[2]
# ax.plot(x_vals, y_vals, ':', color='green', label='2nd Order Polynomial Regression')

# # Add a legend to the subplot
# ax.legend()
# ax.grid(True)
# # Show the figure
# plt.show()

#%%% Plotting 7

# # Create a new figure for the comparison plot
# fig, ax = plt.subplots(figsize=(5, 5))

# # Convert list to 1D numpy array
# corrected_slopes_array = np.ravel(corrected_slopes)

# # Create a histogram of the array
# plt.hist(corrected_slopes_array, bins=100, density=True, fc='blue', ec='k')

# # Add labels to the plot
# plt.xlabel('Slope')
# plt.ylabel('Count')
# plt.title('Histogram of Corrected Slopes')

# # Show the plot
# plt.show()





























    
    
    