# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:31:58 2023

@author: luuka
"""

#%% IMPORT PACKAGES
import os
import numpy as np
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
# from celluloid import Camera
from auxiliary_fuctions import * 
from contour_properties import *
import seaborn as sns
import pandas as pd
#%% CLOSE ALL FIGURES
plt.close("all")

#%% SET CONSTANTS
sep = os.path.sep # OS separator

#%% FIGURE SETTINGS
# Use default parameters in plots
# plt.rcParams.update(mpl.rcParamsDefault)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
# Use Latex font in plots
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "font.size": 14.0})


fig_scale = 1
default_fig_dim = mpl.rcParams["figure.figsize"]

viridis = mpl.cm.viridis
viridis_20 = mpl.cm.get_cmap('viridis', 20)
viridis_40 = mpl.cm.get_cmap('viridis', 40)

jet = mpl.cm.jet
jet_20 = mpl.cm.get_cmap('jet', 20)
jet_40 = mpl.cm.get_cmap('jet', 40)

blues = mpl.cm.Blues
blues_20 = mpl.cm.get_cmap('Blues', 20)
blues_40 = mpl.cm.get_cmap('Blues', 40)

reds = mpl.cm.Reds
reds_20 = mpl.cm.get_cmap('Reds', 20)
reds_40 = mpl.cm.get_cmap('Reds', 40)


#%% MAIN FUNCTIONS
def read_piv_data(image_nr):
    
    # Obtain contour number and corresponding image number
    contour_nr = flame.frames[frame_nr].contour_data.accepted_images.index(image_nr) #534 DNG
    image_nr = flame.frames[frame_nr].contour_data.accepted_images[contour_nr] # n_images_used - 1
    
    image_prefix = get_image_prefix(image_nr)
  
    # Transient file name and scaling parameters from headers of file
    xyuv_file = piv_dir + image_prefix + str(image_nr) + '.txt'
    
    piv_file = open(xyuv_file, "r")
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
    
    segmented_contour_x =  segmented_contour[:,0,0]
    segmented_contour_y =  segmented_contour[:,0,1]
    
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

def determine_local_flame_speed(ax, selected_segments, list_of_final_segments, colors):
    
    for t, final_segments in enumerate(selected_segments):
        
        for final_segment in final_segments:
            
            segment = final_segment[0]
            
            X = segment[4]
            Y = segment[5]
            V_nx = segment[6]
            V_ny = segment[7]
            V_n = segment[8]
            # V_tx = filtered_segment[9]
            # V_ty = filtered_segment[10]
            # V_t = filtered_segment[11]
            # V_x = filtered_segment[12]
            # V_y = filtered_segment[13]
            
            # x_intersect_ref = final_segment[1]
            y_intersect_ref = final_segment[2]
            # x_intersect = final_segment[3]
            y_intersect = final_segment[4]
            V_n = final_segment[5]
            flame_front_velocity = final_segment[6]
       
            if y_intersect_ref <= y_intersect:
                
                if t == 0:
                        
                        S_f = V_n - flame_front_velocity
                        
                elif t == 1:
                        
                        S_f = V_n + flame_front_velocity
                            
                final_segment.append(S_f)
                
                list_of_final_segments.append(final_segment)
                
                
                plot_local_flame_speed(ax, final_segment, colors[t]) 
                
    print(image_nr, len(list_of_final_segments), len(contour_corrected))
    
    return list_of_final_segments
    
def local_flame_speed_from_images(image_nr, n_time_steps=1, n_nearest_coords=1, threshold_angle=30):
    
    # Set color map
    colormap = jet(np.linspace(0, 1, 2))
    
    # Create color iterator for plotting the data
    colormap_iter_time = iter(colormap)
    
    image_nr_t0 = image_nr
    image_nr_t1 = image_nr_t0 + n_time_steps
    
    image_nrs = [image_nr_t0, image_nr_t1]
    
    # Time interval between two contours
    dt = n_time_steps*(1/flame.image_rate)
    
    list_of_final_segments = []
    
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
        selected_segments_2_ref= second_segment_selection_procedure(contour_corrected, selected_segments_1_ref, dt, threshold_angle)
        selected_segments_2.append(selected_segments_2_ref)
        
        if i == 0:
            
            # Create figure for velocity field + frame front contour
            fig, ax = plt.subplots()
            
            ax.set_title("Flame " + str(flame_nr) + ": " + "$\phi$=" + str(flame.phi) + ", $H_{2}\%$=" + str(flame.H2_percentage)+ "\n" +
                         "$D_{in}$=" + str(flame.D_in) + " mm, $Re_{D_{in}}$=" + str(flame.Re_D) + "\n" + 
                         "Image: " + str(image_nr_t0) + " - " + str(image_nr_t1) + ", Frame:" + str(frame_nr))
            
            if normalized:
                
                ax.set_xlabel('$r/D$')
                ax.set_ylabel('$x/D$')
                
            else:
                
                ax.set_xlabel('$r$ [mm]')
                ax.set_ylabel('$x$ [mm]')
            
            # Plot velocity vectors
            plot_velocity_vectors(fig, ax, x_ref, y_ref, u_ref, v_ref, scale)
            
            # Plot velocity field
            quantity = velocity_abs_ref
            
            # Choose 'imshow' or 'pcolor' by uncommenting the correct line
            plot_velocity_field_imshow(fig, ax, x_ref, y_ref, quantity)
            # plot_velocity_field_pcolor(fig, ax, x_t0, y_t0, quantity)

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
    
    list_of_final_segments = determine_local_flame_speed(ax, selected_segments_2, list_of_final_segments, colors)
        
    return list_of_final_segments


def local_flame_speed_from_frames(image_nr, n_nearest_coords=1, threshold_angle=30):
    
    # Set color map
    colormap = jet(np.linspace(0, 1, 2))
    
    # Create color iterator for plotting the data
    colormap_iter_time = iter(colormap)
    
    frame0 = 0
    frame1 = 1
    
    frame_nrs = [frame0, frame1]
    
    # Time interval between two contours
    dt = flame.dt*1e-6
    
    list_of_final_segments = []
    
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
        
        if i == 0:
            
            # Create figure for velocity field + frame front contour
            fig, ax = plt.subplots()
            
            ax.set_title("Flame " + str(flame_nr) + ": " + "$\phi$=" + str(flame.phi) + ", $H_{2}\%$=" + str(flame.H2_percentage)+ "\n" +
                         "$D_{in}$=" + str(flame.D_in) + " mm, $Re_{D_{in}}$=" + str(flame.Re_D) + "\n" + 
                         "Image:" + str(image_nr) + ", Frame:" + str(frame0) + ' - ' + str(frame1))
            
            if normalized:
                
                ax.set_xlabel('$r/D$')
                ax.set_ylabel('$x/D$')
                
            else:
                
                ax.set_xlabel('$r$ [mm]')
                ax.set_ylabel('$x$ [mm]')
            
            # Plot velocity vectors
            plot_velocity_vectors(fig, ax, x, y, u, v, scale)
            
            # Plot velocity field
            quantity = velocity_abs
            
            # Choose 'imshow' or 'pcolor' by uncommenting the correct line
            plot_velocity_field_imshow(fig, ax, x, y, quantity)
            # plot_velocity_field_pcolor(fig, ax, x, y, quantity)

        # Create timestamp for plot
        timestamp = ((frame_nrs[0]-frame0)/flame.image_rate)*1e3
        
        # Plot flame front contour t0
        c = next(colormap_iter_time)
        colors.append(c)
        ax = plot_contour_single_color(ax, contour_corrected_ref, timestamp, c)
        
        # Turn on legend
        ax.legend()
        
        # Tighten figure
        fig.tight_layout()
        
        # Important: This operation reverses the frame_nrs, so that the reference time step changes from frame0 to frame1
        frame_nrs.reverse()
    
    list_of_final_segments = determine_local_flame_speed(ax, selected_segments_2, list_of_final_segments, colors)
        
    return list_of_final_segments

def first_segment_selection_procedure(contour_correction, n_windows_x, n_windows_y, x, y, u, v, n_nearest_coords):
    
    contour_x = contour_correction[:,0,0]
    contour_y = contour_correction[:,0,1]
    
    # Create closed contour to define coordinates (of the interrogation windows) in unburned and burned region 
    contour_open = zip(contour_x, contour_y)
    contour_open = list(contour_open)
    contour_open.append((contour_x[0], contour_y[0]))
    contour_closed_path = mpl.path.Path(np.array(contour_open))
    
    selected_segments_1 = []
    
    for i in range(0, len(contour_x) - 1):
        
        # Initialize velocity data related to a segment (index 0: unburnt, index 1: burnt)
        x_loc = [0, 0]
        y_loc = [0, 0]
        V_x = [0, 0]
        V_y = [0, 0]
        V_n = [0, 0]
        V_t = [0, 0]
        
        coords_u = []
        coords_b = []
        
        nearest_coords = []
        
        x_A, y_A, x_B, y_B  = contour_x[i], contour_y[i], contour_x[i+1], contour_y[i+1]
        x_mid, y_mid = (x_A + x_B)/2, (y_A + y_B)/2
        dx, dy = x_B-x_A, y_B-y_A
        
        # Segment length and angle
        # segment_angle = np.arctan2(dy, dx)
        L, segment_angle = segment_properties(dy, dx)
                
        for j in range(n_windows_y):
            
            for i in range(n_windows_x):
                
                X = x[j][i]
                Y = y[j][i]
                
                distance_to_segment = np.sqrt((x_mid - X)**2 + (y_mid - Y)**2)/D
                
                distance_u_threshold_lower = 0 # units: mm 
                distance_u_threshold_upper = 5 # units: mm 
                distance_b_threshold_lower = 0 # units: mm
                distance_b_threshold_upper = 5 # units: mm 
                    
                if contour_closed_path.contains_point((X, Y)):
                    
                    if distance_u_threshold_lower/D <= distance_to_segment <= distance_u_threshold_upper/D:
                        
                        coords_u.append(([j, i, distance_to_segment, X, Y, u[j][i], v[j][i]])) 
                else:
                    
                    if distance_b_threshold_lower/D <= distance_to_segment <= distance_b_threshold_upper/D:
                        
                        coords_b.append(([j, i, distance_to_segment, X, Y, u[j][i], v[j][i]]))
                    
        # Sort coordinates to get candidate coordinates based on distance closest to segment
        coords_u.sort(key = lambda i: i[2])
        coords_b.sort(key = lambda i: i[2])
        
        candidate_coords_u = coords_u[0:n_nearest_coords]
        candidate_coords_b = coords_b[0:n_nearest_coords]
        both_sides_candidate_coords = [candidate_coords_u, candidate_coords_b]
        
        # Check if velocity vector normal of the reference time step intersects the segment itself 
        for side, candidate_coords in enumerate(both_sides_candidate_coords):
            
            nearest_coords = []
            
            for candidate_coord in candidate_coords:
        
                    X = candidate_coord[3]
                    Y = candidate_coord[4]
                    Vx = candidate_coord[5]
                    Vy = candidate_coord[6]
                    
                    if (Vy*dx - Vx*dy) == 0:
                        
                        k = 0
                        
                    else:
                        
                        k = (Vx*(y_A-Y) - Vy*(x_A-X))/(Vy*dx - Vx*dy)
                    
                    if Vx == 0:
                        
                        l = 0
                        
                    else:
                        
                        l = (k*dx + x_A - X)/Vx
                    
                    x_intersect = x_A + k*dx
                    y_intersect = y_A + k*dy
                    
                    distance_A = np.sqrt((x_intersect - x_A)**2 + (y_intersect - y_A)**2)
                    distance_B = np.sqrt((x_intersect - x_B)**2 + (y_intersect - y_B)**2)
                    
                    
                    if distance_A <= L and distance_B <= L:
                        
                        if (side==0 and l>=0) or (side==1 and l<=0):
                            
                            nearest_coords.append(candidate_coord)
            
            x_avg_list = []
            y_avg_list = []
            Vx_avg_list = []
            Vy_avg_list = []
            
            for j, i, distance_dummy, x_dummy, y_dummy, u_dummy, v_dummy in nearest_coords:
                    
                x_avg_list.append(x_dummy)
                y_avg_list.append(y_dummy)
                Vx_avg_list.append(u_dummy)
                Vy_avg_list.append(v_dummy)
        
            if nearest_coords:
                
                x_avg = np.nanmean(x_avg_list)
                y_avg = np.nanmean(y_avg_list)
                Vx_avg = np.nanmean(Vx_avg_list)
                Vy_avg = np.nanmean(Vy_avg_list)
                
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
            
        
        if 0 in V_n:
            
            pass
        
        else:
            
            for side in range(len(both_sides_candidate_coords)): # len(all_candidate_coords)
                
                if side == 0:
                    
                    V_x = V_x[side]
                    V_y = V_y[side]
                    
                    V_tx = V_t[side]*np.cos(segment_angle)
                    V_ty = V_t[side]*np.sin(segment_angle)
                    
                    V_nx = -V_n[side]*np.sin(segment_angle)
                    V_ny = V_n[side]*np.cos(segment_angle)
                    
                    selected_segment_1 = [x_A, y_A, x_B, y_B, x_loc[side], y_loc[side], V_nx, V_ny, V_n[side], V_tx, V_ty, V_t[side], V_x, V_y]
                    
                    selected_segments_1.append(selected_segment_1)

    return selected_segments_1

def second_segment_selection_procedure(contour_correction, selected_segments_1, dt, threshold_angle):
    
    contour_x = contour_correction[:,0,0]
    contour_y = contour_correction[:,0,1]
    
    selected_segments_2 = []
    
    for selected_segment in selected_segments_1:
        
        # Check if velocity vector normal to the segment of the reference time step intersects the segment itself 
        x_A, y_A, x_B, y_B  = selected_segment[0], selected_segment[1], selected_segment[2], selected_segment[3]
        x_mid_ref, y_mid_ref = (x_A + x_B)/2, (y_A + y_B)/2
        dx, dy = x_B-x_A, y_B-y_A
        
        # Segment length and angle
        # segment_angle_ref = np.arctan2(dy, dx)
        L_ref, segment_angle_ref = segment_properties(dy, dx)
        
        # Velocity data of reference time step
        X = selected_segment[4]
        Y = selected_segment[5]
        V_nx = selected_segment[6]
        V_ny = selected_segment[7]
        V_n = selected_segment[8]
        
        if (V_ny*dx - V_nx*dy) == 0:
            
            k = 0
            
        else:
            
            k = (V_nx*(y_A-Y) - V_ny*(x_A-X))/(V_ny*dx - V_nx*dy)
            
        if V_nx == 0:
            
            l = 0
            
        else:
            
            l = (k*dx + x_A - X)/V_nx
        
        x_intersect_ref = x_A + k*dx
        y_intersect_ref = y_A + k*dy
        
        distance_A = np.sqrt((x_intersect_ref - x_A)**2 + (y_intersect_ref - y_A)**2)
        distance_B = np.sqrt((x_intersect_ref - x_B)**2 + (y_intersect_ref - y_B)**2)
        
        # Check if velocity vector normal to the segment intersects with the segment of the reference time step
        if distance_A <= L_ref and distance_B <= L_ref:
            
            # Check if the coordinate is on the unburnt side
            if (l >= 0):
                
                # Check if velocity vector normal to the segment of the reference time step 
                # intersects a segment in the chosen time step 

                for i in range(0, len(contour_x) - 1):
                    
                    x_A, y_A, x_B, y_B  = contour_x[i], contour_y[i], contour_x[i+1], contour_y[i+1]
                    x_mid, y_mid = (x_A + x_B)/2, (y_A + y_B)/2
                    dx, dy = x_B-x_A, y_B-y_A
                    
                    # Segment length and angle
                    # segment_angle = np.arctan2(dy, dx)
                    L, segment_angle = segment_properties(dy, dx)
                    
                    if (V_ny*dx - V_nx*dy) == 0:
                        
                        k = 0
                        
                    else:
                        
                        k = (V_nx*(y_A-Y) - V_ny*(x_A-X))/(V_ny*dx - V_nx*dy)
                    
                    if V_nx == 0:
                        
                        l = 0
                        
                    else:
                        
                        l = (k*dx + x_A - X)/V_nx
                    
                    x_intersect = x_A + k*dx
                    y_intersect = y_A + k*dy
                    
                    distance_A = np.sqrt((x_intersect - x_A)**2 + (y_intersect - y_A)**2)
                    distance_B = np.sqrt((x_intersect - x_B)**2 + (y_intersect - y_B)**2)
                    
                    # Check if velocity vector normal to segment intersects with the segment of the chosen time step
                    if distance_A <= L and distance_B <= L:
                            
                            distance_between_segment = np.sqrt((x_mid_ref - x_mid)**2 + (y_mid_ref - y_mid)**2)
                            
                            if (l >= 0) and (distance_between_segment < 3*L):
                                
                                if (np.abs(segment_angle - segment_angle_ref)) < np.deg2rad(threshold_angle):
                                    
                                    flame_front_movement = np.sqrt((x_intersect_ref - x_intersect)**2 + (y_intersect_ref - y_intersect)**2)
                                    
                                    flame_front_velocity = flame_front_movement*(D*1e-3)/dt
                                    
                                    flame_front_velocity /= U_bulk
                                         
                                    selected_segment_2 = [selected_segment, x_intersect_ref, y_intersect_ref, x_intersect, y_intersect, V_n, flame_front_velocity]
                                    
                                    selected_segments_2.append(selected_segment_2)
                                        
                                                          
                             
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
    
    # bar = fig.colorbar(quantity_plot)
    # bar.set_label('velocity [ms$^-1$]') #('$u$/U$_{b}$ [-]')

def plot_contour_single_color(ax, contour, timestamp=0, c='b'):
    
    contour_x, contour_y =  contour[:,0,0], contour[:,0,1]
    
    # # Plot flame front contour
    label = 't = ' + str(timestamp) + ' ms'
    ax.plot(contour_x, contour_y, color=c, marker='None', lw=2, label=label)
    
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
    
    X = segment[4]
    Y = segment[5]
    V_nx = segment[6]
    V_ny = segment[7]
    # V_n = segment[8]
    # V_tx = filtered_segment[9]
    # V_ty = filtered_segment[10]
    # V_t = filtered_segment[11]
    # V_x = filtered_segment[12]
    # V_y = filtered_segment[13]
    
    x_intersect_ref = final_segment[1]
    y_intersect_ref = final_segment[2]
    x_intersect = final_segment[3]
    y_intersect = final_segment[4]
    V_n = final_segment[5]
    flame_front_velocity = final_segment[6]
    S_f = final_segment[7]
    
    ax.text(x_intersect, y_intersect, str(np.round(flame_front_velocity, 3)), color='k')
    
    ax.text(X, Y, np.round(V_n, 3), c='k')
    
    ax.text((x_intersect_ref + x_intersect)/2 , (y_intersect_ref + y_intersect)/2, np.round(S_f, 3), color='w')
    
    ax.plot(x_intersect_ref, y_intersect_ref, c='r', marker='x')
    ax.plot(x_intersect, y_intersect, c='r', marker='x')
    
    ax.plot(X, Y, c=color, marker='o')
    ax.quiver(X, Y, V_nx, V_ny, angles='xy', scale_units='xy', scale=scale, ls='-', fc=color, ec=color)
    
#%% AXUILIARY PLOTTING FUNCTIONS
def save_animation(camera, filename, interval=500, repeat_delay=1000):
    
    animation = camera.animate(interval = interval, repeat = True, repeat_delay = repeat_delay)
    animation.save(filename + ".gif", writer = 'pillow')


def save_figure(fig, fig_name):
    
    fig.savefig(fig_name + '.png', dpi=1200) 
    
#%% START OF CODE
# Before executing code, Python interpreter reads source file and define few special variables/global variables. 
# If the python interpreter is running that module (the source file) as the main program, it sets the special __name__ variable to have a value “__main__”. If this file is being imported from another module, __name__ will be set to the module’s name. Module’s name is available as value to __name__ global variable. 
if __name__ == "__main__":
    
    #%%% CHOOSE FLAME AND READ FLAME OBJECT
    flame_nr = 4
    filtered_data = 0 # 1:True, 0:False
    frame_nr = 0
    segment_length_mm = 1 # units: mm
    window_size = 27 # units: pixels

    spydata_dir = 'spydata'

    file_pickle = 'flame_' + str(flame_nr) + '_' + 'unfiltered' + '_' +'segment_length_' + str(segment_length_mm) + 'mm_' + 'wsize_' + str(window_size) + 'pixels.pkl'

    with open(spydata_dir + sep + file_pickle, 'rb') as f:
        flame = pickle.load(f)

    #%%% NON_DIMENSIONALIZE TOGGLE
    normalized = False

    # Set if plot is normalized or non-dimensionalized
    if normalized:
        D = flame.D_in
        U_bulk = flame.Re_D*flame.properties.nu_u/(flame.D_in*1e-3)
        scale = 20
    else:
        D = 1
        U_bulk = 1
        scale = 2

    #%%% READ RAW IMAGE DATA
    data_dir = "pre_data"
    post_dir = "post_data"
    piv_folder = "_PIV_MP(3x12x12_0%ov)"

    piv_dir = os.getcwd() + sep + data_dir + sep + flame.name + sep + flame.record_name + piv_folder + sep

    # Raw file name and scaling parameters from headers of file
    raw_dir = os.getcwd() + sep + data_dir + sep + flame.name + sep + flame.record_name + sep

    raw_file = open(raw_dir +'B0001.txt', "r")
    scaling_info = raw_file.readline()
    raw_file.close()
    scaling_info_raw = scaling_info.split()

    n_windows_x_raw = int(scaling_info_raw[3])
    n_windows_y_raw = int(scaling_info_raw[4])

    window_size_x_raw = float(scaling_info_raw[6])
    x_origin_raw = float(scaling_info_raw[7])
    window_size_y_raw = float(scaling_info_raw[10])
    y_origin_raw = float(scaling_info_raw[11])

    x_left_raw = x_origin_raw
    x_right_raw = x_origin_raw + (n_windows_x_raw - 1)*window_size_x_raw
    y_bottom_raw = y_origin_raw + (n_windows_y_raw - 1)*window_size_y_raw
    y_top_raw = y_origin_raw

    extent_raw =  np.array([
                            x_left_raw - window_size_x_raw / 2,
                            x_left_raw + (n_windows_x_raw - 0.5) * window_size_x_raw,
                            y_top_raw + (n_windows_y_raw - 0.5) * window_size_y_raw,
                            y_top_raw - window_size_y_raw / 2
                            ])/D

    #%%% READ PIV IMAGE DIMENSIONS
    image_nr = 1

    contour_nr, n_windows_x, n_windows_y, x, y, u, v, velocity_abs = read_piv_data(image_nr)

    window_size_x = np.mean(np.diff(x[0,:]))
    window_size_y = np.mean(np.diff(y[:,0]))

    extent_piv =  np.array([
                            x.min() - window_size_x / 2,
                            x.min() + (n_windows_x - 0.5) * window_size_x,
                            y.max() + (n_windows_y - 0.5) * window_size_y,
                            y.max() - window_size_y / 2
                            ])

    #%%% SET PARAMETERS of INTEREST

    image_nr = 360
    n_time_steps = 1
    n_nearest_coords = 1
    threshold_angle = 30
    
    #%% PLOT CONTOUR
    
    # Create figure for velocity field + frame front contour
    fig, ax = plt.subplots()
    ax.set_xlabel('$r/D$')
    ax.set_ylabel('$x/D$')
    
    ax.set_title("Flame " + str(flame_nr) + ": " + "$\phi$=" + str(flame.phi) + ", $H_{2}\%$=" + str(flame.H2_percentage)+ "\n" +
                  "$D_{in}$=" + str(flame.D_in) + " mm, $Re_{D_{in}}$=" + str(flame.Re_D) + "\n" + 
                  "Image \#: " + str(image_nr))
    
    # Read PIV data
    contour_nr, n_windows_x, n_windows_y, x, y, u, v, velocity_abs = read_piv_data(image_nr)
    
    # Contour correction RAW --> (non-dimensionalized) WORLD [reference]
    contour_corrected = contour_correction(contour_nr)
    
    contour = flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
    plot_contour_single_color(ax, contour_corrected)
    plt.savefig('Single_front_demonstration', bbox_inches = 'tight')
    
    #%% CHECK SLOPE OF CONTOUR
    contour_slopes = flame.frames[frame_nr].contour_data.slopes_of_segmented_contours[contour_nr]
    # contour_slopes = slope(contour_corrected)
    
    for i, slope in enumerate(contour_slopes):
        
        # This is because slopes_of_segmented_contours was calculated with the origin of the image,
        # which is top left instead of the conventional bottom left
        slope = -slope
    
        x_coord_text = (contour_corrected[i,0,0] + contour_corrected[i+1,0,0])/2
        y_coord_text = (contour_corrected[i,0,1] + contour_corrected[i+1,0,1])/2
    #     ax.text(x_coord_text, y_coord_text, str(np.round(slope, 3)), color='k')
    
    #%% CHECK TORTOSITY OF CONTOUR
    contour_tortuosity = flame.frames[frame_nr].contour_data.tortuosity_of_segmented_contours[contour_nr]
    
    for i, tort in enumerate(contour_tortuosity):
        
        # This is because slopes_of_segmented_contours was calculated with the origin of the image,
        # which is top left instead of the conventional bottom left
        tort = -tort
        
        x_coord_text = contour_corrected[i+1,0,0] 
        y_coord_text = contour_corrected[i+1,0,1]
        # ax.text(x_coord_text, y_coord_text, str(np.round(tort, 3)), color='k')
    
    #%% CONTOUR DISTRIBUTION
    # contour_distribution = flame.frames[frame_nr].contour_data.contour_distribution
    
    # fig, ax = plt.subplots()
    # ax.imshow(contour_distribution, extent=extent_raw, cmap='viridis')
    
    # quantity_plot = ax.pcolor(contour_distribution, cmap='viridis')
    
    # quantity_plot.set_clim(0, contour_distribution.max())
    # quantity_plot.set_clim(0, 20)
    
    # Set aspect ratio of plot
    # ax.set_aspect('equal')
       
    #%% CHECK
    super_coords = local_flame_speed_from_images(image_nr, n_time_steps, n_nearest_coords, threshold_angle)
    # local_flame_speed_from_frames(image_nr, n_nearest_coords, threshold_angle)
    
    
    
    # final_segment = [selected_segment, x_intersect_ref, y_intersect_ref, x_intersect, y_intersect, V_n, flame_front_velocity]
    # selected_segment = [x_A, y_A, x_B, y_B, x_loc[side], y_loc[side], V_nx, V_ny, V_n[side], V_tx, V_ty, V_t[side], V_x, V_y]
#%% Plotting
    
    #%%% Histogram Tortuosity
    
    # Compute average tortuosity
    avg_tortuosity = np.mean(contour_tortuosity)
    
    # Plot histogram of tortuosity values
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(contour_tortuosity, bins=30, edgecolor='black', alpha=0.5, color='blue')
    
    # Add vertical line for average tortuosity
    ax.axvline(x=avg_tortuosity, linestyle='--', color='r', label='Average')
    
    # Set axis labels and legend
    ax.set_xlabel('Tortuosity')
    ax.set_ylabel('Counts')
    ax.legend(title=f"Average: {avg_tortuosity:.2f}")
    ax.grid()
    
    #%%% Pdf Tortuosity
    import scipy.stats as stats

    # Fit a probability density function to the data
    pdf = stats.gaussian_kde(contour_tortuosity)
    
    # Create a figure for the PDF distribution
    fig, ax = plt.subplots()
    
    # Generate x values for the PDF plot
    x_pdf = np.linspace(np.min(contour_tortuosity), np.max(contour_tortuosity), 100)
    
    # Plot the PDF
    ax.plot(x_pdf, pdf(x_pdf), color='blue')
    
    # Add vertical line for average tortuosity
    ax.axvline(x=avg_tortuosity, linestyle='--', color='r', label='Average')
    
    # Set axis labels and legend
    ax.set_xlabel('Tortuosity')
    ax.set_ylabel('Probability Density')
    ax.legend(title=f"Average: {avg_tortuosity:.2f}")
    ax.grid()

    
    #%%% Histogram Slope
    
    # Compute average tortuosity
    avg_slope = np.mean(contour_slopes)
    
    # Plot histogram of tortuosity values
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(contour_slopes, bins=30, edgecolor='black', alpha=0.5, color='blue')
    
    # Add vertical line for average tortuosity
    ax.axvline(x=avg_slope, linestyle='--', color='r', label='Average')
    
    # Set axis labels and legend
    ax.set_xlabel('Slope')
    ax.set_ylabel('Counts')
    ax.legend(title=f"Average: {avg_slope:.2f}")
    ax.grid()
        
    #%%% Pdf Slope
    import scipy.stats as stats
    
    # Fit a probability density function to the data
    pdf = stats.gaussian_kde(contour_slopes)
    
    # Create a figure for the PDF distribution
    fig, ax = plt.subplots()
    
    # Generate x values for the PDF plot
    x_pdf = np.linspace(np.min(contour_slopes), np.max(contour_slopes), 100)
    
    # Plot the PDF
    ax.plot(x_pdf, pdf(x_pdf), color='blue')
    
    # Add vertical line for average tortuosity
    ax.axvline(x=avg_slope, linestyle='--', color='r', label='Average')
    
    # Set axis labels and legend
    ax.set_xlabel('Slope')
    ax.set_ylabel('Probability Density')
    ax.legend(title=f"Average: {avg_slope:.2f}")
    ax.grid()

    #%%% Scatter tortuosity

    avg_tortuosity = np.mean(contour_tortuosity)
    # define length_contour_corrected as a matrix from 1 to len(contour_corrected)
    length_contour_corrected = np.arange(1, len(contour_corrected)-1)

    
    # create a scatter plot with x-axis as length_contour_corrected and y-axis as contour_tortuosity
    fig, ax = plt.subplots()
    ax.scatter(length_contour_corrected, contour_tortuosity)
    # Add horizontal line for average tortuosity
    ax.axhline(y=avg_tortuosity, linestyle='--', color='r', label='Average')
    
    # set x-axis label and title
    ax.set_xlabel('Segment Number')
    ax.set_ylabel('Tortuosity')
    ax.set_title('Contour Tortuosity vs Number of Segments')
    ax.legend(title=f"Average: {avg_tortuosity:.2f}")
    ax.grid()

    #%%% Scatter Slope
    
    avg_slope = np.mean(contour_slopes)
    # define length_contour_corrected as a matrix from 1 to len(contour_corrected)
    length_contour_corrected = np.arange(1, len(contour_corrected))
    
    
    # create a scatter plot with x-axis as length_contour_corrected and y-axis as contour_tortuosity
    fig, ax = plt.subplots()
    ax.scatter(length_contour_corrected, contour_slopes)
    # Add horizontal line for average tortuosity
    ax.axhline(y=avg_slope, linestyle='--', color='r', label='Average')
    
    # set x-axis label and title
    ax.set_xlabel('Segment Number')
    ax.set_ylabel('Slope')
    ax.set_title('Contour Slopes vs Number of Segments')
    ax.legend(title=f"Average: {avg_slope:.2f}")
    ax.grid()



    #%%% Scatter Slope

    #%%% Figure V_n
    
    fig, ax = plt.subplots()
    
    #Histogram
    bins = 10
    V_n = [super_coords[i][5] for i in range(len(super_coords))]
    ax.hist(V_n, bins=bins, edgecolor='black', alpha=0.5, color='red', label='V_n')
    
    mean_V_n = np.mean(V_n)
    ax.axvline(x=mean_V_n, color='green', linestyle='-.', label='Mean V_n')
    
    # Plot PDF
    kde_kwargs = {'shade': True}
    sns.kdeplot(V_n, ax=ax, color='blue', label='V_n', **kde_kwargs)
    
    ax.set_xlabel('Magnitude (m/s)')
    ax.set_ylabel('Number of Detected V_n')
    ax.set_title('Histogram-PDF of V_n Distribution')
    ax.grid(True)
    ax.legend(title=f"Average: {mean_V_n:.2f}")
    
    #%%% Figure V_n on Contour
    fig, ax = plt.subplots()
    
    # Define the x and y coordinates
    x = [(super_coords[i][0][0]+ super_coords[i][0][2])/2 for i in range(len(super_coords))]
    y = [(super_coords[i][0][1]+ super_coords[i][0][3])/2 for i in range(len(super_coords))]
        
    # Load the velocity data
    V_n = [super_coords[i][5] for i in range(len(super_coords))]
    
    # Create a scatter plot of the points, color coded based on their radius of curvature
    scatter = ax.scatter(x, y, c=V_n, cmap='RdYlBu_r')
    
    # Add a colorbar to each plot
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('V_n [m/s] ')
    
    # Set the x and y axis labels
    ax.set_xlabel('r[mm]')
    ax.set_ylabel('x[mm]')
    ax.grid(True)
    ax.set_title('Distribution of V_n on selected segment midpoints')
    ax.plot(contour_corrected[:,0,0], contour_corrected[:,0,1], linestyle='--', color='black', alpha = 0.4)

    #%%% Figure Flame_front_vel
    
    fig, ax = plt.subplots()
    
    #Histogram
    bins = 10
    Flame_front_vel = [super_coords[i][6] for i in range(len(super_coords))]
    ax.hist(Flame_front_vel, bins=bins, edgecolor='black', alpha=0.5, color='red', label='Flame_front_vel')
    
    mean_Flame_front_vel = np.mean(Flame_front_vel)
    ax.axvline(x=mean_Flame_front_vel, color='green', linestyle='-.', label='Mean Flame_front_vel')
    
    # Plot PDF
    kde_kwargs = {'shade': True}
    sns.kdeplot(Flame_front_vel, ax=ax, color='blue', label='Flame_front_vel', **kde_kwargs)
    
    ax.set_xlabel('Magnitude (m/s)')
    ax.set_ylabel('Number of Detected Flame_front_vel')
    ax.set_title('Histogram-PDF of Flame_front_vel Distribution')
    ax.grid(True)
    ax.legend(title=f"Average: {mean_Flame_front_vel:.2f}")
    
    #%%% Figure Flame_front_vel on Contour
    fig, ax = plt.subplots()
    
    # Define the x and y coordinates
    x = [(super_coords[i][0][0]+ super_coords[i][0][2])/2 for i in range(len(super_coords))]
    y = [(super_coords[i][0][1]+ super_coords[i][0][3])/2 for i in range(len(super_coords))]
        
    # Load the velocity data
    Flame_front_vel = [super_coords[i][6] for i in range(len(super_coords))]
    
    # Create a scatter plot of the points, color coded based on their radius of curvature
    scatter = ax.scatter(x, y, c=Flame_front_vel, cmap='RdYlBu_r')
    
    # Add a colorbar to each plot
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Flame_front_vel [m/s] ')
    
    # Set the x and y axis labels
    ax.set_xlabel('r[mm]')
    ax.set_ylabel('x[mm]')
    ax.grid(True)
    ax.set_title('Distribution of Flame_front_vel on selected segment midpoints')
    ax.plot(contour_corrected[:,0,0], contour_corrected[:,0,1], linestyle='--', color='black', alpha = 0.4)

    #%%% Figure S_f
    
    fig, ax = plt.subplots()
    
    #Histogram
    bins = 10
    S_f = [super_coords[i][7] for i in range(len(super_coords))]
    ax.hist(S_f, bins=bins, edgecolor='black', alpha=0.5, color='red', label='S_f')
    
    mean_S_f = np.mean(S_f)
    ax.axvline(x=mean_S_f, color='green', linestyle='-.', label='Mean S_f')
    
    # Plot PDF
    kde_kwargs = {'shade': True}
    sns.kdeplot(S_f, ax=ax, color='blue', label='S_f', **kde_kwargs)
    
    ax.set_xlabel('Magnitude (m/s)')
    ax.set_ylabel('Number of Detected S_f')
    ax.set_title('Histogram-PDF of S_f Distribution')
    ax.grid(True)
    ax.legend(title=f"Average: {mean_S_f:.2f}")


    #%%% Figure S_f on Contour
    fig, ax = plt.subplots()
    
    # Define the x and y coordinates
    x = [(super_coords[i][0][0]+ super_coords[i][0][2])/2 for i in range(len(super_coords))]
    y = [(super_coords[i][0][1]+ super_coords[i][0][3])/2 for i in range(len(super_coords))]
        
    # Load the velocity data
    S_f = [super_coords[i][7] for i in range(len(super_coords))]
    
    # Create a scatter plot of the points, color coded based on their radius of curvature
    scatter = ax.scatter(x, y, c=S_f, cmap='RdYlBu_r')
    
    # Add a colorbar to each plot
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('S_f [m/s]')
    
    # Set the x and y axis labels
    ax.set_xlabel('r[mm]')
    ax.set_ylabel('x[mm]')
    ax.grid(True)
    ax.set_title('Distribution of S_f on selected segment midpoints')
    # plot_contour_single_color(ax, contour_corrected)
    ax.plot(contour_corrected[:,0,0], contour_corrected[:,0,1], linestyle='--', color='black', alpha = 0.4)


#%%% curvature


# fig, ax = plt.subplots()


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors

# def calculate_signed_curvature(x, y):
#     n = len(x)
#     curvature = np.zeros(n)
#     for i in range(1, n-1):
#         x1, y1 = x[i-1], y[i-1]
#         x2, y2 = x[i], y[i]
#         x3, y3 = x[i+1], y[i+1]

#         # Calculate the lengths of the sides of the triangle formed by the three points
#         a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
#         c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

#         # Calculate the radius of the circle that passes through the three points
#         s = (a + b + c) / 2.0
#         r = a * b * c / (4.0 * np.sqrt(s * (s - a) * (s - b) * (s - c)))

#         # Calculate the curvature of the arc
#         curvature[i] = 1.0 / r

#         # Determine the orientation of the curve
#         v1 = np.array([x2 - x1, y2 - y1])
#         v2 = np.array([x3 - x2, y3 - y2])
#         cross_product = np.cross(v1, v2)
#         sign = np.sign(cross_product)
#         curvature[i] *= sign

#     # Set the curvature for the first and last points to be equal to the second and second-to-last points, respectively
#     curvature[0] = curvature[1]
#     curvature[-1] = curvature[-2]

#     return curvature

# # Assuming you have already calculated the curvature values using the function I provided earlier
# x = contour_corrected[:,0,0]
# y = contour_corrected[:,0,1]

# curvature = calculate_signed_curvature(x, y)

# # Set up the colormap with a diverging distribution
# cmap = plt.cm.seismic
# norm = colors.TwoSlopeNorm(vmin=-abs(curvature).max(), vcenter=0, vmax=abs(curvature).max())

# # Create scatter plot
# plt.scatter(x, y, c=curvature, cmap=cmap, norm=norm)

# # Add colorbar legend
# cbar = plt.colorbar(label='Signed Curvature', ticks=[-abs(curvature).max(), 0, abs(curvature).max()])
# cbar.ax.set_yticklabels(['-max', '0', 'max'])
# ax.plot(contour_corrected[:,0,0], contour_corrected[:,0,1], linestyle='--', color='black', alpha = 0.4)
# ax.grid(True)

# # Set axis labels
# plt.xlabel('X')
# plt.ylabel('Y')

# # Show the plot
# plt.show()

#%%% curvature 2


# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_curvature2(x, y):
#     n = len(x)
#     curvature2 = np.zeros(n)
#     for i in range(1, n-1):
#         x1, y1 = x[i-1], y[i-1]
#         x2, y2 = x[i], y[i]
#         x3, y3 = x[i+1], y[i+1]
        
#         # Calculate the lengths of the sides of the triangle formed by the three points
#         a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
#         c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        
#         # Calculate the radius of the circle that passes through the three points
#         s = (a + b + c) / 2.0
#         r = a * b * c / (4.0 * np.sqrt(s * (s - a) * (s - b) * (s - c)))
        
#         # Calculate the curvature of the arc
#         curvature2[i] = 1.0 / r
        
#     # Set the curvature for the first and last points to be equal to the second and second-to-last points, respectively
#     curvature2[0] = curvature2[1]
#     curvature2[-1] = curvature2[-2]
    
#     return curvature2


# x = contour_corrected[:,0,0]
# y = contour_corrected[:,0,1]
# curvature2 = calculate_curvature2(x, y)

# fig, ax = plt.subplots()

# # Assuming you have already calculated the curvature values using the function I provided earlier
# x = contour_corrected[:,0,0]
# y = contour_corrected[:,0,1]

# curvature2 = calculate_curvature2(x, y)

# # Create scatter plot
# plt.scatter(x, y, c=curvature2, cmap='coolwarm')

# # Add colorbar legend
# plt.colorbar(label='Curvature')

# # Set axis labels
# plt.xlabel('X')
# plt.ylabel('Y')

# # Show the plot
# plt.show()

#%%% curvature 3

# def calculate_curvature3(x, y):
#     n = len(x)
#     curvature3 = np.zeros(n)
#     for i in range(1, n-1):
#         x1, y1 = x[i-1], y[i-1]
#         x2, y2 = x[i], y[i]
#         x3, y3 = x[i+1], y[i+1]

#         # Calculate the distances between the three points
#         a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         c = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        
#         # Calculate the radius of the circle that passes through the two points (x1,y1) and (x3,y3)
#         r = a * a + c * c - 2 * a * c * np.cos(np.arctan2(y3 - y1, x3 - x1) - np.arctan2(y2 - y1, x2 - x1)) / (2 * np.abs(a * np.sin(np.arctan2(y3 - y1, x3 - x1) - np.arctan2(y2 - y1, x2 - x1))))
        
#         # Calculate the curvature of the arc
#         curvature3[i] = 1.0 / r
        
#     # Set the curvature for the first and last points to be equal to the second and second-to-last points, respectively
#     curvature3[0] = curvature3[1]
#     curvature3[-1] = curvature3[-2]
    
#     return curvature3


# fig, ax = plt.subplots()

# # Assuming you have already calculated the curvature values using the function I provided earlier
# x = contour_corrected[:,0,0]
# y = contour_corrected[:,0,1]

# curvature3 = calculate_curvature3(x, y)

# # Create scatter plot
# plt.scatter(x, y, c=curvature3, cmap='coolwarm')

# # Add colorbar legend
# plt.colorbar(label='Curvature')

# # Set axis labels
# plt.xlabel('X')
# plt.ylabel('Y')

# # Show the plot
# plt.show()

#%%% curvature 1 vs tortuosity comparision

# fig, ax = plt.subplots()

# x = range(-1, len(curvature)-1)
# x1 = range(0, len(contour_tortuosity))

# plt.plot(x, -curvature, color='blue', alpha=0.5, label='- Curvature')
# plt.plot(x1, contour_tortuosity, color='red', alpha=0.5, label='Tortuosity')

# plt.xlabel('Segment Numbers')
# plt.ylabel('Value')
# plt.legend()
# ax.grid()

# plt.show()


#%%% Parameters for excel- 10 image 

# print(avg_slope, avg_tortuosity, np.mean(V_n), np.mean(Flame_front_vel), np.mean(S_f), np.mean(curvature))

# print("{:.2f}".format(avg_slope), "{:.2f}".format(avg_tortuosity), "{:.2f}".format(np.mean(V_n)),"{:.2f}".format(np.mean(Flame_front_vel)), "{:.2f}".format(np.mean(np.mean(S_f))), "{:.2f}".format(np.mean(np.mean(curvature))))


#%%% 20 image contours together 


fig, ax = plt.subplots()

ax.plot(contour_corrected[:,0,0], contour_corrected[:,0,1], linestyle='--', color='black', alpha = 0.4)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
ax.grid()

plt.show()




