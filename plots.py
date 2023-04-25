# -*- coding: utf-8 -*-
'''
Created on Mon Mar 27 15:04:13 2023

@author: laaltenburg
'''

#%% IMPORT PACKAGES
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import progressbar
from local_flame_speed import *
from premixed_flame_properties import *

#%% CLOSE ALL FIGURES
plt.close('all')

#%% SET CONSTANTS
sep = os.path.sep # OS separator

#%% FIGURE SETTINGS
# # Use default parameters in plots
# mpl.rcParams.update(mpl.rcParamsDefault)

# Use Latex font in plots
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 14.0})


fig_scale = 1
default_fig_dim = mpl.rcParams['figure.figsize']

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

# Google hex-colors
google_red = '#db3236'
google_green = '#3cba54'
google_blue = '#4885ed'
google_yellow = '#f4c20d'

#%%% SET PARAMETERS OF INTEREST

# image_nrs = [*range(1, 2, 1)]
image_nrs = [*range(1, 1300, 1)]

# Number of steps between two timesteps
n_time_steps = 1

# Set amount of vector considered
n_nearest_coords = 1

# Threshold angle between segments of consecutive timesteps
threshold_angle = 180


dt = n_time_steps*(1/flame.image_rate)


cases = ['by frame', 'by image', 'by image']

n_time_steps = [0, 1, 2]

list_of_S_f_list = []

slope_change_avg_list = []

cases = [0]


# list_of_slope_list = []
# list_of_slope_change_avg_list = []
# list_of_slope_change0_list = []
# list_of_slope_change1_list = []

# for i, case in enumerate(cases):
    
#     list_of_final_segments = []
#     slope_list = []
#     slope_change_avg_list = []
#     slope_change0_list = []
#     slope_change1_list = []
    
#     for image_nr in progressbar.progressbar(image_nrs): 
        
#         contour_nr, n_windows_x, n_windows_y, x, y, u, v, velocity_abs = read_piv_data(image_nr)
#         contour_slopes = flame.frames[frame_nr].contour_data.slopes_of_segmented_contours[contour_nr]
#         contour_slope_changes = flame.frames[frame_nr].contour_data.slope_changes_of_segmented_contours[contour_nr]
        
#         corrected_slopes = []
        
#         for slope in contour_slopes:
            
#             slope = np.abs(slope)
            
#             if  slope >= 0.5:
#                 slope -= 1
            
#             corrected_slopes.append(slope)
            
#         final_segments = local_flame_speed_from_single_image_double_frame(image_nr, n_nearest_coords, threshold_angle)
        
#         for final_segment in final_segments:
            
#             i_segment = final_segment[0][14]
#             contour_slopes = final_segment[0][-1]
            
#             slope_change_avg = 0.0
#             slope_change0 = 0.0
#             slope_change1 = 0.0
            
#             if i_segment > 0 and i_segment < len(contour_slope_changes):
#                 slope_change0 = -contour_slope_changes[i_segment-1]
#                 slope_change1 = -contour_slope_changes[i_segment]
#                 slope_change_avg = (slope_change0 + slope_change1)/2
            
#             slope_list.append(slope_change)
#             slope_change_avg_list.append(slope_change_avg)
#             slope_change0_list.append(slope_change0)
#             slope_change1_list.append(slope_change1)
            
#     # list_of_slope_list.append(slope_list)
#     list_of_slope_change_avg_list.append(slope_change_avg_list)
#     list_of_slope_change0_list.append(slope_change0_list)
#     list_of_slope_change1_list.append(slope_change1_list)

list_of_slope_list = []
list_of_slope_change_avg_list = []
list_of_slope_change0_list = []
list_of_slope_change1_list = []

for i, case in enumerate(cases):
    
    list_of_final_segments = []
    slope_list = []
    slope_change_avg_list = []
    slope_change0_list = []
    slope_change1_list = []
    
    for image_nr in progressbar.progressbar(image_nrs): 
        
        contour_nr, n_windows_x, n_windows_y, x, y, u, v, velocity_abs = read_piv_data(image_nr)
        contour_slopes = flame.frames[frame_nr].contour_data.slopes_of_segmented_contours[contour_nr]
        contour_slope_changes = flame.frames[frame_nr].contour_data.slope_changes_of_segmented_contours[contour_nr]
        
        corrected_slopes = []
        
        for slope in contour_slopes:
            
            slope = np.abs(slope)
            
            if  slope >= 0.5:
                slope -= 1
            
            corrected_slopes.append(slope)
            
        final_segments = local_flame_speed_from_single_image_double_frame(image_nr, n_nearest_coords, threshold_angle)
        
        for final_segment in final_segments:
            
            i_segment = final_segment[0][14]
            contour_slopes = final_segment[0][-1]
            
            slope_change_avg = 0.0
            slope_change0 = 0.0
            slope_change1 = 0.0
            
            if i_segment > 0 and i_segment < len(contour_slope_changes):
                slope_change0 = -contour_slope_changes[i_segment-1]
                slope_change1 = -contour_slope_changes[i_segment]
                slope_change_avg = (slope_change0 + slope_change1)/2
            
            slope_list.append(slope_change)
            slope_change_avg_list.append(slope_change_avg)
            slope_change0_list.append(slope_change0)
            slope_change1_list.append(slope_change1)
            
    list_of_slope_list.append(slope_list)
    list_of_slope_change_avg_list.append(slope_change_avg_list)
    list_of_slope_change0_list.append(slope_change0_list)
    list_of_slope_change1_list.append(slope_change1_list)


# def calculate_tangential_divergence(final_segments_list, time_step):
#     tangential_velocity_list = []
#     tangential_divergence_list = []
    
#     for final_segments in final_segments_list:
#         V_t_list = []
        
#         for final_segment in final_segments:
#             V_t = final_segment[0][11]
#             V_t_list.append(V_t)
        
#         tangential_velocity = np.mean(V_t_list)
#         tangential_velocity_list.append(tangential_velocity)
        
#         tangential_divergence = np.mean(np.diff(V_t_list)) / dt
#         tangential_divergence_list.append(tangential_divergence)
        
#     return tangential_velocity_list, tangential_divergence_list

# tangential_velocity_list, tangential_divergence_list = calculate_tangential_divergence(list_of_final_segments, dt)






# list_of_Tangential_Divergence = []

# for i, case in enumerate(cases):
    
#     list_of_final_segments = []
#     Tangential_Divergence_list = []
    
#     for image_nr in progressbar.progressbar(image_nrs): 
        
#         contour_nr, n_windows_x, n_windows_y, x, y, u, v, velocity_abs = read_piv_data(image_nr)
#         contour_slopes = flame.frames[frame_nr].contour_data.slopes_of_segmented_contours[contour_nr]
#         contour_slope_changes = flame.frames[frame_nr].contour_data.slope_changes_of_segmented_contours[contour_nr]
        
#         corrected_slopes = []
        
#         for slope in contour_slopes:
            
#             slope = np.abs(slope)
            
#             if  slope >= 0.5:
#                 slope -= 1
            
#             corrected_slopes.append(slope)
            
#         final_segments = local_flame_speed_from_single_image_double_frame(image_nr, n_nearest_coords, threshold_angle)
        
#         for final_segment in final_segments:
            
#             i_segment = final_segment[0][14]
#             contour_slopes = final_segment[0][-1]
#             V_t = final_segment[0][11]
            
#             if i_segment > 0 and i_segment < len(contour_slope_changes):
#                 slope_change0 = -contour_slope_changes[i_segment-1]
#                 slope_change1 = -contour_slope_changes[i_segment]
#                 slope_change_avg = (slope_change0 + slope_change1)/2
#                 Tangential_Divergence = (slope_change1 - slope_change0) / (2 * V_t)
#             else:
#                 Tangential_Divergence = 0.0
                
#             Tangential_Divergence_list.append(Tangential_Divergence)
            
#     list_of_Tangential_Divergence.append(Tangential_Divergence_list)




list_of_tortuosity = []
list_of_acc = []

for i, case in enumerate(cases):
    
    image_tortuosity = []
    image_acc = []
    
    for image_nr in progressbar.progressbar(image_nrs): 
        
        # Read the PIV data and contour information
        contour_nr, n_windows_x, n_windows_y, x, y, u, v, velocity_abs = read_piv_data(image_nr)
        contour_slopes = flame.frames[frame_nr].contour_data.slopes_of_segmented_contours[contour_nr]
        contour_slope_changes = flame.frames[frame_nr].contour_data.slope_changes_of_segmented_contours[contour_nr]
        
        # Correct the slopes
        corrected_slopes = []
        
        for slope in contour_slopes:
            
            slope = np.abs(slope)
            
            if slope >= 0.5:
                slope -= 1
            
            corrected_slopes.append(slope)
            
        # Calculate the final segments
        final_segments = local_flame_speed_from_single_image_double_frame(image_nr, n_nearest_coords, threshold_angle)
        
        slope_change0_list = []
        
        for final_segment in final_segments:
            
            i_segment = final_segment[0][14]
            
            # Check if the segment is in the range of slope changes
            if i_segment > 0 and i_segment < len(contour_slope_changes):
                
                # Calculate the slope change
                slope_change0 = -contour_slope_changes[i_segment-1]
                
                # Append the slope change to the list
                slope_change0_list.append(slope_change0)
                
        # Calculate the tortuosity for the current image
        tortuosity = sum(abs(slope_change0) for slope_change0 in slope_change0_list)
        # Calculate the acc for the current image
        acc = sum((slope_change0) for slope_change0 in slope_change0_list)
        
        # Append the tortuosity to the list of image tortuosities
        image_tortuosity.append(tortuosity)
        image_acc.append(acc)
    
    # Append the list of image tortuosities to the list of all tortuosities
    list_of_tortuosity.append(image_tortuosity)
    list_of_acc.append(image_acc)




list_of_S_f_list = []

for i, case in enumerate(cases):
    
    list_of_final_segments = []
    S_f_list = []
    
    for image_nr in progressbar.progressbar(image_nrs): 
        
        final_segments = local_flame_speed_from_single_image_double_frame(image_nr, n_nearest_coords, threshold_angle)
        
        for final_segment in final_segments:
            
            S_f = final_segment[8]
            
            # if V_n > 10:
            #     print(image_nr)
            S_f_list.append(S_f)
        
        list_of_final_segments.append(final_segments)
        
    list_of_S_f_list.append(S_f_list)


list_of_V_n_list = []

for i, case in enumerate(cases):
    
    list_of_final_segments = []
    V_n_list = []
    
    for image_nr in progressbar.progressbar(image_nrs): 
        
        final_segments = local_flame_speed_from_single_image_double_frame(image_nr, n_nearest_coords, threshold_angle)
        
        for final_segment in final_segments:
            
            V_n = final_segment[0][8]
            
            # if V_n > 10:
            #     print(image_nr)
            V_n_list.append(V_n)
        
        list_of_final_segments.append(final_segments)
        
    list_of_V_n_list.append(V_n_list)




list_of_S_d_list = []

for i, case in enumerate(cases):
    
    list_of_final_segments = []
    S_d_list = []
    
    for image_nr in progressbar.progressbar(image_nrs): 
        
        final_segments = local_flame_speed_from_single_image_double_frame(image_nr, n_nearest_coords, threshold_angle)
        
        for final_segment in final_segments:
            
            S_d = final_segment[9]
            
            # if (S_d > 10).any():
            #     print(image_nr)
            S_d_list.append(S_d)
        
        list_of_final_segments.append(final_segments)
        
    list_of_S_d_list.append(S_d_list)

    
list_of_V_t_list = []

for i, case in enumerate(cases):
    
    list_of_final_segments = []
    V_t_list = []
    
    for image_nr in progressbar.progressbar(image_nrs): 
        
        final_segments = local_flame_speed_from_single_image_double_frame(image_nr, n_nearest_coords, threshold_angle)
        
        for final_segment in final_segments:
            
            V_t = final_segment[0][11]
            
            if V_t > 10:
                print(image_nr)
            V_t_list.append(V_t)
        
        list_of_final_segments.append(final_segments)
        
    list_of_V_t_list.append(V_t_list)
    

# list_of_grad_t_list = []

# for i, case in enumerate(cases):
    
#     list_of_final_segments = []
#     grad_t_list = []
    
#     for image_nr in progressbar.progressbar(image_nrs): 
        
#         final_segments = local_flame_speed_from_single_image_double_frame(image_nr, n_nearest_coords, threshold_angle)
        
#         for final_segment in final_segments:
            
#             V_t = final_segment[0][11]
#             dx_t = final_segment[0][3] - final_segment[0][1]
#             dy_t = final_segment[0][4] - final_segment[0][2]
            
#             grad_t = np.array([dx_t, dy_t])
#             grad_t_norm = np.linalg.norm(grad_t)
            
#             if grad_t_norm > 0:
#                 grad_t /= grad_t_norm
#                 grad_t *= V_t
                
#             grad_t_list.append(grad_t)
        
#         list_of_final_segments.append(final_segments)
        
#     list_of_grad_t_list.append(grad_t_list)

#%%% Tortuosity and Acc vs image numbers

# # Plotting
# fig, ax = plt.subplots()
# for i, tortuosity_list in enumerate(list_of_tortuosity):
#     ax.scatter(image_nrs, tortuosity_list, c='k', marker='+', label=f"Case {i}")
# ax.set_xlabel("Image number")
# ax.set_ylabel("Tortuosity")
# ax.legend()
# ax.grid()
# plt.show()

# # Plotting
# fig, ax = plt.subplots()
# for i, acc_list in enumerate(list_of_acc):
#     ax.scatter(image_nrs, acc_list, c='r', marker='+', label=f"Case {i}")
# ax.set_xlabel("Image number")
# ax.set_ylabel("Acc")
# ax.legend()
# ax.grid()
# plt.show()

# Plotting Doublee
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,8), sharex=True)

for i, tortuosity_list in enumerate(list_of_tortuosity):
    ax[0].scatter(image_nrs, tortuosity_list, c='k', marker='+', label=f"Case {i}")
ax[0].set_ylabel("Tortuosity")
ax[0].legend()
ax[0].grid()

for i, acc_list in enumerate(list_of_acc):
    ax[1].scatter(image_nrs, acc_list, c='r', marker='+', label=f"Case {i}")
ax[1].set_xlabel("Image number")
ax[1].set_ylabel("Acc")
ax[1].legend()
ax[1].grid()

plt.show()


#%%% S_f plot vs slope_change1_list

# create scatter plot for S_f vs Slope Change Average
fig, ax = plt.subplots()
ax.scatter(slope_change1_list, S_f_list, c='k', marker='+')
ax.set_xlabel('Slope Change Average (degrees)')
ax.set_ylabel('S_f (m/s)')
ax.set_title('Relationship between Slope Change due to Next Segment and Flame Speed')
ax.grid(True)

# add linear regression line
csf = np.polyfit(slope_change1_list, S_f_list, 1)
polysf = np.poly1d(csf)
ax.plot(slope_change1_list, polysf(slope_change1_list), 'r-', label='Linear regression')

ax.legend(loc='upper right', fontsize=12)
plt.show()

#%%% S_f plot vs slope_change_avg_list

# create scatter plot for S_f vs Slope Change Average
fig, ax = plt.subplots()
ax.scatter(slope_change_avg_list, S_f_list, c='k', marker='+')
ax.set_xlabel('Slope Change Average (degrees)')
ax.set_ylabel('S_f (m/s)')
ax.set_title('Relationship between Slope Change Average and Flame Speed')
ax.grid(True)

# add linear regression line
csf = np.polyfit(slope_change1_list, S_f_list, 1)
polysf = np.poly1d(csf)
ax.plot(slope_change_avg_list, polysf(slope_change_avg_list), 'r-', label='Linear regression')

ax.legend(loc='upper right', fontsize=12)
plt.show()


#%%% V_n plot vs slope_change1_list

# create scatter plot for V_n vs Slope Change Average
fig, ax = plt.subplots()
ax.scatter(slope_change1_list, V_n_list, c='k', marker='+')
ax.set_xlabel('Slope Change Average (degrees)')
ax.set_ylabel('V_n (m/s)')
ax.set_title('Relationship between Slope Change due to Next Segment and V_n')
ax.grid(True)
ax.legend(['Values'], loc='upper right', fontsize=12)

# add linear regression line
cvn = np.polyfit(slope_change1_list, V_n_list, 1)
polyvn = np.poly1d(cvn)
ax.plot(slope_change1_list, polyvn(slope_change1_list), 'r-', label='Linear regression')

ax.legend(loc='upper right', fontsize=12)
plt.show()

#%%% V_n plot vs slope_change_avg_list

# create scatter plot for V_n vs Slope Change Average
fig, ax = plt.subplots()
ax.scatter(slope_change_avg_list, V_n_list, c='k', marker='+')
ax.set_xlabel('Slope Change Average (degrees)')
ax.set_ylabel('V_n (m/s)')
ax.set_title('Relationship between Slope Change Average and Flame Speed')
ax.grid(True)
ax.legend(['Values'], loc='upper right', fontsize=12)

# add linear regression line
cvn = np.polyfit(slope_change_avg_list, V_n_list, 1)
polyvn = np.poly1d(cvn)
ax.plot(slope_change_avg_list, polyvn(slope_change_avg_list), 'r-', label='Linear regression')

ax.legend(loc='upper right', fontsize=12)
plt.show()
#%%% S_d plot vs slope_change1_list

# create scatter plot for V_n vs Slope Change Average
fig, ax = plt.subplots()
ax.scatter(slope_change1_list, S_d_list, c='k', marker='+')
ax.set_xlabel('Slope Change Average (degrees)')
ax.set_ylabel('S_d (m/s)')
ax.set_title('Relationship between Slope Change due to Next Segment and S_d')
ax.grid(True)
ax.legend(['Values'], loc='upper right', fontsize=12)

# add linear regression line
csd = np.polyfit(slope_change1_list, S_f_list, 1)
polysd = np.poly1d(csd)
ax.plot(slope_change1_list, polysd(slope_change1_list), 'r-', label='Linear regression')

ax.legend(loc='upper right', fontsize=12)
plt.show()

#%%% S_d plot vs slope_change_avg_list

# create scatter plot for V_n vs Slope Change Average
fig, ax = plt.subplots()
ax.scatter(slope_change_avg_list, S_d_list, c='k', marker='+')
ax.set_xlabel('Slope Change Average (degrees)')
ax.set_ylabel('S_d (m/s)')
ax.set_title('Relationship between Slope Change Average and S_d')
ax.grid(True)
ax.legend(['Values'], loc='upper right', fontsize=12)

# add linear regression line
csd = np.polyfit(slope_change_avg_list, S_f_list, 1)
polysd = np.poly1d(csd)
ax.plot(slope_change_avg_list, polysd(slope_change_avg_list), 'r-', label='Linear regression')

ax.legend(loc='upper right', fontsize=12)
plt.show()

#%%% V_t plot vs slope_change1_list


# create scatter plot for V_t vs Slope Change Average
fig, ax = plt.subplots()
ax.scatter(slope_change1_list, V_t_list, c='k', marker='+')
ax.set_xlabel('Slope Change Average (degrees)')
ax.set_ylabel('V_t (m/s)')
ax.set_title('Relationship between Slope Change due to Next Segment and V_t')
ax.grid(True)
ax.legend(['Values'], loc='upper right', fontsize=12)

# add linear regression line
cvt = np.polyfit(slope_change1_list, V_t_list, 1)
polyvt = np.poly1d(cvt)
ax.plot(slope_change1_list, polyvt(slope_change1_list), 'r-', label='Linear regression')

ax.legend(loc='upper right', fontsize=12)
plt.show()

#%%% V_t plot vs slope_change_avg_list


# create scatter plot for V_t vs Slope Change Average
fig, ax = plt.subplots()
ax.scatter(slope_change_avg_list, V_t_list, c='k', marker='+')
ax.set_xlabel('Slope Change Average (degrees)')
ax.set_ylabel('V_t (m/s)')
ax.set_title('Relationship between Slope Change Average and V_t')
ax.grid(True)
ax.legend(['Values'], loc='upper right', fontsize=12)

# add linear regression line
cvt = np.polyfit(slope_change_avg_list, V_t_list, 1)
polyvt = np.poly1d(cvt)
ax.plot(slope_change_avg_list, polyvt(slope_change_avg_list), 'r-', label='Linear regression')

ax.legend(loc='upper right', fontsize=12)
plt.show()       
#%%% Number of Interested Segments

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate the number of elements in each list
num_values = [len(l) for l in list_of_final_segments]

# Plot the number of elements for each image as a bar chart
ax.bar(np.arange(len(list_of_final_segments)), num_values, color='black')

# Add labels and legend
ax.set_xlabel('Image Number')
ax.set_ylabel('Number of Elements')
ax.grid(True)

# Add labels with the values of the number of elements
for i, num in enumerate(num_values):
    ax.text(i, num + 0.1, str(num), horizontalalignment='center')

# Set the x-tick labels to the image numbers
ax.set_xticklabels([f'Image {i+1}' for i in range(len(list_of_final_segments))])

plt.show()
#%%% S_F plot


# # Create a figure and axis object
# fig, ax = plt.subplots()

# # Plot a histogram with Seaborn and include a probability density curve (kde=True)
# data = S_f_list
# # sns.histplot(data, stat='count', kde=False, bins=30, ax=ax, color=google_blue) #stat='density'
# ax.hist(data, bins = 100, density=True, fc=google_blue, ec='k')

# # Set axis labels and plot title
# ax.set_xlabel(r'$S_{f}$ $[ms^{-1}]$')
# ax.set_ylabel(r'probability density')
# ax.set_ylabel('Count')

# # ax.set_title(title)
# ax.grid()
# # sns.set_style('ticks')

# S_f_average = np.mean(S_f_list)

# ax.axvline(S_f_average, color=google_red, marker='None', linestyle='--', linewidth=2, label= '$S_{f}=$' + '{0:.3f}'.format(S_f_average) + ' ms$^{-1}$')

# T_u, p_u, phi, H2_percentage = flame.T_lab, flame.p_lab, flame.phi, flame.H2_percentage
# premixed_flame = PremixedFlame(phi, H2_percentage, T_u, p_u)
# premixed_flame.solve_equations()
# ax.axvline(premixed_flame.S_L0, color=google_yellow, marker='None', linestyle='--', linewidth=2, label= '$S_{L0}=$' + '{0:.3f}'.format(premixed_flame.S_L0) + ' ms$^{-1}$')

# print('Average local flame speed: {0:.3f} m/s'.format(S_f_average)) 
# print("Unstretched laminar flame speed: {0:.3f} m/s".format(premixed_flame.S_L0)) 

# ax.legend()
# fig.tight_layout()
# plt.show()
#%%% S_F plot corrected


fig, ax = plt.subplots()

# Flatten the list of lists into a 1D list
flat_list = [val for sublist in list_of_S_f_list for val in sublist]

# Create a histogram with 100 bins and probability density scaling
ax.hist(flat_list, bins=100, density=True, fc=google_blue, ec='k')

# Set axis labels and plot title
ax.set_xlabel(r'$S_{f}$ $[ms^{-1}]$')
ax.set_ylabel(r'probability density')
ax.grid()

# Compute and plot the mean of list_of_S_f_list
S_f_average = np.mean(flat_list)
ax.axvline(S_f_average, color='red', linestyle='--', linewidth=2, label='$S_{f}=$' + '{0:.3f}'.format(S_f_average) + ' ms$^{-1}$')

# Compute and plot the unstretched laminar flame speed
T_u, p_u, phi, H2_percentage = flame.T_lab, flame.p_lab, flame.phi, flame.H2_percentage
premixed_flame = PremixedFlame(phi, H2_percentage, T_u, p_u)
premixed_flame.solve_equations()
ax.axvline(premixed_flame.S_L0, color='yellow', linestyle='--', linewidth=2, label='$S_{L0}=$' + '{0:.3f}'.format(premixed_flame.S_L0) + ' ms$^{-1}$')

# Add legend
ax.legend()

# Add title
plt.title('Distribution of Values in list_of_S_f_list')
print('S_f_avg: {0:.3f} m/s'.format(S_f_average))

# Show the plot
plt.show()




#%%% V_n plot


# Create a figure and axis object
fig, ax = plt.subplots()

# Plot a histogram with Seaborn and include a probability density curve (kde=True)
data = V_n_list
# sns.histplot(data, stat='count', kde=False, bins=30, ax=ax, color=google_blue) #stat='density'
ax.hist(data, bins = 100, density=True, fc=google_blue, ec='k')

# Set axis labels and plot title
ax.set_xlabel(r'$V_n$ $[ms^{-1}]$')
ax.set_ylabel(r'probability density')
# ax.set_ylabel('Count')

# ax.set_title(title)
ax.grid()
# sns.set_style('ticks')

V_n_average = np.mean(V_n_list)

ax.axvline(V_n_average, color=google_red, marker='None', linestyle='--', linewidth=2, label= '$V_n=$' + '{0:.3f}'.format(V_n_average) + ' ms$^{-1}$')

T_u, p_u, phi, H2_percentage = flame.T_lab, flame.p_lab, flame.phi, flame.H2_percentage
premixed_flame = PremixedFlame(phi, H2_percentage, T_u, p_u)
premixed_flame.solve_equations()
ax.axvline(premixed_flame.S_L0, color=google_yellow, marker='None', linestyle='--', linewidth=2, label= '$S_{L0}=$' + '{0:.3f}'.format(premixed_flame.S_L0) + ' ms$^{-1}$')

print('V_n_avg: {0:.3f} m/s'.format(V_n_average))
# print('Average local flame speed: {0:.3f} m/s'.format(S_f_average)) 
# print("Unstretched laminar flame speed: {0:.3f} m/s".format(premixed_flame.S_L0)) 

ax.legend()
fig.tight_layout()
plt.show()

#%%% S_d plot


# Create a figure and axis object
fig, ax = plt.subplots()

# Plot a histogram with Seaborn and include a probability density curve (kde=True)
data = np.ravel(S_d_list)
# sns.histplot(data, stat='count', kde=False, bins=30, ax=ax, color=google_blue) #stat='density'
ax.hist(data, bins = 100, density=True, fc=google_blue, ec='k')

# Set axis labels and plot title
ax.set_xlabel(r'$S_d$ $[ms^{-1}]$')
ax.set_ylabel(r'probability density')
# ax.set_ylabel('Count')

# ax.set_title(title)
ax.grid()
# sns.set_style('ticks')

S_d_average = np.mean(S_d_list)

ax.axvline(S_d_average, color=google_red, marker='None', linestyle='--', linewidth=2, label= '$S_d=$' + '{0:.3f}'.format(S_d_average) + ' ms$^{-1}$')

T_u, p_u, phi, H2_percentage = flame.T_lab, flame.p_lab, flame.phi, flame.H2_percentage
premixed_flame = PremixedFlame(phi, H2_percentage, T_u, p_u)
premixed_flame.solve_equations()
ax.axvline(premixed_flame.S_L0, color=google_yellow, marker='None', linestyle='--', linewidth=2, label= '$S_{L0}=$' + '{0:.3f}'.format(premixed_flame.S_L0) + ' ms$^{-1}$')

print('S_d_avg: {0:.3f} m/s'.format(S_d_average))
# print('Average local dispalacement speed: {0:.3f} m/s'.format(S_d_average)) 
# print("Unstretched laminar flame speed: {0:.3f} m/s".format(premixed_flame.S_L0)) 

ax.legend()
fig.tight_layout()
plt.show()

#%%% V_t plot


# Create a figure and axis object
fig, ax = plt.subplots()

# Plot a histogram with Seaborn and include a probability density curve (kde=True)
data = V_t_list
# sns.histplot(data, stat='count', kde=False, bins=30, ax=ax, color=google_blue) #stat='density'
ax.hist(data, bins = 100, density=True, fc=google_blue, ec='k')

# Set axis labels and plot title
ax.set_xlabel(r'$S_d$ $[ms^{-1}]$')
ax.set_ylabel(r'probability density')
# ax.set_ylabel('Count')

# ax.set_title(title)
ax.grid()
# sns.set_style('ticks')

V_t_average = np.mean(V_t_list)

ax.axvline(V_t_average, color=google_red, marker='None', linestyle='--', linewidth=2, label= '$V_t=$' + '{0:.3f}'.format(V_t_average) + ' ms$^{-1}$')

T_u, p_u, phi, H2_percentage = flame.T_lab, flame.p_lab, flame.phi, flame.H2_percentage
premixed_flame = PremixedFlame(phi, H2_percentage, T_u, p_u)
premixed_flame.solve_equations()
ax.axvline(premixed_flame.S_L0, color=google_yellow, marker='None', linestyle='--', linewidth=2, label= '$S_{L0}=$' + '{0:.3f}'.format(premixed_flame.S_L0) + ' ms$^{-1}$')

print('V_t: {0:.3f} m/s'.format(S_d_average))
print('Average V_t: {0:.3f} m/s'.format(V_t_average)) 
print("Unstretched laminar flame speed: {0:.3f} m/s".format(premixed_flame.S_L0)) 

ax.legend()
fig.tight_layout()
plt.show()

#%%% Tortuosity


# # Create a figure and axis object
# fig, ax = plt.subplots()

# # Plot a histogram with Seaborn and include a probability density curve (kde=True)
# data = list_of_tortuosity
# # sns.histplot(data, stat='count', kde=False, bins=30, ax=ax, color=google_blue) #stat='density'
# ax.hist(data, bins = 100, density=True, fc=google_blue, ec='k')

# # Set axis labels and plot title
# ax.set_xlabel(r'$Tortuosity$ ')
# ax.set_ylabel(r'probability density')
# # ax.set_ylabel('Count')

# # ax.set_title(title)
# ax.grid()
# # sns.set_style('ticks')

# t_average = np.mean(list_of_tortuosity)

# ax.axvline(t_average, color=google_red, marker='None', linestyle='--', linewidth=2, label= '$t=$' + '{0:.3f}'.format(t_average))

# # T_u, p_u, phi, H2_percentage = flame.T_lab, flame.p_lab, flame.phi, flame.H2_percentage
# # premixed_flame = PremixedFlame(phi, H2_percentage, T_u, p_u)
# # premixed_flame.solve_equations()
# # ax.axvline(premixed_flame.S_L0, color=google_yellow, marker='None', linestyle='--', linewidth=2, label= '$S_{L0}=$' + '{0:.3f}'.format(premixed_flame.S_L0) + ' ms$^{-1}$')

# # print('V_t: {0:.3f} m/s'.format(S_d_average))
# # print('Average V_t: {0:.3f} m/s'.format(V_t_average)) 
# # print("Unstretched laminar flame speed: {0:.3f} m/s".format(premixed_flame.S_L0)) 

# ax.legend()
# fig.tight_layout()
# plt.show()
