# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 12:33:00 2023

@author: aleks
"""

"""
Python script to plot chi2 contours for all pairs of variables
in the output file produced by XCLASS myXCLASSFit fit__{algorithm}__call_1.log.chi2

Saves a png version of the plot (by default, same location and name as the input file)
(needs to be run in the run/myXCLASSFit/[job_dir] directory because idk how to
to make it work otherwise.

Plus, I don't actually think the contour plot is that successful.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import sys
from matplotlib import ticker, cm
import matplotlib.ticker as ticker


# dirname = 'C:\\Users\\aleks\\OneDrive\\Dokumenty\\mphys\\chi contour\\'
dirname = ""
filename_genetic = 'fit__Genetic__call_1.log.chi2'
filename_lm = 'fit__LM__call_1.log.chi2'

def get_data(filename):
    file_path = dirname+ filename
    data = np.genfromtxt(file_path, skip_header=1)
    
    #move chi to the end
    #data[:, 1:] = np.roll(data[:, 1:], -1, 1)
    
    #get rid of nan-s
    data = data[~np.isnan(data).any(axis=1)]
    print(data)
    
    return data

def fmt(x, pos):
    """
    for contour plot, currently unused
    """
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def sort_data(data):
    """
    sort data into sep arrays

    """
    size1 = data[:,2]
    temp1 = data[:,3]
    colden1 = data[:,4]
    fwhm1 = data[:,5]
    size2 = data[:,6]
    temp2 = data[:,7]
    colden2 = data[:,8]
    fwhm2 = data[:,9]
    chi = data[:,1]
    return size1, temp1, colden1, fwhm1, size2, temp2, colden2, fwhm2, chi

def sort_meshgrid(size, temp, colden, fwhm):
    """
    creating possible meshgrids:
        size-temp
        size-colden
        size-fwhm
        
        temp-colden
        temp-fwhm
        
        colden-fwhm
    """
    X_st, Y_st = np.meshgrid(size, temp)
    X_sc, Y_sc = np.meshgrid(size, colden)
    X_sf, Y_sf = np.meshgrid(size, fwhm)
    
    X_tc, Y_tc = np.meshgrid(temp, colden)
    X_tf, Y_tf = np.meshgrid(temp, fwhm)
    
    X_cf, Y_cf = np.meshgrid(colden, fwhm)
    return  X_st, Y_st,  X_sc, Y_sc,  X_sf, Y_sf,  X_tc, Y_tc, X_tf, Y_tf, X_cf, Y_cf

def chi_grid(chi, x_var, y_var):
    "chi meshgrid, can use meshgrid arrays for x and y len"
    Z = np.zeros([len(x_var),len(y_var)], dtype = float)

    for i in range(len(Z)):
        Z[i,i] = chi[i]
        
    return Z
        

def one_comp_contour_array(x_mesh_size_c, x_mesh_size_t, x_mesh_size_f, x_mesh_temp_c,
                           x_mesh_temp_f, x_mesh_colden_f, 
                           y_mesh_size_c, y_mesh_size_t, y_mesh_size_f, y_mesh_temp_c,
                           y_mesh_temp_f, y_mesh_colden_f,
                           chi, title_name):
    
    # z = chi_grid(x_mesh_size_c, y_mesh_size_c)
    fig = plt.figure(figsize=(9, 10), dpi=300)
    # fig.figsize=(450, 300)
    fig.suptitle(title_name)
    
    plot_st = plt.subplot2grid((3, 3), (0, 0))
    plot_sc = plt.subplot2grid((3, 3), (1, 0))
    plot_sf = plt.subplot2grid((3, 3), (2, 0))
    plot_tc = plt.subplot2grid((3, 3), (1, 1))
    plot_tf = plt.subplot2grid((3, 3), (2, 1))
    plot_cf = plt.subplot2grid((3, 3), (2, 2))
    
    
    # plot_st.set_xlabel(r'First $T_1$ [K]')
    plot_st.set_ylabel(r'temp [K]')
    plot_st.contour(x_mesh_size_t, y_mesh_size_t, chi_grid(chi, x_mesh_size_t, y_mesh_size_t), 5)

    # plot_sc.set_xlabel(r'First $T_1$ [K]')
    plot_sc.set_ylabel(r'colden [cm$^{-2}$]')
    plot_sc.contour(x_mesh_size_c, y_mesh_size_c, chi_grid(chi, x_mesh_size_c, y_mesh_size_c), 5)

    plot_sf.set_xlabel(r'size [arcmin]')
    plot_sf.set_ylabel(r'fwhm [kms$^{-1}$')
    plot_sf.contour(x_mesh_size_f, y_mesh_size_f, chi_grid(chi, x_mesh_size_f, y_mesh_size_f),5)

    # plot_tc.set_xlabel(r'')
    # plot_tc.set_ylabel(r' $T_2$ [K]')
    plot_tc.contour(x_mesh_temp_c, y_mesh_temp_c, chi_grid(chi, x_mesh_temp_c, y_mesh_temp_c),5)
    
    plot_tf.set_xlabel(r'temp [K]')
    # plot_tf.set_ylabel(r' $T_2$ [K]')
    plot_tf.contour(x_mesh_temp_f, y_mesh_temp_f, chi_grid(chi, x_mesh_temp_f, y_mesh_temp_f),5)
    
    plot_cf.set_xlabel(r'colden [cm$^{-2}$]')
    # plot_cf.set_ylabel(r' $T_2$ [K]')
    plot_cf.contour(x_mesh_colden_f, y_mesh_colden_f, chi_grid(chi, x_mesh_colden_f, y_mesh_colden_f),5)
    
    #fig.show()
    fig.savefig(title_name)
    
    return None



data_genetic = get_data(filename_genetic)
data_lm = get_data(filename_lm)

size_genetic_1, temp_genetic_1, colden_genetic_1, fwhm_genetic_1, size_genetic_2, temp_genetic_2, colden_genetic_2, fwhm_genetic_2, chi_genetic = sort_data(data_genetic)

#1st comp
X_st_g, Y_st_g,  X_sc_g, Y_sc_g,  X_sf_g, Y_sf_g,  X_tc_g, Y_tc_g, X_tf_g, Y_tf_g, X_cf_g, Y_cf_g = sort_meshgrid(size_genetic_1, temp_genetic_1, colden_genetic_1, fwhm_genetic_1)

one_comp_contour_array(X_st_g, X_sc_g, X_sf_g, X_tc_g, X_tf_g, X_cf_g, 
                       Y_st_g, Y_sc_g, Y_sf_g, Y_tc_g, Y_tf_g, Y_cf_g, chi_genetic, "Genetic chi distribution, 1st comp.")

#2nd comp
X_st_g, Y_st_g,  X_sc_g, Y_sc_g,  X_sf_g, Y_sf_g,  X_tc_g, Y_tc_g, X_tf_g, Y_tf_g, X_cf_g, Y_cf_g = sort_meshgrid(size_genetic_2, temp_genetic_2, colden_genetic_2, fwhm_genetic_2)

one_comp_contour_array(X_st_g, X_sc_g, X_sf_g, X_tc_g, X_tf_g, X_cf_g, 
                       Y_st_g, Y_sc_g, Y_sf_g, Y_tc_g, Y_tf_g, Y_cf_g, chi_genetic, "Genetic chi distribution, 2nd comp.")



#LM
size_lm_1, temp_lm_1, colden_lm_1, fwhm_lm_1, size_lm_2, temp_lm_2, colden_lm_2, fwhm_lm_2, chi_lm = sort_data(data_lm)

#1st comp
X_st_l, Y_st_l,  X_sc_l, Y_sc_l,  X_sf_l, Y_sf_l,  X_tc_l, Y_tc_l, X_tf_l, Y_tf_l, X_cf_l, Y_cf_l = sort_meshgrid(size_lm_1, temp_lm_1, colden_lm_1, fwhm_lm_1)

one_comp_contour_array(X_st_l, X_sc_l, X_sf_l, X_tc_l, X_tf_l, X_cf_l, 
                       Y_st_l, Y_sc_l, Y_sf_l, Y_tc_l, Y_tf_l, Y_cf_l, chi_lm, "LM chi distribution, 1st comp.")

#2nd comp
X_st_l, Y_st_l,  X_sc_l, Y_sc_l,  X_sf_l, Y_sf_l,  X_tc_l, Y_tc_l, X_tf_l, Y_tf_l, X_cf_l, Y_cf_l = sort_meshgrid(size_lm_2, temp_lm_2, colden_lm_2, fwhm_lm_2)

one_comp_contour_array(X_st_l, X_sc_l, X_sf_l, X_tc_l, X_tf_l, X_cf_l, 
                       Y_st_l, Y_sc_l, Y_sf_l, Y_tc_l, Y_tf_l, Y_cf_l, chi_lm, "LM chi distribution, 2nd comp.png")

# X, Y = np.meshgrid(data_genetic[:,2], data_genetic[:,3])

# Z = np.zeros([len(X),len(Y)], dtype = float)

# ZZ = np.linspace(0, len(Z)-1, len(Z))

# print(len(ZZ))

# # for x in data[:,-1]:
# #     print("We're on time %d" % (x))

# # print([Z[i][i] for i in range(len(Z))])

# chi = data_genetic[:,1]


# for i in range(len(Z)):
#     Z[i,i] = chi[i]
    
# print(Z)



# plot1 = plt.subplot2grid((3, 3), (0, 0), colspan=1)
# plot2 = plt.subplot2grid((3, 3), (1, 0), rowspan=1, colspan=1)
# plot3 = plt.subplot2grid((3, 3), (2, 0))



# # plt.title(r'$N_{2}$ fit')
# # plt.xlabel(r'First temperature component $T_1$ [K]')
# # plt.ylabel(r'Second temperature component $T_2$ [K]')
# # plt.contour(X, Y, Z)

# plot1.set_xlabel(r'First $T_1$ [K]')
# plot1.set_ylabel(r' $T_2$ [K]')
# plot1.contour(X, Y, Z)


# plot2.set_xlabel(r' $T_1$ [K]')
# plot2.set_ylabel(r'Second  $T_2$ [K]')
# plot2.contour(Y, Y, Z)

# plot3.set_xlabel(r' $T_1$ [K]')
# plot3.set_ylabel(r' $T_2$ [K]')
# plot3.contour(Y, Y, Z)


