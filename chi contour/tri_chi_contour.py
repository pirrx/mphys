# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:10:01 2023

@author: aleks
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm, SymLogNorm, CenteredNorm

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
    # print(data)
    
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


def one_comp_tricontour_array(size, temp, colden, fwhm, chi, title_name, limit):
    
    fig = plt.figure(figsize=(24, 16), dpi=300)
    fig.suptitle(title_name, fontsize=50)
    # fig.tight_layout()
    
    # for i in range(len(chi)):
    #     if chi[i] > 5:
    #         chi[i] = np.nan
            
    size = np.delete(size, np.where(chi > 2))    
    temp = np.delete(temp, np.where(chi > 2))    
    colden = np.delete(colden, np.where(chi > 2))    
    fwhm = np.delete(fwhm, np.where(chi > 2))    
    chi = np.delete(chi, np.where(chi > 2))    
    
    # orig_map=plt.cm.get_cmap('viridis')
    # # reversing the original colormap using reversed() function
    # reversed_map = orig_map.reversed()
    
    plot_st = plt.subplot2grid((3, 3), (0, 0))
    plot_sc = plt.subplot2grid((3, 3), (1, 0))
    plot_sf = plt.subplot2grid((3, 3), (2, 0))
    plot_tc = plt.subplot2grid((3, 3), (1, 1))
    plot_tf = plt.subplot2grid((3, 3), (2, 1))
    plot_cf = plt.subplot2grid((3, 3), (2, 2))
    
    plot_st.set_ylabel(r'temp [K]')
    tri_st = plot_st.tricontourf(size, temp, chi, 20, vmin=0, vmax=limit)#cmap=r'plasma_r', norm=SymLogNorm(linthresh=1, linscale=1))
    fig.colorbar(tri_st, ax=plot_st, format=ticker.FuncFormatter(fmt))

    plot_sc.set_ylabel(r'colden [cm$^{-2}$]')
    tri_sc = plot_sc.tricontourf(size, colden, chi, 20, vmin=0, vmax=limit)#cmap='plasma_r', norm=SymLogNorm(linthresh=1, linscale=1))
    fig.colorbar(tri_sc, ax=plot_sc, format=ticker.FuncFormatter(fmt))

    plot_sf.set_xlabel(r'size [arcmin]')
    plot_sf.set_ylabel(r'fwhm [kms$^{-1}$')
    tri_sf = plot_sf.tricontourf(size, fwhm, chi, 20, vmin=0, vmax=limit)#cmap='plasma_r', norm=SymLogNorm(linthresh=1, linscale=1))
    fig.colorbar(tri_sf, ax=plot_sf, format=ticker.FuncFormatter(fmt))

    tri_tc = plot_tc.tricontourf(temp, colden, chi, 20, vmin=0, vmax=limit)#cmap='plasma_r', norm=SymLogNorm(linthresh=1, linscale=1))
    fig.colorbar(tri_tc, ax=plot_tc, format=ticker.FuncFormatter(fmt))
    
    plot_tf.set_xlabel(r'temp [K]')
    tri_tf = plot_tf.tricontourf(temp, fwhm, chi, 20, vmin=0, vmax=limit)#cmap='plasma_r', norm=SymLogNorm(linthresh=1, linscale=1))
    fig.colorbar(tri_tf, ax=plot_tf, format=ticker.FuncFormatter(fmt))
    
    plot_cf.set_xlabel(r'colden [cm$^{-2}$]')
    tri_cf = plot_cf.tricontourf(colden, fwhm, chi, 20, vmin=0, vmax=limit)#cmap='plasma_r', norm=SymLogNorm(linthresh=1, linscale=1))
    fig.colorbar(tri_cf, ax=plot_cf, format=ticker.FuncFormatter(fmt))
    
    # plt.colorbar(format=ticker.FuncFormatter(fmt))
    
    #fig.show()
    fig.savefig(title_name)
    
    return None




data_genetic = get_data(filename_genetic)
data_lm = get_data(filename_lm)

size_genetic_1, temp_genetic_1, colden_genetic_1, fwhm_genetic_1, size_genetic_2, temp_genetic_2, colden_genetic_2, fwhm_genetic_2, chi_genetic = sort_data(data_genetic)
size_lm_1, temp_lm_1, colden_lm_1, fwhm_lm_1, size_lm_2, temp_lm_2, colden_lm_2, fwhm_lm_2, chi_lm = sort_data(data_lm)


# GENETIC PLOT
one_comp_tricontour_array(size_genetic_1, temp_genetic_1, colden_genetic_1, fwhm_genetic_1, chi_genetic, "Genetic chi distribution, 1st comp", 2)

one_comp_tricontour_array(size_genetic_2, temp_genetic_2, colden_genetic_2, fwhm_genetic_2, chi_genetic, "Genetic chi distribution, 2nd comp", 2)

# LM PLOT
one_comp_tricontour_array(size_lm_1, temp_lm_1, colden_lm_1, fwhm_lm_1, chi_lm, "LM chi distribution, 1st comp", 0.05)

one_comp_tricontour_array(size_lm_2, temp_lm_2, colden_lm_2, fwhm_lm_2, chi_lm, "LM chi distribution, 2nd comp", 0.05)






