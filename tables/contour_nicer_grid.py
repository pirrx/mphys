# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:38:24 2023

@author: aleks

works similarly to the prev contour plot but BETTER lol
sorry for keeping the index system i dont want to re-work how the 
adjust the "populate_mesh_grid" function (the for loop) to adjust the size 
of the grid (0,1,2,3 are basically four rows of the 4x4 matrix,
             the nr of columns does not matter)
its a sort of a very basic code because theres a nicer way of doing it withi think
np.ravel, but this is quciker for now.
adjust the scales accordingly etc, use LogNorm or SymLogNorm or whatever for column density and so on




"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, CenteredNorm
from matplotlib import ticker, cm
import matplotlib.ticker as ticker
from path_file import TABLES_PATH




# FILENAMES = ["radex_e14.csv", "radex_e15.csv", "radex_e16.csv", "radex_e17.csv"]
FILENAMES = ["lte_e14.csv", "lte_e15.csv", "lte_e16.csv", "lte_e17.csv"]

N_SINGLE_T_FITTING = []

FILENAMES = [TABLES_PATH + file for file in FILENAMES]





def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def data_array(filename, index_generated, index_fitted, index_changed):
    data = np.genfromtxt(filename, delimiter=';',
                     comments='%')

    GENERATED = data[:, index_generated]
    FITTED = data[:, index_fitted]
    UPPER = data[:, (index_fitted+1)]
    LOWER = data[:, (index_fitted+2)]

    CHANGED = data[:, index_changed]
    
    return GENERATED, CHANGED, FITTED, UPPER, LOWER


def populate_mesh_grid(x_var, y_var, var_14, var_15, var_16, var_17):
    "chi meshgrid, can use meshgrid arrays for x and y len"
    Z = np.full([len(x_var),len(y_var)], 0.0, dtype = float)

    for i in range(4):
        Z[0,i] = var_14[i]
        Z[1,i] = var_15[i]
        Z[2,i] = var_16[i]
        Z[3,i] = var_17[i]
        
    return Z

"""   CHANGE TO TEMP AS NECESSARY """
X = [1e14, 1e15, 1e16, 1e17]
Y = [1e14, 1e15, 1e16, 1e17]


"""
[generated, fitted]
t1 = 0, 8,
t2 = 1, 11
n1 = 2, 14
n2 = 3, 17
f1 = 4, 20
f2 = 5, 23
s1 = 6, 26
s2 = 7, 29
chi = 32
s t = 33
s n = 36
s f = 39
s s = 42
s chi = 45

temp => changed = 1
colden => changed = 3

"""

""" TECHINCALLY SO FAR ONLY FITTED & single INDEX MATTERS """

index_generated = 3 #comp 2 generated data (n2 or t2 etc)
index_generated_n1 = 2 #other generated data (n1 or t1 etc)
index_fitted = 17 #comp 2 variable for fitting
index_fitted_n1 = 14 #_1 variable for fitting
index_changed = 1 #whats gonna be on the x axis

index_single = 36

# t_index_generated = 3
# t_index_generated_t1 = 2
# t_index_fitted = 17
# t_index_fitted_t1 = 14
# t_index_changed = 1




#TRY
GENERATED_e14, CHANGED_e14, FITTED_e14, UPPER_e14, LOWER_e14 = data_array(FILENAMES[0], index_generated, index_fitted, index_changed)
GENERATED_e14_N1, CHANGED_e14_N1, FITTED_e14_N1, UPPER_e14_N1, LOWER_e14_N1 = data_array(FILENAMES[0], index_generated_n1, index_fitted_n1, index_changed)
GENERATED_e14_single, CHANGED_e14_single, FITTED_e14_single, UPPER_e14_single, LOWER_e14_single = data_array(FILENAMES[0], index_generated_n1, index_single, index_changed)

GENERATED_e15, CHANGED_e15, FITTED_e15, UPPER_e15, LOWER_e15 = data_array(FILENAMES[1], index_generated, index_fitted, index_changed)
GENERATED_e15_N1, CHANGED_e15_N1, FITTED_e15_N1, UPPER_e15_N1, LOWER_e15_N1 = data_array(FILENAMES[1], index_generated_n1, index_fitted_n1, index_changed)
GENERATED_e15_single, CHANGED_e15_single, FITTED_e15_single, UPPER_e15_single, LOWER_e15_single = data_array(FILENAMES[1], index_generated_n1, index_single, index_changed)

GENERATED_e16, CHANGED_e16, FITTED_e16, UPPER_e16, LOWER_e16 = data_array(FILENAMES[2], index_generated, index_fitted, index_changed)
GENERATED_e16_N1, CHANGED_e16_N1, FITTED_e16_N1, UPPER_e16_N1, LOWER_e16_N1 = data_array(FILENAMES[2], index_generated_n1, index_fitted_n1, index_changed)
GENERATED_e16_single, CHANGED_e16_single, FITTED_e16_single, UPPER_e16_single, LOWER_e16_single = data_array(FILENAMES[2], index_generated_n1, index_single, index_changed)

GENERATED_e17, CHANGED_e17, FITTED_e17, UPPER_e17, LOWER_e17 = data_array(FILENAMES[3], index_generated, index_fitted, index_changed)
GENERATED_e17_N1, CHANGED_e17_N1, FITTED_e17_N1, UPPER_e17_N1, LOWER_e17_N1 = data_array(FILENAMES[3], index_generated_n1, index_fitted_n1, index_changed)
GENERATED_e17_single, CHANGED_e17_single, FITTED_e17_single, UPPER_e17_single, LOWER_e17_single = data_array(FILENAMES[3], index_generated_n1, index_single, index_changed)


# ZZ_N2 = []
# ZZ_N2.extend(FITTED_e14)
# ZZ_N2.extend(FITTED_e15)
# ZZ_N2.extend(FITTED_e16)
# ZZ_N2.extend(FITTED_e17)

# ZZ_N1 = []
# ZZ_N1.extend(FITTED_e14_N1)
# ZZ_N1.extend(FITTED_e15_N1)
# ZZ_N1.extend(FITTED_e16_N1)
# ZZ_N1.extend(FITTED_e17_N1)

# ZZ_NS = []
# ZZ_NS.extend(FITTED_e14_single)
# ZZ_NS.extend(FITTED_e15_single)
# ZZ_NS.extend(FITTED_e16_single)
# ZZ_NS.extend(FITTED_e17_single)

#create a meshgrid from our data
Z_Nsingle = populate_mesh_grid(X, Y, FITTED_e14_single, FITTED_e15_single, FITTED_e16_single, FITTED_e17_single)
Z_N1 = populate_mesh_grid(X, Y, FITTED_e14_N1, FITTED_e15_N1, FITTED_e16_N1, FITTED_e17_N1)
Z_N2 = populate_mesh_grid(X, Y, FITTED_e14, FITTED_e15, FITTED_e16, FITTED_e17)

# create axes meshgrid
N_mesh_2, N_mesh_1 = np.meshgrid(X, Y)
"""
^^^ i am frankly unsure on how the axes should be set up but this seems correct
"""

print(N_mesh_1)
print(N_mesh_2)
# Z_N2[1,2]=55.
print(Z_N1)
print(Z_N2)
print(Z_Nsingle)




# GENERALLY: UNLESS PLOTTING COLDEN, LOGSPACE LEVELS (JUST BELOW) 
# AND set_yscale, set_xscale can be commented x- and -axis is not colden
#remember to change the levels
# also ignore the format=ticker thingy if not log
# remember to change the main title and graph names
# make a judgement whether there needs to be one or many colorbars

title_name = "Column density $N$"

title_1 = r'$N_1$'
title_2 = r'$N_2$'
title_3 = r'$N_{\mathrm{single}}$'

# title_1 = r'$T_1$ ($T_1$ = 250K)'
# title_2 = r'$T_2$ ($T_2$ = 100K)'
# title_3 = r'$T_{\mathrm{single}}$'

# title_1 = r'$F_1$ ($F_1$ = 2)'
# title_2 = r'$F_2$ ($F_2$ = 7)'
# title_3 = r'$F_{\mathrm{single}}$'

# title_1 = r'$s_1$ ($s_1$ = 0.7")'
# title_2 = r'$s_2$ ($s_2$ = 0.7")'
# title_3 = r'$s_{\mathrm{single}}$'

levels = np.logspace(14, 17, 15)
# levels = np.linspace(50, 300, 25)
# levels = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
# levels = np.linspace(1, 10, 21)
# levels_1 = np.linspace(1,4, 22)
# levels_2 = np.linspace(4, 8, 22)
# levels = np.linspace(0.3, 1.3, 20)

#-------------------------------------------

# fig = plt.figure(figsize=(20, 5), dpi=400)
fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,5), constrained_layout=True)
plot_1, plot_2, plot_3 = axes.flatten()

fig.suptitle(title_name, fontsize=20)
# fig.tight_layout()

# plot_1 = plt.subplot2grid((1, 5), (0, 0))
# plot_2 = plt.subplot2grid((1, 5), (0, 2))
# plot_3 = plt.subplot2grid((1, 5), (0, 4))

# figsize=(20,5)
plot_1.set_title(title_1, fontsize=15)
plot_1.set_xlabel(r'N$_1$ [cm$^{-2}$]')
plot_1.set_ylabel(r'N$_2$ [cm$^{-2}$]')
plot_1.set_xscale('log')
plot_1.set_yscale('log')
grid_1 = plot_1.contourf(N_mesh_1, N_mesh_2, Z_N1, levels) #cmap=r'plasma_r', norm=SymLogNorm(linthresh=1, linscale=1))
# fig.colorbar(grid_1, ax=plot_1)

plot_2.set_title(title_2, fontsize=15)
plot_2.set_xlabel(r'N$_1$ [cm$^{-2}$]')
# plot_2.set_ylabel(r'N$_2$ [cm$^{-2}$]')
plot_2.set_xscale('log')
plot_2.set_yscale('log')
grid_2 = plot_2.contourf(N_mesh_1, N_mesh_2, Z_N2, levels) #cmap=r'plasma_r', norm=SymLogNorm(linthresh=1, linscale=1))
# fig.colorbar(grid_2, ax=plot_2)

plot_3.set_title(title_3, fontsize=15)
plot_3.set_xlabel(r'N$_1$ [cm$^{-2}$]')
# plot_3.set_ylabel(r'N$_2$ [cm$^{-2}$]')
plot_3.set_xscale('log')
plot_3.set_yscale('log')
grid_3 = plot_3.contourf(N_mesh_1, N_mesh_2, Z_Nsingle, levels) #cmap=r'plasma_r', norm=SymLogNorm(linthresh=1, linscale=1))
fig.colorbar(grid_3, ax=plot_3, format=ticker.FuncFormatter(fmt))



# plt.yscale('log')
# plt.xscale('log')
# plt.title(r'N$_{single}$', fontsize=10)
# plt.xlabel(r'First column density component N$_1$ [cm$^{-2}$]')
# plt.ylabel(r'Second column density component N$_2$ [cm$^{-2}$]')
# # plt.ticks()
# # plt.ylabel(variable_name)
# plt.contourf(N_mesh_1, N_mesh_2, Z_N2, 20,  cmap='RdBu_r')
# plt.colorbar(format=ticker.FuncFormatter(fmt))
# # vmin=1e14, vmax=1e17
# # format=ticker.FuncFormatter(fmt)
# # % distance from generated values
# # norm=LogNorm(), levels=levels
# # cmap='viridis'
# # cmap='RdBu_r'
# #SymLogNorm(linthresh=0.03, linscale=1)

