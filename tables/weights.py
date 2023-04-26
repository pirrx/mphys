# -*- coding: utf-8 -*-
"""
@author: aleks
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

#FILENAMES = ["e14.csv", "e15.csv", "e16.csv", "e17.csv", "low.csv", "high.csv"]

FILENAMES = ["radex_e14.csv", "radex_e15.csv", "radex_e16.csv", "radex_e17.csv"]


comp_1_n = [1e14, 1e15, 1e16, 1e17]
# comp_2_n = [1e14, 5e14, 1e15, 5e15, 1e16, 5e16, 1e17]
comp_2_n = comp_1_n

comp_1_t = [250, 250, 250, 250]
# comp_2_t = [100, 100, 100, 100, 100, 100, 100]
comp_2_t = [100, 100, 100, 100, 100]

comp_1_fwhm = [2, 2, 2, 2]
# comp_2_fwhm = [7, 7, 7, 7, 7, 7, 7]
comp_2_fwhm = [7, 7, 7, 7, 7]

comp_1_s = [0.7, 0.7, 0.7, 0.7]
comp_2_s = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
comp_2_s = comp_1_s

t1 = 250
t2 = 100

fwhm1 = 2
fwhm2 = 7

s = 0.7

# max_intensity_1 = [1.246, 11.831, 74.01, 107.81]
# max_intensity_2 = [0.906, 4.339, 8.226, 27.75, 36.95, 41.51, 41.51]

single_n = [1.86e14, 3.96e14, 8.23e14, 1.01e16, 5.71e16, 4.15e14, 9.18e14, 4.93e15, 9.94e15, 9.96e16,
            7.84e15, 1.13e16, 3.60e16, 3.47e16, 6.18e16, 6.67e16]

single_t = [128, 107, 104, 101, 100, 108, 133, 102, 101, 100, 179, 161, 185, 224, 244, 192]

single_fwhm = [4.15, 6.22, 6.59, 6.96, 8.38, 6.21, 4.06, 6.92, 6.99, 7.00, 6.77, 3.44, 6.96, 8.05, 3.29, 6.49]

single_s = [0.59, 1.08, 0.88, 0.7, 0.8, 0.93, 1.29, 0.71, 0.7, 0.7, 1.12, 1.04, 0.68, 0.86, 0.89, 0.84]

og_data_n = [[1e14, 1e14, 1e14, 1e14, 1e14, 1e15, 1e15, 1e15, 1e15, 1e15, 1e16, 1e16, 1e16, 1e16, 1e17, 1e17],
           [1e14, 5e14, 1e15, 1e16, 1e17, 5e14, 1e15, 5e15, 1e16, 1e17, 5e15, 1e16, 5e16, 1e17, 5e16, 1e17]]

og_data_t = [[250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
           [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]

og_data_fwhm = [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]]

og_data_s = [[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
           [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]]



# $0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'

def axvlines(xs, ax=None, lims=None, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param ax: The axis (or none to use gca)
    :param lims: Optionally the (ymin, ymax) of the lines
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    if lims is None:
        lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scaley = False, **plot_kwargs)
    return plot


def data_array(filename, index_fitted):
    data = np.genfromtxt(filename, delimiter=';',
                     comments='%')

    FITTED = data[:, index_fitted]
    UPPER = data[:, (index_fitted+1)]
    LOWER = data[:, (index_fitted+2)]
    CHI_S = data[:, -1]
    
    return FITTED, UPPER, LOWER, CHI_S

"""
indices:
0 - e14 e14
1 - e14 5e14
2 - e14 e15
3 - e14 e16
4 - e14 e17
5 - e15 5e14
6 - e15 e15
7 - e15 5e15
8 - e15 e16
9 - e15 e17
10 - e16 5e15
11 - e16 e16
12 - e16 5e16 
13 - e16 17
14 - e17 5e16
15 - e17 e17

"""

def mean_value(a,b):
    return (a+b)/2

def weighted_mean(a, b, intensity1, intensity2):
    return (a*(intensity1) + b*(intensity2))/((intensity1)+(intensity2))

# def weighted_mean(a, b, intensity1, intensity2):
#     return (a*np.log(intensity1) + b*np.log(intensity2))/(np.log(intensity1)+np.log(intensity2))

# comp_1 = comp_1_n
# comp_2 = comp_2_n
# og_data = og_data_n
# single = single_n
# variable_name = 'column density' + r'[cm$^{-2}]$'

# comp_1 = comp_1_t
# comp_2 = comp_2_t
# og_data = og_data_t
# single = single_t
# variable_name = 'temperature [K]'

comp_1 = comp_1_fwhm
comp_2 = comp_2_fwhm
og_data = og_data_fwhm
single = single_fwhm
variable_name = 'FWHM'

# comp_1 = comp_1_s
# comp_2 = comp_2_s
# og_data = og_data_s
# single = single_s
# variable_name = 'size ["]'



"""
[fitted]
s t = 33
s n = 36
s f = 39
s s = 42
s chi = 45

# temp => changed = 1
# colden => changed = 3

"""

# sam eindexing as in with residuals
index_fitted = 39 #_2 variable for fitting

FITTED_e14, UPPER_e14, LOWER_e14, CHI_e14 = data_array(FILENAMES[0], index_fitted)
FITTED_e15, UPPER_e15, LOWER_e15, CHI_e15 = data_array(FILENAMES[1], index_fitted)
FITTED_e16, UPPER_e16, LOWER_e16, CHI_e16 = data_array(FILENAMES[2], index_fitted)
FITTED_e17, UPPER_e17, LOWER_e17, CHI_e17 = data_array(FILENAMES[3], index_fitted)


#only use lower and upper for uncertainties, ignore the rest
#or can modify the code to use those instead of arrays defined at the top
#but maybe later

LOWER = []
LOWER.extend(LOWER_e14)
LOWER.extend(LOWER_e15)
LOWER.extend(LOWER_e16)
LOWER.extend(LOWER_e17)


UPPER = []
UPPER.extend(UPPER_e14)
UPPER.extend(UPPER_e15)
UPPER.extend(UPPER_e16)
UPPER.extend(UPPER_e17)

CHI = []
CHI.extend(CHI_e14)
CHI.extend(CHI_e15)
CHI.extend(CHI_e16)
CHI.extend(CHI_e17)


print(LOWER)
print(UPPER)



WEIGHTED_N = []

# WEIGHTED_N = [weighted_mean(comp_1[0], comp_2[0], max_intensity_1[0], max_intensity_2[0]), 
#               (weighted_mean(comp_1[0], comp_2[1], max_intensity_1[0], max_intensity_2[1])),
#               (weighted_mean(comp_1[0], comp_2[2], max_intensity_1[0], max_intensity_2[2])), 
#               (weighted_mean(comp_1[0], comp_2[4], max_intensity_1[0], max_intensity_2[4])),
#               (weighted_mean(comp_1[0], comp_2[6], max_intensity_1[0], max_intensity_2[6])),
#               (weighted_mean(comp_1[1], comp_2[1], max_intensity_1[1], max_intensity_2[1])), 
#               (weighted_mean(comp_1[1], comp_2[2], max_intensity_1[1], max_intensity_2[2])),
#               (weighted_mean(comp_1[1], comp_2[3], max_intensity_1[1], max_intensity_2[3])),
#               (weighted_mean(comp_1[1], comp_2[4], max_intensity_1[1], max_intensity_2[4])),
#               (weighted_mean(comp_1[1], comp_2[6], max_intensity_1[1], max_intensity_2[6])),
#               (weighted_mean(comp_1[2], comp_2[3], max_intensity_1[2], max_intensity_2[3])),
#               (weighted_mean(comp_1[2], comp_2[4], max_intensity_1[2], max_intensity_2[4])),
#               (weighted_mean(comp_1[2], comp_2[5], max_intensity_1[2], max_intensity_2[5])),
#               (weighted_mean(comp_1[2], comp_2[6], max_intensity_1[2], max_intensity_2[6])),
#               (weighted_mean(comp_1[3], comp_2[5], max_intensity_1[3], max_intensity_2[5])),
#               (weighted_mean(comp_1[3], comp_2[6], max_intensity_1[3], max_intensity_2[6]))]

labels = [r'$10^{14}, 10^{14}$' + '\n' + r'$\chi^2_R = $' + str(CHI[0]), 
          r'$10^{14}, 5\times10^{14}$' +' \n' + r'$\chi^2_R = $' + str(CHI[1]), 
          r'$10^{14}, 10^{15}$' + '\n' + r'$\chi^2_R = $' + str(CHI[2]), 
          r'$10^{14}, 10^{16}$' + '\n' + r'$\chi^2_R = $' + str(CHI[3]), 
          r'$10^{14}, 10^{17}$' + '\n' + r'$\chi^2_R = $' + str(CHI[4]),
          r'$10^{15}, 5\times10^{14}$' + '\n' + r'$\chi^2_R = $' + str(CHI[5]), 
          r'$10^{15}, 10^{15}$' + '\n' + r'$\chi^2_R = $' + str(CHI[6]), 
          r'$10^{15}, 5\times10^{15}$' + '\n' + r'$\chi^2_R = $' + str(CHI[7]), 
          r'$10^{15}, 10^{16}$' + '\n' + r'$\chi^2_R = $' + str(CHI[8]), 
          r'$10^{15}, 10^{17}$' + '\n' + r'$\chi^2_R = $' + str(CHI[9]),
          r'$10^{16}, 5\times10^{15}$' + '\n' + r'$\chi^2_R = $' + str(CHI[10]), 
          r'$10^{16}, 10^{16}$' + '\n' + r'$\chi^2_R = $' + str(CHI[11]), 
          r'$10^{16}, 5\times10^{16}$' + '\n' + r'$\chi^2_R = $' + str(CHI[12]), 
          r'$10^{16}, 10^{17}$' + '\n' + r'$\chi^2_R = $' + str(CHI[13]),
          r'$10^{17}, 5\times10^{17}$' + '\n' + r'$\chi^2_R = $' + str(CHI[14]), 
          r'$10^{17}, 10^{17}$' + '\n' + r'$\chi^2_R = $' + str(CHI[15])
          ]
   
    
# print(WEIGHTED_N)


x = np.linspace(0,16,16)
print(x)



fig = plt.figure(figsize=(21, 5), dpi=400)
# plt.yscale('log')
plt.xticks(ticks=x, labels=labels)
plt.xlabel('column density components N1, N2')
plt.ylabel(variable_name)

# plt.scatter(x, WEIGHTED_N, label='weighted average from generated values')
plt.errorbar(x, single, yerr=(LOWER, UPPER), ls='none', marker='o', capsize=3,  label='fit results', color='C1')
plt.vlines(x, ymin=og_data[0], ymax=og_data[1], color='red', linewidth=0.5)
plt.scatter(x, og_data[0], color='red', marker='x', s=15, label='generated comp. 1 and 2 max and min value')
plt.scatter(x, og_data[1], color='red', marker='x', s=15)
plt.legend()

# plt.errorbar(x, single_n, yerr=(og_data[0], og_data[1]), color='r', fmt="none", fillstyle='none', label='fitted')





