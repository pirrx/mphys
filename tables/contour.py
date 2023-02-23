# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:42:42 2023

@author: aleks
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, CenteredNorm
from matplotlib import ticker, cm
import matplotlib.ticker as ticker
from path_file import TABLES_PATH


N1 = [1e14, 1e15, 1e16, 1e17]
N2 = [1e14, 5e14, 1e15, 5e15, 1e16, 5e16, 1e17]

T1 = [50, 100, 150, 200, 250, 300]
T2 = [50, 100, 150, 200, 250, 300]

single_n = [1.86e14, 3.96e14, 8.23e14, 1.01e16, 5.71e16, 4.15e14, 9.18e14, 4.93e15, 9.94e15, 9.96e16,
            7.84e15, 1.13e16, 3.60e16, 3.47e16, 6.18e16, 6.67e16]

single_t = [50.60, 67.21, 74.74, 74.47, 72.72, 69.37, 86.8, 133.14, 172.22, 211.3, 240.2, 239.28]

FILENAMES = ["e14.csv", "e15.csv", "e16.csv", "e17.csv", "low.csv", "high.csv"]

N_SINGLE_T_FITTING = []

FILENAMES = [TABLES_PATH + file for file in FILENAMES]

# for file in FILENAMES:
#     file = TABLES_PATH + file

# FILENAMES = TABLES_PATH + FILENAMES

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


index_generated = 3 #generated data (n2 or t2 etc)
index_generated_n1 = 3 #other generated data (n1 or t1 etc)
index_fitted = 17 #-2 variable for fitting
index_fitted_n1 = 14 #_1 variable for fitting
index_changed = 3 #whats gonna be on the x axis

index_single = 36

t_index_generated = 3
t_index_generated_t1 = 2
t_index_fitted = 17
t_index_fitted_t1 = 14
t_index_changed = 1




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




GENERATED_low, CHANGED_low, FITTED_low, UPPER_low, LOWER_low = data_array(FILENAMES[4], t_index_generated, t_index_fitted, t_index_changed)
GENERATED_low_T1, CHANGED_low_T1, FITTED_low_T1, UPPER_low_T1, LOWER_low_T1 = data_array(FILENAMES[4], t_index_generated_t1, t_index_fitted_t1, t_index_changed)
GENERATED_low_single, CHANGED_low_single, FITTED_low_single, UPPER_low_single, LOWER_low_single = data_array(FILENAMES[4], t_index_generated_t1, index_single, t_index_changed)

GENERATED_high, CHANGED_high, FITTED_high, UPPER_high, LOWER_high = data_array(FILENAMES[5], t_index_generated, t_index_fitted, t_index_changed)
GENERATED_high_T1, CHANGED_high_T1, FITTED_high_T1, UPPER_high_T1, LOWER_high_T1 = data_array(FILENAMES[5],t_index_generated_t1, t_index_fitted_t1, t_index_changed)
GENERATED_high_single, CHANGED_high_single, FITTED_high_single, UPPER_high_single, LOWER_high_single = data_array(FILENAMES[5], t_index_generated_t1, index_single, t_index_changed)


# FITTED_e14 = (FITTED_e14 - GENERATED_e14) / GENERATED_e14 *100
# FITTED_e15 = (FITTED_e15 - GENERATED_e15) /GENERATED_e15 *100
# FITTED_e16 = (FITTED_e16 - GENERATED_e16) / GENERATED_e16 *100
# FITTED_e17 = (FITTED_e17 - GENERATED_e17) / GENERATED_e17 *100
# FITTED_low = (FITTED_low - GENERATED_low) / GENERATED_low *100
# FITTED_high = (FITTED_high - GENERATED_high) /GENERATED_high *100


# FITTED_e14_N1 = (FITTED_e14_N1 - GENERATED_e14_N1) /GENERATED_e14_N1 *100
# FITTED_e15_N1 = (FITTED_e15_N1 - GENERATED_e15_N1) /GENERATED_e15_N1 *100
# FITTED_e16_N1 = (FITTED_e16_N1 - GENERATED_e16_N1) /GENERATED_e16_N1 *100
# FITTED_e17_N1 = (FITTED_e17_N1 - GENERATED_e17_N1) /GENERATED_e17_N1 *100
# FITTED_low_T1 = (FITTED_low_T1 - GENERATED_low_T1) /GENERATED_low_T1 *100
# FITTED_high_T1 =( FITTED_high_T1 - GENERATED_high_T1) /GENERATED_high_T1 *100


ZZ_N2 = []
ZZ_N2.extend(FITTED_e14)
ZZ_N2.extend(FITTED_e15)
ZZ_N2.extend(FITTED_e16)
ZZ_N2.extend(FITTED_e17)

ZZ_N1 = []
ZZ_N1.extend(FITTED_e14_N1)
ZZ_N1.extend(FITTED_e15_N1)
ZZ_N1.extend(FITTED_e16_N1)
ZZ_N1.extend(FITTED_e17_N1)

ZZ_NS = []
ZZ_NS.extend(FITTED_e14_single)
ZZ_NS.extend(FITTED_e15_single)
ZZ_NS.extend(FITTED_e16_single)
ZZ_NS.extend(FITTED_e17_single)


print(ZZ_NS)

ZZ_T2 = []
ZZ_T2.extend(FITTED_low)
ZZ_T2.extend(FITTED_high)

ZZ_T1 = []
ZZ_T1.extend(FITTED_low_T1)
ZZ_T1.extend(FITTED_high_T1)

ZZ_TS = []
ZZ_TS.extend(FITTED_low_single)
ZZ_TS.extend(FITTED_high_single)

print('hi')
print(ZZ_T1)
print(FITTED_low_single)
print(FITTED_high_single)

print(ZZ_N1)
print(ZZ_N2)
print(ZZ_T1)
print(ZZ_T2)


X_N, Y_N = np.meshgrid(N1, N2)
print(X_N[1,2])
print(X_N)
print(Y_N)

single_n = [1.86e14, 3.96e14, 8.23e14, 1.01e16, 5.71e16, 4.15e14, 9.18e14, 4.93e15, 9.94e15, 9.96e16,
            7.84e15, 1.13e16, 3.60e16, 3.47e16, 6.18e16, 6.67e16]
single_n = ZZ_NS

print(single_n)

Z_N1 = np.zeros([7, 4], dtype = float)
Z_N2 = np.zeros([7, 4], dtype = float)
Z_N_SINGLE = np.zeros([7, 4], dtype = float)
# Z_SINGLE.extend(single_n)


Z_N_SINGLE[0,0] = single_n[0]
Z_N_SINGLE[1,0] = single_n[1]
Z_N_SINGLE[2,0] = single_n[2]
Z_N_SINGLE[4,0] = single_n[3]
Z_N_SINGLE[6,0] = single_n[4]

Z_N_SINGLE[1,1] = single_n[5]
Z_N_SINGLE[2,1] = single_n[6]
Z_N_SINGLE[3,1] = single_n[7]
Z_N_SINGLE[4,1] = single_n[8]
Z_N_SINGLE[6,1] = single_n[9]

Z_N_SINGLE[3,2] = single_n[10]
Z_N_SINGLE[4,2] = single_n[11]
Z_N_SINGLE[5,2] = single_n[12]
Z_N_SINGLE[6,2] = single_n[13]

Z_N_SINGLE[5,3] = single_n[14]
Z_N_SINGLE[6,3] = single_n[15]

################

Z_N1[0,0] = ZZ_N1[0]
Z_N1[1,0] = ZZ_N1[1]
Z_N1[2,0] = ZZ_N1[2]
Z_N1[4,0] = ZZ_N1[3]
Z_N1[6,0] = ZZ_N1[4]

Z_N1[1,1] = ZZ_N1[5]
Z_N1[2,1] = ZZ_N1[6]
Z_N1[3,1] = ZZ_N1[7]
Z_N1[4,1] = ZZ_N1[8]
Z_N1[6,1] = ZZ_N1[9]

Z_N1[3,2] = ZZ_N1[10]
Z_N1[4,2] = ZZ_N1[11]
Z_N1[5,2] = ZZ_N1[12]
Z_N1[6,2] = ZZ_N1[13]

Z_N1[5,3] = ZZ_N1[14]
Z_N1[6,3] = ZZ_N1[15]

#####################################


Z_N2[0,0] = ZZ_N2[0]
Z_N2[1,0] = ZZ_N2[1]
Z_N2[2,0] = ZZ_N2[2]
Z_N2[4,0] = ZZ_N2[3]
Z_N2[6,0] = ZZ_N2[4]

Z_N2[1,1] = ZZ_N2[5]
Z_N2[2,1] = ZZ_N2[6]
Z_N2[3,1] = ZZ_N2[7]
Z_N2[4,1] = ZZ_N2[8]
Z_N2[6,1] = ZZ_N2[9]



Z_N2[3,2] = ZZ_N2[10]
Z_N2[4,2] = ZZ_N2[11]
Z_N2[5,2] = ZZ_N2[12]
Z_N2[6,2] = ZZ_N2[13]

Z_N2[5,3] = ZZ_N2[14]
Z_N2[6,3] = ZZ_N2[15]

### START CHANGING STUFF HERE
Z = Z_N_SINGLE
Z[Z_N_SINGLE == 0.] = np.nan

# Z = Z_N1
# Z[Z_N1 == 0.] = np.nan

# Z = Z_N2
# Z[Z_N2 == 0.] = np.nan

print(Z)

levels = np.logspace(14, 17, 20)
# print(levels)

fig = plt.figure(dpi=400)
plt.yscale('log')
plt.xscale('log')
plt.title(r'N$_{single}$', fontsize=10)
plt.xlabel(r'First column density component N$_1$ [cm$^{-2}$]')
plt.ylabel(r'Second column density component N$_2$ [cm$^{-2}$]')
# plt.ticks()
# plt.ylabel(variable_name)
plt.contourf(X_N, Y_N, Z, 20, cmap='RdBu_r', vmin=1e14, vmax=1e17)
plt.colorbar(format=ticker.FuncFormatter(fmt))
# format=ticker.FuncFormatter(fmt)
# % distance from generated values
# norm=LogNorm(), levels=levels
#SymLogNorm(linthresh=0.03, linscale=1)


############################# TEMPERATURE
single_t = [50.60, 67.21, 74.74, 74.47, 72.72, 69.37, 86.8, 133.14, 172.22, 211.3, 240.2, 239.28]
single_t = ZZ_TS
       

X_T, Y_T = np.meshgrid(T1, T2)
print(X_T[1,2])
print(X_T)
print(Y_T)

X_TN = np.logspace(14, 17, 6)
Y_TN = np.logspace(14, 17, 6)

# X_T, Y_T = np.meshgrid(X_T, Y_T)

# print(X_T[1,2])
print(X_TN)
print(Y_TN)



Z_T_SINGLE = np.full([6, 6], 0, dtype = float)
Z_T1 = np.full([6, 6], 0, dtype = float)
Z_T2 = np.full([6, 6], 0, dtype = float)

Z_T_SINGLE[0,0] = single_t[0]
Z_T_SINGLE[1,0] = single_t[1]
Z_T_SINGLE[2,0] = single_t[2]
Z_T_SINGLE[3,0] = single_t[3]
Z_T_SINGLE[4,0] = single_t[4]
Z_T_SINGLE[5,0] = single_t[5]

Z_T_SINGLE[0,4] = single_t[6]
Z_T_SINGLE[1,4] = single_t[7]
Z_T_SINGLE[2,4] = single_t[8]
Z_T_SINGLE[3,4] = single_t[9]
Z_T_SINGLE[4,4] = single_t[10]
Z_T_SINGLE[5,4] = single_t[11]

###########

Z_T1[0,0] = ZZ_T1[0]
Z_T1[1,0] = ZZ_T1[1]
Z_T1[2,0] = ZZ_T1[2]
Z_T1[3,0] = ZZ_T1[3]
Z_T1[4,0] = ZZ_T1[4]
Z_T1[5,0] = ZZ_T1[5]

Z_T1[0,4] = ZZ_T1[6]
Z_T1[1,4] = ZZ_T1[7]
Z_T1[2,4] = ZZ_T1[8]
Z_T1[3,4] = ZZ_T1[9]
Z_T1[4,4] = ZZ_T1[10]
Z_T1[5,4] = ZZ_T1[11]

############

Z_T2[0,0] = ZZ_T2[0]
Z_T2[1,0] = ZZ_T2[1]
Z_T2[2,0] = ZZ_T2[2]
Z_T2[3,0] = ZZ_T2[3]
Z_T2[4,0] = ZZ_T2[4]
Z_T2[5,0] = ZZ_T2[5]

Z_T2[0,4] = ZZ_T2[6]
Z_T2[1,4] = ZZ_T2[7]
Z_T2[2,4] = ZZ_T2[8]
Z_T2[3,4] = ZZ_T2[9]
Z_T2[4,4] = ZZ_T2[10]
Z_T2[5,4] = ZZ_T2[11]

# Z = Z_T_SINGLE

# Z = Z_T1

Z = Z_T2


# Z[Z_T_SINGLE == 0.] = np.nan
# print(Z)

# print(ZZ_T1)
# print(ZZ_T2)

##################################
fig = plt.figure(dpi=400)
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx


plt.title(r'$N_{2}$ fit')
plt.xlabel(r'First temperature component $T_1$ [K]')
plt.ylabel(r'Second temperature component $T_2$ [K]')
plt.contourf(X_T, Y_T, Z, 10, cmap='RdBu_r', norm=CenteredNorm(vcenter=1e15))
plt.colorbar(format=ticker.FuncFormatter(fmt))
###################################

# ax2
#norm=CenteredNorm(vcenter=1e15)

# plt.set_xlabel(r'First colden component $T_1$ [cm$^{-2}$]')  # we already handled the x-label with ax1
# ax2.plot(t, data2, color=color)
# plt.tick_params(axis='y')

# ax2




