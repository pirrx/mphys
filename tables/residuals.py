# -*- coding: utf-8 -*-
"""
@author: aleks
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

#its not residuals but diff between the graphs but go off

FILENAMES = ["e14.csv", "e15.csv", "e16.csv", "e17.csv", "low.csv", "high.csv"]
# FILENAMES = ["radex_e14.csv", "radex_e15.csv", "radex_e16.csv", "radex_e17.csv"]


def file_check(filename):
    """
    Checks if all files are present in the directory.
    Arguments:
        filename: string
    Raises: FileNotFoundError
    Returns: bool
    """
    try:
        file = open(filename, 'r')
        file.close()
        return True
    except FileNotFoundError:
        print(f"{filename}" + " not found. Check the directory and the file.")
        return False
    
    
    
def data_array(filename, index_generated, index_fitted, index_changed):
    data = np.genfromtxt(filename, delimiter=';',
                     comments='%')

    GENERATED = data[:, index_generated]
    FITTED = data[:, index_fitted]
    UPPER = data[:, (index_fitted+1)]
    LOWER = data[:, (index_fitted+2)]

    CHANGED = data[:, index_changed]

    RESIDUALS = GENERATED - FITTED
    GENERATED = GENERATED
    FITTED = FITTED


    
    return RESIDUALS, GENERATED, FITTED, CHANGED, UPPER, LOWER
    
    
"""
FILENAMES index:
    0 - e14 colden
    1 - e15 colden
    2 - e16 colden
    3 - e17 colden
    4 - low temp
    5 - high temp
"""


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


index_generated = 1 #generated data (n2 or t2 etc)
index_generated_n1 = 0 #other generated data (n1 or t1 etc)
index_fitted = 11 #-2 variable for fitting
index_fitted_n1 = 8 #_1 variable for fitting
# index_upper = index_fitted +1
# index_lower = index_fitted +2
index_changed = 3 #whats gonna be on the x axis



#getting all the data sets
RESIDUALS_e14, GENERATED_e14, FITTED_e14, CHANGED_e14, UPPER_e14, LOWER_e14 = data_array(FILENAMES[0], index_generated, index_fitted, index_changed)
RESIDUALS_e14_N1, GENERATED_e14_N1, FITTED_e14_N1, CHANGED_e14_N1, UPPER_e14_N1, LOWER_e14_N1 = data_array(FILENAMES[0], index_generated_n1, index_fitted_n1, index_changed)
print(RESIDUALS_e14)
print(GENERATED_e14)
print(FITTED_e14)
print(CHANGED_e14)

RESIDUALS_e15, GENERATED_e15, FITTED_e15, CHANGED_e15, UPPER_e15, LOWER_e15 = data_array(FILENAMES[1], index_generated, index_fitted, index_changed)
RESIDUALS_e15_N1, GENERATED_e15_N1, FITTED_e15_N1, CHANGED_e15_N1, UPPER_e15_N1, LOWER_e15_N1 = data_array(FILENAMES[1], index_generated_n1, index_fitted_n1, index_changed)

RESIDUALS_e16, GENERATED_e16, FITTED_e16, CHANGED_e16, UPPER_e16, LOWER_e16 = data_array(FILENAMES[2], index_generated, index_fitted, index_changed)
RESIDUALS_e16_N1, GENERATED_e16_N1, FITTED_e16_N1, CHANGED_e16_N1, UPPER_e16_N1, LOWER_e16_N1 = data_array(FILENAMES[2], index_generated_n1, index_fitted_n1, index_changed)

RESIDUALS_e17, GENERATED_e17, FITTED_e17, CHANGED_e17, UPPER_e17, LOWER_e17 = data_array(FILENAMES[3], index_generated, index_fitted, index_changed)
RESIDUALS_e17_N1, GENERATED_e17_N1, FITTED_e17_N1, CHANGED_e17_N1, UPPER_e17_N1, LOWER_e17_N1 = data_array(FILENAMES[3], index_generated_n1, index_fitted_n1, index_changed)

#temp
RESIDUALS_low, GENERATED_low, FITTED_low, CHANGED_low, UPPER_low, LOWER_low = data_array(FILENAMES[4], index_generated, index_fitted, index_changed)
RESIDUALS_high, GENERATED_high, FITTED_high, CHANGED_high, UPPER_high, LOWER_high = data_array(FILENAMES[5], index_generated, index_fitted, index_changed)


 



#PLOT COMPARISON
fig = plt.figure()
plt.xscale('log')
# plt.plot(CHANGED_e14, GENERATED_e14, 'r', label="generated " + f"{GENERATED_e14[0]} K")


# plt.errorbar(CHANGED_e14, FITTED_e14, yerr=(LOWER_e14, UPPER_e14), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e14, GENERATED_e14, color='b', marker='.' , label="generated N1 = " + '10^14 /cm^2')
# plt.axhline(y = GENERATED_e14[0], color = 'blue', linewidth=1, linestyle = '--')
# plt.errorbar(CHANGED_e14_N1, FITTED_e14_N1, yerr=(LOWER_e14_N1, UPPER_e14_N1), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e14_N1, GENERATED_e14_N1, color='b', marker='.' , label="generated N1 = " + '10^14 /cm^2')
# plt.axhline(y = GENERATED_e14_N1[0], color = 'blue', linewidth=1, linestyle = '--')


# plt.errorbar(CHANGED_e15, FITTED_e15, yerr=(LOWER_e15, UPPER_e15), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e15, GENERATED_e15, color='b', marker='.' , label="generated at N1 = " + '10^15 /cm^2')
# plt.errorbar(CHANGED_e15_N1, FITTED_e15_N1, yerr=(LOWER_e15_N1, UPPER_e15_N1), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e15, GENERATED_e15_N1, color='b', marker='.' , label="generated at N1 = " + '10^15 /cm^2')
# plt.axhline(y = GENERATED_e15_N1[0], color = 'blue', linewidth=1, linestyle = '--')



# plt.errorbar(CHANGED_e16, FITTED_e16, yerr=(LOWER_e16, UPPER_e16), color='r', fmt="o", fillstyle='none', label='fitted')
# plt. label="generated at N1 = " + '10^16 /cm^2')
# plt.errorbar(CHANGED_e16_N1, FITTED_e16_N1, yerr=(LOWER_e16_N1, UPPER_e16_N1), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e16, GENERATED_e16_N1, color='b', marker='.' , label="generated at N1 = " + '10^16 /cm^2')
# plt.axhline(y = GENERATED_e16_N1[0], color = 'blue', linewidth=1, lscatter(CHANGED_e16, GENERATED_e16, color='b', marker='.' ,inestyle = '--')


# plt.errorbar(CHANGED_e17, FITTED_e17, yerr=(LOWER_e17, UPPER_e17), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e17, GENERATED_e17, color='b', marker='.' , label="generated at N1 = " + '10^17 /cm^2')
# plt.errorbar(CHANGED_e17_N1, FITTED_e17_N1, yerr=(LOWER_e17_N1, UPPER_e17_N1), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e17, GENERATED_e17_N1, color='b', marker='.' , label="generated at N1 = " + '10^17 /cm^2')
# plt.axhline(y = GENERATED_e17_N1[0], color = 'blue', linewidth=1, linestyle = '--')

# plt.errorbar(CHANGED_e16, FITTED_e16, yerr=(LOWER_e16, UPPER_e16), fmt="o", label='fitted')
# plt.errorbar(CHANGED_e17, FITTED_e17, yerr=(LOWER_e17, UPPER_e17), fmt="o", label='fitted')
    
plt.xticks()
plt.yticks()
    #plt.grid()
plt.xlabel('Column density N1')
plt.ylabel('source size 2')
plt.legend()
# plt.title('colden 1: ' + f"{colden[0] :.3g} K", fontsize=15)    
# fig_name = 'all colden for report' + '.png'
#plt.savefig()





# -------------------------------------------------------------------------


RESIDUAL_FIG = plt.figure(figsize=(15, 7))

N1 = RESIDUAL_FIG.add_subplot(211)
plt.xscale('log')

# plt.errorbar(CHANGED_e14_N1, FITTED_e14_N1, yerr=(LOWER_e14_N1, UPPER_e14_N1), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e14_N1, GENERATED_e14_N1, color='b', marker='.' , label="generated N1 = " + '10^14 /cm^2')
# plt.axhline(y = GENERATED_e14_N1[0], color = 'blue', linewidth=1, linestyle = '--')


plt.errorbar(CHANGED_e15_N1, FITTED_e15_N1, yerr=(LOWER_e15_N1, UPPER_e15_N1), color='r', fmt="o", fillstyle='none', label='fitted')
plt.scatter(CHANGED_e15, GENERATED_e15_N1, color='b', marker='.' , label="generated at N1 = " + '10^15 /cm^2')
plt.axhline(y = GENERATED_e15_N1[0], color = 'blue', linewidth=1, linestyle = '--')


# plt.errorbar(CHANGED_e16_N1, FITTED_e16_N1, yerr=(LOWER_e16_N1, UPPER_e16_N1), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e16, GENERATED_e16_N1, color='b', marker='.' , label="generated at N1 = " + '10^16 /cm^2')
# plt.axhline(y = GENERATED_e16_N1[0], color = 'blue', linewidth=1, linestyle = '--')

# plt.errorbar(CHANGED_e17_N1, FITTED_e17_N1, yerr=(LOWER_e17_N1, UPPER_e17_N1), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e17, GENERATED_e17_N1, color='b', marker='.' , label="generated at N1 = " + '10^17 /cm^2')
# plt.axhline(y = GENERATED_e17_N1[0], color = 'blue', linewidth=1, linestyle = '--')



plt.xticks()
plt.yticks()
    #plt.grid()
# plt.xlabel('Column density N1')
plt.ylabel('Source size s1')
plt.legend()

N2 = RESIDUAL_FIG.add_subplot(212)
plt.xscale('log')

# plt.errorbar(CHANGED_e14, FITTED_e14, yerr=(LOWER_e14, UPPER_e14), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e14, GENERATED_e14, color='b', marker='.' , label="generated N1 = " + '10^14 /cm^2')
# plt.axhline(y = GENERATED_e14[0], color = 'blue', linewidth=1, linestyle = '--')

plt.errorbar(CHANGED_e15, FITTED_e15, yerr=(LOWER_e15, UPPER_e15), color='r', fmt="o", fillstyle='none', label='fitted')
plt.scatter(CHANGED_e15, GENERATED_e15, color='b', marker='.' , label="generated at N1 = " + '10^15 /cm^2')
plt.axhline(y = GENERATED_e15[0], color = 'blue', linewidth=1, linestyle = '--')

# plt.errorbar(CHANGED_e16, FITTED_e16, yerr=(LOWER_e16, UPPER_e16), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e16, GENERATED_e16, color='b', marker='.' , label="generated at N1 = " + '10^16 /cm^2')
# plt.axhline(y = GENERATED_e16[0], color = 'blue', linewidth=1, linestyle = '--')

# plt.errorbar(CHANGED_e17, FITTED_e17, yerr=(LOWER_e17, UPPER_e17), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e17, GENERATED_e17, color='b', marker='.' , label="generated at N1 = " + '10^17 /cm^2')
# plt.axhline(y = GENERATED_e17[0], color = 'blue', linewidth=1, linestyle = '--')


plt.xticks()
plt.yticks()
    #plt.grid()
plt.xlabel('Column density N1')
plt.ylabel('Source size s2')
plt.legend()



# --------------------------------------------------------------------------

# CHANGED_e15_new = CHANGED_e15[0:-1]
# RESIDUALS_e15_new = RESIDUALS_e15[0:-1]
# LOWER_e15_new = LOWER_e15[0: -1]
# UPPER_e15_new = UPPER_e15[0:-1]



#PLOT RESIDUALS
fig = plt.figure()
plt.xscale('log')
# plt.yscale('log')
plt.xlabel('N1 Column density')
plt.ylabel('Source size residuals')
# plt.scatter(CHANGED, RESIDUALS)
# plt.errorbar(CHANGED_e14, RESIDUALS_e14, yerr=(LOWER_e14, UPPER_e14), fmt="o", label='N2 (changing) residuals')
# plt.errorbar(CHANGED_e14_N1, RESIDUALS_e14_N1, yerr=(LOWER_e14_N1, UPPER_e14_N1), fmt="o", label='N1 (=10^14) residuals')

plt.errorbar(CHANGED_e15, RESIDUALS_e15, yerr=(LOWER_e15, UPPER_e15), fmt="o", label='N2 (changing) residuals')
plt.errorbar(CHANGED_e15_N1, RESIDUALS_e15_N1, yerr=(LOWER_e15_N1, UPPER_e15_N1), fmt="o", label='N1 (=10^15) residuals')

# plt.errorbar(CHANGED_e16, RESIDUALS_e16, yerr=(LOWER_e16, UPPER_e16), fmt="o", label='T2 (100 K) residuals')
# plt.errorbar(CHANGED_e16_N1, RESIDUALS_e16_N1, yerr=(LOWER_e16_N1, UPPER_e16_N1), fmt="o", label='T1 (250 K) residuals')

# plt.errorbar(CHANGED_e17, RESIDUALS_e17, yerr=(LOWER_e17, UPPER_e17), fmt="o", label='N2 (changing) residuals')
# plt.errorbar(CHANGED_e17_N1, RESIDUALS_e17_N1, yerr=(LOWER_e17_N1, UPPER_e17_N1), fmt="o", label='N1 (=10^17) residuals')


plt.axhline(y = 0.0, color = 'r', linewidth=1, linestyle = '--')
plt.legend()


