# -*- coding: utf-8 -*-
"""
@author: aleks
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

# FILENAMES = ["e14.csv", "e15.csv", "e16.csv", "e17.csv", "low.csv", "high.csv"]
FILENAMES = ["radex_e14.csv", "radex_e15.csv", "radex_e16.csv", "radex_e17.csv"]
# FILENAMES = ["lte_e14.csv", "lte_e15.csv", "lte_e16.csv", "lte_e17.csv"]
# FILENAMES = ["radex_lte_e14.csv", "radex_lte_e15.csv", "radex_lte_e16.csv", "radex_lte_e17.csv"]

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
    
    print(data)
    
    print(data[:, index_fitted])

    GENERATED = data[:, index_generated]
    FITTED = data[:, index_fitted]
    UPPER = data[:, (index_fitted+1)]
    LOWER = data[:, (index_fitted+2)]

    CHANGED = data[:, index_changed]

    RESIDUALS = GENERATED - FITTED
    GENERATED = GENERATED
    FITTED = FITTED

    #changed as in?? what did i do
    #what goes into the x-axis, so for me const N_1??
    
    return RESIDUALS, GENERATED, FITTED, CHANGED, UPPER, LOWER

def data_array_single(filename, index_single_fitted):
    data = np.genfromtxt(filename, delimiter=';',
                     comments='%')
    
    print(data)
    
    print(data[:, index_fitted])

    FITTED = data[:, index_single_fitted]
    UPPER = data[:, (index_single_fitted+1)]
    LOWER = data[:, (index_single_fitted+2)]

    FITTED = FITTED

    #changed as in?? what did i do
    #what goes into the x-axis, so for me const N_1??
    
    return FITTED, UPPER, LOWER
    
    
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


index_generated = 3
index_generated_n1 = 2
index_fitted = 17
index_fitted_n1 = 14
index_single_fitted = 36
# index_upper = index_fitted +1
# index_lower = index_fitted +2
index_changed = 3
# index_tgenerated = 1
# index_generated_t1 = 0
# index_tfitted = 11
# index_fitted_t1 = 8
# index_tchanged = 1



# colden varied
RESIDUALS_e14, GENERATED_e14, FITTED_e14, CHANGED_e14, UPPER_e14, LOWER_e14 = data_array(FILENAMES[0], index_generated, index_fitted, index_changed)
RESIDUALS_e14_N1, GENERATED_e14_N1, FITTED_e14_N1, CHANGED_e14_N1, UPPER_e14_N1, LOWER_e14_N1 = data_array(FILENAMES[0], index_generated_n1, index_fitted_n1, index_changed)
FITTED_e14_single, UPPER_e14_single, LOWER_e14_single = data_array_single(FILENAMES[0], index_single_fitted)


RESIDUALS_e15, GENERATED_e15, FITTED_e15, CHANGED_e15, UPPER_e15, LOWER_e15 = data_array(FILENAMES[1], index_generated, index_fitted, index_changed)
RESIDUALS_e15_N1, GENERATED_e15_N1, FITTED_e15_N1, CHANGED_e15_N1, UPPER_e15_N1, LOWER_e15_N1 = data_array(FILENAMES[1], index_generated_n1, index_fitted_n1, index_changed)
FITTED_e15_single, UPPER_e15_single, LOWER_e15_single = data_array_single(FILENAMES[1], index_single_fitted)



RESIDUALS_e16, GENERATED_e16, FITTED_e16, CHANGED_e16, UPPER_e16, LOWER_e16 = data_array(FILENAMES[2], index_generated, index_fitted, index_changed)
RESIDUALS_e16_N1, GENERATED_e16_N1, FITTED_e16_N1, CHANGED_e16_N1, UPPER_e16_N1, LOWER_e16_N1 = data_array(FILENAMES[2], index_generated_n1, index_fitted_n1, index_changed)
FITTED_e16_single, UPPER_e16_single, LOWER_e16_single = data_array_single(FILENAMES[2], index_single_fitted)


RESIDUALS_e17, GENERATED_e17, FITTED_e17, CHANGED_e17, UPPER_e17, LOWER_e17 = data_array(FILENAMES[3], index_generated, index_fitted, index_changed)
RESIDUALS_e17_N1, GENERATED_e17_N1, FITTED_e17_N1, CHANGED_e17_N1, UPPER_e17_N1, LOWER_e17_N1 = data_array(FILENAMES[3], index_generated_n1, index_fitted_n1, index_changed)
FITTED_e17_single, UPPER_e17_single, LOWER_e17_single = data_array_single(FILENAMES[3], index_single_fitted)



# temperature varied
# RESIDUALS_low, GENERATED_low, FITTED_low, CHANGED_low, UPPER_low, LOWER_low = data_array(FILENAMES[4], index_tgenerated, index_tfitted, index_tchanged)
# RESIDUALS_low_T1, GENERATED_low_T1, FITTED_low_T1, CHANGED_low_T1, UPPER_low_T1, LOWER_low_T1 = data_array(FILENAMES[4], index_generated_t1, index_fitted_t1, index_tchanged)

# RESIDUALS_high, GENERATED_high, FITTED_high, CHANGED_high, UPPER_high, LOWER_high = data_array(FILENAMES[5], index_tgenerated, index_tfitted, index_tchanged)
# RESIDUALS_high_T1, GENERATED_high_T1, FITTED_high_T1, CHANGED_high_T1, UPPER_high_T1, LOWER_high_T1 = data_array(FILENAMES[5], index_generated_t1, index_fitted_t1, index_tchanged)

# print(RESIDUALS_low_T1)
# print(GENERATED_low_T1)
# print(FITTED_low_T1)
 



#PLOT COMPARISON
fig = plt.figure()
plt.xscale('log')
# plt.plot(CHANGED_e14, GENERATED_e14, 'r', label="generated " + f"{GENERATED_e14[0]} K")


plt.errorbar(CHANGED_e14, FITTED_e14, yerr=(LOWER_e14, UPPER_e14), color='r', fmt="o", fillstyle='none', label='fitted')
plt.scatter(CHANGED_e14, GENERATED_e14, color='b', marker='.' , label="generated N1 = " + '10^14 /cm^2')
# plt.errorbar(CHANGED_e14_N1, FITTED_e14_N1, yerr=(LOWER_e14_N1, UPPER_e14_N1), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e14_N1, GENERATED_e14_N1, color='b', marker='.' , label="generated N1 = " + '10^14 /cm^2')
# plt.axhline(y = GENERATED_e14_N1[0], color = 'blue', linewidth=1, linestyle = '--')


# plt.errorbar(CHANGED_e15, FITTED_e15, yerr=(LOWER_e15, UPPER_e15), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e15, GENERATED_e15, color='b', marker='.' , label="generated at N1 = " + '10^15 /cm^2')
# plt.errorbar(CHANGED_e15_N1, FITTED_e15_N1, yerr=(LOWER_e15_N1, UPPER_e15_N1), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e15, GENERATED_e15_N1, color='b', marker='.' , label="generated at N1 = " + '10^15 /cm^2')
# plt.axhline(y = GENERATED_e15_N1[0], color = 'blue', linewidth=1, linestyle = '--')


# change 'low'/'high' here
# plt.errorbar(CHANGED_low, FITTED_low, yerr=(LOWER_low, UPPER_low), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_low, GENERATED_low, color='b', marker='.' , label='generated at T1 = 50K')

# plt.errorbar(CHANGED_e16_N1, FITTED_e16_N1, yerr=(LOWER_e16_N1, UPPER_e16_N1), color='r', fmt="o", fillstyle='none', label='fitted')
# plt.scatter(CHANGED_e16, GENERATED_e16_N1, color='b', marker='.' , label="generated at N1 = " + '10^16 /cm^2')
# plt.axhline(y = GENERATED_e16_N1[0], color = 'blue', linewidth=1, linestyle = '--')


# plt.errorbar(CHANGED_e16, FITTED_e16, yerr=(LOWER_e16, UPPER_e16), fmt="o", label='fitted')
# plt.errorbar(CHANGED_e17, FITTED_e17, yerr=(LOWER_e17, UPPER_e17), fmt="o", label='fitted')



# TO DO:
    # ADD LEGENDS
    # ADD EASIER COMPARISION?
    # IMPROVE RESIDUALS PLOT





#plt.scatter(GENERATED, RESIDUALS)
# plt.plot(GENERATED, RESIDUALS, linewidth = 1.5, color = 'black', label = '250 K')
    # 
    
plt.xticks()
plt.yticks()
    #plt.grid()
plt.xlabel('Temperature T1')
plt.ylabel('Temperature T2')
plt.legend()
# plt.title('colden 1: ' + f"{colden[0] :.3g} K", fontsize=15)    
fig_name = 'temp' + '.png'
#plt.savefig()


#-----------------------------------------------
# start one in everything fig
RESIDUAL_FIG = plt.figure(figsize=(15, 7))
N1 = RESIDUAL_FIG.add_subplot(111) #311
# plt.errorbar(CHANGED_e14_N1, FITTED_e14_N1, yerr=(LOWER_e14_N1, UPPER_e14_N1), color='black', fmt="o", fillstyle='none', linewidth=3, label='fitted N1')
# plt.errorbar(CHANGED_e14, FITTED_e14, yerr=(LOWER_e14, UPPER_e14), color='grey', fmt="o", fillstyle='none', linewidth=2, label='fitted N2')
# plt.errorbar(CHANGED_e14, FITTED_e14_single, yerr=(LOWER_e14_single, UPPER_e14_single), color='r', fmt="o", fillstyle='none', linewidth=1, label='fitted N single')
# plt.scatter(CHANGED_e14_N1, GENERATED_e14_N1, color='b', marker='.' , s=200, label=r"generated N1") #= $10^{14}$
# plt.scatter(CHANGED_e14, GENERATED_e14, color='green', marker='.' , s=100, label=r"generated N2") #= $10^{14}$

# plt.errorbar(CHANGED_e15_N1, FITTED_e15_N1, yerr=(LOWER_e15_N1, UPPER_e15_N1), color='black', fmt="o", fillstyle='none', linewidth=3, label='fitted N1')
# plt.errorbar(CHANGED_e15, FITTED_e15, yerr=(LOWER_e15, UPPER_e15), color='grey', fmt="o", fillstyle='none', linewidth=2, label='fitted N2')
# plt.errorbar(CHANGED_e15, FITTED_e15_single, yerr=(LOWER_e15_single, UPPER_e15_single), color='r', fmt="o", fillstyle='none', linewidth=1, label='fitted N single')
# plt.scatter(CHANGED_e15_N1, GENERATED_e15_N1, color='b', marker='.' , s=200, label=r"generated N1") #= $10^{14}$
# plt.scatter(CHANGED_e15, GENERATED_e15, color='green', marker='.' , s=100, label=r"generated N2") #= $10^{14}$

# plt.errorbar(CHANGED_e16_N1, FITTED_e16_N1, yerr=(LOWER_e16_N1, UPPER_e16_N1), color='black', fmt="o", fillstyle='none', linewidth=3, label='fitted N1')
# plt.errorbar(CHANGED_e16, FITTED_e16, yerr=(LOWER_e16, UPPER_e16), color='grey', fmt="o", fillstyle='none', linewidth=2, label='fitted N2')
# plt.errorbar(CHANGED_e16, FITTED_e16_single, yerr=(LOWER_e16_single, UPPER_e16_single), color='r', fmt="o", fillstyle='none', linewidth=1, label='fitted N single')
# plt.scatter(CHANGED_e16_N1, GENERATED_e16_N1, color='b', marker='.' , s=200, label=r"generated N1") #= $10^{14}$
# plt.scatter(CHANGED_e16, GENERATED_e16, color='green', marker='.' , s=100, label=r"generated N2") #= $10^{14}$

# plt.errorbar(CHANGED_e17_N1, FITTED_e17_N1, yerr=(LOWER_e17_N1, UPPER_e17_N1), color='black', fmt="o", fillstyle='none', linewidth=3, label='fitted N1')
# plt.errorbar(CHANGED_e17, FITTED_e17, yerr=(LOWER_e17, UPPER_e17), color='grey', fmt="o", fillstyle='none', linewidth=2, label='fitted N2')
# plt.errorbar(CHANGED_e17, FITTED_e17_single, yerr=(LOWER_e17_single, UPPER_e17_single), color='r', fmt="o", fillstyle='none', linewidth=1, label='fitted N single')
# plt.scatter(CHANGED_e17_N1, GENERATED_e17_N1, color='b', marker='.' , s=200, label=r"generated N1") #= $10^{14}$
# plt.scatter(CHANGED_e17, GENERATED_e17, color='green', marker='.' , s=100, label=r"generated N2") #= $10^{14}$

# plt.axhline(y = GENERATED_e14_N1[0], color = 'blue', linewidth=1, linestyle = '--')
plt.xticks()
plt.yticks()
plt.yscale('log')
plt.xscale('log')
    #plt.grid()
plt.xlabel('Column density N1')
plt.ylabel('Column density N1')

plt.legend()

# end large figure
#-----------------------------------------------------------------------



#start three sep figs
#-----------------------------------------------------------------------
RESIDUAL_FIG = plt.figure(figsize=(10, 10))
N1 = RESIDUAL_FIG.add_subplot(311) #311
plt.errorbar(CHANGED_e15, FITTED_e15_single, yerr=(LOWER_e15_single, UPPER_e15_single), color='r', fmt="o", fillstyle='none', linewidth=1, label='fitted N single')
plt.scatter(CHANGED_e15_N1, GENERATED_e15_N1, color='b', marker='.', label=r"generated N1 = $10^{15}$") #= $10^{14}$
plt.scatter(CHANGED_e15, GENERATED_e15, color='green', marker='.' , label=r"generated N2") #= $10^{14}$

# plt.axhline(y = GENERATED_e14_N1[0], color = 'blue', linewidth=1, linestyle = '--')
plt.xticks()
plt.yticks()
plt.yscale('log')
plt.xscale('log')
    #plt.grid()
plt.xlabel('Column density N1')
plt.ylabel('Column density N1')
N1.legend(loc='lower right', bbox_to_anchor=(1.3, 0.0))


N2 = RESIDUAL_FIG.add_subplot(312)
plt.scatter(CHANGED_e15_N1, GENERATED_e15_N1, color='b', marker='.', label=r"generated N1 = $10^{15}$") #= $10^{14}$
plt.scatter(CHANGED_e15, GENERATED_e15, color='green', marker='.' , label=r"generated N2") #= $10^{14}$
plt.errorbar(CHANGED_e15_N1, FITTED_e15_N1, yerr=(LOWER_e15_N1, UPPER_e15_N1), color='r', fmt="o", fillstyle='none', linewidth=1, label='fitted N1')
plt.xticks()
plt.yticks()
plt.yscale('log')
plt.xscale('log')
    #plt.grid()
plt.xlabel('Column density N1')
plt.ylabel('Column density N2')
N2.legend(loc='lower right', bbox_to_anchor=(1.3, 0.))

N3 = RESIDUAL_FIG.add_subplot(313)
plt.scatter(CHANGED_e15, GENERATED_e15, color='b', marker='.' , label=r"generated N1 = $10^{15}$")
plt.scatter(CHANGED_e15, GENERATED_e15_N1, color='green', marker='.' , label=r"generated N2")
plt.errorbar(CHANGED_e15, FITTED_e15, yerr=(LOWER_e15, UPPER_e15), color='r', fmt="o", fillstyle='none', linewidth=1, label='fitted N2')
plt.xticks()
plt.yticks()
plt.yscale('log')
plt.xscale('log')
    #plt.grid()
plt.xlabel('Column density N1')
plt.ylabel('Column density N2')
N3.legend(loc='lower right', bbox_to_anchor=(1.3, 0.0))


# #PLOT RESIDUALS
# fig = plt.figure()
# plt.xscale('log')
# # plt.yscale('log')
# plt.xlabel('T1 Temperature')
# plt.ylabel('Residuals')
# # plt.scatter(CHANGED, RESIDUALS)
# plt.errorbar(CHANGED_e14, RESIDUALS_e14, yerr=(LOWER_e14, UPPER_e14), fmt="o", label='N2 (changing) residuals')
# plt.errorbar(CHANGED_e14_N1, RESIDUALS_e14_N1, yerr=(LOWER_e14_N1, UPPER_e14_N1), fmt="o", label='N1 (=10^14) residuals')

# # plt.errorbar(CHANGED_e15_new, RESIDUALS_e15_new, yerr=(LOWER_e15_new, UPPER_e15_new), fmt="o", label='N2 (changing) residuals')
# # plt.errorbar(CHANGED_e15_N1, RESIDUALS_e15_N1, yerr=(LOWER_e15_N1, UPPER_e15_N1), fmt="o", label='N1 (=10^15) residuals')

# # plt.errorbar(CHANGED_low, RESIDUALS_low, yerr=(LOWER_low, UPPER_low), fmt="o", label='T2 (changing) residuals')
# # plt.errorbar(CHANGED_low_T1, RESIDUALS_low_T1, yerr=(LOWER_low_T1, UPPER_low_T1), fmt="o", label='T1 (=50K) residuals')

# plt.axhline(y = 0.0, color = 'r', linewidth=1, linestyle = '--')
# plt.legend()
# plt.show()


