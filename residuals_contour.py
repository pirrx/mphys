# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:32:45 2023

@author: aleks
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from file_path import RESIDUALS_PATH, TABLES_PATH, RESIDUALS_PATH_NONLTE
import matplotlib.ticker as mtick
# from weights import CHI

# NOISE_N_FILENAME = "noise_colden.csv"
# NOISE_T_FILENAME = "noise_extemp.csv"


print(RESIDUALS_PATH)
print(RESIDUALS_PATH_NONLTE)
print(TABLES_PATH)

X = [1e14, 1e15, 1e16, 1e17]
Y = [1e14, 1e15, 1e16, 1e17]

# labels = [r'$10^{14}, 10^{14}$',
#           r'$10^{14}, 10^{15}$',
#           r'$10^{14}, 10^{16}$',
#           r'$10^{14}, 10^{17}$',
          
#           r'$10^{15}, 10^{14}$',
#           r'$10^{15}, 10^{15}$',
#           r'$10^{15}, 10^{16}$',
#           r'$10^{15}, 10^{17}$',

#           r'$10^{16}, 10^{14}$',
#           r'$10^{16}, 10^{15}$',
#           r'$10^{16}, 10^{16}$',
#           r'$10^{16}, 10^{17}$',
          
#           r'$10^{17}, 10^{14}$',
#           r'$10^{17}, 10^{15}$',
#           r'$10^{17}, 10^{16}$',
#           r'$10^{17}, 10^{17}$',
#           ]




def file_check(filename):
    """
    Checks if all files are present in the directory.

    Args:
        filename (string): name of the data set file

    Raises:
        FileNotFoundError

    Returns:
        bool
    """
    try:
        file = open(filename, 'r')
        file.close()
        return True
    except FileNotFoundError:
        print(f"{filename}" + " not found. Check the directory and the file.")
        return False

def get_filenames(directory_path):
    """
    takes all files in the specified in the path
    (so files need to be saved in a separate folder)
    """
    filenames = []
    
    for path in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, path)):
            filenames.append(path)
    return filenames

def get_sorted_files(filenames, option):
    """ separates one and two comp filenames - files need to contain either 'one' 
    or 'two' in their name 
    option: lte_1, lte_2, non_lte_1, non_lte_2
    """
    
    files_array = []
    
    
    files_array = [i for i in filenames if option in i]
    # one_comp_residuals = [i for i in filenames if 'one' in i]
    # two_comp_residuals = [i for i in filenames if 'two' in i]

    return files_array

# def get_noise(filename):
#     file_path = TABLES_PATH + filename
#     data = np.genfromtxt(file_path, dtype='float', delimiter=';', skip_header=1)
#     noise = data[:,1]
#     return noise

def get_data_single_file(filename):
    """

    """
    file_path = RESIDUALS_PATH_NONLTE + filename
    # print(file_path)
    data = np.genfromtxt(file_path, dtype='float', delimiter=' ', skip_header=0)
    # print(data)
    frequency = data[:,0]
    intensity = data[:,1]
    return frequency, intensity

def root_mean_square(array):
    return np.sqrt(np.mean(array**2))

def over_three_sigma(array):
    
    mean = root_mean_square(array)       
    [i] = np.where((3*mean) < np.abs(array))
    print("indices")        
    print(i)
    return len(i)

def populate_mesh_grid(list_over_three_sigma):
    "chi meshgrid, can use meshgrid arrays for x and y len"
    list_over_three_sigma = np.array(list_over_three_sigma)
    Z = np.full([4, 4], 0.0, dtype = float)

    Z = list_over_three_sigma.reshape(4, 4)
        
    return Z


# FILENAMES = get_filenames(RESIDUALS_PATH_NONLTE)
# # # print('filenames')
# # # print(FILENAMES)

# FILES = get_sorted_files(FILENAMES, "non_lte_2")
# print(FILES)

# freq, ins = get_data_single_file(FILES[15])
# print(ins)



# X = root_mean_square(ins)
# print("mean:")
# print(X)

# NR = over_three_sigma(ins)
# print(NR)


# # # # # # ------------------------------------------------------- # # # # # #
#                                    hello                                    #
# # # # # # ------------------------------------------------------- # # # # # #

def get_data_all_files(path, option):
    """
    path as specified in the path file and imported - path to the folder with 
    all residuals disttributions
    
    options: "non_lte_2", "non_lte_1", "lte_2", "lte_2" - to EDIT BECAUSE THIS 
    WILL NOT SORT PROPERLY -> CHANGE FILE NAMES :((( to something more distinct
    and adjust options accordingly
    option is used in "get_sorted_files()" function
        

    """
    FILENAMES = get_filenames(path)
    # # print('filenames')
    # # print(FILENAMES)

    FILES = get_sorted_files(FILENAMES, option)
    print(FILES)
    freq, ins = get_data_single_file(FILES[0])
    print(ins)
    X = root_mean_square(ins)
    print("mean:")
    print(X)
    NR = over_three_sigma(ins)
    print(NR)
    
    # noise = get_noise(noise_file)
    index = 0
    nr_over_limit_values = []
    length = 0

    for file in FILES:
        # noise_value = noise[index]
        print(index)
        print(file)
        freq, intensity = get_data_single_file(file)  
        mean = root_mean_square(intensity) 
        print(mean)        
        # [i] = np.where((3*mean) < np.abs(intensity))
        nr = over_three_sigma(intensity)        
        nr_over_limit_values.append(nr)
        
        # plt.figure()
        # plt.scatter(freq, intensity, 0.5)
        # plt.scatter(freq[i], intensity[i], 0.5)
        # plt.show()
        length = len(intensity)
        index += 1
        
    ### ---------- fin loop ---------- ###
    
    
    print(nr_over_limit_values)
    percentage_over = [x / length * 100 for x in nr_over_limit_values]
    
    X = [1e14, 1e15, 1e16, 1e17]
    Y = [1e14, 1e15, 1e16, 1e17]
    N_mesh_2, N_mesh_1 = np.meshgrid(X, Y)
    print(N_mesh_1)
    print(N_mesh_2)
    
    Z = populate_mesh_grid(percentage_over)
    print(Z)
    
    
    plt.figure(figsize=(6, 5), dpi=400)
    
    # plt.xticks(ticks=x, labels=labels)
    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xlabel('column density N1')
    plt.ylabel('column density N2')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('% over 3 sigma residuals - distribution ' + option + "\n", fontsize = 10)
    plt.contourf(N_mesh_1, N_mesh_2, Z, 10)
    plt.colorbar()
        
        # return



get_data_all_files(RESIDUALS_PATH_NONLTE, 'non_lte_1')





