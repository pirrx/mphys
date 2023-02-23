# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:27:21 2023

@author: aleks
"""

#####

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from path_file import RESIDUALS_PATH, TABLES_PATH
import matplotlib.ticker as mtick
# from weights import CHI

NOISE_N_FILENAME = "noise_colden.csv"
NOISE_T_FILENAME = "noise_extemp.csv"


print(RESIDUALS_PATH)
print(TABLES_PATH)

labels = [r'$10^{14}, 10^{14}$',
          r'$10^{14}, 5\times10^{14}$',
          r'$10^{14}, 10^{15}$',
          r'$10^{14}, 10^{16}$',
          r'$10^{14}, 10^{17}$',
          
          
         r'$10^{15}, 5\times10^{14}$',
          r'$10^{15}, 10^{15}$',
          r'$10^{15}, 5\times10^{15}$',
          r'$10^{15}, 10^{16}$',
          r'$10^{15}, 10^{17}$',

          r'$10^{16}, 10^{16}$',
          r'$10^{16}, 5\times10^{15}$',
          r'$10^{16}, 5\times10^{16}$',
          r'$10^{16}, 10^{17}$',
          
          r'$10^{17}, 10^{17}$',
          r'$10^{17}, 5\times10^{17}$',
          ]




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

def get_comp_nr(filenames):
    "separates one and two comp filenames"
    
    two_comp_residuals = []
    one_comp_residuals = []

    one_comp_residuals = [i for i in filenames if 'one' in i]
    two_comp_residuals = [i for i in filenames if 'two' in i]

    return one_comp_residuals, two_comp_residuals

def get_noise(filename):
    file_path = TABLES_PATH + filename
    data = np.genfromtxt(file_path, dtype='float', delimiter=';', skip_header=1)
    noise = data[:,1]
    return noise


def get_data_single_file(filename):
    """

    """
    file_path = RESIDUALS_PATH + filename
    # print(file_path)
    data = np.genfromtxt(file_path, dtype='float', delimiter='\t', skip_header=3)
    # print(data)
    frequency = data[:,0]
    intensity = data[:,4]
    return frequency, intensity

def root_mean_square(array):
    return np.sqrt(np.mean(array**2))


FILENAMES = get_filenames(RESIDUALS_PATH)
# print('filenames')
# print(FILENAMES)

ONE_COMP, TWO_COMP = get_comp_nr(FILENAMES)
# print(ONE_COMP)
# print(TWO_COMP)

FREQ, INT = get_data_single_file('a. Result subtract e14 e14 rpp50 two.fus')
NOISE_N = get_noise(NOISE_N_FILENAME)
# print(NOISE_N)


def get_data_all_files(filenames, noise_file):
    # file_path = [RESIDUALS_PATH + x for x in filenames]
    # print(file_path)
    noise = get_noise(noise_file)
    index = 0
    nr_over_limit_values = []
    length = 0

    
    for file in filenames:
        noise_value = noise[index]
        print(index)
        freq, intensity = get_data_single_file(file)  
        noise_value = root_mean_square(intensity[0:175])         
        [i] = np.where((3*noise_value) < np.abs(intensity))        
        nr_over_limit_values.append(len(i))
        
        plt.figure()
        plt.scatter(freq, intensity, 0.5)
        plt.scatter(freq[i], intensity[i], 0.5)
        plt.show()
        
        length = len(intensity)

        index += 1
        
    print(nr_over_limit_values)
    x = np.linspace(0, len(noise), len(noise))
    print(noise)
    
    
    percentage_over = [x / length for x in nr_over_limit_values]
    
    plt.figure(figsize=(21, 5), dpi=400)
    plt.xticks(ticks=x, labels=labels)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xlabel('column density components N1, N2')
    plt.ylabel('percentage points away from 3 sigma  ')
    plt.scatter(x, percentage_over)
        
        # return

get_data_all_files(TWO_COMP, NOISE_N_FILENAME)





    

# filename = 'Result subtract e14 e14 rpp50 two.fus'
# file_path = RESIDUALS_PATH + filename
# print(file_path)
# data = np.genfromtxt(file_path, dtype='float', delimiter='\t', skip_header=3)
# print(data)
# print(len(data))

# could add specific noise value into the array of each spectra dataset