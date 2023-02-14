# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:27:21 2023

@author: aleks
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


RESIDUALS_PATH = "D:\\UNIVERSITY\\YEAR 4\\MASTERS\\mphys repository\\mphys\\tables\\residual"
NOISE_N = []



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

def get_data_single_file(filename):
    """

    """
    file_path = RESIDUALS_PATH + filename
    print(file_path)
    data = np.genfromtxt(file_path, dtype='float', delimiter='\t', skip_header=3)
    print(data)
    frequency = data[:,0]
    intensity = data[:,3]
    return frequency, intensity


# FILENAMES = get_filenames(RESIDUALS_PATH)
# print(FILENAMES)

# ONE_COMP, TWO_COMP = get_comp_nr(FILENAMES)
# print(ONE_COMP)
# print(TWO_COMP)

FREQ, INT = get_data_single_file('Result subtract e14 e14 rpp50 two.fus')


def get_data_single(filenames):
    file_path = [RESIDUALS_PATH + x for x in filenames]
    

# filename = 'Result subtract e14 e14 rpp50 two.fus'
# file_path = RESIDUALS_PATH + filename
# print(file_path)
# data = np.genfromtxt(file_path, dtype='float', delimiter='\t', skip_header=3)
# print(data)
# print(len(data))

# could add specific noise value into the array of each spectra dataset