# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:27:21 2023

@author: aleks
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


RESIDUALS_PATH = "C:\\Users\\aleks\\OneDrive\\Dokumenty\\mphys\\colden_noise_fwhm_temp\\residuals\\"

def get_filenames(directory_path):
    
    filenames = []
    
    for path in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, path)):
            filenames.append(path)
    return filenames


FILENAMES = get_filenames(RESIDUALS_PATH)
print(FILENAMES)