# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:42:05 2022

@author: aleks
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from scipy.optimize import fmin



#reading in the file & cleaning it up
DATA = np.genfromtxt("G13_CH3CN_model_bestModel.lis", delimiter=' ',
                     comments='//')

print(DATA)