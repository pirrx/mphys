#======================================================================================
# This script is an example showing how to use the LineAnalysisScripting module with :
# - one component with the LTE model                                                  
#
# Output files in myDirOutput : myName_bestModel.lis, myName.lam, myName.dat, myName.png
# (see below for the definitions of myDirOutput and myName)
# 
# To run this script as it is, the only thing you need to specify is your python path (myPython variable).
# Cassis scripting documentation : http://cassis.irap.omp.eu/docs/script/README.html
#======================================================================================

import time
import ScriptEnvironment
import subprocess
from Range import Range
from LineAnalysisScripting import UserInputs
from Component import Component
from eu.omp.irap.cassis.properties import Software
from java.io import File

import os

#==============================================================================
# INPUTS
# =============================================================================

# Define the frequency ranges for the lines
Range.unit  = "km/s"      # Possible units are GHz and km/s, MHz, cm-1 and micrometer
r1                = Range(-30,30)             # or v1 = Range(-5, 10.0) if Range.unit = "km/s"

# Set some useful variables
plotName          = "one_comp"
speciesName         = "CH3CN"
myModel             = "1comp_lte"
myName              =  plotName + "_" + speciesName + "_model"

# To be changed for your own myDirInput & myDirOutput folders
cassisDataPath      = Software.getCassisPath() + "/delivery/data/"
myDirInput          = "/Users/aleks/OneDrive/Dokumenty/mphys/scripts/"
myDirOutput         = myDirInput
inputFile           = myDirInput + "G013.6562_ch3cn_ch3cch.asc"
outputFile          = myDirOutput + myName + ".dat"

# For contour plots of chi2 values, set the path for your python here 
# Required packages : pandas, numpy, scipy, matplotlib
myPython = "C:/Users/aleks/anaconda3/python38.zip"

myPythonScript = Software.getCassisPath()+"/delivery/script/examples/plot_chi2_RG.py"
# For more information, please open the file and read the explanation at the top of the script. 

# =============================================================================
# USER INPUTS
# =============================================================================
userInputs  = UserInputs(
inputFile   = inputFile,
outputFile  = outputFile,
#telescope   = {apex: [0, 1], hifi: [2, 3]},
telescope   = "alma_400m",
tuningRange = [238.8,239.257],            #  in GHz
tuningBand  = 60.0,                                 # in km/s
aijMin      = 0.0,
eup         = [0.0, 500.0],
kup         = ["*","*"],
template    = "Full CDMS",  # or "Full CDMS" or "Full JPL" or "Full VASTEL" etc...
moltags     = [41505],
tmb2ta      = False,
isoUnique   = False,
plotTitle   = myName,
warning	    = True,
continuum    = 0,   # value of the continuum adapted to your observations (exemple: continuum    = 3.4 or any file that you created)
observing_mode      = "PSw/DBSw",  #or observing_mode      = "FSw",

# Enter here the lines and the corresponding ranges to be taken into
# account in the computation. The lines are sorted by frequency.
selectedLines  = {"1-10": r1},         # or selectedLines       = {"1": v1, "2": v2, "3":v3, "4":v4},  if selection on a velocity range

# rmsLines data (in K)
rmsLines          = {"1-10": 0.200 },

# calibration for the lines 
calLines      = {"1-10": 0.0}
)

# =============================================================================

## MODEL INPUTS
# =============================================================================
# Type of models :
# LTE  : nmol, tex, fwhm, size, vlsr and iso if there are different moltags
# RADEX: nmol, collisionFile, n_collisioners, tkin, fwhm, size, vlsr and iso
# =============================================================================

# Parameters for the first component
# Note: setting the min and max the same for any parameters means that the parameter is in fact fixed
#       In this case since all the mins and maxs are the same, only one model is calculated
# =============================================================================
comp_1 		= Component(
# Needed for any model
nmol 		= {'min':1.1e17, 'max':1.1e17, 'nstep':10,  'log_mode':True},
temp 		= {'min':500.0,   'max':500.0,   'nstep':10,  'log_mode':False},  # warning: for radex, use a T range within the T range valid for the collision file 
fwhm 		= {'min':8.0,    'max':8.0,    'nstep':6,  'log_mode':False},
size 		= {'min':0.6, 'max':0.6,   'nstep':3,  'log_mode':False},
vlsr 		= {'min':-2.0,   'max':-2.0,   'nstep':3,  'log_mode':False},
iso               = {'min':60,     'max':60,     'nstep':10,  'log_mode':False},
interacting       = True,         #False
model             = "lte",           #  model             = "radex",
)
# =============================================================================


# Execution time beginning
timeStart           = time.time()

# Compute the chi2 min using the regular grid and write the data
userInputs.computeChi2MinUsingRG(comp_1)     # userInputs.computeChi2MinUsingRG(comp_1,comp_2)  

# CASSIS Execution time ending
timeEnd             = time.time()

print "Execution time = %.3f seconds." % (timeEnd - timeStart)

#======================================================================================#
# SAVE AND DISPLAY THE RESULTS                                                         #
#======================================================================================#

# A. Plot the best model and save the corresponding spectra and config files
bestLineModel = userInputs.plotBestModel(moltag = [28503,29501], overSampling=5) 
bestLineModel.saveConfig(File(myDirOutput+myName+".lam"))
bestPhysicalModels  = userInputs.getBestPhysicalModels()

# The next line writes out the spectrum given from the physical parameters. If 
# a grid of models had been run, then this would be the best fit model. 
userInputs.saveBestPhysicalModels(myDirOutput+myName+"_bestModel_lines.lis")

# ==============================================================================

# This just uses a system command to slightly alter the header lines on the file
os.system("sed s'|FreqLsb|// FreqLsb|' < "+myDirOutput+myName+"_bestModel_lines.lis | sort  > "+myDirOutput+myName+"_bestModel.lis")
