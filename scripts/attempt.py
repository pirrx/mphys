#======================================================================================
# This script is an attempt to use the LineAnalysisScripting module with :
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
Range.unit  = "GHz"      # Possible units are GHz and km/s, MHz, cm-1 and micrometer
f1                = Range(238.85,239.15)             # or v1 = Range(-5, 10.0) if Range.unit = "km/s"

# Set some useful variables
sourceName          = "G13"
speciesName         = "CH3CN"
myModel             = "1comp_lte"
myName              =  sourceName + "_" + speciesName + "_model"

# To be changed for your own myDirInput & myDirOutput folders
cassisDataPath      = Software.getCassisPath() + "/delivery/data/"
myDirInput          = "/Users/aleks/OneDrive/Dokumenty/mphys/two components/ch3cn/CASSIS data/colden/"
myDirOutput         = myDirInput
inputFile           = myDirInput + "N e15.fus"
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
telescope   = "alma_400m",
tuningRange = [238.85,239.15],            #  in GHz
#TuningBand  = 60.0,                                 # in km/s
aijMin      = 0.0,
eup         = [0.0, 1000.0],
kup         = ["*","*"],
template    = "All Species",  # or "Full CDMS" or "Full JPL" or "Full VASTEL" etc...
moltags     = [41001], #41001 - CH3CN All Species #41505 Full CDMS
tmb2ta      = False,
isoUnique   = False,
plotTitle   = myName,
warning	    = True,
continuum    = 0,   # value of the continuum adapted to your observations (exemple: continuum    = 3.4 or any file that you created)
observing_mode      = "PSw/DBSw",  #or observing_mode      = "FSw",

# Enter here the lines and the corresponding ranges to be taken into
# account in the computation. The lines are sorted by frequency.
selectedLines  = {"1": f1},         # or selectedLines       = {"1": v1, "2": v2, "3":v3, "4":v4},  if selection on a velocity range

# rmsLines data (in K)
rmsLines          = {"1": 0.200 },

# calibration for the lines 
calLines      = {"1": 0.0}
)






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
nmol 		= {'min':1.0e14, 'max':1.0e17, 'nstep':10,  'log_mode':True},
temp 		= {'min':50.0,   'max':300.0,   'nstep':10,  'log_mode':False},  # warning: for radex, use a T range within the T range valid for the collision file 
fwhm 		= {'min':2.0,    'max':10.0,    'nstep':6,  'log_mode':False},
size 		= {'min':0.2, 'max':1.3,   'nstep':3,  'log_mode':False},
vlsr 		= {'min':0.0,   'max':0.0,   'nstep':3,  'log_mode':False},
iso               = {'min':60,     'max':60,     'nstep':10,  'log_mode':False},
interacting       = False,         #True
model             = "lte",           #  model             = "radex",
)
# =============================================================================
# =============================================================================

# Execution time beginning
timeStart = time.time()

# Compute the chi2 min and write the data
userInputs.computeChi2MinUsingRG(comp_1,comp_2)      #userInputs.computeChi2MinUsingRG(comp_1)   

# Execution time ending
timeEnd = time.time()
print "execution time = %.3f seconds." % (timeEnd - timeStart)



#======================================================================================#
# SAVE AND DISPLAY THE RESULTS 
#======================================================================================#

# A. Plot the best model and save the corresponding spectra and config files
bestLineModel = userInputs.plotBestModel(moltag = 41001, overSampling=3)
bestLineModel.saveConfig(File(myDirOutput+myName+".lam"))
bestPhysicalModels  = userInputs.getBestPhysicalModels()
userInputs.saveBestPhysicalModels(myDirOutput+myName+"_bestModel.lis")

# B. Contour plots for chi2
trianglePlot        = [myPython+" "+myPythonScript+" "+outputFile]
subprocess.Popen(trianglePlot, shell=True)
# ==============================================================================




