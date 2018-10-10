'''
2018-09-28

Making a CSV file containing the ranges of time for which the given calibration is applicable.
'''

import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil
# For lab
sys.path.insert(0,"C:/Users/Helium1/Google Drive/Research/Lamb shift measurement/Code")
# For home
#sys.path.insert(0,"E:/Google Drive/Code/Python/Testing/Blah 3.7")
from exp_data_analysis import *
from KRYTAR_109_B_Calib_analysis import *
import re
import time
import math
import copy

import numpy.fft
import scipy.fftpack
import scipy.interpolate

import matplotlib.pyplot as plt
#%%
