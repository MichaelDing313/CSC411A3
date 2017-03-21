## PART 7, 8 and 9 word to Vec
## PLEAE RUN IN PYTHON 2.7

#############
## IMPORTS ##
#############

from pylab import *
import numpy as np
import os

from collections import OrderedDict
from collections import Counter
import re

#import operator
from shutil import copy2

# For reading and saving tar files
import tarfile

import random
import string
## END IMPORTS


####################
## MISC FUNCTIONS ##
####################

def log_event(log_string, level = "ALL"):
    '''
    This function handles info display in console, mainly here to unify logging
    and let me control the level of logging displayed
    
    :param log_string:  Log message to be displayed
    :param level:       Severity of event, higher the number more important
    '''
    # Assign preset log level to number
    if log_level == "ALL": log_lvl_local = 0
    elif log_level == "SHORT": log_lvl_local = 1
    elif log_level == "INFO": log_lvl_local = 2
    elif log_level == "WARNING": log_lvl_local = 3
    elif log_level == "NONE": log_lvl_local = 999
    else: log_lvl_local = log_level
    
    # Check log severity and skip function of no logging needed
    if level < log_lvl_local: return
    
    # Case if we want short logging, plrint a dot for each low level event
    # instead of printing full message.
    # Log level 1 allow full message when "ALL" and dot output when "SHORT"
    if (log_lvl_local == 1 and level == 1):
        print '.' ,
    else:
        print(log_string)

## END MISC FUNCTIONS



####################
## MISC FUNCTIONS ##
####################

def log_event(log_string, level = "ALL"):
    '''
    This function handles info display in console, mainly here to unify logging
    and let me control the level of logging displayed
    
    :param log_string:  Log message to be displayed
    :param level:       Severity of event, higher the number more important
    '''
    # Assign preset log level to number
    if log_level == "ALL": log_lvl_local = 0
    elif log_level == "SHORT": log_lvl_local = 1
    elif log_level == "INFO": log_lvl_local = 2
    elif log_level == "WARNING": log_lvl_local = 3
    elif log_level == "NONE": log_lvl_local = 999
    else: log_lvl_local = log_level
    
    # Check log severity and skip function of no logging needed
    if level < log_lvl_local: return
    
    # Case if we want short logging, plrint a dot for each low level event
    # instead of printing full message.
    # Log level 1 allow full message when "ALL" and dot output when "SHORT"
    if (log_lvl_local == 1 and level == 1):
        print '.' ,
    else:
        print(log_string)

## END MISC FUNCTIONS

############
## PART 7 ##
############

embed = np.load("embeddings.npz")["emb"]
w2i = np.load("embeddings.npz")["word2ind"].flatten()[0]



