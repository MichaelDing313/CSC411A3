import numpy as np
import os
from collections import OrderedDict
import re
from collections import Counter
from shutil import copy2
import math
import string
import heapq
from numpy import random
import tensorflow as tf
from pylab import *
import matplotlib.pyplot as plt
import random as rn



# read polarity dataset
def read_file():
    directories = ['pos','neg']
    keyword = []
    pos_dictionary = {}
    neg_dictionary = {}
    
    for directory in directories:
        path = 'review_polarity/'+'txt_sentoken/' + directory
        for txt_file in os.listdir(path):
            with open(path + '/' + txt_file) as f:

                file_word_list = f.read().lower()
                for pun in string.punctuation:
                    file_word_list = file_word_list.replace(pun," ")
                file_word_list = re.split('\W+', file_word_list)
                file_word_list.remove('')
                file__word_list = list(OrderedDict.fromkeys(file_word_list))

                #merge word_lists
                keyword += file_word_list
    
        store_word = {} # dictionary
        for word in keyword:
            if word not in store_word:
                store_word[word] = 0
            else:
                store_word[word] +=1
        if directory == directories[0]:
            pos_dictionary = store_word
        else:
            neg_dictionary = store_word
            
    return list(OrderedDict.fromkeys(keyword)),pos_dictionary,neg_dictionary

# path = 'review_polarity/'+'txt_sentoken/' + directory
def init_x_and_y_matrix(directories, keyword):
    x_matrix = np.zeros((0,len(keyword))) # []
    x2_matrix = np.zeros((1,len(keyword)))
    y_matrix = np.zeros((0,1))  # []
    
    directories = ['pos','neg']
    for directory in directories: 
        path = 'review_polarity/'+'txt_sentoken/' + directory   
        for txt_file in os.listdir(path):
            with open(path + '/' + txt_file) as f:
                file_word_list = f.read().lower()
                for pun in string.punctuation:
                    file_word_list = file_word_list.replace(pun," ")
                file_word_list = re.split('\W+', file_word_list)
                file_word_list.remove('')
                for word in file_word_list:
                    #x2_matrix first row
                    x2_matrix[0][keyword.index(word)] = 1
                x_matrix = np.vstack((x_matrix,x2_matrix))
                x2_matrix = np.zeros((1,len(keyword))) # reset for the next txt_file
        
            if (directory == 'pos'):
                y_matrix = np.vstack((y_matrix,1))
            else:
                # neg y = 0 
                y_matrix = np.vstack((y_matrix,0))
                
    return x_matrix,y_matrix
    

    
    
    
    
        
##      CALL
keyword,pos_dictionary,neg_dictionary = read_file()

x_matrix,y_matrix = init_x_and_y_matrix('review_polarity/', keyword) 


train_performance = []
vali_performance = []
test_performance = []
