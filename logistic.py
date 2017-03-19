import numpy as np
import os

from collections import OrderedDict
import re
from collections import Counter

import operator

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
                file_word_list = re.split('\W+',f.read().lower())
#   >>> re.split('\W+', 'Words, words, words.')
# ['Words', 'words', 'words', '']
                file_word_list.remove('')
                #print file_word_list
                #merge word_lists
                keyword += file_word_list
    #print keyword
    
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
    
##    
keyword,pos_dictionary,neg_dictionary = read_file()
# find the top 3 words
top_pos = []
top_neg = []
for i in range (3):
    top_pos.append(max(pos_dictionary,key=pos_dictionary.get))
    pos_dictionary[top_pos[i]] = 0
    top_neg.append(max(neg_dictionary,key=neg_dictionary.get))
    neg_dictionary[top_neg[i]] = 0
top_pos
top_neg
    
    
    
# path = 'review_polarity/'+'txt_sentoken/' + directory
# want 200 training; 200 validation
def init_x_and_y_matrix(directories, keyword):

    x_matrix = np.zeros((0,len(keyword))) # []
    x2_matrix = np.zeros((1,len(keyword)))
    #y_matrix = np.array([])
    y_matrix = np.zeros((0,1))  # []
    
    directories = ['pos','neg']
    for directory in directories: 
        path = 'review_polarity/'+'txt_sentoken/' + directory   
        for txt_file in os.listdir(path):
            with open(path + '/' + txt_file ) as f:
                file_word_list = re.split('\W+',f.read().lower())
                file_word_list.remove('')
                for word in file_word_list:
                    #x2_matrix first row
                    x2_matrix[0][keyword.index(word)] = 1
                #print x2_matrix
                #print x1_matrix
                x_matrix = np.vstack((x_matrix,x2_matrix))
                x2_matrix = np.zeros((1,len(keyword))) # reset for the next txt_file
        
            if (directory == 'pos'):
                y_matrix = np.vstack((y_matrix,1))
            else:
                # neg y = 0 
                y_matrix = np.vstack((y_matrix,0))
                
#    print 'x_', x_matrix.shape
#    print 'y_', y_matrix.shape
    # x_ (2000, 39697)
    # y_ (2000, 1)
    return x_matrix,y_matrix

init_x_and_y_matrix('review_polarity/',keyword)                
                
        
