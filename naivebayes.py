#############
## IMPORTS ##
#############

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

############
## PART 1 ##
############

## Read Pol Data Set
def read_data(tar = False, root_path = "./"):
    '''
    This function will read the input data set, extracing the tar file if need
    Contents of each of the text files will be broken down into words, they will
    in the same order as the original document, all lower case.
    
    :param tar:     If the data set is packed in the tar file, functoin will 
                    try to extract the tar file is this is true
                    
    :param root_path:   Root path of which the files reside, this will be where
                        txt_sentoken/ folder or the tar file is.
                        
    :returns:       A dictionay containing the lists of words and their
                    corrosponding labels. They labels (keys) will be in the
                    format of: n_### or p_### for positive or negative sentiment
                    reviews.
    '''

    directories = ('pos','neg')
    keyword = []
    pos_dictionary = {}
    neg_dictionary = {}
    
    if tar:
        # If tar reading is selected, we need to extract data before proceeding
        try:
            tarf = tarfile.open(root_path + "review_polarity.tar.gz")
            log_event("Reading Tar File",1)
            tarf.extractall()
            tarf.close()
        except IOError:
            log_event("FAIL - tarfile not found :(",2)
        except:
            log_event("FAIL - Unknown Error. reading tar file",3)
            
    # Tar reading finished, now read the output folder to find all the files
    

    log_event("Reading Data",2)
    # loop though each directory ('pos/' and 'neg/')
    neg_dic = {}
    pos_dic = {}
    for directory in directories:
        # Construct path to each file, based on neg or pos
        path_to_file = root_path + 'txt_sentoken/' + directory + '/'
        
        # loop though each file in the directory and operate on them
        for txt_file in os.listdir(path_to_file):
            
            # Note using same variable naems so they over write in each step
            # will be a bit more space efficient since our input is big
            txt_str = open(path_to_file + txt_file,"rb")            # open file
            txt_str = txt_str.read().lower()                              # read all strings, convert to lower case
            txt_str = txt_str.replace('\n','')                      # Strip line feed character
            txt_str = txt_str.translate(None, string.punctuation)   # Strip all punctuations from string
            txt_str = txt_str.split()                               # split into list of words  
            
            # Extarct file number to be saved in dictionary key
            key_name = txt_file.split("_")[0].replace('cv','')
            
            if directory == 'pos':
                pos_dic[key_name] = txt_str;
            else:
                neg_dic[key_name] = txt_str;
                
    log_event("End reading data",2)
    return pos_dic,neg_dic

## Collapse all lists in the dictonry into sets, only one occurance of each word is kept
def data_to_set(in_dic):
    '''
    This function takes in a input dictionary and convert lists assigned to each
    key into sets, thus eliminating all duplicat words
    
    ** CONVERSION IS MADE IN PLACE, DICTIONARY IS NOT COPIED **
    
    :param in_dic:    Input dictionary, contains all input data
                    
    :returns:   Same dictionay, with contents converted to set
    
    '''

    log_event("Removed Word Duplicate, {}".format(set_name),2)
    # Read each, convert to set
    for keys in in_dic.keys():
        in_dic[keys] = set(in_dic[keys])
        
    return in_dic


## Break down data into traning, validation and test set    
def split_set(in_dic, set_name = "default", train_size = 600, vali_size = 200, test_size = 200):    
    '''
    This function takes in a input dictionary and outputs 3 seperate dictionaries.
    input is the dictionary containing all input data and outputs training,
    validation and test sets in 3 dictionaries, the key names are retained.
    
    ** INITIALIZE RANDOM SEED BEFORE CALLING THIS FUNCTION **
    
    :param in_dic:    Input dictionary, contains all input data
                    
                        
    :returns:   training, validatoin, test
                3 lists of dictionary keys of the randomly chosen set
    '''

    log_event("Split Input to Sets, {}".format(set_name),2)
    key_list = in_dic.keys()
    
    return key_list[0:train_size],key_list[train_size:train_size+vali_size], \
            key_list[train_size+vali_size:train_size+vali_size+test_size]


def tally_counts(in_dic, in_keys = None, set_name = "default"):
    '''
    This function takes in a input dictionary and counts occurance of each word
    in its content, the input must be converted to sets first. However, if input
    is not in set form, all occurances will be counted in the final tally.
    
    ** INITIALIZE RANDOM SEED BEFORE CALLING THIS FUNCTION **
    
    :param in_dic:    Input dictionary, contains all input data
                    
                        
    :returns:   A dictoinary object, that containes all words that have been used
                in the entire data set and how many occurances.
    '''
    log_event("Tallying Word Usage, {}".format(set_name),2)
    count_dic = {}
    if in_keys == None: in_keys = in_dic.keys()
    
    for i in in_keys:
        # Loop through each dictionary entry
        for word in in_dic[i]:
            # Loop through each word in each input set

            if word in count_dic.keys():
                # If this word is already in the dictionary, we increment count
                count_dic[word] += 1
            else:
                # If this word is new, initialize the counter for this word
                count_dic[word] = 1
                

    log_event("Finished Word Usage Tally, {}".format(set_name),2)
                
    return count_dic
    
    
    
    

## Part 1 function to run part 1 code
def part1():
    # find the top 3 words
    top_pos = []
    top_neg = []
    for i in range (3):
        top_pos.append(max(pos_dictionary,key=pos_dictionary.get))
        print top_pos[i], pos_dictionary[top_pos[i]]
        pos_dictionary[top_pos[i]] = 0
        top_neg.append(max(neg_dictionary,key=neg_dictionary.get))
        print top_neg[i], neg_dictionary[top_neg[i]]
        neg_dictionary[top_neg[i]] = 0
    print top_pos
    print top_neg

## END PART 1


##########################
## EXECUTION BEGIN HERE ##
##########################

## Init code
log_level = "ALL"
## END init code

# read data and convert to set
pos_data,neg_data = read_data()
data_to_set(pos_data)
data_to_set(neg_data)

# obtain train, validation and test data
random.seed(4115555)
pos_train, pos_vali, pos_test = split_set(pos_data)
neg_train, neg_vali, neg_test = split_set(neg_data)

# count training set word usage
count_pos = tally_counts(pos_data,pos_train)
count_neg = tally_counts(neg_data,neg_train)

# sort words by occurance
scount_pos = sorted(count_pos, key=count_pos.get, reverse=False)
scount_neg = sorted(count_pos, key=count_pos.get, reverse=False)






















# Stops program execution here
i = []
i[1]




keyword,pos_dictionary,neg_dictionary = read_file()
part1()


##      RESULTS
#the 41470
#the 76528
#a 20195
#a 38105
#and 19895
#and 35575    
##                   
             


##      CALL
pos_path = 'review_polarity/txt_sentoken/pos'
neg_path = 'review_polarity/txt_sentoken/neg'
# training 150
# vali&test 100 from pos, 100 from neg
randomly_make_set_folders(150,100,100,pos_path,neg_path)

##


##      CALL
train_keyword_set,train_pos_dic,train_neg_dic = read_each_set('train')
vali_keyword_set,vali_pos_dic,vali_neg_dic = read_each_set('vali')
test_keyword_set,test_pos_dic,test_neg_dic = read_each_set('test')

# #CAN ONLY CALL ONCE, BECUASE THE PREVIOUS CALL RESET THE MAX TO ZERO
# train_top_pos = []
# train_top_neg = []
# for i in range (3):
#     train_top_pos.append(max(train_pos_dic,key=train_pos_dic.get))
#     print train_top_pos[i],train_pos_dic[train_top_pos[i]]
#     train_pos_dic[train_top_pos[i]] = 0
#     train_top_neg.append(max(train_neg_dic,key=train_neg_dic.get))
#     print train_top_neg[i], train_neg_dic[train_top_neg[i]]
#     train_neg_dic[train_top_neg[i]] = 0
# print train_top_pos
# print train_top_neg

##
# keyword_list = keyword_set
def probability(keyword_list):
    pass  
                
                                

##      CALL
pos_prob = 0
neg_prob = 0
train_total_dic = {}
