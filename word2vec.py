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


## Reverse Dictionary Lookup
def d_look(dic, val):
    '''
    This Functoin looks up the index of a dictionary, used for reverse
    lookup of array index given a word
    '''
 
    return list(dic.keys())[list(dic.values()).index(val)]

###############################
## LOGISTIC REGRESSION STUFF ##
###############################


## Logistic Output Function
def hf(x, w = None):
    '''
    Hypothesis Function for Logistic Regression, aka sigmoid
    
    h(x) = 1 / (1 + exp( - theta.T * x)
    or
    h(x) = 1 / (1 + exp( -x)
    
    :param x:   Input vector    
    :oparam w:  Weight matrix, same dimension as input, INCLUDE BIAS IN w
                If w is not given, then no weights is used, simply use x to regress
                
    :returns:   Logistic hypothese of calssification of input
                given the weights
    
    '''
    if w:
        return 1/(1+np.exp(-np.dot(x, w)))
    else:
        return 1/(1+np.exp(-x))
    
## Logistic Cost Function
def cf(w, x, y, len = 1):
    '''
    Cost function for logistic regression, this cost funtion in this case is
    the sum of the euclidien distance of each component of the vector
    
    :param w:       Weight matrix, same dimension as input, INCLUDE BIAS IN w
    :param x:       Input vector
    :oparam len:    Length of the list used to normalize input weight, pass this
                    in instead of calculating from data each iteartion, much faster
                    Optional, default to 1 (no normalization)
    
    :returns:   Logistic hypothese of calssification of input
                given the weights
    '''
    cost_mat = hf(w,in_word) - ans
    cost = sum(np.square(cost_mat)) / len
    return cost
    
## Logistic Cost Gradient
def df(w,x,y,len = 1):
    '''    
    :param w:       Weight matrix, same dimension as input, INCLUDE BIAS IN w
    :param x:       Input vector
    :oparam len:    Length of the list used to normalize input weight, pass this
                    in instead of calculating from data each iteartion, much faster
                    Optional, default to 1 (no normalization)
    
    :returns:   "Direction to Descent" of the Logistic Cost Function
    '''
    return 2*sum((y-dot(w.T, x))*x, 1) / len

## Gradient Descent
def rgd(f,df, x, y, init_w, record_steps, learn_rate = 0.01 ,max_iter = 0.00001 ,step_lim = 400000):
    '''
    This is s generic Gradient Descent funtion to perform gradient descent given
    a cost function and a cost gradient. Supply learning rate and stop tolerance, 
    and step limit optionally.
    
    Default Values:
    learning rate = 0.01
    step tolerance = 0.00001
    step limit = 40,000
    
    :param f:   Function that returns the cost function given weight
    :param df:  Function that returns gradient descent vector of f
    :param x:   Input x (word representation in this case)
    :param y:   Result to train against
    :init_w:    Initial weight matrix to gradient descent on
    :record_steps:  Wether to record history of the cost overtime,
                    for training curve plotting
    
    :oparam learn_rate: Learning rate, alpha value for descent
    :oparam max_iter:   Terminateds gradient descent if difference between iteration
                        is less than stop_tol
    :oparam step_lim:   Step limit for gradient descent, stop desccent if total
                        iteration count is above this number
                        
    :returns:   Optimized weight matrix after gradient descent, 
                optinally, if recording is selected, output history of theta
    '''
    
    iter = 0
    w = init_w.copy()
    prev_w = w - 10*stop_tol
    len = shape(x)[1]
    history = []
    while norm(w - prev_w) >  stop_tol and iter < max_iter:
        prev_w = w.copy()
        w -= learn_rate * df(w,x,y,len)
        
        if i%10 == 0 and record_steps: history.append([cf(w,x,y,len), prev_w])
        if i%500 == 0: print("iteration:{} cost:{}".format(iter, cf(w,x,y,len)))
    
    return w, history

#####################################
## Data Set Reading / Manipulation ##
#####################################

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
            key_name = directory+txt_file.split("_")[0].replace('cv','')
            
            if directory == 'pos':
                pos_dic[key_name] = txt_str;
            else:
                neg_dic[key_name] = txt_str;
                
    log_event("End reading data",2)
    return pos_dic,neg_dic

## Collapse all lists in the dictonry into sets, only one occurance of each word is kept
def data_to_set(in_dic, set_name = ""):
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
def split_set(in_dic, set_name = "", train_size = 600, vali_size = 200, test_size = 200):    
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

## Count Occurances of All Words
def tally_counts(in_dic, in_keys = None, set_name = ""):
    '''
    This function takes in a input dictionary and counts occurance of each word
    in its content, the input must be converted to sets first. However, if input
    is not in set form, all occurances will be counted in the final tally.
    
    ** INITIALIZE RANDOM SEED BEFORE CALLING THIS FUNCTION **
    
    :param in_dic:    Input dictionary, contains all input data
    :oparam in_keys:  Input Keys set, use this to specify processing of a subset
                        of the dictionary data, such as for differen data sets.
                    
                        
    :returns:   A dictoinary object, that containes all words that have been used
                in the entire data set and how many occurances.
    '''
    log_event("Tallying Word Usage, {}".format(set_name),2)
    if in_keys == None: in_keys = in_dic.keys()
    
    word_set = set()
    
    for i in in_keys:
        # Loop Through Each Dictionary Entry, build big list of words
        word_set.update(in_dic[i])
    
    # Create a numpy array to store occurances, in the order corrosponding
    # to the big list of words
    
    count_dic = {k: 0 for k in word_set}
    
    for i in in_keys:
        # Loop through each dictionary entry
        for word in in_dic[i]:
            # Loop through each word in each input set
            # and increment count for that word
            count_dic[word] += 1
                

    log_event("Finished Word Usage Tally, {}".format(set_name),2)
                
    return count_dic

## Build context With Word Input
def build_context(in_dic, in_keys = None, set_name=''):
    '''
    This function builds the adjacency lists given input set of a range in a dictionary
    
    :param in_dic:      Input dictionary, contains all input data, must be in original order
    
    :oparam in_keys:    Input Keys set, use this to specify processing of a subset
                        of the dictionary data, such as for differen data sets.
    :oparam set_name:   Optional name used for identification, input a string
    
    
    :returns:           A dictionay with words as keys and a list of words where it is
                        part of its context
    
    
    '''
    
    log_event("Generating Word Context Data, {}".format(set_name),2)
    if in_keys == None: in_keys = in_dic.keys()
    
    word_set = set()
    
    for i in in_keys:
        # Loop Through Each Dictionary Entry, build big list of words
        word_set.update(in_dic[i])
    
    # Create a numpy array to store occurances, in the order corrosponding
    # to the big list of words
    context_dic = {k: set() for k in word_set}
    #import pdb; pdb.set_trace()
    
    for i in in_keys:
        # Loop through each input dictionary entry
        #import pdb; pdb.set_trace()
        curr_list = in_dic[i]
        
        for index, word in enumerate(curr_list):
            # Loop through each word in each input set
            # and add each word left or right of the word as context
            if index < (len(curr_list)-1):  context_dic[word].add(curr_list[index+1])
            if index > 0:                   context_dic[word].add(curr_list[index-1])
                

    log_event("Finished Generating Context, {}".format(set_name),2)
                
    return context_dic

## Fix inconsistency between supplied embedding and word set
def fix_words(w2i, wc):
    '''
    Fix inconsistency with supplied embedding and generated
    word adjacency lists
    '''
    
    log_event("Fixing Word Inconsistencies",2)

    # Make a list of valid that is present in both sets
    valid_words = set([str(i) for i in w2i.values()]).intersection(wc.keys())
    
    # Flip the order of dictionaries, since it is built backwards
    flip_w2i = dict(zip(w2i.values(), w2i.keys()))
    
    fixed_context = dict.fromkeys(valid_words)
    fixed_w2i = dict.fromkeys(valid_words)
    for i in valid_words:
        fixed_context[i] = wc[i]
        fixed_w2i[i] = flip_w2i[i]
        
    return fixed_w2i, fixed_context
    
    log_event("Finished Fixing Word Inconsistencies",2)

## Get Minibatch For Training
def get_batch(context_dic, embed_dic, batch_size):
    '''
    This function extracts a subset of the input data to train logistic regression
    returns two lists, the input word set and answers related to that word
    
    '''
    
    # Not specifying exactly how many positive and negative examples to use
    # because its much easier and faster and more reliabel to not
    
    # List of all words we are looking at
    word_list = context_dic.keys()
    
    ret = np.empty(batch_size,2)
    
    for i in range(batch_size):
        # Loop batch_size times, adding a data pair every time
        
        # Generate two words from the big list
        rand_word = random.sample(word_list, 2)
        
        # Check if these two words are adjacent
        if 
        
############
## PART 7 ##
############
def part7():
    pass



##########################
## EXECUTION BEGIN HERE ##
##########################

## Init code
log_level = "ALL"

part7()

log_event("-------------- PART 7 --------------------",9)
# read data
pos_data,neg_data = read_data()

# obtain train, validation and test data
random.seed(4115555)
pos_train, pos_vali, pos_test = split_set(pos_data)
neg_train, neg_vali, neg_test = split_set(neg_data)

# count training set word usage
count_pos = tally_counts(pos_data,pos_train)
count_neg = tally_counts(neg_data,neg_train)

# Build a combined dictionary containinting both negative and positive words
combined_dic = {}
combined_dic.update(pos_data)
combined_dic.update(neg_data)

# Build adjacency lists for all words
word_context = build_context(combined_dic,pos_train+neg_train,'combined')

# Load embeddings from 
embed = np.load("embeddings.npz")["emb"]
w2i = np.load("embeddings.npz")["word2ind"].flatten()[0]

w2i, word_context = fix_words(w2i, word_context)


