import numpy as np
import os

from collections import OrderedDict
import re
from collections import Counter

#import operator
from shutil import copy2

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
    
##      CALL -> uncommand to run part1
keyword,pos_dictionary,neg_dictionary = read_file()
# # find the top 3 words
# top_pos = []
# top_neg = []
# for i in range (3):
#     top_pos.append(max(pos_dictionary,key=pos_dictionary.get))
#     print top_pos[i], pos_dictionary[top_pos[i]]
#     pos_dictionary[top_pos[i]] = 0
#     top_neg.append(max(neg_dictionary,key=neg_dictionary.get))
#     print top_neg[i], neg_dictionary[top_neg[i]]
#     neg_dictionary[top_neg[i]] = 0
# print top_pos
# print top_neg

##      RESULTS
#the 41470
#the 76528
#a 20195
#a 38105
#and 19895
#and 35575    
##                   

# want randomly draw 200 training; 200 validation               
def randomly_make_set_folders(train_size, vali_size, test_size,pos_path,neg_path):
    sets = ['train','vali','test']
    directories = ['pos','neg']
    
    for set in sets:
        for directory in directories:
            txt_file_name = set + '/' + directory
            if not os.path.exists(txt_file_name):
                os.makedirs(txt_file_name) #train/pos; train/neg
                
    pos = os.listdir(pos_path)
    neg = os.listdir(neg_path)
    np.random.seed(0)
    one_dir = np.random.permutation(1000)   # each has 1000 txt_files
    #print 'one_dir', one_dir
    #print one_dir.size
    
    train_ind = one_dir[0:train_size]
    vali_ind = one_dir[(train_size):(train_size+vali_size)]
    test_ind = one_dir[(train_size+vali_size):(train_size+vali_size+test_size)]

    for x in range(len(train_ind)):
        copy2(pos_path + '/' + pos[train_ind[x]],'train/pos')
        copy2(neg_path + '/' + neg[train_ind[x]],'train/neg')
    for y in range(len(vali_ind)):
        copy2(pos_path + '/' + pos[train_ind[y]],'vali/pos')
        copy2(neg_path + '/' + neg[train_ind[y]],'vali/neg')    
    for z in range(len(test_ind)):
        copy2(pos_path + '/' + pos[test_ind[z]],'test/pos')
        copy2(neg_path + '/' + neg[test_ind[z]],'test/neg')

##      CALL
pos_path = 'review_polarity/txt_sentoken/pos'
neg_path = 'review_polarity/txt_sentoken/neg'
# training 150
# vali&test 100 from pos, 100 from neg
randomly_make_set_folders(150,100,100,pos_path,neg_path)

##
def read_each_set(path):    
    directories = ['pos','neg']
    keyword_set = []
    pos_dictionary = {}
    neg_dictionary = {}
    
    for directory in directories:
        for txt_file in os.listdir(path+'/'+directory):
            with open(path + '/' + directory + '/' + txt_file) as f:
                file_word_list = re.split('\W+',f.read().lower())
                file_word_list.remove('')
                #merge word_lists
                keyword_set += file_word_list
    
        store_word = {} # dictionary
        for word in keyword_set:
            if word not in store_word:
                store_word[word] = 0
            else:
                store_word[word] +=1
        
        if directory == directories[0]:
            pos_dictionary = store_word
        else:
            neg_dictionary = store_word    
    
    return keyword_set,pos_dictionary,neg_dictionary

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
                
                
                                

##      CALL
pos_prob = 0
neg_prob = 0
train_total_dic = {}
