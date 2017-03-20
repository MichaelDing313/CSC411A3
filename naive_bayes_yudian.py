import numpy as np
import os
from collections import OrderedDict
import re
from collections import Counter
#import operator
from shutil import copy2
import math


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
    

## want randomly draw 200 training; 200 validation               
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
 
##
def count_word_prob(path):    
    count_words = 0  #count words appear probablity for all txt_files
    word_dic = {}
    word_prob_dic = {}  # store freq probability in a dictionary
    
    for txt_file in os.listdir(path):
        with open(path + '/' + txt_file) as f:
            file_word_list = re.split('\W+',f.read().lower())
            file_word_list.remove('')
            length = len(file_word_list) # each txt file word list length
            for word in file_word_list:
                if word not in word_prob_dic:
                    word_dic[word] = 0
                else:
                    word_dic[word] += 1
            for word in word_dic:
                if word_dic[word] == 0:
                    word_prob_dic[word] = float(1/length)
                else:
                    word_prob_dic[word] += float(word_dic[word]/length)

            count_words += length
    return count_words,word_prob_dic,word_dic
    

def prob_ai_given_class(word_list,count_words,word_prob_dic,word_dic,cls, m=0.01, k=10):
    # keywords : a1,a2,...
    # a1a2a3...an = exp(log(a1)+log(a2)+...log(an))    
    log = 0
    for word in word_list:
        if cls == 1:
            if word in word_dic:
                count_ai1_cls = word_dic[word]
            else:
                count_ai1_cls = 0
            
            count_cls = count_words    
            p_ai1_given_cls = math.log(((count_ai1_cls) + m*k)/(count_cls + k))
        else:
            if word in word_dic:
                count_ai1_cls = word_dic[word]
            else:
                count_ai1_cls = 0
            
            count_cls = count_words    
            p_ai1_given_cls = math.log(((count_ai1_cls) + m*k)/(count_cls + k))
        log += p_ai1_given_cls
    return log

def prob_cls_given_ai(word_list,count_words,word_prob_dic,word_dic,cls,prob_pos,prob_neg, m=0.01,k=10):
    log_p_ai_cls = prob_ai_given_class(word_list,count_words,word_prob_dic,word_dic,cls, m=0.01, k=10)
    if cls == 1:
        prob_class = prob_pos
    else:
        prob_class = prob_neg
    return log_p_ai_cls + math.log(prob_class)


##      CALL -> select all and uncommand to run
print '---part 1---\n'
# keyword,pos_dictionary,neg_dictionary = read_file()
# # find the top 3 words
# top_pos = []
# top_neg = []
# for i in range (3):
#     top_pos.append(max(pos_dictionary,key=pos_dictionary.get))
#     print 'pos',top_pos[i], pos_dictionary[top_pos[i]]
#     pos_dictionary[top_pos[i]] = 0
#     top_neg.append(max(neg_dictionary,key=neg_dictionary.get))
#     print 'neg',top_neg[i], neg_dictionary[top_neg[i]]
#     neg_dictionary[top_neg[i]] = 0
# print 'most frequently appeared word in positive reviews',top_pos
# print 'most frequently appeared word in negative reviews',top_neg


print '---part 2---\n'
print 'randomly select sets \n'
# pos_path = 'review_polarity/txt_sentoken/pos'
# neg_path = 'review_polarity/txt_sentoken/neg'
# # training 150
# # vali&test 100 from pos, 100 from neg
# randomly_make_set_folders(150,100,100,pos_path,neg_path)


count_words_train_pos, word_prob_dic_train_pos,word_dic_train_pos = count_word_prob('train/pos')
count_words_train_neg, word_prob_dic_train_neg,word_dic_train_neg = count_word_prob('train/neg')


prob_train_pos = prob_cls_given_ai(word_list,count_words,word_prob_dic,word_dic,cls,prob_pos,prob_neg, m=0.01,k=10)





