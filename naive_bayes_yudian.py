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

    for txt_file in os.listdir(path):
        with open(path + '/' + txt_file) as f:

            file_word_list = f.read().lower()
            for pun in string.punctuation:
                file_word_list = file_word_list.replace(pun," ")
            file_word_list = re.split('\W+', file_word_list)
            file_word_list.remove('')
            file_word_list = list(OrderedDict.fromkeys(file_word_list))


            length = len(file_word_list) # each txt file word list length
            for word in file_word_list:
                if word not in word_dic:
                    word_dic[word] = 1.0/length
                else:
                    word_dic[word] += 1.0/length
            count_words += 1.0/length
    return word_dic, count_words

##
def prob_C_given_word(cls, word, m, k):
    word_list = [word]
    prob_word = math.log(train_total_dic[word]/count_train_total)
    prob_word_given_C = log_prob_C_given_words(word_list, m, k, cls)
    return prob_word_given_C - prob_word


def log_prob_C_given_words(word_list, m, k, cls):
    prob_words_cls = add_logs_prob_words_given_C(word_list, m, k, cls)
    if cls == 1:
        prob_C = prob_pos
    else:
        prob_C = prob_neg
    return prob_words_cls + math.log(prob_C)
    
    
def add_logs_prob_words_given_C(word_list, m, k, cls):
    sum = 0
    for word in word_list:
        if cls == 1:
            if word in train_pos_dic:
                word_count = train_pos_dic[word]
            else:
                word_count = 0
            prob_word = math.log((word_count + m * k)/(count_train_pos + k))
        else:
            if word in train_neg_dic:
                word_count = train_neg_dic[word]
            else:
                word_count = 0
            prob_word = math.log((word_count + m * k)/(count_train_neg + k))
        sum += prob_word
    return sum


def predict_review(word_list, m, k):
    prob_pos_review = log_prob_C_given_words(word_list, m, k, 1)
    prob_neg_review = log_prob_C_given_words(word_list, m, k, 0)
    if prob_pos_review >= prob_neg_review:
        return 1
    else:
        return 0
        
        
def performance(path,cls,m,k):
    correct_count = 0
    total = len(os.listdir(path))    
    for txt_file in os.listdir(path):
        f = open(path + '/' + txt_file)

        file_word_list = f.read().lower()
        for pun in string.punctuation:
            file_word_list = file_word_list.replace(pun," ")
        file_word_list = re.split('\W+', file_word_list)
        file_word_list.remove('')
        file_word_list = list(OrderedDict.fromkeys(file_word_list))

        predict = predict_review(file_word_list, m, k)
        if (predict == cls):
            correct_count += 1
    print 'correct count:', correct_count
    print 'out of total', total
    print(path+" performance: "+str(correct_count*100/total)+"%\n")
    return correct_count,total

    
def get_general_performance(path1,path2,m,k):
    pos_correct_count,pos_total = performance(path1,1,m,k)
    neg_correct_count,neg_total = performance(path2,0,m,k)    
    return ((pos_correct_count+neg_correct_count) * 100 / (pos_total+neg_total))    


##      prepare to call
def part1():
    keyword,pos_dictionary,neg_dictionary = read_file()
    # find the top 3 words
    toprob_pos = []
    toprob_neg = []
    for i in range (3):
        toprob_pos.append(max(pos_dictionary,key=pos_dictionary.get))
        print 'pos',toprob_pos[i], pos_dictionary[toprob_pos[i]]
        pos_dictionary[toprob_pos[i]] = 0
        toprob_neg.append(max(neg_dictionary,key=neg_dictionary.get))
        print 'neg',toprob_neg[i], neg_dictionary[toprob_neg[i]]
        neg_dictionary[toprob_neg[i]] = 0
    print 'most frequently appeared word in positive reviews',toprob_pos
    print 'most frequently appeared word in negative reviews',toprob_neg


def part2_creat_sets():
    print 'randomly select sets \n'
    pos_path = 'review_polarity/txt_sentoken/pos'
    neg_path = 'review_polarity/txt_sentoken/neg'
    # training 800
    # vali&test 100 from pos, 100 from neg
    randomly_make_set_folders(800,100,100,pos_path,neg_path)

def part2_part3():
    global train_pos_dic, train_neg_dic, train_total_dic
    global count_train_pos, count_train_neg, count_train_total
    global prob_pos, prob_neg
    
    prob_pos, prob_neg = 0, 0
    
    train_pos_dic = {}
    train_neg_dic = {}
    train_total_dic = {}
    
    count_train_pos = 0
    count_train_neg = 0
    count_train_total = 0
    
    m = 0.1
    print 'm =',m
    k = 50
    print 'k =',k
    
    train_pos_dic, count_train_pos = count_word_prob ('train/pos')
    train_neg_dic, count_train_neg = count_word_prob ('train/neg')
   
    train_total_dic = {k: train_pos_dic.get(k, 0) + train_neg_dic.get(k, 0) for k in set(train_pos_dic) | set(train_neg_dic)}
    
    count_train_total = float(count_train_pos + count_train_neg)
    
    prob_pos = count_train_pos / count_train_total
    prob_neg = count_train_neg / count_train_total
    
    train_accuracy = get_general_performance('train/pos','train/neg', m, k)
    val_accuracy = get_general_performance('vali/pos','vali/neg', m, k)
    test_accuracy = get_general_performance('test/pos','test/neg', m, k)
    
    print ("Training performance: "+str(train_accuracy)+"%")
    print ("Validation performance: "+str(val_accuracy)+"%")
    print ("Test performance: "+str(test_accuracy)+"%")
    
    
    print '\n part3-----'
    h_pos = []
    h_neg = []
    
    for word in train_pos_dic:
        prob_pos_given_word = prob_C_given_word(1, word, m, k)
        heapq.heappush(h_pos, (prob_pos_given_word, word))
    
    for word in train_neg_dic:
        prob_neg_given_word = prob_C_given_word(0, word, m, k)
        heapq.heappush(h_neg, (prob_neg_given_word, word))
        
    top10_pos = heapq.nlargest(10, h_pos)
    top10_neg = heapq.nlargest(10, h_neg)
    
    print 'top 10 pos predicted:', ([voca[1] for voca in top10_pos])
    print '\n'
    print 'top 10 neg predicted:', ([voca[1] for voca in top10_neg])


##  let's call
# print '---run - part 1---\n'
part1()
# print '---run - part 2 ---\n'
# part2_creat_sets()
part2_part3()



