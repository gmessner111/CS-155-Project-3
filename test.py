# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:49:00 2020

@author: rossc
"""

import numpy as np
import pandas as pd
from HMM import *

def generate_syllable_dict(filename):
    dict_words = pd.read_csv(filename, header = None)
    syllables = {}
    for i in range(len(dict_words)):
        row = dict_words[0][i]
        word = ''
        for j in range(len(row)):
            if(row[j]!=' '):
                word = word+row[j]
            else:
                syllable_list = []
                current_string = ''
                for j1 in range(j+1, len(row)):
                    if(row[j1] == ' '):
                        syllable_list.append(current_string)
                        current_string = ''
                    else:
                        current_string = current_string + row[j1]
                        if(j1 == len(row) - 1):
                            syllable_list.append(current_string)

                syllable_list_2 = []     
                for s in syllable_list:
                    if(len(s) == 1):
                        syllable_list_2.append(int(s))
                    else:
                        syllable_list_2.append(int(s[1]))
                syllables[word] = syllable_list_2
                break
    syllables[','] = [0]
    syllables['.'] = [0]
    syllables['!'] = [0]
    syllables['?'] = [0]
    syllables[':'] = [0]
    syllables[';'] = [0]
    return syllables

def generate_sonnet_list(filename):
    raw_data = pd.read_csv(filename,header=None,sep='\n')
    sonnets = []
    sonnet = ''
    for i in range(1,len(raw_data)):
        if(len(raw_data[0][i].strip()) < 4):
            sonnets.append(sonnet)
            sonnet = ''
        elif(i == len(raw_data) - 1):
            sonnet = sonnet + raw_data[0][i] + '\n'
            sonnets.append(sonnet)
        else:
            sonnet = sonnet + raw_data[0][i] + '\n'
    for i, sonnet in enumerate(sonnets):
        sonnets[i] = sonnet.lower()
    return sonnets

def sonnet_to_indices(s):
    s = s.replace('\n', ' ').replace(',', ' ,').replace('.', ' .').replace(':', ' :').replace(';', ' ;').replace('!', ' !').replace('?', ' ?')
    results = list(filter(lambda e: e != '', s.split(' ')))
    for i, j in enumerate(results):
        if j in states:
            results[i] = int(states.index(j))
        elif j.strip("'()") in states:
            results[i] = int(states.index(j.strip("'()")))
    return results

if __name__=="__main__":
    syllables = generate_syllable_dict('Syllable_dictionary.txt')
    sonnets = generate_sonnet_list('shakespeare.txt')
    states = list(syllables.keys())
    
    indexed_sonnets = [sonnet_to_indices(s) for s in sonnets]
    hmm = unsupervised_HMM(indexed_sonnets, 20, 1000)
    
    for i in range(5):
        print('#' * 70)
        print(hmm.generate_sonnet(states, syllables))