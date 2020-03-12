# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:49:00 2020

@author: rossc
"""

import pandas as pd
from HMM import unsupervised_HMM

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

def load_spenser_sonnets():
    sonnets = []
    with open ("spenser.txt", "r") as myfile:
        data=myfile.readlines()
        data = list(filter(lambda x: x != '\n', data))
        i = 1
        while i < len(data):
            sonnets.append(data[i:i+14])
            i += 15
        for i, sonnet in enumerate(sonnets):
            snt = ''
            for line in sonnet:
                snt += line.strip() + ' '
            snt = snt.replace(',', ' ,').replace('.', ' .').replace(':', ' :').replace(';', ' ;').replace('!', ' !').replace('?', ' ?')
            sonnets[i] = snt[:-1].lower().split()
    return sonnets

def sonnet_to_indices(s, whitespace=True):
    results = []
    if whitespace:
        s = s.replace('\n', ' ').replace(',', ' ,').replace('.', ' .').replace(':', ' :').replace(';', ' ;').replace('!', ' !').replace('?', ' ?')
        results = list(filter(lambda e: e != '', s.split(' ')))
    else:
        results = s
    for i, j in enumerate(results):
        if j in states:
            results[i] = int(states.index(j))
        elif j.strip("'()") in states:
            results[i] = int(states.index(j.strip("'()")))
    return results

if __name__=="__main__":
    syllables = generate_syllable_dict('Syllable_dictionary.txt')
    spenser_sonnets = load_spenser_sonnets()
    spenser_indexed_sonnets = [sonnet_to_indices(s, whitespace=False) for s in spenser_sonnets]
    for i, idx in enumerate(spenser_indexed_sonnets):
        spenser_indexed_sonnets[i] = list(filter(lambda x: type(x) == type(int(0)), idx))
    
    sonnets = generate_sonnet_list('shakespeare.txt')
    states = list(syllables.keys())
    
    for s in spenser_indexed_sonnets:
        for i in s:
            if i not in range(len(states)):
                print('Index out of bounds: %d' % i)
    
    
    
    indexed_sonnets = [sonnet_to_indices(s) for s in sonnets]
    all_sonnets = indexed_sonnets + spenser_indexed_sonnets
    hmm = unsupervised_HMM(indexed_sonnets, 10, 1000)
    print("Trained shakespeare")
    spenser_hmm = unsupervised_HMM(spenser_indexed_sonnets, 10, 1000, D=len(states))
    print("Trained spenser")
    all_hmm = unsupervised_HMM(all_sonnets, 10, 1000)
    print("Trained both")
    
    for i in range(1,4):
        print(hmm.generate_sonnet(states, syllables, "sonnet_shakespeare" + str(i) + ".txt"))
        print()
        print(spenser_hmm.generate_sonnet(states, syllables, "sonnet_spenser" + str(i) + ".txt"))
        print()
        print(all_hmm.generate_sonnet(states, syllables, "sonnet_all" + str(i) + ".txt"))
        print()
        print(hmm.generate_haiku(states, syllables, "haiku_shakespeare" + str(i) + ".txt"))
        print(spenser_hmm.generate_haiku(states, syllables, "haiku_spenser" + str(i) + ".txt"))
        print(all_hmm.generate_haiku(states, syllables, "haiku_all" + str(i) + ".txt"))
