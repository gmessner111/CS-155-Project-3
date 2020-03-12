# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:49:00 2020

@author: rossc
"""

import pandas as pd
from HMM import unsupervised_HMM

def generate_syllable_dict(filename):
    '''
    This function reads from the specified file and generates a dictionary
    of the words and their syllable counts. For words which can have a 
    different number of syllables at the end of a line, this is represented
    by a two element list, sorted in ascending order.
       
    returns: dictionary with word as key and syllable count(s) as value
    '''
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
    # Add in punctuation as zero syllable words, see our report for details
    syllables[','] = [0]
    syllables['.'] = [0]
    syllables['!'] = [0]
    syllables['?'] = [0]
    syllables[':'] = [0]
    syllables[';'] = [0]
    return syllables

def generate_sonnet_list(filename):
    '''
    Generate list of sonnets all in lower case. Note that this function only
    works for the shakespeare.txt file format.
    
    returs: list of strings where each string is an all lowercase sonnet
    '''
    raw_data = pd.read_csv(filename,header=None,sep='\n')
    sonnets = []
    sonnet = ''
    for i in range(1,len(raw_data)):
        if(len(raw_data[0][i].strip()) < 4):
            # This corresponds to a numbered heading
            sonnets.append(sonnet)
            sonnet = ''
        elif(i == len(raw_data) - 1):
            sonnet = sonnet + raw_data[0][i] + '\n'
            sonnets.append(sonnet)
        else:
            # Add the line to the sonnet
            sonnet = sonnet + raw_data[0][i] + '\n'
    for i, sonnet in enumerate(sonnets):
        # Conver to lowercase
        sonnets[i] = sonnet.lower()
    return sonnets

def load_spenser_sonnets():
    '''
    Load the sonnets from the spenser.txt file. This function returns the 
    sonnets of a list of lists where each interior list is a list of words as 
    strings.
    
    returns: list of lists of strings corresponding to sonnets brokenup by words
    '''
    sonnets = []
    with open ("spenser.txt", "r") as myfile:
        data=myfile.readlines()
        # Get rid of all the lines which are blank (i.e. newlines)
        data = list(filter(lambda x: x != '\n', data))
        i = 1
        while i < len(data):
            # Each sonnet is 14 lines plus a neumber line ahead of it
            sonnets.append(data[i:i+14])
            i += 15
        for i, sonnet in enumerate(sonnets):
            snt = ''
            for line in sonnet:
                # Convert the sonnet to one space delimited string
                snt += line.strip() + ' '
            # Separate punctuation on both sides by a space since we treat them
            # like words
            snt = snt.replace(',', ' ,').replace('.', ' .').replace(':', ' :').replace(';', ' ;').replace('!', ' !').replace('?', ' ?')
            # Make into a list of space separated words and lowercase them
            sonnets[i] = snt[:-1].lower().split()
    return sonnets

def sonnet_to_indices(s, whitespace=True):
    '''
    Convert the sonnets into lists of integers corresponding to word indices
    in our list of words.
    
    returns: list of lists or ints representing each word in the syllable dict
    '''
    results = []
    if whitespace:
        # Shakespeare data
        s = s.replace('\n', ' ').replace(',', ' ,').replace('.', ' .').replace(':', ' :').replace(';', ' ;').replace('!', ' !').replace('?', ' ?')
        results = list(filter(lambda e: e != '', s.split(' ')))
    else:
        results = s
    for i, j in enumerate(results):
        # If the word is in our list of dictionary keys, conver to index
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
        # Drop any words that we don't have syllable data for
        spenser_indexed_sonnets[i] = list(filter(lambda x: type(x) == type(int(0)), idx))
    
    sonnets = generate_sonnet_list('shakespeare.txt')
    states = list(syllables.keys())
    
    indexed_sonnets = [sonnet_to_indices(s) for s in sonnets]
    all_sonnets = indexed_sonnets + spenser_indexed_sonnets
    
    # Train unsupervised hmms on the three training sets
    hmm = unsupervised_HMM(indexed_sonnets, 10, 1000)
    print("Trained shakespeare")
    spenser_hmm = unsupervised_HMM(spenser_indexed_sonnets, 10, 1000, D=len(states))
    print("Trained spenser")
    all_hmm = unsupervised_HMM(all_sonnets, 10, 1000)
    print("Trained both")
    
    # Generate three emissions for each hmm and each type of poem and save to file
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
