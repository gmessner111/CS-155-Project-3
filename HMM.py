########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random


class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]
        
        # Start with base case of the A_start probabilities (uniform in 
        # this case) and multiply them by the probability of having
        # x[0] emitted from each of the states.
        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]
        
        for j in range(2, M + 1):
            for i in range(self.L):
                # Compute the max probability of getting to the state j, i
                prev = [probs[j-1][k] * self.A[k][i] * self.O[i][x[j-1]] for k in range(self.L)]
                probs[j][i] = max(prev)
                max_index = prev.index(probs[j][i])
                # append the most recent state to the state sequence
                seqs[j][i] = seqs[j - 1][max_index] + str(max_index)

        # Get the final state and append it to the state sequence
        max_prob_idx = probs[M].index(max(probs[M]))
        max_seq = seqs[M][max_prob_idx] + str(max_prob_idx)
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        
        # Update alphas based on recursive definition
        for state in range(self.L):
            alphas[1][state] = self.A_start[state] * self.O[state][x[0]]
        
        for j in range(1, M):
            for state in range(self.L):
                alphas[j + 1][state] = sum(alphas[j][k] * self.A[k][state] * self.O[state][x[j]] for k in range(self.L))
            if normalize:
                # Normalization just requires summing to 1 for each column
                norm_factor = sum(alphas[j+1])
                if norm_factor != 0:
                    for state in range(self.L):
                        alphas[j+1][state] /= norm_factor

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Update betas based on recursive definition
        betas[M] = [1. for _ in range(self.L)]
        
        for i in range(M - 1, 0, -1):
            for z in range(self.L):
                betas[i][z] = sum([betas[i+1][k] * self.A[z][k] \
                      * self.O[k][x[i]] for k in range(self.L)])
            if normalize:
                # Normalization requires columns to sum to 1
                norm_factor = sum(betas[i])
                if norm_factor != 0:
                    for k in range(self.L):
                        betas[i][k] /= norm_factor

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        for a in range(self.L):
            for b in range(self.L):
                num, denom = 0, 0
                for j in range(len(Y)):
                    for i in range(len(Y[j]) - 1):
                        num += 1 if Y[j][i+1] == a and Y[j][i] == b else 0
                        denom += 1 if Y[j][i] == b else 0
                self.A[a][b] = num / denom

        # Calculate each element of O using the M-step formulas.

        for w in range(self.L):
            for z in range(self.D):
                num, denom = 0, 0
                for j in range(len(Y)):
                    for i in range(len(Y[j])):
                        num += 1 if X[j][i] == z and Y[j][i] == w else 0
                        denom += 1 if Y[j][i] == w else 0
                self.O[w][z] = num / denom
                
    def printAO(self):
        # Print the transition matrix.
        print("Transition Matrix:")
        print('#' * 70)
        for i in range(len(self.A)):
            print(''.join("{:<12.3e}".format(self.A[i][j]) for j in range(len(self.A[i]))))
        print('')
        print('')
    
        # Print the observation matrix. 
        print("Observation Matrix:  ")
        print('#' * 70)
        for i in range(len(self.O)):
            print(''.join("{:<12.3e}".format(self.O[i][j]) for j in range(len(self.O[i]))))
        print('')
        print('')


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''  
        import numpy as np
        for _ in range(N_iters):
            
            # Matrices to keep track of the numerators and denominators
            # of the expressions summing over marginals
            A_top = np.zeros((self.L, self.L))
            A_bottom = np.zeros((self.L, 1))
            O_top = np.zeros((self.L, self.D))
            O_bottom = np.zeros((self.L, 1))
            
            for x in X:
                
                # Calculate the alphas and betas
                alpha = self.forward(x, normalize=True)
                beta = self.backward(x, normalize=True)
                M = len(x)
                
                for i in range(1, M+1):
                    # Compute and normalize the one element marginals for each i
                    probs = np.zeros((self.L, 1))
                    for a in range(self.L):
                        probs[a] = alpha[i][a] * beta[i][a]
                    tot = sum(probs)
                    probs = np.divide(probs, tot)
                    
                    # add them to the appropriate indices in all matrices
                    for h in range(self.L):
                        if i < M:
                            A_bottom[h] += probs[h]
                        O_top[h][x[i-1]] += probs[h]
                        O_bottom[h] += probs[h]
                    
                for i in range(1, M):
                    # Compute and normalize the two element marginals for each i
                    two_prob = np.zeros((self.L, self.L))
                    for a in range(self.L):
                        for b in range(self.L):
                            two_prob[a][b] = alpha[i][a] * self.A[a][b] * self.O[b][x[i]] * beta[i+1][b]
                    tot = 0
                    for k in range(len(two_prob)):
                        tot += sum(two_prob[k])
                    two_prob = np.divide(two_prob, tot)
                    
                    # add them to the appropriate indices in all matrices
                    for a in range(self.L):
                        for b in range(self.L):
                            A_top[a][b] += two_prob[a][b]
            
            # Update A and O by dividing the numerators by denominators
            self.A = np.divide(A_top, A_bottom)
            self.O = np.divide(O_top, O_bottom)


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        
        # Choose start state uniformly at random
        s = random.choice(range(self.L))
        
        # Choose an emission from this start state according to the O matrix
        emission.append(random.choices(range(self.D), weights=self.O[s])[0])
        
        # append the state
        states.append(s)
        
        for k in range(M-1):
            # Choose next state from current state according to the A matrix
            s = random.choices(range(self.L), weights=self.A[s])[0]
            
            # Choose an emission from this state according to the O matrix
            emission.append(int(random.choices(range(self.D), weights=self.O[s])[0]))
            
            # append the state
            states.append(s)

        return emission, states
    
    def generate_sonnet(self, idx_to_word, syllable_dict):
        punctuation = [',','.',':',';','!','?']
        sonnet = ''
        state = random.choice(range(self.L))
        for i in range(14):
            
            num_syllables = 0
            while (num_syllables != 10):
                emission_idx = random.choices(range(self.D), weights=self.O[state])[0]
                emission_word = idx_to_word[emission_idx]
                if emission_word in punctuation:
                    if num_syllables == 0 or sonnet[-2] in punctuation:
                        continue
                    else:
                        sonnet = sonnet[:-1]
                if (num_syllables + int(syllable_dict[emission_word][-1]) <= 10):
                    if num_syllables == 0:
                        if len(emission_word) > 1:
                            sonnet += emission_word[0].upper() + emission_word[1:] + ' '
                        else:
                            sonnet += emission_word.upper() + ' '
                    else:
                        if emission_word == 'i':
                            emission_word = 'I'
                        sonnet += emission_word + ' '
                    if emission_word == 'I':
                        emission_word = 'i'
                    num_syllables += int(syllable_dict[emission_word][-1])
                    state = random.choices(range(self.L), weights=self.A[state])[0]
            
            sonnet += '\n'
        sonnet = sonnet[:-2] + '.'
        
        for i in range(2,len(sonnet)):
            prev = sonnet[i-2]
            if prev == '.' or prev == '?' or prev == '!':
                if i+1 < len(sonnet):
                    sonnet = sonnet[:i] + sonnet[i].upper() + sonnet[i+1:]
                else:
                    sonnet = sonnet[:i] + sonnet[i].upper()
        return sonnet
        


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
