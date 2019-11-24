from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
from load_map import *

import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
import pandas as pd
import random
import nltk

STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
  return o_counts[word2wid[word]]

def tw_stemmer(word):
  '''Stems the word using Porter stemmer, unless it is a 
  username (starts with @).  If so, returns the word unchanged.

  :type word: str
  :param word: the word to be stemmed
  :rtype: str
  :return: the stemmed word

  '''
  if word[0] == '@': #don't stem these
    return word
  else:
    return STEMMER.stem(word)

def PMI(c_xy, c_x, c_y, N):
    '''Compute the pointwise mutual information using cooccurrence counts.
    
    :type c_xy: int 
    :type c_x: int 
    :type c_y: int 
    :type N: int
    :param c_xy: coocurrence count of x and y
    :param c_x: occurrence count of x
    :param c_y: occurrence count of y
    :param N: total observation count
    :rtype: float
    :return: the pmi value
    
    '''
    p = (c_xy * N) / (c_x * c_y)
    
    return log(p)/log(2)

#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")

def cos_sim(v0,v1):
    '''Compute the cosine similarity between two sparse vectors.
    
    :type v0: dict
    :type v1: dict
    :param v0: first sparse vector
    :param v1: second sparse vector
    :rtype: float
    :return: cosine between v0 and v1
    '''
    # We recommend that you store the sparse vectors as dictionaries
    # with keys giving the indices of the non-zero entries, and values
    # giving the values at those dimensions.
    
    #You will need to replace with the real function
    v0_keys = list(v0.keys())
    v1_keys = list(v1.keys())
    keys = set(v0_keys + v1_keys)
    inner = 0
    v0_norm = 0
    v1_norm = 0
    
    for key in keys:
        
        if key in v0_keys:
            A0 = v0[key]
        else:
            A0 = 0
            
        if key in v1_keys:
            A1 = v1[key]
        else:
            A1 = 0
            
        inner   += (A0 * A1)
        v0_norm += (A0 ** 2)
        v1_norm += (A1 ** 2)
    
    cos_sim = inner / ( sqrt(v0_norm) * sqrt(v1_norm) )
    #print(cos_sim)
    return cos_sim

def create_ppmi_vectors(wids, o_counts, co_counts, tot_count, normalize=False):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for wid0 in wids:
        ##you will need to change this
        vectors[wid0] = {}
        c_x = o_counts[wid0]
        for wid1 in co_counts[wid0].keys():
            c_y  = o_counts[wid1]
            c_xy = co_counts[wid0][wid1]
            vectors[wid0][wid1] = max( PMI(c_xy, c_x, c_y, tot_count), 0)
        
        if normalize:
            sums = sum(list(vectors[wid0].values()))
            try:
                for wid1 in co_counts[wid0].keys():
                    vectors[wid0][wid1] /= sums
            except:
                print(wid2word[wid0])
            
    return vectors

def read_counts(filename, wids):
    '''Reads the counts from file. It returns counts for all words, but to
    save memory it only returns cooccurrence counts for the words
    whose ids are listed in wids.
    
    :type filename: string
    :type wids: list
    :param filename: where to read info from
    :param wids: a list of word ids
    :returns: occurence counts, cooccurence counts, and tot number of observations
    '''
    o_counts = {} # Occurence counts
    co_counts = {} # Cooccurence counts
    fp = open(filename)
    N = float(next(fp))
    for line in fp:
        line = line.strip().split("\t")
        wid0 = int(line[0])
        o_counts[wid0] = int(line[1])
        if(wid0 in wids):
            co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
    return (o_counts, co_counts, N)

def print_sorted_pairs(similarities, o_counts, first=0, last=100):
    '''Sorts the pairs of words by their similarity scores and prints
    out the sorted list from index first to last, along with the
    counts of each word in each pair.
    
    :type similarities: dict 
    :type o_counts: dict
    :type first: int
    :type last: int
    :param similarities: the word id pairs (keys) with similarity scores (values)
    :param o_counts: the counts of each word id
    :param first: index to start printing from
    :param last: index to stop printing
    :return: none
    '''
    if first < 0: last = len(similarities)
    for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
        word_pair = (wid2word[pair[0]], wid2word[pair[1]])
        print("{:.2f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
                                             o_counts[pair[0]],o_counts[pair[1]]))

def freq_v_sim(sims):
    xs = []
    ys = []
    for pair in sims.items():
        ys.append(pair[1])
        c0 = o_counts[pair[0][0]]
        c1 = o_counts[pair[0][1]]
        xs.append(min(c0,c1))
    plt.clf() # clear previous plots (if any)
    plt.xscale('log') #set x axis to log scale. Must do *before* creating plot
    plt.plot(xs, ys, 'k.') # create the scatter plot
    plt.xlabel('Min Freq')
    plt.ylabel('Similarity')
    print("Freq vs Similarity Spearman correlation = {:.2f}".format(spearmanr(xs,ys)[0]))
    plt.show() #display the set of plots

def make_pairs(items):
    '''Takes a list of items and creates a list of the unique pairs
    with each pair sorted, so that if (a, b) is a pair, (b, a) is not
    also included. Self-pairs (a, a) are also not included.
    
    :type items: list
    :param items: the list to pair up
    :return: list of pairs
    
    '''
    return [(x, y) for x in items for y in items if x < y]


def corr(vector1, vector2):
    '''
    The vectors are dictionaries
    '''
    v1 = []
    v2 = []
    key1 = vector1.keys()
    key2 = vector2.keys()
    keys = set(list(key1) + list(key2))
    for key in keys:
        if key in key1:
            v1.append(vector1[key])
        else:
            v1.append(0)
        if key in key2:
            v2.append(vector2[key])
        else:
            v2.append(0)
    return pearsonr(v1,v2)[0]




#test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]
#test_words = ["love","@justinbieber","wife","husband"]
test_words = ["orange","apple","red","pineapple","purple", "pizza", "pasta", "mozzarella", "tacos", "hamburger", "wife", "father", "woman", "son"]
test_words = ["orange", "apple", "person", "pizza", "pasta"]
test_words = ["football", "volley", "sport", "ball", "stadium", "soccer", "bieber", "song", "justin"]
test_words  = ["justin", "bieber", "trump", "america"]
test_words = ["oil", "wine", "bread", "egg", "flour", "meat", "vegan", "sweet", "sour", "chocolate", "milk", "car", "train", "plane", "flight", "ticket"]
test_words = ["kind", "gentle", "nice", "awesome", "good","polite", "bad", "evil"]
test_words = ["justin", "bieber", "trump", "america", "food", "instagram", "orange","apple","red","pineapple","purple", "pizza", "pasta", "mozzarella", "oil", "wine", "bread", "egg", "flour", "meat", "vegan", "sweet", "sour", "chocolate", "milk", "car", "train", "plane", "flight", "ticket", "tacos", "hamburger", "wife", "father", "woman", "son"]

test_words = []
test_words_lab = []
fp = open('testwords.txt', "r", encoding="utf-8-sig")
for line in fp:
    line = line.strip("\n").split(",")
    print(line)
    label = line[0]
    words = [(x.strip().lower(),label) for x in line]
    test_words_lab += words
    test_words += [word[0] for word in test_words_lab]

stemmed_words = [tw_stemmer(w) for w in test_words]
all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them
#all_wids = random.sample(list(o_counts.keys()),1000)

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs 
wid_pairs = make_pairs(all_wids)


#read in the count information (same as in lab)
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/lab8/counts", all_wids)

#make the word vectors
vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N, normalize=True)

data = pd.DataFrame(vectors).T.fillna(0)
data.to_csv('example.csv')

#rint(data)
pca = PCA()
pca.fit(data)
pca.components_
print(sum(pca.explained_variance_ratio_[:2]))

import umap

embedding = umap.UMAP(n_neighbors=15,
                      min_dist=0.3,
                      metric='cosine').fit_transform(data)


data_pca = pca.fit_transform(data)
plt.scatter(data_pca[:,0], data_pca[:,1], alpha=0.7)

# compute cosine similarites for all pairs we consider
c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

print("Sort by cosine similarity")
print_sorted_pairs(c_sims, o_counts)


from sklearn.cluster import KMeans

kmeans = KMeans(4)
kmeans.fit(data_pca)
kmeans.predict(data_pca)




# Jaccard Similarity
## Estimate JS on PPMI vectors
import itertools

def JaccardSimilarity(wid_pairs, vectors):
    jaccards = []
    for k in wid_pairs:
        jaccards.append((1-nltk.jaccard_distance(set(vectors[k[0]].keys()), set(vectors[k[1]].keys())), wid2word[k[0]], wid2word[k[1]])) #1-distance
    return sorted(jaccards)

J = JaccardSimilarity(wid_pairs, vectors) #this metric gives the same importance to all the items in the intersection, but this is not the case!

    
# Comments: no way to distinguish between pairs of synonims and pairs of antynomis
        # make the vector shorter is not really helpful and the JS drops

# Jaccard Similarity 2
        
def JaccardSimilarityW(v1, v2):
    num = 0
    den = 0
    for k in set(list(v1.keys()) + list(v2.keys())):
        if k in v1.keys():
            v1_val = v1[k]
        else:
            v1_val = 0
        if k in v2.keys():
            v2_val = v2[k]
        else:
            v2_val = 0
        num+= min(v1_val,v2_val)
        den += max(v1_val,v2_val)
    return num/den

JS = {(wid0,wid1): JaccardSimilarityW(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
print_sorted_pairs(JS, o_counts)
#Comments: We like it. Performs better than the binary version and returns reasonable clusters (broad categories)


# PCA

pca = PCA()
pca.fit_transform(data)