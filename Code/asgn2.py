from __future__ import division

from nltk.stem.porter import PorterStemmer, wid2word, word2wid, spearmanr
from scipy.stats.stats import pearsonr
from math import log,sqrt

from nltk.stem.porter import *
from nltk.stem import *
from load_map import *

import matplotlib.pyplot as plt
import pandas as pd

# Define the stemmer
STEMMER = PorterStemmer()


def w_count(word):
  """
  Helper function to get the count of a word (string)
  This was part of the code given as part of the assignment
  """
  return o_counts[word2wid[word]]


def tw_stemmer(word):
  '''
  Stems the word using Porter stemmer, unless it is a 
  username (starts with @).  If so, returns the word unchanged.
  
  Input:
      word  a string to be stemmed
      
  Output:
      the stemmed word
      
  This was part of the code given as part the assignment
  '''
  if word[0] == '@': #don't stem these
    return word
  else:
    return STEMMER.stem(word)


def PMI(c_xy, c_x, c_y, N):
    '''
    Compute the pointwise mutual information using cooccurrence counts.
    
    Input:
        
        c_xy coocurrence count of x and y
        
        c_x  occurrence count of x
        
        c_y  occurrence count of y
        
        N    total observation count
        
    Output:
        the PMI value
    '''
    p = (c_xy * N) / (c_x * c_y)
    return log(p)/log(2)


#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")


def create_ppmi_vectors(wids, o_counts, co_counts, tot_count, normalize=False):
    '''
    Creates sparse context vectors for the words in wids, using PPMI
    
    Input:
        
        wids      list of the ids of the words to make vectors for
        
        o_counts  a dictionary with the counts of each word, indexed by word id
        
        co_counts a dictionary of dictionaries with the cooccurrence counts of
                  each word pair, indexed by the ids of both words
                  
        tot_count an integer with the total number of observations
        
        normalize a boolean that determines whether to normalize or not
        
    Output:
        a dictionary with the ppmi vectors, indexed by word id
    '''
    
    # Initialize the vector dictionary
    vectors = {}
    
    # For each word id we passed, generate its vector
    for wid0 in wids:
        
        # Initialize the vector and get its occurence count
        vectors[wid0] = {}
        c_x = o_counts[wid0]
        
        # For each word it co-ocurs with, calculate the PMI and store the value
        for wid1 in co_counts[wid0].keys():
            c_y  = o_counts[wid1]
            c_xy = co_counts[wid0][wid1]
            vectors[wid0][wid1] = max( PMI(c_xy, c_x, c_y, tot_count), 0)
        
        # Normalize the vector if the normalize value is True
        if normalize:
            sums = sum(list(vectors[wid0].values()))
            try:
                for wid1 in co_counts[wid0].keys():
                    vectors[wid0][wid1] /= sums
            except:
                print(wid2word[wid0])
    
    # Return the vector       
    return vectors


def cos_sim(v0,v1):
    '''
    Compute the cosine similarity between two sparse vectors
    
    Input:
        Two sparse vectors stored as dictionaries
    
    Output:
        The cosine similarity between v0 and v1
    '''
    
    # Determine the keys of the vectors
    v0_keys = list(v0.keys())
    v1_keys = list(v1.keys())
    keys = set(v0_keys + v1_keys)
    
    # Initialize variables
    inner = 0
    v0_norm = 0
    v1_norm = 0
    
    # For each key
    for key in keys:
        
        # Store the value of v0 for that key
        if key in v0_keys:
            A0 = v0[key]
        else:
            A0 = 0
            
        # Store the value of v1 for that key
        if key in v1_keys:
            A1 = v1[key]
        else:
            A1 = 0
            
        # Add the corresponding part to each norm
        inner   += (A0 * A1)
        v0_norm += (A0 ** 2)
        v1_norm += (A1 ** 2)
    
    # Calculate the cosine similarity
    cos_sim = inner / ( sqrt(v0_norm) * sqrt(v1_norm) )
    return cos_sim


def read_counts(filename, wids):
    '''
    Reads the counts from a file. It returns counts for all words, but to save
    memory it only returns cooccurrence counts for the words whose ids are
    listed in wids.
    
    Input:
        
        filename  a string that states where to read info from
        
        wids      a list of word ids
        
    Outputs:
        occurence counts, cooccurence counts, and tot number of observations
      
    This was part of the code given as part the assignment and was not 
    commented further
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
  '''
  Sorts the pairs of words by their similarity scores and prints out the sorted
  list from index first to last, along with the counts of each word in each
  pair.

  Input:
      
      similarities dictionary with similarity scores indexed by word id pairs
      
      o_counts     dictionary with the counts of each word id
      
      first        index to start printing from
      
      last         index to stop printing
      
    This was part of the code given as part the assignment and was not 
    commented further
  '''
  if first < 0: last = len(similarities)
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    print("{:.2f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
                                         o_counts[pair[0]],o_counts[pair[1]]))


def return_sorted_pairs(similarities, o_counts, co_counts):
    '''
    Sorts the pairs of words by their similarity scores and prints out the 
    sorted list from index first to last, along with the counts of each word in
    each pair.

    Input:
      
      similarities dictionary with similarity scores indexed by word id pairs
      
      o_counts     dictionary with the counts of each word id
      
      first        index to start printing from
      
      last         index to stop printing
      
    This was part of the code given as part the assignment and was not 
    commented further
    '''
    pairs_values = dict()
    first = 0
    last = len(similarities)
    for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
        word_pair = (wid2word[pair[0]], wid2word[pair[1]])
        word_pair_str = str(word_pair)
        word_pair_values = dict()
        if pair[0] in co_counts[pair[1]]:
            co_count = co_counts[pair[0]][pair[1]]
        else:
            co_count = 0       
        word_pair_values["similarity"] = similarities[pair]
        word_pair_values["co_count"] = co_count
        word_pair_values["o_count_1"] = o_counts[pair[0]]
        word_pair_values["o_count_2"] = o_counts[pair[1]]
        pairs_values[word_pair_str] = word_pair_values
            
    return pairs_values


def freq_v_sim(sims):
    '''
    This function plots the words' frequencies against their similarities
      
    This was part of the code given as part the assignment and was not 
    commented further
    '''
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
    '''
    Takes a list of items and creates a list of the unique pairs with each pair
    sorted, so that if (a, b) is a pair, (b, a) is not also included.
    Self-pairs (a, a) are also not included.
    
    Input:
        items: the list to pair up
        
    Output:
        a list of pairs
    
    This was part of the code given as part the assignment and was not 
    commented further
    '''
    return [(x, y) for x in items for y in items if x < y]


#function to compute Jaccard similarity
def JaccardSimilarityW(v1, v2):
    '''
    Takes two PPMI vectors and returns the weighted Jaccard similarity
    '''
    
    # Initialize the counters
    num = 0
    den = 0
    
    keys = set(list(v1.keys()) + list(v2.keys()))
    
    # For each word in the vectors, update the Jaccard similarity
    for k in keys:
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
    
    # Return the similarity
    return num/den


def correlation(vector1, vector2):
    '''
    Takes two vectors and returns the Pearson correlation between them
    '''
    
    # Initialize non-sparse vectors
    v1 = []
    v2 = []
    
    # Get the set of all keys
    key1 = vector1.keys()
    key2 = vector2.keys()
    keys = set(list(key1) + list(key2))
    
    # If the current key appears, add its value to the non-sparse vector
    # else, append a 0
    for key in keys:
        if key in key1:
            v1.append(vector1[key])
        else:
            v1.append(0)
        if key in key2:
            v2.append(vector2[key])
        else:
            v2.append(0)
            
    # Return the correlation between the two non-sparse vectors
    return pearsonr(v1,v2)[0]


# PRELIMINARY task

# Stem the test words and obtain the ids, pairs, ocurrence counts, and
# co-ocurrence counts, as well as the total number of words
stemmed_words = [tw_stemmer(w) for w in test_words]
all_wids = set([word2wid[x] for x in stemmed_words]) 
wid_pairs = make_pairs(all_wids)
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/lab8/counts", all_wids)

# Generate the PMI vectors and get their similarities
vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N, normalize=False)
c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

# Print the word pairs, sorted by cosine similarity
print("Sort by cosine similarity")
return_sorted_pairs(c_sims, o_counts, co_counts)
print_sorted_pairs(c_sims, o_counts)

# Save the results to a csv
dictionary_pairs = return_sorted_pairs(c_sims, o_counts, co_counts)
preliminary_test = pd.DataFrame(dictionary_pairs).T
preliminary_test.to_csv("df_preliminary.csv")


# MAIN task

# Import our test words 
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
     
# Stem the our test words and obtain the ids, pairs, ocurrence counts, and
# co-ocurrence counts, as well as the total number of words
stemmed_words = [tw_stemmer(w) for w in test_words]
all_wids = set([word2wid[x] for x in stemmed_words]) 
wid_pairs = make_pairs(all_wids)
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/lab8/counts", all_wids)

# Generate the word vectors
vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N, normalize=True)

# Create ppmi dataframe
df_ppmi = pd.DataFrame(vectors).T.fillna(0)
df_ppmi.to_csv('df_ppmi.csv')

# Compute COSINE SIMILARITY for all pairs we consider
c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
#Store into a dataframe all the pairs, with similarity values, o_counts and co_counts info
dictionary_pairs = return_sorted_pairs(c_sims, o_counts, co_counts)
df_cosine = pd.DataFrame(dictionary_pairs).T
df_cosine.to_csv("df_cosine.csv")


# Compute JACCARD SIMILARITY for all pairs we consider
c_sims = {(wid0,wid1): JaccardSimilarityW(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
#Store into a dataframe all the pairs, with similarity values, o_counts and co_counts info
dictionary_pairs = return_sorted_pairs(c_sims, o_counts, co_counts)
df_jaccard = pd.DataFrame(dictionary_pairs).T
df_jaccard.to_csv("df_jaccard.csv")


# Compute CORRELATION for all pairs we consider
c_sims = {(wid0,wid1): correlation(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
#Store into a dataframe all the pairs, with similarity values, o_counts and co_counts info
dictionary_pairs = return_sorted_pairs(c_sims, o_counts, co_counts)
df_corr= pd.DataFrame(dictionary_pairs).T
df_corr.to_csv("df_corr.csv")