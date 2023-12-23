#!/usr/bin/env python
# coding: utf-8

"""
This file generates wordclouds for each of the metrics we chose.
Do note that we are assuming that the files are the ones generated with the 
previous scripts.
"""

from wordcloud import WordCloud
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# Set some variables
data_path = "Data/"
cosine_path = data_path + "df_cosine_complete.csv"
jaccard_path = data_path + "Data/df_jaccard_complete.csv"
correlation_path = data_path + "df_correlation_complete.csv"
info_path = data_path + "word_test_df.csv"


# Load the data
cosine = pd.read_csv(cosine_path, index_col=[0])
jaccard = pd.read_csv(jaccard_path, index_col=[0] )
correlation = pd.read_csv(correlation_path,  index_col=[0])
test_info = pd.read_csv(info_path, index_col=[0])


# Print the most similar words for latex
print(cosine.sort_values("similarity", ascending=False)["similarity"].head(10).to_latex(),
      jaccard.sort_values("similarity", ascending=False)["similarity"].head(10).to_latex(),
      correlation.sort_values("similarity", ascending=False)["similarity"].head(10).to_latex())


# Print the least similar words for latex
print(cosine.sort_values("similarity", ascending=True)["similarity"].head(10).to_latex(),
      jaccard.sort_values("similarity", ascending=True)["similarity"].head(10).to_latex(),
      correlation.sort_values("similarity", ascending=True)["similarity"].head(10).to_latex())


# Convert the dataframes into dictionaries to get the co-occurencing words as keys
dict_cosine = cosine.T.to_dict()
dict_jaccard = jaccard.T.to_dict()
dict_correlation = correlation.T.to_dict()


# Function to get the all the words that occur with a target word in a dictionary
def get_word_cloud(dic, word):
    '''
    This function returns the words that co-ocur with a given target word and
    their similarity score.
    
    Input:
        
        dic   a dictionary where:
              the keys are strings containing two words separated by a space
              the values denote the similarity between the two words
        
        word  the target word
        
    Output:
        A dictionary where the keys are words and the values are the
        co-ocurrence score of the key and the target word
    '''
    
    # Initialize an empty dictionary
    sim_word = dict()
    
    # For each pair of words
    for pairs in dic.keys():
        
        # We split the words
        lst = re.sub('[^A-Za-z]', ' ', pairs).strip().split()
        
        if word in lst:  
            
            # If the first word of the two is the target, we add it to the
            # output dictionary.
            if word == lst[0]:
                try:
                    W = test_info.loc[test_info["stem"] == lst[1]].index[0]
                except IndexError:
                    W = lst[1]
                sim_word[W] = dic[pairs]["similarity"]
                
            # If the second word of the two is the target, we add it to the
            # output dictionary.
            if word == lst[1]:
                try:
                    W = test_info.loc[test_info["stem"] == lst[0]].index[0]
                except IndexError:
                    W = lst[0]
                sim_word[W] = dic[pairs]["similarity"]
            sim_word[word] = 1
            
    return sim_word


# Generate the wordcloud for the jaccard similarity
figure(figsize=(10,10))
cloud_Jaccard = WordCloud(background_color="white", max_words=40,relative_scaling=0.5,normalize_plurals=False, colormap="Reds").generate_from_frequencies(get_word_cloud(dict_jaccard, "pizza"))
plt.imshow(cloud_Jaccard)
plt.axis('off')
#plt.show()
plt.savefig('jaccardWC.png', bbox_inches='tight')


# Generate the wordcloud for the correlation
figure(figsize=(10,10))
cloud_correlation = WordCloud(background_color="white", max_words=40,relative_scaling=0.5,normalize_plurals=False, colormap="Blues").generate_from_frequencies(get_word_cloud(dict_correlation, "pizza"))
plt.imshow(cloud_correlation)
plt.axis('off')
#plt.show()
plt.savefig('correlationWC.png', bbox_inches='tight')


# Generate the wordcloud for the cosine similarity
figure(figsize=(10,10))
cloud_cosine = WordCloud(background_color="white", max_words=40,relative_scaling=0.5,normalize_plurals=False, colormap="Greens").generate_from_frequencies(get_word_cloud(dict_cosine, "pizza"))
plt.imshow(cloud_cosine)
plt.axis('off')
#plt.show()
plt.savefig('cosineWC.png', bbox_inches='tight')

