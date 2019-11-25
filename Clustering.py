#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# We import the relevant libraries
import umap
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
from load_map import *


# In[ ]:


# Import the data generated from the other files
data = pd.read_csv("df_ppmi.csv", index_col=0)
word_data = pd.read_csv("word_test_df.csv", index_col=0)
word_data = word_data.replace("#arabspring", "ARAB SPRING")


# In[ ]:


# These parts will help us plot the plots with the non-stemmed words
keys = word_data["stem"]
clusters = word_data["cluster"].unique()
names = ["correlation", "jaccard", "cosine"]
words = dict()
for n, i in enumerate(data.index):
    words[i] = wid2word[i]


# In[ ]:


# We make an UMAP for each of our metrics
for name in names: 
    embedding = umap.UMAP(n_neighbors=10,
                          min_dist=0.1,
                          metric=name).fit_transform(data)

    fig,ax = plt.subplots(figsize=(25,15))
    colors = ["red","blue","green","orange","yellow","purple"]
    color_key = dict(zip(clusters,colors))
    for n, i in enumerate(data.index):
        cluster = word_data.loc[word_data["stem"] == wid2word[i]]["cluster"].iloc[0]
        word = word_data.loc[word_data["stem"] == wid2word[i]].index[0]
        #print(cluster)
        plt.plot(embedding[n,0],embedding[n,1], alpha=0.7, linestyle='', ms=12, marker="o",
                 label=cluster, color=color_key[cluster])
        ax.annotate(word, (embedding[n,0], embedding[n,1]), fontsize=20)
    custom_lines = [Line2D([0], [0], color=c, lw=4) for c in colors]
    plt.axis("off")
    ax.legend(custom_lines, clusters)
    plt.show()

    fig.savefig(name + "_clusters.png", bbox_inches='tight')


# In[ ]:


# We make a heatmap for each of our metrics
for name in names:
    
    filepath = "df_" + name + ".csv"
    dist_data = pd.read_csv(filepath, index_col=0)    
    
    L = len(keys)
    corr = np.zeros((L,L))
    for i,a in enumerate(keys):
        for j,b in enumerate(keys):
            if i == j:
                corr[i,j] = 0
            else:
                try:
                    corr[i,j] = dist_data.loc[str((a,b))]["similarity"]
                except KeyError:
                    try:
                        corr[i,j] = dist_data.loc[str((b,a))]["similarity"]
                    except KeyError:
                        corr[i,j] = 0
                        
    corr = pd.DataFrame(corr)
    sns.heatmap(corr,
            xticklabels=False,
            yticklabels=False)
    plt.savefig(name + "_heatmap.png", bbox_inches='tight')
    plt.show()


# In[ ]:




