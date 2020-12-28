#!/usr/bin/env python
# coding: utf-8

"""
This file generates umap clusters for each of the metrics we chose.
Do note that we are assuming that the files are the ones generated with the 
previous scripts.
"""


# We import the relevant libraries
import umap

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib.lines import Line2D
from load_map import *


# Import the data generated from the other files
data = pd.read_csv("df_ppmi.csv", index_col=0)
word_data = pd.read_csv("word_test_df.csv", index_col=0)
word_data = word_data.replace("#arabspring", "ARAB SPRING")


# These parts will help us use the non-stemmed words in the plots
keys = word_data["stem"]
clusters = word_data["cluster"].unique()
names = ["correlation", "jaccard", "cosine"]
words = dict()
for n, i in enumerate(data.index):
    words[i] = wid2word[i]


# We make an UMAP for each of our metrics
for name in names:
    
    # Generate the umap embeddings
    embedding = umap.UMAP(n_neighbors=10,
                          min_dist=0.1,
                          metric=name).fit_transform(data)

    # Determine the colors for the classes
    fig,ax = plt.subplots(figsize=(25,15))
    colors = ["red","blue","green","orange","yellow","purple"]
    color_key = dict(zip(clusters,colors))
    
    # Find each datapoint and add it to the plot
    for n, i in enumerate(data.index):
        cluster = word_data.loc[word_data["stem"] == wid2word[i]]["cluster"].iloc[0]
        word = word_data.loc[word_data["stem"] == wid2word[i]].index[0]
        #print(cluster)
        plt.plot(embedding[n,0],embedding[n,1], alpha=0.7, linestyle='', ms=12, marker="o",
                 label=cluster, color=color_key[cluster])
        ax.annotate(word, (embedding[n,0], embedding[n,1]), fontsize=20)
    
    # Adding the legend to the plot
    custom_lines = [Line2D([0], [0], color=c, lw=4) for c in colors]
    plt.axis("off")
    ax.legend(custom_lines, clusters)
    
    # Show and/or save the plot
    plt.show()
    fig.savefig(name + "_clusters.png", bbox_inches='tight')


# We make a heatmap for each of our metrics
for name in names:
    
    # Load the data
    filepath = "df_" + name + ".csv"
    dist_data = pd.read_csv(filepath, index_col=0)    
    
    # Generate a numpy array with for the heatmap values
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
                        
    # Generate a seaborn heatmap
    corr = pd.DataFrame(corr)
    sns.heatmap(corr,
            xticklabels=False,
            yticklabels=False)
    
    # Show and/or save the plot
    plt.savefig(name + "_heatmap.png", bbox_inches='tight')
    plt.show()
