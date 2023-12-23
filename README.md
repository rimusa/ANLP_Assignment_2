# Word Similarity

This project was made by [Antonia Donvito](https://github.com/Antos23) and [Ricardo Muñoz Sánchez](https://github.com/rimusa). It is based on the second assignment for the Accelerated Natural Language Processing (ANLP)  course imparted at the University of Edinburgh by Sharon Goldwater and Shay Cohen during the Autumn 2019 semester.


## Assignment

The idea of the project was to study different ways in which words can be considered similar.
In order to do this, we used the co-occurrence of words in Tweets as our sparse representations, from which we would then compute the similarity using several metrics. 
The project consisted of both a preliminary and a main task to explore different similarity metrics.

The preliminary task was to compute the cosine similarity between the words in a given list. The results of this task can be found on Table 3 of our report.

The main task was to study different kinds of word similarities. We decided to use three metrics (cosine similarity, Jaccard similarity, and correlation) and to visualize the differences between them through heatmaps, wordclouds, and dimensionality reduction using [UMAP](https://umap-learn.readthedocs.io/en/latest/).

All of the results and analyses we made can be found on our report, as well as the corresponding visualizations.



## Documents

The `documents` folder contains the report for the assignment, where we discuss our methodology and results more in detail than in the previous section. The main difference between this file and the one we handed in is that it has our names instead of our student numbers and that we increased the size of our plots so that they would be more readable.


## Code

On the `Code` folder we have the following files:

- `asgn2` is the main file. It first runs the preliminary task, then imports the list of 60 test words and calculates and stores the PPMI and the similarity values for each metric. NOTE: The cosine similarity part takes a lot of time to run, so the file can take from one to two
hours to run.
- `clustering.py` is a file that generates the UMAP clusters and the heatmaps.
- `load_map.py` is a supporting file that was given to us for the assignment. It loads two dictionaries that map words to their respective IDs.
- `wordcloud.py` is a file that generates wordclouds for each of the similarity metrics given as word.

[comment]: # (To run it, you need [DATA]. With this, you can run `asgn2.py` to get the similarity scores.)

[comment]: # (Finally, run `wordcloud.py` and `clustering.py` to generate the tables and plots from the report.)