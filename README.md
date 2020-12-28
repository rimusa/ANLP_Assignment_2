# Word Similarity

This project was made by [Antonia Donvito](https://github.com/Antos23) and [Ricardo Muñoz Sánchez](https://github.com/rimusa). It is based on the second assignment for the Accelerated Natural Language Processing (ANLP)  course imparted at the University of Edinburgh by Sharon Goldwater and Shay Cohen during the Autumn 2019 semester.


## Assignment

[Introduce the assignment]
Preliminary task - compute similarity between among a list of words.

[Explain the word representations]

[Explain the similarity metrics used]

[Explain our exploration of the clusters]


## Documents

The documents folder contains the report for the assignment, where we discuss our methodology and results more in detail than in the previous section. The main difference between this file and the one we handed in is that it has our names instead of our student numbers and that we increased the size of our plots so that they would be more readable.


## Code

On the ``Code`` folder we have the following files:

- `asgn2` is the main file. It first runs the preliminary task, then imports the list of 60 test words and calculates and stores the PPMI and the similarity values for each metric. NOTE: The cosine similarity part takes a lot of time to run, so the file can take from one to two
hours to run.
- `clustering.py` is a file that generates the UMAP clusters and the heatmaps.
- `load_map.py` is a supporting file that was given to us for the assignment. It loads two dictionaries that map words to their respective IDs.
- `wordcloud.py` is a file that generates wordclouds for each of the similarity metrics given as word.

[comment]: # (To run it, you need [DATA]. With this, you can run `asgn2.py` to get the similarity scores.)

[comment]: # (Finally, run `wordcloud.py` and `clustering.py` to generate the tables and plots from the report.)