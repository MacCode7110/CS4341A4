# dbworld_bodies_stemmed and dbworld_subjects_stemmed.
# corresponding to the email body and email subject respectively.
# The data is currently represented as a binary stemmed bag-of-words and requires no
# additional NLP.

# Notes from Canvas assignment:
# Each dataset is in a table form with 64 rows and n columns.
# The 1st column is “id” and has values from 1 to 64, corresponding to each of the
# 64 emails (this column can be removed).
# The 2 to n-1 columns are unique words found in all the emails, they have binary
# values i.e. 0 means that the word did not appear in the email and 1 means that the
# word appeared.
# The nth column is CLASS, 0 means discard email and 1 means keep email.

# Parts a and b combined:

import pandas as pd


# Naive Bayes Classifier implementation with Laplacian smoothing:

class NaiveBayesClassifier:

    def __init__(self):
        pass


# Load and split both datasets into training and testing:

dbworld_bodies_stemmed = pd.read_csv('dbworld_bodies_stemmed', sep='\t',
                       header=None)
dbworld_subjects_stemmed = pd.read_csv('dbworld_subjects_stemmed', sep='\t')
