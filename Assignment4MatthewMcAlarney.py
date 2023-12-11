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
# The nth column is CLASS, 0 means discard email (SPAM) and 1 means keep email (HAM).

# Parts a and b combined:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Key assumption: In order to use Bayes Rule in the Naive Bayes Classifier implementation, we must assume that the occurrences of different words are mutually independent events.

# Naive Bayes Classifier implementation with Laplacian smoothing:

class NaiveBayesClassifier:

    def __init__(self, laplace_smoothing_parameter):
        self.laplace_smoothing_parameter = laplace_smoothing_parameter


    def create_vocabulary(self):
        # Create the vocabulary (list of unique words) for each of the training sets:
        bodies_training_data_vocab = []
        subjects_training_data_vocab = []

        for (word, count_per_email) in bodies_training_data_x.items():
            bodies_training_data_vocab.append(word)

        for (word, count_per_email) in subjects_training_data_x.items():
            subjects_training_data_vocab.append(word)

    def train_classifier(self):
        pass

    def test_classifier(self):
        pass

    def generate_confusion_matrix(self):
        pass

    def report_f_measure(self):
        pass

    def run_algorithm(self, run_on_training):
        pass


# Load and split both datasets into training and testing:

dbworld_bodies_stemmed = pd.read_csv('dbworld_bodies_stemmed.csv',
                                     delimiter=',')  # Retrieves the data in the form of a dataframe.
dbworld_subjects_stemmed = pd.read_csv('dbworld_subjects_stemmed.csv',
                                       delimiter=',')  # Retrieves the data in the form of a dataframe.

# Drop the id column from both datasets:
updated_dbworld_bodies_stemmed = dbworld_bodies_stemmed.drop(columns=['id'], axis=1)
updated_dbworld_subjects_stemmed = dbworld_subjects_stemmed.drop(columns=['id'], axis=1)

dbworld_bodies_word_list = updated_dbworld_bodies_stemmed.columns.values.tolist()
dbworld_subjects_word_list = updated_dbworld_subjects_stemmed.columns.values.tolist()

# Remove the last label in each list (CLASS):

dbworld_bodies_word_list.remove(dbworld_bodies_word_list[len(dbworld_bodies_word_list) - 1])
dbworld_subjects_word_list.remove(dbworld_subjects_word_list[len(dbworld_subjects_word_list) - 1])

# All the data is already cleaned, so now we can split both datasets into training and test sets:

feature_dataframe_dbworld_bodies_stemmed = updated_dbworld_bodies_stemmed.loc[:, dbworld_bodies_word_list]
target_dataframe_dbworld_bodies_stemmed = updated_dbworld_bodies_stemmed.loc[:, ['CLASS']]

feature_dataframe_dbworld_subjects_stemmed = updated_dbworld_subjects_stemmed.loc[:, dbworld_subjects_word_list]
target_dataframe_dbworld_subjects_stemmed = updated_dbworld_subjects_stemmed.loc[:, ['CLASS']]

# train_test_split the db_world_bodies data:

bodies_training_data_x, bodies_testing_data_x, bodies_training_data_y, bodies_testing_data_y = train_test_split(
    feature_dataframe_dbworld_bodies_stemmed, target_dataframe_dbworld_bodies_stemmed,
    train_size=0.8, shuffle=True,
    stratify=target_dataframe_dbworld_bodies_stemmed)

# train_test_split the db_world_subjects data:

subjects_training_data_x, subjects_testing_data_x, subjects_training_data_y, subjects_testing_data_y = train_test_split(
    feature_dataframe_dbworld_subjects_stemmed, target_dataframe_dbworld_subjects_stemmed,
    train_size=0.8, shuffle=True,
    stratify=target_dataframe_dbworld_subjects_stemmed)








