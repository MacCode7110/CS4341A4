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
from sklearn.model_selection import train_test_split


# Key assumption: In order to use Bayes Rule in the Naive Bayes Classifier implementation, we must assume that the occurrences of different words are mutually independent events.

# Naive Bayes Classifier implementation with Laplacian smoothing:

class NaiveBayesClassifier:

    def __init__(self, laplace_smoothing_parameter):
        self.laplace_smoothing_parameter = laplace_smoothing_parameter

    @staticmethod
    def get_vocabulary(feature_data):
        # Create the vocabulary (list of unique words) for the current training dataset:
        training_data_vocab = []

        for (word, counts_per_email) in feature_data.items():
            training_data_vocab.append(word)

        return training_data_vocab

    @staticmethod
    def get_class_information(feature_data, target_data):
        spam_emails = target_data[target_data['CLASS'] == 0]
        ham_emails = target_data[target_data['CLASS'] == 1]
        probability_of_spam_email = len(spam_emails) / len(target_data)
        probability_of_ham_email = len(ham_emails) / len(target_data)

        for (word, counts_per_email) in feature_data.items():
            # number_of_words_in_all_spam_emails =
            # number_of_words_in_all_ham_emails =

    def train_classifier(self):
        pass

    def test_classifier(self):
        pass

    def generate_confusion_matrix(self):
        pass

    def report_f_measure(self):
        pass

    def run_algorithm(self, run_on_training, feature_data, target_data):
        if run_on_training:
            vocabulary = self.get_vocabulary(feature_data)
            self.get_class_information(feature_data, target_data)
            self.train_classifier()
        else:
            self.test_classifier()
            self.generate_confusion_matrix()
            self.report_f_measure()


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

# Run the Naive Bayes Classifier on the training and test data:

naive_bayes_classifier_for_bodies = NaiveBayesClassifier(1)
naive_bayes_classifier_for_subjects = NaiveBayesClassifier(1)

naive_bayes_classifier_for_bodies.run_algorithm(True, bodies_training_data_x, bodies_training_data_y)
naive_bayes_classifier_for_bodies.run_algorithm(False, bodies_testing_data_x, bodies_testing_data_y)

naive_bayes_classifier_for_subjects.run_algorithm(True, subjects_training_data_x, subjects_training_data_y)
naive_bayes_classifier_for_subjects.run_algorithm(False, subjects_testing_data_x, subjects_testing_data_y)

# Below we use the scikit learn Naive Bayes classifier on the bodies and subjects datasets and report a comparison of results between the implementation from scratch and the scikit-learn implementation:
