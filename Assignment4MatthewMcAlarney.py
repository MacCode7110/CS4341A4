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

# Key assumptions:
# 1. In order to use the Naive Bayes Classifier implementation, we must assume that the occurrences of different words are mutually independent events.
# 2. There are two possible classes that emails can be classified as; Ham and Spam classes. Thus, the Naive Bayes Classifier below is designed around these two classes.

# Naive Bayes Classifier implementation with Laplacian smoothing:

import pandas as pd
from sklearn.model_selection import train_test_split


class NaiveBayesClassifier:

    def __init__(self, laplace_smoothing_parameter):
        self.ham_conditional_probabilities = []
        self.spam_conditional_probabilities = []
        self.ham_probability_prior = None
        self.spam_probability_prior = None
        self.laplace_smoothing_parameter = laplace_smoothing_parameter

    @staticmethod
    def get_vocabulary(feature_data):
        # Create the vocabulary (list of unique words) for the current training dataset:
        training_data_vocab = []
        for (word, counts_for_every_email) in feature_data.items():
            training_data_vocab.append(word)
        return training_data_vocab

    @staticmethod
    def get_spam_emails(combined_dataframe):
        spam_emails = combined_dataframe[combined_dataframe['CLASS'] == 0]
        return spam_emails

    @staticmethod
    def get_ham_emails(combined_dataframe):
        ham_emails = combined_dataframe[combined_dataframe['CLASS'] == 1]
        return ham_emails

    def calculate_ham_probability_prior(self, ham_emails, combined_dataframe):
        self.ham_probability_prior = len(ham_emails) / len(combined_dataframe)

    def calculate_spam_probability_prior(self, spam_emails, combined_dataframe):
        self.spam_probability_prior = len(spam_emails) / len(combined_dataframe)

    @staticmethod
    def get_total_number_of_words_in_all_spam_emails(spam_emails):
        # The following code produces the equivalent affect to summing the occurrences/tokens of each unique word in the Spam class as shown in the algorithm on the slides.
        updated_spam_emails = spam_emails.drop("CLASS", axis='columns')
        number_of_words_per_spam_email = updated_spam_emails.sum(axis='columns')
        total_number_of_words_in_all_spam_emails = number_of_words_per_spam_email.sum()
        total_number_of_words_in_all_spam_emails_to_int = total_number_of_words_in_all_spam_emails.item()
        return total_number_of_words_in_all_spam_emails_to_int

    @staticmethod
    def get_total_number_of_words_in_all_ham_emails(ham_emails):
        # The following code produces the equivalent affect to summing the occurrences/tokens of each unique word in the Ham class as shown in the algorithm on the slides.
        updated_ham_emails = ham_emails.drop("CLASS", axis='columns')
        number_of_words_per_ham_email = updated_ham_emails.sum(axis='columns')
        total_number_of_words_in_all_ham_emails = number_of_words_per_ham_email.sum()
        total_number_of_words_in_all_ham_emails_to_int = total_number_of_words_in_all_ham_emails.item()
        return total_number_of_words_in_all_ham_emails_to_int

    def train_classifier(self, vocabulary, ham_emails, spam_emails, total_number_of_words_in_all_ham_emails, total_number_of_words_in_all_spam_emails):
        # Store a list of dictionaries for the conditional probabilities (probabilities of each word occurring given the Ham and Spam categories):
        for unique_word in vocabulary:
            # Sum the occurrences of the unique word in each ham email (in the current column).
            occurrences_of_unique_word_in_ham = ham_emails[unique_word].sum()
            # In this equation, note that laplace smoothing is implemented in the numerator.
            # In addition, adding the length of the vocabulary to the total number of words in the current class is equivalent to the conditional probability equation shown on the slides; in the denominator, 1 is added to the token count for the current unique word, which ultimately means that 1 is summed the same number of times as the number of unique words (the number of unique words is the length of the vocabulary).
            conditional_probability_of_unique_word_given_ham = (occurrences_of_unique_word_in_ham + self.laplace_smoothing_parameter) / (total_number_of_words_in_all_ham_emails + len(vocabulary))
            self.ham_conditional_probabilities.append({unique_word: conditional_probability_of_unique_word_given_ham})
            # Sum the occurrences of the unique word in each spam email (in the current column).
            occurrences_of_unique_word_in_spam = spam_emails[unique_word].sum()
            # In this equation, note that laplace smoothing is implemented in the numerator.
            # In addition, adding the length of the vocabulary to the total number of words in the current class is equivalent to the conditional probability equation shown on the slides; in the denominator, 1 is added to the token count for the current unique word, which ultimately means that 1 is summed the same number of times as the number of unique words (the number of unique words is the length of the vocabulary).
            conditional_probability_of_unique_word_given_spam = (occurrences_of_unique_word_in_spam + self.laplace_smoothing_parameter) / (total_number_of_words_in_all_spam_emails + len(vocabulary))
            self.spam_conditional_probabilities.append({unique_word: conditional_probability_of_unique_word_given_spam})

    def test_classifier(self, feature_data):
        # The goal of test_classifier is to classify each email in the feature data as either Ham or Spam.
        # We need to iterate through each word in the current email in order to provide a classification.
        predicted_classifications_per_email = []

    def generate_confusion_matrix(self):
        pass

    def report_f_measure(self):
        pass

    def run_algorithm(self, run_on_training, feature_data, target_data):
        combined_dataframe = feature_data.join(target_data)
        if run_on_training:
            vocabulary = self.get_vocabulary(feature_data)
            ham_emails = self.get_ham_emails(combined_dataframe)
            spam_emails = self.get_spam_emails(combined_dataframe)
            self.calculate_ham_probability_prior(ham_emails, combined_dataframe)
            self.calculate_spam_probability_prior(spam_emails, combined_dataframe)
            number_of_words_in_all_ham_emails = self.get_total_number_of_words_in_all_ham_emails(ham_emails)
            number_of_words_in_all_spam_emails = self.get_total_number_of_words_in_all_spam_emails(spam_emails)
            self.train_classifier(vocabulary, ham_emails, spam_emails, number_of_words_in_all_ham_emails, number_of_words_in_all_spam_emails)
        else:
            self.test_classifier(feature_data)
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
# naive_bayes_classifier_for_bodies.run_algorithm(False, bodies_testing_data_x, bodies_testing_data_y)

# naive_bayes_classifier_for_subjects.run_algorithm(True, subjects_training_data_x, subjects_training_data_y)
# naive_bayes_classifier_for_subjects.run_algorithm(False, subjects_testing_data_x, subjects_testing_data_y)

# Below we use the scikit learn Naive Bayes classifier on the bodies and subjects datasets and report a comparison of results between the implementation from scratch and the scikit-learn implementation:
