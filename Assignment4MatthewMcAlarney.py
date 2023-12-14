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

# When training the classifier and computing the conditional probabilities of each unique word with laplace smoothing, I consulted the following resources to gain further insight into how conditional probabilities are calculated in Multinomial Naive Bayes, and figured out that the conditional probability equations I use in my code are equivalent to and essentially synthesize the conditional probability equation shown on the slides:
# https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece
# https://www.analyticsvidhya.com/blog/2021/04/improve-naive-bayes-text-classifier-using-laplace-smoothing/
# https://devopedia.org/naive-bayes-classifier

# Parts a and b combined:

# Key assumptions:
# 1. In order to use the Naive Bayes Classifier implementation, we must assume that the occurrences of different words are mutually independent events.
# 2. There are two possible classes that emails can be classified as; Ham and Spam classes. Thus, the Naive Bayes Classifier below is designed around these two classes.

# Naive Bayes Classifier implementation with Laplacian smoothing:

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split


class NaiveBayesClassifier:

    def __init__(self, laplace_smoothing_parameter):
        self.ham_conditional_probabilities = np.empty(0)
        self.spam_conditional_probabilities = np.empty(0)
        self.ham_probability_prior = None
        self.spam_probability_prior = None
        self.laplace_smoothing_parameter = laplace_smoothing_parameter

    @staticmethod
    def get_vocabulary(feature_data):
        # Create the vocabulary (list of unique words) for the current training dataset:
        training_data_vocab = np.empty(0)
        for (word, counts_for_every_email) in feature_data.items():
            training_data_vocab = np.append(training_data_vocab, word)
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
        # Summing the occurrences of each unique word in the current class is equal to calculating the total number of words in the current class.
        updated_spam_emails = spam_emails.drop("CLASS", axis='columns')
        number_of_words_per_spam_email = updated_spam_emails.sum(axis='columns')
        total_number_of_words_in_all_spam_emails = number_of_words_per_spam_email.sum()
        total_number_of_words_in_all_spam_emails_to_int = total_number_of_words_in_all_spam_emails.item()
        return total_number_of_words_in_all_spam_emails_to_int

    @staticmethod
    def get_total_number_of_words_in_all_ham_emails(ham_emails):
        # The following code produces the equivalent affect to summing the occurrences/tokens of each unique word in the Ham class as shown in the algorithm on the slides.
        # Summing the occurrences of each unique word in the current class is equal to calculating the total number of words in the current class.
        updated_ham_emails = ham_emails.drop("CLASS", axis='columns')
        number_of_words_per_ham_email = updated_ham_emails.sum(axis='columns')
        total_number_of_words_in_all_ham_emails = number_of_words_per_ham_email.sum()
        total_number_of_words_in_all_ham_emails_to_int = total_number_of_words_in_all_ham_emails.item()
        return total_number_of_words_in_all_ham_emails_to_int

    def train_classifier(self, vocabulary, ham_emails, spam_emails, total_number_of_words_in_all_ham_emails, total_number_of_words_in_all_spam_emails):
        # Store a list of dictionaries for the conditional probabilities (probabilities of each word occurring given the Ham and Spam categories):
        for unique_word in vocabulary:
            # Sum the occurrences of the unique word in the ham category (in the current column).
            occurrences_of_unique_word_in_ham = ham_emails[unique_word].sum()
            # In this equation, note that laplace smoothing is implemented in the numerator and denominator.
            # In addition, adding (the length of the vocabulary (number of dimensions in the feature data) * laplace smoothing parameter) to the total number of words in the current class is equivalent to the conditional probability equation shown on the slides;
            # In the denominator, 1 is added to the token count for the current unique word, which ultimately means that 1 is summed the same number of times as the number of unique words (the number of unique words is the length of the vocabulary).
            # Furthermore, if the laplace smoothing parameter was increased to 2, adding (2 * length of vocabulary) is equivalent to summing the token counts for each unique word and adding 2 to each token count.
            # This is because when 2 is added to each token count, 2 is added per unique word; this is equal to multiplying the length of the set of unique words (the length of the vocabulary) by 2.
            conditional_probability_of_unique_word_given_ham = (occurrences_of_unique_word_in_ham + self.laplace_smoothing_parameter) / (total_number_of_words_in_all_ham_emails + (self.laplace_smoothing_parameter * len(vocabulary)))
            self.ham_conditional_probabilities = np.append(self.ham_conditional_probabilities, {unique_word: conditional_probability_of_unique_word_given_ham})
            # Sum the occurrences of the unique word in the spam category (in the current column).
            occurrences_of_unique_word_in_spam = spam_emails[unique_word].sum()
            # In this equation, note that laplace smoothing is implemented in the numerator and denominator.
            # In addition, adding (the length of the vocabulary (number of dimensions in the feature data) * laplace smoothing parameter) to the total number of words in the current class is equivalent to the conditional probability equation shown on the slides;
            # In the denominator, 1 is added to the token count for the current unique word, which ultimately means that 1 is summed the same number of times as the number of unique words (the number of unique words is the length of the vocabulary).
            # Furthermore, if the laplace smoothing parameter was increased to 2, adding (2 * length of vocabulary) is equivalent to summing the token counts for each unique word and adding 2 to each token count.
            # This is because when 2 is added to each token count, 2 is added per unique word; this is equal to multiplying the length of the set of unique words (the length of the vocabulary) by 2.
            conditional_probability_of_unique_word_given_spam = (occurrences_of_unique_word_in_spam + self.laplace_smoothing_parameter) / (total_number_of_words_in_all_spam_emails + (self.laplace_smoothing_parameter * len(vocabulary)))
            self.spam_conditional_probabilities = np.append(self.spam_conditional_probabilities, {unique_word: conditional_probability_of_unique_word_given_spam})

    def test_classifier(self, feature_data):
        # The goal of test_classifier is to classify each email in the feature data as either Ham or Spam.
        # We need to iterate through each word in the current email in order to provide a classification for each email.
        predicted_classifications_per_email = [0] * len(feature_data.index)
        probability_of_ham_per_email = np.empty(0)
        probability_of_spam_per_email = np.empty(0)
        email_indices_list = []
        list_index_placeholder = 0

        for email_index, email in feature_data.iterrows():
            email_indices_list.append(email_index)
            probability_of_ham_per_email = np.append(probability_of_ham_per_email, {email_index: math.log(self.ham_probability_prior)})
            probability_of_spam_per_email = np.append(probability_of_spam_per_email, {email_index: math.log(self.spam_probability_prior)})
            unique_word_index = 0
            for unique_word in feature_data.columns:
                probability_of_ham_per_email[list_index_placeholder].update({email_index: (probability_of_ham_per_email[list_index_placeholder].get(email_index) + math.log(self.ham_conditional_probabilities[unique_word_index].get(unique_word)))})
                probability_of_spam_per_email[list_index_placeholder].update({email_index: (probability_of_spam_per_email[list_index_placeholder].get(email_index) + math.log(self.spam_conditional_probabilities[unique_word_index].get(unique_word)))})
                unique_word_index += 1
            list_index_placeholder += 1

        for predicted_classification_index in range(len(predicted_classifications_per_email)):
            if probability_of_ham_per_email[predicted_classification_index].get(email_indices_list[predicted_classification_index]) > probability_of_spam_per_email[predicted_classification_index].get(email_indices_list[predicted_classification_index]):
                predicted_classifications_per_email[predicted_classification_index] = 1

        return predicted_classifications_per_email

    @staticmethod
    def generate_confusion_matrix(predicted_classifications_per_email, target_data):
        target_data_nd_array = target_data.to_numpy()
        target_data_nd_array = target_data_nd_array.flatten()
        confusion_matrix = np.zeros((2, 2))
        for prediction_index in range(len(predicted_classifications_per_email)):
            # 1 corresponds to HAM and 0 corresponds to SPAM. In this case, we are counting a "positive" email as HAM and a "negative" email as SPAM.
            # (0, 0) = TP, (0, 1) = FP, (1, 0) = FN, (1, 1) = TN
            if predicted_classifications_per_email[prediction_index] == 1 and int(
                target_data_nd_array[prediction_index]) == 1:
                # True positive
                confusion_matrix[0, 0] += 1
            elif predicted_classifications_per_email[prediction_index] == 1 and int(
                target_data_nd_array[prediction_index]) == 0:
                # False positive
                confusion_matrix[0, 1] += 1
            elif predicted_classifications_per_email[prediction_index] == 0 and int(
                target_data_nd_array[prediction_index]) == 0:
                # True negative
                confusion_matrix[1, 1] += 1
            elif predicted_classifications_per_email[prediction_index] == 0 and int(
                target_data_nd_array[prediction_index]) == 1:
                # False negative
                confusion_matrix[1, 0] += 1

        return confusion_matrix

    @staticmethod
    def report_f_measure(confusion_matrix, testing_dataset_name):
        precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
        recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
        f_measure = (2 * precision * recall) / (precision + recall)
        print("The f-measure for " + testing_dataset_name + " is " + str(f_measure))

    def run_algorithm(self, run_on_training, feature_data, target_data, testing_dataset_name):
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
            predicted_classifications_per_email = self.test_classifier(feature_data)
            confusion_matrix = self.generate_confusion_matrix(predicted_classifications_per_email, target_data)
            self.report_f_measure(confusion_matrix, testing_dataset_name)


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

# Remove the last label (CLASS) in each list:

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

naive_bayes_classifier_for_bodies.run_algorithm(True, bodies_training_data_x, bodies_training_data_y, None)
naive_bayes_classifier_for_bodies.run_algorithm(False, bodies_testing_data_x, bodies_testing_data_y, "Email bodies stemmed testing data")

naive_bayes_classifier_for_subjects.run_algorithm(True, subjects_training_data_x, subjects_training_data_y, None)
naive_bayes_classifier_for_subjects.run_algorithm(False, subjects_testing_data_x, subjects_testing_data_y, "Email subjects stemmed testing data")

# Below we use the scikit learn Naive Bayes classifier on the bodies and subjects datasets and report a comparison of results between the implementation from scratch and the scikit-learn implementation:
