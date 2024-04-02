#  -*- coding: utf-8 -*-
import math
import argparse
import nltk
import codecs
from collections import defaultdict

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter. 
        """
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.log_prob = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0

    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: True if the entire file could be processed, False otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(
                    int, f.readline().strip().split(' '))
                # YOUR CODE HERE
                # Read the unigram counts
                for _ in range(self.unique_words):
                    index, word, count = f.readline().strip().split(' ')
                    self.index[word] = int(index)
                    self.word[int(index)] = word
                    self.unigram_count[int(index)] = int(count)

                # Read the bigram probabilities
                while True:
                    line = f.readline().strip()
                    if line == '-1' or line == '':
                        break
                    index1, index2, log_prob = line.split(' ')
                    self.bigram_prob[int(index1)][int(
                        index2)] = float(log_prob)

                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def compute_entropy_cumulatively(self, word):
        # YOUR CODE HERE

        try:
            # Get the index of the word
            index = self.index[word]
            unigram_prob = self.unigram_count[index] / self.total_words
        except KeyError:
            # If the word is unknown, the index is -1
            index = -1
            unigram_prob = 0.0

        # If the bigram probability is not in the dictionary, it is 0.
        bigram_prob = 0.0
        if index in self.bigram_prob[self.last_index]:
            bigram_prob = math.exp(self.bigram_prob[self.last_index][index])

        # Update the average log-probability
        self.log_prob -= math.log(self.lambda1 * bigram_prob +
                                  self.lambda2 * unigram_prob + self.lambda3) / len(self.tokens)

        self.test_words_processed += 1
        self.last_index = index

    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower())
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
            return True
        except IOError:
            print('Error reading testfile')
            return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,
                        required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str,
                        required=True, help='test corpus')

    arguments = parser.parse_args()

    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)
    print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(
        bigram_tester.test_words_processed, bigram_tester.log_prob))


if __name__ == "__main__":
    main()
