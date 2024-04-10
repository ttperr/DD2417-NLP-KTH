import os
import pickle
from parse_dataset import Dataset
from dep_parser import Parser
from logreg import LogisticRegression
import numpy as np


class TreeConstructor:
    """
    This class builds dependency trees and evaluates using unlabeled arc score (UAS) and sentence-level accuracy
    """

    def __init__(self, parser):
        self.__parser = parser

    def build(self, model, words, tags, ds):
        """
        Build the dependency tree using the logistic regression model `model` for the sentence containing
        `words` pos-tagged by `tags`

        :param      model:  The logistic regression model
        :param      words:  The words of the sentence
        :param      tags:   The POS-tags for the words of the sentence
        :param      ds:     Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE
        #
        i, stack, pred_tree = 0, [], [0] * len(words)
        move_log = []

        while i < len(pred_tree) or len(stack) > 1:
            datapoint = ds.dp2array(words, tags, i, stack)
            prediction = model.get_log_probs(datapoint)
            moves = np.argsort(prediction)[::-1]

            valid_moves = self.__parser.valid_moves(i, stack, pred_tree)
            selected_move = next(move for move in moves if move in valid_moves)
            i, stack, pred_tree = self.__parser.move(
                i, stack, pred_tree, selected_move)
            move_log.append(selected_move)

        return move_log

    def evaluate(self, model, test_file, ds):
        """
        Evaluate the model on the test file `test_file` using the feature representation given by the dataset `ds`

        :param      model:      The model to be evaluated
        :param      test_file:  The CONLL-U test file
        :param      ds:         Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE
        #
        correct_sentences = 0
        total_sentences = 0
        uas_total = 0
        uas_correct = 0
        with open(test_file, 'r') as source:
            for words, tags, tree, relations in self.__parser.trees(source):
                # Call the build function
                correct_moves = self.__parser.compute_correct_moves(tree)
                moves = self.build(model, words, tags, ds)

                total_sentences += 1
                is_correct = True
                for i in range(len(correct_moves)):
                    if correct_moves[i] != self.__parser.SH:
                        uas_total += 1

                        if correct_moves[i] == moves[i]:
                            uas_correct += 1
                    if correct_moves[i] != moves[i]:
                        is_correct = False
                correct_sentences += is_correct
        print(
            f'Sentence-level accuracy: {correct_sentences / total_sentences * 100:.2f}%')
        print(f'UAS accuracy: {uas_correct / uas_total * 100:.2f}%')


if __name__ == '__main__':

    # Create parser
    p = Parser()

    # Create training dataset
    ds = p.create_dataset("en-ud-train-projective.conllu", train=True)

    # Train LR model
    if os.path.exists('model.pkl'):
        # if model exists, load from file
        print("Loading existing model...")
        lr = pickle.load(open('model.pkl', 'rb'))
        ds.to_arrays()
    else:
        # train model using minibatch GD
        lr = LogisticRegression()
        lr.fit(*ds.to_arrays())
        pickle.dump(lr, open('model.pkl', 'wb'))

    # Create test dataset
    test_ds = p.create_dataset("en-ud-dev-projective.conllu")
    # Copy feature maps to ensure that test datapoints are encoded in the same way
    test_ds.copy_feature_maps(ds)
    # Compute move-level accuracy
    lr.classify_datapoints(*test_ds.to_arrays())

    # Compute UAS and sentence-level accuracy
    t = TreeConstructor(p)
    t.evaluate(lr, 'en-ud-dev-projective.conllu', ds)
