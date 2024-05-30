import numpy as np
from tqdm import tqdm
import sys
sys.path.append('d:/dars/MIR project 2024/IMBD_IR_System')
from Logic.core.word_embedding.fasttext_model import FastText


class BasicClassifier:
    def __init__(self):
        pass

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def prediction_report(self, x, y):
        pass

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        return sum(self.predict([sentence])[0] == 'positive' for sentence in sentences) / len(sentences)

