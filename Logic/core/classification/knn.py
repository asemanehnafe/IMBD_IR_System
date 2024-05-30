import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader
import pandas as pd
import sys
sys.path.append('d:/dars/MIR project 2024/IMBD_IR_System')
from Logic.core.word_embedding.fasttext_model import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.x_train = x
        self.y_train = y

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    
    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        predictions = []
        for sample in tqdm(x):
            distances = [self._euclidean_distance(sample, train_sample) for train_sample in self.x_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            majority_vote = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(majority_vote)
        return np.array(predictions)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        predictions = self.predict(x)
        return classification_report(y, predictions)


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    data_loader = ReviewLoader(file_path = './IMDB Dataset.csv')
    data_loader.load_data()
    data_loader.get_embeddings()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    model = KnnClassifier(5)
    model.fit(X_train, y_train)
    print(model.prediction_report(X_test, y_test))
