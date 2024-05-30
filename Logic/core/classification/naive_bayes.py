import numpy as np
import pandas as pd
import sys
sys.path.append('d:/dars/MIR project 2024/IMBD_IR_System')

from Logic.core.word_embedding.fasttext_model import preprocess_text

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.number_of_samples, self.number_of_features = x.shape

        self.prior = np.zeros(self.num_classes)
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))

        for idx, label in enumerate(self.classes):
            x_class = x[y == label]
            self.prior[idx] = x_class.shape[0] / self.number_of_samples
            self.feature_probabilities[idx, :] = (np.sum(x_class, axis=0) + self.alpha) / (x_class.shape[0] + self.alpha * self.number_of_features)

        self.log_probs = np.log(self.feature_probabilities)
        return self

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
        log_priors = np.log(self.prior)
        predictions = np.dot(x, self.log_probs.T) + log_priors
        return self.classes[np.argmax(predictions, axis=1)]

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

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        x = self.cv.transform(sentences)
        predictions = self.predict(x.toarray())
        positive_reviews = np.sum(predictions == 'positive')
        return (positive_reviews / len(sentences)) * 100


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    # Example usage
    df = pd.read_csv('./IMDB Dataset.csv')
    
    labels = df['sentiment'].values

    reviews = [preprocess_text(text) for text in df['review']]
    count_vectorizer = CountVectorizer(max_features=5000)
    X = count_vectorizer.fit_transform(reviews).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    nb_classifier = NaiveBayes(count_vectorizer)
    nb_classifier.fit(X_train, y_train)

    print(nb_classifier.prediction_report(X_test, y_test))