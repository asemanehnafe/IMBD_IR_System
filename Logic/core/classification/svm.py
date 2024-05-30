import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
from basic_classifier import BasicClassifier
from data_loader import ReviewLoader
import sys
sys.path.append('d:/dars/MIR project 2024/IMBD_IR_System')
from Logic.core.word_embedding.fasttext_model import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

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
        return self.model.predict(x)

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


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    #Example usage
    df = pd.read_csv('./IMDB Dataset.csv')
    labels = df['sentiment'].values
    df['review'] = df['review'].apply(preprocess_text)
    reviews = df['review'].values
    count_vectorizer = CountVectorizer(max_features=300)
    X = count_vectorizer.fit_transform(reviews).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model = SVMClassifier()
    model.fit(X_train, y_train)
    print(model.prediction_report(X_test, y_test))
