import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('d:/dars/MIR project 2024/IMBD_IR_System')
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.word_embedding.fasttext_model import preprocess_text


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = None
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        data = pd.read_csv(self.file_path)

        reviews = data['review'].values
        sentiments = data['sentiment'].values
        label_encoder = LabelEncoder()
        sentiments = label_encoder.fit_transform(sentiments)

        self.review_tokens = [preprocess_text(review) for review in reviews]
        self.sentiments = sentiments

        self.fasttext_model = FastText()
        self.fasttext_model.load_model(path='./FastText_model.bin')
        self.get_embeddings()

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        if not self.review_tokens:
            self.load_data()
        self.embeddings = [self.fasttext_model.get_query_embedding(tokens) for tokens in self.review_tokens]


    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        # Ensure embeddings are generated
        if not self.embeddings:
            self.get_embeddings()

        # Convert lists to numpy arrays
        embeddings = np.array(self.embeddings)
        sentiments = np.array(self.sentiments)

        x_train, x_test, y_train, y_test = train_test_split(embeddings, sentiments, test_size=test_data_ratio, random_state=42)

        return x_train, x_test, y_train, y_test
