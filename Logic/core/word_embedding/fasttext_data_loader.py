import pandas as pd
from tqdm import tqdm
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string

sys.path.append('d:/dars/MIR project 2024/IMBD_IR_System')

from sklearn.preprocessing import LabelEncoder
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader

class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path
        pass

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        data = Index_reader(self.file_path, Indexes.DOCUMENTS).index
        titles = []
        genres = []
        synopses = []
        summaries = []
        reviews = []

        for _, details in tqdm(data.items()):
            titles.append(details.get('title', ''))
            genres.append(details.get('genres', ''))
            synopses.append(details.get('synopsis', ''))
            summaries.append(details.get('summaries', ''))
            reviews.append(details.get('reviews', ''))


        df = pd.DataFrame({
            'synopsis': synopses,
            'summary': summaries,
            'reviews': reviews,
            'title': titles,
            'genre': genres
        })
        return df

    def list_review_preproces(self, list):
        reviews = ''
        if list is not None:
            for review in list:
                text = review[0]
                text = text.lower()
                text = re.sub('['+string.punctuation+']', '', text)
                stop_words = set(stopwords.words('english'))
                words = word_tokenize(text)
                filtered_words = [word for word in words if word not in stop_words]
                text = ' '.join(filtered_words)
                reviews += text
        return reviews
    
    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        df['preprocessed_reviews'] = df['reviews'].apply(self.list_review_preproces)
        

        df['first_genre'] = df['genre'].str[0].fillna('nothing')
        
        label_encoder = LabelEncoder()
        df['encoded_genre'] = label_encoder.fit_transform(df['first_genre'].astype(str))

        X = df['preprocessed_reviews'].values.astype(str)[:1000]
        y = df['encoded_genre'].values[:1000]

        return X, y
