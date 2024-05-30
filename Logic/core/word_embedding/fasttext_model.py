import fasttext
import re
import string
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
import tempfile

from .fasttext_data_loader import FastTextDataLoader


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if lower_case:
        text = text.lower()

    if punctuation_removal:
            text = re.sub('['+string.punctuation+']', '', text)
    if stopword_removal:
        stop_words = set(stopwords.words('english'))
        if stopwords_domain:
            stop_words.update(stopwords_domain)
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in stop_words]

        text = ' '.join(filtered_words)

    if minimum_length > 0:
        words = word_tokenize(text)
        filtered_words = [word for word in words if len(word) >= minimum_length]

        text = ' '.join(filtered_words)

    return text

class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None


    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(temp_file.name, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        self.model = fasttext.train_unsupervised(input=temp_file.name, model=self.method)
        temp_file.close()

    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        # tokens = query.split()
        # embeddings = []
        # for token in tokens:
        #     embeddings.append(self.model.get_word_vector(token))
        # query_embedding = np.mean(embeddings, axis=0)
        # return query_embedding
        return self.model.get_sentence_vector(query)
    
    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        emb_word1 = self.model.get_word_vector(word1)
        emb_word1 = self.model.get_word_vector(word1)
        emb_word2 = self.model.get_word_vector(word2)
        emb_word3 = self.model.get_word_vector(word3)

        emb_result = emb_word2 - emb_word1 + emb_word3

        other_emb = {word: self.model[word] for word in self.model.words if word not in {word1, word2, word3}}

        closest_word, min_distance = min(
            other_emb.items(),
            key=lambda item: distance.cosine(emb_result, item[1]),
            default=(None, float('inf'))
        )

        return closest_word

    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)



    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if mode == 'save':
            self.save_model(path)

if __name__ == "__main__":
    ft_model = FastText(method='skipgram')

    path = './index'
    ft_data_loader = FastTextDataLoader(path)

    X, y = ft_data_loader.create_train_data()
    np.savez('arrays.npz', arr1=X, arr2=y)
    ft_model.train(X)
    ft_model.prepare(None, mode = "save")

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "men"
    word3 = "woman"
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
    ft_model.save_model()