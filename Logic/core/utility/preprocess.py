import re
from typing import List

class Preprocessor:

    def __init__(self, documents, stopwords_path = 'Logic/core/utility/stopwords.txt'):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents
        self.stopwords = self.load_stopwords(stopwords_path)

    def load_stopwords(self, stopwords_path: str):
        """
        Load stopwords from a file.

        Parameters
        ----------
        stopwords_path : str
            The path to the stopwords file.

        Returns
        ----------
        set
            A set containing stopwords.
        """
        with open(stopwords_path, 'r') as file:
            stopwords = set(file.read().splitlines())
        return stopwords
    
    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        preprocessed_documents = []
        for doc in self.documents:
            if isinstance(doc, str):
                doc = self.preprocess_doc(doc)
            elif isinstance(doc, dict):
                for field in doc:
                    if isinstance(doc[field], str):
                        doc[field] = self.preprocess_doc(doc[field])
                    elif isinstance(doc[field], list):
                        doc[field] = self.tmp(doc[field], field)
            preprocessed_documents.append(doc)
        return preprocessed_documents
    
    def tmp(self, val, field):
        new_list = []
        if field == 'reviews':
            for a in val:
                new_list.append(self.preprocess_list(a))
        else:
            new_list = self.preprocess_list(val)
        return new_list

    def preprocess_list(self, val): 
        new_list= []
        for a in val:
            new_list.append(self.preprocess_doc(a))
        return new_list

        # if val== ['']:
        #     return val
        # if isinstance(val,  List[List[str]].__origin__):
        #     new_list = []
        #     for a in val:
        #         new_list.append(self.preprocess_list(a))
        #     return new_list
        # if isinstance(val, List[str].__origin__):
        #     new_list= []
        #     for a in val:
        #         new_list.append(self.preprocess_doc(a))
        #     return new_list

    
    def preprocess_doc(self, doc):
            doc = self.remove_links(doc)
            doc = self.remove_punctuations(doc)
            doc = self.normalize(doc)
            doc = self.remove_stopwords(doc)
            return doc

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        return text.lower()

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        return re.sub(r'[^\w\s]', '', text)


    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        #return word_tokenize(text)
        return text.split()

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
