import numpy as np

class Scorer:    
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self,query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))
    
    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            df = len(self.index.get(term, {}))
            if df != 0:
                idf = np.log(self.N / df)
            else:
                idf = 0
            self.idf[term] = idf
        return idf
    
    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        query_tfs = {}
        for term in query:
            query_tfs[term] = query_tfs.get(term, 0) + 1
        return query_tfs

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        scores = {}
        query_tfs = self.get_query_tfs(query)
        for document_id in self.get_list_of_documents(query):
            scores[document_id] = self.get_vector_space_model_score(query, query_tfs, document_id, method[0:3], method[4:7])
        return scores
    
    def cal(self, tf, idf, method):
        for i in range(len(tf)):
            tf[i] += 0.1
        if method == "nnn":
            return tf
        elif method == "nnc":
            return tf / np.linalg.norm(tf)
        elif method == "ntn":
            return tf * idf
        elif method == "ntc":
            w = tf * idf
            return w / np.linalg.norm(w)
        elif method == "lnn":
            return 1 + np.log(tf)
        elif method == "lnc":
            w = 1 + np.log(tf)
            return w / np.linalg.norm(w)
        elif method == "ltn":
            return (1 + np.log(tf)) * idf
        elif method == "ltc":
            w =  (1 + np.log(tf)) * idf 
            return w / np.linalg.norm(w) 

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """ 
        tf = [self.index.get(term, {}).get(document_id, 0) for term in query]
        idf = [self.get_idf(term) for term in query]
        qtf = [query_tfs.get(term, 0) for term in query]
        score = np.dot(self.cal(tf, idf, document_method), self.cal(qtf, idf, query_method))
        return score
    
    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        scores = {}
        for document_id in self.get_list_of_documents(query):
            scores[document_id] = self.get_okapi_bm25_score(query, document_id, average_document_field_length, document_lengths)
        return scores
    
    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """
        score = 0
        k1 = 1.2
        b= 0.75
        for term in query:
            tf = self.index.get(term, {}).get(document_id, 0)
            idf = self.get_idf(term)
            dl = document_lengths[document_id]
            score += idf * ((k1+ 1) * tf )/(k1 * ((1-b) + b * dl/average_document_field_length) + tf)
        return score
    
    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """

        scores = {}
        for document_id in document_lengths:
            score = self.get_unigram_model_score(
                query, document_id, smoothing_method, document_lengths, alpha, lamda
            )
            scores[document_id] = score
        return scores

    def get_unigram_model_score(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """        
        document_score = 0
        for term in query:
            tf = self.index.get(term, {}).get(document_id, 0)
            cf = sum(self.index.get(term, {}).values()) / sum(document_lengths.values())
            term_prob = self.compute_term_probability(tf, smoothing_method, alpha, lamda, document_lengths[document_id], cf)
            document_score += np.log(term_prob)
        return document_score

    def compute_term_probability(self, tf, smoothing_method, alpha, lamda, document_lenght, cf):
            """
            Computes the probability of a term occurring in the document based on the unigram model.

            Parameters
            ----------
            tf : int
                The term frequency in the document.
            idf : float
                The inverse document frequency of the term.
            smoothing_method : str (bayes | naive | mixture)
                The method used for smoothing the probabilities in the unigram model.
            alpha : float
                The parameter used in bayesian smoothing method.
            lamda : float
                The parameter used in some smoothing methods to balance between the document
                probability and the collection probability.

            Returns
            -------
            float
                The probability of the term occurring in the document.
            """
            prob = 0
            if smoothing_method == "bayes":
                if (document_lenght + alpha > 0):
                    prob = (tf + alpha * cf) / (document_lenght + alpha)
            elif smoothing_method == "naive":
                M = len(self.index)
                prob = (tf + 1/ M) / (document_lenght + 1)
            elif smoothing_method == "mixture":
                if document_lenght != 0:
                    prob = (1 - lamda) * cf + lamda * (tf / document_lenght)
            else:
                raise ValueError("Invalid smoothing method")
            
            return prob