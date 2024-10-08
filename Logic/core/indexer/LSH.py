import json
import numpy as np
import itertools
import random


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def get_hash(slef, hash, string):
        for char in string:
            char = ord(char)
            hash = char + (hash << 6 ) + (hash << 16) - hash
        return hash % pow(2, 32)
    
    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        tokens = document.split()
        for i in range(len(tokens) - k + 1):
            shingles.add(' '.join(tokens[i: i + k]))
        return shingles
        # shingles = set()
        # for i in range(len(document) - k + 1):
        #     shingle = document[i:i+k]
        #     shingles.add(shingle)
        # return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        #TODO: some itteraations are redundant

        all_shingles = set()
        for document in self.documents:
            shingles = self.shingle_document(document)
            all_shingles.update(shingles)

        characteristic_matrix = np.zeros((len(self.documents), len(all_shingles)), dtype=bool)
        shingle_to_index = {shingle: i for i, shingle in enumerate(all_shingles)}

        for doc_index, document in enumerate(self.documents):
            shingles = self.shingle_document(document)
            for shingle in shingles:
                shingle_index = shingle_to_index[shingle]
                characteristic_matrix[doc_index, shingle_index] = True
        return characteristic_matrix
    
    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """

        signatures = np.full((self.num_hashes, len(self.documents)), np.inf)
        seeds = []
        for i in range(self.num_hashes):
            seeds.append(random.randint(0, pow(2,32)))
        for i, seed in enumerate(seeds):
            for j, doc in enumerate(self.documents):
                for shingle in self.shingle_document(doc):
                    hash_value =  self.get_hash(seed, shingle)
                    if  signatures[i][j] > hash_value:
                        signatures[i][j] = hash_value
        return signatures

    def lsh_buckets(self, signature, bands=25, rows_per_band=4):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        buckets = {}
        rows_per_band= signature.shape[0] // bands

        for band in range(bands):
            band_signatures = signature[band * rows_per_band : (band + 1) * rows_per_band]
            hashes = [hash(tuple(row)) for row in band_signatures.T]

            for i, h in enumerate(hashes):
                if h not in buckets:
                    buckets[h] = set()
                buckets[h].add(i)

        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        signature = self.min_hash_signature()
        buckets = self.lsh_buckets(signature)
        return buckets

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersection / union if union != 0 else 0
    
    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)
        print(correct_near_duplicates, all_near_duplicates)


def read_from_file_as_json(path):
    with open(path, 'r') as f:
        movies = json.load(f)        
    return movies

def main():
    fake_movies = read_from_file_as_json('logic/core/LSHFakeData.json')
    crawled_movies = read_from_file_as_json('./IMDB_crawled.json')
    all_movies = fake_movies + crawled_movies
    docs = []
    for movie in all_movies:
        if movie['summaries'] != ['']:
            docs.append(" ".join(movie['summaries']))
    m = MinHashLSH(docs, 100)
    buckets = m.perform_lsh()
    m.jaccard_similarity_test(buckets, docs)
if __name__ == '__main__':
    main()