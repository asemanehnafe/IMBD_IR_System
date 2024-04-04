class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        
        #Create shingle
        shingles = set()
        if len(word) < k:
            shingles.add(word)
        else:
            for i in range(len(word) - k + 1):
                shingles.add(word[i:i+k])
        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersection / union if union != 0 else 0

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        #Create shingled words dictionary and word counter dictionary here.
        for document in all_documents:
            for word in document.split():
                if word not in word_counter:
                    word_counter[word] = 1
                else:
                    word_counter[word] += 1
                if word not in all_shingled_words:
                    shingles = self.shingle_word(word)
                    all_shingled_words[word] = shingles
                
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        # TODO: Find 5 nearest candidates here.
        query_shingles = self.shingle_word(word)
        scores = []
        for candidate, shingles in self.all_shingled_words.items():
            jaccard = self.jaccard_score(query_shingles, shingles)
            scores.append((candidate, jaccard))
        top5_candidates = [candidate for candidate, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:5]]   
        return top5_candidates
    
    def find_best(self, query, nearest_words):
        query_shingles = self.shingle_word(query)
        max_tf = max(self.word_counter[word] for word in nearest_words)
        max_score = -1
        for word in nearest_words:
            shingles = self.shingle_word(word)
            jaccard = self.jaccard_score(query_shingles, shingles)
            score = self.word_counter[word] / max_tf * jaccard
            if score > max_score:
                max_score = score
                final_result = word
        return final_result
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""
        for word in query.lower().split():
            if word in self.word_counter:
                final_result += word
            else:
                nearest_words = self.find_nearest_words(word)
                if nearest_words:
                    final_result += self.find_best(word, nearest_words)
                else:
                    final_result += word

        return final_result
