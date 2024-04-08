class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # Remove stop words from the query
        stopwords_path = 'Logic/core/stopwords.txt'
        with open(stopwords_path, 'r') as file:
            stop_words = set(file.read().splitlines())
        query_words = query.split()
        query_without_stop_words = " ".join([word for word in query_words if word.lower() not in stop_words])
        return query_without_stop_words


    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        # Split the document and query into words
        doc_words = doc.split()
        query_words = self.remove_stop_words_from_query(query).split()

        # Iterate through the query words
        for query_word in query_words:
            # Find the index of the query word in the document
            occurrences = [i for i, word in enumerate(doc_words) if word.lower() == query_word.lower()]
            if not occurrences:
                not_exist_words.append(query_word)
                continue
            
            # Find the nearest occurrence of the query word in the document
            nearest_occurrence = occurrences[-1]

            # Get the snippet around the query word
            start_index = max(0, nearest_occurrence - self.number_of_words_on_each_side)
            end_index = min(len(doc_words), nearest_occurrence + self.number_of_words_on_each_side + 1)
            snippet_words = doc_words[start_index:end_index]

            # Highlight the query word in the snippet
            for i, word in enumerate(snippet_words):
                if word.lower() == query_word.lower():
                    snippet_words[i] = f"***{query_word}***"

            # Concatenate the snippet words
            snippet = " ".join(snippet_words)

            # Add the snippet to the final snippet
            final_snippet += snippet + " ... "

        # Remove the trailing " ..." and strip any extra spaces
        final_snippet = final_snippet.rstrip(" ... ").strip()

        return final_snippet, not_exist_words

snippet = Snippet()
print(snippet.find_snippet('hello, I am Asemaneh the king of the Iran','Asemaneh the king this'))