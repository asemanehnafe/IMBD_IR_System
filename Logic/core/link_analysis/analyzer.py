from graph import LinkGraph
import networkx as nx
import json
import random

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            self.graph.add_node(movie["title"])
            for star in movie["stars"]:
                self.graph.add_edge(movie["title"], star)
                if star not in self.authorities:
                    self.authorities.append(star)
            self.hubs.append(movie["title"])

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            for star in movie["stars"]:
                for root_movie in self.root_set:
                    if root_movie["stars"] == star:
                        self.graph.add_node(movie["title"])
                        self.graph.add_edge(movie["title"], star)
                        if star not in self.authorities:
                            self.authorities.append(star)
                        if movie["title"] not in self.hubs:
                            self.hubs.append(movie["title"])

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """

        # Run Hits algorithm
        h, a = nx.hits(self.graph.get_graph(), max_iter=num_iteration)
        # Sort the hubs and authorities by their scores
        sorted_hubs = sorted(h, key=h.get, reverse=True)[:max_result]
        sorted_authorities = sorted(a, key=a.get, reverse=True)[:max_result]
        return sorted_authorities, sorted_hubs

def load_dataset():
    with open('IMDB_crawled.json', 'r') as f:
        return json.load(f)
    
if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    corpus = load_dataset()    # TODO: it shoud be your crawled data
    root_set = random.sample(corpus, 770)   # TODO: it shoud be a subset of your corpus

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
