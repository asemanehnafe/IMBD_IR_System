from typing import Dict, List
from .core.search import SearchEngine
from .core.utility.spell_correction import SpellCorrection
from .core.utility.snippet import Snippet
from .core.indexer.indexes_enum import Indexes, Index_types
import json

def load_dataset():
    with open('IMDB_crawled.json', 'r') as f:
        return json.load(f)
    
def get_all_doc_string():
    all_doc_string = []
    for movie in movies_dataset:
        for field in movie:
            if isinstance(movie[field], str):
                all_doc_string.append('  '.join(star.lower() for star in movie[field]))
            elif field != 'reviews':
                all_doc_string.append('  '.join(star.lower() for star in movie[field]))
            else:
                for review in  movie[field]:
                    all_doc_string.append('  '.join(star.lower() for star in review))

    # for movie in movies_dataset:
    #         all_doc_string.append('  '.join(star for star in movie['stars']))
    #         all_doc_string.append('  '.join(star for star in movie['genres']))
    #         all_doc_string.append('  '.join(star for star in movie['summaries']))
    return all_doc_string

movies_dataset = load_dataset()
search_engine = SearchEngine()
all_movies_string = get_all_doc_string()


def correct_text(text: str, all_documents: List[str]) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text
    all_documents : list of str
        The input documents.

    Returns
    str
        The corrected form of the given text
    """
    # TODO: You can add any preprocessing steps here, if needed!
    # spell_correction_obj = SpellCorrection(all_documents)
    # text = spell_correction_obj.spell_check(text)
    return text


def search(
    query: str,
    max_result_count: int,
    method: str = "ltn-lnn",
    weights: list = [0.3, 0.3, 0.4],
    safe_method: str = 'safe',
    smoothing_method=None,
    alpha: int=0.5,
    lamda: int=0.5,
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    query:
        The query text

    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    weights:
        The list, containing importance weights in the search result for each of these items:
            Indexes.STARS: weights[0],
            Indexes.GENRES: weights[1],
            Indexes.SUMMARIES: weights[2],

    preferred_genre:
        A list containing preference rates for each genre. If None, the preference rates are equal.
        (You can leave it None for now)

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    weights = {
        Indexes.STARS: weights[0],
        Indexes.GENRES: weights[1],
        Indexes.SUMMARIES: weights[2]
    }
    safe_ranking = False
    if (safe_method == 'safe'):
        safe_ranking = True
    return search_engine.search(
        query, method, weights, safe_ranking, max_result_count, smoothing_method, alpha, lamda
    )

def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """
    result = {}
    for movie in movies_dataset:
        if movie.get("id") == id:
            result = movie

    result["Image_URL"] = (
        "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"  # a default picture for selected movies
    )
    result["URL"] = (
        f"https://www.imdb.com/title/{result['id']}"  # The url pattern of IMDb movies
    )
    return result