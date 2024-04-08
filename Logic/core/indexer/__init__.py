# import json
# from .indexes_enum import Indexes
# import sys
# sys.path.append('d:\\dars\\MIR project 2024\\IMBD_IR_System')
# from Logic.core.preprocess import Preprocessor
# from .tiered_index import Tiered_index
# from .index import Index
# from .metadata_index import Metadata_index
# from .document_lengths_index import DocumentLengthsIndex

# with open('IMDB_crawled.json', 'r') as f:
#     movies = json.load(f)
# preprocessor = Preprocessor(movies)
# movies = preprocessor.preprocess()
# indexer = Index(movies)
# indexer.store_index('./index', Indexes.DOCUMENTS.value)
# indexer.store_index('./index', Indexes.STARS.value)
# indexer.store_index('./index', Indexes.GENRES.value)
# indexer.store_index('./index', Indexes.SUMMARIES.value)
# indexer.check_add_remove_is_correct()
# indexer.load_index('./index')
# indexer.check_if_indexing_is_good(Indexes.SUMMARIES)
# tiered = Tiered_index(
#     path="index/"
# )
# meta_index = Metadata_index()
# document_lengths_index = DocumentLengthsIndex()
# print('Document lengths index stored successfully.')