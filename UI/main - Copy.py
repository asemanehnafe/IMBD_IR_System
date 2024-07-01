import streamlit as st
import sys
import os
import random
import time
from enum import Enum
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy

# sys.path.append("../")
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append('d:\\dars\\MIR project 2024\\IMBD_IR_System')
from Logic import utils
from Logic.core.utility.snippet import Snippet
from Logic.core.link_analysis.analyzer import LinkAnalyzer
from Logic.core.indexer.index_reader import Index_reader, Indexes

snippet_obj = Snippet()

class color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    WHITE = "#FFFFFF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"

def get_top_x_movies_by_rank(x: int, results: list):
    path = "./index"  # Link to the index folder
    document_index = Index_reader(path, Indexes.DOCUMENTS)
    corpus = []
    root_set = []
    for movie_id, movie_detail in document_index.index.items():
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        corpus.append({"id": movie_id, "title": movie_title, "stars": stars})

    for element in results:
        movie_id = element[0]
        movie_detail = document_index.index[movie_id]
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        root_set.append({"id": movie_id, "title": movie_title, "stars": stars})
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=x)
    return actors, movies

def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
                )
    return summary

def search_time(start, end):
    st.success("Search took: {:.6f} milli-seconds".format((end - start) * 1e3))

def retrieve_webpage(link):
    """Retrieves the content of a webpage."""
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    with requests.Session() as sess:
        sess.headers.update({'User-Agent': user_agent})
        try:
            resp = sess.get(link)
            resp.raise_for_status()  # Raise HTTPError for bad responses
            return resp.content
        except requests.exceptions.RequestException as err:
            print(f"Failed to fetch the page: {err}")
            return None

def find_image_url(soup_obj):
    """Finds the image URL within a BeautifulSoup object."""
    img_container = soup_obj.select_one('div.ipc-media__img')
    if img_container:
        img_tag = img_container.find('img')
        if img_tag and img_tag.get('src'):
            return img_tag['src']
        else:
            print("Image tag missing or 'src' attribute not found.")
    else:
        print("Image container not located.")
    return None

def fetch_image_url(link):
    """Primary function to get the image URL from a provided webpage link."""
    page_content = retrieve_webpage(link)
    if page_content:
        soup_obj = BeautifulSoup(page_content, "html.parser")
        img_url = find_image_url(soup_obj)
    else:
        img_url = None

    return img_url

def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
    safe_method,
    unigram_smoothing,
    alpha,
    lamda,
    filter_button,
    num_filter_results,
):
    if filter_button:
        if "search_results" in st.session_state:
            top_actors, top_movies = get_top_x_movies_by_rank(
                num_filter_results, st.session_state["search_results"]
            )
            st.markdown(f"**Top {num_filter_results} Actors:**")
            actors_ = ", ".join(top_actors)
            st.markdown(
                f"<span style='color:{random.choice(list(color)).value}'>{actors_}</span>",
                unsafe_allow_html=True,
            )
            st.divider()

        st.markdown(f"**Top {num_filter_results} Movies:**")
        for i in range(len(top_movies)):
            card = st.columns([3, 1])
            info = utils.get_movie_by_id(top_movies[i], utils.movies_dataset)
            with card[0].container():
                st.title(info["title"])
                st.markdown(f"[Link to movie]({info['URL']})")
                st.markdown(
                    f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                    unsafe_allow_html=True,
                )

            with st.container():
                st.markdown("**Directors:**")
                num_authors = len(info["directors"])
                for j in range(num_authors):
                    st.text(info["directors"][j])

            with st.container():
                st.markdown("**Stars:**")
                num_authors = len(info["stars"])
                stars = "".join(star + ", " for star in info["stars"])
                st.text(stars[:-2])

                topic_card = st.columns(1)
                with topic_card[0].container():
                    st.write("Genres:")
                    num_topics = len(info["genres"])
                    for j in range(num_topics):
                        st.markdown(
                            f"<span style='color:{random.choice(list(color)).value}'>{info['genres'][j]}</span>",
                            unsafe_allow_html=True,
                        )
            with card[1].container():
                st.image(fetch_image_url[info['URL']], use_column_width=True)

            st.divider()
        return
    if search_button:
        corrected_query = utils.correct_text(search_term, utils.all_movies_string)

        if corrected_query.lower() != search_term.lower():
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            
            result = utils.search(
                query = search_term,
                max_result_count = search_max_num,
                method = search_method,
                weights = search_weights,
                safe_method = safe_method,
                smoothing_method = unigram_smoothing,
                alpha = alpha,
                lamda = lamda,
            )
            if "search_results" in st.session_state:
                st.session_state["search_results"] = result
            print(f"Result: {result}")
            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

        for i in range(len(result)):
            card = st.columns([3, 1])
            info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
            with card[0].container():
                st.title(info["title"])
                st.markdown(f"[Link to movie]({info['URL']})")
                st.write(f"Relevance Score: {result[i][1]}")
                st.markdown(
                    f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                    unsafe_allow_html=True,
                )

            with st.container():
                st.markdown("**Directors:**")
                num_authors = len(info["directors"])
                for j in range(num_authors):
                    st.text(info["directors"][j])

            with st.container():
                st.markdown("**Stars:**")
                num_authors = len(info["stars"])
                stars = "".join(star + ", " for star in info["stars"])
                st.text(stars[:-2])

                topic_card = st.columns(1)
                with topic_card[0].container():
                    st.write("Genres:")
                    num_topics = len(info["genres"])
                    for j in range(num_topics):
                        st.markdown(
                            f"<span style='color:{random.choice(list(color)).value}'>{info['genres'][j]}</span>",
                            unsafe_allow_html=True,
                        )
            with card[1].container():
                st.image(info["Image_URL"], use_column_width=True)

            st.divider()

        st.session_state["search_results"] = result
        if "filter_state" in st.session_state:
            st.session_state["filter_state"] = (
                "search_results" in st.session_state
                and len(st.session_state["search_results"]) > 0
            )

def main():
    st.set_page_config(page_title="Movie Genre Predictor", page_icon=":clapper:", layout="centered")
    st.title("Search Engine")
    st.write(
        "This is a simple search engine for IMDB movies. You can search through IMDB dataset and find the most relevant movie to your search terms."
    )
    st.markdown(
        '<span style="color:yellow">Developed By: MIR Team at Sharif University</span>',
        unsafe_allow_html=True,
    )

    if "user_avatar" not in st.session_state:
        st.session_state["user_avatar"] = None

    user_type = st.sidebar.radio(
        "Login as:",
        ("Guest", "User")
    )

    if user_type == "User":
        login_option = st.sidebar.radio(
            "Choose an option:",
            ("Login", "Register")
        )

        if login_option == "Login":
            st.sidebar.header("Login")
            login_email = st.sidebar.text_input("Email")
            login_password = st.sidebar.text_input("Password", type="password")
            login_button = st.sidebar.button("Login")

            if login_button:
                # Placeholder login logic
                st.sidebar.success("Logged in successfully!")
                st.sidebar.header(f"Welcome, {login_email}")

        if login_option == "Register":
            st.sidebar.header("Register")
            register_username = st.sidebar.text_input("Username")
            register_email = st.sidebar.text_input("Email")
            register_password = st.sidebar.text_input("Password", type="password")
            register_avatar = st.sidebar.file_uploader("Upload Avatar", type=["png", "jpg", "jpeg"])

            if register_avatar is not None:
                st.session_state["user_avatar"] = Image.open(register_avatar)

            register_button = st.sidebar.button("Register")

            if register_button:
                if register_avatar is not None:
                    st.session_state["user_avatar"] = Image.open(register_avatar)
                # Placeholder registration logic
                st.sidebar.success("Registered successfully!")
                st.sidebar.header(f"Welcome, {register_username}")
    
    elif user_type == "Guest":
        st.sidebar.header("Continue as Guest")
        st.session_state["user_avatar"] = None

    if st.session_state["user_avatar"]:
        st.sidebar.image(st.session_state["user_avatar"], width=100, caption="Your Avatar")

    search_term = st.text_input("Search Term")
    with st.expander("Advanced Search"):
        search_max_num = st.number_input(
            "Maximum number of results", min_value=5, max_value=100, value=10, step=5
        )
        weight_stars = st.slider(
            "Weight of stars in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_genres = st.slider(
            "Weight of genres in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_summary = st.slider(
            "Weight of summary in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )
        slider_ = st.slider("Select the number of top movies to show", 1, 10, 5)

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox(
            "Search method", ("ltn.lnn", "ltc.lnc", "OkapiBM25", "unigram")
        )
        safe_method = st.selectbox(
            "safe/unsafe",("safe", "unsafe"),
        )

        unigram_smoothing = None
        alpha, lamda = None, None
        if search_method == "unigram":
            unigram_smoothing = st.selectbox(
                "Smoothing method",
                ("naive", "bayes", "mixture"),
            )
            if unigram_smoothing == "bayes":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
            if unigram_smoothing == "mixture":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
                lamda = st.slider(
                    "Lambda",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []

    search_button = st.button("Search!")
    filter_button = st.button("Filter movies by ranking")

    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
        safe_method,
        unigram_smoothing,
        alpha,
        lamda,
        filter_button,
        slider_,
    )

if __name__ == "__main__":
    main()
