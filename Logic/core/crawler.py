from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json
import time

class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = []
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_response(self, url):
        res = get(url, headers=self.headers)
        while(not res.ok):
            print("error in getting response: ", res, ' ', url)
            time.sleep(5)
            res = get(url, headers=self.headers) 
        return res
    
    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        return URL.split('/')[4]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        not_crawled_list = list(self.not_crawled)
        with open('IMDB_crawled.json', 'w') as f:
            json.dump(self.crawled, f)

        with open('IMDB_not_crawled.json', 'w') as f:
            json.dump(not_crawled_list, f)

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = json.load(f)
        with open('IMDB_not_crawled.json', 'r') as f:
            data = json.load(f)
        self.not_crawled.extend(data)
        
        self.added_ids = set(u['id'] for u in self.crawled) | set(self.get_id_from_URL(u) for u in self.not_crawled)
        
    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        try:
            res = self.get_response(URL)
            return res
        except Exception as e:
            print(f"Error crawling URL: {URL}")
            print(e)
        return None
    
    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        response = self.crawl(self.top_250_URL)
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            movies = soup.find_all('a', {'class', 'ipc-title-link-wrapper'})
            for movie in movies:
                if movie['href'].startswith('/title'):
                    url = 'https://www.imdb.com' + movie['href']
                    movie_id = self.get_id_from_URL(url)
                    movie_URL = f'https://www.imdb.com/title/{movie_id}/'
                    if movie_id not in self.added_ids:
                        self.not_crawled.append(movie_URL)
                        self.added_ids.add(movie_id)

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while crawled_counter < self.crawling_threshold and self.not_crawled:
                with self.add_queue_lock:
                    URL = self.not_crawled.popleft()
                futures.append(executor.submit(self.crawl_page_info, URL))
                crawled_counter += 1
                if crawled_counter % 100 == 0:
                    print(f"start crawling {crawled_counter} pages")
                #TODO:
                if len(self.not_crawled) == 0:
                    wait(futures)
                    futures = []
            wait(futures)

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        #print("new iteration")
        res = self.crawl(URL)

        if res:
            movie = self.get_imdb_instance()
            self.extract_movie_info(res, movie, URL)
            with self.add_queue_lock:
                self.crawled.append(movie)
        # else:
        #     print(res,' ', URL)

        for m in movie['related_links']:
            if m not in self.added_ids:
                with self.add_queue_lock:
                    self.not_crawled.append(m)
                with self.add_list_lock:
                    self.added_ids.add(self.get_id_from_URL(m))


    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        movie['id'] = self.get_id_from_URL(URL)
        soup = BeautifulSoup(res.text, 'html.parser')
        movie['title'] = self.get_title(soup)
        movie['first_page_summary'] = self.get_first_page_summary(soup)
        movie['release_year'] = self.get_release_year(soup)
        mpaa_soup = BeautifulSoup(self.get_mpaa_lik(URL).text, 'html.parser')
        movie['mpaa'] = self.get_mpaa(mpaa_soup)
        movie['budget'] = self.get_budget(soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(soup)
        movie['directors'] = self.get_director(soup)
        movie['writers'] = self.get_writers(soup)
        movie['stars'] = self.get_stars(soup)
        movie['related_links'] = self.get_related_links(soup)
        movie['genres'] = self.get_genres(soup)
        movie['languages'] = self.get_languages(soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(soup)
        movie['rating'] = self.get_rating(soup)
        summary_soup = BeautifulSoup(self.get_summary_link(URL).text, 'html.parser')
        movie['summaries'] = self.get_summary(summary_soup)
        movie['synopsis'] = self.get_synopsis(summary_soup)
        review_soup = BeautifulSoup(self.get_review_link(URL).text, 'html.parser')
        movie['reviews'] = self.get_reviews_with_scores(review_soup)
    
    def get_mpaa_lik(self, url):
        """
        Get the link to the mpaa page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/parentalguide is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the mpaa page
        """
        try:
            URL = url+'parentalguide'
            res = self.get_response(URL)
            return res
        except:
            print("failed to get mpaa link")

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            URL = url+'plotsummary'
            res = self.get_response(URL)
            return res
        except:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            URL = url+'reviews'
            res = self.get_response(URL)
            return res
        except:
            print("failed to get review link")

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            title_element = soup.find('span', class_='hero__primary-text')
            return title_element.text.strip()
        except:
            print("failed to get title")

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            summary_element = soup.find('span', class_='sc-466bb6c-0 hlbAws')
            return summary_element.text.strip()
        except:
            print("failed to get first page summary")
            return ''

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            div_element = soup.find('div', class_='sc-67fa2588-3 fZhuJ')
            li_element = div_element.find_all('li', {'data-testid':'title-pc-principal-credit'})[0]
            stars =[s.text.strip() for s in li_element.find_all('li', {'class':'ipc-inline-list__item'})]
            return stars
        except:
            print("failed to get director")
            return ['']

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            div_element = soup.find('div', class_='sc-67fa2588-3 fZhuJ')
            li_element = div_element.find_all('li', {'data-testid':'title-pc-principal-credit'})[2]
            stars =[s.text.strip() for s in li_element.find_all('li', {'class':'ipc-inline-list__item'})]
            return stars
        except:
            print("failed to get stars")
            return ['']

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            div_element = soup.find('div', class_='sc-67fa2588-3 fZhuJ')
            li_element = div_element.find_all('li', {'data-testid':'title-pc-principal-credit'})[1]
            stars =[s.text.strip() for s in li_element.find_all('li', {'class':'ipc-inline-list__item'})]
            return stars
        except:
            print("failed to get writers")
            return ['']

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            urls = []
            section_element = soup.find('section', {'data-testid':'MoreLikeThis'})
            div_element = section_element.find('div',{'class':'ipc-shoveler ipc-shoveler--base ipc-shoveler--page0'})
            movies = div_element.find_all('a',{'class':'ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable'})
            for movie in movies:
                parts = movie['href'].split('/')
                urls.append('https://www.imdb.com' + '/'.join(parts[:-1]) + '/')
            return urls
        except:
            print("failed to get related links")

    def get_summary(self, soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            div_element = soup.find('div',{'data-testid':'sub-section-summaries'})
            li_elements = div_element.find_all('li', {'data-testid':'list-item'})
            summeries = []
            for s in li_elements:
                s = s.find('div',{'class':'ipc-html-content-inner-div'})
                summary_text = s.get_text(strip=True)
                # Remove the author's name from the summary text
                author = s.find('span')
                if(author):
                    author_name = author.text.strip()
                    summary_text = summary_text.replace(author_name, '')
                summeries.append(summary_text)
            return summeries
        except:
            print("failed to get summary")
            return ['']

    def get_synopsis(self, soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            div_element = soup.find('div',{'data-testid':'sub-section-synopsis'})
            return [div_element.text.strip()]
        except:
            print("failed to get synopsis")
            return ['']

    def get_reviews_with_scores(self, soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            reviews_with_scores = []

            review_elements = soup.find_all('div',{'class':'review-container'})

            for review_element in review_elements:
                review_text = review_element.find('div', class_='content').get_text(strip=True)

                score_element = review_element.find('span', class_='rating-other-user-rating')
                if score_element:
                    score = score_element.find('span').get_text(strip=True)
                    reviews_with_scores.append((review_text, score))
                else:
                    reviews_with_scores.append((review_text, ''))

            return reviews_with_scores
        except:
            print("failed to get reviews")

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            geners = soup.find_all('a', {'ipc-chip ipc-chip--on-baseAlt'})
            return [g.text.strip() for g in geners]
        except:
            print("Failed to get generes")

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            div_section = soup.find('div',{'data-testid':'hero-rating-bar__aggregate-rating'}).find(
                'div', {'data-testid':'hero-rating-bar__aggregate-rating__score'})
            return div_section.text.strip()
        except:
            print("failed to get rating")
            return ''

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            section_element = soup.find('tr', {'id':'mpaa-rating'}).find_all('td')[1]
            return section_element.text.strip()
        except:
            print("failed to get mpaa")
            return ''

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            div_element = soup.find('div', {'class':'sc-491663c0-3 bdjVSf'})
            div_element = div_element.find('div', {'class':'sc-67fa2588-0 cFndlt'})
            li = div_element.find('li',{'class':'ipc-inline-list__item'})
            return li.text.strip()
        
        except:
            print("failed to get release year")
            return ''

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            ans = []
            section_element = soup.find('section', {'data-testid':'Details'})
            div_element = section_element.find('div', {'data-testid':'title-details-section'})
            li_element = div_element.find('li', {'data-testid':'title-details-languages'})
            languages = li_element.find_all('a')
            for lan in languages:
                ans.append(lan.text.strip())
            return ans    
        except:
            print("failed to get languages")
            return ['']

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            ans = []
            section_element = soup.find('section', {'data-testid':'Details'})
            div_element = section_element.find('div', {'data-testid':'title-details-section'})
            li_element = div_element.find('li', {'data-testid':'title-details-origin'})
            countries = li_element.find_all('a')
            for country in countries:
                ans.append(country.text.strip())
            return ans
        except:
            print("failed to get countries of origin")
            return['']

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            section_element = soup.find('section', {'data-testid':'BoxOffice'})
            div_element = section_element.find('div', {'data-testid':'title-boxoffice-section'})
            li_element = div_element.find('li', {'data-testid':'title-boxoffice-budget'})
            span_element = li_element.find('span', {'class':'ipc-metadata-list-item__list-content-item'})
            return span_element.text.strip()
        except:
            print("failed to get budget")
            return ''

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            section_element = soup.find('section', {'data-testid':'BoxOffice'})
            div_element = section_element.find('div', {'data-testid':'title-boxoffice-section'})
            li_element = div_element.find('li', {'data-testid':'title-boxoffice-grossdomestic'})
            span_element = li_element.find('span', {'class':'ipc-metadata-list-item__list-content-item'})
            return span_element.text.strip()
        except:
            print("failed to get gross worldwide")
            return ''


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1000)
    imdb_crawler.read_from_file_as_json()
    #imdb_crawler.start_crawling()
    print(f"{len(imdb_crawler.crawled)} pages crawled")
    #imdb_crawler.write_to_file_as_json()
if __name__ == '__main__':
    main()
