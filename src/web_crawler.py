import requests
import re
from bs4 import BeautifulSoup

class WebScraper:
    def __init__(self, user_agent='windows'):
        # Initialize the scraper with a user agent (default is 'macOS')
        self.headers = self._get_headers(user_agent)

    def _get_headers(self, user_agent):
        # Private method to get headers for the request based on the specified user agent
        if user_agent == 'macOS':
            # Headers for macOS user agent
            return {
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"macOS"',
            }
        else:
            # Headers for Windows user agent
            return {
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
            }

    def get_webpage_html(self, url):
        # Fetch the HTML content of a webpage from a given URL
        response = requests.Response()  # Create an empty Response object
        if url.endswith(".pdf"):
            # Skip PDF files which are time consuming
            # print('pdf')
            return response

        try:
            # Attempt to get the webpage content with specified headers and timeout
            response = requests.get(url, headers=self.headers, timeout=8)
            response.encoding = "utf-8"
        except requests.exceptions.Timeout:
            # Add timeout exception handling here
            # print('timeout')
            return response
        
        return response

    def convert_html_to_soup(self, html):
        # Convert the HTML string to a BeautifulSoup object for parsing
        html_string = html.text
        return BeautifulSoup(html_string, "lxml")

    def extract_main_content(self, html_soup, rule=0):
        # Extract the main content from a BeautifulSoup object
        main_content = []
        # tag_rule = re.compile("^(h[1-6]|p|div)" if rule == 1 else "^(h[1-6]|p)")

        if rule == 1:
            tag_rule = re.compile("^(h[1-6]|p|div|section|article)$")
        else:
            tag_rule = re.compile("^(h[1-6]|p|div)$")

        # Iterate through specified tags and collect their text
        for tag in html_soup.find_all(tag_rule):
            tag_text = tag.get_text().strip()
            if tag_text and len(tag_text.split()) > 10:
                main_content.append(tag_text)
        return "\n".join(main_content).strip()

    def scrape_url(self, url, rule=0):
        # Public method to scrape a URL and extract its main content
        webpage_html = self.get_webpage_html(url)
        # print(webpage_html)
        soup = self.convert_html_to_soup(webpage_html)
        main_content = self.extract_main_content(soup, rule)
        return main_content

# Example usage
if __name__ == "__main__":
    scraper = WebScraper(user_agent='windows')
    test_url = "https://www.kkday.com/zh-tw/blog/29786/asia-china-shanghai-must-go"
    main_content = scraper.scrape_url(test_url)
    print(main_content)
    