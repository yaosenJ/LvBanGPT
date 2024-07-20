import requests
import re
import json
import yaml
import os
serper_api_key = os.environ.get("serper_api_key")
class SerperClient:
    def __init__(self):
        # Set up the URL and headers for the Serper API
        self.url = "https://google.serper.dev/search"
        self.headers = {
            "X-API-KEY": serper_api_key,  # API key from config file
            "Content-Type": "application/json"
        }

    def serper(self, query: str):
        # Configure the query parameters for Serper API
        serper_settings = {"q": query, "page": 2}

        # Check if the query contains Chinese characters and adjust settings accordingly
        if self._contains_chinese(query):
            serper_settings.update({"gl": "cn", "hl": "zh-cn",})

        payload = json.dumps(serper_settings)

        # Perform the POST request to the Serper API and return the JSON response
        response = requests.request("POST", self.url, headers=self.headers, data=payload)
        return response.json()

    def _contains_chinese(self, query: str):
        # Check if a string contains Chinese characters using a regular expression
        pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(pattern.search(query))

    def extract_components(self, serper_response: dict):
        # Initialize lists to store the extracted components
        titles, links, snippets = [], [], []

        # Iterate through the 'organic' section of the response and extract information
        for item in serper_response.get("organic", []):
            titles.append(item.get("title", ""))
            links.append(item.get("link", ""))
            snippets.append(item.get("snippet", ""))

        # Retrieve additional information from the response
        query = serper_response.get("searchParameters", {}).get("q", "")
        count = len(links)
        language = "zh-cn" if self._contains_chinese(query) else "en-us"

        # Organize the extracted data into a dictionary and return
        output_dict = {
            'query': query, 
            'language': language, 
            'count': count, 
            'titles': titles, 
            'links': links, 
            'snippets': snippets
        }

        return output_dict

# Usage example
if __name__ == "__main__":    
    client = SerperClient()
    query = "上海旅游攻略。"
    response = client.serper(query)
    print(response)
    print('*'*50)
    components = client.extract_components(response)
    print(components)
