import requests

class WebSearch:
    def __init__(self, api_key=None, engine='google'):
        self.api_key = api_key
        self.engine = engine

    def search(self, query, top_k=5):
        # Mock web search results if API key not provided
        if not self.api_key:
            return [
                {
                    'title': f'Web Search Result for "{query}" - 1',
                    'link': 'https://example.com/result1',
                    'snippet': f'This is a simulated search result for the query: "{query}". In a production environment, this would be a real result from Serper, Bing, or another search API.',
                    'source': 'web (simulated)'
                },
                {
                    'title': f'Web Search Result for "{query}" - 2',
                    'link': 'https://example.com/result2',
                    'snippet': f'A second simulated result for "{query}". Add your Serper API key in .streamlit/secrets.toml to get real search results.',
                    'source': 'web (simulated)'
                }
            ]
        
        # Real search using API (if key is provided)
        try:
            url = "https://serper.dev/search"
            headers = {'X-API-KEY': self.api_key}
            params = {'q': query, 'num': top_k}
            r = requests.get(url, headers=headers, params=params)
            data = r.json()
            results = []
            for item in data.get('organic', []):
                results.append({
                    'title': item.get('title'), 
                    'link': item.get('link'), 
                    'snippet': item.get('snippet'), 
                    'source': 'web'
                })
            return results
        except Exception as e:
            print(f"Web search error: {e}")
            return [
                {
                    'title': f'Web Search Error',
                    'link': 'https://example.com/error',
                    'snippet': f'Error performing web search: {str(e)}',
                    'source': 'web (error)'
                }
            ]
