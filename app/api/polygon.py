import requests
from typing import Dict, List, Any


# Request Status 200 Good Request  429 Too many requests
class Polygon:
    def __init__(self, key:str) -> None:
        self.key = key
        pass

    @staticmethod
    def result_status(func: Any) -> List[Any]:
        def wrapper(*args, **kwargs):
            result=func(*args, **kwargs)
            if "Note" in result or "error" in result or not result:
                return [False, result]
            return [True, result]
        return wrapper

    @result_status
    def get_old_news(self, tiempo: str) -> Dict[str, str]:
        response = requests.get(
            f'https://api.polygon.io/v2/reference/news?limit=1000&order=ascending&sort=published_utc&published_utc.gte={tiempo}&apiKey={self.key}')
        return response.json()

    @result_status
    def get_news(self, tiempo: str) -> Dict[str, str]:
        response = requests.get(
            f'https://api.polygon.io/v2/reference/news?limit=1000&order=desc&sort=published_utc&published_utc.gte={tiempo}&apiKey={self.key}')
        return response.json()

    @result_status
    def get_news_ticker(self, tiempo: str, ticker: str) -> Dict[str, str]:
        response = requests.get(
            f'https://api.polygon.io/v2/reference/news?limit=10&order=descending&sort=published_utc&ticker={ticker}&published_utc={tiempo}&apiKey={self.key}')
        return response.json()

    @result_status
    def get_market_status(self) -> Dict[str, str]:
        response = requests.get(
            f'https://api.polygon.io/v1/marketstatus/now?apiKey={self.key}')
        return response.json()

    @result_status
    def daily_result(self, ticker:str, time:str):
        response = requests.get(
            f'https://api.polygon.io/v1/open-close/crypto/{ticker}/USD/{time}?adjusted=true&apiKey={self.key}')
        return response.json()

    @result_status
    def previous_close(self,ticker:str='BTCUSD'):
        response = requests.get(
            f'https://api.polygon.io/v2/aggs/ticker/X:{ticker}/prev?adjusted=true&apiKey={self.key}')
        return response.json()