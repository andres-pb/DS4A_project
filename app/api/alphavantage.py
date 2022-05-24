import requests
from typing import Dict, List, Any
from os import environ
import logging
_LOG = logging.getLogger(__name__)
class Alphavantage:
    def __init__(self, key: str) -> None:
        self.key = key
        
    @staticmethod
    def result_status(result: Dict[str, str]) -> List[Any]:
        if any([x in result for x in ["Note", "Error Message"]]) or not result:
            return [False, result]
        if any([x in result for x in ["Information"]]):
            return [False, {'FATAL_ERROR'}]
        return [True, result]
    
    def connection_check(fun) -> any:
        def wrapper(self, *args, **kwargs):
            import time
            count=0
            while True:
                try:
                    value=fun(self, *args, **kwargs)
                    return value
                except:
                    _LOG.debug(f"There was an error with ALPHAVANTAGE the connection. This is the [{count}] try")
                    if count >=3:
                        return self.result_status({})
                    time.sleep(20)
                    count+=1
                    pass
        return wrapper

    @connection_check
    def intraday(self, ticker:str) -> Dict[str, str]:
        response = requests.get(f'https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol={ticker}&market=USD&interval=5min&apikey={self.key}')
        return self.result_status(response.json())
    
    @connection_check
    def daily(self, ticker:str) -> Dict[str, str]:
        response = requests.get(f'https://www.alphavantage.co/query?function=FX_MONTHLY&from_symbol={ticker}&to_symbol=USD&apikey={self.key}')
        return self.result_status(response.json())
    
    @connection_check
    def exchange_rate(self, ticker:str) -> Dict[str, str]:
        response = requests.get(f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={ticker}&to_currency=CNY&apikey={self.key}')
        return self.result_status(response.json())