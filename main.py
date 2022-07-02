from app.log import logging_config
# from os.path import dirname, join

import logging
from app import database
from app.util.message import starting_message
from app import globals_variable
from os import environ
# from app import yahoo_finance, Polygon, Alphavantage, Twitter, Statistical, Predict, LSTM_model



#load_dotenv(dotenv_path=join(dirname(__file__), '.env'))
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass
#from os import environ

class Main:

    def run(self) -> None:
        logging_config.init_logging()
        __LOG = logging.getLogger(__name__)
        __LOG.info('...... Initialization Completed  ......')
        __LOG.info(starting_message())

        database.init_app()
        globals_variable.COINS_SELECTION=[
                                            {'name':'NMC - Namecoin', 'ticker':'NMC-USD'},
                                            {'name':'FTC - Feathercoin', 'ticker':'FTC-USD'},
                                            {'name':'PPC - Peercoin', 'ticker':'PPC-USD'},
                                            {'name':'LTC -  Litecoin', 'ticker':'LTC-USD'},
                                            {'name':'BTC - Bitcoin', 'ticker':'BTC-USD'},
                                            {'name':'ETH - Ethereum', 'ticker':'ETH-USD'},
                                        ]
        globals_variable.EXCHANGES=[
                                    {'name':'Dolar', 'ticker':'USD'},
                                    {'name':'Peso Colombiano', 'ticker':'COP=X'},
                                    {'name':'Bitcoin', 'ticker':'BTC-USD'}
                                     ]
        globals_variable.STATISTICAL_MODELS=[
                                    {'name':'Volume', 'function':''},
                                    {'name':'Mean Price', 'function':'mean_price'},
                                    {'name':'Rolling Mean Price', 'function':'rolling_mean'},
                                    {'name':'Rolling Standard Deviation', 'function':'rolling_std'},
                                    {'name':'Exponential WMA', 'function':'exponential_wma'},
                                    {'name':'Bollinger Bands', 'function':'BBANDS'},
                                    {'name':'Average Directional Movement Index', 'function':'ADX'},
                                    {'name':'Average True Range', 'function':'ATR'},
                                    {'name':'Moving Average Convergence/Divergence', 'function':'MACD'},
                                     ]
        from app.dashboard import dashboard_app
        print(environ.get("DS4A_ENV"))
        host, debug, port = {
            'development':('127.0.0.1',True, 8888),
            'production':('0.0.0.0',True, 8050)
                }[environ.get("DS4A_ENV")]
        dashboard_app.run_server(host=host,debug=debug, port=port)
Main().run()








"""#Here you can see how implement yahoo module
status, yahoo_data = yahoo_finance.market_value('BTC-USD')
if status:
    print(yahoo_data)


#Here you can see how implement plygon module
polygon=Polygon(environ.get("POLYGON_KEY"))
print('polygon')
status, polygon_data = polygon.get_market_status()
if status:
    print(polygon_data)

status, polygon_data = polygon.get_news_ticker('2022-05-23', 'BTC')
if status:
    print(polygon_data)

status, polygon_data = polygon.get_old_news('2022-05-30')
if status:
    print(polygon_data)


#Here you can see how implement plygon module
alphavantage=Alphavantage(environ.get("ALPHA_KEY"))
print('polygon')
status, alphavantage_data = alphavantage.intraday('ETH')
if status:
    print(alphavantage_data)

#Here you can see how implement twitter
twt = Twitter()
twt.get_tweets_df(query="#Bitcoin OR Bitcoin", limit=20).to_csv('twitter.csv')
print(twt.get_tweets_df(query="#Bitcoin OR Bitcoin", limit=20))


#Here you can see how implement statistical
statistical = Statistical('BTC-USD')
print(statistical.volume())
print('!'*100)
print(statistical.MACD())


#Here you can see how implement predict
predict = Predict()
print(predict.SEX('BTC-USD',10))
print(predict.sequential())


#Here you can see how implement LSTM
for x in range (10):
    print(LSTM_model('BTC-USD', x))

    """