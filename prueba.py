from app.log import logging_config
from os.path import dirname, join
from dotenv import load_dotenv
import logging
from app.util.message import starting_message
from app import yahoo_finance, Polygon, Alphavantage
import pandas as pd
import time
from datetime import datetime 