import logging
from logging.handlers import TimedRotatingFileHandler
from os.path import join


class LoggingConfig:

    def __init__(self):pass

    def init_logging(self) -> None:
        logging.basicConfig(
            format='[%(thread)d] %(name)s - %(asctime)s - %(levelname)s - %(message)s',
            level=logging.DEBUG,
            handlers={
                TimedRotatingFileHandler(join(__file__, '../../..', 'messages', "information.log"), when='midnight', interval=1)
            }
        )
        return None