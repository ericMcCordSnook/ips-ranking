import logging

class Logger:
    def __init__(self, filename, level):
        logging.basicConfig(filename=filename, level=level)

    def debug(self, text):
        logging.debug(text)

    def info(self, text):
        logging.info(text)

    def warning(self, text):
        logging.warning(text)
