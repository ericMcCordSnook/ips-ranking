import logging

class Logger:
    def __init__(self, filename, level=logging.DEBUG):
        logging.basicConfig(level=level,
                            format="%(asctime)s %(levelname)-8s %(message)s",
                            datefmt="%m-%d-%Y %H:%M:%S",
                            filename=filename,
                            filemode='w')

    def debug(self, text):
        logging.debug(text)

    def info(self, text):
        logging.info(text)

    def warning(self, text):
        logging.warning(text)
