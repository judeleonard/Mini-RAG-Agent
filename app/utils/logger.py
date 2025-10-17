import logging

def error_logger():
    # Setup logger
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)

    return logger