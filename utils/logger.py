import logging


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Create and configure a logger instance for the given module or name.

    Args:
        name (str, optional): Name of the logger. Defaults to the module name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        format="%(asctime)s:%(module)s:%(funcName)s:%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger