import logging
import sys


def configure_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    root.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stdout)

    formatter = logging.Formatter(
        fmt=f"%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    root.addHandler(handler)
