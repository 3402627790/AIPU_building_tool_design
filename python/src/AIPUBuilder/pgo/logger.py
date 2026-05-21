import logging
import os

logger = logging.getLogger("my_app_logger")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


log_file_path = os.path.join(os.getcwd(), "pgo.log")
if os.path.exists(log_file_path):
    with open(log_file_path, 'w') as f:
        f.truncate(0)

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)


formatter = logging.Formatter('[%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)


logger.propagate = False