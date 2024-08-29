import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
import tempfile
TEMP_DIR = tempfile.gettempdir()

BASE_PATH = "core"
DB_PATH = os.path.join(BASE_PATH, "database", "zodb.fs")
IMAGES_DIR = os.path.join(BASE_PATH, "static", "images")
VECTOR_SIZE = 512
MAX_SCORE = -1
API_KEY = os.getenv("API_KEY")
LOG_PATH_FILE = os.path.join(BASE_PATH, "logs")
NUMBER_WORKER = int( min(16, os.cpu_count() or 1) / 2)
MAX_ORIGIN_IMAGES = 4
MAX_IMAGES = 20
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
