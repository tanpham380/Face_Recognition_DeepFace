import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BASE_PATH = "core"
DB_PATH = os.path.join(BASE_PATH, "database", "face_recognition.db")
IMAGES_DIR = os.path.join(BASE_PATH, "static", "images")
VECTOR_SIZE = 512
MAX_SCORE = -1
API_KEY = os.getenv("API_KEY")
LOG_PATH_FILE = os.path.join(BASE_PATH, "logs")
NUMBER_WORKER = 4  
