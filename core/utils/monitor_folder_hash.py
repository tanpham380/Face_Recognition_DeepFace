import hashlib
import os
from typing import Optional

from core.service import recreate_DB
from core.utils.logging import get_logger
from core.utils.static_variable import IMAGES_DIR
from core.utils.threading import add_task_to_queue

logger = get_logger()
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_read(filepath):
    with open(filepath, 'rb') as file:
        return file.read()

def has_directory_changed(directory_path: str, previous_hash: Optional[str] = None) -> (bool, str): # type: ignore
    current_hash = hash_directory(directory_path)
    return previous_hash is None or current_hash != previous_hash, current_hash

def hash_directory(directory_path: str) -> str:
    hash_obj = hashlib.md5()
    for root, _, files in os.walk(directory_path):
        for filename in sorted(files):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(root, filename)
                try:
                    stat = os.stat(filepath)
                    file_content = cached_read(filepath)
                    hash_obj.update(f"{filename}{stat.st_size}{stat.st_mtime}".encode())
                    hash_obj.update(file_content)
                except FileNotFoundError:
                    continue
    return hash_obj.hexdigest()





def check_and_update_directory_hash(dir_name: str, dir_path: str , app):
    """Kiểm tra hash của thư mục và cập nhật vào ZoDB nếu có thay đổi."""
    directory_hash = app.config["ZoDB"].get_directory_hash(dir_name)
    previous_hash = directory_hash.hash_value
    changed, _ = has_directory_changed(dir_path, previous_hash)
    if changed:
        add_task_to_queue(
            recreate_DB,
            img_path=IMAGES_DIR,
            app = app,
            uid=dir_name,
        )
        # app.config["ZoDB"].set_directory_hash(dir_name, current_hash)
