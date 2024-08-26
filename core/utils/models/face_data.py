from typing import Optional
from persistent import Persistent

class FaceData(Persistent):
    def __init__(self, uid: str, image_paths: list, directory_hash_key: Optional[str] = None):
        self.uid = uid
        self.image_paths = image_paths
        self.embedding = []
        self.directory_hash_key = directory_hash_key  # Reference to the directory hash key


    def add_image(self, image_path: str, embedding: Optional[list] = None):
        self.image_paths.append(image_path)
        if embedding:
            self.embedding.append(embedding)

    def remove_image(self, image_path: str):
        if image_path in self.image_paths:
            index = self.image_paths.index(image_path)
            self.image_paths.remove(image_path)
            if index < len(self.embedding):
                self.embedding.pop(index)
    def to_dict(self):
        return {
            "uid": self.uid,
            "image_paths": self.image_paths,
            # "embedding": self.embedding,
        }
    def to_dict_embedding(self):
        return {
            "uid": self.uid,
            "image_paths": self.image_paths,
            "embedding": self.embedding,
        }