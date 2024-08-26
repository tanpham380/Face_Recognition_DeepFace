from typing import Optional, List
from persistent import Persistent

class FaceData(Persistent):
    def __init__(self, uid: str, image_paths: List[str], embedding: Optional[List[list]] = None):
        self.uid = uid
        self.image_paths = image_paths
        self.embedding = embedding if embedding is not None else []

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
            "embedding": self.embedding[0][:2],  
        }

    def to_dict_embedding(self):
        return {
            "uid": self.uid,
            "image_paths": self.image_paths,
            "embedding": self.embedding,  # Ensure this is being correctly populated
        }
