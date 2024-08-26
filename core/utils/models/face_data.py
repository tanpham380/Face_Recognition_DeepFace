from persistent import Persistent

class FaceData(Persistent):
    def __init__(self, uid: str, image_paths: list , embedder : None):
        self.uid = uid
        self.image_paths = image_paths

    def add_image(self, image_path: str):
        self.image_paths.append(image_path)

    def remove_image(self, image_path: str):
        if image_path in self.image_paths:
            self.image_paths.remove(image_path)
