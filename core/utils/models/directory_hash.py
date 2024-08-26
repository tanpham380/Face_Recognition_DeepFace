from persistent import Persistent

class DirectoryHash(Persistent):
    def __init__(self, directory_name: str, hash_value: str):
        self.directory_name = directory_name
        self.hash_value = hash_value

    def update_hash(self, new_hash_value: str):
        self.hash_value = new_hash_value
