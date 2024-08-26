from typing import Optional
from ZODB import DB, FileStorage
import transaction
from persistent.dict import PersistentDict

from core.utils.models.directory_hash import DirectoryHash
from core.utils.models.face_data import FaceData
from core.utils.models.task_queue import TaskQueue
from core.utils.static_variable import IMAGES_DIR, MAX_IMAGES , MAX_ORIGIN_IMAGES
from core.utils.logging import get_logger

logger= get_logger()
class ZoDB_Manager:
    def __init__(self, db_path='database/zodb.fs'):
        storage = FileStorage.FileStorage(db_path)
        self.db = DB(storage)
        self.connection = None
        self.root = None

    def connect(self):
        self.connection = self.db.open()
        self.root = self.connection.root()

        # Initialize persistent objects if they don't exist
        if 'directory_hashes' not in self.root:
            self.root['directory_hashes'] = PersistentDict()
        if 'task_queue' not in self.root:
            self.root['task_queue'] = PersistentDict()
        if 'face_data' not in self.root:
            self.root['face_data'] = PersistentDict()
        transaction.commit()

    def close(self):
        if self.connection:
            if self.connection.isReadOnly() or transaction.isDoomed():
                # If the transaction is doomed, or the connection is read-only, just abort the transaction
                transaction.abort()
            else:
                # Otherwise, commit any pending transactions
                transaction.commit()

            self.connection.close()
        self.db.close()

    def get_directory_hash(self, dir_name: str) -> Optional[DirectoryHash]:
        return self.root['directory_hashes'].get(dir_name)

    def set_directory_hash(self, dir_name: str, hash_value: str):
        self.root['directory_hashes'][dir_name] = DirectoryHash(dir_name, hash_value)
        transaction.commit()

    def del_directory_hash(self, dir_name: str) -> bool:
        """Delete a directory hash by its name."""
        if dir_name in self.root['directory_hashes']:
            del self.root['directory_hashes'][dir_name]
            transaction.commit()
            return True
        return False

    def get_task(self, task_id: str) -> Optional[TaskQueue]:
        return self.root['task_queue'].get(task_id)

    def add_task(self, task_id: str, status: str, created_at: float):
        self.root['task_queue'][task_id] = TaskQueue(task_id, status, created_at)
        transaction.commit()

    def get_face_data(self, uid: str) -> Optional[FaceData]:
        return self.root['face_data'].get(uid)

    def add_face_data(self, uid: str, image_paths: list):
        self.root['face_data'][uid] = FaceData(uid, image_paths)
        transaction.commit()
        
        
    def add_face_embedding(self, uid: str, image_path: str, embedding: list):
        face_data = self.get_face_data(uid)
        if not face_data:
            face_data = FaceData(uid, [image_path], [embedding])
            self.root['face_data'][uid] = face_data
        else:
            # Split UID to differentiate between original and augmented images
            is_augmented = 'aug' in uid

            # Define limits
            max_original =  MAX_ORIGIN_IMAGES
            max_augmented = MAX_IMAGES

            if is_augmented:
                if len([path for path in face_data.image_paths if 'aug' in path]) >= max_augmented:
                    oldest_augmented_index = next(i for i, path in enumerate(face_data.image_paths) if 'aug' in path)
                    face_data.image_paths.pop(oldest_augmented_index)
                    face_data.embedding.pop(oldest_augmented_index)
            else:
                if len([path for path in face_data.image_paths if 'aug' not in path]) >= max_original:
                    oldest_original_index = next(i for i, path in enumerate(face_data.image_paths) if 'aug' not in path)
                    face_data.image_paths.pop(oldest_original_index)
                    face_data.embedding.pop(oldest_original_index)

            face_data.add_image(image_path, embedding)
        transaction.commit()

    def get_face_embedding(self, uid: str) -> Optional[list]:
        face_data = self.get_face_data(uid)
        if face_data:
            return face_data.embedding
        return None
    def list_face_data(self, uid_filter: Optional[str] = None) -> dict:
        """List all face data or filter by UID."""
        if uid_filter:
            face_data = self.root['face_data'].get(uid_filter)
            return {uid_filter: face_data.to_dict()} if face_data else {}
        else:
            return {uid: data.to_dict() for uid, data in self.root['face_data'].items()}

    def list_face_data_embedding(self, uid_filter: Optional[str] = None) -> dict:
        """List all face data or filter by UID."""
        if uid_filter:
            face_data = self.root['face_data'].get(uid_filter)
            return {uid_filter: face_data.to_dict_embedding()} if face_data else {}
        else:
            return {uid: data.to_dict_embedding() for uid, data in self.root['face_data'].items()}
    def list_all_directory_hashes(self) -> dict:
        """List all directory hashes."""
        return {dir_name: hash_obj.hash_value for dir_name, hash_obj in self.root['directory_hashes'].items()}

    def list_all_tasks(self) -> dict:
        """List all tasks in the task queue."""
        return {task_id: task.status for task_id, task in self.root['task_queue'].items()}
    
    def delete_face_embedding(self, uid: str, image_path: Optional[str] = None) -> bool:
        """
        Delete embedding data for a specific UID. If image_path is provided, 
        delete the embedding for that specific image; otherwise, delete all embeddings for the UID and its augmentations.
        """
        # Attempt to delete all UIDs starting with the base UID (to include augmentations)
        base_uid = uid.split('-')[0]

        keys_to_delete = [key for key in self.root['face_data'].keys() if key.startswith(base_uid)]
        if not keys_to_delete:
            logger.error(f"No entries found for UID {base_uid} or its augmentations.")
            return False

        if image_path:
            # Delete only the specified image path
            face_data = self.get_face_data(uid)
            if face_data and image_path in face_data.image_paths:
                index = face_data.image_paths.index(image_path)
                face_data.image_paths.pop(index)
                if index < len(face_data.embedding):
                    face_data.embedding.pop(index)
                transaction.commit()
                logger.info(f"Deleted image and embedding for UID {uid} at path {image_path}.")
                return True
            else:
                logger.error(f"Image path {image_path} not found for UID {uid}.")
                return False  # Image path not found
        else:
            # Delete all data for the UID and its augmentations
            for key in keys_to_delete:
                del self.root['face_data'][key]
            transaction.commit()
            logger.info(f"Deleted all embeddings and data for UID {base_uid} and its augmentations.")
            return True

