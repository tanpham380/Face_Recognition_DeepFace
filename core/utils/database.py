from typing import Optional
from ZODB import DB, FileStorage
import transaction
from persistent.dict import PersistentDict

from core.utils.models.directory_hash import DirectoryHash
from core.utils.models.face_data import FaceData
from core.utils.models.task_queue import TaskQueue
from core.utils.static_variable import IMAGES_DIR, MAX_IMAGES, MAX_ORIGIN_IMAGES
from core.utils.logging import get_logger

logger = get_logger()

class ZoDB_Manager:
    def __init__(self, db_path='database/zodb.fs', commit_threshold=10):
        self.storage = FileStorage.FileStorage(db_path)
        self.db = DB(self.storage)
        self.connection = None
        self.root = None
        self.change_count = 0  # Track the number of changes
        self.commit_threshold = commit_threshold
        self.pending_changes = False  # Track if there are pending changes

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        if not self.connection:
            self.connection = self.db.open()
            self.root = self.connection.root()
            self._initialize_persistent_objects()

    def close(self):
        if self.pending_changes:
            self._commit_transaction()
        if self.connection:
            self.connection.close()
        self.db.close()

    def _commit_transaction(self):
        try:
            transaction.commit()
            self.change_count = 0
            self.pending_changes = False
        except Exception as e:
            logger.error(f"Transaction commit failed: {e}")
            transaction.abort()
            raise

    def _initialize_persistent_objects(self):
        if 'directory_hashes' not in self.root:
            self.root['directory_hashes'] = PersistentDict()
            self.pending_changes = True
        if 'task_queue' not in self.root:
            self.root['task_queue'] = PersistentDict()
            self.pending_changes = True
        if 'face_data' not in self.root:
            self.root['face_data'] = PersistentDict()
            self.pending_changes = True
        self._maybe_commit()  # Commit initialization changes if any

    def _maybe_commit(self):
        if self.change_count >= self.commit_threshold:
            self._commit_transaction()

    # Directory Hashes
    def get_directory_hash(self, dir_name: str) -> Optional[DirectoryHash]:
        return self.root['directory_hashes'].get(dir_name)

    def set_directory_hash(self, dir_name: str, hash_value: str):
        self.root['directory_hashes'][dir_name] = DirectoryHash(dir_name, hash_value)
        self.change_count += 1
        self.pending_changes = True
        self._maybe_commit()

    def del_directory_hash(self, dir_name: str) -> bool:
        """Delete a directory hash by its name."""
        if dir_name in self.root['directory_hashes']:
            del self.root['directory_hashes'][dir_name]
            self.change_count += 1
            self.pending_changes = True
            self._maybe_commit()
            return True
        return False

    def list_all_directory_hashes(self) -> dict:
        """List all directory hashes."""
        return {dir_name: hash_obj.hash_value for dir_name, hash_obj in self.root['directory_hashes'].items()}

    # Task Queue
    def get_task(self, task_id: str) -> Optional[TaskQueue]:
        return self.root['task_queue'].get(task_id)

    def add_task(self, task_id: str, status: str, created_at: float):
        self.root['task_queue'][task_id] = TaskQueue(task_id, status, created_at)
        self.change_count += 1
        self.pending_changes = True
        self._maybe_commit()

    def list_all_tasks(self) -> dict:
        """List all tasks in the task queue."""
        return {task_id: task for task_id, task in self.root['task_queue'].items()}

    def update_task_status(self, task_id: str, new_status: str) -> bool:
        """Update the status of a task in the task queue."""
        task = self.get_task(task_id)
        if task:
            task.update_status(new_status)  # Update status and timestamp
            self.change_count += 1
            self.pending_changes = True
            self._maybe_commit()
            logger.info(f"Updated status for task {task_id} to {new_status}.")
            return True
        else:
            logger.error(f"Task with ID {task_id} not found.")
            return False

    # Face Data
    def get_face_data(self, uid: str) -> Optional[FaceData]:
        return self.root['face_data'].get(uid)

    def add_face_data(self, uid: str, image_paths: list):
        self.root['face_data'][uid] = FaceData(uid, image_paths)
        self.change_count += 1
        self.pending_changes = True
        self._maybe_commit()

    # def add_face_embedding(self, uid: str, image_path: str, embedding: list):
    #     face_data = self.get_face_data(uid)
    #     if not face_data:
    #         face_data = FaceData(uid, [image_path], [embedding])
    #         self.root['face_data'][uid] = face_data
    #     else:
    #         is_augmented = 'aug' in uid
    #         max_images = MAX_IMAGES if is_augmented else MAX_ORIGIN_IMAGES
    #         filtered_paths = [path for path in face_data.image_paths if ('aug' in path) == is_augmented]

    #         if len(filtered_paths) >= max_images:
    #             oldest_index = next(i for i, path in enumerate(face_data.image_paths) if path == filtered_paths[0])
    #             face_data.image_paths.pop(oldest_index)
    #             face_data.embedding.pop(oldest_index)

    #         face_data.add_image(image_path, embedding)
    #     self.change_count += 1
    #     self.pending_changes = True
    #     self._maybe_commit()
    
    
    def add_face_embedding(self, uid: str, image_path: str, embedding: list):
        face_data = self.get_face_data(uid)
        if not face_data:
            face_data = FaceData(uid, [image_path], [embedding])
            self.root['face_data'][uid] = face_data
        else:
            face_data.image_paths = [image_path]  # Ensure there's only one image path
            face_data.embedding = [embedding]     # Ensure there's only one embedding
        self.change_count += 1
        self.pending_changes = True
        self._maybe_commit()


    def get_face_embedding(self, uid: str) -> Optional[list]:
        face_data = self.get_face_data(uid)
        if face_data:
            return face_data.embedding
        return None

    def list_face_data(self, uid_filter: Optional[str] = None) -> dict:
        """List all face data or filter by UID."""
        if uid_filter:
            matching_data = {
            uid: data.to_dict_embedding()
            for uid, data in self.root['face_data'].items()
            if uid.startswith(uid_filter)
        }
            return matching_data
        else:
            return {uid: data.to_dict() for uid, data in self.root['face_data'].items()}

    def list_face_data_embedding(self, uid_filter: Optional[str] = None) -> dict:
        """List all face data or filter by UID."""
        if uid_filter:
            matching_data = {
            uid: data.to_dict_embedding()
            for uid, data in self.root['face_data'].items()
            if uid.startswith(uid_filter)
        }
            return matching_data
        else:
            return {uid: data.to_dict_embedding() for uid, data in self.root['face_data'].items()}

    def delete_face_embedding(self, uid: str, image_path: Optional[str] = None) -> bool:
        """
        Delete embedding data for a specific UID. If image_path is provided, 
        delete the embedding for that specific image; otherwise, delete all embeddings for the UID and its augmentations.
        """
        base_uid = uid.split('-')[0]

        keys_to_delete = [key for key in self.root['face_data'].keys() if key.startswith(base_uid)]
        if not keys_to_delete:
            logger.error(f"No entries found for UID {base_uid} or its augmentations.")
            return False

        if image_path:
            face_data = self.get_face_data(uid)
            if face_data and image_path in face_data.image_paths:
                index = face_data.image_paths.index(image_path)
                face_data.image_paths.pop(index)
                if index < len(face_data.embedding):
                    face_data.embedding.pop(index)
                self.change_count += 1
                self.pending_changes = True
                self._maybe_commit()
                logger.info(f"Deleted image and embedding for UID {uid} at path {image_path}.")
                return True
            else:
                logger.error(f"Image path {image_path} not found for UID {uid}.")
                return False
        else:
            for key in keys_to_delete:
                del self.root['face_data'][key]
            self.change_count += 1
            self.pending_changes = True
            self._maybe_commit()
            logger.info(f"Deleted all embeddings and data for UID {base_uid} and its augmentations.")
            return True
