from typing import Optional, Union
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
    _instance = None  # Singleton instance

    def __new__(cls, db_path='database/zodb.fs'):
        if cls._instance is None:
            cls._instance = super(ZoDB_Manager, cls).__new__(cls)
            cls._instance._initialize(db_path)
        return cls._instance

    def _initialize(self, db_path: str):
        storage = FileStorage.FileStorage(db_path)
        self.db = DB(storage)
        self.connection = None
        self.root = None

    def connect(self):
        if self.connection is None:
            self.connection = self.db.open()
            self.root = self.connection.root()
            self._initialize_root()

    def close(self):
        if self.connection:
            if self.connection.isReadOnly() or transaction.isDoomed():
                transaction.abort()
            else:
                transaction.commit()

            self.connection.close()
            self.connection = None
        self.db.close()

    def _initialize_root(self):
        if 'directory_hashes' not in self.root:
            self.root['directory_hashes'] = PersistentDict()
        if 'task_queue' not in self.root:
            self.root['task_queue'] = PersistentDict()
        if 'face_data' not in self.root:
            self.root['face_data'] = PersistentDict()
        transaction.commit()

    def _get_root_item(self, item_name: str) -> PersistentDict:
        return self.root.get(item_name, PersistentDict())

    ### FaceData Management Functions ###
    
    def get_face_data(self, uid: str) -> Optional[FaceData]:
        return self._get_root_item('face_data').get(uid)

    def add_face_data(self, uid: str, image_paths: list):
        self.root['face_data'][uid] = FaceData(uid, image_paths)
        transaction.commit()

    def add_face_embedding(self, uid: str, image_path: str, embedding: list):
        face_data = self.get_face_data(uid)
        if not face_data:
            face_data = FaceData(uid, [image_path], [embedding])
            self.root['face_data'][uid] = face_data
        else:
            self._handle_image_embedding_limit(face_data, uid)
            face_data.add_image(image_path, embedding)
        transaction.commit()

    def add_face_data_with_directory_hash(self, uid: str, image_paths: list, dir_name: str):
        directory_hash = self.get_directory_hash(dir_name)
        if directory_hash:
            face_data = FaceData(uid, image_paths, dir_name)
            self.root['face_data'][uid] = face_data
            transaction.commit()
            logger.info(f"Face data added for UID {uid} with directory hash {dir_name}.")
        else:
            logger.error(f"Directory hash {dir_name} not found for UID {uid}.")

    def _handle_image_embedding_limit(self, face_data: FaceData, uid: str):
        is_augmented = 'aug' in uid
        max_original = MAX_ORIGIN_IMAGES
        max_augmented = MAX_IMAGES

        if is_augmented:
            self._remove_oldest(face_data, 'aug', max_augmented)
        else:
            self._remove_oldest(face_data, '', max_original)

    def _remove_oldest(self, face_data: FaceData, key: str, limit: int):
        paths = [path for path in face_data.image_paths if key in path]
        if len(paths) >= limit:
            oldest_index = face_data.image_paths.index(paths[0])
            face_data.image_paths.pop(oldest_index)
            face_data.embedding.pop(oldest_index)

    def get_face_embedding(self, uid: str) -> Optional[list]:
        face_data = self.get_face_data(uid)
        return face_data.embedding if face_data else None

    def list_face_data(self, uid_filter: Optional[str] = None) -> dict:
        return self._list_items('face_data', uid_filter, 'to_dict')

    def list_face_data_embedding(self, uid_filter: Optional[str] = None) -> dict:
        return self._list_items('face_data', uid_filter, 'to_dict_embedding')

    def delete_face_embedding(self, uid: str, image_path: Optional[str] = None) -> bool:
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
                transaction.commit()
                logger.info(f"Deleted image and embedding for UID {uid} at path {image_path}.")
                return True
            else:
                logger.error(f"Image path {image_path} not found for UID {uid}.")
                return False
        else:
            for key in keys_to_delete:
                del self.root['face_data'][key]
            transaction.commit()
            logger.info(f"Deleted all embeddings and data for UID {base_uid} and its augmentations.")
            return True

    ### DirectoryHash Management Functions ###

    def get_directory_hash(self, dir_name: str) -> Optional[DirectoryHash]:
        return self._get_root_item('directory_hashes').get(dir_name)

    def set_directory_hash(self, dir_name: str, hash_value: str):
        self.root['directory_hashes'][dir_name] = DirectoryHash(dir_name, hash_value)
        transaction.commit()

    def del_directory_hash(self, dir_name: str) -> bool:
        return self._delete_item('directory_hashes', dir_name)

    def list_all_directory_hashes(self) -> dict:
        return self._list_simple_items('directory_hashes')

    ### TaskQueue Management Functions ###

    def get_task(self, task_id: str) -> Optional[TaskQueue]:
        return self._get_root_item('task_queue').get(task_id)

    def add_task(self, task_id: str, status: str, created_at: float):
        self._ensure_task_limit()  # Ensure only 100 tasks are stored
        self.root['task_queue'][task_id] = TaskQueue(task_id, status, created_at)
        transaction.commit()

    def update_task_status(self, task_id: str, new_status: str):
        task = self.get_task(task_id)
        if task:
            task.update_status(new_status)
            transaction.commit()
        else:
            logger.error(f"Task {task_id} not found in ZoDB.")

    def list_all_tasks(self) -> dict:
        return self._list_simple_items('task_queue')

    def _ensure_task_limit(self):
        task_count = len(self.root['task_queue'])
        if task_count >= 100:
            # Remove oldest tasks
            self._remove_oldest_tasks(task_count - 99)

    def _remove_oldest_tasks(self, excess_count: int):
        task_queue_items = list(self.root['task_queue'].items())
        # Sort tasks by creation time to identify the oldest ones
        task_queue_items.sort(key=lambda item: item[1].created_at)

        for task_id, _ in task_queue_items[:excess_count]:
            del self.root['task_queue'][task_id]

        transaction.commit()
        logger.info(f"Removed {excess_count} oldest tasks to maintain the task limit.")

    ### Helper Functions ###

    def _delete_item(self, root_item: str, key: str) -> bool:
        if key in self.root[root_item]:
            del self.root[root_item][key]
            transaction.commit()
            return True
        return False

    def _list_items(self, root_item: str, uid_filter: Optional[str], method: str) -> dict:
        items = self._get_root_item(root_item)
        if uid_filter:
            item = items.get(uid_filter)
            return {uid_filter: getattr(item, method)()} if item else {}
        else:
            return {uid: getattr(data, method)() for uid, data in items.items()}

    def _list_simple_items(self, root_item: str) -> dict:
        return {key: item.hash_value if root_item == 'directory_hashes' else item.status
                for key, item in self._get_root_item(root_item).items()}
