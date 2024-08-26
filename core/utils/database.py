from typing import Optional
from ZODB import DB, FileStorage
import transaction
from persistent.dict import PersistentDict

from core.utils.models.directory_hash import DirectoryHash
from core.utils.models.face_data import FaceData
from core.utils.models.task_queue import TaskQueue

class ZoDB:
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

    def list_face_data(self, uid_filter: Optional[str] = None) -> dict:
        """List all face data or filter by UID."""
        if uid_filter:
            return {uid_filter: self.root['face_data'].get(uid_filter)} if uid_filter in self.root['face_data'] else {}
        else:
            return {uid: data for uid, data in self.root['face_data'].items()}

    def list_all_directory_hashes(self) -> dict:
        """List all directory hashes."""
        return {dir_name: hash_obj.hash_value for dir_name, hash_obj in self.root['directory_hashes'].items()}

    def list_all_tasks(self) -> dict:
        """List all tasks in the task queue."""
        return {task_id: task.status for task_id, task in self.root['task_queue'].items()}
