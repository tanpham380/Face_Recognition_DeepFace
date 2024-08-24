import logging
from flask import g, current_app
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.spatial.distance import cosine
from ZODB import FileStorage, DB
import transaction
from persistent import Persistent
from BTrees.OOBTree import OOBTree

class TaskStatus(Persistent):
    def __init__(self, task_id: str, uid: Optional[str], status: str, is_final_task: bool):
        self.task_id = task_id
        self.uid = uid
        self.status = status
        self.is_final_task = is_final_task
        self.created_at = None  # Can add timestamp handling if needed
        self.updated_at = None  # Can add timestamp handling if needed

class FaceEmbedding(Persistent):
    def __init__(self, uid: str, embedding: bytes):
        self.uid = uid
        self.embedding = embedding

class ZODBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        storage = FileStorage.FileStorage(db_path)
        self.db = DB(storage)
        connection = self.db.open()
        self.root = connection.root()

        # Initialize trees if they do not exist
        self._initialize_trees()

    def _initialize_trees(self):
        if not hasattr(self.root, 'embeddings'):
            
            self.root.embeddings = OOBTree()
            self.logger.info("Initialized 'embeddings' tree in the database.")
        if not hasattr(self.root, 'tasks'):
            self.root.tasks = OOBTree()
            self.logger.info("Initialized 'tasks' tree in the database.")
        transaction.commit()

    def close_connection(self):
        
        self.db.close()
        self.logger.info("Closed the database connection.")




    def insert_task(self, task_id: str, uid: Optional[str], status: str, is_final_task: bool):
        task = TaskStatus(task_id, uid, status, is_final_task)
        self.root.tasks[task_id] = task
        transaction.commit()

    def update_task_status(self, task_id: str, status: str):
        task = self.root.tasks.get(task_id)
        if task:
            task.status = status
            transaction.commit()

    def get_active_final_task(self, uid: str) -> Optional[str]:
        for task_id, task in self.root.tasks.items():
            if task.uid == uid and task.status == 'pending':
                return task_id
        return None

    def cancel_task(self, task_id: str):
        if task_id in self.root.tasks:
            del self.root.tasks[task_id]
            transaction.commit()

    def insert_or_update_embedding(self, uid: str, embedding: Any):
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32).tobytes()


        self.root.embeddings[uid] = FaceEmbedding(uid, embedding)
        transaction.commit()


    def delete_embedding_by_uid(self, uid: str):
        if uid in self.root.embeddings:
            del self.root.embeddings[uid]
            transaction.commit()

    def get_embedding_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        embedding = self.root.embeddings.get(uid)
        if embedding:
            return {"uid": embedding.uid, "embedding": embedding.embedding}
        return None

    def get_embeddings(self, uid: str) -> List[Dict[str, Any]]:
        embeddings = []
        for emb_uid, emb in self.root.embeddings.items():
            if emb_uid == uid:
                embeddings.append({"uid": emb.uid, "embedding": emb.embedding})
        return embeddings

    def delete_oldest_embeddings(self, uid: str, n: int):

        embeddings = sorted(
            (emb for emb in self.root.embeddings.values() if emb.uid == uid),
            key=lambda emb: emb.created_at  
        )
        to_delete = embeddings[:n]
        for emb in to_delete:
            del self.root.embeddings[emb.uid]
        transaction.commit()

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        return [{"uid": emb.uid, "embedding": np.frombuffer(emb.embedding, dtype='float32').tolist()} 
                for emb in self.root.embeddings.values()]

    def get_all_uids(self) -> List[str]:
        return list(self.root.embeddings.keys())

    def uid_exists(self, uid: str) -> bool:
        return uid in self.root.embeddings

    def fetch_similar_embeddings(self, target_embedding: List[float], threshold: float) -> List[Dict[str, Any]]:
        def calculate_similarity(embedding1: bytes, embedding2: bytes) -> float:
            emb1 = np.frombuffer(embedding1, dtype='float32')
            emb2 = np.frombuffer(embedding2, dtype='float32')
            return 1 - cosine(emb1, emb2)

        similar_embeddings = []
        for emb in self.root.embeddings.values():
            confidence = calculate_similarity(target_embedding, emb.embedding)
            if confidence >= threshold:
                similar_embeddings.append({"uid": emb.uid, "confidence": confidence})

        return similar_embeddings

    def close(self):
        self.close_connection()
