import sqlite3
import logging
import threading
from flask import g, current_app
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.spatial.distance import cosine

class SQLiteManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.local = threading.local()  # Thread-local storage for thread-specific connections

    def optimize_sqlite(self):
        """Optimize SQLite settings for better performance."""
        with self.get_connection() as conn:
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA page_size = 4096")  # Adjust page size as needed

    def get_connection(self):
        if current_app:
            if 'db_connection' not in g:
                g.db_connection = sqlite3.connect(self.db_path)
                g.db_connection.row_factory = sqlite3.Row
                g.db_connection.execute("PRAGMA busy_timeout = 5000")  # 5 seconds
            return g.db_connection
        else:
            if not hasattr(self.local, 'connection'):
                self.local.connection = sqlite3.connect(self.db_path)
                self.local.connection.row_factory = sqlite3.Row
                self.local.connection.execute("PRAGMA busy_timeout = 5000")  # 5 seconds
            return self.local.connection


    def close_connection(self, error=None):
        if current_app and 'db_connection' in g:
            g.db_connection.close()
            g.pop('db_connection', None)
        elif hasattr(self.local, 'connection'):
            self.local.connection.close()
            del self.local.connection

    def is_database_locked(self) -> bool:
        """Check if the database is locked."""
        try:
            with self.get_connection() as conn:
                conn.execute("SELECT 1")
            return False  # No lock if the query succeeds
        except sqlite3.OperationalError as e:
            if 'database is locked' in str(e):
                self.logger.warning("Database is locked.")
                return True
            else:
                raise  # Re-raise if it's not a locking issue

    def execute_with_timeout(self, func, args=(), kwargs=None, timeout=5):
        """Run a database function with a timeout."""
        if kwargs is None:
            kwargs = {}

        result = [None]
        exception = [None]

        def wrapper():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=wrapper)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            self.logger.error("Operation timed out")
            raise TimeoutError("The operation exceeded the timeout limit")

        if exception[0] is not None:
            raise exception[0]

        return result[0]

    def create_table(self):
        self.execute_with_timeout(self._create_table)

    def _create_table(self):
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(uid, embedding)
                )
            """)

    def insert_task(self, task_id: str, uid: Optional[str], status: str, is_final_task: bool):
        self.execute_with_timeout(self._insert_task, args=(task_id, uid, status, is_final_task))

    def _insert_task(self, task_id: str, uid: Optional[str], status: str, is_final_task: bool):
        with self.transaction() as conn:
            try:
                self.logger.info(f"Inserting task {task_id} for UID {uid} with status {status}, final_task: {is_final_task}")
                conn.execute("""
                    INSERT INTO task_status (task_id, uid, status, is_final_task)
                    VALUES (?, ?, ?, ?)
                """, (task_id, uid, status, is_final_task))
            except sqlite3.IntegrityError:
                self.logger.info(f"Updating task {task_id} for UID {uid} with status {status}")
                conn.execute("""
                    UPDATE task_status SET uid = ?, status = ?, updated_at = CURRENT_TIMESTAMP, is_final_task = ?
                    WHERE task_id = ?
                """, (uid, status, is_final_task, task_id))

    def update_task_status(self, task_id: str, status: str):
        self.execute_with_timeout(self._update_task_status, args=(task_id, status))

    def _update_task_status(self, task_id: str, status: str):
        with self.transaction() as conn:
            conn.execute("""
                UPDATE task_status
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
            """, (status, task_id))

    def get_active_final_task(self, uid: str) -> Optional[str]:
        return self.execute_with_timeout(self._get_active_final_task, args=(uid,))

    def _get_active_final_task(self, uid: str) -> Optional[str]:
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT task_id FROM task_status
                WHERE uid = ? AND status = 'pending'
            """, (uid,))
            row = cursor.fetchone()
            if row:
                return row['task_id']
            return None

    def cancel_task(self, task_id: str):
        self.execute_with_timeout(self._cancel_task, args=(task_id,))

    def _cancel_task(self, task_id: str):
        with self.transaction() as conn:
            conn.execute("""
                DELETE FROM task_status WHERE task_id = ?
            """, (task_id,))

    def create_task_status_table(self):
        self.execute_with_timeout(self._create_task_status_table)

    def _create_task_status_table(self):
        with self.transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL UNIQUE,
                    uid TEXT,
                    status TEXT NOT NULL,
                    is_final_task BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_uid ON task_status (uid)
            """)

    def insert_or_update_embedding(self, uid: str, embedding: Any):
        self.execute_with_timeout(self._insert_or_update_embedding, args=(uid, embedding))

    def _insert_or_update_embedding(self, uid: str, embedding: Any):
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32).tobytes()

        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO face_embeddings (uid, embedding)
                VALUES (?, ?)
            """, (uid, sqlite3.Binary(embedding)))

    def delete_embedding_by_uid(self, uid: str):
        self.execute_with_timeout(self._delete_embedding_by_uid, args=(uid,))

    def _delete_embedding_by_uid(self, uid: str):
        with self.transaction() as conn:
            conn.execute("DELETE FROM face_embeddings WHERE uid = ?", (uid,))

    def get_embedding_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        return self.execute_with_timeout(self._get_embedding_by_uid, args=(uid,))

    def _get_embedding_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT uid, embedding FROM face_embeddings WHERE uid = ?", (uid,))
            row = cursor.fetchone()
            if row:
                return {"uid": row["uid"], "embedding": row["embedding"]}
            return None

    def get_embeddings(self, uid: str) -> List[Dict[str, Any]]:
        return self.execute_with_timeout(self._get_embeddings, args=(uid,))

    def _get_embeddings(self, uid: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, embedding FROM face_embeddings WHERE uid = ? ORDER BY id ASC", (uid,))
            rows = cursor.fetchall()
            return [{"id": row["id"], "embedding": row["embedding"]} for row in rows]

    def delete_oldest_embeddings(self, uid: str, n: int):
        self.execute_with_timeout(self._delete_oldest_embeddings, args=(uid, n))

    def _delete_oldest_embeddings(self, uid: str, n: int):
        with self.transaction() as conn:
            cursor = conn.execute(
                "SELECT id FROM face_embeddings WHERE uid = ? ORDER BY created_at ASC LIMIT ?", (uid, n))
            ids_to_delete = [row["id"] for row in cursor.fetchall()]

            if ids_to_delete:
                conn.executemany(
                    "DELETE FROM face_embeddings WHERE id = ?", [(id,) for id in ids_to_delete]
                )

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        return self.execute_with_timeout(self._get_all_embeddings)

    def _get_all_embeddings(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT uid, embedding FROM face_embeddings")
            rows = cursor.fetchall()
            return [{"uid": row["uid"], "embedding": np.frombuffer(row["embedding"], dtype='float32').tolist()} for row in rows]

    def get_all_uids(self) -> List[str]:
        return self.execute_with_timeout(self._get_all_uids)

    def _get_all_uids(self) -> List[str]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT uid FROM face_embeddings")
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def uid_exists(self, uid: str) -> bool:
        return self.execute_with_timeout(self._uid_exists, args=(uid,))

    def _uid_exists(self, uid: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM face_embeddings WHERE uid = ? LIMIT 1", (uid,))
            return cursor.fetchone() is not None

    def fetch_similar_embeddings(self, target_embedding: List[float], threshold: float) -> List[Dict[str, Any]]:
        return self.execute_with_timeout(self._fetch_similar_embeddings, args=(target_embedding, threshold))

    def _fetch_similar_embeddings(self, target_embedding: List[float], threshold: float) -> List[Dict[str, Any]]:
        def calculate_similarity(embedding1: bytes, embedding2: bytes) -> float:
            emb1 = np.frombuffer(embedding1, dtype='float32')
            emb2 = np.frombuffer(embedding2, dtype='float32')
            return 1 - cosine(emb1, emb2)

        with self.get_connection() as conn:
            cursor = conn.execute("SELECT uid, embedding FROM face_embeddings")
            rows = cursor.fetchall()
            similar_embeddings = [
                {
                    "uid": row["uid"],
                    "confidence": calculate_similarity(target_embedding, row["embedding"])
                }
                for row in rows
                if calculate_similarity(target_embedding, row["embedding"]) >= threshold
            ]
            return similar_embeddings

    def execute_raw_sql(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        return self.execute_with_timeout(self._execute_raw_sql, args=(query, params))

    def _execute_raw_sql(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def transaction(self):
        """Context manager for transaction management."""
        class TransactionContext:
            def __init__(self, manager):
                self.manager = manager
                self.conn = None

            def __enter__(self):
                self.conn = self.manager.get_connection()
                self.manager.logger.info("Starting a new transaction...")
                self.conn.execute("BEGIN TRANSACTION")
                return self.conn

            def __exit__(self, exc_type, exc_value, traceback):
                if exc_type is None:
                    try:
                        self.manager.logger.info("Committing the transaction...")
                        self.conn.execute("COMMIT")
                    except sqlite3.Error as e:
                        self.manager.logger.error(f"Failed to commit transaction: {e}")
                        self.conn.execute("ROLLBACK")
                        raise
                else:
                    self.manager.logger.info("Rolling back the transaction due to an exception...")
                    self.conn.execute("ROLLBACK")

        return TransactionContext(self)

    def close(self):
        self.close_connection()