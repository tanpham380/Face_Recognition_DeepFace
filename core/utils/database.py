from flask import g
import sqlite3
import json
from typing import List, Dict, Any


class SQLiteManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_connection(self):
        if 'db_connection' not in g:
            g.db_connection = sqlite3.connect(self.db_path)
        return g.db_connection
    def close_connection(self, error):
        if 'db_connection' in g:
            g.db_connection.close()
            g.pop('db_connection', None)
            
            
            
            
    def create_table(self):
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid TEXT UNIQUE,
                    embedding TEXT
                )
            """)
    def create_task_status_table(self):
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    def get_embedding_by_uid(self, uid: str) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT uid, embedding FROM face_embeddings WHERE uid = ?", (uid,))
            row = cursor.fetchone()
            if row:
                return {"uid": row[0], "embedding": json.loads(row[1])}
            return None

    # def get_embeddings_by_uids(self, uids: List[str]) -> List[Dict[str, Any]]:
    #     with self.get_connection() as conn:
    #         cursor = conn.execute(f"SELECT uid, embedding FROM face_embeddings WHERE uid IN ({
    #                                          ','.join('?' for _ in uids)})", uids)
    #         rows = cursor.fetchall()
    #         return [{"uid": uid, "embedding": json.loads(embedding)} for uid, embedding in rows]
    def get_embeddings(self, uid: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, embedding FROM face_embeddings WHERE uid = ? ORDER BY id ASC", (uid,))
            rows = cursor.fetchall()
            return [{"id": row[0], "embedding": json.loads(row[1])} for row in rows]
    def delete_oldest_embeddings(self, uid: str, n: int):
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM face_embeddings WHERE uid = ? ORDER BY id ASC LIMIT ?", (uid, n))
            rows = cursor.fetchall()
            
            if rows:
                ids_to_delete = [row[0] for row in rows]
                conn.executemany(
                    "DELETE FROM face_embeddings WHERE id = ?", [(id,) for id in ids_to_delete]
                )

    def get_embeddings_with_condition(self, condition: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                f"SELECT uid, embedding FROM face_embeddings WHERE {condition}", params)
            rows = cursor.fetchall()
            return [{"uid": uid, "embedding": json.loads(embedding)} for uid, embedding in rows]

    def get_count_by_uid_prefix(self, prefix: str) -> int:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM face_embeddings WHERE uid LIKE ?", (f"{prefix}%",))
            return cursor.fetchone()[0]

    def get_all_uids(self) -> List[str]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT uid FROM face_embeddings")
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def get_count_of_embeddings(self) -> int:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM face_embeddings")
            return cursor.fetchone()[0]

    def insert_or_update_embedding(self, uid: str, embedding: List[float]):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO face_embeddings (uid, embedding)
                VALUES (?, ?)
            """, (uid, json.dumps(embedding)))

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT uid, embedding FROM face_embeddings")
            rows = cursor.fetchall()
            return [{"uid": uid, "embedding": json.loads(embedding)} for uid, embedding in rows]

    def delete_embedding_by_uid(self, uid: str) -> None:
        with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM face_embeddings WHERE uid = ?", (uid,))

    def update_embedding(self, uid: str, new_embedding: List[float]) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE face_embeddings
                SET embedding = ?
                WHERE uid = ?
            """, (json.dumps(new_embedding), uid))

    def uid_exists(self, uid: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM face_embeddings WHERE uid = ? LIMIT 1", (uid,))
            return cursor.fetchone() is not None

    def drop_table(self):
        with self.get_connection() as conn:
            conn.execute("DROP TABLE IF EXISTS face_embeddings")

    def execute_raw_sql(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict((cursor.description[i][0], value) for i, value in enumerate(row)) for row in rows]
        
        
    def fetch_similar_embeddings(self, target_embedding: List[float], threshold: float) -> List[Dict[str, Any]]:
        from scipy.spatial.distance import cosine

        def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
            return 1 - cosine(embedding1, embedding2)

        with self.get_connection() as conn:
            cursor = conn.execute("SELECT uid, embedding FROM face_embeddings")
            rows = cursor.fetchall()
            similar_embeddings = [
                {
                    "uid": uid,
                    "confidence": calculate_similarity(target_embedding, json.loads(embedding))
                }
                for uid, embedding in rows
                if calculate_similarity(target_embedding, json.loads(embedding)) >= threshold
            ]
            return similar_embeddings


    def begin_transaction(self):
        with self.get_connection() as conn:
            conn.execute("BEGIN TRANSACTION")

    def commit_transaction(self):
        with self.get_connection() as conn:
            conn.execute("COMMIT")

    def rollback_transaction(self):
        with self.get_connection() as conn:
            conn.execute("ROLLBACK")

    def close(self):
        pass
