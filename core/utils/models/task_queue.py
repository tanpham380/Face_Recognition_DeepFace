import time
from persistent import Persistent

class TaskQueue(Persistent):
    def __init__(self, task_id: str, status: str, created_at: float):
        self.task_id = task_id
        self.status = status
        self.created_at = created_at
        self.updated_at = None

    def update_status(self, new_status: str):
        self.status = new_status
        self.updated_at =time.strftime("%d/%m/%Y %H:%M", time.localtime())
        
    def to_dict(self):
        return {
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }