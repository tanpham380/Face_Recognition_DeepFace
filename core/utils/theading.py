import threading
import queue
import time
import uuid

from core.utils.database import SQLiteManager
from core.utils.logging import get_logger
from core.utils.static_variable import NUMBER_WORKER

task_queue = queue.Queue()
logger = get_logger()
threads = []

def generate_unique_task_id(db_manager: SQLiteManager) -> str:
    while True:
        task_id = f"task-{int(time.time())}-{uuid.uuid4().hex}"
        with db_manager.get_connection() as conn:
            result = conn.execute("""
                SELECT task_id FROM task_status WHERE task_id = ?
            """, (task_id,)).fetchone()
        if result is None:
            return task_id

def add_task_to_queue(func, db_manager: SQLiteManager, *args, **kwargs):
    task_id = generate_unique_task_id(db_manager)
    uid = kwargs.get('uid', None) 
    logger.info(f"Generated unique task ID: {task_id}")
    from core.service import check_and_run_final_task

    db_manager.insert_task(task_id, uid, "pending", is_final_task=(func == check_and_run_final_task))
    
    task_queue.put((task_id, lambda: func(db_manager, *args, **kwargs)))
    logger.info(f"Added task {task_id} to the queue")

def worker(app, db_manager: SQLiteManager):
    logger.info("Worker thread started")
    with app.app_context():
        while True:
            logger.info("Worker waiting for a task")
            task_id, task = task_queue.get()
            if task is None:
                break
            logger.info(f"Worker received task {task_id}")
            try:
                db_manager.update_task_status(task_id, "processing")

                logger.info(f"Worker picked up task {task_id}")
                task()  

                db_manager.update_task_status(task_id, "completed")

                logger.info(f"Worker completed task {task_id}")

            except Exception as e:
                logger.error(f"Error in worker thread: {str(e)}")
                db_manager.update_task_status(task_id, "failed")
            finally:
                task_queue.task_done()

def start_workers(app, db_manager: SQLiteManager):
    global threads
    for i in range(NUMBER_WORKER):
        t = threading.Thread(target=worker, args=(app, db_manager), name=f"Worker-{i}")
        t.start()
        threads.append(t)
        logger.info(f"Worker thread {i} started")

def stop_workers():
    for _ in range(NUMBER_WORKER):
        task_queue.put((None, None))
    for t in threads:
        t.join()
