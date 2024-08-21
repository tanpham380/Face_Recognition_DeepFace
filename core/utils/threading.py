import threading
import queue
import time

from core.utils.database import SQLiteManager
from core.utils.logging import get_logger
from core.utils.static_variable import NUMBER_WORKER

# Create a queue for background tasks
task_queue = queue.Queue()
logger = get_logger()

threads = []

def add_task_to_queue(func, db_manager: SQLiteManager, *args, **kwargs):
    """Add a task to the queue with an ID."""
    task_id = f"task-{int(time.time())}"
    
    # Insert the task into the database with a "pending" status
    with db_manager.get_connection() as conn:
        conn.execute("""
            INSERT INTO task_status (task_id, status)
            VALUES (?, ?)
        """, (task_id, "pending"))
    
    # Add the task to the queue
    task_queue.put((task_id, lambda: func(db_manager, *args, **kwargs)))

    logger.info(f"Added task {task_id} to the queue")
    
    return task_id



def worker(app, db_manager: SQLiteManager):
    """Worker function to process tasks from the queue."""
    logger.info("Worker thread started")
    with app.app_context():  # Push the application context
        while True:
            logger.info("Worker waiting for a task")
            task_id, task = task_queue.get()  # Get a task from the queue
            if task is None:  # Check for a stop signal
                break
            logger.info(f"Worker received task {task_id}")
            try:
                with db_manager.get_connection() as conn:
                    conn.execute("""
                        UPDATE task_status
                        SET status = ?
                        WHERE task_id = ?
                    """, ("processing", task_id))
                
                logger.info(f"Worker picked up task {task_id}")
                task()  # Execute the task
            
                with db_manager.get_connection() as conn:
                    conn.execute("""
                        UPDATE task_status
                        SET status = ?
                        WHERE task_id = ?
                    """, ("completed", task_id))
                
                logger.info(f"Worker completed task {task_id}")
                

                
            except Exception as e:
                logger.error(f"Error in worker thread: {str(e)}")
                with db_manager.get_connection() as conn:
                    conn.execute("""
                        UPDATE task_status
                        SET status = ?
                        WHERE task_id = ?
                    """, ("failed", task_id))
            finally:
                task_queue.task_done()  # Mark the task as done

def start_workers(app, db_manager: SQLiteManager):
    """Start worker threads."""
    global threads
    for i in range(NUMBER_WORKER):
        t = threading.Thread(target=worker, args=(app, db_manager))
        t.start()
        threads.append(t)
        logger.info(f"Worker thread {i} started")

def stop_workers():
    """Send stop signal to all workers."""
    for _ in range(NUMBER_WORKER):
        task_queue.put((None, None))
    for t in threads:
        t.join()
