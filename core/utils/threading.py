import threading
import queue
import time
import uuid
from core.utils.logging import get_logger
from core.utils.static_variable import NUMBER_WORKER
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=NUMBER_WORKER)

task_queue = queue.Queue(maxsize=100)  # Bounded queue to prevent overflow
logger = get_logger()
threads = []
queue_lock = threading.Lock()

def generate_unique_task_id() -> str:
    return f"task-{int(time.time())}-{uuid.uuid4().hex}"

def add_task_to_queue(func, app, *args, **kwargs):
    task_id = generate_unique_task_id()

    with queue_lock:
        if task_queue.full():
            logger.warning(f"Task queue is full, waiting to add task {task_id}")

        # Add task to the database
        app.config["ZoDB"].add_task(task_id, "queued", time.time())

        # Pass `app` as an additional argument to the task
        task_queue.put((task_id, lambda: func(app, *args, **kwargs)))
        logger.info(f"Added task {task_id} to the queue")

def worker(app):
    logger.info("Worker thread started")
    with app.app_context():
        while True:
            task_id, task = task_queue.get()
            if task is None:
                break
            start_time = time.time()

            try:
                # Update task status to 'running'
                app.config["ZoDB"].update_task_status(task_id, "running")
                logger.info(f"Worker picked up task {task_id}")

                future = executor.submit(task)
                result = future.result()  # This will block until the task is done

                # Update task status to 'completed'
                app.config["ZoDB"].update_task_status(task_id, "completed")
                logger.info(
                    f"Worker completed task {task_id} in {time.time() - start_time:.2f} seconds"
                )
            except Exception as e:
                # Update task status to 'failed'
                app.config["ZoDB"].update_task_status(task_id, "failed")
                logger.error(f"Error in worker thread: {str(e)}")
            finally:
                task_queue.task_done()

def start_workers(app):
    global threads
    if threads:  # Check if workers are already started
        logger.warning("Workers already started.")
        return

    for i in range(NUMBER_WORKER):
        t = threading.Thread(target=worker, args=(app,), name=f"Worker-{i}")
        t.start()
        threads.append(t)
        logger.info(f"Worker thread {i} started")

def stop_workers():
    global threads
    for _ in range(NUMBER_WORKER):
        task_queue.put((None, None))
    for t in threads:
        t.join()

    # Cleanup
    threads = []
    logger.info("All workers stopped")
