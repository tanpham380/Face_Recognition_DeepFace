import atexit
import os
from flask import Flask, current_app
from core.deepface_controller.controller import DeepFaceController
from core.service import recreate_DB
from core.utils.logging import get_logger
from core.utils.static_variable import BASE_PATH, IMAGES_DIR
from core.utils.theading import stop_workers, start_workers, add_task_to_queue
from core.routes import blueprint

logger = get_logger()
deepface_controller = DeepFaceController()

def has_directory_changed(directory_path, previous_stat=None):
    current_stat = os.stat(directory_path)
    return previous_stat is None or current_stat.st_mtime != previous_stat.st_mtime, current_stat


def create_app():
    app = Flask(__name__)
    with app.app_context():
        app.config['deepface_controller'] = deepface_controller        
    
    # Create necessary directories
    directories = ['static', 'static/images', 'database']
    base_paths = [os.path.join(BASE_PATH, d) for d in directories]
    for path in base_paths:
        os.makedirs(path, exist_ok=True)

    # Monitor changes in all subdirectories within IMAGES_DIR
    previous_stats = {}
    for dir_name in os.listdir(IMAGES_DIR):
        dir_path = os.path.join(IMAGES_DIR, dir_name)
        if os.path.isdir(dir_path):
            changed, previous_stat = has_directory_changed(dir_path, previous_stats.get(dir_name))
            if changed:
                logger.info(f"Directory {dir_name} has changed. Adding task to queue.")
                add_task_to_queue(recreate_DB, img_path=IMAGES_DIR, app=app, uid=dir_name)
            previous_stats[dir_name] = previous_stat

    # Start workers
    start_workers(app)

    # # Cleanup function on exit
    # def cleanup():
    #     logger.info("Cleaning up before shutdown...")
    #     stop_workers()

    atexit.register(lambda: stop_workers() or logger.info("Cleaning up before shutdown..."))
    app.register_blueprint(blueprint)

    return app
