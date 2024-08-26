import atexit
import os
from flask import Flask, current_app
from core.deepface_controller.controller import DeepFaceController
from core.utils.database import ZoDB
from core.utils.logging import get_logger
from core.utils.monitor_folder_hash import check_and_update_directory_hash
from core.utils.static_variable import BASE_PATH, DB_PATH, IMAGES_DIR
from core.utils.threading import stop_workers, start_workers
from core.routes import blueprint
import os.path

logger = get_logger()
deepface_controller = DeepFaceController()

def create_app():
    app = Flask(__name__)
    
    # Create necessary directories
    directories = ["static", "static/images", "database"]
    base_paths = [os.path.join(BASE_PATH, d) for d in directories]
    for path in base_paths:
        os.makedirs(path, exist_ok=True)

    with app.app_context():
        app.config["deepface_controller"] = deepface_controller
        # Connection can be created when needed instead of at startup
        app.config["ZoDB"] = ZoDB(db_path=DB_PATH)
        app.config["ZoDB"].connect()
        logger.info("Application started successfully")

    for dir_name in os.listdir(IMAGES_DIR):
        dir_path = os.path.join(IMAGES_DIR, dir_name)
        if os.path.isdir(dir_path):
            check_and_update_directory_hash(dir_name, dir_path, app)

    # Start workers
    start_workers(app)

    # Cleanup function on exit
    def cleanup():
        logger.info("Cleaning up before shutdown...")
        stop_workers()
        app.config["ZoDB"].close()

    atexit.register(cleanup)

    app.register_blueprint(blueprint)

    return app

