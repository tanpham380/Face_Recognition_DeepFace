import atexit
import os
from flask import Flask, current_app
from core.deepface_controller.controller import DeepFaceController
from core.utils.database import SQLiteManager
from core.utils.logging import get_logger
from core.utils.static_variable import BASE_PATH, DB_PATH, IMAGES_DIR
from core.utils.theading import stop_workers, start_workers
from core.routes import blueprint

logger = get_logger()
deepface_controller = DeepFaceController()


def preload_models(controller):
    # Load the models by calling a dummy detection
    controller.find(
        img_path=os.path.join(BASE_PATH, "static", "temp.png"), # Pass a dummy or default image path
        db_path=IMAGES_DIR,  # Pass a dummy or default db_manager
        model_name="Facenet512",
        detector_backend="retinaface",
        anti_spoofing=True
    )
    
def create_app():
    app = Flask(__name__)

    # Initialize SQLite manager
    
    
    
    with app.app_context():
        db_manager = SQLiteManager(DB_PATH)
        db_manager.optimize_sqlite()
        db_manager.create_table()
        db_manager.create_task_status_table()
        app.config['deepface_controller'] = deepface_controller
        preload_models(deepface_controller)

    app.config['DB_MANAGER'] = db_manager
    logger.info(f"Starting Flask app: {app.name}")
    # Create required directories
    directories = ['static', 'static/images', 'database']
    base_paths = [os.path.join(BASE_PATH, d) for d in directories]
    for path in base_paths:
        os.makedirs(path, exist_ok=True)

    # Start worker threads after the DB manager is initialized
    start_workers(app, db_manager)

    # Register cleanup tasks to be performed on application shutdown
    def cleanup():
        logger.info("Cleaning up before shutdown...")
        stop_workers()
        with db_manager.get_connection() as conn:
            conn.execute("DELETE FROM task_status")
            logger.info("Cleared task_status table.")
    
    atexit.register(cleanup)

    # Register blueprint
    app.register_blueprint(blueprint) #Register your blueprints here

    return app

