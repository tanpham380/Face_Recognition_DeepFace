import atexit
import os
from flask import Flask, current_app
from core.deepface_controller.controller import DeepFaceController
from core.service import recreate_DB
from core.utils.logging import get_logger
from core.utils.static_variable import BASE_PATH, IMAGES_DIR
from core.utils.theading import stop_workers, start_workers
from core.routes import blueprint
from core.utils.theading import add_task_to_queue

logger = get_logger()
deepface_controller = DeepFaceController()





def create_app():
    app = Flask(__name__)
    with app.app_context():
        app.config['deepface_controller'] = deepface_controller        
        add_task_to_queue(recreate_DB, img_path=IMAGES_DIR, app=app, uid="tst")

    directories = ['static', 'static/images', 'database']
    base_paths = [os.path.join(BASE_PATH, d) for d in directories]
    for path in base_paths:
        os.makedirs(path, exist_ok=True)
    start_workers(app)

    def cleanup():
        logger.info("Cleaning up before shutdown...")
        stop_workers()

    atexit.register(cleanup)
    app.register_blueprint(blueprint)

    return app
