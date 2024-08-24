import atexit
import os
from flask import Flask, current_app
from core.deepface_controller.controller import DeepFaceController
from core.utils.logging import get_logger
from core.utils.static_variable import BASE_PATH, IMAGES_DIR
from core.utils.theading import stop_workers, start_workers
from core.routes import blueprint

logger = get_logger()
deepface_controller = DeepFaceController()


def preload_models():
    deepface_controller.find(
        img_path=os.path.join(BASE_PATH, "static", "temp.png"),
        db_path=IMAGES_DIR, 
        model_name="Facenet512",
        detector_backend="retinaface",
        anti_spoofing=True
    )


def create_app():
    app = Flask(__name__)
    with app.app_context():
        app.config['deepface_controller'] = deepface_controller
        preload_models()

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
