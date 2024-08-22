from flask import Blueprint, current_app, request
from core import service
from core.utils.middleware import require_api_key
from core.utils.logging import get_logger
logger = get_logger()
blueprint = Blueprint("routes", __name__)

# Assuming you have an instance of DeepFaceController
@blueprint.route("/")
def home():
    version = service.check_version()
    return version


@blueprint.route("/information", methods=["GET"])
@require_api_key
def version():
    db_manager = current_app.config['DB_MANAGER']
    embeddings = db_manager.is_database_locked()
    return {"information": embeddings}

@blueprint.route("/tasks/status", methods=["GET"])
@require_api_key
def get_task_status():
    
    """API để xem trạng thái các tác vụ."""
    try :
        db_manager = current_app.config['DB_MANAGER']
        with db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT task_id, status, created_at
                FROM task_status
                ORDER BY created_at DESC
            """)
            tasks = cursor.fetchall()
            tasks_list = [{"task_id": row[0], "status": row[1], "created_at": row[2]} for row in tasks]
            return {"message": "Thành công", "data": tasks_list , "success": True }
    except Exception as e:
        return {"message": "Failed to get task status", "data": None, "success": False}, 500




@blueprint.route("/list", methods=["GET"])
@require_api_key
def list_faces():
    try: 
        db_manager = current_app.config['DB_MANAGER']
        uids = db_manager.get_all_uids()

        # Use a set to remove duplicates and then convert it back to a list
        unique_uids = list(set(uids))
        return {"message": "Thành công", "data": unique_uids , "success": True }
    except Exception as e:
        return {"message": "Failed to list faces", "data": None, "success": False}, 500

    



@blueprint.route("/embedding", methods=["GET"])
@require_api_key
def get_embedding_by_uid():
    uid = request.args.get("uid")
    if not uid:
        return {"message": "UID is required"}, 400

    db_manager = current_app.config['DB_MANAGER']
    embedding = db_manager.get_embedding_by_uid(uid)

    if embedding:
        return {"embedding": embedding}
    else:
        return {"message": "UID not found"}, 404


@blueprint.route("/recent_embeddings", methods=["GET"])
@require_api_key
def get_most_recent_embeddings():
    limit = request.args.get("limit", default=10, type=int)
    db_manager = current_app.config['DB_MANAGER']
    embeddings = db_manager.get_most_recent_embeddings(limit=limit)
    return {"embeddings": embeddings}


@blueprint.route("/embeddings_by_range", methods=["GET"])
@require_api_key
def get_embeddings_by_id_range():
    start_id = request.args.get("start_id", type=int)
    end_id = request.args.get("end_id", type=int)

    if start_id is None or end_id is None:
        return {"message": "start_id and end_id are required"}, 400

    db_manager = current_app.config['DB_MANAGER']
    embeddings = db_manager.get_embeddings_by_id_range(
        start_id=start_id, end_id=end_id)
    return {"embeddings": embeddings}


@blueprint.route("/register", methods=["POST"])
@require_api_key
def register():
    uid = request.form.get("uid")
    if not uid:
        return {"message": "UID is required"}, 400

    if 'image' not in request.files:
        return {"message": "Image is required"}, 400

    image = request.files["image"]
    db_manager = current_app.config['DB_MANAGER']

    try:
        response = service.register_face(image, uid, db_manager , current_app)
        return response
    except Exception as e:
        return {"message": "Failed to register face", "error": str(e)}, 500


@blueprint.route("/delete", methods=["POST"])
@require_api_key
def delete_face():
    uid = request.form.get("uid")
    if not uid:
        return {"message": "UID is required"}, 400

    db_manager = current_app.config['DB_MANAGER']

    try:
        response = service.delete_face(uid, db_manager , current_app)
        return response
    except Exception as e:
        return {"message": "Failed to delete face", "error": str(e)}, 500


@blueprint.route("/recognize_db", methods=["POST"])
@require_api_key
def recognize_db():
    if 'image' not in request.files:
        return {"message": "Image is required"}, 400

    image = request.files["image"]
    if image is None:
        return {"message": "Image is required"}, 400
    db_manager = current_app.config['DB_MANAGER']
    try:
        response = service.recognize_face_with_database(image, db_manager)
        return response
    except Exception as e:
        return {"message": "Failed to recognize face", "error": str(e)}, 500

@blueprint.route("/recognize", methods=["POST"])
@require_api_key
def recognize():
    if 'image' not in request.files:
        return {"message": "Image is required"}, 400

    image = request.files["image"]
    try:
        response = service.recognize_face(image)
        return response
    except Exception as e:
        return {"message": "Failed to recognize face", "error": str(e)}, 500

@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    obj = service.represent(
        img_path=img_path,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
        max_faces=input_args.get("max_faces"),
    )

    logger.debug(obj)

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1") or input_args.get("img1_path")
    img2_path = input_args.get("img2") or input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

    verification = service.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        distance_metric=input_args.get("distance_metric", "cosine"),
        align=input_args.get("align", True),
        enforce_detection=input_args.get("enforce_detection", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(verification)

    return verification


@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    demographies = service.analyze(
        img_path=img_path,
        actions=input_args.get("actions", ["age", "gender", "emotion", "race"]),
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(demographies)

    return demographies