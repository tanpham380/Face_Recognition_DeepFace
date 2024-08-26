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



@blueprint.route("/users", methods=["GET"])
@require_api_key
def get_users():
    uid_filter = request.args.get("uid")  # Get UID from query parameter, if provided
    try:
        users = service.list_users(uid_filter)
        return users
    except Exception as e:
        logger.error(f"Failed to retrieve user list: {str(e)}")
        return {"message": "Failed to retrieve user list", "data": None, "success": False}, 500



@blueprint.route("/register", methods=["POST"])
@require_api_key
def register():
    uid = request.form.get("uid")
    if not uid:
        return {"message": "UID is required", "data": None, "success": False}, 400

    if 'image' not in request.files:
        return {"message": "Image is required", "data": None, "success": False}, 400

    image = request.files["image"]

    try:
        response = service.register_face(image, uid, current_app)
        return response
    except Exception as e:
        logger.error(f"Failed to register face: {str(e)}")
        return {"message": "Failed to register face", "data": None, "success": False}, 500


@blueprint.route("/delete", methods=["POST"])
@require_api_key
def delete_face():
    uid = request.form.get("uid")
    if not uid:
        return {"message": "UID is required", "data": None, "success": False}, 400


    try:
        response = service.delete_face(uid, current_app)
        return response
    except Exception as e:
        logger.error(f"Failed to delete face: {str(e)}")
        return {"message": "Failed to delete face", "data": None, "success": False}, 500



@blueprint.route("/recognize", methods=["POST"])
@require_api_key
def recognize():
    if 'image' not in request.files:
        return {"message": "Image is required", "data": None, "success": False}, 400
    uid = request.form.get("uid")
    
    image = request.files["image"]
    try:
        response = service.recognize_face(image , uid)
        return response
    except Exception as e:
        logger.error(f"Failed to recognize face: {str(e)}")
        return {"message": "Failed to recognize face", "data": None, "success": False}, 500


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "Empty input set passed", "data": None, "success": False}, 400

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "You must pass img_path input", "data": None, "success": False}, 400

    try:
        obj = service.represent(
            img_path=img_path,
            model_name=input_args.get("model_name", "VGG-Face"),
            detector_backend=input_args.get("detector_backend", "opencv"),
            enforce_detection=input_args.get("enforce_detection", True),
            align=input_args.get("align", True),
            anti_spoofing=input_args.get("anti_spoofing", False),
            max_faces=input_args.get("max_faces"),
        )
        return {"message": "Representation successful", "data": obj, "success": True}
    except Exception as e:
        logger.error(f"Failed to represent face: {str(e)}")
        return {"message": "Failed to represent face", "data": None, "success": False}, 500


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "Empty input set passed", "data": None, "success": False}, 400

    img1_path = input_args.get("img1") or input_args.get("img1_path")
    img2_path = input_args.get("img2") or input_args.get("img2_path")

    if img1_path is None:
        return {"message": "You must pass img1_path input", "data": None, "success": False}, 400

    if img2_path is None:
        return {"message": "You must pass img2_path input", "data": None, "success": False}, 400

    try:
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
        return {"message": "Verification successful", "data": verification, "success": True}
    except Exception as e:
        logger.error(f"Failed to verify faces: {str(e)}")
        return {"message": "Failed to verify faces", "data": None, "success": False}, 500

@blueprint.route("/analyze", methods=["POST"])
def analyze():
    try: 
        input_args = request.get_json()

        if input_args is None:
            return {"message": "Empty input set passed"}

        img_path = input_args.get("img") or input_args.get("img_path")
        if img_path is None:
            return {"message": "You must pass img_path input"}

        demographies = service.analyze(
            img_path=img_path,
            actions=input_args.get("actions", ["age", "gender", "emotion", "race"]),
            detector_backend=input_args.get("detector_backend", "opencv"),
            enforce_detection=input_args.get("enforce_detection", True),
            align=input_args.get("align", True),
            anti_spoofing=input_args.get("anti_spoofing", False),
        )

        logger.debug(demographies)

        return {"message": "Analysis successful", "data": demographies , "success": True}
    except Exception as e:
        logger.error(f"Failed to analyze face: {str(e)}")
        return {"message": "Failed to analyze face", "data": None, "success": False}, 500
