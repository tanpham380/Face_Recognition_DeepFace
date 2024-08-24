import glob
import flask
import os.path
import traceback
from typing import Optional, List, Dict, Any
import os
import numpy as np
from core.utils.augment_images import augment_image
from core.utils.theading import add_task_to_queue
from core.utils.file_utils import delete_images_for_uid, extract_base_identity, save_image
from core.utils.static_variable import BASE_PATH, IMAGES_DIR, TEMP_DIR
from core.utils.logging import get_logger

# Initialize the logger
logger = get_logger()


def get_deepface_controller():
    return flask.current_app.config['deepface_controller']


def list_users(uid_filter: str = None) -> List[Dict[str, Any]]:
    users = []
    base_dirs = [d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))]

    for base_dir in base_dirs:
        user_files = glob.glob(os.path.join(IMAGES_DIR, base_dir, '*.png'))
        unique_uids = set()

        for file_path in user_files:
            uid = extract_base_identity(file_path)
            if uid_filter:
                if uid_filter in uid:
                    unique_uids.add(uid)
            else:
                unique_uids.add(uid)

        for uid in unique_uids:
            users.append({
                "uid": uid,
                "base_dir": base_dir,
                "images": [os.path.basename(f) for f in user_files if uid in f]
            })
    if users:
        users.sort(key=lambda x: x['uid'])
        
    if users is None:
        return {"message": "No users found", "data": [], "success": False}
    return {"message": "Version fetched successfully", "data": users, "success": True}


def check_version() -> Dict[str, Any]:
    try:
        version = get_deepface_controller().check_version()
        return {"message": "Version fetched successfully", "data": version, "success": True}
    except Exception as e:
        logger.error(f"Exception while checking version: {str(e)} - {traceback.format_exc()}")
        return {"message": "Failed to fetch version", "data": None, "success": False}


def delete_face(uid: str, current_app) -> Dict[str, Any]:
    try:
        base_uid = uid.split('-')[0]
        delete_images_for_uid(uid, base_uid)
        logger.info(f"Deleted all face embeddings and images for UID {uid}")

        add_task_to_queue(recreate_DB, IMAGES_DIR, current_app._get_current_object(), uid=uid)
        logger.info(f"Final task scheduled for re-rendering vector database for UID {uid}")

        return {"message": "Face deleted successfully and final task scheduled!", "data": {"uid": uid}, "success": True}

    except Exception as e:
        logger.error(f"Exception while deleting face with UID {uid}: {str(e)}")
        return {"message": "Failed to delete face", "data": None, "success": False}

def recreate_DB(img_path, app, uid) -> Dict[str, Any]:
    logger.info(f"Starting recreate_DB with img_path: {img_path}, uid: {uid}")
    try:
        with app.app_context():
            get_deepface_controller().find(
                img_path=os.path.join(BASE_PATH, "static", "temp.png"),
                db_path=img_path,
                model_name="Facenet512",
                detector_backend="retinaface",
                anti_spoofing=False
            )
            logger.info("recreate_DB completed successfully.")
    except Exception as e:
        logger.error(f"Error in recreate_DB: {str(e)}")




def register_face(image: Any, uid: str, current_app) -> Dict[str, Any]:
    try:
        image_path, _ = save_image(image, uid, IMAGES_DIR, uid)
        augmented_images = augment_image(image)

        for i, img in enumerate(augmented_images, start=1):
            save_image(img, uid, IMAGES_DIR, f"{uid}_aug{i}")

        add_task_to_queue(recreate_DB, img_path=IMAGES_DIR, app=current_app._get_current_object(), uid=uid)

        return {"message": "Face registered successfully!.", "data": {"uid": uid}, "success": True}

    except Exception as e:
        logger.error(f"Exception while registering face with UID {uid}: {str(e)}")
        return {"message": "Failed to register face", "data": None, "success": False}

def recognize_face(image: Any) -> Dict[str, Any]:
    try:
        image_path, _ = save_image(image, "query_results", TEMP_DIR, "")

        value_objs_anti_spoofing = get_deepface_controller().extract_faces(
            img_path=image_path,
            detector_backend="retinaface",
            anti_spoofing=True
        )[0]

        value_objs = get_deepface_controller().find(
            img_path=image_path,
            db_path=IMAGES_DIR,
            model_name="Facenet512",
            detector_backend="retinaface",
            anti_spoofing=False
        )

        if value_objs and not value_objs[0].empty:
            best_match = value_objs[0].iloc[0]

            best_match_identity = extract_base_identity(os.path.splitext(os.path.basename(best_match['identity']))[0])
            best_match_confidence = round(float((1 - best_match['distance'] / best_match['threshold']) * 100), 2)
            response = {
                "message": "Face recognized successfully!",
                "data": {
                    "best_match": {
                        "identity": best_match_identity,
                        "confidence": best_match_confidence
                    },
                    "is_real": bool(value_objs_anti_spoofing.get('is_real', False)),
                    "antispoof_score": round(float(value_objs_anti_spoofing.get('antispoof_score', 0)), 2) * 100
                },
                "success": True
            }
            return response

        else:
            return {"message": "No faces detected in the image", "data": [], "success": False}

    except Exception as e:
        logger.error(f"Exception while recognizing face: {str(e)} - {traceback.format_exc()}")
        return {"message": "Failed to recognize face", "data": None, "success": False}
    finally:
        pass


def represent(
    img_path: str,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        embedding_objs = get_deepface_controller().represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        return {"message": "Representation fetched successfully", "data": embedding_objs, "success": True}
    except Exception as err:
        logger.error(f"Exception while representing: {str(err)} - {traceback.format_exc()}")
        return {"message": "Failed to represent image", "data": None, "success": False}


def verify(
    img1_path: str,
    img2_path: str,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
) -> Dict[str, Any]:
    try:
        result = get_deepface_controller().verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return {"message": "Verification successful", "data": result, "success": True}
    except Exception as err:
        logger.error(f"Exception while verifying: {str(err)} - {traceback.format_exc()}")
        return {"message": "Failed to verify images", "data": None, "success": False}


def analyze(
    img_path: str,
    actions: List[str],
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
) -> Dict[str, Any]:
    try:
        demographies = get_deepface_controller().analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
            anti_spoofing=anti_spoofing,
        )
        return {"message": "Analysis successful", "data": demographies, "success": True}
    except Exception as err:
        logger.error(f"Exception while analyzing: {str(err)} - {traceback.format_exc()}")
        return {"message": "Failed to analyze image", "data": None, "success": False}
