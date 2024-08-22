
import flask
import os.path
import time
import traceback
from typing import Optional, List, Dict, Any
import os
import numpy as np
from core.utils.augment_images import augment_image
from core.utils.theading import add_task_to_queue, task_queue
from core.utils.database import SQLiteManager
from core.utils.progress_file import delete_images_for_uid, extract_base_identity, save_image
# from core.utils.search_embleeding import find_in_db
from core.utils.static_variable import BASE_PATH, IMAGES_DIR, TEMP_DIR
from core.utils.logging import get_logger



# Initialize the logger
logger = get_logger()
def get_deepface_controller():
    return flask.current_app.config['deepface_controller']


def check_version() -> Dict[str, Any]:
    try:
        version = get_deepface_controller().check_version()
        return {"message": "Version fetched successfully", "data": version, "success": True}
    except Exception as e:
        logger.error(f"Exception while checking version: {str(e)} - {traceback.format_exc()}")
        return {"message": "Failed to fetch version", "data": None, "success": False}

def delete_face(uid: str, db_manager: SQLiteManager, current_app) -> Dict[str, Any]:
    try:
        # Cancel any pending or processing tasks related to this UID
        task_id = db_manager.get_active_final_task(uid)
        if task_id:
            db_manager.cancel_task(task_id)
            logger.info(f"Canceled pending or processing task {task_id} for UID {uid}")

        # Extract base_uid from the given uid
        base_uid = uid.split('-')[0]

        # Delete all embeddings associated with this UID
        db_manager.delete_embedding_by_uid(uid)

        # Delete images associated with the base_uid
        delete_images_for_uid(uid, base_uid)
        logger.info(f"Deleted all face embeddings and images for UID {uid}")

        # Schedule the final task to re-render the database vector using DeepFace
        add_task_to_queue(recreate_DB, db_manager, IMAGES_DIR, current_app._get_current_object(), uid=uid)
        logger.info(f"Final task scheduled for re-rendering vector database for UID {uid}")

        return {"message": "Face deleted successfully and final task scheduled!", "data": {"uid": uid}, "success": True}

    except Exception as e:
        logger.error(f"Exception while deleting face with UID {uid}: {str(e)} - {traceback.format_exc()}")
        return {"message": "Failed to delete face", "data": None, "success": False}

def recreate_DB(db_manager: SQLiteManager, img_path, app, uid) -> Dict[str, Any]:
    with app.app_context():
        logger.info(f"Recreating DB for UID {uid} using image path {img_path}")
        get_deepface_controller().find(
            img_path=os.path.join(BASE_PATH, "static", "temp.png"),
            db_path=img_path,
            model_name="Facenet512",
            detector_backend="retinaface",
            anti_spoofing=False
        )
        logger.info(f"Completed recreating DB for UID {uid}")

def check_and_run_final_task(db_manager: SQLiteManager, img_path, app, uid):
    logger.info(f"Running final task for UID {uid}")
    
    pending_task_id = db_manager.get_active_final_task(uid)
    
    if pending_task_id:
        logger.info(f"Cannot run final task for UID {uid}, other task {pending_task_id} is still pending.")
        return
    
    if task_queue.empty():
        with app.app_context():
            get_deepface_controller().find(
                img_path=img_path,
                db_path=IMAGES_DIR,
                model_name="Facenet512",
                detector_backend="retinaface",
                anti_spoofing=False
            )
    else:
        logger.info(f"New tasks are still in the queue, skipping final task execution for UID {uid}.")

def handle_background_tasks(db_manager: SQLiteManager, image_paths, uid, app, max_embeddings=60):
    logger.info(f"Running background processing for UID {uid}")
    with app.app_context():
        try:
            for img_path in image_paths:
                logger.info(f"Processing image: {img_path}")
                
                time.sleep(5)
                if not task_queue.empty():
                    logger.info("New request detected, stopping current task and waiting for final task.")
                    return
                
                represent_objs = get_deepface_controller().represent(
                    img_path=img_path,
                    model_name="Facenet512",
                    detector_backend="retinaface",
                    enforce_detection=True,
                    align=True,
                    anti_spoofing=False
                )

                if represent_objs:
                    embedding = np.array(represent_objs[0]["embedding"], dtype="float32").tolist()
                    existing_embeddings = db_manager.get_embeddings(uid)

                    if len(existing_embeddings) >= max_embeddings:
                        db_manager.delete_oldest_embeddings(uid, len(existing_embeddings) - max_embeddings + 1)

                    db_manager.insert_or_update_embedding(uid, embedding)
                    logger.info(f"Face registered successfully for UID {uid}")
                else:
                    logger.warning(f"Could not detect any face in image {img_path}")

        except Exception as e:
            logger.error(f"Exception during background processing for UID {uid}: {str(e)} - {traceback.format_exc()}")

def register_face(image: Any, uid: str, db_manager: SQLiteManager, current_app) -> Dict[str, Any]:
    try:
        image_path, _ = save_image(image, uid, IMAGES_DIR, uid)
        augmented_images = augment_image(image)

        augmented_image_paths = []
        for i, img in enumerate(augmented_images, start=1):
            augmented_image_path, _ = save_image(img, uid, IMAGES_DIR, f"{uid}_aug{i}")
            augmented_image_paths.append(augmented_image_path)

        add_task_to_queue(handle_background_tasks, db_manager, [image_path] + augmented_image_paths, uid, current_app._get_current_object())
        add_task_to_queue(check_and_run_final_task, db_manager, img_path=image_path, app=current_app._get_current_object(), uid=uid)

        return {"message": "Face registered successfully!.", "data": {"uid": uid}, "success": True}

    except Exception as e:
        logger.error(f"Exception while registering face with UID {uid}: {str(e)} - {traceback.format_exc()}")
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

def recognize_face_with_database(image: Any, db_manager: SQLiteManager) -> Dict[str, Any]:
    try:
        image_path, _  = save_image(image, "query_results", TEMP_DIR, "")

        value_objs = get_deepface_controller().verify_faces_db(
            img_path=image_path,
            db_manager=db_manager,
            model_name="Facenet512",
            detector_backend="retinaface",
            anti_spoofing=True
        )

        if value_objs and 'matches' in value_objs[0] and value_objs[0]['matches']:
            threshold = value_objs[0].get('threshold', None)
            if threshold is None:
                raise KeyError("Threshold is missing in the response")

            best_match = value_objs[0]['matches'][0]

            confidence = (1 - best_match['distance'] / threshold) * 100
            confidence = round(confidence, 2)

            response = {
                "message": "Face recognized successfully!",
                "data": {
                    "best_match": {
                        "identity": best_match['identity'],
                        "confidence": round(confidence, 2)
                    },
                    "is_real": value_objs[0]['is_real'],
                    "antispoof_score": round(value_objs[0]['antispoof_score'], 2) * 100
                },
                "success": True
            }
            return response

        else:
            return {"message": "No faces detected in the image or no matches found", "data": [], "success": False}

    except KeyError as ke:
        logger.error(f"KeyError while recognizing face: {str(ke)} - {traceback.format_exc()}")
        return {"message": "Failed to recognize face due to missing data", "data": None, "success": False}
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
