import glob
import flask
import os.path
import traceback
from typing import Optional, List, Dict, Any
import os
import numpy as np
from core.utils.augment_images import augment_image
from core.utils.threading import add_task_to_queue
from core.utils.images_handler import (
    delete_directory_if_empty,
    delete_images_for_uid,
    extract_base_identity,
    save_image,
)
from core.utils.static_variable import BASE_PATH, IMAGES_DIR, TEMP_DIR
from core.utils.logging import get_logger


# Initialize the logger
logger = get_logger()


def get_deepface_controller():
    return flask.current_app.config["deepface_controller"]


def hash_directory_data(app) -> Dict[str, Any]:
    try:
        hash_data = app.config["ZoDB"].list_all_directory_hashes()
        return {
            "message": "Hash data fetched successfully",
            "data": hash_data,
            "success": True,
        }
    except Exception as e:
        logger.error(
            f"Exception while fetching hash data: {e} - {traceback.format_exc()}"
        )
        return {"message": "Failed to fetch hash data", "data": None, "success": False}
def get_Face_embedding(uid: str, current_app) -> Dict[str, Any]:
    try:
        face_User = current_app.config["ZoDB"].get_face_data(uid)
        if face_User:
            data = face_User.to_dict()
        else:
            data = None
        
        return {
            "message": "Face embedding fetched successfully",
            "data": data,
            "success": True,
        }
    except Exception as e:
        logger.error(
            f"Exception while fetching face embedding: {e} - {traceback.format_exc()}"
        )
        return {"message": "Failed to fetch face embedding", "data": None, "success": False}
def get_all_faces(current_app) -> Dict[str, Any]:
    try:
        # Fetch all face data
        all_face_data = current_app.config["ZoDB"].list_face_data()
        return {
            "message": "All face data fetched successfully",
            "data": all_face_data,
            "success": True,
        }
    except Exception as e:
        logger.error(
            f"Exception while fetching all face data: {e} - {traceback.format_exc()}"
        )
        return {"message": "Failed to fetch all face data", "data": None, "success": False}

def get_status_task(task_id: Optional[str] = None, app=None) -> Dict[str, Any]:
    try:
        if task_id:
            # Retrieve specific task by task_id
            task = app.config["ZoDB"].get_task(task_id)
            if task:
                data = {
                    "task_id": task.task_id,
                    "status": task.status,
                    "created_at": task.created_at,
                    "updated_at": task.updated_at,
                }
            else:
                return {"message": "Task not found", "success": False, "data": None}
        else:
            # Retrieve and return all tasks with full details
            tasks = app.config["ZoDB"].list_all_tasks()
            
            if tasks:
                data = [
                    {
                        "task_id": task_id,
                        "status": task.status,
                        "created_at": task.created_at,
                        "updated_at": task.updated_at,
                    }
                    for task_id, task in tasks.items()
                ]
            else:
                return {"message": "No tasks found", "success": False, "data": None}

        return {
            "message": "Tasks fetched successfully" if data else "No tasks found",
            "success": True if data else False,
            "data": data,
        }
    except Exception as e:
        logger.error(f"Exception while fetching tasks: {e} - {traceback.format_exc()}")
        return {"message": "Task not found", "success": False, "data": None}

def list_users(
    uid_filter: Optional[str] = None, app: Optional[Any] = None
) -> Dict[str, Any]:
    users = []
    for base_dir in filter(
        lambda d: os.path.isdir(os.path.join(IMAGES_DIR, d)), os.listdir(IMAGES_DIR)
    ):
        user_files = glob.glob(os.path.join(IMAGES_DIR, base_dir, "*.png"))
        unique_uids = {
            extract_base_identity(f)
            for f in user_files
            if not uid_filter or uid_filter in extract_base_identity(f)
        }
        directory_hash = app.config["ZoDB"].get_directory_hash(base_dir)
        hash_value = directory_hash.hash_value
        for uid in unique_uids:
            users.append(
                {
                    "uid": uid,
                    "base_dir": base_dir + "-" + hash_value,
                    "images": [os.path.basename(f) for f in user_files if uid in f],
                }
            )

    return {
        "message": "Users fetched successfully" if users else "No users found",
        "data": users,
        "success": bool(users),
    }


def check_version() -> Dict[str, Any]:
    try:
        version = get_deepface_controller().check_version()
        return {
            "message": "Version fetched successfully",
            "data": version,
            "success": True,
        }
    except Exception as e:
        logger.error(
            f"Exception while checking version: {e} - {traceback.format_exc()}"
        )
        return {"message": "Failed to fetch version", "data": None, "success": False}


# def recreate_DB(img_path: str, app: Any, uid: str) -> None:
#     logger.info(f"Starting recreate_DB with img_path: {img_path}, uid: {uid}")
#     from core.utils.monitor_folder_hash import (
#         hash_directory,
#     )

#     try:
#         with app.app_context():
#             base_uid = uid.split("-")[0]
#             save_dir = os.path.join(img_path, base_uid)
#             get_deepface_controller().find(
#                 img_path=os.path.join(BASE_PATH, "static", "temp.png"),
#                 db_path=save_dir,
#                 model_name="Facenet512",
#                 detector_backend="retinaface",
#                 anti_spoofing=False,
#             )
#             current_hash = hash_directory(save_dir)
#             logger.info(
#                 f"Setting initial hash for directory {save_dir} with current hash {current_hash}."
#             )
#             app.config["ZoDB"].set_directory_hash(base_uid, current_hash)
#             logger.info("recreate_DB completed successfully.")
#     except Exception as e:
#         logger.error(f"Error in recreate_DB: {e} - {traceback.format_exc()}")

def import_db(app: Any, img_path: str, uid: str) -> Dict[str, Any]:
    try:
        with app.app_context():
            controller = get_deepface_controller()
            embedding_objs = controller.represent(
                img_path=img_path,
                model_name="Facenet512",
                detector_backend="retinaface",
                anti_spoofing=False,
            )

            if embedding_objs:
                embedding = embedding_objs[0].get("embedding")
                if embedding:
                    
                    
                    app.config["ZoDB"].add_face_embedding(uid, img_path, embedding)
                    
    except Exception as e:
        logger.error(f"Error in import_db: {e} - {traceback.format_exc()}")



def recreate_DB(app: Any, img_path: str, uid: str) -> None:
    logger.info(f"Starting recreate_DB with img_path: {img_path}, uid: {uid}")
    from core.utils.monitor_folder_hash import hash_directory

    try:
        with app.app_context():
            base_uid = uid.split("-")[0]
            save_dir = os.path.join(img_path, base_uid)
            get_deepface_controller().find(
                img_path=os.path.join(BASE_PATH, "static", "temp.png"),
                db_path=save_dir,
                model_name="Facenet512",
                detector_backend="retinaface",
                anti_spoofing=False,
            )
            current_hash = hash_directory(save_dir)
            logger.info(
                f"Setting initial hash for directory {save_dir} with current hash {current_hash}."
            )
            
            app.config["ZoDB"].set_directory_hash(base_uid, current_hash)

            logger.info("recreate_DB completed successfully.")
    except Exception as e:
        logger.error(f"Error in recreate_DB: {e} - {traceback.format_exc()}")



# def register_face(image: Any, uid: str, current_app) -> Dict[str, Any]:
#     try:
#         # Save the original image
#         image_path, _ = save_image(image, uid, IMAGES_DIR, uid)

#         # Augment and save images
#         augmented_images = augment_image(image)
#         for i, img in enumerate(augmented_images, start=1):
#             aug_image_path, _ = save_image(img, uid, IMAGES_DIR, f"{uid}_aug{i}")
#             add_task_to_queue(
#                 import_db,
#                 img_path=aug_image_path,
#                 app=current_app._get_current_object(),
#                 uid=f"{uid}_aug{i}",
#             )

#         # Register the original image
#         add_task_to_queue(
#             import_db,
#             img_path=image_path,
#             app=current_app._get_current_object(),
#             uid=uid,
#         )

#         # Schedule database recreation task
#         add_task_to_queue(
#             recreate_DB,
#             img_path=IMAGES_DIR,
#             app=current_app._get_current_object(),
#             uid=uid,
#         )

#         return {
#             "message": "Face registered successfully!",
#             "data": {"uid": uid},
#             "success": True,
#         }
#     except Exception as e:
#         logger.error(f"Exception while registering face with UID {uid}: {e} - {traceback.format_exc()}")
#         return {"message": "Failed to register face.", "data": None, "success": False}
def register_face(image: Any, uid: str, current_app) -> Dict[str, Any]:
    try:
        # Save the original image
        image_path, _ = save_image(image, uid, IMAGES_DIR, uid)

        # Augment and save images
        augmented_images = augment_image(image)
        image_paths = [image_path]

        for i, img in enumerate(augmented_images, start=1):
            aug_image_path, _ = save_image(img, uid, IMAGES_DIR, f"{uid}_aug{i}")
            image_paths.append(aug_image_path)
            # Add augmented images to database with directory hash
            add_task_to_queue(
                import_db,
                img_path=aug_image_path,
                app=current_app._get_current_object(),
                uid=f"{uid}_aug{i}",
            )

        

        # Schedule database recreation task
        add_task_to_queue(
            recreate_DB,
            
            img_path=IMAGES_DIR,
            app=current_app._get_current_object(),
            uid=uid,
        )

        return {
            "message": "Face registered successfully!",
            "data": {"uid": uid},
            "success": True,
        }
    except Exception as e:
        logger.error(f"Exception while registering face with UID {uid}: {e} - {traceback.format_exc()}")
        return {"message": "Failed to register face.", "data": None, "success": False}
    
def delete_face(uid: str, current_app) -> Dict[str, Any]:
    try:
        base_uid = uid.split("-")[0]
        save_dir = os.path.join(IMAGES_DIR, base_uid)

        # Delete images and check if directory is empty
        delete_images_for_uid(uid, base_uid)
        
        if delete_directory_if_empty(save_dir):
            current_app.config["ZoDB"].delete_face_embedding(uid=uid)

            return {
                "message": "Face deleted successfully and directory removed.",
                "data": {"uid": uid},
                "success": True,
            }
        
        # Delete all embeddings for a specific UID and its augmentations
        if current_app.config["ZoDB"].delete_face_embedding(uid=uid):
            add_task_to_queue(
                recreate_DB,
                img_path=IMAGES_DIR,
                app=current_app._get_current_object(),
                uid=uid,
            )
            return {
                "message": "Face deleted successfully and final task scheduled!",
                "data": {"uid": uid},
                "success": True,
            }
        else:
            return {"message": "Failed to delete face data", "data": None, "success": False}
    except Exception as e:
        logger.error(
            f"Exception while deleting face with UID {uid}: {e} - {traceback.format_exc()}"
        )
        return {"message": "Failed to delete face", "data": None, "success": False}


def recognize_face(image: Any, uid: Optional[str] = None) -> Dict[str, Any]:
    try:
        save_dir = IMAGES_DIR if uid is None else os.path.join(IMAGES_DIR, uid.split('-')[0])
        image_path, _ = save_image(image, "query_results", TEMP_DIR, "")
        logger.info(f"Recognizing face with UID {uid} in directory {save_dir}")
        

        if not any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(save_dir)):
            return {"message": "No faces detected in the image", "data": [], "success": False}
        anti_spoofing_results = get_deepface_controller().extract_faces(img_path=image_path, detector_backend="retinaface", anti_spoofing=True)[0]

        recognition_results = get_deepface_controller().find(img_path=image_path, db_path=save_dir, model_name="Facenet512", detector_backend="retinaface", anti_spoofing=False)

        if recognition_results and not recognition_results[0].empty:
            best_match = recognition_results[0].iloc[0]
            best_match_identity = extract_base_identity(os.path.splitext(os.path.basename(best_match['identity']))[0])
            best_match_confidence = round(float((1 - best_match['distance'] / best_match['threshold']) * 100), 2)

            return {
                "message": "Face recognized successfully!",
                "data": {
                    "best_match": {"identity": best_match_identity, "confidence": best_match_confidence},
                    "is_real": bool(anti_spoofing_results.get('is_real', False)),
                    "antispoof_score": round(float(anti_spoofing_results.get('antispoof_score', 0)), 2) * 100
                },
                "success": True
            }
        else:
            return {"message": "No faces detected in the image", "data": [], "success": False}
    except Exception as e:
        logger.error(f"Exception while recognizing face: {e} - {traceback.format_exc()}")
        return {"message": "Failed to recognize face", "data": None, "success": False}
def recognize_face_db(image: Any, uid = None,app=None) -> Dict[str, Any]:
    try:
        # Save the input image to a temporary directory
        image_path, _ = save_image(image, "query_results", TEMP_DIR, "")
        # Perform face recognition using the database manager
        recognition_results = get_deepface_controller().verify_faces_db(
            img_path=image_path,
            db_manager=app.config["ZoDB"],
            model_name="Facenet512",
            detector_backend="retinaface",
            anti_spoofing=True,
            uid = uid ,
        )

        if recognition_results and recognition_results[0].get("matches"):
            best_match = recognition_results[0]["matches"][0]
            
            # 20240826_154010_duclong-4_aug3
            best_match['identity'] = "20240826_154010_"+best_match['identity']
            
                        
            best_match_identity = extract_base_identity(best_match['identity'])
            
            
            best_match_confidence = round(float((1 - best_match['distance'] / recognition_results[0]['threshold']) * 100), 2)

            return {
                "message": "Face recognized successfully!",
                "data": {
                    "best_match": {
                        "identity": best_match_identity,
                        "confidence": best_match_confidence
                    },
                    "is_real": recognition_results[0].get('is_real', True),
                    "antispoof_score": round(float(recognition_results[0].get('antispoof_score', 0)), 2) * 100
                },
                "success": True
            }
        else:
            return {"message": "No matching faces found in the database", "data": [], "success": False}
    except Exception as e:
        logger.error(f"Exception while recognizing face: {e} - {traceback.format_exc()}")
        return {"message": "Failed to recognize face", "data": None, "success": False}

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
        return {
            "message": "Representation fetched successfully",
            "data": embedding_objs,
            "success": True,
        }
    except Exception as err:
        logger.error(
            f"Exception while representing: {str(err)} - {traceback.format_exc()}"
        )
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
        logger.error(
            f"Exception while verifying: {str(err)} - {traceback.format_exc()}"
        )
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
        logger.error(
            f"Exception while analyzing: {str(err)} - {traceback.format_exc()}"
        )
        return {"message": "Failed to analyze image", "data": None, "success": False}
