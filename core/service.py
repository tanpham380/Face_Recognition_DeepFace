import datetime
import json
import time
import traceback
from typing import Optional, List, Dict, Any
import os
import numpy as np
from core.utils.augment_images import augment_image
from core.utils.database import SQLiteManager
from core.utils.save_file import delete_images, save_image
# from core.utils.search_embleeding import find_in_db
from core.utils.static_variable import IMAGES_DIR
from core.utils.logging import get_logger
from core.deepface_controller.controller import DeepFaceController
import threading

from core.utils.threading import add_task_to_queue
from concurrent.futures import ThreadPoolExecutor, as_completed


# Initialize the logger
logger = get_logger()

deepface_controller = DeepFaceController()


def check_version() -> Dict[str, Any]:
    try:
        version = deepface_controller.check_version()
        return {"message": "Version fetched successfully", "data": version, "success": True}
    except Exception as e:
        logger.error(f"Exception while checking version: {
                     str(e)} - {traceback.format_exc()}")
        return {"message": "Failed to fetch version", "data": None, "success": False}


def delete_face(uid: str, db_manager: SQLiteManager) -> Dict[str, Any]:
    try:
        if not db_manager.uid_exists(uid):
            return {"message": "UID not found", "data": None, "success": False}

        db_manager.delete_embedding_by_uid(uid)
        return {"message": "Face deleted successfully!", "data": {"uid": uid}, "success": True}
    except Exception as e:
        logger.error(f"Exception while deleting face with UID {
                     uid}: {str(e)} - {traceback.format_exc()}")
        return {"message": "Failed to delete face", "data": None, "success": False}

def check_and_run_final_task(db_manager: SQLiteManager, app):
    """Check if all tasks are done and run the final task if they are."""
    with db_manager.get_connection() as conn:
        cursor = conn.execute("""
            SELECT COUNT(*) FROM task_status WHERE status != 'completed'
        """)
        remaining_tasks = cursor.fetchone()[0]
    logger.info(f"Remaining tasks: {remaining_tasks}")
    if remaining_tasks == 0:
        # Run the final deepface_controller.find task
        logger.info("All tasks completed. Running final deepface_controller.find task.")
        with app.app_context():
            deepface_controller.find(
                # Specify the parameters required
                img_path=None,  # Adjust this based on your logic
                db_path=IMAGES_DIR,
                model_name="Facenet512",
                detector_backend="retinaface",
                anti_spoofing=False
            )

            
def handle_background_tasks(db_manager: SQLiteManager, image_paths, uid, app, max_embeddings=20):
    logger.info(f"Running background processing for UID {uid}")
    with app.app_context():
        try:
            for img_path in image_paths:
                try:
                    logger.info(f"Processing image: {img_path}")
                    represent_objs = deepface_controller.represent(
                        img_path=img_path,
                        model_name="Facenet512",
                        detector_backend="retinaface",
                        enforce_detection=True,
                        align=True,
                        anti_spoofing=False
                    )

                    if represent_objs:
                        embedding = np.array(represent_objs[0]["embedding"], dtype="float32").tolist()

                        # Get existing embeddings from the database
                        existing_embeddings = db_manager.get_embeddings(uid)

                        # Delete old embeddings if they exceed the limit
                        if len(existing_embeddings) >= max_embeddings:
                            db_manager.delete_oldest_embeddings(
                                uid, len(existing_embeddings) - max_embeddings + 1)

                        # Insert or update the new embedding in the database
                        db_manager.insert_or_update_embedding(uid, embedding)

                        logger.info(f"Face registered successfully for UID {uid}")
                    else:
                        logger.warning(f"Could not detect any face in image {img_path}")

                except Exception as e:
                    logger.error(f"Exception during background processing for UID {uid}: {str(e)} - {traceback.format_exc()}")
        finally:
            check_and_run_final_task(db_manager, app)


def handle_background_tasks2(db_manager: SQLiteManager, image_paths, uid, app, max_embeddings=20):
    logger.info(f"Running background processing for UID {uid}")
    
    with app.app_context():
        try:
            new_embeddings = []

            # Process images in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(deepface_controller.represent, img_path=img_path,
                                           model_name="Facenet512",
                                           detector_backend="retinaface",
                                           enforce_detection=True,
                                           align=True,
                                           anti_spoofing=False): img_path for img_path in image_paths}

                for future in as_completed(futures):
                    img_path = futures[future]
                    try:
                        represent_objs = future.result()

                        if represent_objs:
                            embedding = np.array(represent_objs[0]["embedding"], dtype="float32").tolist()
                            new_embeddings.append(embedding)
                            logger.info(f"Face registered successfully for UID {uid} from image {img_path}")
                        else:
                            logger.warning(f"Could not detect any face in image {img_path}")
                    except Exception as e:
                        logger.error(f"Exception during background processing for image {img_path} of UID {uid}: {str(e)} - {traceback.format_exc()}")

            if new_embeddings:
                # Get existing embeddings from the database
                existing_embeddings = db_manager.get_embeddings(uid)

                # Combine old and new embeddings and enforce max limit
                combined_embeddings = existing_embeddings + new_embeddings
                if len(combined_embeddings) > max_embeddings:
                    combined_embeddings = combined_embeddings[-max_embeddings:]

                # Update the database with the combined embeddings
                db_manager.update_embeddings(uid, combined_embeddings)

        finally:
            # Run this after all images are processed, not after each image
            deepface_controller.find(
                # Could be the last image or another if needed
                img_path=image_paths[-1],
                db_path=IMAGES_DIR,
                model_name="Facenet512",
                detector_backend="retinaface",
                anti_spoofing=False
            )

def register_face(image: Any, uid: str, db_manager: SQLiteManager, current_app) -> Dict[str, Any]:
    try:
        # Save the original image and get only the image path
        image_path, _ = save_image(image, uid, IMAGES_DIR, uid)

        # Create augmented images
        augmented_images = augment_image(image)

        # Save augmented images and store only their paths
        augmented_image_paths = []
        for i, img in enumerate(augmented_images, start=1):
            augmented_image_path, _ = save_image(img, uid, IMAGES_DIR, f"{uid}_aug{i}")
            augmented_image_paths.append(augmented_image_path)

        # Add processing task to the queue
        add_task_to_queue(handle_background_tasks, db_manager, [image_path] + augmented_image_paths, uid, current_app._get_current_object())

        # Return result immediately to the user
        return {"message": "Face registered successfully! Processing in background.", "data": {"uid": uid}, "success": True}

    except Exception as e:
        logger.error(f"Exception while registering face with UID {uid}: {str(e)} - {traceback.format_exc()}")
        return {"message": "Failed to register face", "data": None, "success": False}


# def register_face(image: Any, uid: str, db_manager: SQLiteManager) -> Dict[str, Any]:
#     try:
#         # Save the image using the previous save_image function
#         image_path = save_image(image, uid, IMAGES_DIR, uid)

#         # Get face embeddings using DeepFace
#         represent_objs = deepface_controller.represent(
#             img_path=image_path,
#             model_name="Facenet512",
#             detector_backend="retinaface",
#             enforce_detection=True,
#             align=True,
#             anti_spoofing=False
#         )

#         if represent_objs:
#             embedding = np.array(represent_objs[0]["embedding"], dtype="float32").tolist()

#             # Retrieve current embeddings from the database
#             existing_embeddings = db_manager.get_embeddings(uid)

#             # If the number of embeddings exceeds the limit, remove the oldest embeddings
#             max_embeddings = 10
#             if len(existing_embeddings) >= max_embeddings:
#                 db_manager.delete_oldest_embeddings(uid, len(existing_embeddings) - max_embeddings + 1)

#             # Insert or update the embedding in the database
#             db_manager.insert_or_update_embedding(uid, embedding)

#             return {"message": "Face registered successfully!", "data": {"uid": uid}, "success": True}
#         else:
#             return {"message": "Unknown faces in the image", "data": None, "success": False}

#     except Exception as e:
#         logger.error(f"Exception while registering face with UID {uid}: {str(e)} - {traceback.format_exc()}")
#         return {"message": "Failed to register face", "data": None, "success": False}


def recognize_face(image: Any) -> Dict[str, Any]:
    try:
        # Save the image temporarily
        image_path, _  = save_image(image, "query_results", "/tmp", "")
        
        value_objs_anti_spoofing = deepface_controller.extract_faces(
            img_path=image_path,
            detector_backend="retinaface",
            anti_spoofing=True
        )[0]
        

        # Perform face recognition using DeepFace
        value_objs = deepface_controller.find(
            img_path=image_path,
            db_path=IMAGES_DIR,
            model_name="Facenet512",
            detector_backend="retinaface",
            anti_spoofing=False
        )

        # If results are returned, construct the response
        if value_objs and not value_objs[0].empty:
            # Extract the most similar match (first match, as the DataFrame should be sorted by distance)
            best_match = value_objs[0].iloc[0]

            # Convert necessary fields to Python native types
            best_match_identity = str(best_match['identity'])
            best_match_confidence = round(float((1 - best_match['distance'] / best_match['threshold']) * 100), 2)

            # Include all matches in the data
            all_matches = value_objs[0].apply(
                lambda row: {
                    "identity": str(row['identity']),
                    "distance": round(float(row['distance']), 2),
                    "threshold": round(float(row['threshold']), 2),
                    "confidence": round((1 - float(row['distance']) / float(row['threshold'])) * 100, 2)
                },
                axis=1
            ).tolist()

            # Construct the response
            response = {
                "message": "Face recognized successfully!",
                "data": {
                    "best_match": {
                        "identity": best_match_identity,
                        "confidence": best_match_confidence  # Rounded to 2 decimal places
                    },
                    "all_matches": all_matches,
                    "source_region": {
                        "source_x": int(best_match['source_x']),
                        "source_y": int(best_match['source_y']),
                        "source_w": int(best_match['source_w']),
                        "source_h": int(best_match['source_h']),
                    },
                    "is_real": bool(value_objs_anti_spoofing.get('is_real', False)),  # Assuming this is a boolean or similar
                    "antispoof_score": round(float(value_objs_anti_spoofing.get('antispoof_score', 0)), 2) * 100
                },
                "success": True
            }
            return response

        # If no matches were found
        else:
            return {"message": "No faces detected in the image", "data": [], "success": False}

    except Exception as e:
        logger.error(f"Exception while recognizing face: {str(e)} - {traceback.format_exc()}")
        return {"message": "Failed to recognize face", "data": None, "success": False}
    finally:
        # Clean up temporary images
        delete_images(os.path.join("/tmp", "query_results"))



def recognize_face_with_database(image: Any, db_manager: SQLiteManager) -> Dict[str, Any]:
    try:
        image_path, _ = save_image(image, "query_results", "/tmp", "")

        value_objs = deepface_controller.verify_faces_db(
            img_path=image_path,
            db_manager=db_manager,
            model_name="Facenet512",
            detector_backend="retinaface",
            anti_spoofing=True
        )

        if value_objs and 'matches' in value_objs[0] and value_objs[0]['matches']:
            # Extract the threshold value from the top-level dictionary
            threshold = value_objs[0].get('threshold', None)
            if threshold is None:
                raise KeyError("Threshold is missing in the response")

            # Extract the most similar match (first match, as the list is sorted by distance)
            best_match = value_objs[0]['matches'][0]

            # Calculate confidence using the threshold from the top level
            confidence = (1 - best_match['distance'] / threshold) * 100
            confidence = round(confidence, 2)

            # Include all matches in the data
            all_matches = [
                {
                    "identity": match['identity'],
                    "distance": match['distance'],
                    "threshold": threshold,  # Use the threshold from the top level
                    "confidence": round((1 - match['distance'] / threshold) * 100, 2)
                }
                for match in value_objs[0]['matches']
            ]

            # Construct the response
            response = {
                "message": "Face recognized successfully!",
                "data": {
                    "best_match": {
                        "identity": best_match['identity'],
                        "confidence": round(confidence, 2)
                    },
                    "all_matches": all_matches,
                    "source_region": value_objs[0]['source_region'],
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
        delete_images(os.path.join("/tmp", "query_results"))




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
        embedding_objs = deepface_controller.represent(
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
        logger.error(f"Exception while representing: {
                     str(err)} - {traceback.format_exc()}")
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
        result = deepface_controller.verify(
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
        logger.error(f"Exception while verifying: {
                     str(err)} - {traceback.format_exc()}")
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
        demographies = deepface_controller.analyze(
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
        logger.error(f"Exception while analyzing: {
                     str(err)} - {traceback.format_exc()}")
        return {"message": "Failed to analyze image", "data": None, "success": False}
