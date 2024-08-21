import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from core.utils.database import SQLiteManager
from deepface.modules import detection, representation, verification
from core.utils.logging import get_logger
logger = get_logger()
# def find_in_db(
#     img_path: Union[str, np.ndarray],
#     db_manager: SQLiteManager,
#     model_name: str = "VGG-Face",
#     distance_metric: str = "cosine",
#     enforce_detection: bool = True,
#     detector_backend: str = "opencv",
#     align: bool = True,
#     expand_percentage: int = 0,
#     threshold: Optional[float] = None,
#     normalization: str = "base",
#     silent: bool = False,
#     anti_spoofing: bool = False,
# ) -> List[Dict[str, Any]]:
#     """
#     Identify individuals in a database by matching embeddings.

#     Args:
#         img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
#             or a base64 encoded image. If the source image contains multiple faces, the result will
#             include information for each detected face.

#         db_manager (SQLiteManager): Database manager instance to interact with the database.

#         model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
#             OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

#         distance_metric (string): Metric for measuring similarity. Options: 'cosine',
#             'euclidean', 'euclidean_l2'.

#         enforce_detection (boolean): If no face is detected in an image, raise an exception.
#             Default is True. Set to False to avoid the exception for low-resolution images.

#         detector_backend (string): Face detector backend. Options: 'opencv', 'retinaface',
#             'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'.

#         align (boolean): Perform alignment based on the eye positions.

#         expand_percentage (int): Expand detected facial area with a percentage (default is 0).

#         threshold (float): Specify a threshold to determine whether a pair represents the same
#             person or different individuals. This threshold is used for comparing distances.
#             If left unset, default pre-tuned threshold values will be applied based on the specified
#             model name and distance metric (default is None).

#         normalization (string): Normalize the input image before feeding it to the model.
#             Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

#         silent (boolean): Suppress or allow some log messages for a quieter analysis process.

#         anti_spoofing (boolean): Flag to enable anti-spoofing (default is False).


#     Returns:
#         results (List[Dict[str, Any]]): A list of dictionaries, each representing a match found in the database.
#     """

#     tic = time.time()

#     # Detect faces and get embeddings from the provided image
#     source_objs = detection.extract_faces(
#         img_path=img_path,
#         detector_backend=detector_backend,
#         grayscale=False,
#         enforce_detection=enforce_detection,
#         align=align,
#         expand_percentage=expand_percentage,
#         anti_spoofing=anti_spoofing,
#     )

#     # Get embeddings from the database
#     db_embeddings = db_manager.get_all_embeddings()

#     if not db_embeddings:
#         logger.info("No embeddings found in the database.")
#         return []

#     resp_obj = []

#     # for source_obj in source_objs:
#     #     if anti_spoofing and not source_obj.get("is_real", True):
#     #         raise ValueError("Spoof detected in the given image.")

#     source_img = source_objs["face"]
#     source_region = source_objs["facial_area"]
#     target_embedding_obj = representation.represent(
#         img_path=source_img,
#         model_name=model_name,
#         enforce_detection=enforce_detection,
#         detector_backend="skip",
#         align=align,
#         normalization=normalization,
#     )

#     target_representation = target_embedding_obj[0]["embedding"]

#     distances = []
#     identities = []

#     for db_entry in db_embeddings:
#         source_representation = db_entry['embedding']

#         distance = verification.find_distance(
#             source_representation, target_representation, distance_metric
#         )

#         distances.append(distance)
#         identities.append(db_entry['uid'])

#     target_threshold = threshold or verification.find_threshold(model_name, distance_metric)

#     result_df = pd.DataFrame({
#         "identity": identities,
#         "distance": distances
#     })

#     result_df["threshold"] = target_threshold
#     result_df = result_df[result_df["distance"] <= target_threshold]
#     result_df = result_df.sort_values(by=["distance"], ascending=True).reset_index(drop=True)

#     if not result_df.empty:
#         resp_obj.append({
#             "source_region": source_region,
#             "matches": result_df.to_dict(orient='records'),
#             "is_real": source_objs.get("is_real", True),
#             "antispoof_score": source_objs["antispoof_score", 0.0]
            
#         })

#     toc = time.time()
#     logger.info(f"find_in_db function duration {toc - tic} seconds")

#     return resp_obj


