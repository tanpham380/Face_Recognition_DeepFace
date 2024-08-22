from deepface.commons import package_utils, folder_utils
from deepface import __version__
from deepface.modules import (
    modeling,
    representation,
    verification,
    recognition,
    demography,
    detection,
    streaming,
    preprocessing,
)
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import warnings
import logging
from typing import Any, Dict, List, Union, Optional
from multiprocessing import Pool, cpu_count

from core.utils.database import SQLiteManager
from core.utils.logging import get_logger

logger = get_logger()

# -----------------------------------
# configurations for dependencies

# Users should install tf_keras package if they are using tf 2.16 or later versions
package_utils.validate_for_keras3()

warnings.filterwarnings("ignore")
tf_version = package_utils.get_tf_major_version()

if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
messange = None
# Configure TensorFlow to use GPU
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    # Ensure that TensorFlow uses the GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    messange = ("Using GPU for TensorFlow operations")
    logger.info("Using GPU for TensorFlow operations")
else:
    logger.info("No GPU found. Running on CPU")
    messange = ("Using CPU for TensorFlow operations")
# -----------------------------------

# Create required folders if necessary to store model weights
folder_utils.initialize_folder()

def parallel_distance_computation(embedding_chunk, target_representation, distance_metric, threshold):
    try:
        # Convert the embeddings to a NumPy array
        embedding_array = np.array([np.frombuffer(e['embedding'], dtype='float32') for e in embedding_chunk])

        # Calculate distances based on the specified metric
        if distance_metric == "cosine":
            distances = 1 - np.dot(embedding_array, target_representation) / (
                np.linalg.norm(embedding_array, axis=1) * np.linalg.norm(target_representation)
            )
        elif distance_metric == "euclidean":
            distances = np.linalg.norm(embedding_array - target_representation, axis=1)
        elif distance_metric == "euclidean_l2":
            distances = np.linalg.norm(embedding_array - target_representation, axis=1) ** 2

        # Filter results by the threshold and collect matches
        matches = [
            {"identity": embedding_chunk[i]['uid'], "distance": distances[i]}
            for i in range(len(distances))
            if distances[i] <= threshold
        ]
    except Exception as e:
        logger.error(f"Error in parallel_distance_computation: {e}")
        matches = []

    return matches
class DeepFaceController:
    def __init__(self):
        self._face_detector_backend = [
            'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
            'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface'
        ]
        self._model_name = [
            "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
            "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"
        ]
        self._distance_metric = ["cosine", "euclidean", "euclidean_l2"]
        self._face_detector_threshold = 0.6
        self._face_detector_enforce_detection = False
        self._anti_spoofing = False
        self._align = True
        self._actions = ['age', 'gender', 'race', 'emotion']

    @property
    def face_detector_backend(self) -> List[str]:
        return self._face_detector_backend

    @face_detector_backend.setter
    def face_detector_backend(self, backend: Union[str, List[str]]):
        if isinstance(backend, str):
            self._face_detector_backend = [backend]
        elif isinstance(backend, list):
            self._face_detector_backend = backend

    @property
    def face_detector_threshold(self) -> float:
        return self._face_detector_threshold

    @face_detector_threshold.setter
    def face_detector_threshold(self, threshold: float):
        self._face_detector_threshold = threshold

    @property
    def face_detector_enforce_detection(self) -> bool:
        return self._face_detector_enforce_detection

    @face_detector_enforce_detection.setter
    def face_detector_enforce_detection(self, enforce: bool):
        self._face_detector_enforce_detection = enforce

    @property
    def model_name(self) -> List[str]:
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: Union[str, List[str]]):
        if isinstance(model_name, str):
            self._model_name = [model_name]
        elif isinstance(model_name, list):
            self._model_name = model_name

    @property
    def anti_spoofing(self) -> bool:
        return self._anti_spoofing

    @anti_spoofing.setter
    def anti_spoofing(self, anti_spoofing: bool):
        self._anti_spoofing = anti_spoofing

    @property
    def align(self) -> bool:
        return self._align

    @align.setter
    def align(self, align: bool):
        self._align = align

    @property
    def distance_metric(self) -> List[str]:
        return self._distance_metric

    @distance_metric.setter
    def distance_metric(self, metric: Union[str, List[str]]):
        if isinstance(metric, str):
            self._distance_metric = [metric]
        elif isinstance(metric, list):
            self._distance_metric = metric

    @property
    def actions(self) -> List[str]:
        return self._actions

    @actions.setter
    def actions(self, actions: Union[str, List[str]]):
        if isinstance(actions, str):
            self._actions = [actions]
        elif isinstance(actions, list):
            self._actions = actions

    def check_version(self) -> str:
        return {"message": messange, "version": __version__}

    def verify_faces_db(
            self,
            img_path: Union[str, np.ndarray],
            db_manager: SQLiteManager,
            model_name: Optional[Union[str, List[str]]] = None,
            distance_metric: Optional[Union[str, List[str]]] = None,
            enforce_detection: Optional[bool] = None,
            detector_backend: Optional[Union[str, List[str]]] = None,
            align: Optional[bool] = None,
            expand_percentage: int = 0,
            threshold: Optional[float] = None,
            normalization: str = "base",
            silent: bool = False,
            anti_spoofing: Optional[bool] = None,
        ) -> List[Dict[str, Any]]:
            tic = time.time()

            # Use class attributes if parameters are not provided
            model_name = model_name or self._model_name[0]
            distance_metric = distance_metric or self._distance_metric[0]
            enforce_detection = enforce_detection if enforce_detection is not None else self._face_detector_enforce_detection
            detector_backend = detector_backend or self._face_detector_backend[0]
            align = align if align is not None else self._align
            anti_spoofing = anti_spoofing if anti_spoofing is not None else self._anti_spoofing

            # Detect faces and get embeddings from the provided image
            source_objs = detection.extract_faces(
                img_path=img_path,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
                expand_percentage=expand_percentage,
                anti_spoofing=anti_spoofing,
            )

            if not source_objs:
                return []

            source_obj = source_objs[0]  # Use only the first detected face

            # Get all embeddings from the database
            db_embeddings = db_manager.get_all_embeddings()
            if not db_embeddings:
                return []

            source_img = source_obj["face"]
            target_representation = representation.represent(
                img_path=source_img,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )[0]["embedding"]

            # Convert target representation to NumPy array
            target_representation_array = np.array(target_representation, dtype='float32')

            # Use the number of CPU cores for parallel processing
            num_cores = cpu_count()
            chunk_size = len(db_embeddings) // num_cores + 1
            embedding_chunks = [db_embeddings[i:i + chunk_size] for i in range(0, len(db_embeddings), chunk_size)]

            # Use multiprocessing to calculate distances in parallel with batch processing
            target_threshold = threshold or verification.find_threshold(model_name, distance_metric)
            with Pool(processes=num_cores) as pool:
                results = pool.starmap(
                    parallel_distance_computation, 
                    [(chunk, target_representation_array, distance_metric, target_threshold) for chunk in embedding_chunks]
                )

            # Flatten the results and sort by distance
            matches = [item for sublist in results for item in sublist]
            matches.sort(key=lambda x: x["distance"])

            if not matches:
                return []

            resp_obj = [{
                "source_region": source_obj["facial_area"],
                "matches": matches,
                "threshold": target_threshold,
                "is_real": source_obj.get("is_real", True),
                "antispoof_score": source_obj.get("antispoof_score", 0.0)
            }]

            toc = time.time()
            resp_obj.append({"time_excuse": toc - tic})

            return resp_obj
        
    
    def build_model(self, model_name: str) -> Any:
        """
        This function builds a deepface model
        Args:
            model_name (string): face recognition or facial attribute model
                VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
                Age, Gender, Emotion, Race for facial attributes
        Returns:
            built_model
        """
        return modeling.build_model(model_name=model_name)

    def verify(
        self,
        img1_path: Union[str, np.ndarray, List[float]],
        img2_path: Union[str, np.ndarray, List[float]],
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        distance_metric: str = "cosine",
        enforce_detection: bool = True,
        align: bool = True,
        expand_percentage: int = 0,
        normalization: str = "base",
        silent: bool = False,
        threshold: Optional[float] = None,
        anti_spoofing: bool = False,
    ) -> Dict[str, Any]:
        return verification.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            normalization=normalization,
            silent=silent,
            threshold=threshold,
            anti_spoofing=anti_spoofing,
        )

    def analyze(
        self,
        img_path: Union[str, np.ndarray],
        actions: Union[tuple, list] = ("emotion", "age", "gender", "race"),
        enforce_detection: bool = True,
        detector_backend: str = "opencv",
        align: bool = True,
        expand_percentage: int = 0,
        silent: bool = False,
        anti_spoofing: bool = False,
    ) -> List[Dict[str, Any]]:
        return demography.analyze(
            img_path=img_path,
            actions=actions,
            enforce_detection=enforce_detection,
            detector_backend=detector_backend,
            align=align,
            expand_percentage=expand_percentage,
            silent=silent,
            anti_spoofing=anti_spoofing,
        )

    def find(
        self,
        img_path: Union[str, np.ndarray],
        db_path: str,
        model_name: str = "VGG-Face",
        distance_metric: str = "cosine",
        enforce_detection: bool = True,
        detector_backend: str = "opencv",
        align: bool = True,
        expand_percentage: int = 0,
        threshold: Optional[float] = None,
        normalization: str = "base",
        silent: bool = False,
        refresh_database: bool = True,
        anti_spoofing: bool = False,
    ) -> List[pd.DataFrame]:
        """
        Identify individuals in a database
        Args:
            img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
                or a base64 encoded image. If the source image contains multiple faces, the result will
                include information for each detected face.

            db_path (string): Path to the folder containing image files. All detected faces
                in the database will be considered in the decision-making process.

            model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
                OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

            distance_metric (string): Metric for measuring similarity. Options: 'cosine',
                'euclidean', 'euclidean_l2' (default is cosine).

            enforce_detection (boolean): If no face is detected in an image, raise an exception.
                Set to False to avoid the exception for low-resolution images (default is True).

            detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
                'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
                (default is opencv).

            align (boolean): Perform alignment based on the eye positions (default is True).

            expand_percentage (int): expand detected facial area with a percentage (default is 0).

            threshold (float): Specify a threshold to determine whether a pair represents the same
                person or different individuals. This threshold is used for comparing distances.
                If left unset, default pre-tuned threshold values will be applied based on the specified
                model name and distance metric (default is None).

            normalization (string): Normalize the input image before feeding it to the model.
                Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base).

            silent (boolean): Suppress or allow some log messages for a quieter analysis process
                (default is False).

            refresh_database (boolean): Synchronizes the images representation (pkl) file with the
                directory/db files, if set to false, it will ignore any file changes inside the db_path
                (default is True).

            anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        Returns:
            results (List[pd.DataFrame]): A list of pandas dataframes. Each dataframe corresponds
                to the identity information for an individual detected in the source image.
                The DataFrame columns include:

            - 'identity': Identity label of the detected individual.

            - 'target_x', 'target_y', 'target_w', 'target_h': Bounding box coordinates of the
                    target face in the database.

            - 'source_x', 'source_y', 'source_w', 'source_h': Bounding box coordinates of the
                    detected face in the source image.

            - 'threshold': threshold to determine a pair whether same person or different persons

            - 'distance': Similarity score between the faces based on the
                    specified model and distance metric
        """
        return recognition.find(
            img_path=img_path,
            db_path=db_path,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            detector_backend=detector_backend,
            align=align,
            expand_percentage=expand_percentage,
            threshold=threshold,
            normalization=normalization,
            silent=silent,
            refresh_database=refresh_database,
            anti_spoofing=anti_spoofing,
        )

    def represent(
        self,
        img_path: Union[str, np.ndarray],
        model_name: str = "VGG-Face",
        enforce_detection: bool = True,
        detector_backend: str = "opencv",
        align: bool = True,
        expand_percentage: int = 0,
        normalization: str = "base",
        anti_spoofing: bool = False,
    ) -> List[Dict[str, Any]]:
        return representation.represent(
            img_path=img_path,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend=detector_backend,
            align=align,
            expand_percentage=expand_percentage,
            normalization=normalization,
            anti_spoofing=anti_spoofing,
        )

    def stream(
        self,
        db_path: str = "",
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        distance_metric: str = "cosine",
        enable_face_analysis: bool = True,
        source: Any = 0,
        time_threshold: int = 5,
        frame_threshold: int = 5,
        anti_spoofing: bool = False,
    ) -> None:
        time_threshold = max(time_threshold, 1)
        frame_threshold = max(frame_threshold, 1)

        streaming.analysis(
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enable_face_analysis=enable_face_analysis,
            source=source,
            time_threshold=time_threshold,
            frame_threshold=frame_threshold,
            anti_spoofing=anti_spoofing,
        )

    def extract_faces(
        self,
        img_path: Union[str, np.ndarray],
        detector_backend: str = "opencv",
        enforce_detection: bool = True,
        align: bool = True,
        expand_percentage: int = 0,
        grayscale: bool = False,
        anti_spoofing: bool = False,
    ) -> List[Dict[str, Any]]:
        return detection.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            grayscale=grayscale,
            anti_spoofing=anti_spoofing,
        )

    def cli(self) -> None:
        """
        Command line interface function.
        """
        import fire
        fire.Fire()

    def detectFace(
        self,
        img_path: Union[str, np.ndarray],
        target_size: tuple = (224, 224),
        detector_backend: str = "opencv",
        enforce_detection: bool = True,
        align: bool = True,
    ) -> Union[np.ndarray, None]:
        """
        Deprecated face detection function. Use extract_faces for same functionality.

        Args:
            img_path (str or np.ndarray): Path to the first image. Accepts exact image path
                as a string, numpy array (BGR), or base64 encoded images.

            target_size (tuple): final shape of facial image. black pixels will be
                added to resize the image (default is (224, 224)).

            detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
                'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
                (default is opencv).

            enforce_detection (boolean): If no face is detected in an image, raise an exception.
                Set to False to avoid the exception for low-resolution images (default is True).

            align (bool): Flag to enable face alignment (default is True).

        Returns:
            img (np.ndarray): detected (and aligned) facial area image as numpy array
        """
        logger.warn(
            "Function detectFace is deprecated. Use extract_faces instead.")
        face_objs = self.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            grayscale=False,
        )
        if face_objs:
            extracted_face = face_objs[0]["face"]
            return preprocessing.resize_image(
                img=extracted_face, target_size=target_size)
        return None


