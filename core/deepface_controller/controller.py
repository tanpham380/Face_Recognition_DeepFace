from deepface import DeepFace
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
import warnings
import logging
from typing import Any, Dict, List, Union, Optional
import time
from core.utils.logging import get_logger
from core.utils.database import ZoDB_Manager
logger = get_logger()

# Configurations for dependencies
from deepface.commons import package_utils, folder_utils

package_utils.validate_for_keras3()

warnings.filterwarnings("ignore")
tf_version = package_utils.get_tf_major_version()

if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)

# Configure TensorFlow to use GPU
messange = None
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    messange = "Using GPU for TensorFlow operations"
    logger.info(messange)
else:
    messange = "No GPU found. Running on CPU"
    logger.info(messange)

folder_utils.initialize_folder()



class DeepFaceController():
    def __init__(self):
        # super().__init__()
        self._face_detector_backend = [
            'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
            'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface'
        ]
        self._face_detector_threshold = 0.6
        self._face_detector_enforce_detection = False
        self._model_name = [
            "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
            "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"
        ]
        self._anti_spoofing = False
        self._align = True
        self._distance_metric = ["cosine", "euclidean", "euclidean_l2"]
        self._actions = ['age', 'gender', 'race', 'emotion']
    # Getter and Setter for attributes

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
    def check_version(self) -> str :
        return {"message": messange , "version": __version__} 

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
    
    
        
    def verify_faces_db(
        self,
        img_path: Union[str, np.ndarray],
        db_manager: ZoDB_Manager,
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

        # Get embeddings from the database
        db_embeddings = []
        all_face_data = db_manager.list_face_data_embedding()
        for uid, face_data in all_face_data.items():
            for embedding in face_data['embedding']:
                db_embeddings.append({
                    'uid': uid,
                    'embedding': embedding
                })

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

        # Calculate distances and match identities
        distances = []
        identities = []

        for db_entry in db_embeddings:
            distance = verification.find_distance(
                db_entry['embedding'], target_representation, distance_metric
            )
            distances.append(distance)
            identities.append(db_entry['uid'])

        target_threshold = threshold or verification.find_threshold(
            model_name, distance_metric
        )

        # Filter results based on threshold
        result_df = pd.DataFrame({
            "identity": identities,
            "distance": distances
        })
        result_df = result_df[result_df["distance"] <= target_threshold]
        result_df = result_df.sort_values(by="distance").reset_index(drop=True)

        resp_obj = []
        if not result_df.empty:
            resp_obj.append({
                "source_region": source_obj["facial_area"],
                "matches": result_df.to_dict(orient='records'),
                "threshold": target_threshold,
                "is_real": source_obj.get("is_real", True),
                "antispoof_score": source_obj.get("antispoof_score", 0.0)
            })

        toc = time.time()
        resp_obj.append({
            "time_excuse": toc - tic,
        })
        return resp_obj

