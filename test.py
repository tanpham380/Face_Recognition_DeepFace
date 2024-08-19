import os
from deepface import DeepFace


os.environ['KMP_DUPLICATE_LIB_OK']='True'

a = DeepFace.analyze(img_path = "file2.jpg" , anti_spoofing = True, actions=  ("age"),) 


face_objs = DeepFace.extract_faces(
  img_path = "file2.jpg", detector_backend = "opencv", grayscale = False, enforce_detection = True, align = True, anti_spoofing = True
)
print(face_objs)