import sys
import os

import cv2
from keras.models import load_model
import numpy as np

from face_classification.src.utils.datasets import get_labels
from face_classification.src.utils.inference import detect_faces
from face_classification.src.utils.inference import draw_text
from face_classification.src.utils.inference import draw_bounding_box
from face_classification.src.utils.inference import apply_offsets
from face_classification.src.utils.inference import load_detection_model
from face_classification.src.utils.inference import load_image
from face_classification.src.utils.preprocessor import preprocess_input

import logging

logger = logging.getLogger('FaceEvaluator')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class FaceEvaluator:

    emotion_labels = get_labels('fer2013')
    gender_labels = get_labels('imdb')
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # hyper-parameters for bounding boxes shape
    gender_offsets = (30, 60)
    gender_offsets = (10, 10)
    emotion_offsets = (20, 40)
    emotion_offsets = (0, 0)

    def load_params(cd):
        # parameters for loading data and images
        FaceEvaluator.detection_model_path = os.path.join(cd, 'models/face_classification/trained_models/detection_models/haarcascade_frontalface_default.xml') 
        FaceEvaluator.emotion_model_path = os.path.join(cd, 'models/face_classification/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')
        FaceEvaluator.gender_model_path = os.path.join(cd, 'models/face_classification/trained_models/gender_models/simple_CNN.81-0.96.hdf5')

        # loading models
        FaceEvaluator.face_detection = load_detection_model(FaceEvaluator.detection_model_path)
        FaceEvaluator.emotion_classifier = load_model(FaceEvaluator.emotion_model_path, compile=False)
        FaceEvaluator.gender_classifier = load_model(FaceEvaluator.gender_model_path, compile=False)

        # getting input model shapes for inference
        FaceEvaluator.emotion_target_size = FaceEvaluator.emotion_classifier.input_shape[1:3]
        FaceEvaluator.gender_target_size = FaceEvaluator.gender_classifier.input_shape[1:3]

    @staticmethod
    def describe_face(npimg):
        # loading images
        rgb_image = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(npimg, cv2.COLOR_BGR2GRAY)
        
        #gray_image = npimg.convert('LA')
        #gray_image = np.squeeze(gray_image)
        #gray_image = gray_image.astype('uint8')

        #cv2.imwrite('c:/users/andre/documents/github/hcEye/test/name.png', gray_image)
        
        faces = detect_faces(FaceEvaluator.face_detection, gray_image)
        face_features = []
        
        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, FaceEvaluator.gender_offsets)
            rgb_face = rgb_image[y1:y2, x1:x2]

            x1, x2, y1, y2 = apply_offsets(face_coordinates, FaceEvaluator.emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                rgb_face = cv2.resize(rgb_face, (FaceEvaluator.gender_target_size))
                gray_face = cv2.resize(gray_face, (FaceEvaluator.emotion_target_size))
            except:
                continue

            rgb_face = preprocess_input(rgb_face, False)
            rgb_face = np.expand_dims(rgb_face, 0)

            #cv2.imwrite('c:/users/andre/documents/github/hcEye/test/face.png', gray_face)

            gender_prediction = FaceEvaluator.gender_classifier.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = FaceEvaluator.gender_labels[gender_label_arg]
            
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = FaceEvaluator.emotion_classifier.predict(gray_face)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = FaceEvaluator.emotion_labels[emotion_label_arg]
            FaceEvaluator.emotion_probability = np.max(emotion_prediction)
            
            face_features.append([face_coordinates.tolist(), gender_text, emotion_text])
            #print(face_features)


        return face_features
