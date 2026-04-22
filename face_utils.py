# face_utils.py
import dlib
import numpy as np
import face_recognition
from config import SHAPE_PREDICTOR_PATH, LEFT_EYE_START, LEFT_EYE_END, RIGHT_EYE_START, RIGHT_EYE_END
from extensions import db
from models import User

# Inisialisasi dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# Penyimpanan global encoding wajah
known_face_encodings = []
known_face_names = []

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def load_encodings_from_db():
    global known_face_encodings, known_face_names
    users = User.query.all()
    known_face_encodings = [u.face_encoding for u in users]
    known_face_names = [u.user_id for u in users]
    print(f"[INFO] {len(known_face_encodings)} wajah dimuat dari DB.")
    