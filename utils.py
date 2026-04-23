# face_utils.py
import dlib
import numpy as np
import face_recognition
from config import SHAPE_PREDICTOR_PATH, LEFT_EYE_START, LEFT_EYE_END, RIGHT_EYE_START, RIGHT_EYE_END
from extensions import db
from models import User
import math

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


#Haversine
def haversine(lat1, lon1, lat2, lon2):
    """
    Hitung jarak antara dua titik koordinat (latitude, longitude) dalam meter.
    Menggunakan rumus Haversine.
    """
    R = 6371000  # radius bumi dalam meter
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    distance = R * c
    return distance

def is_within_allowed_location(user_lat, user_lon, allowed_lat, allowed_lon, max_radius_meters):
    """
    Cek apakah koordinat pengguna berada dalam radius yang diizinkan.
    """
    if user_lat is None or user_lon is None:
        return False
    distance = haversine(user_lat, user_lon, allowed_lat, allowed_lon)
    return distance <= max_radius_meters