# config.py
import os

from flask import current_app

# Database
SQLALCHEMY_DATABASE_URI = 'sqlite:///attendance.db'  # relatif terhadap instance folder
SQLALCHEMY_TRACK_MODIFICATIONS = False

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
SECRET_KEY = "123"

# Liveness detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
REQUIRED_BLINKS = 2

# Landmark mata (indeks dlib 68 points)
LEFT_EYE_START, LEFT_EYE_END = 42, 48
RIGHT_EYE_START, RIGHT_EYE_END = 36, 42

# Path model dlib (pastikan file ada)
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Folder dataset untuk menyimpan gambar saat registrasi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
DATASET_FOLDER = os.path.join(STATIC_FOLDER, 'dataset')

PREPARE_TIME = 1.0          # waktu persiapan (mata terbuka)
TIME_WINDOW = 4.0           # waktu maksimal untuk menyelesaikan kedipan
MAX_BLINK_FRAMES = 10       # maksimal frame tertutup agar tidak dianggap kedipan panjang
FACE_LOST_THRESHOLD = 5     # berapa frame wajah hilang sebelum reset
FAILED_DELAY = 1.0          # jeda sebelum mencoba ulang setelah gagal

# Parameter lama tetap dipertahankan
EYE_AR_THRESH = 0.22        # lebih rendah dari 0.25 untuk sensitivitas
EYE_AR_CONSEC_FRAMES = 2    # minimal frame tertutup untuk dihitung kedipan
REQUIRED_BLINKS = 2         # (tidak dipakai karena pakai random, bisa dihapus)

# ALLOWED_LOCATION = {
#     "latitude": 0.9386021,   
#     "longitude": 100.3808693,
#     "radius_meters": 50 
# }

#PERPUS
ALLOWED_LOCATION = {
    "latitude": -0.809354,   
    "longitude": 100.373301,
    "radius_meters": 50 
}

#Pecel ayam
# ALLOWED_LOCATION = {
#     "latitude": -0.819843,   
#     "longitude": 100.361887,
#     "radius_meters": 50 
# }

MORNING_START = "07:00"
MORNING_END = "16:00"
AFTERNOON_START = "16:00"
AFTERNOON_END = "18:00"