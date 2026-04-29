# config.py
import os

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
DATASET_FOLDER = "dataset"
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

PREPARE_TIME = 1.0          # waktu persiapan (mata terbuka)
TIME_WINDOW = 4.0           # waktu maksimal untuk menyelesaikan kedipan
MAX_BLINK_FRAMES = 10       # maksimal frame tertutup agar tidak dianggap kedipan panjang
FACE_LOST_THRESHOLD = 5     # berapa frame wajah hilang sebelum reset
FAILED_DELAY = 1.0          # jeda sebelum mencoba ulang setelah gagal

# Parameter lama tetap dipertahankan
EYE_AR_THRESH = 0.22        # lebih rendah dari 0.25 untuk sensitivitas
EYE_AR_CONSEC_FRAMES = 2    # minimal frame tertutup untuk dihitung kedipan
REQUIRED_BLINKS = 2         # (tidak dipakai karena pakai random, bisa dihapus)

ALLOWED_LOCATION = {
    "latitude": 0.9386021,   
    "longitude": 100.3808693,
    "radius_meters": 50 
}
# ALLOWED_LOCATION = {
#     "latitude": -0.9448492,   
#     "longitude": 100.3715785,
#     "radius_meters": 50 
# }

MORNING_START = "07:00"
MORNING_END = "09:00"
AFTERNOON_START = "16:00"
AFTERNOON_END = "18:00"