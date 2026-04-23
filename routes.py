# routes.py
from flask import Blueprint, render_template, Response, request, jsonify
import json
import base64
import cv2
import numpy as np
import face_recognition
import os
from extensions import db, status_queue
from models import User, Attendance
from utils import load_encodings_from_db
from video_stream import generate_frames
from config import DATASET_FOLDER
from control import request_scan

from utils import is_within_allowed_location
from config import ALLOWED_LOCATION

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/admin')
def admin_dashboard():
    logs_query = db.session.query(Attendance, User).join(User, User.user_id == Attendance.user_id).order_by(Attendance.timestamp.desc()).all()
    return render_template('admin.html', logs=logs_query)

@main_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@main_bp.route('/status_stream')
def status_stream():
    def generate():
        while True:
            status = status_queue.get()
            yield f"data: {json.dumps(status)}\n\n"
    return Response(generate(), mimetype='text/event-stream')

# @main_bp.route('/log_attendance', methods=['POST'])
# def log_attendance_route():
#     data = request.get_json()
#     new_log = Attendance(
#         user_id=data['user_id'],
#         attendance_type=data['attendance_type'],
#         latitude=data['latitude'],
#         longitude=data['longitude']
#     )
#     db.session.add(new_log)
#     db.session.commit()
#     status_queue.put({"status": "SUCCESS", "message": f"Absen {data['attendance_type']} berhasil!"})
#     return jsonify({"success": True})

@main_bp.route('/admin/register', methods=['POST'])
def admin_register():
    data = request.get_json()
    user_id = data.get('user_id')
    name = data.get('name')
    image_data_url = data.get('image_data')

    if not all([user_id, name, image_data_url]):
        return jsonify({"success": False, "error": "Data tidak lengkap."})

    if User.query.filter_by(user_id=user_id).first():
        return jsonify({"success": False, "error": "ID Pengguna sudah terdaftar."})

    try:
        _, encoded = image_data_url.split(",", 1)
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"success": False, "error": "Gambar tidak valid"})

    # Simpan gambar ke folder dataset
    user_folder = os.path.join(DATASET_FOLDER, user_id)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    image_path = os.path.join(user_folder, f"{name.replace(' ', '_')}.jpg")
    cv2.imwrite(image_path, frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        os.remove(image_path)
        return jsonify({"success": False, "error": "Tidak ada wajah terdeteksi"})

    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
    new_user = User(user_id=user_id, name=name, face_encoding=face_encoding)
    db.session.add(new_user)
    db.session.commit()
    load_encodings_from_db()
    return jsonify({"success": True})


@main_bp.route('/start_scan', methods=['POST'])
def start_scan():
    """Memulai proses scan wajah dari frontend"""
    request_scan()
    return jsonify({"success": True, "message": "Scan dimulai"})


@main_bp.route('/log_attendance', methods=['POST'])
def log_attendance_route():
    data = request.get_json()
    
    user_lat = data.get('latitude')
    user_lon = data.get('longitude')
    
    if user_lat is not None and user_lon is not None:
        if not is_within_allowed_location(
            user_lat, 
            user_lon, 
            ALLOWED_LOCATION["latitude"], 
            ALLOWED_LOCATION["longitude"], 
            ALLOWED_LOCATION["radius_meters"]
        ):
            return jsonify({
                "success": False, 
                "error": f"Anda berada di luar radius {ALLOWED_LOCATION['radius_meters']} meter dari lokasi absen"
            }), 403
    
    new_log = Attendance(
        user_id=data['user_id'],
        attendance_type=data['attendance_type'],
        latitude=user_lat,
        longitude=user_lon
    )
    db.session.add(new_log)
    db.session.commit()
    status_queue.put({"status": "SUCCESS", "message": f"Absen {data['attendance_type']} berhasil!"})
    return jsonify({"success": True})