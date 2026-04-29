# main_routes.py
from flask import Blueprint, render_template, Response, request, jsonify
import json
from extensions import db, status_queue
from models import Attendance
from video_stream import generate_frames
from control import request_scan
from datetime import datetime
from config import MORNING_START, MORNING_END, AFTERNOON_START, AFTERNOON_END
from utils import is_within_allowed_location
from config import ALLOWED_LOCATION

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

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

@main_bp.route('/start_scan', methods=['POST'])
def start_scan():
    request_scan()
    return jsonify({"success": True, "message": "Scan dimulai"})


# Helper validasi waktu
def is_allowed_time(attendance_type):
    now = datetime.now().strftime("%H:%M")
    if attendance_type == "masuk":
        return MORNING_START <= now <= MORNING_END
    elif attendance_type == "pulang":
        return AFTERNOON_START <= now <= AFTERNOON_END
    return False


@main_bp.route('/log_attendance', methods=['POST'])
def log_attendance():
    data = request.get_json()
    user_id = data.get('user_id')
    attendance_type = data.get('attendance_type')
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    if not is_allowed_time(attendance_type):
        return jsonify({"success": False, "error": "Waktu tidak diizinkan"}), 400

    if latitude and longitude:
        if not is_within_allowed_location(
            latitude,
            longitude,
            ALLOWED_LOCATION["latitude"],
            ALLOWED_LOCATION["longitude"],
            ALLOWED_LOCATION["radius_meters"]
        ):
            return jsonify({"success": False, "error": "Di luar area absen"}), 403

    new_log = Attendance(
        user_id=user_id,
        attendance_type=attendance_type,
        latitude=latitude,
        longitude=longitude
    )
    db.session.add(new_log)
    db.session.commit()

    status_queue.put({"status": "SUCCESS", "message": "Absen berhasil"})
    return jsonify({"success": True})


@main_bp.route('/attendance_history/<user_id>')
def attendance_history(user_id):
    logs = Attendance.query.filter_by(user_id=user_id)\
        .order_by(Attendance.timestamp.desc()).all()

    history = [{
        "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "type": log.attendance_type,
        "latitude": log.latitude,
        "longitude": log.longitude
    } for log in logs]

    return jsonify({"success": True, "history": history})

@main_bp.route('/reset_state', methods=['POST'])
def reset_state():
    from control import reset_system_state
    reset_system_state()   # Fungsi untuk mengubah STATE menjadi IDLE dan hapus recognized_name
    return jsonify({"success": True})