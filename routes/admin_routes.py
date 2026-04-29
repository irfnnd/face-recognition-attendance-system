# admin_routes.py
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
from extensions import db
from models import User, Attendance
from utils import load_encodings_from_db
from config import ADMIN_USERNAME, ADMIN_PASSWORD
import base64
import cv2
import numpy as np
import face_recognition
import os
from config import DATASET_FOLDER

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Decorator login untuk admin
def admin_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin.login'))
        return f(*args, **kwargs)
    return decorated_function

# Halaman login admin
@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin.dashboard'))
        else:
            return render_template('login.html', error='Username atau password salah')
    return render_template('login.html')

# Logout admin
@admin_bp.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('main.index'))

# Dashboard admin (tampilkan semua user dan log)
@admin_bp.route('/')
@admin_login_required
def dashboard():
    users = User.query.all()
    # Ambil log absen dengan join User, urutkan terbaru
    logs = db.session.query(Attendance, User).join(User, User.user_id == Attendance.user_id).order_by(Attendance.timestamp.desc()).all()
    return render_template('admin.html', users=users, logs=logs)

# Registrasi user baru (via admin)
@admin_bp.route('/register', methods=['POST'])
@admin_login_required
def register():
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
    load_encodings_from_db()  # refresh global encodings
    return jsonify({"success": True})

# Edit data user (nama)
@admin_bp.route('/user/<user_id>/edit', methods=['POST'])
@admin_login_required
def edit_user(user_id):
    data = request.get_json()
    new_name = data.get('name')
    user = User.query.filter_by(user_id=user_id).first()
    if not user:
        return jsonify({"success": False, "error": "User tidak ditemukan"})
    user.name = new_name
    db.session.commit()
    return jsonify({"success": True})

# Hapus user (dan semua absensi terkait, serta file gambar)
@admin_bp.route('/user/<user_id>/delete', methods=['DELETE'])
@admin_login_required
def delete_user(user_id):
    user = User.query.filter_by(user_id=user_id).first()
    if not user:
        return jsonify({"success": False, "error": "User tidak ditemukan"})
    # Hapus semua absensi user
    Attendance.query.filter_by(user_id=user_id).delete()
    # Hapus file gambar
    user_folder = os.path.join(DATASET_FOLDER, user_id)
    if os.path.exists(user_folder):
        import shutil
        shutil.rmtree(user_folder)
    # Hapus user
    db.session.delete(user)
    db.session.commit()
    load_encodings_from_db()  # refresh
    return jsonify({"success": True})

# Hapus satu record absen
@admin_bp.route('/attendance/<int:attendance_id>/delete', methods=['DELETE'])
@admin_login_required
def delete_attendance(attendance_id):
    att = Attendance.query.get(attendance_id)
    if not att:
        return jsonify({"success": False, "error": "Record tidak ditemukan"})
    db.session.delete(att)
    db.session.commit()
    return jsonify({"success": True})

# Filter absen berdasarkan user_id dan tanggal (opsional)
@admin_bp.route('/attendance/filter', methods=['GET'])
@admin_login_required
def filter_attendance():
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    query = db.session.query(Attendance, User).join(User, User.user_id == Attendance.user_id)
    if user_id:
        query = query.filter(Attendance.user_id == user_id)
    if start_date:
        query = query.filter(Attendance.timestamp >= start_date)
    if end_date:
        query = query.filter(Attendance.timestamp <= end_date + ' 23:59:59')
    logs = query.order_by(Attendance.timestamp.desc()).all()
    # Untuk keperluan AJAX, kita kembalikan JSON
    data = []
    for att, user in logs:
        data.append({
            'id': att.id,
            'user_id': att.user_id,
            'name': user.name,
            'timestamp': att.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'type': att.attendance_type,
            'latitude': att.latitude,
            'longitude': att.longitude
        })
    return jsonify(data)