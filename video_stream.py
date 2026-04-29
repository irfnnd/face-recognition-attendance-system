import cv2
import time
import numpy as np
import face_recognition
import random
import dlib
from config import EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES
from utils import detector, predictor, known_face_encodings, known_face_names, eye_aspect_ratio
import utils
from extensions import status_queue
from config import LEFT_EYE_START, LEFT_EYE_END, RIGHT_EYE_START, RIGHT_EYE_END
from control import consume_scan_request

# Parameter liveness
TIME_WINDOW = 4.0          # Waktu maksimal melakukan kedipan
PREPARE_TIME = 1.0         # Waktu persiapan mata terbuka sebelum tantangan
MAX_BLINK_FRAMES = 10      # Maksimal frame untuk satu kedipan
FACE_LOST_THRESHOLD = 5     # Frame wajah hilang sebelum gagal
VERIFIED_TIMEOUT = 10.0     # Timeout state VERIFIED jika tidak ada absen

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka kamera")
        return

    STATE = "IDLE"          # IDLE, RECOGNIZE, LIVENESS, VERIFIED, FAILED
    recognized_name = None
    frame_count = 0

    # Variabel liveness
    challenge_blinks = 0
    blink_counter = 0
    consecutive_frames = 0
    challenge_started = False
    challenge_time = 0
    eyes_open_start = None
    face_lost_counter = 0
    failed_time = 0
    verified_start_time = 0

    capture_frame = None
    face_locations = []

    # Untuk throttling pengiriman event (agar tidak overload)
    last_prepare_event = 0
    last_countdown_event = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        now = time.time()

        # ===================== IDLE =====================
        if STATE == "IDLE":
            # Deteksi wajah ringan (setiap 2 frame)
            if frame_count % 2 == 0:
                small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small)
                face_locations = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in face_locations]

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

            if consume_scan_request():
                if len(face_locations) > 0:
                    capture_frame = frame.copy()
                    STATE = "RECOGNIZE"
                    status_queue.put({"status": "RECOGNIZE_START", "message": "Mengenali wajah..."})
                else:
                    status_queue.put({"status": "ERROR", "message": "Tidak ada wajah terdeteksi. Ulangi scan."})

        # ===================== RECOGNIZE =====================
        elif STATE == "RECOGNIZE":
            if capture_frame is not None:
                rgb = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2RGB)
                face_locs = face_recognition.face_locations(rgb)
                if len(face_locs) == 0:
                    STATE = "IDLE"
                    status_queue.put({"status": "ERROR", "message": "Tidak ada wajah terdeteksi saat pengenalan."})
                else:
                    encodings = face_recognition.face_encodings(rgb, face_locs)
                    if len(encodings) == 0:
                        STATE = "IDLE"
                        status_queue.put({"status": "ERROR", "message": "Gagal mengekstrak ciri wajah."})
                    else:
                        matches = face_recognition.compare_faces(utils.known_face_encodings, encodings[0], tolerance=0.5)
                        if True in matches:
                            idx = matches.index(True)
                            recognized_name = utils.known_face_names[idx]
                            STATE = "LIVENESS"
                            challenge_blinks = random.randint(1, 2)
                            blink_counter = 0
                            consecutive_frames = 0
                            challenge_started = False
                            eyes_open_start = None
                            face_lost_counter = 0
                            failed_time = 0
                            status_queue.put({"status": "LIVENESS_START", "message": f"Silakan kedip {challenge_blinks} kali"})
                        else:
                            STATE = "IDLE"
                            status_queue.put({"status": "ERROR", "message": "Wajah tidak dikenal. Silakan daftar atau scan ulang."})
            else:
                STATE = "IDLE"
                status_queue.put({"status": "ERROR", "message": "Gagal mengambil gambar."})

        # ===================== LIVENESS =====================
        elif STATE == "LIVENESS":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Deteksi wajah dengan dlib (setiap 2 frame)
            if frame_count % 2 == 0:
                small_gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
                rects_small = detector(small_gray, 0)
                rects = []
                for r in rects_small:
                    rects.append(dlib.rectangle(
                        int(r.left()*2), int(r.top()*2),
                        int(r.right()*2), int(r.bottom()*2)
                    ))
            else:
                rects = []

            if len(rects) == 0:
                face_lost_counter += 1
                if face_lost_counter >= FACE_LOST_THRESHOLD:
                    STATE = "IDLE"
                    status_queue.put({"status": "FAILED", "message": "Wajah hilang. Scan ulang."})
                    continue
            else:
                face_lost_counter = 0

            for rect in rects:
                shape = predictor(gray, rect)
                shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                leftEye = shape[LEFT_EYE_START:LEFT_EYE_END]
                rightEye = shape[RIGHT_EYE_START:RIGHT_EYE_END]
                ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

                # ---------- TAHAP PERSIAPAN ----------
                if not challenge_started:
                    # Kirim event persiapan (maks 2x per detik)
                    if now - last_prepare_event > 0.5:
                        status_queue.put({
                            "status": "PREPARE_LIVENESS",
                            "message": "Tatap kamera dan buka mata lebar-lebar..."
                        })
                        last_prepare_event = now

                    if ear > EYE_AR_THRESH:
                        if eyes_open_start is None:
                            eyes_open_start = now
                        elif now - eyes_open_start >= PREPARE_TIME:
                            challenge_started = True
                            challenge_time = now
                            blink_counter = 0
                            consecutive_frames = 0
                            status_queue.put({
                                "status": "LIVENESS_PROGRESS",
                                "message": f"Mulai! Kedip {challenge_blinks} kali dalam {TIME_WINDOW} detik"
                            })
                    else:
                        eyes_open_start = None

                    # Timeout persiapan (5 detik)
                    if eyes_open_start and (now - eyes_open_start) > 5.0:
                        STATE = "FAILED"
                        break

                # ---------- TAHAP TANTANGAN KEDIP ----------
                else:
                    remaining = TIME_WINDOW - (now - challenge_time)
                    if remaining <= 0:
                        STATE = "FAILED"
                        break

                    # Kirim countdown setiap 0.3 detik
                    if now - last_countdown_event > 0.3:
                        status_queue.put({
                            "status": "LIVENESS_COUNTDOWN",
                            "remaining": round(remaining, 1),
                            "message": f"Waktu tersisa {remaining:.1f} detik"
                        })
                        last_countdown_event = now

                    # Deteksi kedip
                    if ear < EYE_AR_THRESH:
                        consecutive_frames += 1
                    else:
                        if consecutive_frames >= EYE_AR_CONSEC_FRAMES:
                            if consecutive_frames <= MAX_BLINK_FRAMES:
                                blink_counter += 1
                                status_queue.put({
                                    "status": "BLINK_DETECTED",
                                    "message": f"Kedip {blink_counter}/{challenge_blinks}"
                                })
                            else:
                                STATE = "FAILED"
                                break
                        consecutive_frames = 0

                    # Cek sukses
                    if blink_counter >= challenge_blinks:
                        if ear > EYE_AR_THRESH:
                            STATE = "VERIFIED"
                            status_queue.put({
                                "status": "VERIFIED",
                                "user_id": recognized_name,
                                "message": f"Selamat datang {recognized_name}"
                            })
                            verified_start_time = now
                            break

                # Gambar bounding box kuning selama liveness
                cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0,255,255), 2)

            if STATE == "FAILED":
                status_queue.put({"status": "FAILED", "message": "Liveness gagal. Ulangi scan."})

        # ===================== VERIFIED =====================
        elif STATE == "VERIFIED":
            if now - verified_start_time > VERIFIED_TIMEOUT:
                STATE = "IDLE"
                status_queue.put({"status": "IDLE", "message": "Sesi kadaluwarsa. Scan ulang."})
                recognized_name = None
            # Opsional: gambar bounding box hijau
            if frame_count % 2 == 0:
                small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                face_locs = face_recognition.face_locations(rgb_small)
                for (top, right, bottom, left) in face_locs:
                    top*=2; right*=2; bottom*=2; left*=2
                    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        # ===================== FAILED =====================
        elif STATE == "FAILED":
            if failed_time == 0:
                failed_time = now
            if now - failed_time > 2.0:
                STATE = "IDLE"
                failed_time = 0
                status_queue.put({"status": "IDLE", "message": "Siap untuk scan baru"})

        # TIDAK ADA TEKS APAPUN DI VIDEO
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()