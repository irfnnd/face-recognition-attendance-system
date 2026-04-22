# video_stream.py
import cv2
import time
import numpy as np
import face_recognition
import random
import dlib
from config import EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, REQUIRED_BLINKS
from face_utils import detector, predictor, known_face_encodings, known_face_names, eye_aspect_ratio
import face_utils
from extensions import status_queue
from config import LEFT_EYE_START, LEFT_EYE_END, RIGHT_EYE_START, RIGHT_EYE_END
from control import consume_scan_request


# Parameter liveness
TIME_WINDOW = 4.0
PREPARE_TIME = 1.0
MAX_BLINK_FRAMES = 10
FACE_LOST_THRESHOLD = 5

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka kamera")
        return

    STATE = "IDLE"          # IDLE, RECOGNIZE, LIVENESS, VERIFIED, FAILED
    recognized_name = None
    frame_count = 0

    # Variabel untuk liveness
    challenge_blinks = 0
    blink_counter = 0
    consecutive_frames = 0
    challenge_started = False
    challenge_time = 0
    eyes_open_start = None
    face_lost_counter = 0
    failed_time = 0

    # Variabel untuk pengenalan (capture frame)
    capture_frame = None

    # Untuk deteksi wajah ringan di IDLE
    face_locations = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        overlay_text = ""

        # ===================== STATE: IDLE =====================
        if STATE == "IDLE":
            overlay_text = "Tekan tombol SCAN untuk memulai"
            # Deteksi wajah ringan untuk menampilkan kotak
            if frame_count % 2 == 0:
                small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small)
                face_locations = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in face_locations]

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

            # Cek scan request
            if consume_scan_request():
                if len(face_locations) > 0:
                    # Ambil frame saat ini untuk pengenalan
                    capture_frame = frame.copy()
                    STATE = "RECOGNIZE"
                    status_queue.put({"status": "RECOGNIZE_START", "message": "Mengenali wajah..."})
                else:
                    status_queue.put({"status": "ERROR", "message": "Tidak ada wajah terdeteksi. Ulangi scan."})

        # ===================== STATE: RECOGNIZE (pengenalan wajah) =====================
        elif STATE == "RECOGNIZE":
            overlay_text = "Mengenali wajah, mohon tunggu..."
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
                        matches = face_recognition.compare_faces(face_utils.known_face_encodings, encodings[0], tolerance=0.5)
                        if True in matches:
                            idx = matches.index(True)
                            recognized_name = face_utils.known_face_names[idx]
                            # Lanjut ke liveness
                            STATE = "LIVENESS"
                            # Reset variabel liveness
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

        # ===================== STATE: LIVENESS (challenge kedipan) =====================
        elif STATE == "LIVENESS":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

                if not challenge_started:
                    overlay_text = f"Persiapan... tatap kamera"
                    if ear > EYE_AR_THRESH:
                        if eyes_open_start is None:
                            eyes_open_start = time.time()
                        elif time.time() - eyes_open_start >= PREPARE_TIME:
                            challenge_started = True
                            challenge_time = time.time()
                            blink_counter = 0
                            consecutive_frames = 0
                    else:
                        eyes_open_start = None
                else:
                    remaining = TIME_WINDOW - (time.time() - challenge_time)
                    overlay_text = f"Kedip {challenge_blinks} kali ({remaining:.1f}s)"
                    if ear < (EYE_AR_THRESH + 0.02):
                        consecutive_frames += 1
                    else:
                        if consecutive_frames >= EYE_AR_CONSEC_FRAMES:
                            if consecutive_frames <= MAX_BLINK_FRAMES:
                                blink_counter += 1
                            else:
                                STATE = "FAILED"
                        consecutive_frames = 0

                    if blink_counter > challenge_blinks:
                        STATE = "FAILED"
                    elif blink_counter == challenge_blinks:
                        if ear > (EYE_AR_THRESH + 0.03):
                            # Liveness sukses
                            STATE = "VERIFIED"
                            status_queue.put({"status": "VERIFIED", "user_id": recognized_name, "message": f"Selamat datang {recognized_name}"})
                            break

                    if time.time() - challenge_time > TIME_WINDOW:
                        STATE = "FAILED"

                cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0,255,255), 2)

            if STATE == "FAILED":
                status_queue.put({"status": "FAILED", "message": "Liveness gagal. Ulangi scan."})

        # ===================== STATE: VERIFIED =====================
        elif STATE == "VERIFIED":
            overlay_text = f"Verifikasi berhasil: {recognized_name}. Silakan pilih absen."
            cv2.putText(frame, "TERVERIFIKASI", (100, 240), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0), 2)

        # ===================== STATE: FAILED =====================
        elif STATE == "FAILED":
            overlay_text = "Proses gagal. Kembali ke mode siaga..."
            if failed_time == 0:
                failed_time = time.time()
            if time.time() - failed_time > 2.0:
                STATE = "IDLE"
                failed_time = 0
                status_queue.put({"status": "IDLE", "message": "Siap untuk scan baru"})

        # Tampilkan teks pada frame
        if overlay_text:
            cv2.putText(frame, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"State: {STATE}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()