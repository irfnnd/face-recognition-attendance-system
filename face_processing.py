import cv2
import dlib
import numpy as np
import face_recognition
import os
import time
import random

# --- LOAD DATASET ---
known_face_encodings = []
known_face_names = []

dataset_path = "datasets/"

for file_name in os.listdir(dataset_path):
    image = face_recognition.load_image_file(f"{dataset_path}/{file_name}")
    encodings = face_recognition.face_encodings(image)
    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        known_face_names.append(file_name.split(".")[0])

print("[INFO] Dataset loaded:", len(known_face_names))

# --- LOAD MODEL ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# --- PARAMETER ---
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 2
MAX_BLINK_FRAMES = 10

TIME_WINDOW = 4.0
PREPARE_TIME = 1.0

FACE_LOST_THRESHOLD = 5
FAILED_DELAY = 1.0

# --- UTILS ---
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# --- MAIN ---
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    STATE = "SEARCHING"
    recognized_name = None

    blink_counter = 0
    consecutive_frames = 0
    frame_count = 0

    challenge_started = False
    challenge_time = 0
    eyes_open_start = None
    challenge_blinks = random.randint(1, 2)

    rects = []
    face_lost_counter = 0
    failed_time = 0

    process_this_frame = True
    process_landmark = True

    print("[INFO] Kamera dimulai...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        status_text = ""

        # =====================
        # STATE: SEARCHING
        # =====================
        if STATE == "SEARCHING":
            status_text = "Mencari wajah..."

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            recognized_name = known_face_names[best_match_index]
                            STATE = "VALIDATING"

                            # reset
                            blink_counter = 0
                            consecutive_frames = 0
                            challenge_started = False
                            eyes_open_start = None
                            challenge_blinks = random.randint(1, 2)
                            break

            process_this_frame = not process_this_frame

        # STATE: VALIDATING
        
        elif STATE == "VALIDATING":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 🔥 DETECTOR DI-OPTIMASI (RESIZE)
            if frame_count % 3 == 0:
                small_gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
                rects_small = detector(small_gray, 0)

                rects = []
                for r in rects_small:
                    rects.append(dlib.rectangle(
                        int(r.left()*2),
                        int(r.top()*2),
                        int(r.right()*2),
                        int(r.bottom()*2)
                    ))

            # 🔥 FACE LOST COUNTER
            if len(rects) == 0:
                face_lost_counter += 1
            else:
                face_lost_counter = 0

            if face_lost_counter >= FACE_LOST_THRESHOLD:
                STATE = "SEARCHING"
                face_lost_counter = 0
                continue

            for rect in rects:
                if process_landmark:
                    shape = predictor(gray, rect)
                    shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]

                    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

                    # PREPARE

                    if not challenge_started:
                        status_text = f"Halo {recognized_name}, siapkan wajah..."

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

                    # ----------------
                    # CHALLENGE
                    # ----------------
                    else:
                        remaining = TIME_WINDOW - (time.time() - challenge_time)
                        status_text = f"Kedip {challenge_blinks}x ({remaining:.1f}s)"

                        if ear < (EYE_AR_THRESH + 0.02):
                            consecutive_frames += 1
                        else:
                            if consecutive_frames >= EYE_AR_CONSEC_FRAMES:
                                if consecutive_frames <= MAX_BLINK_FRAMES:
                                    blink_counter += 1
                                else:
                                    STATE = "FAILED"
                            consecutive_frames = 0

                        # VALIDASI
                        if blink_counter > challenge_blinks:
                            STATE = "FAILED"

                        elif blink_counter == challenge_blinks:
                            if ear > (EYE_AR_THRESH + 0.03):
                                STATE = "VERIFIED"

                        if time.time() - challenge_time > TIME_WINDOW:
                            STATE = "FAILED"

                (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

            process_landmark = not process_landmark

        # STATE: VERIFIED

        elif STATE == "VERIFIED":
            status_text = "Berhasil diverifikasi!"

        # STATE: FAILED

        elif STATE == "FAILED":
            status_text = f"EAR: {ear:.2f} - Gagal! Mengulang..."

            if failed_time == 0:
                failed_time = time.time()

            if time.time() - failed_time > FAILED_DELAY:
                challenge_started = False
                eyes_open_start = None
                blink_counter = 0
                consecutive_frames = 0
                challenge_blinks = random.randint(1, 2)
                failed_time = 0
                STATE = "VALIDATING"

        # =====================
        # UI TEXT (GLOBAL)
        # =====================
        cv2.putText(frame, status_text, (20,40),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# RUN
if __name__ == "__main__":
    generate_frames()