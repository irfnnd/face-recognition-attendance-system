import cv2
import dlib
import numpy as np
import face_recognition
import os
import time
from queue import Queue

#load dataset

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

#load dlib model

print("[INFO] Memuat model facial landmark...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

#konfigurasi liveness
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
REQUIRED_BLINKS = 2

status_queue = Queue()

#utils
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    STATE = "SEARCHING"
    recognized_name = None

    blink_counter = 0
    consecutive_frames = 0
    frame_count = 0
    process_this_frame = True

    rects = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera gagal")
            break

        frame_count += 1
        status_text = ""

        # PREPROCESS (OPTIMASI)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # STATE: SEARCHING

        if STATE == "SEARCHING":
            status_text = "Mencari wajah..."

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for face_encoding in face_encodings:
                    if len(known_face_encodings) > 0:
                        matches = face_recognition.compare_faces(
                            known_face_encodings, face_encoding, tolerance=0.5
                        )
                        face_distances = face_recognition.face_distance(
                            known_face_encodings, face_encoding
                        )

                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            recognized_name = known_face_names[best_match_index]
                            STATE = "VALIDATING"
                            blink_counter = 0
                            consecutive_frames = 0
                            break

            process_this_frame = not process_this_frame

            # gambar kotak
            for (top, right, bottom, left) in face_locations:
                top *= 4; right *= 4; bottom *= 4; left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (255,255,255), 2)

        # STATE: VALIDATING

        elif STATE == "VALIDATING":
            status_text = f"Halo {recognized_name}, kedip {REQUIRED_BLINKS} kali."

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frame_count % 3 == 0:
                rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

                if ear < EYE_AR_THRESH:
                    consecutive_frames += 1
                else:
                    if consecutive_frames >= EYE_AR_CONSEC_FRAMES:
                        blink_counter += 1
                    consecutive_frames = 0

                if blink_counter >= REQUIRED_BLINKS:
                    STATE = "VERIFIED"

                (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # STATE: VERIFIED

        elif STATE == "VERIFIED":
            status_text = f"{recognized_name} terverifikasi!"
            cv2.putText(frame, "VERIFIED", (100,240),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0), 2)

        # UI TEXT

        cv2.putText(frame, status_text, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # SHOW

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# MAIN

if __name__ == "__main__":
    generate_frames()