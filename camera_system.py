import face_recognition
import cv2
import os
import numpy as np

# ===============================
# 1. LOAD DATASET
# ===============================
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


# ===============================
# 2. DETEKSI + RECOGNITION
# ===============================
def mulai_deteksi():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Inisialisasi variabel untuk menyimpan hasil terakhir
    face_locations = []
    face_names = []
    
    # Counter untuk mengatur kapan harus melakukan encoding berat
    frame_count = 0
    proses_setiap_x_frame = 10 # Encoding hanya jalan setiap 10 frame (sekitar 0.3 detik)

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1

        # 1. Perkecil frame untuk semua proses (biar cepat)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # 2. DETEKSI LOKASI (Selalu jalan agar kotak tidak telat/delay)
        # Mencari "di mana wajah" jauh lebih ringan daripada mencari "siapa ini"
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

        # 3. ENCODING IDENTITAS (Hanya jalan sesekali)
        if frame_count % proses_setiap_x_frame == 0:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                if len(known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < 0.5:
                        name = known_face_names[best_match_index]
                face_names.append(name)

        # 4. GAMBAR HASIL
        for i, (top, right, bottom, left) in enumerate(face_locations):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            
            name = face_names[i] if i < len(face_names) else "Scanning..."
            
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if __name__ == "__main__":
    mulai_deteksi()
