import cv2
import face_recognition
import numpy as np

def mulai_deteksi():
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Kamera tidak terdeteksi!")
        return

    print("[INFO] Kamera aktif. Tekan 'q' untuk keluar.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue  # skip frame rusak

        frame_count += 1

        # === FRAME SKIPPING (stabil + ringan) ===
        if frame_count % 1 != 0:
            cv2.imshow("Deteksi Wajah", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # === RESIZE ===
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # === BGR → RGB ===
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # === DETEKSI WAJAH ===
        try:
            face_locations = face_recognition.face_locations(
                rgb_small_frame,
                model="hog"  # ringan & stabil
            )
        except RuntimeError:
            continue  # fail-safe kalau frame aneh

        # === GAMBAR KOTAK ===
        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
            # cv2.putText(
            #     frame,
            #     "Wajah",
            #     (left, top - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.8,
            #     (0, 255, 0),
            #     2
            # )

        cv2.imshow("Deteksi Wajah", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mulai_deteksi()
