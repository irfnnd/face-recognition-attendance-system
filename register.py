import cv2
import face_recognition
import sqlite3
import pickle

DB_PATH = "attendance.db"

def save_encoding(user_id, name, encoding):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Buat tabel jika belum ada (sederhana)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE,
            name TEXT,
            face_encoding BLOB
        )
    ''')
    try:
        cursor.execute(
            "INSERT INTO user (user_id, name, face_encoding) VALUES (?, ?, ?)",
            (user_id, name, pickle.dumps(encoding))
        )
        conn.commit()
        print(f"[SUCCESS] {name} ({user_id}) tersimpan.")
    except sqlite3.IntegrityError:
        print(f"[ERROR] ID '{user_id}' sudah ada.")
    conn.close()

print("=== REGISTRASI CEPAT ===")
user_id = input("User ID: ").strip()
name = input("Nama: ").strip()
if not user_id or not name:
    exit()

cap = cv2.VideoCapture(0)
print("Arahkan wajah, tekan SPACE (1x) untuk ambil gambar, ESC batal.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Register", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    if key == ord(' '):
        # Proses encoding hanya sekali
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb)
        if not face_locs:
            print("Wajah tidak terdeteksi, coba lagi.")
            continue
        encoding = face_recognition.face_encodings(rgb, face_locs)[0]
        save_encoding(user_id, name, encoding)
        break

cap.release()
cv2.destroyAllWindows()