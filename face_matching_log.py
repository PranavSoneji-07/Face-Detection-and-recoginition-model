import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from insightface.app import FaceAnalysis
from datetime import datetime
import csv
import os

# ---------------------------
# Eye Blink Detection
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_blink(landmarks, w, h, thresh=0.20):
    left = [(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE]
    right = [(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE]
    ear_left = eye_aspect_ratio(left)
    ear_right = eye_aspect_ratio(right)
    ear_avg = (ear_left + ear_right) / 2.0
    return ear_avg < thresh

# ---------------------------
# Load InsightFace
# ---------------------------
app = FaceAnalysis(name="buffalo_l",
                   providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load embeddings DB
data = np.load("faces_db.npz", allow_pickle=True)
face_db = {person: data[person] for person in data.files}

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def recognize_face(face, threshold=0.45):
    emb = face.normed_embedding.flatten()
    best_match, best_score = "Unknown", -1
    for name, db_embs in face_db.items():
        for ref_emb in db_embs:
            score = cosine_similarity(emb, ref_emb.flatten())
            if score > best_score:
                best_score = score
                best_match = name
    if best_score > threshold:
        return best_match
    return "Unknown"

# ---------------------------
# Logging Setup
# ---------------------------
LOG_FILE = "attendance_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Entry Time", "Exit Time", "Duration (minutes)"])

ENTRY_BUFFER = 120  # 2 min
EXIT_BUFFER = 5   # 2 min

active_sessions = {}  # {name: entry_time}
last_seen = {}        # {name: timestamp}
live_users = {}       # {name: True/False}

def log_attendance(name, now):
    # Face detected → mark as live
    live_users[name] = True
    last_seen[name] = now

    # --- ENTRY LOGIC ---
    if name not in active_sessions:
        last_entry = last_seen.get(f"{name}_entry", None)
        if not last_entry or (now - last_entry).total_seconds() > ENTRY_BUFFER:
            active_sessions[name] = now
            last_seen[f"{name}_entry"] = now

            print(f"✅ Entry logged for {name} at {now}")
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, now, "", ""])  # Entry only

    # --- EXIT LOGIC ---
    else:
        entry_time = active_sessions[name]
        # If person not seen for EXIT_BUFFER seconds → exit
        if (now - last_seen[name]).total_seconds() > EXIT_BUFFER:
            exit_time = now
            duration = (exit_time - entry_time).total_seconds() / 60

            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, entry_time, exit_time, round(duration, 2)])

            print(f"🚪 Exit logged for {name} at {exit_time}, duration {duration:.2f} min")

            del active_sessions[name]
            del live_users[name]

# ---------------------------
# Main Loop
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not captured, exiting...")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now = datetime.now()

    # --- Blink Detection ---
    result = face_mesh.process(rgb)
    blink_detected = False
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            if detect_blink(face_landmarks.landmark, w, h):
                blink_detected = True

    # --- Face Recognition ---
    faces = app.get(rgb)
    for face in faces:
        name = recognize_face(face)
        box = face.bbox.astype(int)

        # Draw box
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(frame, name, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if name != "Unknown":
            if blink_detected:  # Only count live users
                log_attendance(name, now)

    # --- Status Overlay ---
    if blink_detected:
        cv2.putText(frame, "LIVE USER", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "SUSPECT", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Door Camera - Entry/Exit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
