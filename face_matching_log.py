import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from insightface.app import FaceAnalysis
from datetime import datetime, timedelta

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

def detect_blink(landmarks, w, h, thresh=0.25):
    left = [(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE]
    right = [(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE]
    ear_left = eye_aspect_ratio(left)
    ear_right = eye_aspect_ratio(right)
    ear_avg = (ear_left + ear_right) / 2.0
    if ear_avg < thresh:
        print(f"Blink detected! EAR={ear_avg:.3f}")
        return True
    return False


app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load face database (embedding dict)
data = np.load("faces_db_avg.npz", allow_pickle=True)
face_db = {person: data[person] for person in data.files}

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def recognize_face(face):
    emb = face.normed_embedding.flatten()
    best_match, best_score = "Unknown", -1
    for name, db_emb in face_db.items():
        score = cosine_similarity(emb, db_emb.flatten())
        if score > best_score:
            best_score = score
            best_match = name
    if best_score > 0.45:
        return best_match
    return "Unknown"

# ---------------------------
# Main Loop
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

blink_times = []  # store timestamp of each blink
BLINK_WINDOW = timedelta(seconds=3)  # 3-second window
MIN_BLINKS = 2  # require at least 2 blinks in the window

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
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            if detect_blink(face_landmarks.landmark, w, h):
                blink_times.append(now)

    # Remove old blinks outside the window
    blink_times = [t for t in blink_times if now - t <= BLINK_WINDOW]
    blink_detected = len(blink_times) >= MIN_BLINKS

    # --- Face Recognition ---
    faces = app.get(rgb)
    for face in faces:
        name = recognize_face(face)
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(frame, name, (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # --- Combined Decision (blink count in window) ---
    if blink_detected:
        cv2.putText(frame, "LIVE USER", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "SUSPECT", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Face + Blink Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
