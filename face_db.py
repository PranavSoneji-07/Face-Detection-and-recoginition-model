import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Load model
model = FaceAnalysis(name="buffalo_l", root="C:/Users/PRANAV/.insightface")
model.prepare(ctx_id=0, det_size=(640, 640))

# Register a single image
def register_face(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        return None

    faces = model.get(img)
    if not faces:
        print(f"No face detected in {img_path}")
        return None

    return faces[0].embedding.reshape(1, -1)


# Append new embeddings to main DB
def append_to_database(person_folder, save_path="faces_db.npz", avg_save_path="faces_db_avg.npz"):
    face_db = {}

    # Load existing DB if available
    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
        for person in data.files:
            face_db[person] = data[person]
        print(f"Loaded existing DB with {len(face_db)} people")

    # Person name from folder
    person_name = os.path.basename(person_folder.rstrip("/\\"))

    embeddings = []
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        emb = register_face(img_path)
        if emb is not None:
            embeddings.append(emb)

    if embeddings:
        new_embs = np.vstack(embeddings)
        if person_name in face_db:
            face_db[person_name] = np.vstack([face_db[person_name], new_embs])
            print(f"Appended {len(embeddings)} embeddings to {person_name}")
        else:
            face_db[person_name] = new_embs
            print(f"Added new person {person_name} with {len(embeddings)} embeddings")

    # Save updated full DB
    np.savez(save_path, **face_db)
    print(f"Database updated and saved to {save_path}")

    # --- Build Averaged DB ---
    avg_db = {}
    for person in face_db:
        embs = face_db[person]
        avg_emb = np.mean(embs, axis=0, keepdims=True)
        avg_db[person] = avg_emb
        print(f"Averaged {embs.shape[0]} embeddings for {person}")

    np.savez(avg_save_path, **avg_db)
    print(f"Averaged DB updated and saved to {avg_save_path}")


# Example usage
# This will append to DB and also refresh averaged DB
append_to_database("Faces/Pranav_Soneji", "faces_db.npz", "faces_db_avg.npz")
