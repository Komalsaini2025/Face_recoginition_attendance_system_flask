from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
import pickle
import os
from datetime import datetime
import sys
import os

# Ensure project root is on sys.path so imports like `config` and `models`
# (which live in the repository root) can be imported when running
# this file from the subfolder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import SessionLocal
from models import Face, Attendance
import pyttsx3

ENCODINGS_FILE = "encodings/face_encodings.pkl"
FACES_FOLDER = "static/faces"

app = Flask(__name__)

os.makedirs("encodings", exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

# Load encodings
if os.path.exists(ENCODINGS_FILE):
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            known_encodings, known_names = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        # Empty or corrupted file — start fresh
        known_encodings = []
        known_names = []
else:
    known_encodings = []
    known_names = []


def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        pass


def save_encodings():
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_encodings, known_names), f)


# ------------------------------------------------------------------------------
# 📌 API: Recognize Face
# ------------------------------------------------------------------------------

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    # Read image
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    session = SessionLocal()

    # Detect faces
    face_locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, face_locations)

    results = []

    for encoding, loc in zip(encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
        distances = face_recognition.face_distance(known_encodings, encoding)

        if len(distances) > 0:
            best = np.argmin(distances)

            if matches[best] and distances[best] < 0.45:
                name = known_names[best]

                # Mark attendance
                now = datetime.now()
                entry = Attendance(name=name, date=now.date(), time=now.time())
                session.add(entry)
                session.commit()

                speak(f"Attendance marked for {name}")

                results.append({
                    "name": name,
                    "status": "recognized"
                })
                continue

        # Unknown → return face crop for registration
        top, right, bottom, left = loc
        crop = frame[top:bottom, left:right]

        filename = f"unknown_{int(datetime.now().timestamp())}.jpg"
        filepath = os.path.join(FACES_FOLDER, filename)
        cv2.imwrite(filepath, crop)

        results.append({
            "name": "Unknown",
            "image": filename,
            "status": "unrecognized"
        })

    session.close()
    return jsonify(results)


# Simple health endpoint so visiting `/` in a browser doesn't return 405.
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "routes": [
            "/recognize (POST)",
            "/register (POST)",
            "/attendance (GET)"
        ]
    })


# Browser-friendly test form for /recognize (so visiting in a browser won't show 405)
@app.route("/recognize", methods=["GET"])
def recognize_form():
    return (
        "<html><body>"
        "<h2>Recognize Face (POST)</h2>"
        "<form method=\"post\" action=\"/recognize\" enctype=\"multipart/form-data\">"
        "Image: <input type=\"file\" name=\"image\" accept=\"image/*\"><br><br>"
        "<input type=\"submit\" value=\"Upload and Recognize\">"
        "</form>"
        "</body></html>"
    )


# Browser-friendly test form for /register
@app.route("/register", methods=["GET"])
def register_form():
    return (
        "<html><body>"
        "<h2>Register Face (POST)</h2>"
        "<form method=\"post\" action=\"/register\" enctype=\"multipart/form-data\">"
        "Name: <input type=\"text\" name=\"name\"><br><br>"
        "Image: <input type=\"file\" name=\"image\" accept=\"image/*\"><br><br>"
        "<input type=\"submit\" value=\"Register\">"
        "</form>"
        "</body></html>"
    )


# ------------------------------------------------------------------------------
# 📌 API: Register New Face
# ------------------------------------------------------------------------------

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    file = request.files.get("image")

    if not name or not file:
        return jsonify({"error": "Name and image required"}), 400

    # Save image
    filepath = os.path.join(FACES_FOLDER, f"{name}.jpg")
    file.save(filepath)

    # Encode
    img = face_recognition.load_image_file(filepath)
    enc = face_recognition.face_encodings(img)

    if not enc:
        return jsonify({"error": "No face found in image"}), 400

    encoding = enc[0]

    # Save to known data
    known_encodings.append(encoding)
    known_names.append(name)
    save_encodings()

    # Save DB
    session = SessionLocal()
    new_face = Face(name=name, image_path=filepath)
    session.add(new_face)
    session.commit()
    session.close()

    return jsonify({"message": f"Face registered for {name}"})


# ------------------------------------------------------------------------------
# 📌 API: View Attendance
# ------------------------------------------------------------------------------

@app.route("/attendance", methods=["GET"])
def attendance():
    session = SessionLocal()
    rows = session.query(Attendance).all()
    data = [{"name": r.name, "date": str(r.date), "time": str(r.time)} for r in rows]
    session.close()
    return jsonify(data)


# ------------------------------------------------------------------------------
# Run Flask
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
