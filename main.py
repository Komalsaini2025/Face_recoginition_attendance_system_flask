from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
import pickle
import os
from datetime import datetime
import sys
import os
import threading
import time

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

        # Camera control globals
        camera_thread = None
        camera_running = False
        # track last attendance mark time per person to avoid duplicates (seconds)
        last_marked = {}
        MARK_DEBOUNCE_SECONDS = 60
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


def _mark_attendance_for_name(name):
    """Create an Attendance DB entry for `name` if debounce allows."""
    now = datetime.now()
    last = last_marked.get(name)
    if last and (now - last).total_seconds() < MARK_DEBOUNCE_SECONDS:
        return False

    session = SessionLocal()
    try:
        entry = Attendance(name=name, date=now.date(), time=now.time())
        session.add(entry)
        session.commit()
        last_marked[name] = now
        speak(f"Attendance marked for {name}")
        return True
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def camera_loop():
    """Background loop that captures frames from default camera and recognizes faces."""
    global camera_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        camera_running = False
        return

    try:
        while camera_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                face_locations = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, face_locations)
            except Exception:
                encodings = []

            for encoding in encodings:
                if len(known_encodings) == 0:
                    continue

                distances = face_recognition.face_distance(known_encodings, encoding)
                best = np.argmin(distances)
                is_match = False
                try:
                    is_match = face_recognition.compare_faces([known_encodings[best]], encoding, tolerance=0.45)[0]
                except Exception:
                    is_match = False

                if is_match and distances[best] < 0.45:
                    name = known_names[best]
                    try:
                        _mark_attendance_for_name(name)
                    except Exception:
                        # ignore DB errors in background loop
                        pass

            # small sleep to reduce CPU usage
            time.sleep(0.2)
    finally:
        try:
            cap.release()
        except Exception:
            pass


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


# Start camera background processing
@app.route("/start_camera", methods=["POST", "GET"])
def start_camera():
    global camera_thread, camera_running
    if camera_running:
        return jsonify({"status": "already_running"})

    camera_running = True
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    return jsonify({"status": "started"})


@app.route("/stop_camera", methods=["POST", "GET"])
def stop_camera():
    global camera_thread, camera_running
    if not camera_running:
        return jsonify({"status": "not_running"})

    camera_running = False
    # thread is daemon; give it a moment to stop
    time.sleep(0.5)
    camera_thread = None
    return jsonify({"status": "stopped"})


@app.route("/camera_status", methods=["GET"])
def camera_status():
    return jsonify({
        "running": bool(camera_running),
        "tracked_people": list(last_marked.keys())
    })


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
