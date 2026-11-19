"""SmartAttend AI web UI.
Reuses your own code from test_face_recognition.py (Config, MongoDB, Telegram,
FaceRecognizer, AttendanceManager). That file stays untouched.

Run:  python app.py   ->  open http://localhost:5000
Do NOT run test_face_recognition.py at the same time (same webcam, same port).
"""
import os, re, csv, glob, base64, threading
from datetime import datetime
import cv2, numpy as np
from flask import Flask, request, jsonify, send_file
from deepface import DeepFace

import test_face_recognition as core
from test_face_recognition import Config

app = Flask(__name__)
STUDENTS = "students.csv"
lock = threading.Lock()          # guards the recognizer (its cascade + temp file): one scan at a time
Config.VOICE_ENABLED = False     # pyttsx3 can freeze web requests; set True to try it

# FIX for "cv2.error: invalid vector subscript":
# an OpenCV CascadeClassifier is NOT thread-safe, and /api/detect and /api/recognize
# run at the same time. So /api/detect gets its OWN cascade with its OWN lock, and the
# recognizer's cascade is only ever touched while holding `lock`.
detect_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detect_lock = threading.Lock()


class AsyncTelegram:
    """Sends Telegram photos in the background so the page never waits on the network."""
    def __init__(self, t): self.t = t
    def send_photo(self, name, path):
        threading.Thread(target=self.t.send_photo, args=(name, path), daemon=True).start()
    def send_message(self, msg):
        threading.Thread(target=self.t.send_message, args=(msg,), daemon=True).start()


class DashboardStub:
    def add_attendance(self, name, time_str): pass


recognizer = core.FaceRecognizer()
attendance = core.AttendanceManager(core.ESP32Controller(), core.VoiceAnnouncer(),
                                    AsyncTelegram(core.TelegramNotifier()), DashboardStub())


def decode(data_url):
    raw = base64.b64decode(data_url.split(",")[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def students():
    return read_csv(STUDENTS)


def today():
    return datetime.now().strftime("%Y-%m-%d")


# Remember who was already marked today, even after a restart
attendance.marked_today.update(r["Name"] for r in read_csv(Config.ATTENDANCE_FILE)
                               if r.get("Date") == today())


@app.errorhandler(Exception)
def on_error(e):
    import traceback
    traceback.print_exc()  # full details appear in this terminal
    code = e.code if isinstance(getattr(e, "code", None), int) else 500
    return jsonify(error=f"{type(e).__name__}: {e}"), code


@app.get("/")
def home():
    return send_file("index.html")


@app.get("/api/students")
def list_students():
    return jsonify(students())


@app.post("/api/enroll")
def enroll():
    d = request.json
    sid, section = d["id"].strip(), d["section"].strip()
    name = re.sub(r'[\\/:*?"<>|]', "", d["name"]).strip()
    if not (sid and name and section and d.get("images")):
        return jsonify(error="ID, name, section and photos are required."), 400
    rows = [s for s in students() if s["id"] != sid] + [{"id": sid, "name": name, "section": section}]
    with open(STUDENTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "section"])
        w.writeheader()
        w.writerows(rows)
    for i, img in enumerate(d["images"]):
        suffix = "" if i == 0 else f"_{i + 1}"
        cv2.imwrite(os.path.join(Config.FACES_DB, f"{name}{suffix}.jpg"), decode(img))
    try:  # also store the student in your MongoDB
        core.client[Config.MONGO_DB_NAME]["students"].update_one(
            {"id": sid}, {"$set": {"id": sid, "name": name, "section": section}}, upsert=True)
    except Exception as e:
        print(f"MongoDB student save failed: {e}")
    return jsonify(ok=True)


@app.post("/api/train")
def train():
    for p in glob.glob(os.path.join(Config.FACES_DB, "ds_model_*.pkl")):
        os.remove(p)  # DeepFace rebuilds this cache on the next search
    with lock:
        DeepFace.find(img_path=np.zeros((160, 160, 3), np.uint8), db_path=Config.FACES_DB,
                      model_name=Config.MODEL_NAME, enforce_detection=False, silent=True)
    return jsonify(ok=True)


@app.post("/api/detect")
def detect():
    """Fast face finder (no recognition) so the page can draw a live face box."""
    frame = decode(request.json["image"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    with detect_lock:  # separate cascade + lock, never shared with the recognizer
        found = detect_cascade.detectMultiScale(gray, 1.3, 5)
    boxes = sorted(([int(x), int(y), int(w), int(h)] for x, y, w, h in found),
                   key=lambda b: -b[2] * b[3])
    return jsonify(boxes=boxes, size=[frame.shape[1], frame.shape[0]])


@app.post("/api/recognize")
def recognize():
    frame = decode(request.json["image"])
    with lock:  # detect_faces + recognize_face both use the recognizer's cascade
        faces = recognizer.detect_faces(frame)
        if len(faces) == 0:
            return jsonify(found=False, face=False)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        name, conf, emotion = recognizer.recognize_face(frame)
        if not name:
            return jsonify(found=False, face=True)
        name = re.sub(r"_\d+$", "", name)  # "kenneth crismas_2" -> "kenneth crismas"
        student = next((s for s in students() if s["name"] == name), {"id": "", "name": name})
        new = attendance.mark_attendance(name, Config.LOCATION, frame)
    return jsonify(found=True, face=True, box=[int(x), int(y), int(w), int(h)],
                   student=student, new=new, confidence=round(conf, 1), emotion=emotion)


@app.post("/api/manual")
def manual():
    d = request.json
    s = next((s for s in students() if s["id"] == d["id"]), None)
    if not s:
        return jsonify(error="Student not found."), 404
    frame = decode(d["image"]) if d.get("image") else np.zeros((240, 320, 3), np.uint8)
    with lock:
        new = attendance.mark_attendance(s["name"], Config.LOCATION, frame)
    return jsonify(ok=True, new=new)


@app.get("/api/attendance")
def today_list():
    sec = {s["name"]: s["section"] for s in students()}
    seen = {}
    for r in read_csv(Config.ATTENDANCE_FILE):
        if r.get("Date") == today() and r["Name"] in attendance.marked_today:
            seen[r["Name"]] = {"time": r["Time"], "name": r["Name"], "section": sec.get(r["Name"], "")}
    return jsonify(list(seen.values()))


@app.post("/api/reset")
def reset():
    attendance.reset()  # same as pressing R in your original app
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)