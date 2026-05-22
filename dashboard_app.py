import os
import cv2
import threading
import time
import sqlite3
import requests
import json
from collections import deque
from ultralytics import YOLO
from fastapi import FastAPI, responses, Form, Request
import uvicorn
from contextlib import asynccontextmanager

# Local runtime path variables
DB_FILE = "v380_analytics.db"
AI_DETECTION_INTERVAL = 0.2

# Extract hardcoded AI profiles into cleaner global constants
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

raw_device = os.getenv("YOLO_DEVICE", "cpu").lower().strip()
if raw_device == "gpu":
    YOLO_DEVICE = "0"
else:
    YOLO_DEVICE = raw_device

RTSP_URL = os.getenv("RTSP_URL")
if not RTSP_URL:
    raise ValueError("[❌] Critical Error: RTSP_URL environment variable is missing! Please export it before running.")

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_URL = f"{OLLAMA_BASE.rstrip('/')}/api/chat"

class LowLatencyRingBuffer:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.buffer = deque(maxlen=1)
        self.ret = False
        self.running = True

    def start(self):
        threading.Thread(target=self._capture_worker, daemon=True).start()
        return self

    def _capture_worker(self):
        consecutive_failures = 0
        while self.running:
            if not self.cap.isOpened():
                sleep_time = min(2 ** consecutive_failures, 60)
                print(f"[⚠️] Camera connection down. Attempting re-initialization in {sleep_time} seconds...")
                time.sleep(sleep_time)
                consecutive_failures += 1
                self.cap.open(self.src)
                continue
                
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.buffer.append(frame)
                consecutive_failures = 0  
            else:
                consecutive_failures += 1
                sleep_time = min(2 ** consecutive_failures, 60)
                print(f"[⚠️] Frame drop or read failure ({consecutive_failures}). Cool-down for {sleep_time}s before link reset...")
                self.cap.release()
                time.sleep(sleep_time)
                self.cap.open(self.src)

    def get_frame(self):
        if self.ret and len(self.buffer) > 0:
            return True, self.buffer[0].copy()
        return False, None

def init_db():
    with sqlite3.connect(DB_FILE, timeout=15.0) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now', 'localtime')),
                    target_class TEXT,
                    action_defined TEXT,
                    peak_confidence REAL,
                    duration_secs REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_target ON security_logs(target_class)")
            cursor.execute("DELETE FROM security_logs WHERE timestamp < datetime('now', '-30 days');")
            print("[🗄️] Database initialized and data older than 30 days pruned successfully.")
            conn.commit()
        finally:
            cursor.close()

def log_event_to_db(target, action, confidence, duration):
    with sqlite3.connect(DB_FILE, timeout=15.0) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("""
                INSERT INTO security_logs (target_class, action_defined, peak_confidence, duration_secs)
                VALUES (?, ?, ?, ?)
            """, (target, action, round(confidence, 2), round(duration, 1)))
            conn.commit()
        finally:
            cursor.close()

def query_db(sql, params=()):
    with sqlite3.connect(DB_FILE, timeout=15.0) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute(sql, params)
            return cursor.fetchall()
        finally:
            cursor.close()

class AdvancedSpatialTracker:
    def __init__(self, target_name):
        self.target_name = target_name
        self.active_tracks = {}
        self.lock = threading.Lock()
        self.id_lost_grace_period = 2.0  

    def process_frame_tracks(self, current_frame_detections):
        now = time.time()
        
        with self.lock:
            for det in current_frame_detections:
                tid = det["track_id"]
                conf = det["conf"]
                x1, y1, x2, y2 = det["box"]
                
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + (w / 2), y1 + (h / 2)
                aspect_ratio = w / float(h) if h > 0 else 0

                if tid not in self.active_tracks:
                    self.active_tracks[tid] = {
                        "start_time": now,
                        "last_seen": now,
                        "peak_confidence": conf,
                        "current_action": "Calibrating",
                        "last_state_change": now,
                        "history": deque(maxlen=15)
                    }

                track = self.active_tracks[tid]
                track["last_seen"] = now
                track["peak_confidence"] = max(track["peak_confidence"], conf)
                track["history"].append({"cx": cx, "cy": cy, "ar": aspect_ratio, "h": h})

                if len(track["history"]) == track["history"].maxlen and (now - track["last_state_change"] > 1.5):
                    proposed_action = track["current_action"]
                    
                    if self.target_name == "Human":
                        avg_ar = sum(f["ar"] for f in track["history"]) / len(track["history"])
                        dx = abs(track["history"][-1]["cx"] - track["history"][0]["cx"]) / h
                        dy = abs(track["history"][-1]["cy"] - track["history"][0]["cy"]) / h
                        total_velocity = dx + dy

                        if total_velocity > 0.18:
                            proposed_action = "Standing/Moving"
                        elif avg_ar > 0.72:
                            proposed_action = "Sitting/Working"
                        else:
                            proposed_action = "Stationary/Standing"
                            
                    elif self.target_name == "Doggo":
                        dx = abs(track["history"][-1]["cx"] - track["history"][0]["cx"])
                        dy = abs(track["history"][-1]["cy"] - track["history"][0]["cy"])
                        total_velocity = dx + dy

                        if total_velocity > 40:
                            proposed_action = "Pacing/Moving"
                        else:
                            proposed_action = "Bedded/Sleeping"

                    if proposed_action != track["current_action"]:
                        track["current_action"] = proposed_action
                        track["last_state_change"] = now

            stale_ids = [tid for tid, t_meta in self.active_tracks.items() if now - t_meta["last_seen"] > self.id_lost_grace_period]
            for tid in stale_ids:
                t_meta = self.active_tracks[tid]
                duration = t_meta["last_seen"] - t_meta["start_time"]
                
                if duration > 1.5 and t_meta["current_action"] != "Calibrating":
                    log_event_to_db(self.target_name, t_meta["current_action"], t_meta["peak_confidence"], duration)
                
                del self.active_tracks[tid]

    def get_current_action(self, track_id):
        with self.lock:
            if track_id in self.active_tracks:
                return self.active_tracks[track_id]["current_action"]
            return "Calibrating"

stream_bridge = None
yolo_model = None
human_tracker = AdvancedSpatialTracker("Human")
dog_tracker = AdvancedSpatialTracker("Doggo")
latest_processed_frame = None
frame_lock = threading.Lock()

def sanitize_telemetry_logs(rows):
    translation_layer = {
        "Calibrating": "entering or settling into the room briefly",
        "Sitting/Working": "sitting down at the desk working",
        "Standing/Moving": "standing or walking around the room",
        "Stationary/Standing": "standing completely still inside the room",
        "Pacing/Moving": "actively pacing around the room",
        "Bedded/Sleeping": "lying down on the floor resting or sleeping"
    }

    if not rows:
        return "The camera tracking logs are currently empty."

    compressed_events = []
    for r in reversed(rows):
        timestamp, target, action, duration = r
        friendly_action = translation_layer.get(action, action.lower())
        
        if compressed_events and compressed_events[-1]['target'] == target and compressed_events[-1]['friendly_action'] == friendly_action:
            compressed_events[-1]['duration'] += duration
        else:
            compressed_events.append({
                "timestamp": timestamp,
                "target": "human" if target == "Human" else "dog",
                "friendly_action": friendly_action,
                "duration": duration
            })

    clean_text = ""
    for event in compressed_events:
        clean_text += f"- At {event['timestamp']}, a {event['target']} was detected {event['friendly_action']} for approximately {round(event['duration'], 1)} total seconds.\n"
    
    return clean_text

def background_ai_inference_worker():
    global stream_bridge, yolo_model, human_tracker, dog_tracker, latest_processed_frame

    while True:
        t_start = time.time()
        success, frame = stream_bridge.get_frame()
        if not success or frame is None:
            time.sleep(0.01)
            continue

        results = yolo_model.track(frame, persist=True, device=YOLO_DEVICE, verbose=False, imgsz=640)[0]

        human_detections_this_frame = []
        dog_detections_this_frame = []

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id in [0, 16] and conf > 0.60 and box.id is not None:
                    is_human = (cls_id == 0)
                    track_id = int(box.id[0])
                    coords = list(map(int, box.xyxy[0]))
                    x1, y1, x2, y2 = coords

                    det_payload = {"track_id": track_id, "conf": conf, "box": (x1, y1, x2, y2)}
                    
                    if is_human:
                        human_detections_this_frame.append(det_payload)
                    else:
                        dog_detections_this_frame.append(det_payload)

                    tracker_instance = human_tracker if is_human else dog_tracker
                    act_label = tracker_instance.get_current_action(track_id)
                    
                    color = (0, 0, 255) if is_human else (0, 255, 0)
                    lbl = f"{'Human' if is_human else 'Doggo'} #{track_id} ({act_label}): {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        human_tracker.process_frame_tracks(human_detections_this_frame)
        dog_tracker.process_frame_tracks(dog_detections_this_frame)

        system_active = len(human_detections_this_frame) > 0 or len(dog_detections_this_frame) > 0
        txt = "SYSTEM MONITOR: " + ("Tracking Metrics Active" if system_active else "Clear")
        clr = (0, 0, 255) if len(human_detections_this_frame) > 0 else (0, 255, 0) if len(dog_detections_this_frame) > 0 else (255, 165, 0)
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 2)

        with frame_lock:
            latest_processed_frame = frame.copy()

        elapsed = time.time() - t_start
        sleep_t = max(0.001, AI_DETECTION_INTERVAL - elapsed)
        time.sleep(sleep_t)

def sync_ollama_models():
    base_url = OLLAMA_BASE.rstrip('/')
    max_retries = 6
    retry_delay = 5
    connected = False

    print(f"[🤖] Initiating pre-flight verification on Ollama host: {base_url}")
    
    for attempt in range(1, max_retries + 1):
        try:
            requests.get(base_url, timeout=3)
            connected = True
            print(f"[🤖] Successfully established link to Ollama endpoint on attempt {attempt}.")
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(f"[⏳] Ollama service is initializing... (Attempt {attempt}/{max_retries} failed. Retrying in {retry_delay}s)")
            time.sleep(retry_delay)

    if not connected:
        print("[❌] Pre-flight Check Failed: Ollama host could not be reached. Automated pulling skipped.")
        return

    try:
        pull_url = f"{base_url}/api/pull"
        print(f"[🤖] Connection active. Synchronizing {OLLAMA_MODEL} weights...")
        # Replaced hardcoded string with dynamic context variable configuration
        requests.post(pull_url, json={"name": OLLAMA_MODEL}, timeout=None)
        print("[✅] Local model synchronization complete.")
    except Exception as e:
        print(f"[⚠️] Unexpected engine alert encountered during model initialization: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stream_bridge, yolo_model, YOLO_DEVICE
    init_db()
    
    threading.Thread(target=sync_ollama_models, daemon=True).start()

    print(f"[🚀] Validating YOLO tracking inference hardware device target: {YOLO_DEVICE}")
    yolo_model = YOLO("yolo11n_openvino_model/", task="detect")
    
    try:
        dummy_frame = cv2.Mat.zeros(height=640, width=640, type=cv2.CV_8UC3)
        yolo_model.track(dummy_frame, persist=True, device=YOLO_DEVICE, verbose=False, imgsz=640)
    except Exception as e:
        print(f"[⚠️] Pre-flight Check: Execution failed for target ({YOLO_DEVICE}). Enforcing fallback to CPU runtime engine.")
        YOLO_DEVICE = "cpu"

    stream_bridge = LowLatencyRingBuffer(RTSP_URL).start()
    threading.Thread(target=background_ai_inference_worker, daemon=True).start()
    print("[🚀] Headless AI Geometric Action Matrix Engaged Flawlessly.")
    yield
    if stream_bridge:
        stream_bridge.running = False

app = FastAPI(title="V380 Pro AI Analytics Center", lifespan=lifespan)

@app.get("/video_feed")
def video_feed_endpoint():
    return responses.StreamingResponse(generate_live_web_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def generate_live_web_frames():
    global stream_bridge, latest_processed_frame
    while True:
        with frame_lock:
            if latest_processed_frame is not None:
                display_frame = latest_processed_frame.copy()
            else:
                success, display_frame = stream_bridge.get_frame()
                if not success or display_frame is None:
                    time.sleep(0.01)
                    continue

        ret, jpeg_buffer = cv2.imencode('.jpg', display_frame)
        if not ret:
            time.sleep(0.01)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.get("/api/chart_data")
def get_chart_data_endpoint():
    hourly_query = query_db("""
        SELECT strftime('%H', timestamp) as hr, COUNT(*)
        FROM security_logs GROUP BY hr ORDER BY hr ASC
    """)
    composition_query = query_db("SELECT target_class, COUNT(*) FROM security_logs GROUP BY target_class")

    hourly_dict = {f"{i:02d}:00": 0 for i in range(24)}
    for row in hourly_query:
        hourly_dict[f"{row[0]}:00"] = row[1]

    comp_dict = {"Human": 0, "Doggo": 0}
    for row in composition_query:
        if row[0] == "Human": comp_dict["Human"] = row[1]
        elif row[0] == "Doggo": comp_dict["Doggo"] = row[1]

    # Added summary and recent rows endpoints to service the dynamic frontend repainting handler
    summary_data = query_db("""
        SELECT target_class, action_defined, COUNT(*), AVG(duration_secs)
        FROM security_logs GROUP BY target_class, action_defined
    """)
    recent_logs = query_db("""
        SELECT timestamp, target_class, action_defined, peak_confidence, duration_secs
        FROM security_logs ORDER BY id DESC LIMIT 5
    """)

    summary_rows = "".join([
        f"<tr><td>{row[0]}</td><td><span class='badge bg-secondary'>{row[1]}</span></td><td><b>{row[2]}</b></td><td>{row[3]:.1f}s</td></tr>"
        for row in summary_data
    ]) or "<tr><td colspan='4'>No behavioral tracking metrics accumulated yet.</td></tr>"

    recent_rows = "".join([
        f"<tr><td>{row[0]}</td><td><span class='badge {row[1].lower()}'>{row[1]}</span></td><td><small>{row[2]}</small></td><td>{row[3]*100:.0f}%</td><td>{row[4]:.1f}s</td></tr>"
        for row in recent_logs
    ]) or "<tr><td colspan='5'>Awaiting operational sequences...</td></tr>"

    return {
        "hourly_labels": list(hourly_dict.keys()),
        "hourly_values": list(hourly_dict.values()),
        "comp_labels": list(comp_dict.keys()),
        "comp_values": list(comp_dict.values()),
        "summary_rows_html": summary_rows,
        "recent_rows_html": recent_rows
    }

@app.post("/api/chat")
def chatbot_endpoint(history: str = Form(...)):
    try:
        chat_history = json.loads(history)
        
        rows = query_db("""
            SELECT timestamp, target_class, action_defined, duration_secs
            FROM security_logs ORDER BY id DESC LIMIT 15
        """)

        sanitized_timeline_context = sanitize_telemetry_logs(rows) if rows else "The camera tracking logs are currently empty."

        system_prompt = (
            "You are a friendly room security chatbot assistant. Answer the user's question accurately using "
            "the provided tracking logs. Speak like a normal person living in a house—never say things like 'the individual "
            "entered the building' or mention 'calibrating on a wall'. If the user asks about what happened, map the timestamps "
            "to their request. Keep answers clear, direct, and under 3 sentences."
            f"\n\nSanitized Room Activity Logs:\n{sanitized_timeline_context}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        payload = {
            "model": OLLAMA_MODEL, # Set dynamically from global profile definition configurations
            "messages": messages,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return {"response": response.json().get("message", {}).get("content", "Parsing error.").strip()}
        return {"response": f"Ollama service returned error status code: {response.status_code}."}

    except json.JSONDecodeError:
        return {"response": "[❌] Content Malformed: Failed to cleanly reconstruct conversation context metadata."}
    except requests.exceptions.Timeout:
        return {"response": f"[⏳] Chat request timed out. {OLLAMA_MODEL} is taking too long to process parameters."}
    except requests.exceptions.ConnectionError:
        return {"response": "[❌] Connection Refused: Could not reach Ollama. Please ensure your native engine service is active."}
    except Exception as e:
        return {"response": f"Chat engine exception caught: {e}"}

@app.get("/", response_class=responses.HTMLResponse)
def serve_dashboard():
    summary_data = query_db("""
        SELECT target_class, action_defined, COUNT(*), AVG(duration_secs)
        FROM security_logs GROUP BY target_class, action_defined
    """)
    recent_logs = query_db("""
        SELECT timestamp, target_class, action_defined, peak_confidence, duration_secs
        FROM security_logs ORDER BY id DESC LIMIT 5
    """)

    summary_rows = "".join([
        f"<tr><td>{row[0]}</td><td><span class='badge bg-secondary'>{row[1]}</span></td><td><b>{row[2]}</b></td><td>{row[3]:.1f}s</td></tr>"
        for row in summary_data
    ]) or "<tr><td colspan='4'>No behavioral tracking metrics accumulated yet.</td></tr>"

    recent_rows = "".join([
        f"<tr><td>{row[0]}</td><td><span class='badge {row[1].lower()}'>{row[1]}</span></td><td><small>{row[2]}</small></td><td>{row[3]*100:.0f}%</td><td>{row[4]:.1f}s</td></tr>"
        for row in recent_logs
    ]) or "<tr><td colspan='5'>Awaiting operational sequences...</td></tr>"

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>V380 Pro AI Analytics Panel</title>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'>
        <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
        <style>
            body { background-color: #121214; color: #e1e1e6; font-family: sans-serif; padding: 20px 0; }
            .card { background-color: #202024; border: 1px solid #323238; margin-bottom: 20px; }
            .table { color: #e1e1e6; border-color: #323238; }
            th { color: #8d8d99 !important; }
            .badge.human { background-color: #f75a68; }
            .badge.doggo { background-color: #00b37e; }
            .live-feed { max-width: 100%; height: auto; border: 2px solid #323238; }
            canvas { max-height: 240px; width: 100% !important; }
            .chat-box { height: 260px; overflow-y: auto; background-color: #1a1a1e; border: 1px solid #29292e; padding: 15px; border-radius: 6px; }
            .msg { margin-bottom: 12px; padding: 8px 12px; border-radius: 8px; max-width: 85%; display: inline-block; }
            .msg.user { background-color: #29292e; color: #e1e1e6; float: right; clear: both; }
            .msg.ai { background-color: #00b37e; color: #121214; font-weight: 500; float: left; clear: both; }
            .chart-placeholder { height: 240px; display: flex; align-items: center; justify-content: center; color: #8d8d99; font-style: italic; }
        </style>
    </head>
    <body>
        <div class='container'>
            <div class='d-flex justify-content-between align-items-center mb-4'>
                <h2>V380 Pro Behavioral Action Definer Dashboard</h2>
                <span class='badge bg-success'>Local Storage Engine</span>
            </div>

            <div class='row'>
                <div class='col-lg-6'>
                    <div class='card shadow-sm p-3 text-center'>
                        <h4 class='text-warning mb-3 text-start'>Live Camera Stream</h4>
                        <img src='/video_feed' class='live-feed rounded shadow-sm' alt='Live Video'>
                    </div>
                </div>

                <div class='col-lg-6'>
                    <div class='card shadow-sm p-4'>
                        <h4 class='text-primary mb-3'>Activity Metrics Aggregates</h4>
                        <table class='table'>
                            <thead>
                                <tr><th>Profile</th><th>Action State</th><th>Total Events</th><th>Avg Duration</th></tr>
                            </thead>
                            <tbody id="summaryTableBody">{SUMMARY_ROWS}</tbody>
                        </table>
                    </div>

                    <div class='card shadow-sm p-4'>
                        <h4 class='text-success mb-3'>Recent Transitions</h4>
                        <table class='table table-striped-columns'>
                            <thead>
                                <tr><th>Timestamp</th><th>Profile</th><th>Action Defined</th><th>Accuracy</th><th>Duration</th></tr>
                            </thead>
                            <tbody id="recentTableBody">{RECENT_ROWS}</tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div class='row'>
                <div class='col-12'>
                    <div class='card shadow-sm p-4' style='border: 1px solid #323238;'>
                        <h4 class='text-success mb-3'>Room Activity Query</h4>
                        <div class='chat-box' id='chatBox'>
                            <div class='msg ai'>Ask me what happened or specify a recent timeframe.</div>
                        </div>
                        <form id='chatForm' class='input-group mt-3' onsubmit='submitChatRequest(event)'>
                            <input type='text' id='userInput' class='form-control bg-dark text-white border-secondary' placeholder='e.g., What happened at 02:19 AM?' autocomplete='off' required>
                            <button type='submit' class='btn btn-success'>Ask AI</button>
                        </form>
                    </div>
                </div>
            </div>

            <div class='row'>
                <div class='col-md-8'>
                    <div class='card shadow-sm p-4'>
                        <h5 class='text-info mb-3'>Hourly Activity Distribution Map</h5>
                        <div id='hourlyContainer'>
                            <canvas id='hourlyChart'></canvas>
                        </div>
                    </div>
                </div>
                <div class='col-md-4'>
                    <div class='card shadow-sm p-4'>
                        <h5 class='text-info mb-3'>Target Tracking Profiles Split</h5>
                        <div id='compositionContainer'>
                            <canvas id='compositionChart'></canvas>
                        </div>
                    </div>
                </div>
            </div>
            <button class='btn btn-outline-secondary btn-sm' onclick='renderAnalyticsCharts()'>Refresh View Metrics</button>
        </div>

        <script>
            let hourlyChartInstance = null;
            let compositionChartInstance = null;
            let conversationHistory = [];

            async function renderAnalyticsCharts() {
                try {
                    const response = await fetch('/api/chart_data');
                    const data = await response.json();

                    // Update metrics table content asynchronously without disrupting ongoing chat session context 
                    if(data.summary_rows_html) document.getElementById('summaryTableBody').innerHTML = data.summary_rows_html;
                    if(data.recent_rows_html) document.getElementById('recentTableBody').innerHTML = data.recent_rows_html;

                    const totalHourlyEvents = data.hourly_values.reduce((a, b) => a + b, 0);

                    if (totalHourlyEvents === 0) {
                        document.getElementById('hourlyContainer').innerHTML = "<div class='chart-placeholder'>Awaiting room tracking metrics data logs...</div>";
                        document.getElementById('compositionContainer').innerHTML = "<div class='chart-placeholder'>Awaiting profile distributions...</div>";
                        return;
                    }

                    // Restore canvas anchor environments if empty layout placeholders were injected previously
                    if(!document.getElementById('hourlyChart')) {
                        document.getElementById('hourlyContainer').innerHTML = '<canvas id="hourlyChart"></canvas>';
                    }
                    if(!document.getElementById('compositionChart')) {
                        document.getElementById('compositionContainer').innerHTML = '<canvas id="compositionChart"></canvas>';
                    }

                    if (hourlyChartInstance) hourlyChartInstance.destroy();
                    if (compositionChartInstance) compositionChartInstance.destroy();

                    hourlyChartInstance = new Chart(document.getElementById('hourlyChart'), {
                        type: 'bar',
                        data: {
                            labels: data.hourly_labels,
                            datasets: [{
                                label: 'Activity Logs Count',
                                data: data.hourly_values,
                                backgroundColor: '#00b37e',
                                borderColor: '#00e6a0',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                x: { ticks: { color: '#8d8d99' }, grid: { color: '#29292e' } },
                                y: { ticks: { color: '#8d8d99', stepSize: 1 }, grid: { color: '#29292e' } }
                            }
                        }
                    });

                    compositionChartInstance = new Chart(document.getElementById('compositionChart'), {
                        type: 'pie',
                        data: {
                            labels: data.comp_labels,
                            datasets: [{
                                data: data.comp_values,
                                backgroundColor: ['#f75a68', '#00b37e'],
                                borderWidth: 0
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { labels: { color: '#e1e1e6' } } }
                        }
                    });
                } catch (error) {
                    console.error("Error fetching or rendering analytics charts:", error);
                }
            }

            async function submitChatRequest(e) {
                e.preventDefault();
                const inputEl = document.getElementById('userInput');
                const boxEl = document.getElementById('chatBox');
                const msgText = inputEl.value.trim();

                if (msgText !== "") {
                    boxEl.innerHTML += "<div class='msg user'>" + msgText + "</div>";
                    inputEl.value = "";
                    
                    setTimeout(() => { boxEl.scrollTop = boxEl.scrollHeight; }, 10);

                    conversationHistory.push({ role: "user", content: msgText });

                    if (conversationHistory.length > 20) {
                        conversationHistory = conversationHistory.slice(-20);
                    }

                    const loadId = "load_" + Date.now();
                    boxEl.innerHTML += "<div class='msg ai' id='" + loadId + "'>Processing query...</div>";
                    setTimeout(() => { boxEl.scrollTop = boxEl.scrollHeight; }, 10);

                    try {
                        const formData = new FormData();
                        formData.append("history", JSON.stringify(conversationHistory));

                        const response = await fetch('/api/chat', { method: 'POST', body: formData });
                        const resData = await response.json();

                        document.getElementById(loadId).innerText = resData.response;
                        conversationHistory.push({ role: "assistant", content: resData.response });
                    } catch(err) {
                        document.getElementById(loadId).innerText = "Unable to connect to local language model service.";
                    }
                    
                    setTimeout(() => { boxEl.scrollTop = boxEl.scrollHeight; }, 50);
                }
            }

            window.onload = async () => {
                await renderAnalyticsCharts();
            };
        </script>
    </body>
    </html>
    """
    
    return html_template.replace("{SUMMARY_ROWS}", summary_rows).replace("{RECENT_ROWS}", recent_rows)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)
