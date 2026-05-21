import os
import cv2
import threading
import time
import sqlite3
import requests
import json
from collections import deque
from ultralytics import YOLO
from fastapi import FastAPI, responses, Form
import uvicorn

app = FastAPI(title="V380 Pro AI Analytics Center")

# --- SYSTEM CONFIGURATION CONFIGS ---
DB_FILE = "v380_analytics.db"
AI_DETECTION_INTERVAL = 0.2

# Pull credentials safely from environment variables to secure public Git pushes
RTSP_URL = os.getenv("RTSP_URL")
if not RTSP_URL:
    raise ValueError("[❌] Critical Error: RTSP_URL environment variable is missing! Please export it before running.")

# Split base host from the specific API endpoint for clean networking abstraction
OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_URL = f"{OLLAMA_BASE.rstrip('/')}/api/generate"

# --- NETWORK VIDEO STREAM MANAGEMENT ---
class LowLatencyRingBuffer:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.buffer = deque(maxlen=1)
        self.ret = False
        self.running = True

    def start(self):
        threading.Thread(target=self._capture_worker, daemon=True).start()
        return self

    def _capture_worker(self):
        while self.running:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.buffer.append(frame)
            else:
                time.sleep(0.01)

    def get_frame(self):
        if self.ret and len(self.buffer) > 0:
            return True, self.buffer[0].copy()
        return False, None

# --- RELATIONAL STORAGE SCHEMA & TRANSACTIONS ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
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
    conn.commit()
    conn.close()

def log_event_to_db(target, action, confidence, duration):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO security_logs (target_class, action_defined, peak_confidence, duration_secs)
        VALUES (?, ?, ?, ?)
    """, (target, action, round(confidence, 2), round(duration, 1)))
    conn.commit()
    conn.close()

# --- SPATIAL GEOMETRY & TELEMETRY PROCESSING ---
class AdvancedSpatialTracker:
    def __init__(self, target_name):
        self.target_name = target_name
        self.is_active = False
        self.start_time = 0.0
        self.peak_confidence = 0.0
        self.history = deque(maxlen=10)
        self.current_action = "Calibrating"

    def process(self, is_visible, current_conf, box_coords=None):
        if is_visible and box_coords is not None:
            self.peak_confidence = max(self.peak_confidence, current_conf)
            x1, y1, x2, y2 = box_coords
            w = x2 - x1
            h = y2 - y1
            cx = x1 + (w / 2)
            cy = y1 + (h / 2)
            aspect_ratio = w / float(h) if h > 0 else 0

            self.history.append({"cx": cx, "cy": cy, "ar": aspect_ratio, "h": h})

            if len(self.history) == self.history.maxlen:
                if self.target_name == "Human":
                    avg_ar = sum(f["ar"] for f in self.history) / len(self.history)
                    if avg_ar > 0.72:
                        self.current_action = "Sitting/Working"
                    else:
                        self.current_action = "Standing/Moving"
                elif self.target_name == "Doggo":
                    dx = abs(self.history[-1]["cx"] - self.history[0]["cx"])
                    dy = abs(self.history[-1]["cy"] - self.history[0]["cy"])
                    total_velocity = dx + dy

                    if total_velocity > 40:
                        self.current_action = "Pacing/Moving"
                    else:
                        self.current_action = "Bedded/Sleeping"

            if not self.is_active:
                self.is_active = True
                self.start_time = time.time()
        else:
            if self.is_active:
                duration = time.time() - self.start_time
                self.is_active = False
                log_event_to_db(self.target_name, self.current_action, self.peak_confidence, duration)
                self.peak_confidence = 0.0
                self.history.clear()
                self.current_action = "Calibrating"

stream_bridge = None
yolo_model = None
human_tracker = AdvancedSpatialTracker("Human")
dog_tracker = AdvancedSpatialTracker("Doggo")
latest_processed_frame = None
frame_lock = threading.Lock()

# --- DATA SANITIZATION & TRANSLATION LAYER ---
def sanitize_telemetry_logs(rows):
    translation_layer = {
        "Calibrating": "entering or settling into the room briefly",
        "Sitting/Working": "sitting down at the desk working",
        "Standing/Moving": "standing or walking around the room",
        "Pacing/Moving": "actively pacing around the room",
        "Bedded/Sleeping": "lying down on the floor resting or sleeping"
    }

    clean_text = ""
    for r in reversed(rows):
        timestamp, target, action, duration = r
        friendly_action = translation_layer.get(action, action.lower())

        if target == "Human":
            clean_text += f"- At {timestamp}, a human was detected {friendly_action} for {duration} seconds.\n"
        else:
            clean_text += f"- At {timestamp}, a dog was tracked {friendly_action} for {duration} seconds.\n"
    return clean_text

# --- BACKGROUND COMPUTER VISION WORKER ---
def background_ai_inference_worker():
    global stream_bridge, yolo_model, human_tracker, dog_tracker, latest_processed_frame

    while True:
        t_start = time.time()
        success, frame = stream_bridge.get_frame()
        if not success or frame is None:
            time.sleep(0.01)
            continue

        results = yolo_model.predict(frame, device="cpu", verbose=False, imgsz=640)[0]

        f_human, f_dog = False, False
        h_conf, d_conf = 0.0, 0.0
        h_box, d_box = None, None

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id in [0, 16] and conf > 0.60:
                is_human = (cls_id == 0)
                coords = map(int, box.xyxy[0])
                x1, y1, x2, y2 = coords

                if is_human:
                    f_human = True
                    h_conf = max(h_conf, conf)
                    h_box = (x1, y1, x2, y2)
                else:
                    f_dog = True
                    d_conf = max(d_conf, conf)
                    d_box = (x1, y1, x2, y2)

                act_label = human_tracker.current_action if is_human else dog_tracker.current_action
                color = (0, 0, 255) if is_human else (0, 255, 0)
                lbl = f"{'Human' if is_human else 'Doggo'} ({act_label}): {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        human_tracker.process(f_human, h_conf, h_box)
        dog_tracker.process(f_dog, d_conf, d_box)

        txt = "SYSTEM MONITOR: " + ("Tracking Metrics Active" if (f_human or f_dog) else "Clear")
        clr = (0, 0, 255) if f_human else (0, 255, 0) if f_dog else (255, 165, 0)
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 2)

        with frame_lock:
            latest_processed_frame = frame.copy()

        elapsed = time.time() - t_start
        sleep_t = max(0.001, AI_DETECTION_INTERVAL - elapsed)
        time.sleep(sleep_t)

@app.on_event("startup")
def startup_event():
    global stream_bridge, yolo_model
    init_db()
    yolo_model = YOLO("yolo11n.pt")
    stream_bridge = LowLatencyRingBuffer(RTSP_URL).start()
    threading.Thread(target=background_ai_inference_worker, daemon=True).start()
    print("[🚀] Headless AI Geometric Action Matrix Engaged Flawlessly.")

def query_db(sql, params=()):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(sql, params)
    data = cursor.fetchall()
    conn.close()
    return data

# --- WEB APPLICATION BACKEND ENDPOINTS ---
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

    return {
        "hourly_labels": list(hourly_dict.keys()),
        "hourly_values": list(hourly_dict.values()),
        "comp_labels": list(comp_dict.keys()),
        "comp_values": list(comp_dict.values())
    }

# --- LOCAL GENERATIVE LLM CHAT MIDDLEWARE ---
@app.post("/api/chat")
def chatbot_endpoint(user_message: str = Form(...)):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, target_class, action_defined, duration_secs
            FROM security_logs ORDER BY id DESC LIMIT 15
        """)
        rows = cursor.fetchall()
        conn.close()

        sanitized_timeline_context = sanitize_telemetry_logs(rows) if rows else "The camera tracking logs are currently empty."

        system_prompt = (
            "You are a friendly room security chatbot assistant. Answer the user's question accurately using "
            "the provided tracking logs. Speak like a normal person living in a house—never say things like 'the individual "
            "entered the building' or mention 'calibrating on a wall'. If the user asks about what happened, map the timestamps "
            "to their request. Keep answers clear, direct, and under 3 sentences."
        )

        prompt_payload = (
            f"{system_prompt}\n\n"
            f"Sanitized Room Activity Logs:\n{sanitized_timeline_context}\n\n"
            f"User Question: {user_message}\n\n"
            f"Answer:"
        )

        payload = {
            "model": "llama3.2:1b",
            "prompt": prompt_payload,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return {"response": response.json().get("response", "Parsing error.").strip()}
        return {"response": f"Ollama service returned error status code: {response.status_code}."}

    except requests.exceptions.Timeout:
        return {"response": "[⏳] Chat request timed out. Llama 3.2 is taking too long to wake up or process tensors on your system hardware."}
    except requests.exceptions.ConnectionError:
        return {"response": "[❌] Connection Refused: Could not reach Ollama. Please ensure your native engine service is active via 'sudo systemctl start ollama'."}
    except Exception as e:
        return {"response": f"Chat engine exception caught: {e}"}

# --- WEB APPLICATION ROUTER & USER INTERFACE ---
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

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>V380 Pro AI Analytics Panel</title>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'>
        <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
        <style>
            body {{ background-color: #121214; color: #e1e1e6; font-family: sans-serif; padding: 20px 0; }}
            .card {{ background-color: #202024; border: 1px solid #323238; margin-bottom: 20px; }}
            .table {{ color: #e1e1e6; border-color: #323238; }}
            th {{ color: #8d8d99 !important; }}
            .badge.human {{ background-color: #f75a68; }}
            .badge.doggo {{ background-color: #00b37e; }}
            .live-feed {{ max-width: 100%; height: auto; border: 2px solid #323238; }}
            canvas {{ max-height: 240px; width: 100% !important; }}
            .chat-box {{ height: 260px; overflow-y: auto; background-color: #1a1a1e; border: 1px solid #29292e; padding: 15px; border-radius: 6px; }}
            .msg {{ margin-bottom: 12px; padding: 8px 12px; border-radius: 8px; max-width: 85%; display: inline-block; }}
            .msg.user {{ background-color: #29292e; color: #e1e1e6; float: right; clear: both; }}
            .msg.ai {{ background-color: #00b37e; color: #121214; font-weight: 500; float: left; clear: both; }}
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
                            <tbody>{summary_rows}</tbody>
                        </table>
                    </div>

                    <div class='card shadow-sm p-4'>
                        <h4 class='text-success mb-3'>Recent Transitions</h4>
                        <table class='table table-striped-columns'>
                            <thead>
                                <tr><th>Timestamp</th><th>Profile</th><th>Action Defined</th><th>Accuracy</th><th>Duration</th></tr>
                            </thead>
                            <tbody>{recent_rows}</tbody>
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
                        <canvas id='hourlyChart'></canvas>
                    </div>
                </div>
                <div class='col-md-4'>
                    <div class='card shadow-sm p-4'>
                        <h5 class='text-info mb-3'>Target Tracking Profiles Split</h5>
                        <canvas id='compositionChart'></canvas>
                    </div>
                </div>
            </div>
            <button class='btn btn-outline-secondary btn-sm' onclick='window.location.reload()'>Refresh View Metrics</button>
        </div>

        <script>
            let hourlyChartInstance = null;
            let compositionChartInstance = null;

            async function renderAnalyticsCharts() {{
                try {{
                    const response = await fetch('/api/chart_data');
                    const data = await response.json();

                    if (hourlyChartInstance) hourlyChartInstance.destroy();
                    if (compositionChartInstance) compositionChartInstance.destroy();

                    hourlyChartInstance = new Chart(document.getElementById('hourlyChart'), {{
                        type: 'bar',
                        data: {{
                            labels: data.hourly_labels,
                            datasets: [{{
                                label: 'Activity Logs Count',
                                data: data.hourly_values,
                                backgroundColor: '#00b37e',
                                borderColor: '#00e6a0',
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{ legend: {{ display: false }} }},
                            scales: {{
                                x: {{ ticks: {{ color: '#8d8d99' }}, grid: {{ color: '#29292e' }} }},
                                y: {{ ticks: {{ color: '#8d8d99', stepSize: 1 }}, grid: {{ color: '#29292e' }} }}
                            }}
                        }}
                    }});

                    compositionChartInstance = new Chart(document.getElementById('compositionChart'), {{
                        type: 'pie',
                        data: {{
                            labels: data.comp_labels,
                            datasets: [{{
                                data: data.comp_values,
                                backgroundColor: ['#f75a68', '#00b37e'],
                                borderWidth: 0
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{ legend: {{ labels: {{ color: '#e1e1e6' }} }} }}
                        }}
                    }});
                }} catch (error) {{
                    console.error("Error fetching or rendering analytics charts:", error);
                }}
            }}

            async function submitChatRequest(e) {{
                e.preventDefault();
                const inputEl = document.getElementById('userInput');
                const boxEl = document.getElementById('chatBox');
                const msgText = inputEl.value;

                if (msgText.trim() !== "") {{
                    boxEl.innerHTML += "<div class='msg user'>" + msgText + "</div>";
                    inputEl.value = "";
                    boxEl.scrollTop = boxEl.scrollHeight;

                    const loadId = "load_" + Date.now();
                    boxEl.innerHTML += "<div class='msg ai' id='" + loadId + "'>Processing query...</div>";
                    boxEl.scrollTop = boxEl.scrollHeight;

                    try {{
                        const formData = new FormData();
                        formData.append("user_message", msgText);

                        const response = await fetch('/api/chat', {{ method: 'POST', body: formData }});
                        const resData = await response.json();

                        document.getElementById(loadId).innerText = resData.response;
                    }} catch(err) {{
                        document.getElementById(loadId).innerText = "Unable to connect to local language model service.";
                    }}
                    boxEl.scrollTop = boxEl.scrollHeight;
                }}
            }}

            window.onload = async () => {{
                await renderAnalyticsCharts();
            }};
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)
