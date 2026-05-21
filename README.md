# SeeSee: Edge AI Telemetry Pipeline

Basically a local, cloud-free telemetry setup for my CCTV camera. It takes a live network stream, runs it through some vision math to track what me and my dog are doing in the room, logs everything to a database, and lets me talk to Llama 3.2:1b completely offline to summarize what happened.

---

## How It Works under the hood

                   ┌──────────────────────┐
                   │  V380 IP CCTV Camera │
                   └──────────┬───────────┘
                              │ (Live RTSP Stream)
                              ▼
                ┌────────────────────────────┐
                │  Low-Latency Ring Buffer   │
                └─────────────┬──────────────┘
                              │ (Decoupled Decoded Frames)
                              ▼
                 ┌──────────────────────────┐
                 │   YOLO11 Object Tensor   │
                 └────────────┬─────────────┘
                              │ (Spatial Coordinate Matrix)
                              ▼
           ┌──────────────────────────────────────┐
           │ Advanced Geometric Telemetry Tracker │
           └──────────────────┬───────────────────┘
                              │ (Calculated Behavioral Transitions)
                              ▼
                  ┌───────────────────────┐
                  │    SQLite Database    │
                  └───────────┬───────────┘
                              │ (Sanitized Relational Log Timeline)
                              ▼

┌──────────────┐     ┌──────────────────────┐     ┌────────────────┐
│ Browser UI   ├────►│ FastAPI Dashboard    ├────►│ Local Ollama   │
│ Chat Query   │◄────┤ Chat Middleware UI   │◄────┤ (Llama 3.2:1b) │
└──────────────┘     └──────────────────────┘     └────────────────┘


- **Smooth Video Feeds**: Used custom threading to isolate the camera stream frames from the heavy computer vision loops. The live feed stays at a fluid 30 FPS and my PC doesn't lag out when I am playing games.
- **Lightweight Math Tracking**: Avoided resource-intensive deep learning video models by writing simple geometric math that tracks bounding box aspect ratios and center-point speeds. This dynamically logs states like `Sitting/Working` or `Standing/Moving` without pinning down my CPU.
- **No Code Jargon**: Added a custom translation layer to clear out ugly database timestamps and strings before passing them to the local model, so Llama talks like a normal home assistant without hallucinating or spitting out random code syntax.

---

## The Tech Stack
- **OS**: Cross-Platform (Linux, macOS, Windows)
- **Backend Framework**: FastAPI (Uvicorn ASGI Engine)
- **Computer Vision**: OpenCV-Python (Headless Framework) & Ultralytics YOLO11
- **Database**: SQLite 3
- **AI Engine**: Ollama running `llama3.2:1b`

---

## Project Structure
```text
SeeSee/
├── .gitignore               # Keeps local DB logs, virtual envs, and model weights out of GitHub
├── README.md                # This file right here
├── requirements.txt         # The Python package dependencies list
└── dashboard_app.py         # Production-ready main source code script

Run Natively

Since the pipeline uses headless matrix tracking components and standard RTSP streaming, it is completely cross-platform and runs identically on Linux, macOS, and Windows.
1. Set Up a Virtual Environment

Initialize a clean Python environment for your operating system:

    Linux / macOS:
    Bash

    python -m venv venv
    source venv/bin/activate

    Windows (PowerShell):
    PowerShell

    python -m venv venv
    .\venv\Scripts\Activate.ps1

    Windows (Command Prompt):
    DOS

    python -m venv venv
    .\venv\Scripts\activate.bat

2. Install Dependencies

Install the lightweight, unbloated requirements list (uses headless modules to prevent server GUI driver collisions):
Bash

pip install -r requirements.txt

3. Spin Up Ollama (Local AI Engine)

Ensure your local language model engine is active on localhost:11434 and running the required model profile:

    Linux (systemd): sudo systemctl start ollama

    Linux (Non-systemd / Manual): ollama serve &

    Windows / macOS: Launch the native Ollama Desktop Application from your system application menu.

Make sure you have downloaded the weights into your engine instance:
Bash

ollama run llama3.2:1b

4. Export Credentials & Boot the Pipeline

Set your camera's live RTSP stream path network details as environment variables and execute the launch runner:

    Linux / macOS:
    Bash

    export RTSP_URL="rtsp://<username>:<password>@<camera-ip-address>:554/live/ch00_0"
    python dashboard_app.py

    Windows (PowerShell):
    PowerShell

    $env:RTSP_URL="rtsp://<username>:<password>@<camera-ip-address>:554/live/ch00_0"
    python dashboard_app.py

    Windows (Command Prompt):
    DOS

    set RTSP_URL=rtsp://<username>:<password>@<camera-ip-address>:554/live/ch00_0
    python dashboard_app.py

Open your browser and navigate to http://localhost:8050 to access your live analytics dashboard center!
Privacy Details

    100% Offline: Zero tracking, remote keys, or cloud subscriptions. Everything stays entirely on the local drive.

    Safe Version Control: Uses environment variables via os.getenv so your real credentials and IP location never get leaked onto public code repository lines.
