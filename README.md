Markdown

# V380 Pro AI Analytics Center

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
- **OS**: Arch Linux
- **Backend Framework**: FastAPI (Uvicorn ASGI Engine)
- **Computer Vision**: OpenCV-Python & Ultralytics YOLO11
- **Database**: SQLite 3
- **AI Engine**: Ollama running `llama3.2:1b`

---

## Project Structure
```text
v380-yolo/
├── .gitignore               # Keeps local DB logs, .env files, and model weights out of GitHub
├── README.md                # This file right here
├── requirements.txt         # The Python package dependencies list
├── Dockerfile               # Build configuration for the application container
├── docker-compose.yml       # Multi-container orchestrator layout
└── dashboard_app.py         # Main source code script

Quick Start
Run via Docker (Recommended)

Docker handles the entire isolated environment so you don't have to break your brain reinstalling system libraries or model weights.

    Clone the repo:
    Bash

    git clone <YOUR_REPOSITORY_URL>
    cd v380-yolo

    Add your camera credentials:
    Open docker-compose.yml and replace the placeholder credentials inside the environment block with your camera's active username, password, and local IP:
    YAML

    environment:
      - RTSP_URL=rtsp://admin:your_secret_password@192.168.100.188:554/live/ch00_0

    Spin it up:
    Bash

    docker-compose up --build -d

    Pull the model files:
    Since it's the first initialization, run this into the container to download the local model parameters (it's going to pull a few gigabytes of tensors, just let it do its thing):
    Bash

    docker exec -it ollama_engine ollama run llama3.2:1b

    Open the interface: Open your browser and go to http://localhost:8050.

Run Natively

    Set up a virtual environment:
    Bash

    python -m venv venv
    source venv/bin/activate

    Install the dependencies:
    Bash

    pip install -r requirements.txt

    Export your secrets:
    Bash

    export RTSP_URL="rtsp://admin:your_secret_password@192.168.100.188:554/live/ch00_0"
    export OLLAMA_URL="[http://127.0.0.1:11434/api/generate](http://127.0.0.1:11434/api/generate)"

    Run the script:
    Bash

    python dashboard_app.py

Privacy Details

    100% Offline: Zero tracking, remote keys, or cloud subscriptions. Everything stays entirely on the local drive.

    Safe Version Control: Uses environment variables via os.getenv so your real credentials and IP location never get leaked onto public code repository lines.
