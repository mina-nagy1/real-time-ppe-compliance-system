# Real-Time PPE Compliance System

A real-time Personal Protective Equipment (PPE) detection and compliance monitoring system for construction sites. Built with YOLO11s, BoxMOT, FastAPI, and SQLite — fine-tuned on the Construction Site Safety dataset with full MLflow experiment tracking.

---

## Overview

The system processes a live video feed and performs detection, tracking, and compliance analysis in real time. Each worker is assigned a persistent ID across frames, their PPE state is evaluated every frame, and all violation events are logged to a queryable SQLite database. Results are streamed to a live dark-mode dashboard via WebSocket.

---

## Features

- **PPE Detection** — Fine-tuned YOLO11s detecting 10 classes including Hardhat, Safety Vest, Mask, and their negative counterparts
- **Multi-Object Tracking** — BoxMOT (ByteTrack) assigns persistent worker IDs across frames with motion trail visualization
- **Compliance Engine** — Per-worker compliance evaluation with configurable required equipment rules
- **Live Dashboard** — Dark-mode UI with MJPEG stream, real-time Chart.js metrics, and violation event log
- **REST API + WebSocket** — FastAPI backend with MJPEG stream, REST endpoints, and WebSocket push at 4Hz
- **Event Persistence** — SQLite event store logging violations, zone occupancy, line crossings, and crowd alerts
- **Experiment Tracking** — MLflow logging of per-epoch loss curves, mAP, precision, recall, and model artifacts
- **Docker Support** — Fully containerized for portable deployment

---

## System Architecture

```
Video Feed
    │
    ▼
Detector (YOLO11s)
    │  bounding boxes + class labels
    ▼
Tracker (BoxMOT / ByteTrack)
    │  persistent worker IDs + motion history
    ▼
Engine (Compliance + Analytics)
    │  violation flags + crowd alerts + heatmap
    ▼
┌───────────────┬──────────────────┐
│  Visualizer   │  Database        │
│  (cv2 draw)   │  (SQLite events) │
└───────┬───────┴──────────────────┘
        │
        ▼
Pipeline (background thread)
        │
        ▼
FastAPI Server
    ├── /stream     (MJPEG)
    ├── /ws         (WebSocket)
    ├── /api/...    (REST)
    └── /           (Dashboard)
```

---

## Dataset

**Construction Site Safety** by Roboflow Universe Projects

| Split | Images |
|---|---|
| Train | 2,605 |
| Val | 114 |
| Test | 82 |

**10 Classes:** `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`, `vehicle`

---

## Model Performance

| Metric | Value |
|---|---|
| mAP50 (val) | 0.836 |
| mAP50-95 (val) | 0.513 |
| Precision | 0.909 |
| Recall | 0.770 |
| mAP50 (test) | 0.705 |

> Test set mAP50 reflects performance on 82 fully unseen images.

---

## Project Structure

```
real-time-ppe-compliance-system/
├── configs/                  # YAML configs for model, tracker, stream, analytics
├── src/
│   ├── detector.py           # YOLO inference wrapper
│   ├── tracker.py            # BoxMOT multi-object tracking
│   ├── engine.py             # Compliance checking + analytics
│   ├── visualizer.py         # All cv2 rendering
│   ├── pipeline.py           # Orchestrator background thread
│   ├── database.py           # SQLite event persistence
│   ├── config_loader.py      # Pydantic config validation
│   ├── utils.py              # IoU, cross-product, FPS, color palette
│   └── logger.py             # Structured JSON logging
├── api/
│   ├── server.py             # FastAPI app + MJPEG + WebSocket
│   ├── routes.py             # REST endpoints
│   ├── websocket.py          # WebSocket connection manager
│   └── schemas.py            # Pydantic request/response models
├── assets/
│   ├── loss_curves.png
│   └── confusion_matrix.png
├── dashboard/
│   └── index.html            # Dark-mode live dashboard
├── notebooks/
│   └── train.ipynb           # YOLO11s fine-tuning with MLflow
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
├── models/
│   └── labels.txt            # Class names
└── requirements.txt
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/mina-nagy1/real-time-ppe-compliance-system.git
cd real-time-ppe-compliance-system
pip install -r requirements.txt
```

### 2. Add model weights

Place your trained `best.pt` in the `models/` directory. To train your own model see `notebooks/train.ipynb`.

### 3. Configure video source

Edit `configs/stream_config.yaml`:

```yaml
source: 0                          # webcam
# source: "data/samples/site.mp4" # video file
# source: "rtsp://..."            # IP camera
```

### 4. Run

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Docker

```bash
# Place best.pt in models/ first
cd docker
docker-compose up --build
```

Open `http://localhost:8000`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Live dashboard |
| GET | `/stream` | MJPEG video stream |
| WS | `/ws` | WebSocket analytics push |
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Live pipeline stats |
| GET | `/api/violations` | Violations in last 24h |
| GET | `/api/violations/top` | Top violators by worker ID |
| GET | `/api/violations/by-type` | Count per violation type |
| GET | `/api/violations/hourly` | Hourly violation trends |
| GET | `/api/crossings` | Counting line crossing counts |
| GET | `/api/occupancy/peak` | Peak zone occupancy |
| GET | `/api/alerts` | Recent crowd density alerts |

---

## Configuration

All behaviour is controlled via YAML files in `configs/`:

| File | Controls |
|---|---|
| `model_config.yaml` | Weights path, confidence, IoU, class map, compliance rules |
| `tracker_config.yaml` | Tracker backend (ByteTrack/DeepOCSORT/BoTSORT), trail length |
| `stream_config.yaml` | Video source, resolution, FPS, JPEG quality |
| `analytics_config.yaml` | Counting lines, crowd threshold, occupancy interval |

---

## Training

Open `notebooks/train.ipynb` to fine-tune on your own dataset. The notebook handles:

- Roboflow dataset download
- YOLO11s fine-tuning with full augmentation config
- MLflow experiment tracking (per-epoch metrics, loss curves, artifacts)
- Validation on test split
- ONNX export

View experiments:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

---

## Future Work

- INT8 quantization for edge device deployment
- Multi-camera support with cross-camera re-identification
- Segmentation masks with a polygon-annotated dataset
- Alert notification system (email / webhook)

---

## License

MIT
