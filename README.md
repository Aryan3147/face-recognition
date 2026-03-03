# Real-Time Face Registration & Detection System

A Python-based face recognition system using OpenCV Haar Cascades with Make.com webhook integration for automated attendance tracking.

## Features

- Real-time face detection from webcam
- Face registration via webcam
- Face recognition (identifies registered faces)
- Detects frontal and profile (side) faces
- Attendance logging with timestamps
- **Make.com webhook integration** — auto-sends attendance data after each session
- **Duplicate prevention** — 10-second cooldown per person to avoid log spam

## Requirements

- Python 3.8+
- OpenCV (opencv-python)
- NumPy
- Requests
- Webcam

## Installation

```bash
pip install opencv-python numpy requests
```

## Usage

```bash
python face_system.py
```

### Menu Options

1. **Start Detection** - Opens webcam for real-time face detection/recognition
2. **Register New Face** - Add a new person to the database
3. **List Registered Faces** - View all registered faces
4. **Remove a Face** - Delete a person from database
5. **Exit** - Close application

### Controls

- `r` - Register a new face (during detection)
- `q` - Quit detection mode and send attendance to Make.com

## Project Structure

```
face-recognition/
├── face_system.py         # Main application
├── test_webhook.py        # Standalone webhook connectivity tester
├── requirements.txt       # Dependencies
├── face_database/         # Created on first run
│   ├── encodings.pkl      # Face embeddings database
│   └── images/            # Registered face images
└── attendance.log         # Detection logs
```

## How It Works

1. **Detection**: Uses Haar Cascades for frontal and profile face detection
2. **Embedding**: Extracts histogram features from face images
3. **Recognition**: Compares embeddings using histogram correlation
4. **Threshold**: Similarity > 0.5 = match
5. **Logging**: Each recognized person is logged with a timestamp (10-second cooldown)
6. **Webhook**: On session end (press `q`), all attendance records are POSTed to Make.com

## Make.com Integration

Attendance data is automatically sent to a Make.com webhook at the end of every detection session.

### Webhook Configuration

The webhook URL is set at the top of `face_system.py`:

```python
MAKE_WEBHOOK_URL = "https://hook.eu1.make.com/qj676s4s6r0gipdpyfa39xxtbvus73b9"
```

### Make.com Scenario

**Scenario:** Attendance Log Processor v3  
**Trigger:** Custom Webhook (fires on each POST from Python)  
**Flow:** Webhook Trigger → Webhook Response  

The scenario receives CSV-formatted attendance data in this format:
```
Aryan,2026-03-03 11:28:00
Aryan,2026-03-03 11:28:10
```

### Make.com Workflow Diagram

```
┌─────────────────────┐     POST CSV data      ┌─────────────────────┐
│   face_system.py    │ ──────────────────────▶ │  Make.com Webhook   │
│   (Python Script)   │                         │      Trigger        │
└─────────────────────┘                         └──────────┬──────────┘
                                                            │
                                                            ▼
                                                 ┌─────────────────────┐
                                                 │  Webhook Response   │
                                                 │   (HTTP 200 OK)     │
                                                 └─────────────────────┘
```

**Data format sent to Make.com:**
```
Name,Timestamp
Aryan,2026-03-03 11:28:00
Aryan,2026-03-03 11:28:10
```

### Testing the Webhook

Before running the full system, verify the webhook is reachable:

```bash
python test_webhook.py
```

Expected output:
```
Testing Make webhook...
URL: https://hook.eu1.make.com/qj676s4s6r0gipdpyfa39xxtbvus73b9
SUCCESS - HTTP 200
Response: Accepted
```

## Notes

- Works best with good lighting
- One face at a time for registration
- Frontal face detection is more accurate than profile
- Make sure the Make.com scenario is **active** before running detection