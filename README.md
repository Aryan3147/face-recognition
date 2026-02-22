# Real-Time Face Registration & Detection System

A Python-based face recognition system using OpenCV Haar Cascades.

## Features

- Real-time face detection from webcam
- Face registration via webcam
- Face recognition (identifies registered faces)
- Detects frontal and profile (side) faces
- Attendance logging with timestamps

## Requirements

- Python 3.8+
- OpenCV (opencv-python)
- NumPy
- Webcam

## Installation

```bash
pip install opencv-python numpy
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
- `q` - Quit detection mode

## Project Structure

```
face-recognition/
├── face_system.py         # Main application
├── requirements.txt       # Dependencies
├── face_database/         # Created on first run
│   ├── encodings.pkl    # Face embeddings database
│   └── images/          # Registered face images
└── attendance.log        # Detection logs
```

## How It Works

1. **Detection**: Uses Haar Cascades for frontal and profile face detection
2. **Embedding**: Extracts histogram features from face images
3. **Recognition**: Compares embeddings using histogram correlation
4. **Threshold**: Similarity > 0.5 = match

## Notes

- Works best with good lighting
- One face at a time for registration
- Frontal face detection is more accurate than profile
