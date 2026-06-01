# Face Security

Face Security is an academic MVP for residential security support using local
facial recognition. It opens a webcam, compares captured frames with registered
images in `data/known_faces`, classifies the attention level, shows a visual
alert, saves local evidence for unrecognized people, and registers local logs.

The system does not make accusations, predict incidents, or check external
background data. It only distinguishes registered/authorized people from
unrecognized people in a local demonstration environment.

## Requirements

- Python 3.10 or newer
- Webcam
- Registered face images in `data/known_faces`

Each registered image should use the person's display identifier as the file
name, for example:

```text
data/known_faces/integrante_01.jpeg
data/known_faces/integrante_02.jpeg
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bat
venv\Scripts\activate
```

Install runtime and test dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

If OpenCV reports `libGL.so.1` missing on Linux, install the system OpenGL
runtime package for your distribution before running the webcam demo.

## Running Tests

Automated tests do not require a real webcam or real DeepFace execution.

```bash
pytest
```

## Running The System

Start the main webcam flow:

```bash
python main.py
```

Press `q` in the webcam window to stop.

## Manual Webcam Test

Use the manual script only when testing the real webcam:

```bash
python scripts/test_webcam_manual.py
```

For a video demonstration:

1. Add at least one clear face image to `data/known_faces`.
2. Run `python scripts/test_webcam_manual.py`.
3. Show a registered person and confirm the `AUTHORIZED` / `LOW` alert.
4. Show an unrecognized person and confirm the `UNRECOGNIZED` / `ATTENTION`
   alert.
5. Show that evidence images are saved in `data/unknown_faces`.
6. Show that logs are written to `data/logs/events.jsonl`.

## Project Structure

```text
app/
  alert_service.py
  camera.py
  face_recognition_service.py
  logger.py
  risk_analyzer.py
data/
  known_faces/
  unknown_faces/
  logs/
scripts/
  test_webcam_manual.py
tests/
main.py
```

## Local Outputs

- Evidence images: `data/unknown_faces`
- Event logs: `data/logs/events.jsonl`
