# Face Security

Face Security is an academic MVP for residential security support using local
facial recognition. It opens a webcam, shows the camera feed, compares captured
frames with registered images in `data/known_faces` only after a face is visible,
classifies the attention level, saves local evidence for unrecognized people,
and registers local logs.

The system does not make accusations, predict incidents, or check external
background data. It only distinguishes registered/authorized people from
unrecognized people in a local demonstration environment.

## Requirements

- Python 3.10 or newer
- Webcam
- Registered face images in `data/known_faces`

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

## Configuration

Use `.env.example` as a reference and export the same variables in your shell,
Docker configuration, or runtime environment:

```text
KNOWN_FACES_DIR=data/known_faces
UNKNOWN_FACES_DIR=data/unknown_faces
LOGS_DIR=data/logs
CAMERA_INDEX=0
RECOGNITION_MODEL=Facenet
DISTANCE_METRIC=cosine
RECOGNITION_THRESHOLD=0.60
ENFORCE_DETECTION=false
REQUIRE_FACE_DETECTION=true
REBUILD_FACE_CACHE=true
SAVE_UNKNOWN_EVIDENCE=true
SHOW_CAMERA_WINDOW=true
RECOGNITION_INTERVAL_FRAMES=30
RECOGNITION_INTERVAL_SECONDS=1.5
UNKNOWN_EVIDENCE_COOLDOWN_SECONDS=5
```

`REBUILD_FACE_CACHE=true` removes DeepFace `representations_*.pkl` files from
`data/known_faces` before recognition starts. This helps newly added photos get
picked up. To refresh manually, delete any `representations_*.pkl` file under
`data/known_faces`.

`REQUIRE_FACE_DETECTION=true` prevents recognition from running on empty frames.
When no face is visible, the camera overlay shows `NO_FACE_DETECTED` and waits;
it does not classify the frame as an unrecognized person and does not save
unknown evidence.

OpenCV face detection runs every frame so the face box can follow the person in
real time. DeepFace recognition is intentionally slower: it runs only every
`RECOGNITION_INTERVAL_FRAMES` frames or every `RECOGNITION_INTERVAL_SECONDS`
seconds, whichever comes first. This keeps the webcam feed fluid while the last
recognition result remains visible near the face box.

`UNKNOWN_EVIDENCE_COOLDOWN_SECONDS` limits how often unknown evidence images are
saved, so an unrecognized person does not create a new file on every frame.

## How to register known people

To recognize someone, add clear face photos to `data/known_faces`. The system
only recognizes people with photos in `known_faces`; anyone without a matching
registered photo can only be classified as `UNRECOGNIZED`.

The service scans `data/known_faces` recursively and accepts `.jpg`, `.jpeg`,
and `.png` files. Non-image files and `representations_*.pkl` files are ignored.

Format 1: image files directly inside `known_faces`:

```text
data/known_faces/rayston_01.jpg
data/known_faces/rayston_02.jpg
data/known_faces/member_01.jpg
```

For direct files, the identity comes from the filename stem with a trailing
number suffix removed, so `rayston_01.jpg` and `rayston_02.jpg` both become
`rayston`.

Format 2: folders per person:

```text
data/known_faces/rayston/photo_01.jpg
data/known_faces/rayston/photo_02.jpg
data/known_faces/ana/photo_01.jpg
data/known_faces/rayston/rayston_01.jpg
```

For folder-based files, the identity is the folder name, such as `rayston` or
`ana`.

Photo guidelines:

- Use clear frontal face photos.
- Use good lighting.
- Avoid sunglasses, masks, or covered faces.
- Use 2 to 5 photos per person if possible.
- Prefer one person per image.
- Use filenames like `rayston_01.jpg`, `rayston_02.jpg`.
- Or use folders like `data/known_faces/rayston/photo_01.jpg`.

After adding or changing photos, clear the DeepFace cache if needed:

```bash
find data/known_faces -name "representations_*.pkl" -delete
```

Unknown face images are saved in `data/unknown_faces` when a visible face is not
recognized and the evidence cooldown allows a new save.

## Running Tests

Automated tests do not require a real webcam or real DeepFace execution.

```bash
pytest
```

## Manual Recognition Test

Capture one frame, run recognition, print the result, and save unknown evidence
without opening a GUI window:

```bash
python scripts/test_recognition_manual.py
```

If the result is `UNRECOGNIZED` and `SAVE_UNKNOWN_EVIDENCE=true`, the captured
frame is saved in `data/unknown_faces` and the event is logged in
`data/logs/events.jsonl`. If no face is detected, the script returns
`NO_FACE_DETECTED` and does not save an unknown evidence image.

## Running The System

Start the main webcam flow:

```bash
python main.py
```

By default, `SHOW_CAMERA_WINDOW=true`, so the app opens a webcam window and draws
the current recognition status on the frame. Press `q` in the webcam window to
stop. To run without a GUI window:

```bash
SHOW_CAMERA_WINDOW=false python main.py
```

In no-window mode, stop it with `Ctrl+C`.

## Local Outputs

- Unknown evidence images: `data/unknown_faces`
- Event logs: `data/logs/events.jsonl`

Recognized events are logged with `AUTHORIZED` and `LOW`. Unrecognized events
are logged with `UNRECOGNIZED` and `ATTENTION`, and unknown evidence is saved in
`data/unknown_faces` when `SAVE_UNKNOWN_EVIDENCE=true` and the cooldown has
elapsed.

Empty camera frames are shown as `NO_FACE_DETECTED` / `NONE` and are not saved as
unknown evidence.

## Project Structure

```text
app/
  alert_service.py
  camera.py
  face_detector.py
  face_recognition_service.py
  logger.py
  risk_analyzer.py
data/
  known_faces/
  unknown_faces/
  logs/
scripts/
  test_recognition_manual.py
  test_webcam_manual.py
tests/
main.py
```
