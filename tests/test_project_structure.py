from pathlib import Path


def test_required_project_structure_exists():
    project_root = Path(__file__).resolve().parents[1]

    required_paths = [
        "main.py",
        "app",
        "app/__init__.py",
        "app/camera.py",
        "app/face_detector.py",
        "app/face_recognition_service.py",
        "app/recognition_worker.py",
        "app/recognition_process.py",
        "app/risk_analyzer.py",
        "app/alert_service.py",
        "app/logger.py",
        "data/known_faces",
        "data/unknown_faces",
        "data/logs",
        "tests",
        "scripts/test_webcam_manual.py",
        "scripts/test_recognition_manual.py",
    ]

    missing_paths = [
        path for path in required_paths if not (project_root / path).exists()
    ]

    assert missing_paths == []
