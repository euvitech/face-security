import runpy
from pathlib import Path


def test_manual_webcam_script_delegates_to_main(mocker):
    project_root = Path(__file__).resolve().parents[1]
    main_function = mocker.patch("main.main")

    runpy.run_path(
        str(project_root / "scripts" / "test_webcam_manual.py"),
        run_name="__main__",
    )

    main_function.assert_called_once()
