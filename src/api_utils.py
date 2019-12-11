import os
from pathlib import Path
from starlette.responses import JSONResponse


working_dir: Path = Path.cwd()
trainn_dir: Path = working_dir / "custom_training" / Path(os.getenv('TRAIN_NAME') + "_" + os.getenv("TRAIN_START_TIME"))
yolo_events_log_path: Path =  trainn_dir / Path("yolo_events.log")
yolo_events_log_path_1: Path = trainn_dir / Path("yolo_events.log.1")
pid_path: Path = working_dir / "pid.txt"


def check_error() -> any:
    # Check if output file exists
    result: dict = {"success": True, "message": "Training has not started yet"}
    get_time()
    # Training hasn't started if the logging file was not created yet
    if not yolo_events_log_path.exists():
        return result

    # If the logging file is still empty, the training hasn't started yet
    # but the first log file could be empty since it is a rotating file
    # To know if the training has started, the second log file should exist
    with open(yolo_events_log_path, "r") as logReader:
        line: list = logReader.readlines()
        if line == [] and not yolo_events_log_path_1.exists():
            return result

    # Check if process was killed
    try:
        with open(pid_path, "r") as pidReader:
            pid: str = pidReader.read()

        # Check if training has ended
        if pid == "Done":
            result = {
                "success": True,
                "message": "Training has ended :) ... Check trainings/{}/weights folder to get all saved weights files".format(
                    str(get_time())
                ),
            }
            return result
        os.kill(int(pid), 0)
    except OSError:
        result = {
            "success": False,
            "message": "Training was killed :( ... Check yolo_events.log and yolo_events.log.1 for possible error messages or try restarting the training with different parameters ",
        }
        return JSONResponse(status_code=400, content=result)

    return False


def get_time():
    return os.environ["TRAIN_START_TIME"]
