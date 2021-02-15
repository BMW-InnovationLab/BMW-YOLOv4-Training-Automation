import os
from pathlib import Path
from starlette.responses import JSONResponse
from PIL import Image
import subprocess
from io import BytesIO
import shutil
import re
from typing import Any

working_dir: Path = Path.cwd()
trainn_dir: Path = working_dir / "custom_training" / Path(os.getenv('TRAIN_NAME') + "_" + os.getenv("TRAIN_START_TIME"))
yolo_events_log_path: Path =  trainn_dir / Path("yolo_events.log")
yolo_events_log_path_1: Path = trainn_dir / Path("yolo_events.log.1")
pid_path: Path = working_dir / "pid.txt"


def check_error() -> Any:
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


def perform_prediction(image, use_default_weights: bool, is_video: bool) -> dict:
    """Runs the last saved weights to infer on the given image.

    Args:
        image (bytes): the image to run the prediction on
        use_default_weights (bool): whether to use the default YOLOv4 weights

    Returns:
        dict
    """

    prediction_path: Path = working_dir / "predictions"
    training_path: Path = trainn_dir
    weights_path: Path = training_path / "weights"
    last_weights: list = list(weights_path.glob("*_last.weights"))

    if not last_weights:
        result: dict = {
            "success": True,
            "start_time": get_time(),
            "message": "No predictions yet",
        }
        return result

    if not prediction_path.exists():
        # Create folder in working directory symlinked to darknet/data/labels because it is needed by darknet to label the bounding boxes
        Path.mkdir(prediction_path)
        os.chdir(prediction_path)
        os.mkdir(Path("data"))
        os.symlink(
            working_dir / "darknet/data/labels", prediction_path / "data/labels"
        )

    if is_video:
        input_path: str = str(working_dir / "predictions/video.mp4")
        output_path = str(prediction_path / "video_out.mp4")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    else:
        img: Image = Image.open(BytesIO(image)).convert("RGB")
        img.save("image.jpg")
        input_path  = str(working_dir / "predictions/image.jpg")
        output_path = 'predictions.jpg'

    config_file_path: Path = training_path / "config"
    data_path: str = str(list(config_file_path.glob("*.data"))[0])
    cfg_path: str = str(list(config_file_path.glob("*.cfg"))[0])
    last_weights: str = str(last_weights[0])

    if use_default_weights:
        data_path: str = f'{working_dir}/darknet/cfg/coco.data'
        cfg_path: str = f'{working_dir}/config/darknet/yolov4_default_cfgs/yolov4.cfg'
        last_weights: str = f'{working_dir}/config/darknet/yolov4_default_weights/yolov4.weights'

    darknet_exec_path: Path = working_dir / "darknet/darknet"
    command: list = [
        darknet_exec_path,
        'detector',
        'demo' if is_video else 'test',
        data_path,
        cfg_path,
        last_weights,
        '-dont_show',
        input_path
    ]
    if is_video:
        command.append('-ext_output')
        command.append('-out_filename')
        command.append(output_path)

    with open(str(prediction_path / 'darknet_prediction.out'), "w") as out_log:
        with open(str(prediction_path / 'darknet_prediction.err'), "w") as err_log:
            subprocess.call(command, stdout=out_log, stderr=err_log)

    return {'output_path': output_path}

def get_time():
    return os.environ["TRAIN_START_TIME"]

def get_bb_results() -> dict:
    """Gives a dictionary of bounding boxes for each frame of the last predicted video

    Returns:
        dict: dict in form {0: 'aeroplane confidence left_x top_y width height', ...} in pixel coordinates
    """
    prediction_path: Path = working_dir / "predictions"
    with open(str(prediction_path / 'darknet_prediction.out'), "r") as out_log:
        input = out_log.readlines()
        lines = ''.join(input)
        matches = re.findall(
            '^([a-zA-z0-9]+): ([0-9]+)[%].+left_x: [ ]+([0-9]+)[ ]+top_y:[ ]+([0-9]+)[ ]+width:[ ]+([0-9]+)[ ]+height:[ ]+([0-9]+)[)]$',
            lines,
            flags=re.MULTILINE
        )
        indices = [i for i, x in enumerate(input) if x == 'Objects:\n']
        frame_id, box_id = 1, 0
        result = dict()

        # Match the boxes with the frames.
        while frame_id < len(indices):
            box_count = indices[frame_id] - indices[frame_id-1] - 6
            result[frame_id] = []

            if box_count > 0:
                for _ in range(box_count):
                    result[frame_id].append(' '.join(matches[box_id]))
                    box_id += 1

            frame_id += 1

        # Add remaining matches to last frame.
        result[frame_id] = []
        for box_id in range(box_id, len(matches)):
            result[frame_id].append(' '.join(matches[box_id]))

        return result
