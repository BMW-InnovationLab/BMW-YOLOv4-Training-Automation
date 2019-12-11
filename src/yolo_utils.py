import os
import re
import json
import subprocess

from pathlib import Path
from tensorboardX import SummaryWriter

current_map: float = None
current_iteration: int = 0
tensorboard_writer: bool = False
working_dir: str = None
summaries_path: Path = None
status_path: Path = None
pid_path: Path = None

# Create summary writer for tensorboard
writer: SummaryWriter = SummaryWriter(flush_secs=1)


def get_prediction(
    darknet_path: str,
    dot_data_path: str,
    yolo_cfg_path: str,
    weights_path: str,
    image: str,
) -> None:
    command: list = [
        darknet_path,
        "detector",
        "test",
        dot_data_path,
        yolo_cfg_path,
        weights_path,
        image,
        "-dont_show",
    ]
    with open(os.devnull, "w") as DEVNULL:
        subprocess.call(command, stderr=DEVNULL, stdout=DEVNULL)


def update_arg(tensorboard: bool, working_directory: str) -> None:
    global tensorboard_writer
    tensorboard_writer = tensorboard
    global working_dir
    working_dir = working_directory
    global summaries_path
    summaries_path = working_dir / "summaries.txt"
    global status_path
    status_path = working_dir / "status.txt"
    global pid_path
    pid_path = working_dir / "pid.txt"

def update_summary_map(summary_map: float) -> int:
    global current_map
    global current_iteration
    current_map = summary_map
    return current_iteration


# Define the summary of the yolo output in the needed format needed by the API
def define_summary(
    current_iter: str,
    total_loss: str,
    avg_loss_error: str,
    current_lr: str,
    total_time: str,
    nb_images: str,
) -> dict:
    global current_map
    global current_iteration
    if current_iter:
        current_iteration = int(current_iter)
    else:
        current_iteration += 1
    result: dict = {
        "current_training_iteration": str(current_iteration),
        "total_loss": total_loss,
        "average_loss_error": avg_loss_error,
        "current_learning_rate": current_lr,
        "total_time": total_time,
        "number_of_images": nb_images,
        "mAP": current_map,
    }

    return result


# Define the status of the yolo output in the needed format needed by the API
def define_status(
    iou: str,
    _cls: str,
    region: str,
    avg_iou: str,
    avg_giou: str,
    _class: str,
    obj: str,
    no_obj: str,
    R5: str,
    R75: str,
    count: str,
) -> dict:
    result: dict = {
        "normalizer": {"iou": iou, "cls": _cls},
        "region {}".format(region): {"avg": {"iou": avg_iou, "giou": avg_giou}},
        "class": _class,
        "obj": obj,
        "no_obj": no_obj,
        ".5R": R5,
        ".75R": R75,
        "count": count,
    }

    return result


def process_output(line):
    summary: list = re.findall(
        "(\d+)(: )(\d+\.\d+)(, )(\d+\.\d+)( avg loss, )(\d+\.\d+)( rate, )(\d+\.\d+)( seconds, )(\d+)( images *\\n*)",
        line,
    )
    summary_map: list = re.findall(
        "(mean_average_precision \(mAP@0.5\) *= *)([0-9]*\.[0-9]*)", line
    )
    status: list = re.findall(
        "(.*iou: *)(\d+\.\d+)(, cls: )(\d+\.\d+)(\) Region *)(\d+)( Avg \(IOU: )(\d+\.\d+)(, *GIOU: )(\d+\.\d+)(\), Class: *)(\d+\.\d+)(, Obj: )(\d+\.\d+)(, No Obj: )(\d+\.\d+)(, .5R: )(\d+\.\d+)(, .75R: )(\d+\.\d+)(, count: )(\d+)",
        line,
    )
    training_ended: list = re.findall("Saving weights to .*_final.weights", line)

    global tensorboard_writer
    global working_dir
    global writer

    if summary:
        last_summary: list = summary[0]
        result: dict = define_summary(
            current_iter=last_summary[0],
            total_loss=last_summary[2],
            avg_loss_error=last_summary[4],
            current_lr=last_summary[6],
            total_time=last_summary[8],
            nb_images=last_summary[10],
        )
        if tensorboard_writer:
            writer.add_scalar(
                "loss",
                float(result["total_loss"]),
                int(result["current_training_iteration"]),
            )
        with open(summaries_path, "a") as summaries_file:
            summaries_file.write("{}\n".format(json.dumps(result)))

    if status:
        last_status: list = status[0]
        result: dict = define_status(
            iou=last_status[1],
            _cls=last_status[3],
            region=last_status[5],
            avg_iou=last_status[7],
            avg_giou=last_status[9],
            _class=last_status[11],
            obj=last_status[13],
            no_obj=last_status[15],
            R5=last_status[17],
            R75=last_status[19],
            count=last_status[21],
        )
        with open(status_path, "a") as status_file:
            status_file.write("{}\n".format(json.dumps(result)))

    if summary_map:
        summary_map = summary_map[0][1]
        iteration_nb: int = update_summary_map(summary_map)
        if tensorboard_writer:
            writer.add_scalar("mAP", float(summary_map), int(iteration_nb))

    if training_ended:
        with open(pid_path, "w") as pid:
            pid.write("Done")
