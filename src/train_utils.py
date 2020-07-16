import re
import json
import random
import numpy as np

from cv2 import cv2
from tqdm import tqdm
from exception_utils import *
from shutil import copy2, copyfile
from pathlib import Path, PurePath
from sklearn.model_selection import train_test_split


def copy_weights_file(source: Path, destination: Path, weights_name: str) -> None:
    for _ in tqdm(range(1)):
        copyfile(source, destination / weights_name)


def create_obj_names(classes: list) -> str:
    return "\n".join(classes)


def create_obj_data(
    classes: list,
    train_txt_path: Path,
    test_txt_path: Path,
    dot_names_path: Path,
    saved_weights_path: Path,
) -> str:

    if not test_txt_path:
        test_txt_path: Path = train_txt_path

    return "\n".join(
        [
            "classes={}".format(len(classes)),
            "train={}".format(train_txt_path),
            "valid={}".format(test_txt_path),
            "names={}".format(dot_names_path),
            "backup={}".format(saved_weights_path),
        ]
    )


def update_label_file(file_txt: str, dataset_path: Path) -> str:
    return re.sub(
        r"(.*images)([\/])([^\/]*)(\n)", "{}/{}\n".format(dataset_path, r"\3"), file_txt
    )


def split_train_test(
    train_ratio, classes_count, images_path, labels_path, extension
) -> tuple:
    train_ratio: float = float(train_ratio)
    test_ratio: float = 1 - train_ratio
    images_path_list: list = [
        str(path)
        for path in list((images_path).glob("*"))
        if path.suffix in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    ]
    images_per_class: dict = dict()
    found_labels: list = []
    # Storing all images per label
    for image_path in images_path_list:
        label_path: Path = labels_path / Path(Path(image_path).stem)
        label_path: str = "{}{}".format(str(label_path), extension)
        with open(label_path, "r") as labels_file:
            for label in labels_file.readlines():
                label_class: int = int(label.split(" ")[0])
                if label_class in images_per_class.keys():
                    images_per_class[label_class].add(image_path)
                else:
                    found_labels.append(label_class)
                    images_per_class[label_class] = set()
                    images_per_class[label_class].add(image_path)

    # Counting images per label
    images_per_class_count: list = [
        len(images_per_class[label_class]) for label_class in found_labels
    ]
    sorted_indices_images_per_class_count: np = np.argsort(images_per_class_count)
    train_images_set: set = set()
    test_images_set: set = set()
    # Splitting images per class to test train sets starting with the label having the least images
    for label_class in sorted_indices_images_per_class_count:
        images: set = images_per_class[found_labels[label_class]]
        X_train, X_test, _, _ = train_test_split(
            list(images),
            np.zeros(len(images)),
            train_size=train_ratio,
            test_size=test_ratio,
            random_state=None,
        )
        for image in X_train:
            if not image in test_images_set:
                train_images_set.add(image)
        for image in X_test:
            if not image in train_images_set:
                test_images_set.add(image)

    train_images_list: list = list(train_images_set)
    test_images_list: list = list(test_images_set)

    random.shuffle(train_images_list)
    random.shuffle(test_images_list)

    return train_images_list, test_images_list

def customize_yolo_cfg_v4(yolo_cfg: str, classes_nb: int, batch_size: int, subdivisions: int, max_batches: int,
                          image_width: int, image_height: int, angle: int, saturation: float, exposure: float,
                          hue: float,
                          custom_anchors: str, mosaic:bool, blur:bool, learning_rate_yolov4:float ) -> str:
    custom_yolo_cfg: str = yolo_cfg
    # Set classes
    custom_yolo_cfg = re.sub(
            r"(classes *= *)(.+)", r"\1 {}".format(classes_nb), custom_yolo_cfg
    )
    # Set batch_size
    custom_yolo_cfg = re.sub(
            r"(batch *= *)(.+)", r"\1 {}".format(batch_size), custom_yolo_cfg
    )
    # Set subdivisions
    custom_yolo_cfg = re.sub(
            r"(subdivisions *= *)(.+)", r"\1 {}".format(subdivisions), custom_yolo_cfg
    )
    # Set image_width
    custom_yolo_cfg = re.sub(
            r"(width *= *)(.+)", r"\1 {}".format(image_width), custom_yolo_cfg
    )
    # Set image_height
    custom_yolo_cfg = re.sub(
            r"(height *= *)(.+)", r"\1 {}".format(image_height), custom_yolo_cfg
    )
    # Set max_batches
    custom_yolo_cfg = re.sub(
            r"(max_batches *= *)(.+)", r"\1 {}".format(max_batches), custom_yolo_cfg
    )
    # Set steps
    steps = [int(0.8 * max_batches), int(0.9 * max_batches)]
    custom_yolo_cfg = re.sub(
            r"(steps *= *)(.+)",
            r"\1 {}".format("{},{}".format(steps[0], steps[1])),
            custom_yolo_cfg,
    )
    # Set angle
    custom_yolo_cfg = re.sub(
            r"(angle *= *)(.+)", r"\1 {}".format(angle), custom_yolo_cfg
    )
    # Set saturation
    custom_yolo_cfg = re.sub(
            r"(saturation *= *)(.+)", r"\1 {}".format(saturation), custom_yolo_cfg
    )
    # Set exposure
    custom_yolo_cfg = re.sub(
            r"(exposure *= *)(.+)", r"\1 {}".format(exposure), custom_yolo_cfg
    )
    # Set hue
    custom_yolo_cfg = re.sub(r"(hue *= *)(.+)", r"\1 {}".format(hue), custom_yolo_cfg)

    #set learning rate
    custom_yolo_cfg = re.sub(r"(learning_rate *= *)(.+)", r"\1 {}".format(learning_rate_yolov4), custom_yolo_cfg)
    # set mosaic augmentation
    custom_yolo_cfg = re.sub(r"(mosaic *= *)(.+)", r"\1 {}".format(int(mosaic)), custom_yolo_cfg)
    # set blur augmentation
    custom_yolo_cfg = re.sub(r"(blur *= *)(.+)", r"\1 {}".format(int(blur)), custom_yolo_cfg)
    
    # Set custom_anchors
    if custom_anchors:
        custom_yolo_cfg = re.sub(
                r"(anchors *= *)(.+)", r"\1 {}".format(custom_anchors), custom_yolo_cfg
        )
    # Set custom filters
    all_cfg_lines: list = custom_yolo_cfg.split("\n")
    yolo_indexes: list = [i for i, line in enumerate(all_cfg_lines) if "[yolo]" in line]
    for yolo_layer in yolo_indexes:
        for i, cfg_line in enumerate(reversed(all_cfg_lines[:yolo_layer])):
            if "filters" in cfg_line:
                all_cfg_lines[yolo_layer - i - 1] = re.sub(
                        r"(filters *= *)(.+)",
                        r"\1 {}".format((classes_nb + 5) * 3),
                        cfg_line,
                )
                break

    return "\n".join(all_cfg_lines)

def customize_yolo_cfg(
    yolo_cfg: str,
    classes_nb: int,
    batch_size: int,
    subdivisions: int,
    max_batches: int,
    image_width: int,
    image_height: int,
    angle: int,
    saturation: float,
    exposure: float,
    hue: float,
    custom_anchors: str,
    learning_rate_yolov3:float
) -> str:

    custom_yolo_cfg: str = yolo_cfg
    # Set classes
    custom_yolo_cfg = re.sub(
        r"(classes *= *)(.+)", r"\1 {}".format(classes_nb), custom_yolo_cfg
    )
    # Set batch_size
    custom_yolo_cfg = re.sub(
        r"(batch *= *)(.+)", r"\1 {}".format(batch_size), custom_yolo_cfg
    )
    # Set subdivisions
    custom_yolo_cfg = re.sub(
        r"(subdivisions *= *)(.+)", r"\1 {}".format(subdivisions), custom_yolo_cfg
    )
    # Set image_width
    custom_yolo_cfg = re.sub(
        r"(width *= *)(.+)", r"\1 {}".format(image_width), custom_yolo_cfg
    )
    # Set image_height
    custom_yolo_cfg = re.sub(
        r"(height *= *)(.+)", r"\1 {}".format(image_height), custom_yolo_cfg
    )
    # Set max_batches
    custom_yolo_cfg = re.sub(
        r"(max_batches *= *)(.+)", r"\1 {}".format(max_batches), custom_yolo_cfg
    )
    # Set steps
    steps = [int(0.8 * max_batches), int(0.9 * max_batches)]
    custom_yolo_cfg = re.sub(
        r"(steps *= *)(.+)",
        r"\1 {}".format("{},{}".format(steps[0], steps[1])),
        custom_yolo_cfg,
    )
    # Set angle
    custom_yolo_cfg = re.sub(
        r"(angle *= *)(.+)", r"\1 {}".format(angle), custom_yolo_cfg
    )
    # Set saturation
    custom_yolo_cfg = re.sub(
        r"(saturation *= *)(.+)", r"\1 {}".format(saturation), custom_yolo_cfg
    )
    # Set exposure
    custom_yolo_cfg = re.sub(
        r"(exposure *= *)(.+)", r"\1 {}".format(exposure), custom_yolo_cfg
    )
    # Set hue
    custom_yolo_cfg = re.sub(r"(hue *= *)(.+)", r"\1 {}".format(hue), custom_yolo_cfg)

    #set learning rate
    custom_yolo_cfg = re.sub(
        r"(learning_rate *= *)(.+)", r"\1 {}".format(learning_rate_yolov3), custom_yolo_cfg
    )
    # Set custom_anchors
    if custom_anchors:
        custom_yolo_cfg = re.sub(
            r"(anchors *= *)(.+)", r"\1 {}".format(custom_anchors), custom_yolo_cfg
        )
    # Set custom filters
    all_cfg_lines: list = custom_yolo_cfg.split("\n")
    yolo_indexes: list = [i for i, line in enumerate(all_cfg_lines) if "[yolo]" in line]
    for yolo_layer in yolo_indexes:
        for i, cfg_line in enumerate(reversed(all_cfg_lines[:yolo_layer])):
            if "filters" in cfg_line:
                all_cfg_lines[yolo_layer - i - 1] = re.sub(
                    r"(filters *= *)(.+)",
                    r"\1 {}".format((classes_nb + 5) * 3),
                    cfg_line,
                )
                break

    return "\n".join(all_cfg_lines)


def data_checkup(
    framework: str,
    model2config_path: Path,
    classes: list,
    images_path: Path,
    labels_path: Path,
    model_name: str,
    weights_path: Path,
    extension: str,
) -> None:
    # Check classes not empty
    # logger.info('Checking if classes is empty')
    if not classes:
        raise ConfigError(classes, "An empty array of classes was passed")

    # Check if images are there
    # logger.info('Checking if images are found in the provided folder')
    images_list: list = [
        path
        for path in list(Path(images_path).glob("*"))
        if path.suffix in [".png", ".jpg", "jpeg", ".PNG", ".JPG", ".JPEG"]
    ]

    if not images_list:
        raise ConfigError(images_path, "No images were found in the specified path")

    # Check if labels are there
    # logger.info('Checking if labels are found in the provided folder')
    labels_list: list = list(Path(labels_path).glob("*{}".format(extension)))
    if not labels_list:
        raise ConfigError(labels_path, "No files were found in the specified path")

    # Check if labels and images are matching
    # logger.info('Checking if labels and images are matching')
    images_list = [name.stem for name in images_list]
    labels_list = [name.stem for name in labels_list]

    images_without_labels: set = set(images_list) - set(labels_list)
    if images_without_labels:
        raise ConfigError(images_without_labels, "No labels found for these images")

    # Check if weights file exists
    with open("model2config.json", "r", encoding="utf-8") as convertionReader:
        model2config = json.load(convertionReader)

    model_config: str = model2config.get(framework).get(model_name, None)
    if not model_config:
        raise ConfigError(model_name, "Model does not exist")

    if weights_path:
        if not weights_path.exists():
            raise ConfigError(
                weights_path.stem, "Custom weights not found in the provided folder"
            )


def draw_bb(image: str, classes: list, destination: Path) -> None:
    img: np.ndarray = cv2.imread(image, cv2.IMREAD_COLOR)
    image_height, image_width, channels = img.shape

    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 1.0
    thickness: int = 3
    colors: list = [
        (255, 255, 00),
        (00, 255, 00),
        (255, 00, 255),
        (00, 255, 255),
        (00, 00, 255),
        (255, 255, 255),
    ]

    labelsName: list = classes

    detections_path: str = "{}.txt".format(str(Path(image).with_suffix("")))
    with open(detections_path, "r", encoding="utf-8") as detectionsReader:
        detections: list = detectionsReader.readlines()

        for detection in detections:
            labelId, x, y, w, h = map(float, detection.split(" ")[0:5])
            labelId: int = int(labelId)
            x *= image_width
            y *= image_height
            w *= image_width
            h *= image_height
            text_size: tuple = cv2.getTextSize(
                labelsName[labelId].rstrip(), font, font_scale, thickness
            )
            (text_width, text_height) = text_size[0]
            color: tuple = colors[labelId % len(colors)]
            cv2.rectangle(
                img,
                (int(x - (w / 2)), int(y - (h / 2))),
                (int(x + (w / 2)), int(y + (h / 2))),
                color,
                thickness,
            )

            if int(y - (h / 2) - text_height - 9) < 0:
                cv2.rectangle(
                    img,
                    (int(x - (w / 2) - 1), int(y - (h / 2))),
                    (int(x - (w / 2) + text_width), int(y - (h / 2) + text_height + 9)),
                    color,
                    cv2.FILLED,
                )
                cv2.putText(
                    img,
                    labelsName[labelId].rstrip(),
                    (int(x - (w / 2)), int(y - (h / 2) + text_height + 3)),
                    font,
                    font_scale,
                    (0, 0, 0),
                )
            else:
                cv2.rectangle(
                    img,
                    (int(x - (w / 2) - 1), int(y - (h / 2) - text_height) - 9),
                    (int(x - (w / 2) + text_width), int(y - (h / 2))),
                    color,
                    cv2.FILLED,
                )
                cv2.putText(
                    img,
                    labelsName[labelId].rstrip(),
                    (int(x - (w / 2)), int(y - (h / 2) - 3)),
                    font,
                    font_scale,
                    (0, 0, 0),
                )

    cv2.imwrite(str(destination), img)
