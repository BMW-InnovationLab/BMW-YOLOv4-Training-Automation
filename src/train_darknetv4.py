import os
import json
import socket
import shutil
import logging
import subprocess

from train import Coach
from pathlib import Path
from yolo_utils import *
from train_utils import *
from PIL import Image, JpegImagePlugin
from logging.handlers import RotatingFileHandler


class DarknetCoachV4(Coach):

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, model_name: str = "yolov4", generate_custom_anchors: bool = True, angle: int = 0,
                 hue: float = 0.1, exposure: float = 1.5, saturation: float = 1.5, calculate_map: bool = True,
                 web_ui: bool = False, web_ui_port: int = 8090, learning_rate_yolov3: float = 0.001,
                 mosaic: bool = False,
                 blur: bool = False, learning_rate_yolov4: float = 0.00261, *args,
                 **kwargs) -> None:
        print("using yolo v4 version")
        super().__init__(model_name=model_name, *args, **kwargs)
        self._images_path: Path = self._working_dir / "assets/images"
        self._labels_path: Path = self._working_dir / "assets/labels/yolo"
        self._train_txt_path: Path = self._working_dir / "assets/train.txt"
        self._test_txt_path: Path = self._working_dir / "assets/test.txt"
        self._model2config_path: Path = self._working_dir / "model2config.json"
        self._generate_custom_anchors: bool = generate_custom_anchors
        self._angle: int = angle
        self._saturation: float = saturation
        self._exposure: float = exposure
        self._hue: float = hue
        self._calculate_map: bool = calculate_map
        self._web_ui: bool = web_ui
        self._web_ui_port: int = web_ui_port
        self._learning_rate_yolov3: float = learning_rate_yolov3
        self._learning_rate_yolov4: float = learning_rate_yolov4
        self._mosaic: bool = mosaic
        self._blur: bool = blur

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # TODO is_data_valid() return true false
    def check_data(self):
        self._logger.info("Checking if all data is valid")
        weights_path: Path = None
        if self._custom_weights:
            weights_path = self._working_dir / "assets" / self._custom_weights_name

        data_checkup(
            framework="darknet",
            model2config_path=self._model2config_path,
            classes=self._classes,
            images_path=self._images_path,
            labels_path=self._labels_path,
            model_name=self._model_name,
            weights_path=weights_path,
            extension=".txt",
        )

    # TODO parameters to where to create the files and how
    # return something to know about failure or success
    def create_output_files(self):
        # Symlink images and labels folders to same 'dataset' folder
        # since darknet requires them to be in the same directory
        dataset_path: Path = self._working_dir / "dataset"
        self._dataset_path: Path = Path(dataset_path)
        if dataset_path.exists():
            shutil.rmtree(dataset_path)

        Path.mkdir(dataset_path)
        self._logger.info("Loading all images")
        for _ in tqdm(range(1)):
            command = 'for f in {}; do ln -sf "$f" {}; done'.format(
                str(self._images_path / "*"), str(dataset_path)
            )
            os.system(command)
        self._logger.info("Loading all labels")
        for _ in tqdm(range(1)):
            command = 'for f in {}; do ln -sf "$f" {}; done'.format(
                str(self._labels_path / "*"), str(dataset_path)
            )
            os.system(command)

        # Create custom_folder ex: hello00_20190822:15:26:20
        self._logger.info("Creating your training folder")

        _custom_training_folder_path: Path = self._custom_training_dir / Path(
            os.getenv('TRAIN_NAME') + "_" + os.getenv("TRAIN_START_TIME"))
        if (os.path.exists(_custom_training_folder_path)):
            shutil.rmtree(_custom_training_folder_path)
        self._custom_training_folder_path: Path = _custom_training_folder_path
        Path.mkdir(_custom_training_folder_path)

        # Create config and weights folders
        self._logger.info("Creating needed folders: weights and config")
        self._custom_config_path: Path = self._custom_training_folder_path / "config"
        self._custom_weights_path: Path = self._custom_training_folder_path / "weights"

        Path.mkdir(self._custom_config_path)
        Path.mkdir(self._custom_weights_path)

        with open(self._model2config_path, "r", encoding="utf-8") as convertionReader:
            self._model2config: dict = json.load(convertionReader)

        if self._custom_weights:
            weights_path: str = self._working_dir / "assets" / self._custom_weights_name
        else:
            weights_path: str = Path(
                self._model2config.get("darknet").get(self._model_name).get("weights")
            )

        self._logger.info(
            "Copying {} file to needed location".format(weights_path.stem)
        )
        copy_weights_file(
            source=weights_path,
            destination=self._custom_weights_path,
            weights_name="initial.weights",
        )

    def split_train_test(self):
        destination_train_txt_path: Path = self._custom_training_folder_path / "train.txt"
        destination_test_txt_path: Path = self._custom_training_folder_path / "test.txt"
        if self._train_txt_path.exists():
            self._logger.info(
                "Updating paths in train.txt and copying it to needed location"
            )
            with open(self._train_txt_path, "r", encoding="utf-8") as train_txt_file:
                train_txt: str = "".join(train_txt_file.readlines())

            with open(
                    destination_train_txt_path, "w", encoding="utf-8"
            ) as train_txt_file:
                train_txt_file.write(
                    update_label_file(
                        file_txt=train_txt, dataset_path=self._dataset_path,
                    )
                )

            if self._test_txt_path.exists():
                self._logger.info(
                    "Updating paths in test.txt and copying it to needed location"
                )
                with open(self._test_txt_path, "r", encoding="utf-8") as test_txt_file:
                    test_txt: str = "".join(test_txt_file.readlines())

                with open(
                        destination_test_txt_path, "w", encoding="utf-8"
                ) as test_txt_file:
                    test_txt_file.write(
                        update_label_file(
                            file_txt=test_txt, dataset_path=self._dataset_path,
                        )
                    )

                test_txt_path: Path = destination_test_txt_path

            else:
                test_txt_path: Path = None

        else:
            self._logger.info("train.txt and test.txt not found")
            self._logger.info("Creating train.txt and test.txt in needed location")
            train_images_list, test_images_list = split_train_test(
                train_ratio=self._train_ratio,
                classes_count=len(self._classes),
                images_path=self._dataset_path,
                labels_path=self._dataset_path,
                extension=".txt",
            )

            with open(
                    self._custom_training_folder_path / "train.txt", "w+"
            ) as train_file:
                train_file.writelines("\n".join(train_images_list))

            with open(
                    self._custom_training_folder_path / "test.txt", "w+"
            ) as test_file:
                test_file.writelines("\n".join(test_images_list))

            test_txt_path: Path = destination_test_txt_path

        self._train_txt_path = destination_train_txt_path
        self._test_txt_path = test_txt_path

        prediction_image: str = None
        if test_txt_path:
            self._logger.info("Choosing an image from test.txt for inference")
            with open(test_txt_path, "r") as test_txt_file:
                test_images_list: list = test_txt_file.readlines()
                if test_images_list:
                    prediction_image = test_images_list[0].rstrip()
                    self._prediction_image_extension: str = Path(
                        prediction_image
                    ).suffix
                    draw_bb(
                        image=prediction_image,
                        classes=self._classes,
                        destination=self._working_dir / "ground_truth.jpg",
                    )

        self._prediction_image: str = prediction_image

    # Create obj.names and obj.data
    def create_training_files(self):
        self._logger.info("Creating .data file")
        dot_data: str = "{}.data".format(self._name)
        self._logger.info("Creating .names file")
        dot_names: str = "{}.names".format(self._name)
        _custom_dot_data_path: Path = self._custom_config_path / dot_data
        _custom_dot_names_path: Path = self._custom_config_path / dot_names

        with open(_custom_dot_names_path, "w", encoding="utf-8") as obj_names_file:
            obj_names_file.write(create_obj_names(classes=self._classes))

        with open(_custom_dot_data_path, "w", encoding="utf-8") as obj_data_file:
            obj_data_file.write(
                create_obj_data(
                    classes=self._classes,
                    train_txt_path=self._train_txt_path,
                    test_txt_path=self._test_txt_path,
                    dot_names_path=_custom_dot_names_path,
                    saved_weights_path=self._custom_weights_path,
                )
            )

        self._custom_dot_data_path: Path = _custom_dot_data_path

    def generate_anchors(self):
        _darknet_dir: Path = self._working_dir / "darknet"
        _darknet_exec: Path = _darknet_dir / "darknet"
        self._darknet_exec: str = str(_darknet_exec)
        self._custom_anchors: str = None

        if self._generate_custom_anchors:
            self._logger.info("Recalculating anchors")
            command: list = [
                self._darknet_exec,
                "detector",
                "calc_anchors",
                self._custom_dot_data_path,
                "-num_of_clusters",
                str(9),
                "-width",
                str(self._image_width),
                "-height",
                str(self._image_height),
            ]
            for _ in tqdm(range(1)):
                with open(os.devnull, "w") as DEVNULL:
                    subprocess.call(command, stdout=DEVNULL, stderr=DEVNULL)

            anchors_path: Path = self._working_dir / "anchors.txt"
            with open(anchors_path, "r") as anchors_file:
                self._custom_anchors = anchors_file.read()

    def update_config(self):
        config_path: str = self._model2config.get("darknet").get(self._model_name).get(
            "config"
        )
        with open(config_path, "r", encoding="utf-8") as default_yolo_cfg_file:
            default_yolo_cfg: str = "".join(default_yolo_cfg_file.readlines())

        saved_config_path: Path = self._custom_config_path / self._model_name
        saved_config_path: str = "{}.cfg".format(str(saved_config_path))
        self._logger.info("Modifying configuration file as needed")
        with open(saved_config_path, "w", encoding="utf-8") as customize_cfg_file:
            customize_cfg_file.write(
                customize_yolo_cfg_v4(
                    yolo_cfg=default_yolo_cfg,
                    classes_nb=len(self._classes),
                    batch_size=self._batch_size,
                    subdivisions=self._subdivisons,
                    max_batches=self._max_batches,
                    image_width=self._image_width,
                    image_height=self._image_height,
                    angle=self._angle,
                    saturation=self._saturation,
                    exposure=self._exposure,
                    hue=self._hue,
                    custom_anchors=self._custom_anchors,
                    learning_rate_yolov4=self._learning_rate_yolov4,
                    mosaic=self._mosaic,
                    blur=self._blur
                )
            )

    def start_training(self):
        yolo_cfg_path: str = "{}.cfg".format(
            str(self._custom_config_path / self._model_name)
        )
        command: list = [
            self._darknet_exec,
            "detector",
            "train",
            str(self._custom_dot_data_path),
            yolo_cfg_path,
            str(self._custom_weights_path / "initial.weights"),
            "-dont_show",
            "-clear",
            "1",
        ]

        if self._gpus:
            arg: str = ",".join(str(gpu) for gpu in self._gpus)
            command.append("-gpus")
            command.append(arg)

        if self._web_ui:
            self._logger.info(
                "Running web_ui on port {}".format(str(self._web_ui_port))
            )
            command.append("-mjpeg_port")
            command.append(str(self._web_ui_port))

        if self._calculate_map:
            command.append("-map")

        if self._custom_api:
            self._logger.info(
                "Running YOLO API on port {}".format(str(self._custom_api_port))
            )
            with open(os.devnull, "w") as DEVNULL:
                subprocess.Popen(
                    [
                        "uvicorn",
                        "api:app",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(self._custom_api_port),
                    ],
                    stderr=DEVNULL,
                    stdout=DEVNULL,
                )

        if self._tensorboard:
            self._logger.info(
                "Running Tensorboard on port {}".format(str(self._tensorboard_port))
            )
            with open(os.devnull, "w") as DEVNULL:
                subprocess.Popen(
                    ["tensorboard", "--logdir", "./runs","--port", str(self._tensorboard_port)], stderr=DEVNULL, stdout=DEVNULL
                )

        # Create a rotating log
        yolo_training_logger: logging.Logger = logging.getLogger("Rotating Log")
        yolo_training_logger.setLevel(logging.INFO)

        # Add a rotating handler for yolo training output
        yolo_log_path = self._custom_training_folder_path / "yolo_events.log"
        yolo_handler: RotatingFileHandler = RotatingFileHandler(
            yolo_log_path, maxBytes=51200, backupCount=1
        )
        yolo_training_logger.addHandler(yolo_handler)
        Path.mkdir(self._working_dir / "data")
        os.symlink(
            self._working_dir / "darknet/data/labels", self._working_dir / "data/labels"
        )

        update_arg(self._tensorboard, self._working_dir)
        self._logger.info("Starting YOLO training")
        process: subprocess.Popen = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        print(
            "\nYou can now monitor the training using any of the provided means or by viewing the logs saved in the custom training folder\n"
        )
        pid_path = self._working_dir / "pid.txt"
        with open(pid_path, "w+") as pid:
            pid.write(str(process.pid))

        last_weights_path = "{}/{}_last.weights".format(
            str(self._custom_weights_path), self._model_name
        )
        labeled_prediction_image_path = "{}/predictions.jpg".format(
            str(self._working_dir)
        )

        step: int = 0
        while True:
            line: bytes = process.stdout.readline()
            if not line:
                break
            line_decoded: str = line.decode("utf-8")
            yolo_training_logger.info(line_decoded)
            process_output(line_decoded)
            checkpoint: list = re.findall(
                "Saving weights to .*_last.weights", line_decoded
            )
            if checkpoint:
                if self._prediction_image:
                    get_prediction(
                        darknet_path=self._darknet_exec,
                        dot_data_path=str(self._custom_dot_data_path),
                        yolo_cfg_path=yolo_cfg_path,
                        weights_path=last_weights_path,
                        image=self._prediction_image,
                    )
                    if self._tensorboard:
                        img: JpegImagePlugin.JpegImageFile = Image.open(
                            labeled_prediction_image_path
                        )
                        img: np.ndarray = np.transpose(img, (2, 0, 1))
                        writer.add_image("predictions", img, step)
                        step += 1
