import sys
import logging

from pathlib import Path
from time import time, strftime
from abc import ABC, abstractmethod


class Coach(ABC):
    """
    This is the base class for all training frameworks
    """

    def __init__(
        self,
        name: str = "obj",
        enable_training = False,
        classes: list = [],
        train_ratio: float = 0.8,
        model_name: str = None,
        batch_size: int = 2,
        subdivisions: int = 1,
        channels: int = 3,
        image_width: int = 416,
        image_height: int = 416,
        max_batches: int = None,
        gpus: list = None,
        custom_api: bool = True,
        custom_api_port: int = 8000,
        dashboard: bool = True,
        dashboard_port: int = 3000,
        tensorboard: bool = True,
        tensorboard_port: int = 6006,
        custom_weights: bool = False,
        custom_weights_name: str = "darknet53.conv.74",
        *args,
        **kwargs
    ) -> None:

        self._working_dir: Path = Path.cwd()
        self._name: str = name
        self._enable_training: str = enable_training
        self._classes: list = classes
        self._train_ratio: int = train_ratio
        self._model_name: int = model_name
        self._batch_size: int = batch_size
        self._subdivisons: int = subdivisions
        self._channels: int = channels
        self._image_width: int = image_width
        self._image_height: int = image_height
        self._max_batches: int = max_batches
        self._gpus: list = gpus
        self._custom_api: bool = custom_api
        self._custom_api_port: int = custom_api_port
        self._dashboard: bool = dashboard
        self._dashboard_port: int = dashboard_port
        self._tensorboard: bool = tensorboard
        self._tensorboard_port: int = tensorboard_port
        self._start_time: str = strftime("%Y%m%d_%H:%M:%S")
        self._custom_training_dir: Path = Path.cwd() / "custom_training"
        self._custom_weights: bool = custom_weights
        self._custom_weights_name: str = custom_weights_name

        # Creating a logger
        logger = logging.getLogger("Events Log")
        logger.setLevel(logging.INFO)

        # Add a rotating handler for logging
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(name)-12s: %(levelname)-8s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self._logger = logger

        Path.mkdir(self._custom_training_dir, exist_ok=True)
        logger.info("Creating custom trainings folder")

    # Check if data is valid
    @abstractmethod
    def check_data(self) -> None:
        raise NotImplementedError

    # Create the training folder and subfolders containing
    # configuration file, data and pretrained model
    @abstractmethod
    def create_output_files(self) -> None:
        raise NotImplementedError

    # Split the data into train and test
    @abstractmethod
    def split_train_test(self) -> None:
        raise NotImplementedError

    # Create the file/files needed in the correct structure for training
    @abstractmethod
    def create_training_files(self) -> None:
        raise NotImplementedError

    # Generate custom anchors in case of Darknet framework
    def generate_anchors(self) -> None:
        pass

    # Modify the configuration file according to the specified training parameters
    @abstractmethod
    def update_config(self) -> None:
        raise NotImplementedError

    # Execute training command
    @abstractmethod
    def start_training(self) -> None:
        raise NotImplementedError

    # Template method to execute all needed steps in the correct order
    # TODO use asserts to make sure the functions ran as expected
    def train(self) -> None:
        self.check_data()
        self.create_output_files()
        self.split_train_test()
        self.create_training_files()
        self.generate_anchors()
        self.update_config()
        self.start_training()
