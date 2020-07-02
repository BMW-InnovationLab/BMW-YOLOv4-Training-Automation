from pathlib import Path
from train_darknet import *
from train_darknetv4 import *
from exception_utils import *


class CoachFactory(object):
    def init_coach_from_config(self, train_config_path: str) -> Coach:
        if not Path(train_config_path).exists():
            raise ConfigError(
                "train_config.json", "Configuration file was not provided"
            )

        with open(train_config_path, "r", encoding="utf-8") as yolo_train_config_file:
            train_config: dict = json.load(yolo_train_config_file)

        # TODO add JSON Schema validation because this is bad

        framework: str = train_config.get("model").get("framework")
        model_name: str = train_config.get("model").get("model_name")
        classes: list = train_config.get("data").get("classes")
        name: str = train_config.get("data").get("name", "obj")
        train_ratio: float = train_config.get("data").get("train_ratio", 0.8)
        batch_size: int = train_config.get("model").get("batch_size", 2)
        subdivisions: int = train_config.get("model").get("subdivisions", 1)
        image_width: int = train_config.get("model").get("train_image_width", 416)
        image_height: int = train_config.get("model").get("train_image_height", 416)
        max_batches: int = train_config.get("model").get("max_batches", None)
        learning_rate_yolov3 : float = train_config.get("model").get("yolov3_config").get("learning_rate", 0.001)


        # modification by hadi to get yolov4 specific variable of data augmentation
        learning_rate_yolov4 : float = train_config.get("model").get("yolov4_config").get("learning_rate",0.0013)
        mosaic: bool = train_config.get("model").get("yolov4_config").get("mosaic", False)
        blur: bool = train_config.get("model").get("yolov4_config").get("blur", False)
        # end of modification

        gpus: list = train_config.get("training").get("gpus", None)
        custom_api_el = train_config.get("training").get(
            "custom_api", {"enable": False, "port": 8000}
        )
        custom_api: bool = custom_api_el["enable"]
        custom_api_port: int = custom_api_el["port"]
        dashboard_el = train_config.get("training").get(
            "dashboard", {"enable": False, "port": 3000}
        )
        dashboard: bool = dashboard_el["enable"]
        dashboard_port: int = dashboard_el["port"]
        tensorboard_el = train_config.get("training").get(
            "tensorboard", {"enable": False, "port": 6006}
        )
        tensorboard: bool = tensorboard_el["enable"]
        tensorboard_port: int = tensorboard_el["port"]
        custom_weights_el = train_config.get("model").get(
            "custom_weights", {"enable": False, "name": "darknet53.conv.74"}
        )
        custom_weights: bool = custom_weights_el["enable"]
        custom_weights_name: str = custom_weights_el["name"]

        if not framework:
            raise ConfigError(framework, "Framework was not provided")

        if not model_name:
            raise ConfigError(model_name, "Model name was not provided")

        if not classes:
            raise ConfigError(classes, "Classes were not provided")

        if not max_batches:
            max_batches = 2000 * len(classes)

        if str(framework).lower() == "darknet"and str(model_name).lower() == "yolov3":
            generate_custom_anchors: bool = train_config.get("model").get(
                "generate_custom_anchors", False
            )
            angle: int = train_config.get("model").get("angle", 0)
            saturation: float = train_config.get("model").get("saturation", 1.5)
            exposure: float = train_config.get("model").get("exposure", 1.5)
            hue: float = train_config.get("model").get("hue", 0.1)
            calculate_map: bool = train_config.get("training").get(
                "calculate_map", True
            )
            web_ui_el = train_config.get("training").get(
                "web_ui", {"enable": False, "port": 8090}
            )
            web_ui: bool = web_ui_el["enable"]
            web_ui_port: int = web_ui_el["port"]

            return DarknetCoach(
                model_name=model_name,
                name=name,
                train_ratio=train_ratio,
                classes=classes,
                generate_custom_anchors=generate_custom_anchors,
                batch_size=batch_size,
                max_batches=max_batches,
                subdivisions=subdivisions,
                image_width=image_width,
                image_height=image_height,
                angle=angle,
                saturation=saturation,
                exposure=exposure,
                hue=hue,
                gpus=gpus,
                calculate_map=calculate_map,
                custom_api=custom_api,
                custom_api_port=custom_api_port,
                dashboard=dashboard,
                dashboard_port=dashboard_port,
                tensorboard=tensorboard,
                tensorboard_port=tensorboard_port,
                web_ui=web_ui,
                web_ui_port=web_ui_port,
                custom_weights=custom_weights,
                custom_weights_name=custom_weights_name,
                learning_rate_yolov3 = learning_rate_yolov3
            )
        elif str(framework).lower() == "darknet" and str(model_name).lower() == "yolov4":
            generate_custom_anchors: bool = train_config.get("model").get(
                "generate_custom_anchors", False
            )
            angle: int = train_config.get("model").get("angle", 0)
            saturation: float = train_config.get("model").get("saturation", 1.5)
            exposure: float = train_config.get("model").get("exposure", 1.5)
            hue: float = train_config.get("model").get("hue", 0.1)
            calculate_map: bool = train_config.get("training").get(
                "calculate_map", True
            )
            web_ui_el = train_config.get("training").get(
                "web_ui", {"enable": False, "port": 8090}
            )
            web_ui: bool = web_ui_el["enable"]
            web_ui_port: int = web_ui_el["port"]

            return DarknetCoachV4(
                model_name=model_name,
                name=name,
                train_ratio=train_ratio,
                classes=classes,
                generate_custom_anchors=generate_custom_anchors,
                batch_size=batch_size,
                max_batches=max_batches,
                subdivisions=subdivisions,
                image_width=image_width,
                image_height=image_height,
                angle=angle,
                saturation=saturation,
                exposure=exposure,
                hue=hue,
                gpus=gpus,
                calculate_map=calculate_map,
                custom_api=custom_api,
                custom_api_port=custom_api_port,
                dashboard=dashboard,
                dashboard_port=dashboard_port,
                tensorboard=tensorboard,
                tensorboard_port=tensorboard_port,
                web_ui=web_ui,
                web_ui_port=web_ui_port,
                custom_weights=custom_weights,
                custom_weights_name=custom_weights_name,
                learning_rate_yolov4=learning_rate_yolov4,
                mosaic=mosaic,
                blur=blur,
            )        

        else:
            raise ConfigError(framework, "No such available framework")
