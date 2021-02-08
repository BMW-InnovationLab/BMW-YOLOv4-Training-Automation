import os
import json
import subprocess
import numpy as np

from PIL import Image
from io import BytesIO
from src.api_utils import *
from pydantic import BaseModel
from starlette.responses import FileResponse
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.middleware.cors import CORSMiddleware

status_path: Path = working_dir / "status.txt"
summary_path: Path = working_dir / "summaries.txt"
config_path: Path = working_dir / "assets/train_config.json"
prediction_image_path: Path = working_dir / "predictions.jpg"
ground_truth_image_path = working_dir / "ground_truth.jpg"


app: FastAPI = FastAPI(
    title="BMW InnovationLab YOLO Training Automation",
    description='<b>API for Monitoring YOLO Training <br><br><br>Contact the developers:<br><font color="#808080">Nour Azzi: <a href="https://github.com/nourazzii"></a>https://github.com/nourazzii<br>Lynn Nassif: <a href=""></a><br>Hadi Koubeissy:  <a href="https://github.com/hadikoub"></a>https://github.com/hadikoub <br>BMW Innovation Lab: <a href="mailto:innovation-lab@bmw.de">innovation-lab@bmw.de</a></font></b>',
    docs_url="/",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Custom_Error(BaseModel):
    success: str
    message: str


@app.get(
    "/summary",
    responses={400: {"model": Custom_Error}},
    summary="Get last batch output",
    tags=["Batch Output"],
)
def get_summary() -> dict:
    """Returns the summary of the last iteration"""
    result = check_error()
    if result:
        return result

    if not Path(summary_path).exists():
        summary: list = []

    else:
        with open(summary_path, "r") as summaryReader:
            summary: list = summaryReader.readlines()[-1]
            summary = json.loads(summary)

    result: dict = {"success": True, "start_time": get_time(), "data": summary}

    return result


@app.get(
    "/summaries",
    responses={400: {"model": Custom_Error}},
    summary="Get all batches outputs",
    tags=["Batch Output"],
)
def get_history_summary() -> dict:
    """Returns all the summaries so far"""
    result = check_error()

    if result:
        return result

    if not Path(summary_path).exists():
        summaries: list = []

    else:

        with open(summary_path, "r") as summaryReader:
            lines: list = summaryReader.readlines()
            summaries: list = [json.loads(summary) for summary in lines]

    result: dict = {
        "success": True,
        "start_time": get_time(),
        "data": {"summary": summaries},
    }

    return result


@app.get(
    "/status",
    responses={400: {"model": Custom_Error}},
    summary="Get last subdivision output",
    tags=["Subdivision Output"],
)
def get_status() -> dict:
    """Returns the last status"""
    result = check_error()

    if result:
        return result

    if not Path(status_path).exists():
        status: list = []

    else:
        with open(status_path, "r") as statusReader:
            status: str = statusReader.readlines()[-1]
            status: list = json.loads(status)

    result = {"success": True, "start_time": get_time(), "data": status}

    return result


@app.get(
    "/statuses",
    responses={400: {"model": Custom_Error}},
    summary="Get all subdivisions outputs",
    tags=["Subdivision Output"],
)
def get_history_status() -> dict:
    """Returns all the status so far"""
    result = check_error()

    if result:
        return result

    if not Path(status_path).exists():
        statuses: list = []

    else:

        with open(status_path, "r") as statusReader:
            lines: list = statusReader.readlines()
            statuses: list = [json.loads(status) for status in lines]

    result: dict = {
        "success": True,
        "start_time": get_time(),
        "data": {"status": statuses},
    }

    return result


@app.get("/config", summary="Get configuration file", tags=["Configuration File"])
def get_configuration_file() -> dict:
    """Returns the predefined configuration file"""

    with open(config_path, "r") as config:
        result: dict = {
            "success": True,
            "start_time": get_time(),
            "data": {"configuration_file": json.load(config)},
        }

    return result


@app.get(
    "/validate",
    summary="Get predictions on a preselected validation image next to the ground-truth image",
    tags=["Inference"],
)
def get_validation():
    """Returns prediction on image with the latest saved weights \n
       Left image is the one predicted by the model and Right image is the ground truth"""
    if Path(prediction_image_path).exists():
        list_im: list = [prediction_image_path, ground_truth_image_path]
        imgs: list = [Image.open(str(i)) for i in list_im]

        min_shape: list = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb: np = np.hstack(list((np.asarray(i.resize(min_shape)) for i in imgs)))

        imgs_comb: Image = Image.fromarray(imgs_comb)
        collage_path: Path = working_dir / "collage.jpg"
        imgs_comb.save(collage_path)

        return FileResponse(collage_path, media_type="image/jpg")

    else:
        if ground_truth_image_path.exists():
            message: str = "No predictions yet"
        else:
            message: str = "No testing set was provided"

        result: dict = {"success": True, "start_time": get_time(), "message": message}
        return result


@app.post(
    "/predict_video",
    summary="Upload video and get its predictions using last saved weights",
    tags=["Inference"],
)
async def get_prediction_video(
    video: UploadFile = File(..., description="Video to perform inference on", content_type='video/mp4'),
    use_default_weights: bool = False
):
    """Runs the last saved weights to infer on the given image"""

    try:
        result = perform_prediction(video, use_default_weights, True)
    except Exception as ex:
        raise HTTPException(
            422,
            detail=f"Error while reading request image. Please make sure it is a valid image. Error: {ex}",
        )

    if 'output_path' in result:
        if not os.path.exists(result['output_path']):
            return {'error': 'No output generated.'}

        return FileResponse(result['output_path'], media_type="video/mp4")

    return result

@app.post(
    "/predict_image",
    summary="Upload image and get its predictions using last saved weights",
    tags=["Inference"],
)
async def get_prediction_image(
    image: bytes = File(..., description="Image to perform inference on"),
    use_default_weights: bool = False
):
    """Runs the last saved weights to infer on the given image"""

    try:
        result = perform_prediction(image, use_default_weights, False)
    except Exception as ex:
        raise HTTPException(
            422,
            detail=f"Error while reading request image. Please make sure it is a valid image. Error: {ex}",
        )

    if 'output_path' in result:
        if not os.path.exists(result['output_path']):
            return {'error': 'No output generated.'}

        return FileResponse(result['output_path'], media_type="image/jpg")

    return result
