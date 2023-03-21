import json
import os
import tensorflow as tf
import cv2 as cv
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Any


app = FastAPI()


class GarbageResponseSchema(BaseModel):
    status_code: int = 200
    results: list[Any]


class DefaultSchema(BaseModel):
    message: str
    status_code: int


def process_output_garbage_result():

    with open('config/config.json') as f:
        config = json.load(f)

    image = "ggb.png"
    image = cv.imread(image)
    image = cv.resize(image, (320, 320))

    model = tf.keras.models.model_from_config(config)
    model.load_weights('garbage.h5')
    image = tf.keras.applications.xception.preprocess_input(image)
    return model(np.array([image]))


def array_np():
    with open('config/config.json') as f:
        config = json.load(f)

    image = "ggb.png"
    image = cv.imread(image)
    image = cv.resize(image, (320, 320))

    model = tf.keras.models.model_from_config(config)
    model.load_weights('garbage.h5')
    image = tf.keras.applications.xception.preprocess_input(image)
    print(tf.__version__)
    return model(np.array([image]))


@app.get('/', response_model=DefaultSchema)
def index() -> DefaultSchema:
    return DefaultSchema(message='Garbage identifiers', status_code=200)


@app.post('/uploads', response_model=GarbageResponseSchema)
async def garbage_response(
    image: UploadFile = File(...)
) -> GarbageResponseSchema:

    if not image:
        raise HTTPException(status_code=404, detail="Please provide image")

    content_type = image.content_type.split('/')
    if content_type[0] != 'image':
        raise HTTPException(status_code=400, detail="The uploaded files is not a image")
    
    file_path = os.path.join("uploads", 'ggb.png')
    # with open(file_path, "wb") as buffer:
    #     buffer.write(await image.read())
    #

    # read_img = cv.imread(file_path)
    # resize_image = cv.resize(read_img, (320, 320))

    result = process_output_garbage_result()

    return GarbageResponseSchema(
        status_code=200,
        results=result
    )
