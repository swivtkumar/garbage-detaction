import tensorflow as tf
import cv2 as cv
import os
import numpy as np
import json
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Any
from tensorflow.keras.applications import xception
import uvicorn
from  pathlib import Path
import random
import string


app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class GarbageResponseSchema(BaseModel):
    status_code: int = 200
    results: list[Any]


class DefaultSchema(BaseModel):
    message: str
    status_code: int


def predict_result(file_path:str):
    config_path = os.path.join(BASE_DIR, 'config/config.json')
    model_path = os.path.join(BASE_DIR, 'garbage.h5')

    with open(config_path, 'r') as f:
        model = tf.keras.models.model_from_json(f.read())

    model.load_weights(model_path)
    read_img = cv.imread(file_path)
    resize_image = cv.resize(read_img, (320,320))
    input_image = np.expand_dims(resize_image, axis=0)
    output = model.predict(input_image)

    # delete file after response
    if os.path.exists(file_path):
        os.remove(file_path)

    return output[0].tolist()


async def process_output_garbage_result(image: Any, file_path: str):
    config_path = os.path.join(BASE_DIR, 'garbage_model.json')
    model_path = os.path.join(BASE_DIR, 'garbage_model_weights.h5')
    with open(config_path) as f:
        config = json.load(f)

    model = tf.keras.models.model_from_config(config)
    model.load_weights(model_path)
    image = xception.preprocess_input(image)
    result = model(np.array([image]))
    tf_constant = tf.constant(result)
    # delete file after response
    #if os.path.exists(file_path):
    #    os.remove(file_path)

    return tf_constant[0].numpy().tolist()


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

    uploads_path = os.path.join(BASE_DIR, 'uploads')
    image_name = ''.join(random.choice(string.ascii_lowercase) for i in range(7)) + image.filename
    file_path = os.path.join(uploads_path, image_name)
    with open(file_path, "wb") as buffer:
        buffer.write(await image.read())

    #result = await process_output_garbage_result(image=resize_image, file_path=file_path)
    result = predict_result(file_path=file_path)

    return GarbageResponseSchema(
        status_code=200,
        results=result,
        
    )


if __name__ == "__main__":
    uvicorn.run(f"{Path(__file__).stem}:app", host='127.0.0.1', port=9000, reload=True)

