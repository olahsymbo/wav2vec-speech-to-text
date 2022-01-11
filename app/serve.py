import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from wav2vec_speech_to_text.train.predict import stt_predict
from wav2vec_speech_to_text.utils.logger import logger

app = FastAPI()


@app.post("/predict")
async def predict_keyword(file: UploadFile = File(...)):
    if not file:
        logger.error("No file part in the request")
        raise HTTPException(status_code=400, detail="No file part in the request")

    if file.filename == "":
        logger.error("No file selected for upload")
        raise HTTPException(status_code=400, detail="No file selected")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        prediction = stt_predict(tmp_path)
        logger.info(f"Prediction result: {prediction}")
        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"Temporary file {tmp_path} removed after prediction")
