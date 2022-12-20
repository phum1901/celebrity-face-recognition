from fastapi import FastAPI, File, UploadFile 
from recognition.celeb_face_recognition import CelebFaceRecognition

app = FastAPI()
model = CelebFaceRecognition()

@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    if image is None:
        return {
            "success": "false"
        }
    name, uid = model.predict(image.file)
    return {
        "success": "true",
        "account": name,
        "uid": uid
    }