from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
from manager import process_audio

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_PATH = "uploads/input.wav"


@app.get("/")
def home():
    return {"message": "AI Audio Translator API Running"}


@app.post("/translate")
async def translate_audio(file: UploadFile = File(...), language: str = "ta"):

    with open(UPLOAD_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output = process_audio(UPLOAD_PATH, language)

    return {
        "status": "success",
        "translated_audio": output
    }