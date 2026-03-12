from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from fastapi.responses import FileResponse
from manager import process_audio_detailed

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Audio Translator API Running"}

@app.post("/translate")
async def translate_audio_endpoint(file: UploadFile = File(...), language: str = "ta"):
    # Generate unique input filename to avoid collisions
    input_path = f"uploads/input_{uuid.uuid4().hex}.wav"
    os.makedirs("uploads", exist_ok=True)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run detailed pipeline
    result = process_audio_detailed(input_path, language)

    if "error" in result:
        return {"status": "error", "message": result["error"]}

    # Convert local paths to download URLs
    audio_url = f"/download/audio/{os.path.basename(result['audio_file'])}"
    metadata_url = f"/download/metadata/{os.path.basename(result['metadata_file'])}"

    return {
        "status": "success",
        "audio_url": audio_url,
        "metadata_url": metadata_url,
        "segments": result["segments"]
    }

@app.get("/download/audio/{filename}")
async def download_audio(filename: str):
    file_path = os.path.join("uploads", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
    return {"error": "File not found"}

@app.get("/download/metadata/{filename}")
async def download_metadata(filename: str):
    file_path = os.path.join("uploads", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/plain", filename=filename)
    return {"error": "File not found"}