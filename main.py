from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from manager import process_audio_detailed, warm_up_engines
import threading
import asyncio

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Director Rule 13: Background Pre-warming ---
@app.on_event("startup")
async def startup_event():
    """Warms up models in the background after server start."""
    # We do this in a thread to not block the server from accepting connections
    # though FastAPI's startup event can be async.
    # To be safe and fast, we trigger it immediately.
    print("Director: Server starting, initiating background warm-up...")
    threading.Thread(target=warm_up_engines, daemon=True).start()

# Serve static files (CSS, JS)
# We don't have a separate static dir, so we serve from root for simplicity in this project
# Alternatively, we could create a 'static' folder.
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/style.css")
async def serve_css():
    return FileResponse("style.css", media_type="text/css")

@app.get("/script.js")
async def serve_js():
    return FileResponse("script.js", media_type="application/javascript")

@app.post("/translate")
async def translate_audio_endpoint(file: UploadFile = File(...), language: str = "ta"):
    # Generate unique input filename to avoid collisions
    input_path = f"uploads/input_{uuid.uuid4().hex}.wav"
    os.makedirs("uploads", exist_ok=True)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run detailed pipeline
    result = await process_audio_detailed(input_path, language)

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