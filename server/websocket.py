import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app import face_detection

ws = FastAPI()

# Your CORS
ws.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

users = {}
@ws.websocket('/wss')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        img = await websocket.receive_text()
        img = img.split(",")[1] if "data:image" in img else img

        list_boxes = face_detection.get_boxes(img)

        await websocket.send_text(json.dumps(list_boxes))