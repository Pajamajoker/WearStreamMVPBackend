# server.py
import os
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

app = FastAPI()

# Audio config
SAMPLE_RATE = 16000        # Hz
CHANNELS = 1               # mono
BITS_PER_SAMPLE = 16       # 16-bit PCM
BYTES_PER_SAMPLE = BITS_PER_SAMPLE // 8
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE

TARGET_SECONDS = 10
TARGET_BYTES = BYTES_PER_SECOND * TARGET_SECONDS

RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# In-memory buffers per client
buffers: Dict[str, bytearray] = {}


@app.get("/")
async def root():
    return {"status": "ok", "message": "Audio WebSocket server running"}


@app.get("/debug")
async def debug():
    return JSONResponse({
        "active_buffers": {k: len(v) for k, v in buffers.items()},
        "target_bytes": TARGET_BYTES,
    })


@app.websocket("/ws/audio/{device_id}")
async def audio_websocket(websocket: WebSocket, device_id: str):
    """
    WebSocket endpoint for audio streaming.
    Client should connect to: ws://<host>:8000/ws/audio/{device_id}
    """
    await websocket.accept()
    client_key = f"{device_id}_{id(websocket)}"
    buffers[client_key] = bytearray()

    print(f"[WS OPEN] device_id={device_id}, client_key={client_key}")

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk_size = len(data)
            print(f"[RECV] device_id={device_id} chunk={chunk_size} bytes")

            buf = buffers[client_key]
            buf.extend(data)
            print(f"[BUFFER] device_id={device_id} size={len(buf)} bytes (target={TARGET_BYTES})")

            if len(buf) >= TARGET_BYTES:
                print(f"[SAVE] device_id={device_id} reached ~10s of audio")
                save_buffer_as_wav(device_id, buf)
                buf.clear()
                await websocket.send_text("saved_10s_clip")
    except WebSocketDisconnect:
        print(f"[WS CLOSE] device_id={device_id}, client_key={client_key}")
    except Exception as e:
        print(f"[WS ERROR] device_id={device_id}: {e}")
    finally:
        buffers.pop(client_key, None)


def save_buffer_as_wav(device_id: str, buffer: bytearray):
    """
    Save PCM buffer as WAV.
    """
    pcm_bytes = bytes(buffer)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{device_id}_{timestamp}.wav"
    filepath = os.path.join(RECORDINGS_DIR, filename)

    print(f"[WAV] Writing file: {filepath}, bytes={len(pcm_bytes)}")

    header = create_wav_header(
        num_channels=CHANNELS,
        sample_rate=SAMPLE_RATE,
        bits_per_sample=BITS_PER_SAMPLE,
        num_samples=len(pcm_bytes) // BYTES_PER_SAMPLE,
    )

    try:
        with open(filepath, "wb") as f:
            f.write(header)
            f.write(pcm_bytes)
        print(f"[WAV] Saved: {filepath}")
    except Exception as e:
        print(f"[WAV ERROR] Failed to save {filepath}: {e}")


def create_wav_header(num_channels: int, sample_rate: int,
                      bits_per_sample: int, num_samples: int) -> bytes:
    """
    Create a standard 44-byte WAV header.
    """
    import struct

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * num_channels * bits_per_sample // 8
    riff_chunk_size = 36 + data_size

    header = b"RIFF"
    header += struct.pack("<I", riff_chunk_size)
    header += b"WAVE"

    # fmt subchunk
    header += b"fmt "
    header += struct.pack("<I", 16)                  # Subchunk1Size for PCM
    header += struct.pack("<H", 1)                   # AudioFormat = 1 (PCM)
    header += struct.pack("<H", num_channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", bits_per_sample)

    # data subchunk
    header += b"data"
    header += struct.pack("<I", data_size)

    return header


if __name__ == "__main__":
    import uvicorn
    # IMPORTANT: module name is "server" (this file)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
