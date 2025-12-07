# ws_server.py
import os
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

app = FastAPI()

SAMPLE_RATE = 16000        # Hz
CHANNELS = 1               # mono
BITS_PER_SAMPLE = 16       # 16-bit PCM
BYTES_PER_SAMPLE = BITS_PER_SAMPLE // 8
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE

TARGET_SECONDS = 10
TARGET_BYTES = BYTES_PER_SECOND * TARGET_SECONDS

RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

buffers = {}  # key: client id, value: bytearray


@app.get("/")
async def root():
    return {"status": "ok", "message": "WS audio server running (ws_server.py)"}


@app.get("/debug")
async def debug():
    return JSONResponse({
        "active_buffers": {k: len(v) for k, v in buffers.items()},
        "target_bytes": TARGET_BYTES,
    })


@app.websocket("/ws")
async def audio_ws(websocket: WebSocket):
    """
    SUPER SIMPLE: single WebSocket endpoint at /ws.
    """
    await websocket.accept()
    client_key = f"client_{id(websocket)}"
    buffers[client_key] = bytearray()

    print(f"[WS OPEN] {client_key}")

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk_size = len(data)
            print(f"[RECV] {client_key} chunk={chunk_size} bytes")

            buf = buffers[client_key]
            buf.extend(data)
            print(f"[BUFFER] {client_key} size={len(buf)} bytes (target={TARGET_BYTES})")

            if len(buf) >= TARGET_BYTES:
                print(f"[SAVE] {client_key} reached ~10s of audio")
                save_buffer_as_wav(client_key, buf)
                buf.clear()
                await websocket.send_text("saved_10s_clip")
    except WebSocketDisconnect:
        print(f"[WS CLOSE] {client_key}")
    except Exception as e:
        print(f"[WS ERROR] {client_key}: {e}")
    finally:
        buffers.pop(client_key, None)


def save_buffer_as_wav(client_key: str, buffer: bytearray):
    pcm_bytes = bytes(buffer)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{client_key}_{timestamp}.wav"
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
    import struct

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * num_channels * bits_per_sample // 8
    riff_chunk_size = 36 + data_size

    header = b"RIFF"
    header += struct.pack("<I", riff_chunk_size)
    header += b"WAVE"

    header += b"fmt "
    header += struct.pack("<I", 16)
    header += struct.pack("<H", 1)
    header += struct.pack("<H", num_channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", bits_per_sample)

    header += b"data"
    header += struct.pack("<I", data_size)

    return header
