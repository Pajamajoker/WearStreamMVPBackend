# ws_server.py
#
# WebSocket server:
# - Receives raw 16kHz mono 16-bit PCM from phone
# - Buffers per client and saves ~10s WAV files to audio_recordings/
# - Feeds the same PCM stream into StreamingAudioClassifier (audio_model.py)
# - Exposes /scores endpoints for live confidence monitoring

import os
from datetime import datetime
from fastapi.responses import HTMLResponse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from audio_model import classifier  # <-- our streaming classifier

app = FastAPI()

# ---- Audio config (must match audio_model.py) ----
SAMPLE_RATE = 16000      # Hz
CHANNELS = 1             # mono
BITS_PER_SAMPLE = 16     # 16-bit PCM
BYTES_PER_SAMPLE = BITS_PER_SAMPLE // 8
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE

TARGET_SECONDS = 10
TARGET_BYTES = BYTES_PER_SECOND * TARGET_SECONDS

# ---- Folder for recordings ----
RECORDINGS_DIR = "audio_recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# In-memory buffers per client for WAV saving
buffers = {}  # key: client id, value: bytearray


@app.get("/")
async def root():
    return {"status": "ok", "message": "WS audio server with streaming classifier"}


@app.get("/debug")
async def debug():
    return JSONResponse({
        "active_buffers": {k: len(v) for k, v in buffers.items()},
        "target_bytes": TARGET_BYTES,
        "classifier_clients": list(classifier.clients.keys()),
    })


@app.get("/scores")
async def get_all_scores():
    """
    Get latest scores for all clients.
    """
    return classifier.get_all_scores()


@app.get("/scores/{client_id}")
async def get_scores(client_id: str):
    """
    Get latest scores for a specific client.
    """
    scores = classifier.get_scores(client_id)
    if scores is None:
        raise HTTPException(status_code=404, detail="No scores for this client yet")
    return scores


@app.websocket("/ws")
async def audio_ws(websocket: WebSocket):
    """
    Simple WebSocket endpoint at /ws.
    Each connection is treated as a separate client stream.
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

            # 1) Feed into streaming classifier
            scores = classifier.ingest_pcm(client_key, data)
            if scores is not None:
                # Log the rolling emergency score; you could also send this to the client.
                er = scores.get("emergency_rolling", 0.0)
                print(f"[CLS] {client_key} emergency_rolling={er:.3f} "
                      f"rolling={scores.get('rolling')}")

            # 2) Buffer for WAV saving (10 second chunks)
            buf = buffers[client_key]
            buf.extend(data)
            print(f"[BUFFER] {client_key} size={len(buf)} bytes (target={TARGET_BYTES})")

            if len(buf) >= TARGET_BYTES:
                print(f"[SAVE] {client_key} reached ~10s of audio, saving WAV")
                #save_buffer_as_wav(client_key, buf)
                buf.clear()

                # Optional: notify client that a clip was saved
                await websocket.send_text("saved_10s_clip")

    except WebSocketDisconnect:
        print(f"[WS CLOSE] {client_key}")
    except Exception as e:
        print(f"[WS ERROR] {client_key}: {e}")
    finally:
        buffers.pop(client_key, None)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    # Simple HTML page with Chart.js and polling of /scores
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Audio Emergency Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 0;
            padding: 16px;
            background: #111827;
            color: #e5e7eb;
        }
        h1 {
            margin-top: 0;
            font-size: 20px;
        }
        #top-bar {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }
        #client-id {
            font-weight: 600;
            color: #a5b4fc;
        }
        #status {
            font-size: 14px;
        }
        #scores-panel {
            margin-top: 12px;
            font-size: 14px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .score-box {
            background: #020617;
            border-radius: 8px;
            padding: 8px 10px;
            min-width: 120px;
        }
        .score-label {
            font-size: 12px;
            text-transform: uppercase;
            color: #9ca3af;
        }
        .score-value {
            font-size: 16px;
            font-weight: 600;
        }
        canvas {
            background: #020617;
            border-radius: 12px;
            padding: 12px;
        }
    </style>
</head>
<body>
    <h1>Audio Emergency Monitor</h1>
    <div id="top-bar">
        <div>Active client: <span id="client-id">none</span></div>
        <div id="status">Waiting for scores...</div>
    </div>
    <canvas id="chart" height="120"></canvas>

    <div id="scores-panel">
        <div class="score-box">
            <div class="score-label">Vocal distress</div>
            <div class="score-value" id="vocal-score">0.00</div>
        </div>
        <div class="score-box">
            <div class="score-label">Gunshot</div>
            <div class="score-value" id="gunshot-score">0.00</div>
        </div>
        <div class="score-box">
            <div class="score-label">Alarm</div>
            <div class="score-value" id="alarm-score">0.00</div>
        </div>
        <div class="score-box">
            <div class="score-label">Explosion</div>
            <div class="score-value" id="explosion-score">0.00</div>
        </div>
        <div class="score-box">
            <div class="score-label">Emergency (rolling max)</div>
            <div class="score-value" id="emergency-score">0.00</div>
        </div>
    </div>

    <script>
        let chart;
        let labels = [];
        const MAX_POINTS = 60;  // ~60 seconds visible

        const datasets = {
            vocal: {
                label: 'Vocal distress',
                borderColor: 'rgb(239, 68, 68)',
                borderWidth: 2,
                tension: 0.3,
                data: [],
            },
            gunshot: {
                label: 'Gunshot',
                borderColor: 'rgb(249, 115, 22)',
                borderWidth: 2,
                tension: 0.3,
                data: [],
            },
            alarm: {
                label: 'Alarm',
                borderColor: 'rgb(59, 130, 246)',
                borderWidth: 2,
                tension: 0.3,
                data: [],
            },
            explosion: {
                label: 'Explosion',
                borderColor: 'rgb(132, 204, 22)',
                borderWidth: 2,
                tension: 0.3,
                data: [],
            },
            emergency: {
                label: 'Emergency (rolling max)',
                borderColor: 'rgb(244, 244, 245)',
                borderWidth: 2,
                borderDash: [4, 4],
                tension: 0.3,
                data: [],
            }
        };

        function initChart() {
            const ctx = document.getElementById('chart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        datasets.vocal,
                        datasets.gunshot,
                        datasets.alarm,
                        datasets.explosion,
                        datasets.emergency,
                    ],
                },
                options: {
                    responsive: true,
                    animation: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#e5e7eb',
                                font: { size: 11 }
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: '#9ca3af',
                                maxTicksLimit: 8
                            },
                            title: {
                                display: true,
                                text: 'Time (seconds)',
                                color: '#9ca3af',
                                font: { size: 11 }
                            },
                            grid: {
                                color: '#1f2937'
                            }
                        },
                        y: {
                            suggestedMin: 0,
                            suggestedMax: 1,
                            ticks: {
                                color: '#9ca3af',
                                stepSize: 0.2
                            },
                            title: {
                                display: true,
                                text: 'Confidence',
                                color: '#9ca3af',
                                font: { size: 11 }
                            },
                            grid: {
                                color: '#1f2937'
                            }
                        }
                    }
                }
            });
        }

        async function pollScores() {
            try {
                const res = await fetch('/scores');
                if (!res.ok) {
                    document.getElementById('status').textContent = 'Error fetching scores';
                    return;
                }
                const data = await res.json();
                const clientIds = Object.keys(data);

                if (clientIds.length === 0) {
                    document.getElementById('client-id').textContent = 'none';
                    document.getElementById('status').textContent = 'No active streams yet...';
                    return;
                }

                // For now: just take the first client
                const clientId = clientIds[0];
                document.getElementById('client-id').textContent = clientId;
                document.getElementById('status').textContent = 'Receiving live scores';

                const scores = data[clientId];
                const rolling = scores.rolling || {};
                const emergencyRolling = scores.emergency_rolling || 0.0;

                // Update numeric display
                const v = rolling.vocal || 0.0;
                const g = rolling.gunshot || 0.0;
                const a = rolling.alarm || 0.0;
                const e = rolling.explosion || 0.0;

                document.getElementById('vocal-score').textContent = v.toFixed(3);
                document.getElementById('gunshot-score').textContent = g.toFixed(3);
                document.getElementById('alarm-score').textContent = a.toFixed(3);
                document.getElementById('explosion-score').textContent = e.toFixed(3);
                document.getElementById('emergency-score').textContent = emergencyRolling.toFixed(3);

                // Append new point
                const t = labels.length > 0 ? labels[labels.length - 1] + 1 : 0;
                labels.push(t);
                datasets.vocal.data.push(v);
                datasets.gunshot.data.push(g);
                datasets.alarm.data.push(a);
                datasets.explosion.data.push(e);
                datasets.emergency.data.push(emergencyRolling);

                // Keep only last MAX_POINTS
                if (labels.length > MAX_POINTS) {
                    labels.shift();
                    Object.values(datasets).forEach(ds => ds.data.shift());
                }

                chart.data.labels = labels;
                chart.update('none');

            } catch (err) {
                console.error('Error in pollScores:', err);
                document.getElementById('status').textContent = 'Error talking to server';
            }
        }

        window.addEventListener('load', () => {
            initChart();
            // Poll once per second; you can lower to 500ms, but model updates are 1s anyway
            setInterval(pollScores, 1000);
        });
    </script>
</body>
</html>
    """

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
    """
    Create a standard 44-byte WAV header for PCM data.
    """
    import struct

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * num_channels * bits_per_sample // 8
    riff_chunk_size = 36 + data_size

    header = b"RIFF"
    header += struct.pack("<I", riff_chunk_size)
    header += b"WAVE"

    header += b"fmt "
    header += struct.pack("<I", 16)                 # fmt chunk size
    header += struct.pack("<H", 1)                  # PCM
    header += struct.pack("<H", num_channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", bits_per_sample)

    header += b"data"
    header += struct.pack("<I", data_size)

    return header
