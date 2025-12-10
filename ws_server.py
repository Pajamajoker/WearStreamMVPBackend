# ws_server.py
#
# WebSocket server:
# - Receives raw 16kHz mono 16-bit PCM from phone
# - Buffers per client and (optionally) saves ~10s WAV files to audio_recordings/
# - Feeds the same PCM stream into StreamingAudioClassifier (audio_model.py)
# - Exposes /scores endpoints for live confidence monitoring
# - /dashboard shows live chart + alert status + alert history + log stream

import os
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

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
                er = scores.get("emergency_rolling", 0.0)
                overall = scores.get("overall_level", "none")
                print(f"[CLS] {client_key} overall={overall} "
                      f"emergency_rolling={er:.3f} "
                      f"rolling={scores.get('rolling')}")

            # 2) Buffer for WAV saving (10 second chunks) — optional
            buf = buffers[client_key]
            buf.extend(data)
            print(f"[BUFFER] {client_key} size={len(buf)} bytes (target={TARGET_BYTES})")

            if len(buf) >= TARGET_BYTES:
                print(f"[SAVE] {client_key} reached ~10s of audio, clearing buffer")
                # If you want to save, uncomment:
                # save_buffer_as_wav(client_key, buf)
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
    # HTML page with Chart.js, polling /scores and showing:
    # - Rolling confidence lines
    # - Current alert status
    # - Alert history
    # - Log stream (log_line from audio_model)
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Audio Emergency Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg: #020617;
            --bg-elevated: #020617;
            --bg-page: #020617;
            --bg-alt: #0f172a;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --border-subtle: #1f2937;
            --danger: #ef4444;
            --warning: #f97316;
            --ok: #22c55e;
        }
        * { box-sizing: border-box; }
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 0;
            padding: 16px;
            background: var(--bg-page);
            color: var(--text);
        }
        h1 {
            margin-top: 0;
            font-size: 20px;
        }
        #layout {
            display: grid;
            grid-template-columns: 2fr 1fr;
            grid-template-rows: auto 1fr;
            grid-template-areas:
              "main alerts"
              "main logs";
            gap: 16px;
        }
        #main-panel { grid-area: main; }
        #alerts-panel { grid-area: alerts; }
        #logs-panel { grid-area: logs; }

        #top-bar {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }
        #client-id {
            font-weight: 600;
            color: #a5b4fc;
        }
        #status {
            font-size: 13px;
            color: var(--muted);
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .badge-none {
            background: rgba(148, 163, 184, 0.15);
            color: var(--muted);
        }
        .badge-potential {
            background: rgba(249, 115, 22, 0.12);
            color: var(--warning);
        }
        .badge-emergency {
            background: rgba(239, 68, 68, 0.12);
            color: var(--danger);
            box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.4);
        }

        #alert-banner {
            margin-top: 8px;
            padding: 8px 10px;
            border-radius: 8px;
            border: 1px solid rgba(239, 68, 68, 0.5);
            background: rgba(127, 29, 29, 0.8);
            color: #fee2e2;
            font-size: 13px;
            display: none;
        }
        #alert-banner strong {
            margin-right: 6px;
        }

        #scores-panel {
            margin-top: 12px;
            font-size: 14px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .score-box {
            background: var(--bg-elevated);
            border-radius: 8px;
            padding: 8px 10px;
            min-width: 140px;
            border: 1px solid var(--border-subtle);
        }
        .score-label {
            font-size: 11px;
            text-transform: uppercase;
            color: var(--muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .score-value {
            font-size: 16px;
            font-weight: 600;
            margin-top: 2px;
        }
        .score-level {
            font-size: 11px;
            padding: 1px 6px;
            border-radius: 999px;
            border: 1px solid transparent;
        }
        .level-none {
            color: var(--muted);
            border-color: transparent;
        }
        .level-potential {
            color: var(--warning);
            border-color: rgba(249, 115, 22, 0.4);
        }
        .level-emergency {
            color: var(--danger);
            border-color: rgba(239, 68, 68, 0.6);
        }

        canvas {
            background: var(--bg-elevated);
            border-radius: 12px;
            padding: 12px;
            border: 1px solid var(--border-subtle);
        }

        /* Alerts column */
        .panel {
            background: var(--bg-alt);
            border-radius: 12px;
            border: 1px solid var(--border-subtle);
            padding: 10px 12px;
            font-size: 13px;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .panel-title {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
        }
        .panel-body {
            overflow-y: auto;
            max-height: 260px;
            padding-right: 4px;
        }
        .history-item {
            padding: 6px 4px;
            border-bottom: 1px solid rgba(31, 41, 55, 0.7);
        }
        .history-item:last-child {
            border-bottom: none;
        }
        .history-meta {
            font-size: 11px;
            color: var(--muted);
            margin-bottom: 2px;
        }
        .history-main {
            font-size: 13px;
        }

        /* Log panel */
        #logs-list {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 11px;
            color: #e5e7eb;
        }
        .log-line {
            padding: 3px 0;
            border-bottom: 1px solid rgba(31, 41, 55, 0.7);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        @media (max-width: 900px) {
            #layout {
                grid-template-columns: 1fr;
                grid-template-areas:
                  "main"
                  "alerts"
                  "logs";
            }
        }
    </style>
</head>
<body>
    <h1>Audio Emergency Monitor</h1>
    <div id="layout">
        <div id="main-panel">
            <div id="top-bar">
                <div>Active client: <span id="client-id">none</span></div>
                <div id="status">Waiting for scores...</div>
                <div id="overall-badge" class="badge badge-none">No signal</div>
            </div>
            <div id="alert-banner">
                <strong>🚨 Emergency detected</strong>
                <span id="alert-banner-text"></span>
            </div>

            <canvas id="chart" height="120"></canvas>

            <div id="scores-panel">
                <div class="score-box">
                    <div class="score-label">
                        <span>Vocal distress</span>
                        <span id="vocal-level" class="score-level level-none">none</span>
                    </div>
                    <div class="score-value" id="vocal-score">0.00</div>
                </div>
                <div class="score-box">
                    <div class="score-label">
                        <span>Gunshot</span>
                        <span id="gunshot-level" class="score-level level-none">none</span>
                    </div>
                    <div class="score-value" id="gunshot-score">0.00</div>
                </div>
                <div class="score-box">
                    <div class="score-label">
                        <span>Alarm</span>
                        <span id="alarm-level" class="score-level level-none">none</span>
                    </div>
                    <div class="score-value" id="alarm-score">0.00</div>
                </div>
                <div class="score-box">
                    <div class="score-label">
                        <span>Explosion</span>
                        <span id="explosion-level" class="score-level level-none">none</span>
                    </div>
                    <div class="score-value" id="explosion-score">0.00</div>
                </div>
                <div class="score-box">
                    <div class="score-label">
                        <span>Emergency (rolling max)</span>
                        <span class="score-level level-none" id="emergency-level-small">none</span>
                    </div>
                    <div class="score-value" id="emergency-score">0.00</div>
                </div>
            </div>
        </div>

        <div id="alerts-panel" class="panel">
            <div class="panel-header">
                <div class="panel-title">Alert history</div>
                <div id="alerts-count" style="font-size:11px;color:var(--muted);">0</div>
            </div>
            <div class="panel-body" id="alerts-list">
                <!-- alerts go here -->
            </div>
        </div>

        <div id="logs-panel" class="panel">
            <div class="panel-header">
                <div class="panel-title">Model log</div>
                <div style="font-size:11px;color:var(--muted);">Newest first</div>
            </div>
            <div class="panel-body" id="logs-list">
                <!-- log lines go here -->
            </div>
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

        // Alert + log state
        let lastOverallLevel = "none";
        let lastLogTimestamp = 0;
        const MAX_ALERTS = 50;
        const MAX_LOGS = 200;

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
                                text: 'Time (windows)',
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

        function setOverallBadge(level) {
            const badge = document.getElementById('overall-badge');
            badge.className = 'badge'; // reset
            let text = '';

            if (level === 'emergency') {
                badge.classList.add('badge-emergency');
                text = 'Emergency';
            } else if (level === 'potential') {
                badge.classList.add('badge-potential');
                text = 'Potential risk';
            } else {
                badge.classList.add('badge-none');
                text = 'No emergency detected';
            }

            badge.textContent = text;
        }

        function setLevelChip(elemId, level) {
            const el = document.getElementById(elemId);
            el.className = 'score-level';
            if (level === 'emergency') {
                el.classList.add('level-emergency');
            } else if (level === 'potential') {
                el.classList.add('level-potential');
            } else {
                el.classList.add('level-none');
            }
            el.textContent = level;
        }

        function showAlertBanner(text) {
            const banner = document.getElementById('alert-banner');
            const span = document.getElementById('alert-banner-text');
            span.textContent = text || '';
            banner.style.display = 'block';

            // Auto-fade after 6 seconds
            setTimeout(() => {
                banner.style.display = 'none';
            }, 6000);
        }

        function addAlertHistoryEntry(clientId, scores) {
            const scoresRolling = scores.rolling || {};
            const levels = scores.per_group_level || {};
            const ts = scores.timestamp || Date.now() / 1000;
            const date = new Date(ts * 1000);
            const tsStr = date.toLocaleTimeString([], { hour12: false });

            const v = (scoresRolling.vocal || 0).toFixed(3);
            const g = (scoresRolling.gunshot || 0).toFixed(3);
            const a = (scoresRolling.alarm || 0).toFixed(3);
            const e = (scoresRolling.explosion || 0).toFixed(3);

            const overall = scores.overall_level || 'emergency';

            const item = document.createElement('div');
            item.className = 'history-item';

            item.innerHTML = `
                <div class="history-meta">
                    <span>${tsStr}</span>
                    <span style="margin-left:6px;color:#a5b4fc;">${clientId}</span>
                    <span style="margin-left:6px;">level: <strong>${overall}</strong></span>
                </div>
                <div class="history-main">
                    vocal=${v} (${levels.vocal || 'none'}),
                    gunshot=${g} (${levels.gunshot || 'none'}),
                    alarm=${a} (${levels.alarm || 'none'}),
                    explosion=${e} (${levels.explosion || 'none'})
                </div>
            `;

            const list = document.getElementById('alerts-list');
            // Newest on top
            list.insertBefore(item, list.firstChild);

            // Trim
            while (list.children.length > MAX_ALERTS) {
                list.removeChild(list.lastChild);
            }

            document.getElementById('alerts-count').textContent =
                list.children.length.toString();
        }

        function addLogLine(scores) {
            const logLine = scores.log_line;
            const ts = scores.timestamp || Date.now() / 1000;
            const list = document.getElementById('logs-list');

            if (!logLine) return;

            // Avoid duplicates by timestamp
            if (ts <= lastLogTimestamp) return;
            lastLogTimestamp = ts;

            const div = document.createElement('div');
            div.className = 'log-line';
            div.textContent = logLine;

            // Newest first
            list.insertBefore(div, list.firstChild);

            while (list.children.length > MAX_LOGS) {
                list.removeChild(list.lastChild);
            }
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
                    setOverallBadge('none');
                    return;
                }

                // For now: just take the first client
                const clientId = clientIds[0];
                const scores = data[clientId];

                document.getElementById('client-id').textContent = clientId;
                document.getElementById('status').textContent = 'Receiving live scores';

                const rolling = scores.rolling || {};
                const emergencyRolling = scores.emergency_rolling || 0.0;
                const perGroupLevel = scores.per_group_level || {};
                const overallLevel = scores.overall_level || 'none';

                // Update overall badge
                setOverallBadge(overallLevel);

                // Update numeric display + per-group severity chips
                const v = rolling.vocal || 0.0;
                const g = rolling.gunshot || 0.0;
                const a = rolling.alarm || 0.0;
                const e = rolling.explosion || 0.0;

                document.getElementById('vocal-score').textContent = v.toFixed(3);
                document.getElementById('gunshot-score').textContent = g.toFixed(3);
                document.getElementById('alarm-score').textContent = a.toFixed(3);
                document.getElementById('explosion-score').textContent = e.toFixed(3);
                document.getElementById('emergency-score').textContent = emergencyRolling.toFixed(3);

                setLevelChip('vocal-level', perGroupLevel.vocal || 'none');
                setLevelChip('gunshot-level', perGroupLevel.gunshot || 'none');
                setLevelChip('alarm-level', perGroupLevel.alarm || 'none');
                setLevelChip('explosion-level', perGroupLevel.explosion || 'none');
                setLevelChip('emergency-level-small', overallLevel);

                // Append new point to chart
                const t = labels.length > 0 ? labels[labels.length - 1] + 1 : 0;
                labels.push(t);
                datasets.vocal.data.push(v);
                datasets.gunshot.data.push(g);
                datasets.alarm.data.push(a);
                datasets.explosion.data.push(e);
                datasets.emergency.data.push(emergencyRolling);

                if (labels.length > MAX_POINTS) {
                    labels.shift();
                    Object.values(datasets).forEach(ds => ds.data.shift());
                }

                chart.data.labels = labels;
                chart.update('none');

                // Log line
                addLogLine(scores);

                // ALERT logic: trigger when we *enter* emergency
                if (lastOverallLevel !== 'emergency' && overallLevel === 'emergency') {
                    const top = scores.top_labels || [];
                    const topText = top
                        .slice(0, 3)
                        .map(t => `${t.label}:${t.prob.toFixed(2)}`)
                        .join(', ');

                    showAlertBanner(topText || 'High emergency confidence');

                    addAlertHistoryEntry(clientId, scores);
                }
                lastOverallLevel = overallLevel;

            } catch (err) {
                console.error('Error in pollScores:', err);
                document.getElementById('status').textContent = 'Error talking to server';
                setOverallBadge('none');
            }
        }

        window.addEventListener('load', () => {
            initChart();
            // Poll once per second; model updates once per audio window
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
