# audio_model.py
#
# Streaming wrapper around MIT/ast-finetuned-audioset-16-16-0.442
# Takes raw 16kHz mono 16-bit PCM chunks, keeps a per-client buffer,
# uses a multi-second context window for the model, and maintains
# rolling scores + human-readable logs.

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

# ---- Audio + window config (must match your pipeline) ----
TARGET_SR = 16000          # Hz
WINDOW_SEC = 1             # how often we advance the stream (seconds)
BYTES_PER_SAMPLE = 2       # int16 = 2 bytes
BYTES_PER_SECOND = TARGET_SR * BYTES_PER_SAMPLE  # mono

# We still step the stream in 1s chunks, but the model
# will see the *last CONTEXT_SEC seconds* of audio.
CONTEXT_SEC = 3            # length of context passed to model
CONTEXT_BYTES = CONTEXT_SEC * BYTES_PER_SECOND

# Rolling aggregation over last N windows (seconds)
AGG_WINDOW_CHUNKS = 4      # rolling average over last N seconds

# ---- Model config ----
MODEL_ID = "MIT/ast-finetuned-audioset-16-16-0.442"

# ---- Label groups (AudioSet labels) ----
VOCAL_DISTRESS_LABELS = [
    "Screaming",
    "Yell",
    "Whoop",
    "Groan",
    "Grunt",
    "Bellow",
    "Crying, sobbing",
    "Wail, moan",
    "Whimper",
]

GUNSHOT_LABELS = [
    "Gunshot, gunfire",
    "Machine gun",
    "Cap gun",
]

ALARM_LABELS = [
    "Fire alarm",
    "Smoke alarm",
    "Siren",
    "Police car (siren)",
    "Ambulance (siren)",
    "Fire engine, fire truck (siren)",
    "Alarm",
]

EXPLOSION_LABELS = [
    "Explosion",
    "Bang",
    "Burst, pop",
]

GROUPS = {
    "vocal": VOCAL_DISTRESS_LABELS,
    "gunshot": GUNSHOT_LABELS,
    "alarm": ALARM_LABELS,
    "explosion": EXPLOSION_LABELS,
}

# ========= CALIBRATION / THRESHOLDS =========
# Your tuned gains (can be adjusted further)

CALIBRATION = {
    # Screams are undersensitive -> boost
    "vocal": {
        "gain": 2,
        "bias": 0.0,
    },
    # Gunshots are undersensitive -> boost more
    "gunshot": {
        "gain": 10.0,
        "bias": 0.0,
    },
    # Alarms oversensitive -> you had 0.5 before, let's keep closer to 1
    "alarm": {
        "gain": 1.0,
        "bias": 0.0,
    },
    # Explosions somewhere in between
    "explosion": {
        "gain": 4.5,
        "bias": 0.0,
    },
}

# Detection / UI thresholds (per-group "emergency" level)
THRESHOLDS = {
    "vocal": 0.40,
    "gunshot": 0.30,
    "alarm": 0.50,
    "explosion": 0.40,
    "emergency": 0.50,  # global overall emergency threshold (for future use)
}

# We derive "potential" = 0.5 * emergency threshold per group
SEVERITY_ORDER = {"none": 0, "potential": 1, "emergency": 2}


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def apply_calibration(x: float, kind: str) -> float:
    """
    Apply per-class gain + bias, clamp into [0, 1].
    x is usually a rolling score (0–1).
    """
    cfg = CALIBRATION.get(kind, {"gain": 1.0, "bias": 0.0})
    y = cfg["gain"] * x + cfg["bias"]
    return clamp01(y)


def classify_severity(score: float, group: str) -> str:
    """
    Map a final rolling score into: none / potential / emergency
    using group-specific thresholds from THRESHOLDS.
    """
    em_thr = THRESHOLDS.get(group, 0.5)
    pot_thr = 0.5 * em_thr

    if score >= em_thr:
        return "emergency"
    elif score >= pot_thr:
        return "potential"
    else:
        return "none"


def format_ts(ts: float) -> str:
    """Return an ISO-ish timestamp string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


@dataclass
class ClientState:
    """Keeps streaming state for a single client."""
    # Raw 1s step buffer (used only to decide when to process)
    pcm_buffer: bytearray = field(default_factory=bytearray)

    # Rolling context buffer of last CONTEXT_SEC seconds (for model input)
    pcm_context: bytearray = field(default_factory=bytearray)

    # History of raw per-second group scores (before calibration)
    history: Dict[str, List[float]] = field(
        default_factory=lambda: {g: [] for g in GROUPS.keys()}
    )

    # Last computed scores + log info
    last_scores: Optional[dict] = None


class StreamingAudioClassifier:
    """
    Streaming classifier around AST AudioSet model.

    Usage:
        classifier = StreamingAudioClassifier()
        classifier.ingest_pcm(client_id, pcm_bytes)
        scores = classifier.get_scores(client_id)
    """

    def __init__(self):
        print("[MODEL] Loading processor and model...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
        self.model.eval()

        # Choose device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[MODEL] Loaded on device: {self.device}")

        # Label mappings
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Reverse mapping: AudioSet label -> list of our groups
        self.label_to_groups: Dict[str, List[str]] = {}
        for group_name, labels in GROUPS.items():
            for lab in labels:
                self.label_to_groups.setdefault(lab, []).append(group_name)

        # Per-client state
        self.clients: Dict[str, ClientState] = {}

    # ---------------- Public API ----------------

    def ingest_pcm(self, client_id: str, pcm_bytes: bytes) -> Optional[dict]:
        """
        Feed raw 16-bit PCM (little-endian) into the stream for a client.
        Internally accumulates a buffer; every time it reaches 1 second,
        runs a classification step and updates rolling scores.

        Returns latest scores dict for that client (or None if not enough data yet).
        """
        state = self.clients.get(client_id)
        if state is None:
            state = ClientState()
            self.clients[client_id] = state

        state.pcm_buffer.extend(pcm_bytes)

        updated = False
        # Process as many full 1-second windows as we have.
        while len(state.pcm_buffer) >= BYTES_PER_SECOND:
            window_bytes = state.pcm_buffer[:BYTES_PER_SECOND]
            del state.pcm_buffer[:BYTES_PER_SECOND]

            self._process_window(state, window_bytes)
            updated = True

        return state.last_scores if updated else state.last_scores

    def get_scores(self, client_id: str) -> Optional[dict]:
        state = self.clients.get(client_id)
        if state is None:
            return None
        return state.last_scores

    def get_all_scores(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for cid, state in self.clients.items():
            if state.last_scores is not None:
                out[cid] = state.last_scores
        return out

    # ---------------- Internal helpers ----------------

    def _process_window(self, state: ClientState, window_bytes: bytes) -> None:
        """
        Advance stream by 1 second:
        - update context buffer (last CONTEXT_SEC seconds)
        - run model on that multi-second context
        - update rolling + severity + log.
        """
        # 1) Update context (last CONTEXT_SEC seconds)
        state.pcm_context.extend(window_bytes)
        if len(state.pcm_context) > CONTEXT_BYTES:
            # keep only the last CONTEXT_BYTES
            state.pcm_context = state.pcm_context[-CONTEXT_BYTES:]

        # Use full context for model; if we are at the very beginning,
        # we may have less than CONTEXT_SEC seconds -> that's fine.
        context_bytes = state.pcm_context if state.pcm_context else window_bytes
        context_sec = len(context_bytes) / BYTES_PER_SECOND

        # 2) PCM int16 LE -> float32 in [-1, 1]
        samples = np.frombuffer(context_bytes, dtype="<i2").astype("float32") / 32768.0
        waveform = torch.from_numpy(samples)  # shape [time]

        # 3) AST processor expects mono waveform
        inputs = self.processor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )

        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 4) Forward pass
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.sigmoid(logits)[0].cpu()  # multi-label probs, shape [num_labels]

        # 5) Top-k model labels (true AudioSet labels)
        TOP_K = 5
        top_indices = probs.topk(TOP_K).indices.tolist()
        top_labels = []
        for idx in top_indices:
            label = self.id2label[idx]
            p = float(probs[idx])
            mapped_groups = self.label_to_groups.get(label, [])
            top_labels.append({
                "label": label,
                "prob": p,
                "groups": mapped_groups,
            })

        # 6) Per-group instant scores (sum of relevant AudioSet probs)
        instant_scores: Dict[str, float] = {}
        for group_name, labels in GROUPS.items():
            s = self._sum_labels(probs, labels)
            instant_scores[group_name] = s
            state.history[group_name].append(s)

        # 7) Rolling averages (raw, over last AGG_WINDOW_CHUNKS seconds)
        rolling_raw: Dict[str, float] = {}
        for group_name, hist in state.history.items():
            if not hist:
                rolling_raw[group_name] = 0.0
            else:
                recent = hist[-AGG_WINDOW_CHUNKS:]
                rolling_raw[group_name] = float(sum(recent) / len(recent))

        # 8) Apply calibration on rolling scores
        rolling_calibrated: Dict[str, float] = {}
        for group_name, raw_val in rolling_raw.items():
            rolling_calibrated[group_name] = apply_calibration(raw_val, group_name)

        # For now, our FINAL scores are the calibrated ones (no extra trend/corr)
        rolling_final: Dict[str, float] = dict(rolling_calibrated)

        # 9) Per-group severity classification
        per_group_level: Dict[str, str] = {}
        overall_level = "none"
        for group_name, score in rolling_final.items():
            level = classify_severity(score, group_name)
            per_group_level[group_name] = level
            if SEVERITY_ORDER[level] > SEVERITY_ORDER[overall_level]:
                overall_level = level

        # 10) Emergency aggregates
        emergency_instant = max(instant_scores.values()) if instant_scores else 0.0
        emergency_rolling_raw = max(rolling_raw.values()) if rolling_raw else 0.0
        emergency_rolling_final = max(
            rolling_final.values()
        ) if rolling_final else 0.0

        ts = time.time()

        # 11) Build a nice log line for dashboard / debugging
        def fmt_score(name: str) -> str:
            s = rolling_final.get(name, 0.0)
            lvl = per_group_level.get(name, "none")
            return f"{name}={s:.3f}({lvl})"

        parts = [
            f"[{format_ts(ts)}]",
            f"ctx={context_sec:.1f}s",
            fmt_score("vocal"),
            fmt_score("gunshot"),
            fmt_score("alarm"),
            fmt_score("explosion"),
            f"overall={overall_level}",
        ]

        top_str = ", ".join(
            f"{t['label']}:{t['prob']:.3f}"
            for t in top_labels
        )
        parts.append(f"top=[{top_str}]")

        log_line = " | ".join(parts)

        # 12) Store everything in last_scores (what /metrics and WS server can expose)
        state.last_scores = {
            # raw per-second group probs from model
            "instant": instant_scores,

            # rolling averages BEFORE calibration (for debugging)
            "rolling_raw": rolling_raw,

            # rolling AFTER calibration (used by plots + severity)
            "rolling": rolling_final,

            # top-k AudioSet labels
            "top_labels": top_labels,

            # per-group and overall severity levels
            "per_group_level": per_group_level,
            "overall_level": overall_level,

            # aggregate emergency scores
            "emergency_instant": emergency_instant,
            "emergency_rolling_raw": emergency_rolling_raw,
            "emergency_rolling": emergency_rolling_final,

            # thresholds (so UI can draw horizontal lines etc.)
            "thresholds": THRESHOLDS,

            # how much context the model actually saw here
            "context_seconds": context_sec,

            # timestamp + human-readable log
            "timestamp": ts,
            "log_line": log_line,
        }

    def _sum_labels(self, probs: torch.Tensor, labels: List[str]) -> float:
        """
        Sum probabilities of all AudioSet labels in this group.
        Skips labels that are not present in this model's config.
        """
        score = 0.0
        for lab in labels:
            idx = self.label2id.get(lab)
            if idx is not None:
                score += float(probs[idx])
        return score


# Global singleton classifier used by ws_server.py
classifier = StreamingAudioClassifier()
