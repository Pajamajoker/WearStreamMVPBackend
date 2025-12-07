# audio_model.py
#
# Streaming wrapper around MIT/ast-finetuned-audioset-16-16-0.442
# Takes raw 16kHz mono 16-bit PCM chunks, keeps a per-client buffer,
# runs 1-second windows through AST, and maintains rolling scores.

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

# ---- Audio + window config (must match your pipeline) ----
TARGET_SR = 16000          # Hz
WINDOW_SEC = 1             # length of each classification window
BYTES_PER_SAMPLE = 2       # int16 = 2 bytes
BYTES_PER_SECOND = TARGET_SR * BYTES_PER_SAMPLE  # mono
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


@dataclass
class ClientState:
    """Keeps streaming state for a single client."""
    pcm_buffer: bytearray = field(default_factory=bytearray)
    history: Dict[str, List[float]] = field(
        default_factory=lambda: {g: [] for g in GROUPS.keys()}
    )
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

        # Process as many full 1-second windows as we have.
        updated = False
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
        Run one 1-second window through the model and update scores.
        """
        # PCM int16 LE -> float32 in [-1, 1]
        samples = np.frombuffer(window_bytes, dtype="<i2").astype("float32") / 32768.0
        waveform = torch.from_numpy(samples)  # shape [time]

        # AST processor expects mono waveform
        inputs = self.processor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        )

        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.sigmoid(logits)[0].cpu()  # multi-label probs

        # Compute group scores
        instant_scores = {}
        for group_name, labels in GROUPS.items():
            s = self._sum_labels(probs, labels)
            instant_scores[group_name] = s
            state.history[group_name].append(s)

        # Rolling averages
        rolling_scores = {}
        for group_name, hist in state.history.items():
            if not hist:
                rolling_scores[group_name] = 0.0
            else:
                recent = hist[-AGG_WINDOW_CHUNKS:]
                rolling_scores[group_name] = float(sum(recent) / len(recent))

        # Aggregate emergency scores
        emergency_instant = max(instant_scores.values()) if instant_scores else 0.0
        emergency_rolling = max(rolling_scores.values()) if rolling_scores else 0.0

        state.last_scores = {
            "instant": instant_scores,
            "rolling": rolling_scores,
            "emergency_instant": emergency_instant,
            "emergency_rolling": emergency_rolling,
            "timestamp": time.time(),
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
