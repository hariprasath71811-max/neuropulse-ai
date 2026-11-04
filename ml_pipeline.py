"""NeuroPulse AI - ML pipeline for synthetic EEG signal classification and anomaly detection."""

import time
import numpy as np
from scipy.signal import butter, filtfilt, welch
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

CHANNELS = ["Fp1", "Fp2", "C3", "C4", "P3", "P4"]
FS = 256
BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}
STATES = ["Calm / Relaxed", "Focused / Active", "Fatigued", "Abnormal Spike"]


class NeuroPulsePipeline:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.fs = FS
        self.channels = list(CHANNELS)
        self.scaler = StandardScaler()
        self.iso = IsolationForest(contamination=0.12, random_state=42, n_estimators=200)
        self.rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        self.trained = False
        self.anomaly_log = []
        self.session_log = []
        self._train()

    # ------------------------------------------------------------------ #
    # Signal processing
    # ------------------------------------------------------------------ #
    def _bandpass(self, data, lo: float = 0.5, hi: float = 45.0, order: int = 4):
        nyq = 0.5 * self.fs
        b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
        return filtfilt(b, a, data, axis=-1)

    def generate_signal(self, state: str, duration: float = 2.0, noise: float = 0.5,
                        inject_anomaly: bool = False, t_offset: float = 0.0):
        n = int(duration * self.fs)
        t = np.arange(n) / self.fs + t_offset
        sig = np.zeros((len(self.channels), n), dtype=float)
        for ci in range(len(self.channels)):
            if state == "Calm / Relaxed":
                base = 0.6 * np.sin(2 * np.pi * 10.0 * t) + 0.3 * np.sin(2 * np.pi * 9.0 * t)
            elif state == "Focused / Active":
                base = 0.5 * np.sin(2 * np.pi * 20.0 * t) + 0.4 * np.sin(2 * np.pi * 16.0 * t)
            elif state == "Fatigued":
                base = 0.7 * np.sin(2 * np.pi * 3.0 * t) + 0.4 * np.sin(2 * np.pi * 1.5 * t)
            else:  # Abnormal Spike
                base = 0.4 * np.sin(2 * np.pi * 10.0 * t) + 1.2 * np.sin(2 * np.pi * 40.0 * t)
            sig[ci] = base + self.rng.normal(0.0, noise, n)
            sig[ci] *= (1.0 + 0.08 * ci)
        if inject_anomaly:
            k = int(self.rng.integers(3, 7))
            idx = self.rng.integers(0, n - 3, size=k)
            for i in idx:
                sig[:, i:i + 3] += self.rng.normal(0.0, 4.5, (len(self.channels), 3))
        sig = self._bandpass(sig)
        return sig, t

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    def extract_features(self, sig):
        feats = []
        agg_band = {b: 0.0 for b in BANDS}
        per_channel = []
        for ci in range(sig.shape[0]):
            f, psd = welch(sig[ci], fs=self.fs, nperseg=min(256, sig.shape[1]))
            bp = {}
            for name, (lo, hi) in BANDS.items():
                mask = (f >= lo) & (f <= hi)
                bp[name] = float(np.trapezoid(psd[mask], f[mask])) if mask.any() else 0.0
            total = sum(bp.values()) + 1e-12
            for name in BANDS:
                feats.append(bp[name] / total)
                agg_band[name] += bp[name]
            per_channel.append(bp)
            feats.append(float(np.var(sig[ci])))
            feats.append(float(np.ptp(sig[ci])))
            feats.append(float(np.sqrt(np.mean(sig[ci] ** 2))))
        return np.array(feats, dtype=float), agg_band, per_channel

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _train(self):
        X = []
        y = []
        label_map = {s: i for i, s in enumerate(STATES)}
        for state in STATES:
            reps = 60 if state != "Abnormal Spike" else 40
            for _ in range(reps):
                inject = (state == "Abnormal Spike")
                sig, _ = self.generate_signal(state, duration=2.0, noise=0.45,
                                             inject_anomaly=inject)
                feats, _, _ = self.extract_features(sig)
                X.append(feats)
                y.append(label_map[state])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.iso.fit(Xs)
        self.rf.fit(Xs, y)
        self.trained = True
        self.n_features = X.shape[1]

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, sig, noise_level: float = 0.5):
        t0 = time.perf_counter()
        feats, agg_band, per_channel = self.extract_features(sig)
        Xs = self.scaler.transform(feats.reshape(1, -1))
        rf_pred = int(self.rf.predict(Xs)[0])
        rf_proba = self.rf.predict_proba(Xs)[0]
        iso_pred = int(self.iso.predict(Xs)[0])
        iso_score = float(self.iso.decision_function(Xs)[0])

        state = STATES[rf_pred]
        abn_idx = STATES.index("Abnormal Spike")

        if iso_pred == -1:
            risk = "Critical"
            risk_pct = float(np.clip((0.5 - iso_score) * 120.0, 70.0, 99.0))
        elif state == "Abnormal Spike":
            risk = "Critical"
            risk_pct = float(np.clip(75.0 + rf_proba[abn_idx] * 20.0, 75.0, 99.0))
        else:
            abn_p = float(rf_proba[abn_idx])
            if abn_p > 0.25:
                risk = "Medium"
                risk_pct = float(np.clip(30.0 + abn_p * 60.0, 30.0, 70.0))
            else:
                risk = "Low"
                risk_pct = float(np.clip(5.0 + abn_p * 30.0, 0.0, 30.0))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        dominant = max(agg_band, key=lambda k: agg_band[k])

        anomaly_event = None
        if iso_pred == -1 or state == "Abnormal Spike":
            anomaly_event = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "state": state,
                "risk": risk,
                "risk_pct": round(risk_pct, 1),
                "dominant_band": dominant,
            }
            self.anomaly_log.append(anomaly_event)
            if len(self.anomaly_log) > 200:
                self.anomaly_log = self.anomaly_log[-200:]

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state": state,
            "risk": risk,
            "risk_pct": round(risk_pct, 1),
            "dominant_band": dominant,
            "latency_ms": round(latency_ms, 2),
        }
        for b in BANDS:
            row[f"{b}_power"] = round(float(agg_band[b]), 4)
        self.session_log.append(row)
        if len(self.session_log) > 5000:
            self.session_log = self.session_log[-5000:]

        return {
            "state": state,
            "risk": risk,
            "risk_pct": risk_pct,
            "dominant_band": dominant,
            "latency_ms": latency_ms,
            "band_power": agg_band,
            "channel_band_power": per_channel,
            "rf_proba": {STATES[i]: float(rf_proba[i]) for i in range(len(STATES))},
            "anomaly_event": anomaly_event,
            "iso_pred": iso_pred,
        }

    def clear_logs(self):
        self.anomaly_log = []
        self.session_log = []