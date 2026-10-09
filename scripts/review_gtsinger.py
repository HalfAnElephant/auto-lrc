#!/usr/bin/env python3
"""Serve a local, blind, original-sample boundary annotation workspace."""
import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
from urllib.parse import parse_qs, urlparse
import wave

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from t2l.datasets.annotations import IDENTITY_FIELDS, LABEL_FIELDS, ONSET_DEFINITION, row_key, validate_label

ASSETS = REPO_ROOT / "scripts" / "review_ui"


class ReviewStore:
    def __init__(self, form_path, session_path):
        self.session_path = Path(session_path).absolute()
        if self.session_path.is_symlink() or self.session_path == Path(form_path).resolve():
            raise ValueError("Session must not overwrite the input form or a symlink")
        raw = Path(form_path).read_bytes()
        rows = json.loads(raw)
        if not isinstance(rows, list) or not rows:
            raise ValueError("The blind form must contain a nonempty array")
        self.audio, self.original = {}, {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(IDENTITY_FIELDS + LABEL_FIELDS):
                raise ValueError("Use the blind manual-review.json form, without source candidate fields")
            key = row_key(row)
            if key in self.original:
                raise ValueError("Duplicate boundary identity")
            segment_id = row["segment_id"]
            if segment_id not in self.audio:
                path = Path(row["audio_path"])
                if not path.is_absolute() or not path.is_file():
                    raise ValueError("Missing original WAV: " + str(path))
                with wave.open(str(path), "rb") as wav:
                    if wav.getcomptype() != "NONE" or wav.getsampwidth() not in (1, 2, 3, 4):
                        raise ValueError("Only original PCM WAV is supported")
                    metadata = {"sample_rate": wav.getframerate(), "num_samples": wav.getnframes(),
                                "channels": wav.getnchannels(), "sample_width": wav.getsampwidth()}
                    pcm = wav.readframes(metadata["num_samples"])
                    if len(pcm) != metadata["num_samples"] * metadata["channels"] * metadata["sample_width"]:
                        raise ValueError("Truncated WAV")
                self.audio[segment_id] = dict(metadata, path=path, pcm=pcm,
                                             sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            audio = self.audio[segment_id]
            if row["audio_path"] != str(audio["path"]) or row["sample_rate"] != audio["sample_rate"]:
                raise ValueError("Form and original WAV timeline disagree")
            if self.session_path.resolve() == audio["path"].resolve():
                raise ValueError("Session must not overwrite original audio")
            validated = validate_label(row, row, audio["num_samples"])
            if validated["review_status"] != "pending":
                raise ValueError("Start from an unlabelled blind form; restore labels with --session")
            self.original[key] = validated
        self.segment_ids = list(self.audio)
        fingerprint = hashlib.sha256(raw)
        for segment_id in self.segment_ids:
            fingerprint.update(self.audio[segment_id]["sha256"].encode("ascii"))
        self.fingerprint = fingerprint.hexdigest()
        self.rows = list(self.original.values())
        self.revision, self.history = 0, []
        self.lock = threading.RLock()
        if self.session_path.exists():
            saved = json.loads(self.session_path.read_text(encoding="utf8"))
            if saved.get("form_fingerprint") != self.fingerprint or saved.get("protocol") != "blind":
                raise ValueError("Existing session belongs to another form or protocol")
            if not isinstance(saved.get("revision"), int) or isinstance(saved["revision"], bool) or saved["revision"] < 0:
                raise ValueError("Invalid saved revision")
            restored = saved.get("rows", [])
            if len(restored) != len(self.rows) or [row_key(r) for r in restored] != list(self.original):
                raise ValueError("Saved session coverage changed")
            self.rows = [validate_label(self.original[row_key(r)], r, self.audio[r["segment_id"]]["num_samples"])
                         for r in restored]
            self.revision, self.history = saved["revision"], saved.get("history", [])

    def document(self):
        return {"schema_version": 1, "protocol": "blind", "onset_definition": ONSET_DEFINITION,
                "form_fingerprint": self.fingerprint, "revision": self.revision,
                "audio_sources": {key: {field: audio[field] for field in ("sha256", "sample_rate", "num_samples")}
                                  for key, audio in self.audio.items()},
                "rows": self.rows, "history": self.history,
                "independence": "Labels are submitted by the named reviewer; independence and adjudication require external verification."}

    def verify_audio(self, segment_id):
        audio = self.audio[segment_id]
        if hashlib.sha256(audio["path"].read_bytes()).hexdigest() != audio["sha256"]:
            raise ValueError("Original audio changed since this review began")
        return audio

    @contextmanager
    def session_lock(self):
        import fcntl
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.session_path.with_name(self.session_path.name + ".lock")
        if lock_path.is_symlink():
            raise ValueError("Session lock must not be a symlink")
        with lock_path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                if self.session_path.exists():
                    saved = json.loads(self.session_path.read_text(encoding="utf8"))
                    if saved.get("form_fingerprint") != self.fingerprint or saved.get("revision") != self.revision:
                        raise ValueError("Session changed in another process; restart before saving")
                elif self.revision:
                    raise ValueError("Saved session disappeared; restart before saving")
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def update(self, index, label, expected_revision):
        with self.lock, self.session_lock():
            if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self.rows) or
                    not isinstance(expected_revision, int) or isinstance(expected_revision, bool)):
                raise ValueError("Invalid row index or revision")
            if expected_revision != self.revision:
                raise ValueError("Session changed in another tab; reload before saving")
            original = self.rows[index]
            audio = self.verify_audio(original["segment_id"])
            reviewed = validate_label(self.original[row_key(original)], label, audio["num_samples"], stamp=True)
            if reviewed["review_status"] == "pending":
                raise ValueError("Use an explicit reviewed or ambiguous label")
            rows = list(self.rows)
            rows[index] = reviewed
            history = self.history + [{"revision": self.revision + 1, "row_index": index,
                                       "previous": original, "submitted": reviewed}]
            document = dict(self.document(), rows=rows, history=history, revision=self.revision + 1)
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
            if self.session_path.is_symlink():
                raise ValueError("Session path became a symlink")
            fd, name = tempfile.mkstemp(prefix=".review-", dir=self.session_path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf8") as target:
                    json.dump(document, target, ensure_ascii=False, indent=2, allow_nan=False)
                    target.write("\n")
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(name, self.session_path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
            self.rows, self.history, self.revision = rows, history, self.revision + 1
            return self.document()

    def signal(self, segment_index, start, end, width):
        import numpy as np
        if not 0 <= segment_index < len(self.segment_ids):
            raise ValueError("Unknown segment")
        audio = self.verify_audio(self.segment_ids[segment_index])
        if not 0 <= start < end <= audio["num_samples"] or not 64 <= width <= 2000:
            raise ValueError("Invalid signal window")
        sample_width = audio["sample_width"]
        if sample_width == 3:
            raw = np.frombuffer(audio["pcm"], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            values = raw[:, 0] | raw[:, 1] << 8 | raw[:, 2] << 16
            values = (values ^ (1 << 23)) - (1 << 23)
        elif sample_width == 1:
            values = np.frombuffer(audio["pcm"], dtype=np.uint8).astype(np.int32) - 128
        else:
            values = np.frombuffer(audio["pcm"], dtype="<i" + str(sample_width))
        values = values.astype(np.float64).reshape(-1, audio["channels"]).mean(axis=1)
        values /= float(1 << (sample_width * 8 - 1))
        points = np.linspace(start, end, min(width, end - start) + 1, dtype=int)
        waveform = [[float(values[a:b].min()), float(values[a:b].max())] for a, b in zip(points[:-1], points[1:])]
        nfft = min(1024, end - start)
        hop = max(1, math.ceil(max(1, end - start - nfft + 1) / 256))
        centers, spectra = [], []
        frequency_indices = np.unique(np.linspace(0, min(nfft // 2, int(nfft * 10000 / audio["sample_rate"])), 96, dtype=int))
        for position in range(start, end - nfft + 1, hop):
            power = np.abs(np.fft.rfft(values[position:position + nfft] * np.hanning(nfft)))
            spectra.append(np.clip(20 * np.log10(np.maximum(power[frequency_indices] / nfft, 1e-6)), -120, 0).tolist())
            centers.append(position + (nfft - 1) / 2)
        return {"start_sample": start, "end_sample": end, "waveform": waveform,
                "spectrogram_db": spectra, "spectrogram_centers_sample": centers,
                "frequencies_hz": (frequency_indices * audio["sample_rate"] / nfft).tolist(),
                "sample_rate": audio["sample_rate"], "num_samples": audio["num_samples"]}


def make_handler(store, token):
    class Handler(BaseHTTPRequestHandler):
        def local_request(self, *, require_origin=False):
            host = self.headers.get("Host")
            allowed = {"127.0.0.1:%d" % self.server.server_port, "localhost:%d" % self.server.server_port}
            origin = self.headers.get("Origin")
            return host in allowed and (origin == "http://" + str(host) or
                                        (origin is None and not require_origin))

        def send(self, payload, content_type="application/json", status=200, extra=None):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if not self.local_request():
                self.send({"error": "Only this local review origin is allowed"}, status=403)
                return
            request = urlparse(self.path)
            try:
                if request.path == "/api/form":
                    with store.lock:
                        metadata = [{"segment_id": key, **{field: store.audio[key][field] for field in
                                     ("sample_rate", "num_samples", "channels")}} for key in store.segment_ids]
                        self.send(dict(store.document(), segments=metadata, token=token))
                elif request.path == "/api/signal":
                    query = parse_qs(request.query)
                    self.send(store.signal(*(int(query[key][0]) for key in ("segment", "start", "end", "width"))))
                elif request.path == "/export":
                    with store.lock:
                        self.send(store.document(), extra={"Content-Disposition": 'attachment; filename="boundary-review.json"'})
                elif request.path.startswith("/audio/"):
                    index = int(request.path[len("/audio/"):])
                    if not 0 <= index < len(store.segment_ids):
                        raise ValueError("Unknown segment")
                    payload = store.verify_audio(store.segment_ids[index])["path"].read_bytes()
                    header = self.headers.get("Range")
                    if header:
                        if not header.startswith("bytes=") or "," in header:
                            raise ValueError("Unsupported audio range")
                        first, last = header[6:].split("-", 1)
                        begin = int(first) if first else max(0, len(payload) - int(last))
                        finish = int(last) + 1 if first and last else len(payload)
                        if not 0 <= begin < finish <= len(payload):
                            raise ValueError("Invalid audio range")
                        self.send(payload[begin:finish], "audio/wav", 206,
                                  {"Content-Range": "bytes %d-%d/%d" % (begin, finish - 1, len(payload)), "Accept-Ranges": "bytes"})
                    else:
                        self.send(payload, "audio/wav", extra={"Accept-Ranges": "bytes"})
                else:
                    names = {"/": ("index.html", "text/html; charset=utf-8"),
                             "/app.js": ("app.js", "application/javascript"),
                             "/style.css": ("style.css", "text/css"),
                             "/lucide.min.js": ("lucide.min.js", "application/javascript")}
                    if request.path not in names:
                        self.send({"error": "Not found"}, status=404)
                        return
                    name, content_type = names[request.path]
                    self.send((ASSETS / name).read_bytes(), content_type)
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                self.send({"error": str(error)}, status=400)

        def do_POST(self):
            if not self.local_request(require_origin=True) or self.path != "/api/label" or self.headers.get("X-Review-Token") != token:
                self.send({"error": "Invalid local review request"}, status=403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                self.send(store.update(request["row_index"], request["label"], request["revision"]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.send({"error": str(error)}, status=400)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--form", required=True, type=Path)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    store = ReviewStore(args.form, args.session)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, secrets.token_urlsafe(32)))
    print("Blind boundary review: http://127.0.0.1:%d" % server.server_port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
