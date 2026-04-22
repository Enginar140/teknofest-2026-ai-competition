"""
Yerel TEKNOFEST değerlendirme API'sinin basit taklidi (Flask).
Gerçek yarışma sunucusu TEKNOFEST tarafından sağlanır; bu sadece bağlantı ve istemci testi içindir.

Çalıştırma (HY kök klasöründen):
  pip install flask pillow
  python mock_evaluation_server.py

Sonra başka bir terminalde:
  cd teknofest_ai_system
  python main.py

Sunucu panelinde URL: http://127.0.0.1:5000/  | Kullanıcı: demo | Şifre: demo

TAKIM main.py için config/.env örneği:
  EVALUATION_SERVER_URL=http://127.0.0.1:5000/
  TEAM_NAME=demo
  PASSWORD=demo
  SESSION_NAME=demo_session
"""

from __future__ import annotations

import argparse
import io
import json
import os
import secrets
from typing import Any, Dict, List, Set

from flask import Flask, Response, jsonify, request

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("pip install pillow gerekli: " + str(e)) from e

app = Flask(__name__)

# Basit bellek içi oturum (üretim değil)
TOKENS: Set[str] = set()
DEMO_USER = os.environ.get("MOCK_TEAM_USER", "demo")
DEMO_PASS = os.environ.get("MOCK_TEAM_PASS", "demo")
NUM_FRAMES = int(os.environ.get("MOCK_FRAMES", "8"))
VIDEO_NAME = os.environ.get("MOCK_VIDEO_NAME", "demo_session")
BASE_HOST = os.environ.get("MOCK_PUBLIC_HOST", "http://127.0.0.1:5000").rstrip("/")

# Önbellekte JPEG baytları
_JPEG_CACHE: Dict[str, bytes] = {}


def _jpeg_for_index(i: int) -> bytes:
    key = str(i)
    if key in _JPEG_CACHE:
        return _JPEG_CACHE[key]
    img = Image.new("RGB", (640, 480), color=((i * 37) % 255, 90, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()
    _JPEG_CACHE[key] = data
    return data


@app.route("/auth/", methods=["POST"])
def auth() -> Any:
    u = request.form.get("username") or request.json and request.json.get("username")
    p = request.form.get("password") or request.json and request.json.get("password")
    if u == DEMO_USER and p == DEMO_PASS:
        t = secrets.token_hex(16)
        TOKENS.add(t)
        return jsonify({"token": t}), 200
    return jsonify({"detail": "Invalid credentials"}), 401


def _require_token() -> Response | None:
    h = request.headers.get("Authorization", "")
    if not h.startswith("Token "):
        return jsonify({"detail": "Authentication required"}), 401
    tok = h.split(None, 1)[1].strip()
    if tok not in TOKENS:
        return jsonify({"detail": "Invalid token"}), 401
    return None


@app.route("/frames/", methods=["GET"])
def get_frames() -> Any:
    err = _require_token()
    if err:
        return err
    frames: List[Dict[str, Any]] = []
    for i in range(NUM_FRAMES):
        frames.append(
            {
                "url": f"{BASE_HOST}/api/frame/{i}/",
                "image_url": f"/images/{i}.jpg",
                "video_name": VIDEO_NAME,
            }
        )
    return Response(json.dumps(frames), mimetype="application/json", status=200)


@app.route("/translation/", methods=["GET"])
def get_translation() -> Any:
    err = _require_token()
    if err:
        return err
    rows: List[Dict[str, Any]] = []
    for i in range(NUM_FRAMES):
        # İlk ~450 kare sağlıklı simülasyonu: kısa testte ilk 3 kare sağlıklı
        healthy = 1 if i < min(3, NUM_FRAMES) else 0
        rows.append(
            {
                "frame_id": i,
                "translation_x": str(round(i * 0.05, 4)),
                "translation_y": str(round(i * 0.02, 4)),
                "translation_z": str(15.0 + i * 0.01),
                "health_status": str(healthy),
                "gps_health_status": str(healthy),
            }
        )
    return Response(json.dumps(rows), mimetype="application/json", status=200)


@app.route("/media/images/<path:name>")
def media_image(name: str) -> Any:
    # İstemci: base + "media" + image_url  →  .../media/images/0.jpg
    base = name.replace(".jpg", "").replace(".jpeg", "")
    if not base.isdigit():
        return "not found", 404
    data = _jpeg_for_index(int(base))
    return Response(data, mimetype="image/jpeg")


@app.route("/prediction/", methods=["POST"])
def prediction() -> Any:
    err = _require_token()
    if err:
        return err
    # Gövdeyi doğrula (log için)
    try:
        body = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"detail": "Invalid JSON"}), 400
    if not body:
        return jsonify({"detail": "Empty body"}), 400
    return "", 201


def main() -> None:
    global BASE_HOST, NUM_FRAMES
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--frames", type=int, default=None, help="Kare sayısı (varsayılan MOCK_FRAMES veya 8)")
    args = p.parse_args()
    if args.frames is not None:
        NUM_FRAMES = max(1, args.frames)
    BASE_HOST = f"http://{args.host}:{args.port}"
    print(f"Mock TEKNOFEST server: {BASE_HOST}/")
    print(f"  User / pass: {DEMO_USER} / {DEMO_PASS}")
    print(f"  Frames: {NUM_FRAMES}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
