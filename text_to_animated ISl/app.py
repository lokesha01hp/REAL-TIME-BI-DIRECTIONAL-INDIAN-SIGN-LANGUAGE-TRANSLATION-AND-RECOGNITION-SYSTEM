from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import speech_recognition as sr
from pathlib import Path
import cv2
import av
import json
import re
import os
import tempfile
import uuid
import fractions
from flask import redirect, current_app
app = Flask(__name__)
socketio = SocketIO(app)

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "isl_index.json"
STATIC_OUTPUT_DIR = BASE_DIR / "static" / "outputs"
STATIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(INDEX_PATH) as f:
    ISL_INDEX = json.load(f)

def _abs(p: str) -> str:
    return str((BASE_DIR / p).resolve())

WORD2VIDEO = {k: _abs(v) for k, v in ISL_INDEX.get("words", {}).items()}
ALPHA2VIDEO = {k: _abs(v) for k, v in ISL_INDEX.get("alphabets", {}).items()}
NUM2VIDEO = {k: _abs(v) for k, v in ISL_INDEX.get("numbers", {}).items()}

GOOGLE_API_KEY = "AIzaSyAxNvj-2Ll0696bX8snJwsf1xjyjyHs760"

def tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

def text_to_video_paths(text: str):
    tokens = tokenize(text)
    if not tokens:
        return []
    paths = [WORD2VIDEO.get(t) for t in tokens]
    if None not in paths:
        return paths
    phrase = "_".join(tokens)
    if WORD2VIDEO.get(phrase):
        return [WORD2VIDEO[phrase]]
    paths = []
    for ch in text.lower():
        if ch.isalpha():
            p = ALPHA2VIDEO.get(ch)
        elif ch.isdigit():
            p = NUM2VIDEO.get(ch)
        else:
            continue
        if p:
            paths.append(p)
    return paths

def concat_videos_to_h264_mp4(video_paths, output_path: Path, speed: float = 1.0):
    if not video_paths:
        raise ValueError("No videos to concatenate")

    cap0 = cv2.VideoCapture(video_paths[0])
    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open: {video_paths[0]}")
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    src_fps = cap0.get(cv2.CAP_PROP_FPS) or 25
    cap0.release()

    out_fps_float = max(1.0, src_fps * speed)
    out_fps = fractions.Fraction(int(round(out_fps_float)), 1)

    out = av.open(str(output_path), mode="w", format="mp4")

    vstream = out.add_stream("h264", rate=out_fps)
    vstream.width = width
    vstream.height = height
    vstream.pix_fmt = "yuv420p"
    vstream.bit_rate = 800000

    frame_index = 0
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            vf = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            vf = vf.reformat(format="yuv420p")
            vf.pts = frame_index
            frame_index += 1
            for pkt in vstream.encode(vf):
                out.mux(pkt)
        cap.release()
    for pkt in vstream.encode(None):
        out.mux(pkt)
    out.close()

def generate_video(text: str, speed: float = 1.0):
    paths = text_to_video_paths(text)
    if not paths:
        return None
    uid = uuid.uuid4().hex[:8]
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", text.lower())[:40] or "isl"
    out_file = STATIC_OUTPUT_DIR / f"{safe}_{uid}.mp4"
    concat_videos_to_h264_mp4(paths, out_file, speed)
    return f"/static/outputs/{out_file.name}"

@app.route('/')
def cover():
    return render_template('cover.html')

@app.route('/text_to_isl')
def text_to_isl():
    return render_template('text_to_isl.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text', '').strip()
    speed = data.get('speed', 1.0)
    if not text:
        return jsonify({'error': 'No text'})
    url = generate_video(text, speed)
    if not url:
        return jsonify({'error': 'No video'})
    return jsonify({'url': url})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files['audio']
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    audio_file.save(temp_path)
    r = sr.Recognizer()
    with sr.AudioFile(temp_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio, key=GOOGLE_API_KEY)
        url = generate_video(text)
        os.unlink(temp_path)
        if url:
            return jsonify({'url': url})
        return jsonify({'error': 'No video'})
    except Exception as e:
        os.unlink(temp_path)
        return jsonify({'error': str(e)})

@app.route('/static/outputs/<filename>')
def serve_video(filename):
    return send_from_directory(STATIC_OUTPUT_DIR, filename)

@app.route('/video_to_isl')
def video_to_isl():
    return redirect('http://localhost:9999')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)