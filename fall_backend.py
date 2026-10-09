# -*- coding: utf-8 -*-
"""人员跌倒检测 Web 后端。"""

from pathlib import Path
from threading import Lock
import atexit
import cv2
from flask import Flask, Response, jsonify, send_file
from fall_detection import FallDetector

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
detector = FallDetector()
camera_lock = Lock()

CURRENT = {
    "status": "idle", "is_fall": False, "ratio": 0.0, "angle": None,
    "confidence": "low", "reason": "等待摄像头", "fall_count": 0,
    "recover_count": 0,
}


def update_status(status):
    CURRENT.update({
        "status": status,
        "is_fall": detector.is_fall,
        "ratio": round(detector.ratio, 3),
        "angle": None if detector.torso_angle is None else round(detector.torso_angle, 1),
        "confidence": detector.conf,
        "reason": detector.reason,
        "fall_count": detector.fall_count,
        "recover_count": detector.recover_count,
    })


def gen_frames():
    """一个浏览器视频流占用一个摄像头，避免重复打开设备。"""
    if not camera_lock.acquire(blocking=False):
        CURRENT.update(status="busy", reason="摄像头正在被另一个页面使用")
        return
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            CURRENT.update(status="camera_error", reason="无法打开摄像头")
            return
        while True:
            ok, frame = cap.read()
            if not ok:
                CURRENT.update(status="camera_error", reason="读取摄像头画面失败")
                break
            frame, status = detector.process(frame)
            update_status(status)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       jpeg.tobytes() + b"\r\n")
    finally:
        cap.release()
        camera_lock.release()


@app.get("/")
def index():
    return send_file(BASE_DIR / "人员跌倒检测.html")


@app.get("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
def status():
    return jsonify(CURRENT)


@app.post("/reset")
def reset():
    detector.reset_alarm()
    update_status("normal")
    return jsonify({"ok": True})


atexit.register(detector.release)

if __name__ == "__main__":
    print("人员跌倒检测已启动：http://127.0.0.1:5000")
    print("按 Ctrl+C 停止")
    app.run(host="127.0.0.1", port=5000, threaded=True)
