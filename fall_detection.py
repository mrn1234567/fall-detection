# -*- coding: utf-8 -*-
"""基于 MediaPipe Pose 的单人跌倒检测器。"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


@dataclass
class PoseAnalysis:
    bbox: tuple[float, float, float, float]
    ratio: float
    torso_angle: Optional[float]
    hip_y: Optional[float]
    fall_candidate: bool
    confidence: str
    reason: str


class FallDetector:
    """实时跌倒检测。角度指躯干与水平线夹角：站立约 90°，平躺约 0°。"""

    def __init__(self, fall_angle=38.0, recover_angle=62.0, fall_ratio=0.92,
                 fall_frames=7, recover_frames=10, min_visibility=0.45):
        if mp is None:
            raise RuntimeError("缺少 mediapipe，请运行：python -m pip install -r requirements.txt")
        self.fall_angle = fall_angle
        self.recover_angle = recover_angle
        self.fall_ratio = fall_ratio
        self.fall_frames = fall_frames
        self.recover_frames = recover_frames
        self.min_vis = min_visibility
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.55, min_tracking_confidence=0.55)
        self.fall_count = self.recover_count = self.missing_count = 0
        self.is_fall = False
        self.bbox = None
        self.conf = "low"
        self.torso_angle = None
        self.ratio = 0.0
        self.hip_y = None
        self.reason = ""

    @staticmethod
    def _visible(point: Any, threshold: float) -> bool:
        return point is not None and getattr(point, "visibility", 0.0) >= threshold

    def _analyze(self, landmarks: Any) -> Optional[PoseAnalysis]:
        points = list(getattr(landmarks, "landmark", landmarks))
        visible = [p for p in points if self._visible(p, self.min_vis)]
        if len(visible) < 6:
            return None

        # 分位数边界可减少挥手和单个关键点漂移造成的宽高比误差。
        xs = np.asarray([p.x for p in visible], dtype=float)
        ys = np.asarray([p.y for p in visible], dtype=float)
        min_x, max_x = np.percentile(xs, [3, 97])
        min_y, max_y = np.percentile(ys, [3, 97])
        width = max(float(max_x - min_x), 1e-6)
        height = max(float(max_y - min_y), 1e-6)
        ratio = height / width

        shoulders, hips = points[11:13], points[23:25]
        torso_ready = all(self._visible(p, self.min_vis) for p in shoulders + hips)
        torso_angle = hip_y = None
        if torso_ready:
            sx = (shoulders[0].x + shoulders[1].x) / 2.0
            sy = (shoulders[0].y + shoulders[1].y) / 2.0
            hx = (hips[0].x + hips[1].x) / 2.0
            hip_y = (hips[0].y + hips[1].y) / 2.0
            # 修复旧算法：直接计算与水平线夹角，不能再用 90-angle。
            torso_angle = float(np.degrees(np.arctan2(abs(hip_y - sy), abs(hx - sx) + 1e-9)))

        lower_ids = (25, 26, 27, 28, 31, 32)
        has_lower = sum(self._visible(points[i], self.min_vis) for i in lower_ids) >= 2
        enough_size = height >= 0.16 or width >= 0.16
        angle_fall = torso_angle is not None and torso_angle <= self.fall_angle
        shape_fall = has_lower and ratio <= self.fall_ratio
        combined_fall = torso_angle is not None and torso_angle <= 52.0 and shape_fall
        fall_candidate = enough_size and (angle_fall or combined_fall)

        reason = ("躯干接近水平" if angle_fall else
                  "躯干倾斜且身体横向" if combined_fall else "姿态正常")
        if torso_ready and has_lower and len(visible) >= 12:
            confidence = "high"
        elif torso_ready or len(visible) >= 9:
            confidence = "medium"
        else:
            confidence = "low"

        pad = 0.025
        bx, by = max(0.0, float(min_x)-pad), max(0.0, float(min_y)-pad)
        bbox = (bx, by, min(1.0-bx, width+2*pad), min(1.0-by, height+2*pad))
        return PoseAnalysis(bbox, ratio, torso_angle, hip_y, fall_candidate,
                            confidence, reason)

    def _update_state(self, a: PoseAnalysis) -> str:
        if a.fall_candidate:
            self.fall_count = min(self.fall_frames, self.fall_count + 1)
            self.recover_count = 0
            if self.fall_count >= self.fall_frames:
                self.is_fall = True
        else:
            self.fall_count = 0
            upright = a.torso_angle is not None and a.torso_angle >= self.recover_angle
            if upright:
                self.recover_count = min(self.recover_frames, self.recover_count + 1)
                if self.recover_count >= self.recover_frames:
                    self.is_fall = False
            else:
                self.recover_count = 0
        return "fall" if self.is_fall else "normal"

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            self.missing_count += 1
            if self.missing_count >= 15 and not self.is_fall:
                self._clear_measurements()
            return frame, "fall" if self.is_fall else "no_person"

        analysis = self._analyze(results.pose_landmarks)
        if analysis is None:
            self.missing_count += 1
            return frame, "fall" if self.is_fall else "few_points"
        self.missing_count = 0
        self.bbox, self.ratio = analysis.bbox, analysis.ratio
        self.torso_angle, self.hip_y = analysis.torso_angle, analysis.hip_y
        self.conf, self.reason = analysis.confidence, analysis.reason
        status = self._update_state(analysis)
        self._draw(frame)
        return frame, status

    def _draw(self, frame):
        if self.bbox is None:
            return
        fh, fw = frame.shape[:2]
        x, y, w, h = self.bbox
        p1, p2 = (int(x*fw), int(y*fh)), (int((x+w)*fw), int((y+h)*fh))
        color = (0, 0, 255) if self.is_fall else (0, 210, 80)
        label = "FALL" if self.is_fall else "NORMAL"
        angle = "--" if self.torso_angle is None else f"{self.torso_angle:.0f}deg"
        cv2.rectangle(frame, p1, p2, color, 3)
        cv2.putText(frame, f"{label}  angle={angle}  ratio={self.ratio:.2f}",
                    (p1[0], max(28, p1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, color, 2, cv2.LINE_AA)

    def _clear_measurements(self):
        self.bbox, self.torso_angle, self.hip_y = None, None, None
        self.ratio, self.reason = 0.0, ""

    def reset_alarm(self):
        self.fall_count = self.recover_count = 0
        self.is_fall = False

    def release(self):
        self.pose.close()
