"""不需要摄像头和 MediaPipe 的核心判定测试。"""
import unittest
from types import SimpleNamespace
from fall_detection import FallDetector


def pose(fallen=False):
    pts = [SimpleNamespace(x=.5, y=.5, visibility=.95) for _ in range(33)]
    if fallen:
        # 肩、髋、膝、踝沿水平方向排列。
        pts[11].x, pts[11].y = .65, .55
        pts[12].x, pts[12].y = .65, .65
        pts[23].x, pts[23].y = .50, .55
        pts[24].x, pts[24].y = .50, .65
        for i, x in zip((25, 26, 27, 28, 31, 32), (.40, .40, .28, .28, .22, .22)):
            pts[i].x, pts[i].y = x, .60
        pts[0].x, pts[0].y = .78, .60
    else:
        pts[11].x, pts[11].y = .44, .30
        pts[12].x, pts[12].y = .56, .30
        pts[23].x, pts[23].y = .46, .55
        pts[24].x, pts[24].y = .54, .55
        for i, x, y in ((25,.46,.72),(26,.54,.72),(27,.45,.90),(28,.55,.90),(31,.44,.94),(32,.56,.94)):
            pts[i].x, pts[i].y = x, y
        pts[0].x, pts[0].y = .50, .12
    return pts


def detector_without_model():
    d = FallDetector.__new__(FallDetector)
    d.fall_angle, d.recover_angle, d.fall_ratio = 38, 62, .92
    d.fall_frames, d.recover_frames, d.min_vis = 7, 10, .45
    d.fall_count = d.recover_count = 0
    d.is_fall = False
    return d


class FallDetectorTests(unittest.TestCase):
    def test_standing_is_normal(self):
        d = detector_without_model()
        a = d._analyze(pose(False))
        self.assertGreater(a.torso_angle, 80)
        self.assertFalse(a.fall_candidate)

    def test_horizontal_pose_is_fall(self):
        d = detector_without_model()
        a = d._analyze(pose(True))
        self.assertLess(a.torso_angle, 10)
        self.assertTrue(a.fall_candidate)

    def test_temporal_confirmation_and_recovery(self):
        d = detector_without_model()
        fallen, standing = d._analyze(pose(True)), d._analyze(pose(False))
        for _ in range(6):
            self.assertEqual(d._update_state(fallen), "normal")
        self.assertEqual(d._update_state(fallen), "fall")
        for _ in range(9):
            self.assertEqual(d._update_state(standing), "fall")
        self.assertEqual(d._update_state(standing), "normal")


if __name__ == "__main__":
    unittest.main()
