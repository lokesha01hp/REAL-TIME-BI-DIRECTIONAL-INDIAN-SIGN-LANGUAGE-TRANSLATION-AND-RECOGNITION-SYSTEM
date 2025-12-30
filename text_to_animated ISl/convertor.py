#!/usr/bin/env python3
"""
ISL Video -> Original Anime Style
(flicker reduced, free-moving body)
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import List, Tuple


class AnimeRenderer:
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height

        self.red = (0, 0, 255)
        self.bg = (30, 30, 30)

        self.finger_colors = {
            "thumb": (0, 165, 255),
            "index": (0, 255, 255),
            "middle": (255, 255, 0),
            "ring": (0, 255, 0),
            "little": (255, 0, 255),
        }

        self.body_th = 8
        self.hand_th = 8
        self.face_th = 6
        self.min_seg_len_sq = 9

    def canvas(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.bg
        return img

    def norm(self, lms, shape):
        pts = []
        for lm in lms:
            x = int(lm.x * self.width)
            y = int(lm.y * self.height)
            pts.append((x, y, lm.z if hasattr(lm, "z") else 0))
        return pts

    def _line(self, img, p1, p2, c, t):
        if not p1 or not p2:
            return

        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        dx, dy = x2 - x1, y2 - y1

        if dx * dx + dy * dy < self.min_seg_len_sq:
            return

        if (
            0 <= x1 < self.width
            and 0 <= y1 < self.height
            and 0 <= x2 < self.width
            and 0 <= y2 < self.height
        ):
            cv2.line(img, (x1, y1), (x2, y2), c, t, cv2.LINE_AA)

    # ---------- BODY ----------
    def draw_pose(self, img, pose):
        if not pose or len(pose) < 33:
            return

        ls, rs, lh, rh = pose[11], pose[12], pose[23], pose[24]

        self._line(img, ls, rs, self.red, self.body_th)
        self._line(img, ls, lh, self.red, self.body_th)
        self._line(img, rs, rh, self.red, self.body_th)
        self._line(img, lh, rh, self.red, self.body_th)

        for s, e, w in [(11, 13, 15), (12, 14, 16)]:
            self._line(img, pose[s], pose[e], self.red, self.body_th)
            self._line(img, pose[e], pose[w], self.red, self.body_th)

    # ---------- HANDS ----------
    def draw_hand(self, img, hand):
        if not hand or len(hand) < 21:
            return

        wx, wy = hand[0][0], hand[0][1]
        scale = 0.9

        hand_pts = []
        for (x, y, z) in hand:
            sx = int(wx + (x - wx) * scale)
            sy = int(wy + (y - wy) * scale)
            hand_pts.append((sx, sy, z))

        palm_color = (200, 200, 200)

        for i, j in [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]:
            self._line(img, hand_pts[i], hand_pts[j], palm_color, self.hand_th)

        fingers = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "little": [17, 18, 19, 20],
        }

        for name, idxs in fingers.items():
            col = self.finger_colors[name]
            for k in range(len(idxs) - 1):
                self._line(
                    img,
                    hand_pts[idxs[k]],
                    hand_pts[idxs[k + 1]],
                    col,
                    self.hand_th,
                )

    # ---------- FACE ----------
    def draw_face(self, img, face):
        if not face or len(face) < 468:
            return

        face_oval = [
            10, 338, 297, 332, 284, 251, 389, 356, 454,
            323, 361, 288, 397, 365, 379, 378, 400, 377,
            152, 148, 176, 149, 150, 136, 172, 58, 132,
            93, 234, 127, 162, 21, 54, 103, 67, 109,
        ]

        left_brow = [70, 63, 105, 66, 107]
        right_brow = [336, 296, 334, 293, 300]
        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263]
        lips_outer = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
        ]

        def poly(idx, closed=False, t=None):
            if t is None:
                t = self.face_th

            pts = []
            for i in idx:
                if i < len(face):
                    x, y = int(face[i][0]), int(face[i][1])
                    pts.append((x, y))

            if len(pts) > 1:
                arr = np.array(pts, np.int32)
                cv2.polylines(img, [arr], closed, self.red, t, cv2.LINE_AA)

        poly(face_oval, False, self.face_th)
        poly(left_brow, False, self.face_th - 2)
        poly(right_brow, False, self.face_th - 2)
        poly(left_eye, True, self.face_th - 2)
        poly(right_eye, True, self.face_th - 2)
        poly(lips_outer, True, self.face_th)


class ISLToAnimeSmooth:
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.renderer = AnimeRenderer(width, height)

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.alpha = 0.85
        self.prev_pose = None
        self.prev_lh = None
        self.prev_rh = None
        self.prev_face = None

        self.last_pose = None
        self.last_lh = None
        self.last_rh = None
        self.last_face = None

        self.max_cache_frames = 2
        self.miss_pose = 0
        self.miss_lh = 0
        self.miss_rh = 0
        self.miss_face = 0

    def _smooth(self, cur, prev):
        if prev is None or not prev or len(prev) != len(cur):
            return cur

        out = []
        a = self.alpha

        for (cx, cy, cz), (px, py, pz) in zip(cur, prev):
            sx = a * px + (1 - a) * cx
            sy = a * py + (1 - a) * cy
            sz = a * pz + (1 - a) * cz
            out.append((int(sx), int(sy), sz))

        return out

    def convert(self, input_path: str, output_path: str) -> int:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, fps, (self.width, self.height)
        )

        frame_count = 0
        all_frames = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.holistic.process(rgb)
            img = self.renderer.canvas()

            pose = None
            if res.pose_landmarks:
                p_raw = self.renderer.norm(
                    res.pose_landmarks.landmark, (self.height, self.width)
                )
                pose = self._smooth(p_raw, self.prev_pose)
                self.prev_pose = pose
                self.last_pose = pose
                self.miss_pose = 0
            elif self.last_pose is not None and self.miss_pose < self.max_cache_frames:
                pose = self.last_pose
                self.miss_pose += 1

            lh = None
            if res.left_hand_landmarks:
                lh_raw = self.renderer.norm(
                    res.left_hand_landmarks.landmark, (self.height, self.width)
                )
                lh = self._smooth(lh_raw, self.prev_lh)
                self.prev_lh = lh
                self.last_lh = lh
                self.miss_lh = 0
            elif self.last_lh is not None and self.miss_lh < self.max_cache_frames:
                lh = self.last_lh
                self.miss_lh += 1

            rh = None
            if res.right_hand_landmarks:
                rh_raw = self.renderer.norm(
                    res.right_hand_landmarks.landmark, (self.height, self.width)
                )
                rh = self._smooth(rh_raw, self.prev_rh)
                self.prev_rh = rh
                self.last_rh = rh
                self.miss_rh = 0
            elif self.last_rh is not None and self.miss_rh < self.max_cache_frames:
                rh = self.last_rh
                self.miss_rh += 1

            face = None
            if res.face_landmarks:
                f_raw = self.renderer.norm(
                    res.face_landmarks.landmark, (self.height, self.width)
                )
                face = self._smooth(f_raw, self.prev_face)
                self.prev_face = face
                self.last_face = face
                self.miss_face = 0
            elif self.last_face is not None and self.miss_face < self.max_cache_frames:
                face = self.last_face
                self.miss_face += 1

            all_frames.append({
                "pose": pose,
                "lh": lh,
                "rh": rh,
                "face": face,
            })

            self.renderer.draw_pose(img, pose)
            self.renderer.draw_face(img, face)
            self.renderer.draw_hand(img, lh)
            self.renderer.draw_hand(img, rh)

            out.write(img)
            frame_count += 1

        cap.release()
        out.release()

        np.save(
            output_path + ".npy",
            np.array(all_frames, dtype=object),
            allow_pickle=True,
        )

        print("Landmarks saved to:", output_path + ".npy")
        return frame_count


def main():
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python3 isl_anime_original_smooth.py "
            "<input_video.mp4> [output_video.mp4]"
        )
        return

    input_video = sys.argv[1]
    output_video = (
        sys.argv[2] if len(sys.argv) > 2 else "Main_dict/what.mp4"
    )

    if not Path(input_video).exists():
        print(f"Error: Video not found: {input_video}")
        return

    conv = ISLToAnimeSmooth(1920, 1080)
    frames = conv.convert(input_video, output_video)
    print("Done. Frames:", frames, "Output:", output_video)


if __name__ == "__main__":
    main()
