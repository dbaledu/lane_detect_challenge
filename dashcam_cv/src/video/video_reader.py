"""Video Reader module for dashcam video decoding with metadata extraction."""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Generator, Dict, Any


@dataclass
class VideoMetadata:
    """Video file metadata information."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    target_width: int
    target_height: int


class VideoReader:
    """Reads video files frame by frame with error resilience and accurate timing."""

    def __init__(self, video_path: str, target_width: int = 960, frame_stride: int = 1):
        self.video_path = video_path
        self.target_width = target_width
        self.frame_stride = max(1, int(frame_stride))
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video file at path: {video_path}")

        orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        if target_width > 0 and orig_w > 0:
            scale = target_width / orig_w
            tgt_w = target_width
            tgt_h = int(round(orig_h * scale))
        else:
            tgt_w = orig_w
            tgt_h = orig_h

        self.meta = VideoMetadata(
            width=orig_w,
            height=orig_h,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
            target_width=tgt_w,
            target_height=tgt_h
        )

        self.current_frame_idx = 0

    def frames(self, max_frames: int = 0) -> Generator[Tuple[bool, Optional[np.ndarray], int, float], None, None]:
        """Generator yielding (success, frame, frame_idx, timestamp_sec)."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            self.current_frame_idx += 1
            if max_frames > 0 and self.current_frame_idx > max_frames:
                break

            # Frame stride skipping
            if (self.current_frame_idx - 1) % self.frame_stride != 0:
                continue

            # Resize if requested
            if (self.meta.target_width != self.meta.width or
                    self.meta.target_height != self.meta.height):
                frame = cv2.resize(frame, (self.meta.target_width, self.meta.target_height),
                                   interpolation=cv2.INTER_AREA)

            timestamp = (self.current_frame_idx - 1) / self.meta.fps
            yield True, frame, self.current_frame_idx, timestamp

    def release(self):
        """Releases the underlying VideoCapture."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
