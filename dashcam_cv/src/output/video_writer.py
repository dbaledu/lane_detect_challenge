"""Video Writer module for saving annotated analysis video."""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class AnnotatedVideoWriter:
    """Writes annotated vision output frames to video file."""

    def __init__(self, output_path: str, fps: float, size: Tuple[int, int], codec: str = "mp4v"):
        self.output_path = output_path
        self.fps = fps
        self.size = size
        self.codec = codec

        parent = Path(output_path).parent
        parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, size)

        if not self.writer.isOpened():
            # Fallback codec if requested codec fails
            fallback_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(output_path, fallback_fourcc, fps, size)
            if not self.writer.isOpened():
                raise IOError(f"Failed to open VideoWriter at {output_path}")

    def write(self, frame: np.ndarray):
        """Writes a single frame."""
        if self.writer is not None and self.writer.isOpened():
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size)
            self.writer.write(frame)

    def release(self):
        """Releases the video writer."""
        if self.writer is not None and self.writer.isOpened():
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
