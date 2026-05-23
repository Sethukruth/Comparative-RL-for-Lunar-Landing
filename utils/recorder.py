"""
GIF / video recorder for evaluation rollouts.

Records frames from ``gymnasium`` environments rendered in ``rgb_array``
mode, then writes them to GIF (imageio) or MP4 (opencv) files.
"""

import os
from typing import List, Optional

import imageio
import numpy as np


class Recorder:
    """Accumulate RGB frames and save them as a GIF or MP4.

    Example::

        rec = Recorder()
        rec.capture(env.render())   # inside step loop
        rec.save_gif("results/landing.gif")
    """

    def __init__(self) -> None:
        self.frames: List[np.ndarray] = []

    def capture(self, frame: np.ndarray) -> None:
        """Append a single RGB frame.

        Args:
            frame: HxWx3 uint8 array from ``env.render()``.
        """
        if frame is not None:
            self.frames.append(frame)

    def reset(self) -> None:
        """Clear all captured frames."""
        self.frames.clear()

    def save_gif(
        self,
        path: str,
        fps: int = 30,
        optimize: bool = True,
    ) -> str:
        """Write captured frames to a GIF file.

        Args:
            path: Output file path.
            fps: Frames per second.
            optimize: Whether to apply size optimization.

        Returns:
            Absolute path to the saved GIF.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if not self.frames:
            raise ValueError("No frames captured — cannot save GIF.")

        imageio.mimsave(
            path,
            self.frames,
            fps=fps,
            loop=0,
        )
        return os.path.abspath(path)

    def save_mp4(self, path: str, fps: int = 30) -> str:
        """Write captured frames to an MP4 file using OpenCV.

        Args:
            path: Output file path.
            fps: Frames per second.

        Returns:
            Absolute path to the saved MP4.
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for MP4 export: pip install opencv-python"
            ) from exc

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if not self.frames:
            raise ValueError("No frames captured — cannot save MP4.")

        h, w, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

        for frame in self.frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        return os.path.abspath(path)

    def __len__(self) -> int:
        return len(self.frames)
