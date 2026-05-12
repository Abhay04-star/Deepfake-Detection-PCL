"""Video/image helpers: frame extraction and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameExtractionResult:
    frames_bgr: list[np.ndarray]
    frame_indices: list[int]
    total_frames: int


def is_video_path(path: str | Path) -> bool:
    p = Path(path)
    ext = p.suffix.lower()
    return ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def read_image_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to read image (unsupported format or corrupted).")
    return img


def extract_video_frames(
    path: str | Path,
    max_frames: int = 32,
) -> FrameExtractionResult:
    """Extract up to max_frames evenly-spaced frames from a video.

    Returns BGR frames (OpenCV default).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # Some codecs don't report count; fallback to reading sequentially up to max_frames
        frames: list[np.ndarray] = []
        indices: list[int] = []
        i = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            indices.append(i)
            i += 1
        cap.release()
        return FrameExtractionResult(frames_bgr=frames, frame_indices=indices, total_frames=i)

    sample = min(max_frames, total)
    # Evenly spaced indices including first and last if possible
    idxs = np.linspace(0, max(total - 1, 0), num=sample, dtype=int).tolist()
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return FrameExtractionResult(frames_bgr=frames, frame_indices=idxs, total_frames=total)


def select_key_frames(
    frames_bgr: list[np.ndarray],
    frame_indices: list[int],
    key_frame_count: int = 8,
    original_total_frames: int | None = None,
    small_size: tuple[int, int] = (64, 64),
) -> FrameExtractionResult:
    """Select key frames for prediction (video-level aggregation).

    Strategy:
    - Resize frames to a small grayscale representation
    - Compute mean absolute differences between consecutive frames
    - Always keep first/last and add frames with the largest changes (peaks)

    Args:
        frames_bgr: Extracted BGR frames.
        frame_indices: Original frame indices corresponding to frames_bgr.
        key_frame_count: Number of frames to keep.
        small_size: Resize for cheap similarity/difference computation.

    Returns:
        FrameExtractionResult with selected frames and indices.
    """
    if len(frames_bgr) != len(frame_indices):
        raise ValueError("frames_bgr and frame_indices must have the same length.")
    n = len(frames_bgr)
    if n == 0:
        return FrameExtractionResult(frames_bgr=[], frame_indices=[], total_frames=0)
    if key_frame_count >= n:
        return FrameExtractionResult(frames_bgr=frames_bgr, frame_indices=frame_indices, total_frames=n)
    if key_frame_count <= 0:
        raise ValueError("key_frame_count must be > 0")

    # Prepare small grayscale frames for diff computation.
    small_gray: list[np.ndarray] = []
    for f in frames_bgr:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA)
        small_gray.append(small.astype(np.float32))

    # Differences between consecutive frames.
    diffs = []
    for i in range(1, n):
        dif = float(np.mean(np.abs(small_gray[i] - small_gray[i - 1])))
        diffs.append(dif)

    # Greedily choose from highest diffs while preserving first/last.
    selected_positions: list[int] = []
    selected_set: set[int] = set()

    selected_positions.append(0)
    selected_set.add(0)
    if n - 1 != 0:
        selected_positions.append(n - 1)
        selected_set.add(n - 1)

    # Order candidate positions (1..n-1) by diff descending.
    # diff index i corresponds to frame i+1 relative to i.
    candidates = list(range(1, n))
    candidates.sort(key=lambda pos: diffs[pos - 1], reverse=True)

    for pos in candidates:
        if len(selected_set) >= key_frame_count:
            break
        selected_set.add(pos)

    selected_positions = sorted(selected_set)
    selected_frames = [frames_bgr[i] for i in selected_positions]
    selected_indices = [frame_indices[i] for i in selected_positions]

    # total_frames: keep the original total for transparency.
    total_frames = int(original_total_frames) if original_total_frames is not None else n
    return FrameExtractionResult(
        frames_bgr=selected_frames,
        frame_indices=selected_indices,
        total_frames=total_frames,
    )


def extract_frames(
    path: str | Path,
    max_frames: int = 32,
) -> FrameExtractionResult:
    """Alias for extract_video_frames to match a simpler API name."""
    return extract_video_frames(path=path, max_frames=max_frames)


def resize_and_normalize_rgb(
    frame_bgr: np.ndarray,
    size: int = 224,
) -> np.ndarray:
    """Convert BGR->RGB, resize to (size,size), normalize to [0,1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    return x


def batch_preprocess_frames(
    frames_bgr: Iterable[np.ndarray],
    size: int = 224,
) -> np.ndarray:
    xs = [resize_and_normalize_rgb(f, size=size) for f in frames_bgr]
    return np.stack(xs, axis=0)

