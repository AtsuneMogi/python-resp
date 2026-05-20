from __future__ import annotations

import argparse
import collections
import dataclasses
import math
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - handled at runtime
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


BREATHING_MIN_HZ = 0.10  # 6 bpm
BREATHING_MAX_HZ = 0.50  # 30 bpm
GRAPH_HEIGHT = 220
GRAPH_WIDTH = 900
CHART_MARGIN = 32
DEFAULT_GRAPH_SPAN_SECONDS = 15.0
FACE_SIGNAL_WEIGHT = 0.35
CHEST_SIGNAL_WEIGHT = 0.65
MIN_SIGNAL_SAMPLES = 10


Rect = tuple[int, int, int, int]


@dataclasses.dataclass
class Sample:
    timestamp: float
    value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate respiration rate from a camera or video stream."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or video file path. Default: 0",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=30.0,
        help="Sliding analysis window in seconds. Default: 30",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=120,
        help="Minimum face width/height for detection in pixels. Default: 120",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.2,
        help="Smoothing factor for face/chest ROI stabilization. Lower values are smoother. Default: 0.2",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Display a live annotated window. This is enabled by default.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the live annotated window.",
    )

    args = parser.parse_args()
    if args.no_show:
        args.show = False
    return args


def open_capture(source: str):
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not installed. Install the dependencies with `pip install -r requirements.txt`."
        ) from _CV2_IMPORT_ERROR

    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    return capture


def load_face_cascade():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Failed to load face cascade: {cascade_path}")
    return cascade


def _detect_largest_face(gray: np.ndarray, cascade, min_face_size: int, scale_factor: float, min_neighbors: int):
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])


def detect_face_rect(frame: np.ndarray, cascade, min_face_size: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    for candidate in (gray, equalized, clahe):
        rect = _detect_largest_face(candidate, cascade, min_face_size, scale_factor=1.1, min_neighbors=5)
        if rect is not None:
            return rect

    # Fallback pass with a slightly more permissive setting for difficult lighting.
    for candidate in (equalized, clahe):
        rect = _detect_largest_face(candidate, cascade, min_face_size, scale_factor=1.05, min_neighbors=3)
        if rect is not None:
            return rect

    return None


def chest_roi_from_face(face_rect: Rect, frame_shape):
    x, y, w, h = face_rect
    chest_x = x - int(w * 0.20)
    chest_y = y + h + int(h * 0.05)
    chest_w = int(w * 1.40)
    chest_h = int(h * 1.80)

    chest_x = max(0, chest_x)
    chest_y = max(0, chest_y)
    chest_w = max(1, min(frame_shape[1] - chest_x, chest_w))
    chest_h = max(1, min(frame_shape[0] - chest_y, chest_h))
    return chest_x, chest_y, chest_w, chest_h


def face_signal_roi_from_face(face_rect: Rect, frame_shape):
    x, y, w, h = face_rect
    face_x = x + int(w * 0.20)
    face_y = y + int(h * 0.20)
    face_w = int(w * 0.60)
    face_h = int(h * 0.55)

    face_x = max(0, face_x)
    face_y = max(0, face_y)
    face_w = max(1, min(frame_shape[1] - face_x, face_w))
    face_h = max(1, min(frame_shape[0] - face_y, face_h))
    return face_x, face_y, face_w, face_h


def smooth_rect(previous_rect: Rect | None, current_rect: Rect, alpha: float) -> Rect:
    if previous_rect is None:
        return tuple(int(value) for value in current_rect)

    blended = [
        (1.0 - alpha) * previous_value + alpha * current_value
        for previous_value, current_value in zip(previous_rect, current_rect)
    ]
    return tuple(int(round(value)) for value in blended)


def extract_motion(prev_gray: np.ndarray | None, current_gray: np.ndarray):
    if prev_gray is None:
        return None

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=2,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    vertical_motion = float(np.mean(flow[..., 1]))
    return vertical_motion


def extract_signal(frame: np.ndarray, roi, prev_gray: np.ndarray | None):
    x, y, w, h = roi
    patch = frame[y : y + h, x : x + w]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    motion = extract_motion(prev_gray, gray)
    return gray, motion


def prepare_uniform_signal(samples: collections.abc.Sequence[Sample]):
    if len(samples) < MIN_SIGNAL_SAMPLES:
        return None

    timestamps = np.array([sample.timestamp for sample in samples], dtype=np.float64)
    values = np.array([sample.value for sample in samples], dtype=np.float64)

    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return None

    sampling_intervals = np.diff(timestamps)
    positive_intervals = sampling_intervals[sampling_intervals > 0]
    if len(positive_intervals) == 0:
        return None

    sampling_rate = 1.0 / float(np.median(positive_intervals))
    if sampling_rate < 2.0:
        return None

    target_count = max(32, int(math.ceil(duration * sampling_rate)))
    uniform_times = np.linspace(timestamps[0], timestamps[-1], target_count)
    interpolated_values = np.interp(uniform_times, timestamps, values)
    return uniform_times, interpolated_values, sampling_rate


def normalize_motion(value: float | None, history: collections.deque[float]):
    if value is None:
        return None
    if len(history) < 8:
        return value

    history_array = np.array(history, dtype=np.float64)
    mean = float(np.mean(history_array))
    std = float(np.std(history_array))
    if std < 1e-6:
        return value - mean
    normalized = (value - mean) / std
    return float(np.clip(normalized, -3.0, 3.0))


def fuse_signals(chest_value: float | None, face_value: float | None):
    available = []
    if chest_value is not None:
        available.append((chest_value, CHEST_SIGNAL_WEIGHT))
    if face_value is not None:
        available.append((face_value, FACE_SIGNAL_WEIGHT))

    if not available:
        return None

    weighted_sum = sum(value * weight for value, weight in available)
    total_weight = sum(weight for _, weight in available)
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def estimate_bpm(samples: list[Sample], min_hz: float = BREATHING_MIN_HZ, max_hz: float = BREATHING_MAX_HZ):
    prepared = prepare_uniform_signal(samples)
    if prepared is None:
        return None

    _, interpolated_values, sampling_rate = prepared
    detrended = interpolated_values - np.mean(interpolated_values)
    window = np.hanning(len(detrended))
    spectrum = np.fft.rfft(detrended * window)
    frequencies = np.fft.rfftfreq(len(detrended), d=1.0 / sampling_rate)
    power = np.abs(spectrum)

    band_mask = (frequencies >= min_hz) & (frequencies <= max_hz)
    if not np.any(band_mask):
        return None

    band_frequencies = frequencies[band_mask]
    band_power = power[band_mask]
    best_index = int(np.argmax(band_power))
    best_frequency = float(band_frequencies[best_index])
    if best_frequency <= 0:
        return None
    return best_frequency * 60.0


def bandpass_breathing_component(samples: collections.deque[Sample], min_hz: float = BREATHING_MIN_HZ, max_hz: float = BREATHING_MAX_HZ):
    prepared = prepare_uniform_signal(samples)
    if prepared is None:
        return None

    uniform_times, interpolated_values, sampling_rate = prepared
    centered = interpolated_values - np.mean(interpolated_values)
    spectrum = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sampling_rate)

    band_mask = (frequencies >= min_hz) & (frequencies <= max_hz)
    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[band_mask] = spectrum[band_mask]
    filtered_values = np.fft.irfft(filtered_spectrum, n=len(centered))
    return uniform_times, filtered_values


def select_graph_samples(samples: collections.deque[Sample], graph_span_seconds: float):
    if len(samples) == 0:
        return collections.deque()

    end_timestamp = samples[-1].timestamp
    cutoff = end_timestamp - graph_span_seconds
    selected = collections.deque(sample for sample in samples if sample.timestamp >= cutoff)
    if len(selected) >= 2:
        return selected
    return collections.deque(samples)


def draw_rect(frame: np.ndarray, rect: Rect, color: tuple[int, int, int], thickness: int = 2):
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


def draw_overlay(frame: np.ndarray, roi, bpm: float | None, sample_count: int):
    draw_rect(frame, roi, (0, 255, 0), 2)

    label = "Respiration (face+chest): -- bpm"
    if bpm is not None:
        label = f"Respiration (face+chest): {bpm:.1f} bpm"

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (50, 230, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_signal_graph(samples: collections.deque[Sample], bpm: float | None, graph_span_seconds: float):
    graph = np.full((GRAPH_HEIGHT, GRAPH_WIDTH, 3), 18, dtype=np.uint8)
    cv2.rectangle(graph, (0, 0), (GRAPH_WIDTH - 1, GRAPH_HEIGHT - 1), (60, 60, 60), 1)

    title = "Breathing component"
    if bpm is not None:
        title = f"Breathing component  |  {bpm:.1f} bpm"

    cv2.putText(
        graph,
        title,
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    graph_samples = select_graph_samples(samples, graph_span_seconds)
    filtered_result = bandpass_breathing_component(graph_samples)
    if filtered_result is None:
        cv2.putText(
            graph,
            "Waiting for motion samples...",
            (16, GRAPH_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        return graph

    timestamps, values = filtered_result
    cv2.putText(
        graph,
        f"span {graph_span_seconds:.0f}s",
        (GRAPH_WIDTH - 120, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    duration = max(1e-6, timestamps[-1] - timestamps[0])

    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if math.isclose(value_min, value_max):
        value_min -= 1.0
        value_max += 1.0

    usable_width = GRAPH_WIDTH - 2 * CHART_MARGIN
    usable_height = GRAPH_HEIGHT - 2 * CHART_MARGIN
    points = []
    for timestamp, value in zip(timestamps, values):
        x = CHART_MARGIN + int(((timestamp - timestamps[0]) / duration) * usable_width)
        y_ratio = (value - value_min) / (value_max - value_min)
        y = CHART_MARGIN + int((1.0 - y_ratio) * usable_height)
        points.append([x, y])

    points_array = np.array(points, dtype=np.int32)
    cv2.polylines(graph, [points_array], False, (80, 220, 255), 2, cv2.LINE_AA)

    mid_y = CHART_MARGIN + usable_height // 2
    cv2.line(graph, (CHART_MARGIN, mid_y), (GRAPH_WIDTH - CHART_MARGIN, mid_y), (80, 80, 80), 1)
    cv2.putText(
        graph,
        f"min {value_min:+.3f}",
        (16, GRAPH_HEIGHT - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        graph,
        f"max {value_max:+.3f}",
        (GRAPH_WIDTH - 120, GRAPH_HEIGHT - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    return graph


def compose_display(frame: np.ndarray, graph: np.ndarray):
    frame_width = max(frame.shape[1], graph.shape[1])
    if frame.shape[1] != frame_width:
        frame = cv2.resize(frame, (frame_width, frame.shape[0]), interpolation=cv2.INTER_AREA)
    if graph.shape[1] != frame_width:
        graph = cv2.resize(graph, (frame_width, graph.shape[0]), interpolation=cv2.INTER_AREA)
    return np.vstack([frame, graph])


def main() -> int:
    args = parse_args()
    smooth_alpha = min(max(args.smooth_alpha, 0.0), 1.0)
    graph_span_seconds = max(2.0, min(args.window_seconds, DEFAULT_GRAPH_SPAN_SECONDS))
    max_face_hold_frames = 8
    capture = open_capture(args.source)
    face_cascade = load_face_cascade()
    samples: collections.deque[Sample] = collections.deque()
    last_bpm: float | None = None
    prev_chest_gray: np.ndarray | None = None
    prev_face_gray: np.ndarray | None = None
    recent_chest_motion: collections.deque[float] = collections.deque(maxlen=120)
    recent_face_motion: collections.deque[float] = collections.deque(maxlen=120)
    smoothed_face_rect = None
    smoothed_chest_roi = None
    face_miss_frames = 0

    if args.show:
        try:
            cv2.namedWindow("Respiration Rate", cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise RuntimeError(
                "Could not open an OpenCV window. On macOS this usually means the GUI backend is unavailable. "
                "Try installing the non-headless OpenCV build and run the script locally with a desktop session."
            ) from exc

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp = time.time()
            face_rect = detect_face_rect(frame, face_cascade, args.min_face_size)

            if face_rect is None and smoothed_face_rect is not None and face_miss_frames < max_face_hold_frames:
                face_rect = smoothed_face_rect
                face_miss_frames += 1
            elif face_rect is None:
                face_miss_frames = 0
            else:
                face_miss_frames = 0

            if face_rect is not None:
                smoothed_face_rect = smooth_rect(smoothed_face_rect, face_rect, smooth_alpha)
                chest_roi = chest_roi_from_face(smoothed_face_rect, frame.shape)
                face_signal_roi = face_signal_roi_from_face(smoothed_face_rect, frame.shape)
                smoothed_chest_roi = smooth_rect(smoothed_chest_roi, chest_roi, smooth_alpha)
                chest_roi = smoothed_chest_roi

                prev_chest_gray, chest_motion = extract_signal(frame, chest_roi, prev_chest_gray)
                prev_face_gray, face_motion = extract_signal(frame, face_signal_roi, prev_face_gray)

                chest_norm = normalize_motion(chest_motion, recent_chest_motion)
                face_norm = normalize_motion(face_motion, recent_face_motion)
                fused_signal = fuse_signals(chest_norm, face_norm)
                if fused_signal is not None:
                    samples.append(Sample(timestamp=timestamp, value=fused_signal))

                if chest_motion is not None:
                    recent_chest_motion.append(chest_motion)
                if face_motion is not None:
                    recent_face_motion.append(face_motion)

                cutoff = timestamp - args.window_seconds
                while samples and samples[0].timestamp < cutoff:
                    samples.popleft()

                if len(samples) >= MIN_SIGNAL_SAMPLES:
                    last_bpm = estimate_bpm(list(samples))

                draw_rect(frame, smoothed_face_rect, (255, 120, 60), 2)
                draw_rect(frame, chest_roi, (0, 255, 0), 2)

                if args.show:
                    draw_overlay(frame, chest_roi, last_bpm, len(samples))
            elif args.show:
                prev_chest_gray = None
                prev_face_gray = None
                smoothed_face_rect = None
                smoothed_chest_roi = None
                cv2.putText(
                    frame,
                    "Face not detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            if args.show:
                graph = draw_signal_graph(samples, last_bpm, graph_span_seconds)
                display = compose_display(frame, graph)
                cv2.imshow("Respiration Rate", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        if last_bpm is None:
            print("Respiration rate could not be estimated from the available frames.")
        else:
            print(f"Estimated respiration rate: {last_bpm:.1f} bpm")
        return 0
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
