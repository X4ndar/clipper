"""M3 reframing: find faces on sampled frames, plan a smoothed 9:16 crop path, drive ffmpeg's crop.

Steps per clip:
  1. shot cuts from tiny grayscale frames at full frame rate (frame-accurate, cheap)
  2. faces on frames sampled at `reframe.sample_fps` (MediaPipe full-range BlazeFace, CPU)
  3. per shot: a target x for the crop center, then a hold-still / pan-smoothly camera path
  4. framing metrics (main face fully inside the crop) for this path and for a plain center crop

The plan is a JSON file next to the clip; render.py turns it into ffmpeg `sendcmd` commands.
Face detection sits behind a tiny interface so tests and other detectors can be swapped in.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Protocol

import numpy as np

CUT_W, CUT_H = 64, 36


class ReframeError(RuntimeError):
    pass


@dataclass
class Face:
    """Normalized to the source frame: 0..1 on both axes."""

    cx: float
    cy: float
    w: float
    h: float
    score: float
    mx: Optional[float] = None  # mouth keypoint, when the detector gives one
    my: Optional[float] = None
    mouth: float = 0.0  # mouth movement since the previous sample (0 = still); see mouth_activity()

    @property
    def left(self) -> float:
        return self.cx - self.w / 2

    @property
    def right(self) -> float:
        return self.cx + self.w / 2


class FaceDetector(Protocol):
    name: str

    def detect(self, rgb: np.ndarray) -> list[Face]: ...


@dataclass
class CropPlan:
    src_w: int
    src_h: int
    crop_w: int
    crop_h: int
    y: int
    mode: str  # "face" | "center"
    times: list[float] = field(default_factory=list)  # clip-relative seconds, one per sample
    xs: list[float] = field(default_factory=list)  # crop left edge in source pixels, per sample
    shot_starts: list[float] = field(default_factory=list)  # clip-relative cut times (first is 0)
    metrics: dict = field(default_factory=dict)

    def x_at(self, t: float) -> float:
        """Crop left edge at clip time t: linear between samples, a hard step at shot cuts."""
        if not self.times:
            return (self.src_w - self.crop_w) / 2
        shot_t0 = max((s for s in self.shot_starts if s <= t), default=0.0)
        nxt = [s for s in self.shot_starts if s > t]
        shot_t1 = nxt[0] if nxt else float("inf")
        idx = [i for i, ts in enumerate(self.times) if shot_t0 <= ts < shot_t1]
        if not idx:  # no sample inside this shot: hold the nearest one
            i = int(np.argmin([abs(ts - t) for ts in self.times]))
            return self.xs[i]
        ts = [self.times[i] for i in idx]
        xs = [self.xs[i] for i in idx]
        return float(np.interp(t, ts, xs))

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "CropPlan":
        return cls(**d)


# ---------------- geometry ----------------


def crop_size(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int]:
    """Largest crop with the output aspect ratio; even sizes for yuv420p."""
    cw = min(src_w, round(src_h * out_w / out_h))
    ch = min(src_h, round(src_w * out_h / out_w))
    return cw - cw % 2, ch - ch % 2


def center_plan(src_w: int, src_h: int, out_w: int, out_h: int) -> CropPlan:
    cw, ch = crop_size(src_w, src_h, out_w, out_h)
    return CropPlan(src_w, src_h, cw, ch, (src_h - ch) // 2, "center")


# ---------------- frame access ----------------


def _read_frames(cmd: list[str], w: int, h: int, channels: int) -> Iterator[np.ndarray]:
    size = w * h * channels
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            shape = (h, w, channels) if channels > 1 else (h, w)
            yield np.frombuffer(buf, np.uint8).reshape(shape)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise ReframeError(f"ffmpeg frame read failed: {err.strip()[-500:]}")


def detect_cuts(src: Path, start: float, dur: float, fps: float, threshold: float = 0.12,
                ratio: float = 2.5) -> list[float]:
    """Clip-relative times where a new shot starts (excluding 0).

    A cut is a single-frame spike in mean absolute difference of 64x36 grayscale frames. Camera
    motion gives runs of similar differences, so a spike must be `ratio` times both neighbors.
    """
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
           "-t", f"{dur:.3f}", "-an", "-vf", f"scale={CUT_W}:{CUT_H}:flags=area,format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    frames = [f.astype(np.float32) / 255 for f in _read_frames(cmd, CUT_W, CUT_H, 1)]
    if len(frames) < 3:
        return []
    d = np.abs(np.diff(np.stack(frames), axis=0)).mean(axis=(1, 2))
    return [round((i + 1) / fps, 3) for i in cuts_from_diffs(d, threshold, ratio)]


def cuts_from_diffs(d: np.ndarray, threshold: float = 0.12, ratio: float = 2.5) -> list[int]:
    """Indices i where frame i+1 starts a new shot, given d[i] = diff(frame i, frame i+1)."""
    out = []
    for i, v in enumerate(d):
        prev = d[i - 1] if i > 0 else 0.0
        nxt = d[i + 1] if i + 1 < len(d) else 0.0
        if v >= threshold and v >= ratio * max(prev, nxt, 1e-6):
            out.append(i)
    return out


def sample_faces(src: Path, start: float, dur: float, fps: float, detector: FaceDetector,
                 max_width: int, src_w: int, src_h: int) -> tuple[list[float], list[list[Face]]]:
    w = min(max_width, src_w)
    w -= w % 2
    h = round(src_h * w / src_w)
    h -= h % 2
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
           "-t", f"{dur:.3f}", "-an", "-vf", f"fps={fps},scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    times, faces = [], []
    prev: list[tuple[Face, np.ndarray, np.ndarray]] = []
    for k, frame in enumerate(_read_frames(cmd, w, h, 3)):
        times.append(round(k / fps, 3))
        found = detector.detect(frame)
        prev = mouth_activity(frame, found, prev)
        faces.append(found)
    return times, faces


def _patch(gray: np.ndarray, cx: float, cy: float, half_w: float, half_h: float) -> np.ndarray:
    H, W = gray.shape
    x0, x1 = int(max(0, (cx - half_w) * W)), int(min(W, (cx + half_w) * W))
    y0, y1 = int(max(0, (cy - half_h) * H)), int(min(H, (cy + half_h) * H))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((8, 8), np.float32)
    p = gray[y0:y1, x0:x1]
    ys = np.linspace(0, p.shape[0] - 1, 8).astype(int)
    xs = np.linspace(0, p.shape[1] - 1, 8).astype(int)
    return p[np.ix_(ys, xs)].astype(np.float32) / 255.0


def mouth_activity(frame: np.ndarray, faces: list[Face], prev: list) -> list:
    """Set Face.mouth: how much the mouth area changed since the same face in the previous sample,
    minus how much the eye area changed (which cancels head and camera movement). A cheap stand-in
    for an active-speaker model. Returns the state to pass in with the next frame."""
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame
    state = []
    for f in faces:
        mx = f.mx if f.mx is not None else f.cx
        my = f.my if f.my is not None else f.cy + 0.22 * f.h
        mouth = _patch(gray, mx, my, 0.22 * f.w, 0.14 * f.h)
        eyes = _patch(gray, f.cx, f.cy - 0.15 * f.h, 0.30 * f.w, 0.12 * f.h)
        best = None
        for pf, pm, pe in prev:
            d = abs(pf.cx - f.cx) + abs(pf.cy - f.cy)
            if d < 0.6 * f.w and 0.7 < pf.w / f.w < 1.4 and (best is None or d < best[0]):
                best = (d, pf, pm, pe)
        if best is not None:
            _, pf, pm, pe = best
            raw = max(0.0, float(np.abs(mouth - pm).mean()) - float(np.abs(eyes - pe).mean()))
            f.mouth = round(0.6 * raw + 0.4 * pf.mouth, 4)
        state.append((f, mouth, eyes))
    return state


# ---------------- path planning ----------------


def main_faces(faces: list[Face], min_rel_size: float) -> list[Face]:
    """Drop background faces: keep those at least `min_rel_size` of the largest face's width."""
    if not faces:
        return []
    big = max(f.w for f in faces)
    return [f for f in faces if f.w >= min_rel_size * big]


def target_center(faces: list[Face], crop_frac: float, prev: Optional[float],
                  follow_speaker: bool = False) -> Optional[float]:
    """Where the crop center should be (0..1 of source width) for one sample, or None if no face."""
    if not faces:
        return None
    left, right = min(f.left for f in faces), max(f.right for f in faces)
    if right - left <= 0.9 * crop_frac:
        return (left + right) / 2  # everyone fits: frame the group
    # Not everyone fits: the largest face. Near-ties go to the one we are already on, or at the start
    # of a shot to the one nearest the middle (where the camera operator put the subject).
    big = max(f.w for f in faces)
    if follow_speaker:
        # The face whose mouth is clearly moving more than anyone else's is the one talking.
        cand = sorted((f for f in faces if f.w >= 0.6 * big), key=lambda f: -f.mouth)
        if cand[0].mouth >= 0.02 and (len(cand) == 1 or cand[0].mouth >= 1.5 * cand[1].mouth):
            return cand[0].cx
    near = [f for f in faces if f.w >= 0.8 * big]
    anchor = prev if prev is not None else 0.5
    return min(near, key=lambda f: abs(f.cx - anchor)).cx


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < 3:
        return x
    r = k // 2
    pad = np.pad(x, r, mode="edge")
    return np.array([np.median(pad[i : i + k]) for i in range(len(x))])


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < 2:
        return x
    r = k // 2
    pad = np.pad(x, r, mode="edge")
    return np.convolve(pad, np.ones(2 * r + 1) / (2 * r + 1), mode="valid")


def smooth_shot(targets: list[Optional[float]], fps: float, crop_frac: float, dead_zone: float,
                smooth_s: float, fallback: float) -> np.ndarray:
    """One shot's crop centers (0..1). Holds still unless the subject leaves the dead zone."""
    known = [i for i, t in enumerate(targets) if t is not None]
    if not known:
        return np.full(len(targets), fallback)
    # Fill gaps by holding the nearest known value (no guessing between detections).
    filled = np.array([targets[min(known, key=lambda k: abs(k - i))] for i in range(len(targets))], dtype=float)
    filled = _median_filter(filled, 5)
    dz = dead_zone * crop_frac
    if filled.max() - filled.min() <= 2 * dz:
        return np.full(len(filled), float(np.median(filled)))  # locked-off camera
    cam = np.empty_like(filled)
    c = filled[0]
    for i, t in enumerate(filled):
        if t > c + dz:
            c = t - dz
        elif t < c - dz:
            c = t + dz
        cam[i] = c
    return _moving_average(cam, max(1, int(round(smooth_s * fps))))


def plan_path(times: list[float], faces: list[list[Face]], cuts: list[float], src_w: int, src_h: int,
              out_w: int, out_h: int, rcfg) -> CropPlan:
    plan = center_plan(src_w, src_h, out_w, out_h)
    plan.mode = "face"
    crop_frac = plan.crop_w / src_w
    half = crop_frac / 2
    shot_starts = [0.0] + [c for c in cuts if c > 0]
    plan.shot_starts = shot_starts
    kept = [main_faces(f, rcfg.min_face_rel_size) for f in faces]
    centers = np.full(len(times), 0.5)
    bounds = shot_starts + [float("inf")]
    for s in range(len(shot_starts)):
        idx = [i for i, t in enumerate(times) if bounds[s] <= t < bounds[s + 1]]
        if not idx:
            continue
        prev: Optional[float] = None  # a new shot is a new composition: no carry-over
        targets = []
        for i in idx:
            tc = target_center(kept[i], crop_frac, prev, getattr(rcfg, "follow_speaker", False))
            if tc is not None:
                prev = tc
            targets.append(tc)
        centers[idx] = smooth_shot(targets, rcfg.sample_fps, crop_frac, rcfg.dead_zone, rcfg.smooth_s, 0.5)
    centers = np.clip(centers, half, 1 - half)
    plan.times = [float(t) for t in times]
    plan.xs = [float(round(c * src_w - plan.crop_w / 2, 1)) for c in centers]
    plan.metrics = framing_metrics(plan, kept)
    return plan


def framing_metrics(plan: CropPlan, kept: list[list[Face]]) -> dict:
    """Over sampled frames with a face: how often at least one main face is fully inside the crop, and
    how often a main face is sliced by the crop edge. Same numbers for a plain center crop, to compare."""
    W, cw = plan.src_w, plan.crop_w
    center_x = (W - cw) / 2

    def judge(x: float, fs: list[Face]) -> tuple[bool, bool]:
        spans = [(f.left * W, f.right * W) for f in fs]
        full = any(x <= l and r <= x + cw for l, r in spans)
        sliced = any(l < x < r or l < x + cw < r for l, r in spans)
        return full, sliced

    with_face = full = cut = full_c = cut_c = 0
    for t, fs in zip(plan.times, kept):
        if not fs:
            continue
        with_face += 1
        a, b = judge(plan.x_at(t), fs)
        c, d = judge(center_x, fs)
        full, cut, full_c, cut_c = full + a, cut + b, full_c + c, cut_c + d
    n = len(plan.times)

    def share(k: int) -> Optional[float]:
        return round(k / with_face, 3) if with_face else None

    return {
        "samples": n,
        "shots": len(plan.shot_starts),
        "face_share": round(with_face / n, 3) if n else 0.0,
        "face_in_frame": share(full),
        "face_cut": share(cut),
        "face_in_frame_center_crop": share(full_c),
        "face_cut_center_crop": share(cut_c),
    }


def concat_plans(plans: list[CropPlan], durations: list[float]) -> CropPlan:
    """Join the plans of a jump-cut clip's pieces onto one timeline; each join is a shot cut."""
    if len(plans) == 1:
        return plans[0]
    b = plans[0]
    out = CropPlan(b.src_w, b.src_h, b.crop_w, b.crop_h, b.y, b.mode)
    off = 0.0
    n = shots = with_face = 0
    acc = {"face_in_frame": 0.0, "face_cut": 0.0, "face_in_frame_center_crop": 0.0, "face_cut_center_crop": 0.0}
    for p, d in zip(plans, durations):
        out.times += [round(t + off, 3) for t in p.times]
        out.xs += p.xs
        out.shot_starts += [round(s + off, 3) for s in p.shot_starts]
        m = p.metrics
        wf = round(m.get("face_share", 0) * m.get("samples", 0))
        n, shots, with_face = n + m.get("samples", 0), shots + m.get("shots", 0), with_face + wf
        for k in acc:
            acc[k] += (m.get(k) or 0.0) * wf
        off += d
    out.metrics = {"samples": n, "shots": shots, "face_share": round(with_face / n, 3) if n else 0.0,
                   **{k: (round(v / with_face, 3) if with_face else None) for k, v in acc.items()}}
    return out


# ---------------- facecam + gameplay ----------------


def detect_facecam(samples: list[list[Face]], min_share: float = 0.6, max_width: float = 0.22
                   ) -> Optional[tuple[float, float, float]]:
    """A streamer's camera overlay: a smallish face that sits in the same place in most samples, even
    across shot cuts. Returns its (cx, cy, face_width), normalized, or None."""
    samples = [s for s in samples if s is not None]
    faces = [f for s in samples for f in s if f.w <= max_width]
    if len(samples) < 8 or not faces:
        return None
    best, best_n = None, 0
    for f in faces[:: max(1, len(faces) // 200)]:
        n = sum(any(abs(g.cx - f.cx) < 0.03 and abs(g.cy - f.cy) < 0.04 and 0.7 < g.w / f.w < 1.4 for g in s)
                for s in samples)
        if n > best_n:
            best, best_n = f, n
    if best is None or best_n / len(samples) < min_share:
        return None
    close = [g for s in samples for g in s if abs(g.cx - best.cx) < 0.03 and abs(g.cy - best.cy) < 0.04]
    return (float(np.median([g.cx for g in close])), float(np.median([g.cy for g in close])),
            float(np.median([g.w for g in close])))


def stack_geometry(src_w: int, src_h: int, out_w: int, out_h: int, top_share: float,
                   cam: tuple[float, float, float, float]) -> dict:
    """Pixel boxes for the stacked layout: camera on top, gameplay below. `cam` is the camera region
    (x, y, w, h, normalized); it is grown to the top panel's aspect ratio and kept inside the frame."""
    top_h = int(out_h * top_share) // 2 * 2
    bot_h = out_h - top_h
    def fit(cx: float, cy: float, w: float, h: float, aspect: float) -> list[int]:
        w, h = max(w, h * aspect), max(h, w / aspect)
        if w > src_w:
            w, h = src_w, src_w / aspect
        if h > src_h:
            w, h = src_h * aspect, src_h
        x = min(max(cx - w / 2, 0), src_w - w)
        y = min(max(cy - h / 2, 0), src_h - h)
        return [int(x) // 2 * 2, int(y) // 2 * 2, int(w) // 2 * 2, int(h) // 2 * 2]
    x, y, w, h = cam
    cam_box = fit((x + w / 2) * src_w, (y + h / 2) * src_h, w * src_w, h * src_h, out_w / top_h)
    game_box = fit(src_w / 2, src_h / 2, 1, src_h, out_w / bot_h)
    return {"cam": cam_box, "game": game_box, "top_h": top_h, "bot_h": bot_h}


def facecam_region(face: tuple[float, float, float], src_w: int, src_h: int) -> tuple[float, float, float, float]:
    """A camera box around a detected facecam face: about three face-widths wide, head near the top third."""
    cx, cy, fw = face
    w = min(1.0, 3.2 * fw)
    h = min(1.0, w * src_w / src_h * 0.6)
    return (min(max(cx - w / 2, 0.0), 1.0 - w), min(max(cy - 0.42 * h, 0.0), 1.0 - h), w, h)


# ---------------- ffmpeg commands ----------------


def sendcmd_lines(plan: CropPlan, duration: float, fps: float, target: str = "crop@rf") -> list[str]:
    """One `crop x` command per output frame where x changes (even pixels for yuv420p)."""
    lines, last = [], None
    n = int(np.ceil(duration * fps)) + 1
    max_x = plan.src_w - plan.crop_w
    for k in range(n):
        t = k / fps
        x = int(round(min(max(plan.x_at(t), 0), max_x)))
        x -= x % 2
        if x != last:
            lines.append(f"{t:.4f} {target} x {x};")
            last = x
    return lines


def write_plan(plan: CropPlan, json_path: Path, cmd_path: Path, duration: float, fps: float) -> None:
    json_path.write_text(json.dumps(plan.to_json(), indent=1), encoding="utf-8")
    cmd_path.write_text("\n".join(sendcmd_lines(plan, duration, fps)) + "\n", encoding="utf-8")


# ---------------- MediaPipe backend ----------------


def ensure_model(rcfg, home: Path) -> Path:
    """The face model file: reframe.model_path if set, else downloaded once and checked by SHA-256."""
    if rcfg.model_path:
        p = Path(rcfg.model_path).expanduser()
        if not p.is_file():
            raise ReframeError(f"reframe.model_path {p} not found")
        return p
    dest = home / "models" / Path(rcfg.model_url).name
    if dest.is_file() and _sha256(dest) == rcfg.model_sha256:
        return dest
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        r = httpx.get(rcfg.model_url, timeout=60, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise ReframeError(f"could not download the face model from {rcfg.model_url}: {e}") from e
    tmp.write_bytes(r.content)
    if _sha256(tmp) != rcfg.model_sha256:
        tmp.unlink(missing_ok=True)
        raise ReframeError("downloaded face model failed its SHA-256 check; not using it")
    tmp.replace(dest)
    return dest


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class MediaPipeFaces:
    name = "mediapipe-blazeface-full-range"

    def __init__(self, model_path: Path, min_confidence: float):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except (ImportError, OSError) as e:  # OSError: missing system libs (libGL, PortAudio) on Linux
            raise ReframeError(
                f"MediaPipe is not available ({e}). Install it with: uv sync --frozen --extra reframe"
            ) from e
        self._mp = mp
        opts = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=str(model_path)),
                                          min_detection_confidence=min_confidence)
        self._det = vision.FaceDetector.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> list[Face]:
        h, w = rgb.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        out = []
        for d in self._det.detect(img).detections:
            b = d.bounding_box
            score = d.categories[0].score if d.categories else 0.0
            kp = getattr(d, "keypoints", None) or []
            mouth = kp[3] if len(kp) > 3 else None  # BlazeFace order: eyes, nose, mouth, ears
            out.append(Face(cx=(b.origin_x + b.width / 2) / w, cy=(b.origin_y + b.height / 2) / h,
                            w=b.width / w, h=b.height / h, score=float(score),
                            mx=None if mouth is None else float(mouth.x),
                            my=None if mouth is None else float(mouth.y)))
        return out


def get_detector(rcfg, home: Path) -> FaceDetector:
    return MediaPipeFaces(ensure_model(rcfg, home), rcfg.min_confidence)


# ---------------- one clip ----------------


def reframe_clip(src: Path, start: float, end: float, src_w: int, src_h: int, src_fps: float,
                 out_w: int, out_h: int, rcfg, detector: FaceDetector, faces_cache: Optional[Path] = None) -> CropPlan:
    dur = max(0.1, end - start)
    key = {"v": 2, "start": round(start, 3), "end": round(end, 3), "sample_fps": rcfg.sample_fps,
           "detector": detector.name, "min_confidence": rcfg.min_confidence, "max_width": rcfg.analysis_width,
           "cut_threshold": rcfg.cut_threshold}
    cached = None
    if faces_cache is not None and faces_cache.is_file():
        blob = json.loads(faces_cache.read_text(encoding="utf-8"))
        if blob.get("key") == key:
            cached = blob
    if cached:
        times, cuts = cached["times"], cached["cuts"]
        faces = [[Face(**f) for f in fs] for fs in cached["faces"]]
    else:
        cuts = detect_cuts(src, start, dur, src_fps or 30.0, rcfg.cut_threshold)
        times, faces = sample_faces(src, start, dur, rcfg.sample_fps, detector, rcfg.analysis_width, src_w, src_h)
        if faces_cache is not None:
            faces_cache.write_text(json.dumps({"key": key, "times": times, "cuts": cuts,
                                               "faces": [[asdict(f) for f in fs] for fs in faces]}),
                                   encoding="utf-8")
    return plan_path(times, faces, cuts, src_w, src_h, out_w, out_h, rcfg)
