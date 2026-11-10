"""Cut, reframe (center crop in M1) and burn in captions with ffmpeg."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import bundled_fonts_dir
from .media import run


FONT_EXTS = {".ttf", ".otf", ".ttc", ".woff", ".woff2"}


def fonts_dirs(ccfg, home: Path) -> list[Path]:
    """Font folders in priority order: configured, $CLIPPER_HOME/fonts, bundled."""
    dirs = []
    if ccfg.fonts_dir:
        dirs.append(Path(ccfg.fonts_dir).expanduser())
    dirs.append(home / "fonts")
    dirs.append(bundled_fonts_dir())
    return [d for d in dirs if d.is_dir()]


def prepare_fonts_dir(ccfg, home: Path, dest: Path) -> Path:
    """libass takes a single fontsdir, so link every font we know about into one folder."""
    dest.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for d in fonts_dirs(ccfg, home):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in FONT_EXTS and f.name not in seen:
                seen.add(f.name)
                link = dest / f.name
                if link.is_symlink() or link.exists():
                    link.unlink()
                try:
                    link.symlink_to(f.resolve())
                except OSError:
                    import shutil

                    shutil.copy2(f, link)
    return dest


def _filter_path(p: Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument."""
    s = str(p.resolve())
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,")


@dataclass
class DynamicCrop:
    """A crop window whose x moves over time, driven by an ffmpeg sendcmd file (see reframe.py)."""

    w: int
    h: int
    x0: int
    y: int
    cmd_path: Path


@dataclass
class StackLayout:
    """Streamer camera on top, gameplay below. Boxes are [x, y, w, h] in source pixels."""

    cam: list[int]
    game: list[int]
    top_h: int
    bot_h: int


def stack_filter(s: StackLayout, width: int) -> str:
    cx, cy, cw, ch = s.cam
    gx, gy, gw, gh = s.game
    return (
        f"split[fc_a][fc_b];[fc_a]crop={cw}:{ch}:{cx}:{cy},scale={width}:{s.top_h}:flags=lanczos[fc_t];"
        f"[fc_b]crop={gw}:{gh}:{gx}:{gy},scale={width}:{s.bot_h}:flags=lanczos[fc_g];[fc_t][fc_g]vstack,setsar=1"
    )


def crop_filter(width: int, height: int) -> str:
    # Center crop to the target aspect, then scale. Works for landscape, square and portrait input.
    ar = f"{width}/{height}"
    return (
        f"crop='min(iw,ih*{ar})':'min(ih,iw/({ar}))',"
        f"scale={width}:{height}:flags=lanczos,setsar=1"
    )


def dynamic_crop_filter(c: DynamicCrop, width: int, height: int) -> str:
    return (
        f"sendcmd=f='{_filter_path(c.cmd_path)}',crop@rf=w={c.w}:h={c.h}:x={c.x0}:y={c.y},"
        f"scale={width}:{height}:flags=lanczos,setsar=1"
    )


def video_filter(rcfg, ass_path: Optional[Path], fonts_dir: Optional[Path],
                 crop=None) -> str:
    """`crop` is None (center crop), a DynamicCrop (moving crop) or a StackLayout."""
    if isinstance(crop, StackLayout):
        vf = stack_filter(crop, rcfg.width)
    elif crop is None:
        vf = crop_filter(rcfg.width, rcfg.height)
    else:
        vf = dynamic_crop_filter(crop, rcfg.width, rcfg.height)
    if ass_path is not None:
        vf += f",ass='{_filter_path(ass_path)}'"
        if fonts_dir is not None:
            vf += f":fontsdir='{_filter_path(fonts_dir)}'"
    return vf


def _afade(dur: float, fade: float) -> str:
    return f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={max(0.0, dur - fade):.3f}:d={fade:.3f}"


def render_clip(src: Path, out: Path, start: float, end: float, rcfg, fade_ms: float,
                ass_path: Optional[Path], fonts_dir: Optional[Path], crop: Optional[DynamicCrop] = None,
                segments: Optional[list[tuple[float, float]]] = None) -> None:
    """Render one clip. With `segments` (several (start, end) source spans), join them with jump cuts
    in the given order; each piece gets its own short audio fade so the joins do not click."""
    fade = max(0.0, fade_ms / 1000)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    encode = [
        "-c:v", rcfg.video_codec, "-preset", rcfg.preset, "-crf", str(rcfg.crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", rcfg.audio_bitrate, "-ar", "48000",
        "-threads", str(rcfg.threads), "-movflags", "+faststart", str(tmp),
    ]
    head = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    vf = video_filter(rcfg, ass_path, fonts_dir, crop)
    if segments and len(segments) > 1:
        inputs, fc = [], []
        for i, (a, b) in enumerate(segments):
            d = max(0.1, b - a)
            inputs += ["-ss", f"{a:.3f}", "-t", f"{d:.3f}", "-i", str(src)]
            fc.append(f"[{i}:v:0]setpts=PTS-STARTPTS[v{i}];[{i}:a:0]asetpts=PTS-STARTPTS,{_afade(d, fade)}[a{i}]")
        joined = "".join(f"[v{i}][a{i}]" for i in range(len(segments)))
        fc.append(f"{joined}concat=n={len(segments)}:v=1:a=1[vc][ac];[vc]{vf}[vout]")
        run(head + inputs + ["-filter_complex", ";".join(fc), "-map", "[vout]", "-map", "[ac]"] + encode)
    else:
        dur = max(0.1, end - start)
        run(head + ["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
                    "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf, "-af", _afade(dur, fade)] + encode)
    tmp.replace(out)


def render_frame(src: Path, out: Path, clip_start: float, at: float, rcfg, ass_path: Optional[Path],
                 fonts_dir: Optional[Path], crop: Optional[DynamicCrop] = None) -> None:
    """One PNG frame at `at` seconds into the clip, with captions, to tune a style in seconds.

    Seeks to just before the frame and lets the subtitle filter see the correct timestamp.
    """
    pre = min(at, 1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Restamp the decoded segment so it starts at `at - pre` (clip-relative time, as in the ASS),
    # then keep the first frame at or after `at`.
    vf = (
        f"setpts=PTS-STARTPTS+{at - pre:.3f}/TB,"
        + video_filter(rcfg, ass_path, fonts_dir, crop)
        + f",select=gte(t\\,{at:.3f})"
    )
    run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, clip_start + at - pre):.3f}", "-i", str(src),
        "-vf", vf, "-frames:v", "1", "-update", "1", str(out),
    ])
