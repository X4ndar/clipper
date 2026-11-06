"""Cut, reframe (center crop in M1) and burn in captions with ffmpeg."""

from __future__ import annotations

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


def crop_filter(width: int, height: int) -> str:
    # Center crop to the target aspect, then scale. Works for landscape, square and portrait input.
    ar = f"{width}/{height}"
    return (
        f"crop='min(iw,ih*{ar})':'min(ih,iw/({ar}))',"
        f"scale={width}:{height}:flags=lanczos,setsar=1"
    )


def video_filter(rcfg, ass_path: Optional[Path], fonts_dir: Optional[Path]) -> str:
    vf = crop_filter(rcfg.width, rcfg.height)
    if ass_path is not None:
        vf += f",ass='{_filter_path(ass_path)}'"
        if fonts_dir is not None:
            vf += f":fontsdir='{_filter_path(fonts_dir)}'"
    return vf


def render_clip(src: Path, out: Path, start: float, end: float, rcfg, fade_ms: float,
                ass_path: Optional[Path], fonts_dir: Optional[Path]) -> None:
    dur = max(0.1, end - start)
    fade = max(0.0, fade_ms / 1000)
    af = f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={max(0.0, dur - fade):.3f}:d={fade:.3f}"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", video_filter(rcfg, ass_path, fonts_dir),
        "-af", af,
        "-c:v", rcfg.video_codec, "-preset", rcfg.preset, "-crf", str(rcfg.crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", rcfg.audio_bitrate, "-ar", "48000",
        "-threads", str(rcfg.threads), "-movflags", "+faststart", str(tmp),
    ]
    run(cmd)
    tmp.replace(out)


def render_frame(src: Path, out: Path, clip_start: float, at: float, rcfg, ass_path: Optional[Path],
                 fonts_dir: Optional[Path]) -> None:
    """One PNG frame at `at` seconds into the clip, with captions, to tune a style in seconds.

    Seeks to just before the frame and lets the subtitle filter see the correct timestamp.
    """
    pre = min(at, 1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Restamp the decoded segment so it starts at `at - pre` (clip-relative time, as in the ASS),
    # then keep the first frame at or after `at`.
    vf = (
        f"setpts=PTS-STARTPTS+{at - pre:.3f}/TB,"
        + video_filter(rcfg, ass_path, fonts_dir)
        + f",select=gte(t\\,{at:.3f})"
    )
    run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, clip_start + at - pre):.3f}", "-i", str(src),
        "-vf", vf, "-frames:v", "1", "-update", "1", str(out),
    ])
