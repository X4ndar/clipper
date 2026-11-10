"""Download a finished stream (VOD) from a link, plus its chat replay when the platform has one.

Video: yt-dlp (optional `fetch` extra), so any site it supports works; Twitch and Kick also get chat.
Chat is written as TwitchDownloader-style JSON, the format signals.load_chat() already reads:
  {"comments": [{"content_offset_seconds": 12.0, "commenter": {"display_name": "a"},
                 "message": {"body": "text", "fragments": [{"text": "LUL", "emoticon": {...}}]}}]}

Only use this on streams you own or have the rights to clip. Live streams are refused: wait for the VOD.

The chat endpoints are the ones the sites' own players call (not documented APIs), so a failure there is
reported as a note and the job goes on with audio peaks alone.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx

Progress = Callable[[str, str, str], None]

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".ts", ".flv", ".m4v")
TWITCH_GQL = "https://gql.twitch.tv/gql"
TWITCH_CHAT_CLIENT_ID = "kd1unb4b3q4t58fwlpcbzcbnm76a8fp"  # public web client id used for VOD comments
TWITCH_COMMENTS_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"
KICK_CHAT = "https://web.kick.com/api/v1/chat/{channel_id}/history"
KICK_STEP_S = 5  # each call returns the messages of a 5-second window
_EMOTE = re.compile(r"\[emote:\d+:[^\]]*\]")


class FetchError(RuntimeError):
    pass


@dataclass
class Fetched:
    video: Path
    chat: Optional[Path]
    title: Optional[str]
    platform: str
    id: str
    url: str
    notes: list[str] = field(default_factory=list)


def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", str(s).strip(), re.I))


def platform_of(url: str) -> str:
    host = re.sub(r"^https?://(www\.|m\.)?", "", url.strip().lower()).split("/")[0]
    if host.endswith("twitch.tv"):
        return "twitch"
    if host.endswith("kick.com"):
        return "kick"
    if host.endswith("youtube.com") or host == "youtu.be":
        return "youtube"
    return "other"


def fmt_bytes(n: Optional[float]) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return "?"


# ---------------- video ----------------


def _ydl(opts: dict):
    try:
        import yt_dlp
    except ImportError:
        raise FetchError("Downloading from a link needs yt-dlp. Install it with: "
                         "uv sync --frozen --extra fetch (add the other extras you use)") from None
    return yt_dlp.YoutubeDL(opts)


def _friendly(e: Exception) -> str:
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).replace("ERROR: ", "").strip()
    low = msg.lower()
    if "subscriber" in low or "sub-only" in low or "403" in low:
        return f"The site refused the download (subscriber-only or private video?). {msg}"
    if "unsupported url" in low:
        return ("This link is not a video page yt-dlp knows. For Kick, use the form "
                "kick.com/<channel>/videos/<id>; for Twitch, twitch.tv/videos/<id>.")
    return msg


def fetch_source(url: str, cfg, progress: Progress, want_chat: bool = True) -> Fetched:
    """Download `url` into $CLIPPER_HOME/downloads/<platform>-<id>/ (reused if already there)."""
    fc = cfg.fetch
    url = url.strip()
    plat = platform_of(url)
    root = Path(fc.dir).expanduser() if fc.dir else cfg.home_path / "downloads"
    base = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True, "retries": 5,
            "fragment_retries": 10, "logger": _Quiet()}
    try:
        with _ydl(base) as ydl:
            info = ydl.extract_info(url, download=False)
    except FetchError:
        raise
    except Exception as e:
        raise FetchError(_friendly(e)) from e
    if info.get("_type") == "playlist" or info.get("entries"):
        raise FetchError("This link is a list of videos. Paste the link of one finished stream.")
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise FetchError("This stream is still live. Wait until it has ended, then use the link of the saved "
                         "stream (VOD).")
    vid = re.sub(r"[^A-Za-z0-9_-]+", "", str(info.get("id") or "")) or "video"
    vid = vid[1:] if plat == "twitch" and vid.startswith("v") and vid[1:].isdigit() else vid
    out = root / f"{plat}-{vid}"
    out.mkdir(parents=True, exist_ok=True)
    title = info.get("title") or info.get("fulltitle")
    notes: list[str] = []

    video = _existing_video(out)
    if video is None:
        last = [0.0]

        def hook(d: dict) -> None:
            if d.get("status") != "downloading" or time.monotonic() - last[0] < 1.0:
                return
            last[0] = time.monotonic()
            done, total = d.get("downloaded_bytes"), d.get("total_bytes") or d.get("total_bytes_estimate")
            fi, fn = d.get("fragment_index"), d.get("fragment_count")
            share = (fi / fn) if fi and fn else ((done / total) if done and total else None)
            pct = f"{share:.0%} · " if share is not None else ""
            speed = f" · {fmt_bytes(d.get('speed'))}/s" if d.get("speed") else ""
            progress("fetch", "info", f"video {pct}{fmt_bytes(done)} of {fmt_bytes(total)}{speed}")

        h = int(fc.max_height)
        opts = {**base, "format": f"b[height<={h}]/bv*[height<={h}]+ba/b/bv*+ba",
                "outtmpl": str(out / "source.%(ext)s"), "merge_output_format": "mp4",
                "concurrent_fragment_downloads": int(fc.fragment_threads), "progress_hooks": [hook]}
        progress("fetch", "info", f"downloading “{title or url}” (up to {h}p)")
        try:
            with _ydl(opts) as ydl:
                ydl.extract_info(url, download=True)
        except FetchError:
            raise
        except Exception as e:
            if e.__class__.__name__ == "Cancelled":
                raise
            raise FetchError(_friendly(e)) from e
        video = _existing_video(out)
        if video is None:
            raise FetchError(f"The download finished but no video file was found in {out}")
    else:
        progress("fetch", "info", f"already downloaded: {video.name}")

    meta = {k: info.get(k) for k in ("id", "title", "uploader", "uploader_id", "channel", "channel_id",
                                     "timestamp", "duration", "webpage_url", "extractor_key")}
    (out / "info.json").write_text(json.dumps({"url": url, "platform": plat, **meta}, indent=2, ensure_ascii=False,
                                              default=str), encoding="utf-8")

    chat: Optional[Path] = out / "chat.json"
    if chat.is_file():
        pass
    elif not (want_chat and fc.chat):
        chat = None
    else:
        chat = None
        try:
            from .media import probe

            length = float(probe(video).duration or 0)
            comments: Optional[list[dict]] = None
            if plat == "twitch":
                comments = twitch_chat(vid, length, int(fc.chat_threads), progress)
            elif plat == "kick" and info.get("channel_id") and info.get("timestamp"):
                comments = kick_chat(str(info["channel_id"]), float(info["timestamp"]), length,
                                     int(fc.chat_threads), progress)
            elif plat == "kick":
                notes.append("Kick did not give the stream's start time, so the chat replay was skipped.")
            if comments is not None:
                if comments:
                    chat = out / "chat.json"
                    write_chat(chat, comments, plat, vid)
                    progress("fetch", "info", f"chat replay: {len(comments)} messages")
                else:
                    notes.append("The chat replay came back empty (old streams lose their chat); "
                                 "using audio peaks alone.")
        except Exception as e:  # chat is a bonus: never fail the job for it
            if e.__class__.__name__ == "Cancelled":
                raise
            notes.append(f"Could not download the chat replay ({e}); using audio peaks alone.")
    for n in notes:
        progress("fetch", "warn", n)
    return Fetched(video=video, chat=chat, title=title, platform=plat, id=vid, url=url, notes=notes)


class _Quiet:
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def _existing_video(folder: Path) -> Optional[Path]:
    files = [p for p in folder.glob("source.*") if p.suffix.lower() in VIDEO_EXTS and p.stat().st_size > 0]
    return max(files, key=lambda p: p.stat().st_size) if files else None


def write_chat(path: Path, comments: list[dict], platform: str, video_id: str) -> None:
    comments = sorted(comments, key=lambda c: c["content_offset_seconds"])
    path.write_text(json.dumps({"video": {"id": video_id, "platform": platform}, "comments": comments},
                               ensure_ascii=False), encoding="utf-8")


def _split(length_s: float, parts: int) -> list[tuple[float, float]]:
    parts = max(1, min(parts, int(length_s // 600) or 1))  # no point splitting under 10 minutes a part
    step = length_s / parts
    return [(i * step, (i + 1) * step) for i in range(parts)]


# ---------------- Twitch chat ----------------


def _twitch_body(video_id: str, offset: Optional[int] = None, cursor: Optional[str] = None) -> dict:
    variables: dict = {"videoID": video_id}
    if cursor:
        variables["cursor"] = cursor
    else:
        variables["contentOffsetSeconds"] = int(offset or 0)
    return {"operationName": "VideoCommentsByOffsetOrCursor", "variables": variables,
            "extensions": {"persistedQuery": {"version": 1, "sha256Hash": TWITCH_COMMENTS_HASH}}}


def twitch_comment(node: dict) -> Optional[dict]:
    """One GQL comment node -> TwitchDownloader-style comment, or None if it has no author."""
    who = node.get("commenter")
    if not who or node.get("contentOffsetSeconds") is None:
        return None
    frags = [{"text": f.get("text") or "", "emoticon": {"emoticon_id": f["emote"].get("emoteID")} if f.get("emote")
              else None} for f in ((node.get("message") or {}).get("fragments") or [])]
    return {"_id": node.get("id"), "content_offset_seconds": float(node["contentOffsetSeconds"]),
            "commenter": {"display_name": who.get("displayName") or who.get("login")},
            "message": {"body": "".join(f["text"] for f in frags), "fragments": frags}}


def _twitch_page(client: httpx.Client, body: dict) -> dict:
    """One page of comments. Twitch sometimes returns a null list under load: retry with a backoff."""
    last = "no response"
    for attempt in range(8):
        try:
            r = client.post(TWITCH_GQL, json=body)
            if r.status_code == 200:
                data = r.json()
                data = data[0] if isinstance(data, list) else data
                if data.get("errors"):
                    last = str(data["errors"][0].get("message"))
                else:
                    video = (data.get("data") or {}).get("video")
                    if video is None:
                        raise FetchError("Twitch says this video does not exist (deleted or expired).")
                    comments = video.get("comments")
                    if comments and comments.get("edges") is not None:
                        return comments
                    last = "empty comment list"
            else:
                last = f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            last = e.__class__.__name__
        time.sleep(0.3 * (attempt + 1))
    raise FetchError(f"Twitch chat request failed ({last})")


def twitch_chat(video_id: str, length_s: float, threads: int, progress: Progress,
                client: Optional[httpx.Client] = None) -> list[dict]:
    own = client is None
    client = client or httpx.Client(headers={"Client-ID": TWITCH_CHAT_CLIENT_ID}, timeout=30)
    chunks = _split(max(length_s, 1.0), threads)
    done = [0]

    def run(chunk: tuple[float, float], last_chunk: bool) -> list[dict]:
        a, b = chunk
        out: list[dict] = []
        body = _twitch_body(video_id, offset=int(a))
        while True:
            page = _twitch_page(client, body)
            edges = page.get("edges") or []
            past_end = False
            for e in edges:
                c = twitch_comment(e.get("node") or {})
                if c is None:
                    continue
                t = c["content_offset_seconds"]
                if t >= b and not last_chunk:
                    past_end = True
                elif t >= a:
                    out.append(c)
            done[0] += len(edges)
            if out and done[0] % 2000 < len(edges):
                progress("fetch", "info", f"chat replay: {done[0]} messages read")
            if past_end or not edges or not (page.get("pageInfo") or {}).get("hasNextPage"):
                return out
            body = _twitch_body(video_id, cursor=edges[-1].get("cursor"))

    try:
        progress("fetch", "info", "downloading the chat replay from Twitch")
        with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            parts = list(ex.map(lambda ic: run(ic[1], ic[0] == len(chunks) - 1), enumerate(chunks)))
    finally:
        if own:
            client.close()
    seen: dict[str, dict] = {}
    for part in parts:
        for c in part:
            seen.setdefault(c["_id"] or f"{c['content_offset_seconds']}{c['message']['body']}", c)
    return sorted(seen.values(), key=lambda c: c["content_offset_seconds"])


# ---------------- Kick chat ----------------


def kick_comment(m: dict, start: datetime) -> Optional[dict]:
    try:
        at = datetime.fromisoformat(str(m["created_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    raw = str(m.get("content") or "")
    emotes = len(_EMOTE.findall(raw))
    body = _EMOTE.sub(lambda x: x.group(0).split(":")[-1].rstrip("]") + " ", raw).strip()
    frags = [{"text": body, "emoticon": None}] + [{"text": "", "emoticon": {"emoticon_id": "kick"}}] * emotes
    return {"_id": m.get("id"), "content_offset_seconds": (at - start).total_seconds(),
            "commenter": {"display_name": (m.get("sender") or {}).get("username")},
            "message": {"body": body, "fragments": frags}}


def kick_chat(channel_id: str, start_ts: float, length_s: float, threads: int, progress: Progress,
              client: Optional[httpx.Client] = None) -> list[dict]:
    """Walk the VOD in 5-second windows (what the Kick player does) and collect the messages."""
    own = client is None
    client = client or httpx.Client(headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}, timeout=30)
    start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    url = KICK_CHAT.format(channel_id=channel_id)
    offsets = list(range(0, int(length_s) + KICK_STEP_S, KICK_STEP_S))
    done, failed = [0], [0]

    def window(off: int) -> list[dict]:
        ts = (start + timedelta(seconds=off)).strftime("%Y-%m-%dT%H:%M:%SZ")
        msgs: list = []
        for attempt in range(4):
            try:
                r = client.get(url, params={"start_time": ts})
                if r.status_code == 200:
                    msgs = ((r.json().get("data") or {}).get("messages")) or []
                    break
                if r.status_code in (401, 403):
                    raise FetchError(f"Kick refused the chat request (HTTP {r.status_code})")
            except httpx.HTTPError:
                pass
            time.sleep(0.5 * (attempt + 1))
        else:
            failed[0] += 1
        done[0] += 1
        if done[0] % 200 == 0:
            progress("fetch", "info", f"chat replay: {done[0] / len(offsets):.0%}")
        return [c for c in (kick_comment(m, start) for m in msgs) if c is not None]

    try:
        progress("fetch", "info", "downloading the chat replay from Kick")
        window(0)  # fail fast if Kick blocks us, before starting the threads
        with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
            parts = list(ex.map(window, offsets))
    finally:
        if own:
            client.close()
    if failed[0] > len(offsets) * 0.2:
        raise FetchError(f"{failed[0]} of {len(offsets)} chat requests failed")
    seen: dict[str, dict] = {}
    for part in parts:
        for c in part:
            if 0 <= c["content_offset_seconds"] <= length_s + 60:
                seen.setdefault(c["_id"] or f"{c['content_offset_seconds']}{c['message']['body']}", c)
    return sorted(seen.values(), key=lambda c: c["content_offset_seconds"])
