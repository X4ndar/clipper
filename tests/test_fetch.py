"""Chat replay download for Twitch and Kick against fake HTTP servers (no network, no yt-dlp)."""

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from clipper.fetch import (
    FetchError, is_url, kick_chat, kick_comment, platform_of, twitch_chat, twitch_comment, write_chat,
)
from clipper.signals import load_chat


def quiet(*_):
    pass


def test_links_and_platforms():
    assert is_url("https://www.twitch.tv/videos/123") and is_url(" HTTP://kick.com/x/videos/abc ")
    assert not is_url("C:\\videos\\stream.mp4") and not is_url("stream.mp4")
    assert platform_of("https://www.twitch.tv/videos/123") == "twitch"
    assert platform_of("https://kick.com/someone/videos/0193-abc") == "kick"
    assert platform_of("https://youtu.be/abc") == "youtube"
    assert platform_of("https://example.org/v/1") == "other"


def node(i, t, text="hello", emote=False, author=True):
    frags = [{"text": text, "emote": None}] + ([{"text": "LUL", "emote": {"emoteID": "425618"}}] if emote else [])
    return {"cursor": f"c{i}", "node": {"id": f"id{i}", "contentOffsetSeconds": t,
                                        "commenter": {"displayName": f"user{i}"} if author else None,
                                        "message": {"fragments": frags}}}


def twitch_server(comments, page=3, calls=None):
    """Serves VideoCommentsByOffsetOrCursor: by offset (first page) or by cursor (next pages)."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.headers["Client-ID"] and body["operationName"] == "VideoCommentsByOffsetOrCursor"
        v = body["variables"]
        if calls is not None:
            calls.append(v)
        if "cursor" in v:
            start = next(i for i, c in enumerate(comments) if c["cursor"] == v["cursor"]) + 1
        else:
            start = next((i for i, c in enumerate(comments) if c["node"]["contentOffsetSeconds"] >= v["contentOffsetSeconds"]),
                         len(comments))
        edges = comments[start:start + page]
        return httpx.Response(200, json={"data": {"video": {"comments": {
            "edges": edges, "pageInfo": {"hasNextPage": start + page < len(comments)}}}}})
    return httpx.Client(transport=httpx.MockTransport(handler), headers={"Client-ID": "x"})


def test_twitch_chat_pages_in_parallel_chunks_without_duplicates(tmp_path):
    comments = [node(i, i * 100.0, emote=(i == 5)) for i in range(40)] + [node(99, 500.0, author=False)]
    comments.sort(key=lambda c: c["node"]["contentOffsetSeconds"])
    calls = []
    got = twitch_chat("123", 4000.0, 4, quiet, client=twitch_server(comments, calls=calls))
    assert [c["_id"] for c in got] == [f"id{i}" for i in range(40)]  # all, once each, in order; no-author one dropped
    assert sum(1 for v in calls if "contentOffsetSeconds" in v) == 4  # one offset request per chunk
    out = tmp_path / "chat.json"
    write_chat(out, got, "twitch", "123")
    msgs = load_chat(out)  # the pipeline's own reader understands the file
    assert len(msgs) == 40 and msgs[5].t == 500.0 and msgs[5].emotes == 1 and msgs[5].text == "helloLUL"


def test_twitch_comment_skips_deleted_authors_and_null_text():
    assert twitch_comment({"id": "a", "contentOffsetSeconds": 3, "commenter": None, "message": {}}) is None
    c = twitch_comment({"id": "a", "contentOffsetSeconds": 3, "commenter": {"login": "bob"},
                        "message": {"fragments": [{"text": None, "emote": None}, {"text": "gg", "emote": None}]}})
    assert c["message"]["body"] == "gg" and c["commenter"]["display_name"] == "bob"


def test_twitch_deleted_video_is_a_clear_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": {"video": None}})))
    with pytest.raises(FetchError, match="does not exist"):
        twitch_chat("1", 100.0, 1, quiet, client=client)


START = datetime(2026, 11, 9, 21, 0, 0, tzinfo=timezone.utc)


def test_kick_chat_walks_five_second_windows(tmp_path):
    msgs = [{"id": f"m{i}", "created_at": (START + timedelta(seconds=i * 3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "content": "nice [emote:37226:KEKW]" if i == 2 else f"msg {i}", "sender": {"username": f"u{i}"}}
            for i in range(20)]
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/668/history"
        t0 = datetime.fromisoformat(request.url.params["start_time"].replace("Z", "+00:00"))
        asked.append(t0)
        win = [m for m in msgs if t0 <= datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")) < t0 + timedelta(seconds=5)]
        return httpx.Response(200, json={"data": {"messages": win, "cursor": "x"}})

    got = kick_chat("668", START.timestamp(), 60.0, 3, quiet, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert [c["_id"] for c in got] == [f"m{i}" for i in range(20)]
    assert got[2]["content_offset_seconds"] == 6.0 and got[2]["message"]["body"] == "nice KEKW"
    assert len(set(asked)) == 13  # 0, 5, ... 60
    out = tmp_path / "chat.json"
    write_chat(out, got, "kick", "abc")
    assert load_chat(out)[2].emotes == 1


def test_kick_refusal_is_reported_and_bad_rows_are_skipped():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    with pytest.raises(FetchError, match="403"):
        kick_chat("668", START.timestamp(), 30.0, 2, quiet, client=client)
    assert kick_comment({"id": "x", "content": "hi"}, START) is None
