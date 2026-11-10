"""Crop-path planning on synthetic face tracks (no MediaPipe, no ffmpeg)."""

import numpy as np
import pytest

from clipper.config import ReframeConfig
from clipper.reframe import (
    CropPlan, Face, crop_size, cuts_from_diffs, main_faces, plan_path, sendcmd_lines, smooth_shot, target_center,
)

W, H = 1280, 720
CFG = ReframeConfig()


def face(cx, w=0.08, score=0.9):
    return Face(cx=cx, cy=0.4, w=w, h=w * 16 / 9, score=score)


def track(fn, seconds=10.0, fps=CFG.sample_fps):
    times = [round(k / fps, 3) for k in range(int(seconds * fps))]
    return times, [fn(t) for t in times]


def test_crop_size_is_9x16_and_even():
    assert crop_size(1280, 720, 1080, 1920) == (404, 720)
    assert crop_size(1920, 1080, 1080, 1920) == (608, 1080)
    assert crop_size(1080, 1920, 1080, 1920) == (1080, 1920)  # already vertical: full frame


def test_cuts_are_spikes_not_motion_runs():
    d = np.full(60, 0.01)
    d[10] = 0.25  # a cut
    d[30:34] = 0.13  # fast camera motion: a run, not a spike
    d[50] = 0.09  # below threshold
    assert cuts_from_diffs(d) == [10]


def test_background_faces_are_dropped():
    faces = [face(0.3, w=0.10), face(0.8, w=0.03)]
    assert [f.cx for f in main_faces(faces, 0.5)] == [0.3]


def test_target_frames_a_group_that_fits_else_the_closest_big_face():
    crop_frac = 404 / W
    assert target_center([face(0.40), face(0.55)], crop_frac, None) == pytest.approx(0.475)
    far = [face(0.15, w=0.10), face(0.75, w=0.09)]  # cannot both fit in a 9:16 crop
    assert target_center(far, crop_frac, prev=0.2) == pytest.approx(0.15)  # near-tie: stay put
    assert target_center(far, crop_frac, prev=None) == pytest.approx(0.75)  # new shot: nearest the middle
    clear = [face(0.15, w=0.10), face(0.6, w=0.05)]
    assert target_center(clear, crop_frac, prev=None) == pytest.approx(0.15)  # clearly larger wins
    assert target_center([], crop_frac, None) is None


def test_subject_choice_does_not_carry_over_a_cut():
    # Shot 1: one person on the left. Shot 2: three equal faces; the middle one is the subject.
    def faces_at(t):
        return [face(0.15)] if t < 5 else [face(0.18), face(0.46), face(0.78)]
    times, faces = track(faces_at)
    plan = plan_path(times, faces, [5.0], W, H, 1080, 1920, CFG)
    assert plan.x_at(7.0) + plan.crop_w / 2 == pytest.approx(0.46 * W, abs=3)


def test_jittery_static_subject_gives_a_locked_off_crop():
    rng = np.random.default_rng(0)
    targets = list(0.3 + rng.normal(0, 0.01, 40))
    x = smooth_shot(targets, 4.0, 404 / W, CFG.dead_zone, CFG.smooth_s, 0.5)
    assert np.ptp(x) == 0 and x[0] == pytest.approx(0.3, abs=0.01)


def test_no_faces_falls_back_and_gaps_hold_the_last_position():
    assert set(smooth_shot([None] * 8, 4.0, 0.3, 0.15, 1.0, 0.5)) == {0.5}
    x = smooth_shot([0.3, 0.3, None, None, 0.3, 0.3], 4.0, 0.3, 0.15, 1.0, 0.5)
    assert np.allclose(x, 0.3)


def test_moving_subject_is_followed_smoothly_and_stays_in_frame():
    times, faces = track(lambda t: [face(0.2 + 0.06 * t)])  # walks from 0.2 to 0.8 of the width
    plan = plan_path(times, faces, [], W, H, 1080, 1920, CFG)
    xs = np.array(plan.xs)
    assert np.all(np.diff(xs) >= -1e-6)  # never pans backwards
    assert np.max(np.abs(np.diff(xs))) < 0.1 * W  # no jumps inside a shot
    assert plan.metrics["face_in_frame"] == 1.0
    assert plan.metrics["face_in_frame_center_crop"] < 0.5
    assert 0 <= xs.min() and xs.max() <= W - plan.crop_w


def test_crop_jumps_at_a_shot_cut():
    times, faces = track(lambda t: [face(0.2)] if t < 5 else [face(0.8)])
    plan = plan_path(times, faces, [5.0], W, H, 1080, 1920, CFG)
    before, after = plan.x_at(4.9), plan.x_at(5.0)
    assert before == pytest.approx(0.2 * W - plan.crop_w / 2, abs=2)
    assert after == pytest.approx(0.8 * W - plan.crop_w / 2, abs=2)
    assert plan.metrics["shots"] == 2 and plan.metrics["face_in_frame"] == 1.0


def test_crop_is_clamped_at_the_frame_edges():
    times, faces = track(lambda t: [face(0.02)])
    plan = plan_path(times, faces, [], W, H, 1080, 1920, CFG)
    assert min(plan.xs) == 0


def test_sendcmd_lines_only_on_change_and_even_pixels():
    plan = CropPlan(W, H, 404, 720, 0, "face", times=[0.0, 1.0, 2.0], xs=[100.0, 100.0, 301.0], shot_starts=[0.0])
    lines = sendcmd_lines(plan, 2.0, 10)
    xs = [int(line.split()[-1].rstrip(";")) for line in lines]
    assert lines[0].startswith("0.0000 crop@rf x 100;")
    assert all(x % 2 == 0 for x in xs) and len(xs) == len(set(xs))
    assert xs[-1] == 300


def test_joined_plans_cut_at_the_join_and_weight_metrics():
    from clipper.reframe import concat_plans

    t1, f1 = track(lambda t: [face(0.2)], seconds=4)
    t2, f2 = track(lambda t: [face(0.8)], seconds=6)
    p1 = plan_path(t1, f1, [], W, H, 1080, 1920, CFG)
    p2 = plan_path(t2, f2, [], W, H, 1080, 1920, CFG)
    j = concat_plans([p1, p2], [4.0, 6.0])
    assert 4.0 in j.shot_starts and j.metrics["shots"] == 2 and j.metrics["samples"] == len(t1) + len(t2)
    assert j.x_at(3.9) == pytest.approx(p1.x_at(3.9)) and j.x_at(4.0) == pytest.approx(p2.x_at(0.0))
    assert j.metrics["face_in_frame"] == 1.0
