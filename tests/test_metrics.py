"""T6: 지표 정의를 손으로 계산한 예시로 확인"""
import numpy as np

from score_results import case_metrics, size_bin


def test_building_add_precision_and_overgen():
    orig = np.ones((64, 64), np.uint8)                     # 전부 배경
    edited = orig.copy(); edited[20:30, 20:30] = 2         # 목표 건물 100픽셀 (E)
    hole = np.zeros_like(orig, bool); hole[10:50, 10:50] = True   # M = 1600, C = 1500
    pred = orig.copy(); pred[20:30, 20:30] = 2             # 목표 100픽셀 모두 건물
    extra = np.zeros_like(hole); extra[10:20, 10:40] = True   # 목표 밖 300픽셀도 건물 (모두 C 안)
    pred[extra] = 2
    img = np.zeros((64, 64, 3), np.uint8)
    row, comps = case_metrics("building_add", orig, edited, hole, img, img, pred, orig.copy())
    assert row["add_recall"] == 1.0 and row["intent_changed"] == 1.0
    assert row["add_precision_M"] == 0.25 and row["add_iou_M"] == 0.25
    assert row["overgen_C"] == 0.2                          # 300 / 1500
    assert row["real_overgen_C"] == 0.0 and row["outside_max_err"] == 0
    assert len(comps) == 1 and comps[0]["size_bin"] == "small" and comps[0]["add_recall"] == 1.0


def test_building_remove_residual_and_keep():
    orig = np.ones((64, 64), np.uint8)
    orig[10:20, 10:20] = 2                                 # 지울 건물 (E, 100픽셀)
    orig[40:50, 40:50] = 2                                 # 남길 건물
    edited = orig.copy(); edited[10:20, 10:20] = 1
    hole = np.zeros_like(orig, bool); hole[5:25, 5:25] = True
    pred = orig.copy(); pred[10:15, 10:20] = 1             # 지운 자리 절반만 배경으로 보임
    img = np.zeros((64, 64, 3), np.uint8)
    row, comps = case_metrics("building_remove", orig, edited, hole, img, img, pred, orig.copy())
    assert row["remove_residual"] == 0.5 and row["fill_match"] == 0.5 and row["gone_changed"] == 0.5
    assert row["keep_buildings"] == 1.0
    assert comps[0]["remove_residual"] == 0.5


def test_outside_change_detected():
    orig = np.ones((32, 32), np.uint8); edited = orig.copy(); edited[5:10, 5:10] = 2
    hole = np.zeros_like(orig, bool); hole[4:11, 4:11] = True
    img = np.zeros((32, 32, 3), np.uint8); fin = img.copy(); fin[0, 0] = 7
    row, _ = case_metrics("building_add", orig, edited, hole, fin, img, None, None)
    assert row["outside_max_err"] == 7


def test_size_bins():
    assert size_bin(399) == "small" and size_bin(400) == "mid" and size_bin(2499) == "mid" and size_bin(2500) == "large"
