# constants_optimized.py
"""High-performance constants/helpers for Othello.

Drop-in replacement for your constants.py with the same public API:
- EMPTY, BLACK, WHITE
- CORNERS, X_SQUARES, C_SQUARES, EDGES
- adjust_position_weight(pos, value, stages=('early','mid','late'))
- opponent(color)

Additions (pure helpers, optional):
- X_SQUARES_SET, C_SQUARES_SET, CORNERS_SET, EDGES_SET (frozenset for fast membership)
- xy_to_idx / idx_to_xy (bit ops)
- get_phase_weights(empty_count) -> one of EARLY/MID/LATE lists
"""
from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple

# ---------------------- Colors ----------------------
EMPTY, BLACK, WHITE = 0, 1, 2  # keep numeric and contiguous

# ---------------------- Positional weights ----------------------
# Keep mutable 2D lists for compatibility; use locals inside functions for speed.
EARLY_WEIGHTS: List[List[int]] = [
    [150, -40, 20,  5,  5, 20, -40, 150],
    [-40, -80, -5, -5, -5, -5, -80, -40],
    [20,   -5, 15,  3,  3, 15,  -5,  20],
    [5,    -5,  3,  0,  0,  3,  -5,   5],
    [5,    -5,  3,  0,  0,  3,  -5,   5],
    [20,   -5, 15,  3,  3, 15,  -5,  20],
    [-40, -80, -5, -5, -5, -5, -80, -40],
    [150, -40, 20,  5,  5, 20, -40, 150],
]

MID_WEIGHTS: List[List[int]] = [
    [120, -30, 25, 10, 10, 25, -30, 120],
    [-30, -60,  0,  0,  0,  0, -60, -30],
    [25,    0, 20, 10, 10, 20,   0,  25],
    [10,    0, 10,  5,  5, 10,   0,  10],
    [10,    0, 10,  5,  5, 10,   0,  10],
    [25,    0, 20, 10, 10, 20,   0,  25],
    [-30, -60,  0,  0,  0,  0, -60, -30],
    [120, -30, 25, 10, 10, 25, -30, 120],
]

LATE_WEIGHTS: List[List[int]] = [
    [200, 50,  50,  30,  30, 50,  50, 200],
    [50,  10,  20,  15,  15, 20,  10,  50],
    [50,  20,  30,  25,  25, 30,  20,  50],
    [30,  15,  25,  10,  10, 25,  15,  30],
    [30,  15,  25,  10,  10, 25,  15,  30],
    [50,  20,  30,  25,  25, 30,  20,  50],
    [50,  10,  20,  15,  15, 20,  10,  50],
    [200, 50,  50,  30,  30, 50,  50, 200],
]

# ---------------------- Key squares ----------------------
X_SQUARES: List[Tuple[int, int]] = [(1, 1), (1, 6), (6, 1), (6, 6)]
C_SQUARES: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, 6), (6, 0), (7, 1), (6, 7), (7, 6), (1, 7)]
CORNERS: List[Tuple[int, int]] = [(0, 0), (0, 7), (7, 0), (7, 7)]
EDGES: List[Tuple[int, int]] = (
    [(i, 0) for i in range(8)]
    + [(i, 7) for i in range(8)]
    + [(0, i) for i in range(8)]
    + [(7, i) for i in range(8)]
)

# Fast membership (use frozenset to avoid accidental mutation)
X_SQUARES_SET = frozenset(X_SQUARES)
C_SQUARES_SET = frozenset(C_SQUARES)
CORNERS_SET = frozenset(CORNERS)
EDGES_SET = frozenset(EDGES)

# ---------------------- Utilities ----------------------

def xy_to_idx(x: int, y: int) -> int:
    """(x,y)-> 0..63 (why: faster than tuple-based dict keys)."""
    return (x << 3) | y


def idx_to_xy(idx: int) -> Tuple[int, int]:
    return (idx >> 3), (idx & 7)


# Precompute 4-way symmetry per square; deduplicated for axes/center
_sym4 = {
    (x, y): tuple({(x, y), (7 - x, y), (x, 7 - y), (7 - x, 7 - y)})
    for x in range(8)
    for y in range(8)
}


def _apply_sym_weight(mat: List[List[int]], x: int, y: int, v: int) -> None:
    # Tight loop with locals to minimize global lookups
    m = mat
    for sx, sy in _sym4[(x, y)]:
        m[sx][sy] = v


def adjust_position_weight(pos: Tuple[int, int], value: int, stages: Sequence[str] = ("early", "mid", "late")) -> None:
    """좌표의 가중치를 대칭으로 수정. 중복 좌표는 자동 제거.
    - stages: ('early','mid','late') 중 선택 적용
    """
    x, y = pos
    if 'early' in stages:
        _apply_sym_weight(EARLY_WEIGHTS, x, y, value)
    if 'mid' in stages:
        _apply_sym_weight(MID_WEIGHTS, x, y, value)
    if 'late' in stages:
        _apply_sym_weight(LATE_WEIGHTS, x, y, value)


# Phase helper (avoids repeated branching at call sites)

def get_phase_weights(empty_count: int) -> List[List[int]]:
    """빈칸 수로 단계별 가중치 테이블 선택."""
    if empty_count > 50:
        return EARLY_WEIGHTS
    if empty_count > 20:
        return MID_WEIGHTS
    return LATE_WEIGHTS


# ---------------------- Opponent (hot path) ----------------------

def opponent(color: int) -> int:
    """상대 색 반환. color는 1(BLACK) 또는 2(WHITE)라고 가정.
    비트 XOR(3) 사용: 1^3=2, 2^3=1. (왜: dict/if보다 빠름)
    """
    return color ^ 3


__all__ = [
    'EMPTY', 'BLACK', 'WHITE',
    'EARLY_WEIGHTS', 'MID_WEIGHTS', 'LATE_WEIGHTS',
    'X_SQUARES', 'C_SQUARES', 'CORNERS', 'EDGES',
    'X_SQUARES_SET', 'C_SQUARES_SET', 'CORNERS_SET', 'EDGES_SET',
    'xy_to_idx', 'idx_to_xy', 'adjust_position_weight', 'get_phase_weights', 'opponent',
]
