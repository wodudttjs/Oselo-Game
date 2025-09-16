"""
UltraAdvancedAI (clean implementation)

This file replaces a previously corrupted version. It provides a robust,
time-limited alpha-beta Othello AI with a corner-first policy and sane
evaluation, compatible with the GUI via `UltraAdvancedAI(color, difficulty, time_limit)`.

Key behaviors:
- Corner-first: if a corner is available, take it.
- Iterative deepening alpha-beta within the time limit.
- Evaluation combines mobility, corners, disc differential, and edge bias.
- Does not mutate the incoming Board instance; operates on internal grids.
"""

from __future__ import annotations

import time
import logging
from typing import List, Tuple, Optional

from constants import BLACK, WHITE, EMPTY, opponent, CORNERS, X_SQUARES, C_SQUARES


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')


DIRS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= x < 8 and 0 <= y < 8


def _copy_grid(grid: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in grid]


def _valid_moves_on_grid(grid: List[List[int]], color: int) -> List[Tuple[int, int]]:
    opp = opponent(color)
    moves: List[Tuple[int, int]] = []
    for x in range(8):
        for y in range(8):
            if grid[x][y] != EMPTY:
                continue
            legal = False
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if not _in_bounds(nx, ny) or grid[nx][ny] != opp:
                    continue
                while True:
                    nx += dx
                    ny += dy
                    if not _in_bounds(nx, ny):
                        break
                    v = grid[nx][ny]
                    if v == opp:
                        continue
                    if v == color:
                        legal = True
                    break
                if legal:
                    moves.append((x, y))
                    break
    return moves


def _apply_move_on_grid(grid: List[List[int]], x: int, y: int, color: int) -> List[List[int]]:
    opp = opponent(color)
    ng = _copy_grid(grid)
    ng[x][y] = color
    for dx, dy in DIRS:
        nx, ny = x + dx, y + dy
        captured: List[Tuple[int, int]] = []
        while _in_bounds(nx, ny) and ng[nx][ny] == opp:
            captured.append((nx, ny))
            nx += dx
            ny += dy
        if _in_bounds(nx, ny) and ng[nx][ny] == color and captured:
            for fx, fy in captured:
                ng[fx][fy] = color
    return ng


def _count_discs(grid: List[List[int]]) -> Tuple[int, int]:
    b = sum(row.count(BLACK) for row in grid)
    w = sum(row.count(WHITE) for row in grid)
    return b, w


class UltraAdvancedAI:
    """Corner-first, time-limited alpha-beta AI compatible with GUI."""

    def __init__(self, color: int, difficulty: str = 'hard', time_limit: float = 10.0) -> None:
        self.color = color
        self.difficulty = difficulty
        self.time_limit = float(time_limit)

        # Depth by difficulty (kept modest for responsiveness)
        if difficulty == 'easy':
            self.max_depth = 3
        elif difficulty == 'medium':
            self.max_depth = 5
        else:
            self.max_depth = 7

        # Search bookkeeping
        self._nodes = 0

    # -------------------- Public API --------------------
    def get_move(self, board) -> Optional[Tuple[int, int]]:
        # Corner-first policy
        grid = board.board
        legal = _valid_moves_on_grid(grid, self.color)
        if not legal:
            return None
        for m in legal:
            if m in CORNERS:
                return m

        # Iterative deepening with time limit
        start = time.time()
        best: Optional[Tuple[int, int]] = None
        last_score = 0
        for depth in range(1, self.max_depth + 1):
            self._nodes = 0
            remaining = self.time_limit - (time.time() - start)
            if remaining <= 0:
                break
            score, move = self._search(grid, depth, self.color, -10**9, 10**9, start)
            if move is not None:
                best = move
                last_score = score
            if time.time() - start > self.time_limit * 0.9:
                break
        logging.info(f"AI depth reached, nodes={self._nodes}, score={last_score}, move={best}")
        return best if best is not None else (legal[0] if legal else None)

    # -------------------- Search --------------------
    def _search(
        self,
        grid: List[List[int]],
        depth: int,
        side: int,
        alpha: int,
        beta: int,
        start_time: float,
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        self._nodes += 1
        # Time guard
        if self._nodes % 2048 == 0 and (time.time() - start_time) > self.time_limit * 0.98:
            return self._evaluate(grid), None

        moves = _valid_moves_on_grid(grid, side)

        if depth == 0 or not moves:
            # Pass or terminal
            if not moves:
                opp = opponent(side)
                opp_moves = _valid_moves_on_grid(grid, opp)
                if not opp_moves:
                    # Game over: exact disc differential scaled heavily
                    b, w = _count_discs(grid)
                    diff = (b - w) if self.color == BLACK else (w - b)
                    return diff * 10000, None
                # Pass without reducing depth
                score, _ = self._search(grid, depth, opp, -beta, -alpha, start_time)
                return -score, None
            return self._evaluate(grid), None

        # Move ordering: corners first, avoid X/C early, prefer edges, prefer higher flips
        def move_key(m: Tuple[int, int]) -> Tuple[int, int, int, int]:
            x, y = m
            # compute flips for ordering only
            flips_est = self._flip_count_estimate(grid, x, y, side)
            return (
                2 if m in CORNERS else 1 if (x == 0 or y == 0 or x == 7 or y == 7) else 0,
                -1 if m in X_SQUARES or m in C_SQUARES else 0,
                flips_est,
                -(x * 8 + y),  # slight index preference for stability
            )

        ordered = sorted(moves, key=move_key, reverse=True)

        best_move: Optional[Tuple[int, int]] = None
        best_score = -10**9
        opp = opponent(side)
        for i, (x, y) in enumerate(ordered):
            child = _apply_move_on_grid(grid, x, y, side)
            if i == 0:
                score, _ = self._search(child, depth - 1, opp, -beta, -alpha, start_time)
                score = -score
            else:
                # PVS window
                score, _ = self._search(child, depth - 1, opp, -alpha - 1, -alpha, start_time)
                score = -score
                if alpha < score < beta:
                    score, _ = self._search(child, depth - 1, opp, -beta, -score, start_time)
                    score = -score

            if score > best_score:
                best_score = score
                best_move = (x, y)
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best_score, best_move

    # -------------------- Evaluation --------------------
    def _evaluate(self, grid: List[List[int]]) -> int:
        # Disc differential
        b, w = _count_discs(grid)
        disc = (b - w) if self.color == BLACK else (w - b)

        # Mobility
        my_moves = len(_valid_moves_on_grid(grid, self.color))
        op_moves = len(_valid_moves_on_grid(grid, opponent(self.color)))
        mobility = 0
        tot = my_moves + op_moves
        if tot:
            mobility = (my_moves - op_moves) * 100 // tot

        # Corners
        corner_score = 0
        for cx, cy in CORNERS:
            v = grid[cx][cy]
            if v == self.color:
                corner_score += 25
            elif v != EMPTY:
                corner_score -= 25

        # Edge presence (light bias)
        edge = 0
        for i in range(8):
            for j in (0, 7):
                v1 = grid[i][j]
                if v1 == self.color:
                    edge += 1
                elif v1 != EMPTY:
                    edge -= 1
                v2 = grid[j][i]
                if v2 == self.color:
                    edge += 1
                elif v2 != EMPTY:
                    edge -= 1

        # Combine
        score = disc * 2 + mobility * 3 + corner_score * 8 + edge
        return int(score)

    def _flip_count_estimate(self, grid: List[List[int]], x: int, y: int, color: int) -> int:
        opp = opponent(color)
        cnt = 0
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            run = 0
            while _in_bounds(nx, ny) and grid[nx][ny] == opp:
                run += 1
                nx += dx
                ny += dy
            if run and _in_bounds(nx, ny) and grid[nx][ny] == color:
                cnt += run
        return cnt

