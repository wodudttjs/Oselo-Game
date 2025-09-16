
from constants import EMPTY, BLACK, WHITE, opponent, CORNERS
from zobrist import ZOBRIST_TABLE
import numpy as np
import copy
=======
# board_optimized.py
# Drop-in, faster Board with zero-deepcopy apply, cached moves, and micro-opts.

from __future__ import annotations
from typing import List, Tuple, Optional

from constants import (
    EMPTY, BLACK, WHITE, opponent,
    CORNERS,
)

# Try to import set variants if available (from constants_optimized)
try:
    from constants import CORNERS_SET  # type: ignore
except Exception:  # pragma: no cover
    CORNERS_SET = frozenset(CORNERS)

# 8 directions (tuples to avoid realloc)
DIRS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)



class Board:
    __slots__ = ("size", "board", "move_history", "_valid_moves_cache", "_b_count", "_w_count", "_empty_count")

    def __init__(self) -> None:
        self.size = 8

        self.board = [[EMPTY] * self.size for _ in range(self.size)]
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        self.move_history = []
        # Initialize Zobrist hash for the starting position
        self._zobrist_hash = self._compute_zobrist()

    def _compute_zobrist(self):
        h = np.uint64(0)
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece != EMPTY:
                    h ^= ZOBRIST_TABLE[i][j][piece]
        return h

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_valid_moves(self, color):
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] != EMPTY:

        b = [[EMPTY] * 8 for _ in range(8)]
        # initial position
        b[3][3] = WHITE
        b[3][4] = BLACK
        b[4][3] = BLACK
        b[4][4] = WHITE
        self.board: List[List[int]] = b
        self.move_history: List[Tuple[int, int, int, List[Tuple[int, int]]]] = []
        self._valid_moves_cache = {BLACK: None, WHITE: None}
        # maintain counts incrementally
        self._b_count = 2
        self._w_count = 2
        self._empty_count = 64 - 4

    # ----------------------------- helpers -----------------------------

    @staticmethod
    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < 8 and 0 <= y < 8

    # ----------------------------- moves -------------------------------

    def _collect_candidates(self, color: int) -> List[Tuple[int, int]]:
        """Collect empty squares adjacent to opponent discs (tight superset of legal moves)."""
        opp = opponent(color)
        b = self.board
        cand_set = set()
        for i in range(8):
            row = b[i]
            for j in range(8):
                if row[j] == opp:
                    for dx, dy in DIRS:
                        nx, ny = i + dx, j + dy
                        if 0 <= nx < 8 and 0 <= ny < 8 and b[nx][ny] == EMPTY:
                            cand_set.add((nx, ny))
        return list(cand_set)

    def get_valid_moves(self, color: int):
        cached = self._valid_moves_cache.get(color)
        if cached is not None:
            return cached

        moves: List[Tuple[int, int]] = []
        b = self.board
        opp = opponent(color)

        # Candidates: only empties next to opponent discs
        for x, y in self._collect_candidates(color):
            # check in 8 dirs
            valid = False
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < 8 and 0 <= ny < 8) or b[nx][ny] != opp:

                    continue
                # found opponent run
                nx += dx
                ny += dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    v = b[nx][ny]
                    if v == opp:
                        nx += dx
                        ny += dy
                        continue
                    if v == color:
                        valid = True
                    break
                if valid:
                    moves.append((x, y))
                    break

        self._valid_moves_cache[color] = moves
        return moves

    def is_valid_move(self, x: int, y: int, color: int) -> bool:
        b = self.board
        if b[x][y] != EMPTY:
            return False
        opp = opponent(color)
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < 8 and 0 <= ny < 8) or b[nx][ny] != opp:
                continue
            nx += dx
            ny += dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                v = b[nx][ny]
                if v == opp:
                    nx += dx
                    ny += dy
                    continue
                return v == color
        return False


    def apply_move(self, x, y, color):
        new_board = copy.deepcopy(self)
        new_board.board[x][y] = color
        flipped = []

        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in directions:

    def apply_move(self, x: int, y: int, color: int) -> "Board":
        """Return a new Board with the move applied (no deepcopy)."""
        # Create a new instance without running __init__
        nb: Board = Board.__new__(Board)  # type: ignore
        nb.size = 8
        # shallow-copy rows (8x8)
        src = self.board
        b = [src[i][:] for i in range(8)]
        nb.board = b
        # copy counters and history ref replaced later
        nb._b_count = self._b_count
        nb._w_count = self._w_count
        nb._empty_count = self._empty_count

        flipped: List[Tuple[int, int]] = []
        opp = opponent(color)

        # place stone
        b[x][y] = color
        nb._empty_count -= 1
        if color == BLACK:
            nb._b_count += 1
        else:
            nb._w_count += 1

        # flip in 8 dirs
        for dx, dy in DIRS:

            nx, ny = x + dx, y + dy
            to_flip: List[Tuple[int, int]] = []
            # run opponent chain
            while 0 <= nx < 8 and 0 <= ny < 8 and b[nx][ny] == opp:
                to_flip.append((nx, ny))
                nx += dx
                ny += dy
            # bounded by own color? then flip
            if to_flip and 0 <= nx < 8 and 0 <= ny < 8 and b[nx][ny] == color:
                for fx, fy in to_flip:
                    b[fx][fy] = color
                flipped.extend(to_flip)


        # Incremental Zobrist hash update
        # Place the new piece at (x, y)
        new_board._zobrist_hash ^= ZOBRIST_TABLE[x][y][color]
        # Apply flips: toggle opponent -> color
        opp = opponent(color)
        for fx, fy in flipped:
            new_board._zobrist_hash ^= ZOBRIST_TABLE[fx][fy][opp]
            new_board._zobrist_hash ^= ZOBRIST_TABLE[fx][fy][color]

        new_board.move_history = self.move_history + [(x, y, color, flipped)]
        return new_board

        # update counters using flipped length
        f = len(flipped)
        if f:
            if color == BLACK:
                nb._b_count += f
                nb._w_count -= f
            else:
                nb._w_count += f
                nb._b_count -= f

        # history (copy-on-write)
        nb.move_history = self.move_history + [(x, y, color, flipped)]
        # invalidate caches
        nb._valid_moves_cache = {BLACK: None, WHITE: None}
        return nb

    # ----------------------------- counts ---------------------------------


    def count_stones(self):
        # use maintained counters; fall back if missing (older pickles)
        try:
            return self._b_count, self._w_count
        except AttributeError:  # pragma: no cover
            b = sum(row.count(BLACK) for row in self.board)
            w = sum(row.count(WHITE) for row in self.board)
            return b, w

    def get_empty_count(self):

        return sum(row.count(EMPTY) for row in self.board)
        
    def count_score(self, color):
        """Return disc differential from the given color's perspective."""
        b, w = self.count_stones()
        return (b - w) if color == BLACK else (w - b)
        
    def is_stable(self, x, y):
        """Check if a stone at position (x, y) is stable"""
        if self.board[x][y] == EMPTY:

        try:
            return self._empty_count
        except AttributeError:  # pragma: no cover
            return sum(row.count(EMPTY) for row in self.board)

    # ----------------------------- features --------------------------------

    def is_stable(self, x: int, y: int) -> bool:
        """Conservative stability check (kept for compatibility)."""
        b = self.board
        c = b[x][y]
        if c == EMPTY:

            return False
        if (x, y) in CORNERS_SET:
            return True
        # Check 4 axes; if connected to same-color to edge/corner on both sides → stable-ish
        for (dx1, dy1), (dx2, dy2) in (
            ((0, 1), (0, -1)),
            ((1, 0), (-1, 0)),
            ((1, 1), (-1, -1)),
            ((1, -1), (-1, 1)),
        ):
            stable_dir = False
            # forward
            nx, ny = x, y
            while True:
                nx += dx1; ny += dy1
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    stable_dir = True; break
                if b[nx][ny] != c:
                    break
                if (nx, ny) in CORNERS_SET:
                    stable_dir = True; break
            if stable_dir:
                continue
            # backward
            nx, ny = x, y
            while True:
                nx += dx2; ny += dy2
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    stable_dir = True; break
                if b[nx][ny] != c:
                    break
                if (nx, ny) in CORNERS_SET:
                    stable_dir = True; break
            if not stable_dir:
                return False
        return True

    def get_frontier_count(self, color: int) -> int:
        frontier = 0
        b = self.board
        for i in range(8):
            row = b[i]
            for j in range(8):
                if row[j] == color:
                    for dx, dy in DIRS:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < 8 and 0 <= nj < 8 and b[ni][nj] == EMPTY:
                            frontier += 1
                            break
        return frontier

    def get_hash(self):
        """Get a hash representation of the board state"""

    def get_hash(self) -> int:
        # Simple structural hash; your AI uses its own Zobrist, so this is rarely called.

        return hash(tuple(tuple(row) for row in self.board))
