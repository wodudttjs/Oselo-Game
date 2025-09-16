from __future__ import annotations

from typing import List, Tuple

from constants import EMPTY, BLACK, WHITE, opponent, CORNERS


DIRS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


class Board:
    def __init__(self) -> None:
        self.size = 8
        self.board: List[List[int]] = [[EMPTY for _ in range(8)] for _ in range(8)]
        # Initial four stones
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        # Minimal history for compatibility: (x, y, color, flipped_list)
        self.move_history: List[Tuple[int, int, int, List[Tuple[int, int]]]] = []

    @staticmethod
    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < 8 and 0 <= y < 8

    # ---------------- Moves ----------------
    def _flips_for_move(self, x: int, y: int, color: int) -> List[Tuple[int, int]]:
        if self.board[x][y] != EMPTY:
            return []
        opp = opponent(color)
        flips: List[Tuple[int, int]] = []
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            run: List[Tuple[int, int]] = []
            while self.in_bounds(nx, ny) and self.board[nx][ny] == opp:
                run.append((nx, ny))
                nx += dx
                ny += dy
            if run and self.in_bounds(nx, ny) and self.board[nx][ny] == color:
                flips.extend(run)
        return flips

    def is_valid_move(self, x: int, y: int, color: int) -> bool:
        return len(self._flips_for_move(x, y, color)) > 0

    def get_valid_moves(self, color: int) -> List[Tuple[int, int]]:
        moves: List[Tuple[int, int]] = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == EMPTY and self.is_valid_move(i, j, color):
                    moves.append((i, j))
        return moves

    def apply_move(self, x: int, y: int, color: int) -> "Board":
        flips = self._flips_for_move(x, y, color)
        if not flips:
            # Return self unchanged if illegal; caller should check validity
            return self
        # Create a new board instance with copied grid
        nb = Board.__new__(Board)  # type: ignore
        nb.size = 8
        nb.board = [row[:] for row in self.board]
        nb.move_history = self.move_history[:]

        nb.board[x][y] = color
        for fx, fy in flips:
            nb.board[fx][fy] = color
        nb.move_history.append((x, y, color, flips))
        return nb

    # ---------------- Counts & Features ----------------
    def count_stones(self) -> Tuple[int, int]:
        b = sum(row.count(BLACK) for row in self.board)
        w = sum(row.count(WHITE) for row in self.board)
        return b, w

    def get_empty_count(self) -> int:
        return sum(row.count(EMPTY) for row in self.board)

    def count_score(self, color: int) -> int:
        b, w = self.count_stones()
        return (b - w) if color == BLACK else (w - b)

    def get_frontier_count(self, color: int) -> int:
        frontier = 0
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == color:
                    for dx, dy in DIRS:
                        ni, nj = i + dx, j + dy
                        if self.in_bounds(ni, nj) and self.board[ni][nj] == EMPTY:
                            frontier += 1
                            break
        return frontier

    def get_hash(self) -> int:
        return hash(tuple(tuple(row) for row in self.board))

