from constants import EMPTY, BLACK, WHITE, opponent, CORNERS
from zobrist import ZOBRIST_TABLE
import numpy as np
import copy

class Board:
    def __init__(self):
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
                    continue
                if self.is_valid_move(x, y, color):
                    moves.append((x, y))
        return moves

    def is_valid_move(self, x, y, color):
        if self.board[x][y] != EMPTY:
            return False
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            found_opponent = False
            
            while self.in_bounds(nx, ny) and self.board[nx][ny] == opponent(color):
                found_opponent = True
                nx += dx
                ny += dy
                
            if found_opponent and self.in_bounds(nx, ny) and self.board[nx][ny] == color:
                return True
        return False

    def apply_move(self, x, y, color):
        new_board = copy.deepcopy(self)
        new_board.board[x][y] = color
        flipped = []

        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            to_flip = []
            
            while new_board.in_bounds(nx, ny) and new_board.board[nx][ny] == opponent(color):
                to_flip.append((nx, ny))
                nx += dx
                ny += dy
                
            if new_board.in_bounds(nx, ny) and new_board.board[nx][ny] == color and to_flip:
                for fx, fy in to_flip:
                    new_board.board[fx][fy] = color
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

    def count_stones(self):
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
            return False
            
        color = self.board[x][y]
        
        # Corners are always stable
        if (x, y) in CORNERS:
            return True
            
        # Check stability in all directions
        directions = [
            [(0, 1), (0, -1)],   # horizontal
            [(1, 0), (-1, 0)],   # vertical
            [(1, 1), (-1, -1)],  # diagonal
            [(1, -1), (-1, 1)]   # anti-diagonal
        ]
        
        for dir_pair in directions:
            stable_in_direction = False
            
            for dx, dy in dir_pair:
                nx, ny = x, y
                while True:
                    nx += dx
                    ny += dy
                    if not self.in_bounds(nx, ny):
                        # Reached edge
                        stable_in_direction = True
                        break
                    if self.board[nx][ny] != color:
                        # Found different color or empty
                        break
                    if (nx, ny) in CORNERS:
                        # Connected to corner
                        stable_in_direction = True
                        break
                
                if stable_in_direction:
                    break
                    
            if not stable_in_direction:
                return False
                
        return True

    def get_frontier_count(self, color):
        """Count frontier discs (discs adjacent to empty squares)"""
        frontier = 0
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == color:
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if self.in_bounds(ni, nj) and self.board[ni][nj] == EMPTY:
                            frontier += 1
                            break
        return frontier

    def get_hash(self):
        """Get a hash representation of the board state"""
        return hash(tuple(tuple(row) for row in self.board))
