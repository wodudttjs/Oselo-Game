from constants import EMPTY, BLACK, WHITE, opponent, CORNERS
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
            
        new_board.move_history = self.move_history + [(x, y, color, flipped)]
        return new_board

    def count_stones(self):
        b = sum(row.count(BLACK) for row in self.board)
        w = sum(row.count(WHITE) for row in self.board)
        return b, w
        
    def get_empty_count(self):
        return sum(row.count(EMPTY) for row in self.board)
        
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
    # Board 클래스에 추가할 최적화된 apply_move 메서드
def apply_move_fast(self, x, y, color):
    """최적화된 move 적용 (깊은 복사 대신 얕은 복사 + 수동 복원)"""
    # 원본 보드 상태 저장
    original_state = []
    
    # 돌 놓기
    original_state.append((x, y, self.board[x][y]))
    self.board[x][y] = color
    
    flipped = []
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        to_flip = []
        
        while self.in_bounds(nx, ny) and self.board[nx][ny] == opponent(color):
            to_flip.append((nx, ny))
            nx += dx
            ny += dy
            
        if self.in_bounds(nx, ny) and self.board[nx][ny] == color and to_flip:
            for fx, fy in to_flip:
                original_state.append((fx, fy, self.board[fx][fy]))
                self.board[fx][fy] = color
            flipped.extend(to_flip)
    
    return original_state, flipped

def undo_move(self, original_state):
    """move 되돌리기"""
    for x, y, original_color in original_state:
        self.board[x][y] = original_color

# 캐시를 활용한 valid_moves
def __init__(self):
    # ... 기존 초기화 코드 ...
    self._valid_moves_cache = {}
    self._board_hash_cache = None

def get_valid_moves_cached(self, color):
    """캐시된 valid moves"""
    board_hash = self.get_hash()
    cache_key = (board_hash, color)
    
    if cache_key not in self._valid_moves_cache:
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == EMPTY and self.is_valid_move(x, y, color):
                    moves.append((x, y))
        self._valid_moves_cache[cache_key] = moves
    
    return self._valid_moves_cache[cache_key]

def get_hash_cached(self):
    """캐시된 해시"""
    if self._board_hash_cache is None:
        self._board_hash_cache = hash(tuple(tuple(row) for row in self.board))
    return self._board_hash_cache

def _invalidate_cache(self):
    """캐시 무효화 (보드 상태 변경시 호출)"""
    self._valid_moves_cache.clear()
    self._board_hash_cache = None