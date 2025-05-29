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
        
        # Cache for performance
        self._valid_moves_cache = {}
        self._hash_cache = None

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_valid_moves(self, color):
        """Get valid moves with caching for performance"""
        board_hash = self.get_hash()
        cache_key = (board_hash, color)
        
        if cache_key in self._valid_moves_cache:
            return self._valid_moves_cache[cache_key]
        
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] != EMPTY:
                    continue
                if self.is_valid_move(x, y, color):
                    moves.append((x, y))
        
        self._valid_moves_cache[cache_key] = moves
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
        """Apply move and return new board (optimized)"""
        new_board = Board()
        
        # Deep copy the board state
        for i in range(8):
            for j in range(8):
                new_board.board[i][j] = self.board[i][j]
        
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
        """Check if a stone at position (x, y) is stable (enhanced)"""
        if self.board[x][y] == EMPTY:
            return False
            
        color = self.board[x][y]
        
        # Corners are always stable
        if (x, y) in CORNERS:
            return True
            
        # Check if connected to a corner through same-colored stones
        visited = set()
        return self._is_connected_to_stable(x, y, color, visited)
    
    def _is_connected_to_stable(self, x, y, color, visited):
        """Check if position is connected to a stable position"""
        if (x, y) in visited:
            return False
        visited.add((x, y))
        
        # If it's a corner, it's stable
        if (x, y) in CORNERS:
            return True
        
        # Check all 8 directions for connections to stable positions
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (self.in_bounds(nx, ny) and 
                self.board[nx][ny] == color and 
                (nx, ny) not in visited):
                
                if (nx, ny) in CORNERS:
                    return True
                
                # Check if this direction leads to an edge
                if (nx == 0 or nx == 7 or ny == 0 or ny == 7):
                    # Check if the entire edge line is the same color
                    if self._check_edge_stability(nx, ny, color, dx, dy):
                        return True
        
        return False
    
    def _check_edge_stability(self, x, y, color, dx, dy):
        """Check if edge line is stable"""
        # Simple edge stability check
        if x == 0 or x == 7:  # Top or bottom edge
            for j in range(8):
                if self.board[x][j] != color and self.board[x][j] != EMPTY:
                    return False
        elif y == 0 or y == 7:  # Left or right edge
            for i in range(8):
                if self.board[i][y] != color and self.board[i][y] != EMPTY:
                    return False
        return True

    def get_frontier_count(self, color):
        """Count frontier discs (discs adjacent to empty squares) - optimized"""
        frontier = 0
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == color:
                    is_frontier = False
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if self.in_bounds(ni, nj) and self.board[ni][nj] == EMPTY:
                            is_frontier = True
                            break
                    if is_frontier:
                        frontier += 1
        return frontier

    def get_hash(self):
        """Get a hash representation of the board state (cached)"""
        if self._hash_cache is None:
            board_str = ''.join(str(cell) for row in self.board for cell in row)
            self._hash_cache = hash(board_str)
        return self._hash_cache
    
    def get_corner_control(self, color):
        """Get corner control score"""
        corners_controlled = 0
        for corner_x, corner_y in CORNERS:
            if self.board[corner_x][corner_y] == color:
                corners_controlled += 1
        return corners_controlled
    
    def get_edge_control(self, color):
        """Get edge control score"""
        edge_count = 0
        
        # Top and bottom edges
        for j in range(8):
            if self.board[0][j] == color:
                edge_count += 1
            if self.board[7][j] == color:
                edge_count += 1
        
        # Left and right edges (excluding corners already counted)
        for i in range(1, 7):
            if self.board[i][0] == color:
                edge_count += 1
            if self.board[i][7] == color:
                edge_count += 1
        
        return edge_count
    
    def evaluate_mobility_difference(self, color):
        """Evaluate mobility difference efficiently"""
        my_moves = len(self.get_valid_moves(color))
        opp_moves = len(self.get_valid_moves(opponent(color)))
        
        if my_moves + opp_moves == 0:
            return 0
        
        return (my_moves - opp_moves) / (my_moves + opp_moves)
    
    def copy(self):
        """Create a deep copy of the board"""
        new_board = Board()
        for i in range(8):
            for j in range(8):
                new_board.board[i][j] = self.board[i][j]
        new_board.move_history = self.move_history.copy()
        return new_board
    
    def to_string(self):
        """Convert board to string representation for debugging"""
        result = ""
        for row in self.board:
            result += ''.join(['.' if cell == EMPTY else 
                              'B' if cell == BLACK else 'W' for cell in row]) + '\n'
        return result