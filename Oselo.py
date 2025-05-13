import tkinter as tk
from tkinter import messagebox
import copy

EMPTY, BLACK, WHITE = '.', 'B', 'W'

# 평가 가중치 (중앙 약함, 모서리 강함)
WEIGHTS = [
    [100, -20, 10,  5,  5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10,   -2, -1, -1, -1, -1,  -2,  10],
    [5,    -2, -1, -1, -1, -1,  -2,   5],
    [5,    -2, -1, -1, -1, -1,  -2,   5],
    [10,   -2, -1, -1, -1, -1,  -2,  10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10,  5,  5, 10, -20, 100],
]
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

def opponent(color):
    return BLACK if color == WHITE else WHITE

class Board:
    def __init__(self):
        self.size = 8
        self.board = [[EMPTY] * self.size for _ in range(self.size)]
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_valid_moves(self, color):
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] != EMPTY and self.board[x][y] != opponent(color):
                    continue
                if self.is_valid_move(x, y, color):
                    moves.append((x, y))
        return moves

    def is_valid_move(self, x, y, color):
        if self.board[x][y] != EMPTY:
            return False
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
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
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            to_flip = []
            while new_board.in_bounds(nx, ny) and new_board.board[nx][ny] == opponent(color):
                to_flip.append((nx, ny))
                nx += dx
                ny += dy
            if new_board.in_bounds(nx, ny) and new_board.board[nx][ny] == color:
                for fx, fy in to_flip:
                    new_board.board[fx][fy] = color
        return new_board

    def count_stones(self):
        b = sum(row.count(BLACK) for row in self.board)
        w = sum(row.count(WHITE) for row in self.board)
        return b, w

class AlphaBetaAI:
    def __init__(self, color, depth=3):
        self.color = color
        self.depth = depth

    def evaluate(self, board_obj):
        board = board_obj.board

        def positional_score():
            score = 0
            for i in range(8):
                for j in range(8):
                    if board[i][j] == self.color:
                        score += WEIGHTS[i][j]
                    elif board[i][j] == opponent(self.color):
                        score -= WEIGHTS[i][j]
            return score

        def mobility_score():
            my_moves = len(board_obj.get_valid_moves(self.color))
            opp_moves = len(board_obj.get_valid_moves(opponent(self.color)))
            if my_moves + opp_moves == 0:
                return 0
            return 100 * (my_moves - opp_moves) / (my_moves + opp_moves)

        def corner_score():
            my = sum(1 for x, y in CORNERS if board[x][y] == self.color)
            opp = sum(1 for x, y in CORNERS if board[x][y] == opponent(self.color))
            return 25 * (my - opp)

        return positional_score() + mobility_score() + corner_score()

    def alphabeta(self, board, depth, alpha, beta, maximizing):
        current_color = self.color if maximizing else opponent(self.color)
        moves = board.get_valid_moves(current_color)

        if depth == 0 or not moves:
            return self.evaluate(board), None

        best_move = None
        if maximizing:
            value = float('-inf')
            for move in moves:
                new_board = board.apply_move(*move, current_color)
                eval, _ = self.alphabeta(new_board, depth - 1, alpha, beta, False)
                if eval > value:
                    value = eval
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_move
        else:
            value = float('inf')
            for move in moves:
                new_board = board.apply_move(*move, current_color)
                eval, _ = self.alphabeta(new_board, depth - 1, alpha, beta, True)
                if eval < value:
                    value = eval
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_move

    def get_move(self, board):
        _, move = self.alphabeta(board, self.depth, float('-inf'), float('inf'), True)
        return move

class OthelloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Othello (Tkinter)")
        self.cell_size = 60
        self.board = Board()
        self.ai = AlphaBetaAI(WHITE)
        self.current_player = BLACK

        self.canvas = tk.Canvas(self.root, width=self.cell_size*8, height=self.cell_size*8, bg="dark green")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)

        self.status_label = tk.Label(self.root, text="Your turn (Black)", font=("Arial", 14))
        self.status_label.pack()
        self.update_gui()

    def handle_click(self, event):
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        if (y, x) in self.board.get_valid_moves(self.current_player):
            self.board = self.board.apply_move(y, x, self.current_player)
            self.update_gui()
            if not self.board.get_valid_moves(self.ai.color) and not self.board.get_valid_moves(self.current_player):
                self.check_game_end()
                return
            if self.board.get_valid_moves(self.ai.color):
                self.root.after(500, self.ai_move)
            else:
                messagebox.showinfo("Turn Skipped", "AI has no valid move. Your turn again.")

    def ai_move(self):
        if self.board.get_valid_moves(self.ai.color):
            move = self.ai.get_move(self.board)
            if move:
                self.board = self.board.apply_move(*move, self.ai.color)
                self.update_gui()

        if not self.board.get_valid_moves(self.ai.color) and not self.board.get_valid_moves(self.current_player):
            self.check_game_end()
            return

        if not self.board.get_valid_moves(self.current_player):
            messagebox.showinfo("Turn Skipped", "You have no valid move. AI plays again.")
            self.root.after(500, self.ai_move)

    def check_game_end(self):
        b, w = self.board.count_stones()
        msg = f"Game Over!\nBlack: {b} | White: {w}\n"
        msg += "You Win!" if b > w else "AI Wins!" if w > b else "Draw!"
        messagebox.showinfo("Result", msg)
        self.root.destroy()

    def update_gui(self):
        self.canvas.delete("all")
        for i in range(8):
            for j in range(8):
                x0, y0 = j * self.cell_size, i * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black", fill="dark green")

                stone = self.board.board[i][j]
                if stone == BLACK:
                    self.canvas.create_oval(x0+5, y0+5, x1-5, y1-5, fill="black")
                elif stone == WHITE:
                    self.canvas.create_oval(x0+5, y0+5, x1-5, y1-5, fill="white")

        b, w = self.board.count_stones()
        self.status_label.config(text=f"Black: {b} | White: {w}")

# 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = OthelloGUI(root)
    root.mainloop()
