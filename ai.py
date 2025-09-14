# ultra_ai/UltraAdvancedAI_corner_safe.py
# Corner-safety hardened search + Corner-first policy + 2-ply corner avoidance:
#  - If a legal corner move exists this turn, TAKE IT immediately (pre-search).
#  - Filter out moves that give opponent an immediate corner (1-ply) when alternatives exist.
#  - NEW: Filter (when possible) moves that let opponent force a corner in 2 plies.

import time
import logging
import random
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Tuple, Dict, List

from constants import (
    adjust_position_weight,
    BLACK,
    WHITE,
    EMPTY,
    opponent,
    CORNERS,
    X_SQUARES,
    C_SQUARES,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

# ---------------- Zobrist (deterministic) ----------------
_rng = random.Random(1337)
ZOBRIST_TABLE: List[List[List[int]]] = [[[0 for _ in range(3)] for _ in range(8)] for _ in range(8)]
for i in range(8):
    for j in range(8):
        ZOBRIST_TABLE[i][j][1] = _rng.getrandbits(64) or 1  # BLACK
        ZOBRIST_TABLE[i][j][2] = _rng.getrandbits(64) or 1  # WHITE
ZOBRIST_TURN: int = _rng.getrandbits(64) or 1


def _pidx(v: int) -> int:
    return 1 if v == BLACK else 2 if v == WHITE else 0


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(slots=True)
class TTEntry:
    score: float
    move: Optional[Tuple[int, int]]
    depth: int
    node_type: str
    age: int


class UltraAdvancedAI:
    """Corner-first Othello AI with 2-ply corner exposure avoidance.
       Public: get_move(board) -> (x,y) or None
    """

    def __init__(self, color: int, difficulty: str = 'hard', time_limit: float = 10.0) -> None:
        self.color = color
        self.difficulty = difficulty
        self.time_limit = float(time_limit)

        self.tt: Dict[int, TTEntry] = {}
        self.tt_age = 0
        self.max_tt_size = 2**20

        self.killer_moves = defaultdict(list)
        self.counter_moves: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.history_table: List[int] = [0] * 64

        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0

        self.pattern_weights = self._init_patterns()

        self.opening_book: Dict[int, Tuple[int, int]] = {}
        self.endgame_cache: Dict[Tuple[int, int], Tuple[int, Optional[Tuple[int, int]]]] = {}

        self._init_pos_weights()
        if difficulty == 'easy':
            self.max_depth = 6
        elif difficulty == 'medium':
            self.max_depth = 8
        else:
            self.max_depth = 10

        # Precompute X/C → corner map for instant checks
        self.X_TO_CORNER = {(1, 1): (0, 0), (1, 6): (0, 7), (6, 1): (7, 0), (6, 6): (7, 7)}
        self.C_TO_CORNER = {
            (0, 1): (0, 0), (1, 0): (0, 0),
            (0, 6): (0, 7), (1, 7): (0, 7),
            (7, 1): (7, 0), (6, 0): (7, 0),
            (7, 6): (7, 7), (6, 7): (7, 7),
        }

        self.enable_null_move = True
        self.null_move_min_depth = 3
        self.null_move_R = 2
        self.null_move_min_empties = 14
        self.enable_futility = True
        self.futility_margin = 250
        self.futility_max_depth = 2
        self.enable_lmp = True

        # 2-ply corner avoidance tunables
        self.two_ply_check_empties = 20  # apply when empties > this
        self.two_ply_opp_limit = 6      # consider up to N opponent replies
        self.two_ply_our_limit = 6      # consider up to M our replies

    # ---------- init helpers ----------

    def _init_patterns(self):
        return {
            'corner_control': 1000,
            'edge_stability': 300,
            'mobility_ratio': 200,
            'disc_differential': 100,
            'frontier_discs': -50,
        }

    def _init_pos_weights(self) -> None:
        for c in CORNERS:
            adjust_position_weight(c, 500)
        for x in X_SQUARES:
            adjust_position_weight(x, -160, stages=('early', 'mid'))  # 강한 패널티
        for c in C_SQUARES:
            adjust_position_weight(c, -100, stages=('mid',))

    # ---------- hashing ----------

    def _hash(self, board, side: int) -> int:
        h = 0
        grid = board.board
        for i in range(8):
            row = grid[i]
            for j in range(8):
                v = row[j]
                if v:
                    h ^= ZOBRIST_TABLE[i][j][_pidx(v)]
        if side == WHITE:
            h ^= ZOBRIST_TURN
        return h

    # ---------- eval (kept short) ----------

    def evaluate_board_neural(self, board) -> int:
        empties = board.get_empty_count()
        if empties == 0:
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)
            return 10000 if diff > 0 else -10000 if diff < 0 else 0
        my_c = sum(1 for x, y in CORNERS if board.board[x][y] == self.color)
        op_c = sum(1 for x, y in CORNERS if board.board[x][y] == opponent(self.color))
        corner_control = my_c - op_c
        my_moves = len(board.get_valid_moves(self.color))
        op_moves = len(board.get_valid_moves(opponent(self.color)))
        mobility = (my_moves - op_moves) / max(1, my_moves + op_moves)
        b, w = board.count_stones()
        disc_diff = (b - w) if self.color == BLACK else (w - b)
        frontier = board.get_frontier_count(opponent(self.color)) - board.get_frontier_count(self.color)
        pw = self.pattern_weights
        score = 0
        score += corner_control * pw['corner_control']
        score += mobility * pw['mobility_ratio']
        score += disc_diff * pw['disc_differential']
        score += frontier * pw['frontier_discs']
        if empties > 50:
            score *= 1.2
        elif empties <= 20:
            score *= 0.85
        return int(score)

    # ---------- corner safety helpers ----------

    def _has_corner_move(self, board, side: int) -> bool:
        return any(m in CORNERS for m in board.get_valid_moves(side))

    def _is_corner_exposing_move(self, board, move: Tuple[int, int], side: int) -> bool:
        x, y = move
        grid = board.board
        if move in CORNERS:
            return False
        c = self.X_TO_CORNER.get(move)
        if c:
            cx, cy = c
            if grid[cx][cy] == EMPTY:
                return True
        c = self.C_TO_CORNER.get(move)
        if c:
            cx, cy = c
            if grid[cx][cy] == EMPTY:
                return True
        nb = board.apply_move(x, y, side)
        opp = opponent(side)
        if self._has_corner_move(nb, opp):
            return True
        return False

    def _exposes_corner_in_two(self, board, move: Tuple[int, int], side: int,
                               opp_limit: int, our_limit: int) -> bool:
        """Conservative 2-ply trap check.
        Returns True if there exists an opponent reply after our move such that
        for all of our top replies the opponent then has an immediate corner.
        """
        x, y = move
        if move in CORNERS:
            return False
        nb = board.apply_move(x, y, side)
        opp = opponent(side)
        opp_moves = nb.get_valid_moves(opp)
        # Immediate corner already
        if any(m in CORNERS for m in opp_moves):
            return True
        # Prioritize opponent replies that are near empty corners or flip more
        def opp_priority(mv: Tuple[int, int]) -> int:
            px, py = mv
            s = 0
            c = self.X_TO_CORNER.get(mv)
            if c:
                cx, cy = c
                if nb.board[cx][cy] == EMPTY:
                    s += 500
            c = self.C_TO_CORNER.get(mv)
            if c:
                cx, cy = c
                if nb.board[cx][cy] == EMPTY:
                    s += 200
            s += self._estimate_flips_scan(nb.board, px, py, opp)
            return s
        opp_moves = sorted(opp_moves, key=opp_priority, reverse=True)[:opp_limit]
        for om in opp_moves:
            nb2 = nb.apply_move(om[0], om[1], opp)
            our_replies = nb2.get_valid_moves(side)
            if not our_replies:
                # we pass; if they already can corner now, it's a trap
                if self._has_corner_move(nb2, opp):
                    return True
                # else continue (not decisive)
                continue
            # Try to parry with our best replies
            def our_priority(mv: Tuple[int, int]) -> int:
                rx, ry = mv
                return self._estimate_flips_scan(nb2.board, rx, ry, side)
            parried = False
            for rm in sorted(our_replies, key=our_priority, reverse=True)[:our_limit]:
                nb3 = nb2.apply_move(rm[0], rm[1], side)
                if not self._has_corner_move(nb3, opp):
                    parried = True
                    break
            if not parried:
                return True
        return False

    def _corner_exposure_penalty(self, board, move: Tuple[int, int], side: int, empties: int) -> int:
        if self._is_corner_exposing_move(board, move, side):
            return 200000 if empties > 20 else 80000
        return 0

    # ---------- move ordering ----------

    def _static_move_value(self, board, move: Tuple[int, int]) -> int:
        x, y = move
        if move in CORNERS:
            return 20000  # massive bonus
        c = self.X_TO_CORNER.get(move)
        if c:
            cx, cy = c
            if board.board[cx][cy] == EMPTY:
                return -10000
            return 300 if board.board[cx][cy] == self.color else -200
        c = self.C_TO_CORNER.get(move)
        if c:
            cx, cy = c
            if board.board[cx][cy] == EMPTY:
                return -4000
            return -100
        if x == 0 or x == 7 or y == 0 or y == 7:
            return 150
        return 0

    def _estimate_flips_scan(self, grid, x: int, y: int, side: int) -> int:
        if grid[x][y] != EMPTY:
            return 0
        opp = opponent(side)
        total = 0
        for dx, dy in DIRS:
            i, j = x + dx, y + dy
            cnt = 0
            while 0 <= i < 8 and 0 <= j < 8 and grid[i][j] == opp:
                cnt += 1; i += dx; j += dy
            if cnt and 0 <= i < 8 and 0 <= j < 8 and grid[i][j] == side:
                total += cnt
        return total

    def _order_moves(self, board, moves, depth: int, side: int,
                     prev_best: Optional[Tuple[int, int]], prev_move: Optional[Tuple[int, int]], empties: int):
        if not moves:
            return []
        grid = board.board
        key = self._hash(board, side)
        tt = self.tt.get(key)
        tt_move = tt.move if tt else None
        total_hist = sum(self.history_table) or 1

        scored: List[Tuple[int, Tuple[int, int]]] = []
        for mv in moves:
            x, y = mv
            s = 0
            if prev_best and mv == prev_best:
                s += 10000
            if tt_move and mv == tt_move:
                s += 8000
            killers = self.killer_moves.get(depth, [])
            if mv in killers:
                s += 4000 + (2 - killers.index(mv)) * 1000
            if prev_move is not None and prev_move in self.counter_moves and mv == self.counter_moves[prev_move]:
                s += 3000
            s += (self.history_table[(x << 3) | y] * 2000) // total_hist
            s += self._static_move_value(board, mv)
            s -= self._corner_exposure_penalty(board, mv, side, empties)
            s += self._estimate_flips_scan(grid, x, y, side) * 50
            scored.append((s, mv))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored]

    # ---------- search ----------

    def _lmr(self, depth: int, i: int, n: int) -> int:
        if depth >= 3 and i >= 3 and n > 6:
            return min(2, (i - 2) // 3)
        return 0

    def alpha_beta_enhanced(self, board, depth: int, alpha: float, beta: float, side: int,
                             start_time: float, prev_move: Optional[Tuple[int, int]] = None):
        self.nodes_searched += 1
        if (self.nodes_searched & 0x7FF) == 0 and time.time() - start_time >= self.time_limit * 0.95:
            return self.evaluate_board_neural(board), None

        key = self._hash(board, side)
        tt = self.tt.get(key)
        if tt and tt.depth >= depth:
            if tt.node_type == 'exact':
                return tt.score, tt.move
            if tt.node_type == 'lowerbound' and tt.score >= beta:
                return tt.score, tt.move
            if tt.node_type == 'upperbound' and tt.score <= alpha:
                return tt.score, tt.move

        moves = board.get_valid_moves(side)
        empties = board.get_empty_count()
        if depth == 0:
            return self.evaluate_board_neural(board), None
        if not moves:
            opp = opponent(side)
            if not board.get_valid_moves(opp):
                return self.evaluate_board_neural(board), None
            sc, _ = self.alpha_beta_enhanced(board, depth, -beta, -alpha, opp, start_time, prev_move)
            return -sc, None

        # null-move
        if self.enable_null_move and depth >= self.null_move_min_depth and empties >= self.null_move_min_empties and moves:
            opp = opponent(side)
            R = self.null_move_R if depth > 6 else 1
            sc, _ = self.alpha_beta_enhanced(board, depth - 1 - R, -beta, -beta + 1, opp, start_time, prev_move)
            sc = -sc
            if sc >= beta:
                return sc, None

        # futility
        if self.enable_futility and depth <= self.futility_max_depth:
            st = self.evaluate_board_neural(board)
            if st + self.futility_margin <= alpha:
                return st, None

        ordered = self._order_moves(board, moves, depth, side, tt.move if tt else None, prev_move, empties)

        # Filter out corner-exposing moves (1-ply), then 2-ply traps if safe options exist
        if empties > self.two_ply_check_empties:
            safe1 = [m for m in ordered if not self._is_corner_exposing_move(board, m, side)]
            if safe1:
                # try to filter 2-ply traps; if all are traps, keep safe1
                safer2 = [m for m in safe1 if not self._exposes_corner_in_two(board, m, side,
                                                                              self.two_ply_opp_limit,
                                                                              self.two_ply_our_limit)]
                if safer2:
                    ordered = safer2
                else:
                    ordered = safe1

        best_move = ordered[0]
        best_score = float('-inf')
        a0 = alpha
        opp = opponent(side)

        lmp_cut = None
        if self.enable_lmp and depth <= 2:
            lmp_cut = 6 + depth

        for i, mv in enumerate(ordered):
            if lmp_cut is not None and i > lmp_cut and alpha > a0:
                break
            red = self._lmr(depth, i, len(ordered))
            x, y = mv
            nb = board.apply_move(x, y, side)
            child_prev = mv

            if i == 0:
                sc, _ = self.alpha_beta_enhanced(nb, depth - 1 - red, -beta, -alpha, opp, start_time, child_prev)
                sc = -sc
            else:
                sc, _ = self.alpha_beta_enhanced(nb, depth - 1 - red, -alpha - 1, -alpha, opp, start_time, child_prev)
                sc = -sc
                if alpha < sc < beta:
                    sc2, _ = self.alpha_beta_enhanced(nb, depth - 1 - red, -beta, -sc, opp, start_time, child_prev)
                    sc = -sc2

            if sc > best_score:
                best_score = sc
                best_move = mv
            if sc > alpha:
                alpha = sc

            if alpha >= beta:
                kl = self.killer_moves[depth]
                if mv not in kl:
                    if len(kl) >= 2:
                        kl.pop(0)
                    kl.append(mv)
                self.history_table[(x << 3) | y] += depth * depth
                if prev_move is not None:
                    self.counter_moves[prev_move] = mv
                break

        if len(self.tt) >= self.max_tt_size:
            threshold = self.tt_age - 2
            removed = 0
            for k in list(self.tt.keys()):
                if self.tt[k].age < threshold:
                    self.tt.pop(k); removed += 1
                if removed >= max(1, self.max_tt_size // 64):
                    break

        ntype = 'exact'
        if best_score <= a0:
            ntype = 'upperbound'
        elif best_score >= beta:
            ntype = 'lowerbound'
        self.tt[key] = TTEntry(best_score, best_move, depth, ntype, self.tt_age)
        return best_score, best_move

    # ---------- corner-first policy helpers ----------

    @staticmethod
    def _pick_corner_if_available(moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        for m in moves:
            if m in CORNERS:
                return m
        return None

    # ---------- endgame & ID ----------

    def perfect_endgame_solver(self, board, empties: int, side: int):
        if empties > 12:
            return None
        key = (self._hash(board, side), empties)
        if key in self.endgame_cache:
            return self.endgame_cache[key]
        moves = board.get_valid_moves(side)
        opp = opponent(side)
        if empties == 0 or (not moves and not board.get_valid_moves(opp)):
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)
            res = (diff, None)
            self.endgame_cache[key] = res
            return res
        if not moves:
            sc, _ = self.perfect_endgame_solver(board, empties, opp)
            res = (-sc, None)
            self.endgame_cache[key] = res
            return res
        best_sc, best_mv = -10**9, None
        for mv in moves:
            x, y = mv
            nb = board.apply_move(x, y, side)
            sc, _ = self.perfect_endgame_solver(nb, empties - 1, opp)
            sc = -sc
            if sc > best_sc:
                best_sc, best_mv = sc, mv
        res = (best_sc, best_mv)
        self.endgame_cache[key] = res
        return res

    def iterative_deepening_ultimate(self, board):
        start = time.time()
        side = self.color
        moves = board.get_valid_moves(side)
        if not moves:
            return None
        # Corner-first: if any corner is legal now, take it immediately
        corner_now = self._pick_corner_if_available(moves)
        if corner_now is not None:
            logging.info(f"[Corner-Safe AI] Corner available → taking {corner_now}")
            return corner_now
        if len(moves) == 1:
            return moves[0]
        h = self._hash(board, side)
        if h in self.opening_book:
            return self.opening_book[h]

        logging.info(f"[Corner-Safe AI] max_depth={self.max_depth}, moves={len(moves)}")
        best_move = None
        prev_score = 0
        window = 50
        nodes_total = 0
        for depth in range(1, self.max_depth + 1):
            self.nodes_searched = self.tt_hits = self.cutoffs = 0
            self.tt_age += 1
            try:
                if depth >= 4 and best_move is not None:
                    alpha = prev_score - window
                    beta = prev_score + window
                    sc, mv = self.alpha_beta_enhanced(board, depth, alpha, beta, side, start, None)
                    if sc <= alpha:
                        sc, mv = self.alpha_beta_enhanced(board, depth, float('-inf'), beta, side, start, None)
                    elif sc >= beta:
                        sc, mv = self.alpha_beta_enhanced(board, depth, alpha, float('inf'), side, start, None)
                    window = min(100, window + 25)
                else:
                    sc, mv = self.alpha_beta_enhanced(board, depth, float('-inf'), float('inf'), side, start, None)
                if mv is not None:
                    best_move = mv
                    prev_score = sc
                nodes_total += self.nodes_searched
                if abs(sc) >= 9000 or time.time() - start >= self.time_limit * 0.8:
                    break
            except Exception as e:
                logging.exception(f"Error at depth {depth}: {e}")
                break
        return best_move

    def get_move(self, board) -> Optional[Tuple[int, int]]:
        # Corner-first BEFORE any endgame shortcut
        my_moves = board.get_valid_moves(self.color)
        corner_now = self._pick_corner_if_available(my_moves)
        if corner_now is not None:
            logging.info(f"[Corner-Safe AI] Corner available at root → taking {corner_now}")
            return corner_now
        empties = board.get_empty_count()
        end = self.perfect_endgame_solver(board, empties, self.color)
        if end is not None and end[1] is not None:
            logging.info(f"Perfect endgame move: {end[1]}")
            return end[1]
        return self.iterative_deepening_ultimate(board)
