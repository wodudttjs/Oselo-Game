# ultra_ai/UltraAdvancedAI_corner_safe.py
# Corner-safety hardened search + Corner-first policy + 2-ply corner avoidance

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import load_config
from zobrist import ZOBRIST_TABLE, ZOBRIST_TURN
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

# ---- logging ----
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

# ---- sentinel scores ----
MATE_SCORE = 100_000
INF_SCORE = 10**9

# ---- Zobrist (centralized in zobrist.py) ----

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(slots=True)
class TTEntry:
    score: int
    move: Optional[Tuple[int, int]]
    depth: int
    node_type: str  # 'exact' | 'upperbound' | 'lowerbound'
    age: int
    pos_hash: int


class BitBoard:
    """Bit-based board representation for ultra-fast operations."""
    _MASK = 0xFFFFFFFFFFFFFFFF
    _FILE_A = 0x0101010101010101
    _FILE_H = 0x8080808080808080
    _NOT_A = _MASK ^ _FILE_A
    _NOT_H = _MASK ^ _FILE_H

    @staticmethod
    def _shift_n(bb): return (bb >> 8) & BitBoard._MASK
    @staticmethod
    def _shift_s(bb): return (bb << 8) & BitBoard._MASK
    @staticmethod
    def _shift_e(bb): return ((bb & BitBoard._NOT_H) << 1) & BitBoard._MASK
    @staticmethod
    def _shift_w(bb): return ((bb & BitBoard._NOT_A) >> 1) & BitBoard._MASK
    @staticmethod
    def _shift_ne(bb): return ((bb & BitBoard._NOT_H) >> 7) & BitBoard._MASK
    @staticmethod
    def _shift_nw(bb): return ((bb & BitBoard._NOT_A) >> 9) & BitBoard._MASK
    @staticmethod
    def _shift_se(bb): return ((bb & BitBoard._NOT_H) << 9) & BitBoard._MASK
    @staticmethod
    def _shift_sw(bb): return ((bb & BitBoard._NOT_A) << 7) & BitBoard._MASK

    @staticmethod
    def _iter_bits(mask: int):
        while mask:
            lsb = mask & -mask
            idx = (lsb.bit_length() - 1)
            yield idx
            mask ^= lsb

    def __init__(self, board: Optional[object] = None):
        if isinstance(board, BitBoard):
            self.black = int(board.black)
            self.white = int(board.white)
            self.hash_base = np.uint64(getattr(board, 'hash_base', 0))
            return
        if board is not None:
            self.black = 0
            self.white = 0
            grid = board.board  # expects 8x8 ints: EMPTY, BLACK, WHITE
            for i in range(8):
                for j in range(8):
                    pos = (i << 3) | j
                    v = grid[i][j]
                    if v == BLACK:
                        self.black |= (1 << pos)
                    elif v == WHITE:
                        self.white |= (1 << pos)
            self.hash_base = self._compute_hash_base_from_masks(self.black, self.white)
        else:
            # Standard initial Othello position
            self.black = 0x0000001008000000
            self.white = 0x0000000810000000
            self.hash_base = self._compute_hash_base_from_masks(self.black, self.white)

    @staticmethod
    def _compute_hash_base_from_masks(black_mask: int, white_mask: int) -> np.uint64:
        h = np.uint64(0)
        for idx in BitBoard._iter_bits(black_mask):
            x, y = divmod(idx, 8)
            h ^= ZOBRIST_TABLE[x][y][BLACK]
        for idx in BitBoard._iter_bits(white_mask):
            x, y = divmod(idx, 8)
            h ^= ZOBRIST_TABLE[x][y][WHITE]
        return h

    def get_empty_mask(self) -> int:
        return ~(self.black | self.white) & BitBoard._MASK

    @staticmethod
    def popcount(mask: int) -> int:
        return mask.bit_count()

    def _valid_moves_mask(self, own: int, opp: int) -> int:
        empty = ~(own | opp) & BitBoard._MASK
        moves = 0
        for sh in (
            self._shift_n, self._shift_s, self._shift_e, self._shift_w,
            self._shift_ne, self._shift_nw, self._shift_se, self._shift_sw,
        ):
            x = sh(own) & opp
            c = x
            while c:
                c = sh(c) & opp
                x |= c
            moves |= sh(x) & empty
        return moves

    def get_valid_moves_bitboard(self, color: int) -> List[Tuple[int, int]]:
        own = self.black if color == BLACK else self.white
        opp = self.white if color == BLACK else self.black
        mask = self._valid_moves_mask(own, opp)
        moves: List[Tuple[int, int]] = []
        while mask:
            lsb = mask & -mask
            idx = (lsb.bit_length() - 1)
            x, y = divmod(idx, 8)
            moves.append((x, y))
            mask ^= lsb
        return moves

    def _flip_mask_for_move(self, own: int, opp: int, move_bit: int) -> int:
        flips = 0
        for sh in (
            self._shift_n, self._shift_s, self._shift_e, self._shift_w,
            self._shift_ne, self._shift_nw, self._shift_se, self._shift_sw,
        ):
            x = 0
            c = sh(move_bit)
            while c and (c & opp):
                x |= c
                c = sh(c)
            if c and (c & own):
                flips |= x
        return flips

    def apply_move_bitboard(self, x: int, y: int, color: int) -> Optional[BitBoard]:
        idx = (x << 3) | y
        move_bit = 1 << idx
        own = self.black if color == BLACK else self.white
        opp = self.white if color == BLACK else self.black
        flips = self._flip_mask_for_move(own, opp, move_bit)
        if flips == 0:
            return None
        own ^= flips | move_bit
        opp ^= flips
        bb = BitBoard()
        if color == BLACK:
            bb.black, bb.white = own, opp
        else:
            bb.white, bb.black = own, opp
        base = np.uint64(self.hash_base)
        base ^= ZOBRIST_TABLE[x][y][color]
        opp_color = opponent(color)
        for fidx in BitBoard._iter_bits(flips):
            fx, fy = divmod(fidx, 8)
            base ^= ZOBRIST_TABLE[fx][fy][opp_color]
            base ^= ZOBRIST_TABLE[fx][fy][color]
        bb.hash_base = base
        return bb


class Evaluator:
    """고급 오델로 평가함수 - 실전 전략을 반영한 종합적 평가"""
    
    def __init__(self):
        # 위치별 가중치 테이블
        self.position_weights = [
            [1000, -300, 100, 80, 80, 100, -300, 1000],  # 코너=1000, X스퀘어=-300
            [-300, -500, -50, -20, -20, -50, -500, -300], # C스퀘어=-500
            [100, -50, 50, 10, 10, 50, -50, 100],
            [80, -20, 10, 5, 5, 10, -20, 80],
            [80, -20, 10, 5, 5, 10, -20, 80],
            [100, -50, 50, 10, 10, 50, -50, 100],
            [-300, -500, -50, -20, -20, -50, -500, -300],
            [1000, -300, 100, 80, 80, 100, -300, 1000]
        ]
        
        # 방향벡터 (8방향)
        self.directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # 코너와 인접 위치 정의
        self.corners = [(0,0), (0,7), (7,0), (7,7)]
        self.x_squares = [(1,1), (1,6), (6,1), (6,6)]
        self.c_squares = [(0,1), (1,0), (0,6), (1,7), (6,0), (7,1), (7,6), (6,7)]
        self.edges = [(0,i) for i in range(8)] + [(7,i) for i in range(8)] + \
                    [(i,0) for i in range(1,7)] + [(i,7) for i in range(1,7)]

    def evaluate_bitboard(self, bb: BitBoard, ai_color: int) -> int:
        """종합적인 보드 평가"""
        empty_count = 64 - BitBoard.popcount(bb.black | bb.white)
        my_mask = bb.black if ai_color == BLACK else bb.white
        op_mask = bb.white if ai_color == BLACK else bb.black
        
        # 게임 단계 결정
        if empty_count > 50:
            stage = "opening"
        elif empty_count > 20:
            stage = "midgame"
        else:
            stage = "endgame"
        
        score = 0
        
        # 1. 기본 위치 평가
        score += self._evaluate_positions(my_mask, op_mask) * self._get_position_weight(stage)
        
        # 2. 이동성(Mobility) 평가 - 핵심 전략
        score += self._evaluate_mobility(bb, ai_color) * self._get_mobility_weight(stage)
        
        # 3. 프론티어(Frontier) 평가 - 상대가 둘 수 있는 위치 최소화
        score += self._evaluate_frontier(bb, ai_color) * self._get_frontier_weight(stage)
        
        # 4. 코너 안전성 평가
        score += self._evaluate_corner_safety(bb, ai_color) * self._get_corner_weight(stage)
        
        # 5. 안정성(Stability) 평가 - 굳힘돌
        score += self._evaluate_stability(bb, ai_color) * self._get_stability_weight(stage)
        
        # 6. 변(Edge) 컨트롤 평가
        score += self._evaluate_edge_control(bb, ai_color) * self._get_edge_weight(stage)
        
        # 7. 패리티(홀짝) 평가 - 엔드게임에서 중요
        if stage == "endgame":
            score += self._evaluate_parity(empty_count, ai_color) * 100
        
        # 8. 디스크 개수 차이 (엔드게임에서만 중요)
        if stage == "endgame":
            disc_diff = BitBoard.popcount(my_mask) - BitBoard.popcount(op_mask)
            score += disc_diff * 50
        
        return int(score)
    
    def _evaluate_positions(self, my_mask: int, op_mask: int) -> float:
        """위치별 가중치 평가"""
        score = 0
        for i in range(8):
            for j in range(8):
                pos = 1 << ((i << 3) | j)
                if my_mask & pos:
                    score += self.position_weights[i][j]
                elif op_mask & pos:
                    score -= self.position_weights[i][j]
        return score
    
    def _evaluate_mobility(self, bb: BitBoard, ai_color: int) -> float:
        """이동성 평가 - 상대방 선택지 제한이 핵심"""
        my_moves = len(bb.get_valid_moves_bitboard(ai_color))
        op_moves = len(bb.get_valid_moves_bitboard(opponent(ai_color)))
        
        # 상대방 이동성을 제한하는 것이 더 중요
        if my_moves + op_moves == 0:
            return 0
        
        # 절대적 이동성과 상대적 이동성 모두 고려
        mobility_diff = my_moves - op_moves
        relative_mobility = mobility_diff / max(1, my_moves + op_moves)
        
        # 상대방 선택지가 적을수록 좋음 (강한 페널티)
        if op_moves == 0 and my_moves > 0:
            return 2000  # 상대방이 패스해야 하는 상황
        elif op_moves <= 2:
            return 1000 + mobility_diff * 100  # 상대방 선택지가 매우 제한적
        
        return mobility_diff * 80 + relative_mobility * 200
    
    def _evaluate_frontier(self, bb: BitBoard, ai_color: int) -> float:
        """프론티어 평가 - 상대가 인접할 수 있는 빈칸 최소화"""
        my_mask = bb.black if ai_color == BLACK else bb.white
        op_mask = bb.white if ai_color == BLACK else bb.black
        empty_mask = bb.get_empty_mask()
        
        my_frontier = self._count_frontier_discs(my_mask, empty_mask)
        op_frontier = self._count_frontier_discs(op_mask, empty_mask)
        
        # 내 프론티어는 적을수록, 상대 프론티어는 많을수록 좋음
        return (op_frontier - my_frontier) * 50
    
    def _count_frontier_discs(self, mask: int, empty_mask: int) -> int:
        """특정 색깔의 프론티어 디스크 수 계산"""
        frontier_count = 0
        
        for idx in BitBoard._iter_bits(mask):
            x, y = divmod(idx, 8)
            # 8방향 중 하나라도 빈칸과 인접하면 프론티어
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    nidx = (nx << 3) | ny
                    if empty_mask & (1 << nidx):
                        frontier_count += 1
                        break
        
        return frontier_count
    
    def _evaluate_corner_safety(self, bb: BitBoard, ai_color: int) -> float:
        """코너 안전성 평가"""
        my_mask = bb.black if ai_color == BLACK else bb.white
        op_mask = bb.white if ai_color == BLACK else bb.black
        
        score = 0
        for x, y in self.corners:
            pos = 1 << ((x << 3) | y)
            if my_mask & pos:
                score += 1000  # 코너 확보
                # 코너 주변 안정성도 보너스
                score += self._evaluate_corner_stability(bb, (x, y), ai_color)
            elif op_mask & pos:
                score -= 1000
        
        # X 스퀘어와 C 스퀘어 위험도 평가
        score += self._evaluate_dangerous_squares(bb, ai_color)
        
        return score
    
    def _evaluate_corner_stability(self, bb: BitBoard, corner: tuple, ai_color: int) -> float:
        """코너에서 시작하는 안정성 평가"""
        my_mask = bb.black if ai_color == BLACK else bb.white
        stable_score = 0
        
        x, y = corner
        # 코너에서 변을 따라 연속된 내 돌들은 안정
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if abs(dx) + abs(dy) != 1:
                continue
            
            nx, ny = x + dx, y + dy
            consecutive = 0
            while 0 <= nx < 8 and 0 <= ny < 8:
                pos = 1 << ((nx << 3) | ny)
                if my_mask & pos:
                    consecutive += 1
                    nx += dx
                    ny += dy
                else:
                    break
            
            stable_score += consecutive * 100
        
        return stable_score
    
    def _evaluate_dangerous_squares(self, bb: BitBoard, ai_color: int) -> float:
        """X 스퀘어, C 스퀘어 위험도 평가"""
        my_mask = bb.black if ai_color == BLACK else bb.white
        op_mask = bb.white if ai_color == BLACK else bb.black
        empty_mask = bb.get_empty_mask()
        
        score = 0
        
        # X 스퀘어 평가
        for x, y in self.x_squares:
            pos = 1 << ((x << 3) | y)
            corner_x = 0 if x == 1 else 7
            corner_y = 0 if y == 1 else 7
            corner_pos = 1 << ((corner_x << 3) | corner_y)
            
            # 해당 코너가 비어있을 때만 X 스퀘어가 위험
            if empty_mask & corner_pos:
                if my_mask & pos:
                    score -= 300  # 내가 X 스퀘어에 두면 위험
                elif empty_mask & pos:
                    # 상대방이 X 스퀘어에 둘 수 있는지 확인
                    if self._can_move_to(bb, (x, y), opponent(ai_color)):
                        score += 150  # 상대방이 X 스퀘어에 두게 만들면 유리
        
        # C 스퀘어 평가도 유사하게
        for x, y in self.c_squares:
            pos = 1 << ((x << 3) | y)
            # C 스퀘어와 인접한 코너 찾기
            corner_pos = self._get_adjacent_corner(x, y)
            if corner_pos and empty_mask & corner_pos:
                if my_mask & pos:
                    score -= 200
                elif empty_mask & pos and self._can_move_to(bb, (x, y), opponent(ai_color)):
                    score += 100
        
        return score
    
    def _evaluate_stability(self, bb: BitBoard, ai_color: int) -> float:
        """안정성 평가 - 뒤집힐 수 없는 돌들"""
        my_mask = bb.black if ai_color == BLACK else bb.white
        op_mask = bb.white if ai_color == BLACK else bb.black
        
        my_stable = self._count_stable_discs(my_mask, op_mask)
        op_stable = self._count_stable_discs(op_mask, my_mask)
        
        return (my_stable - op_stable) * 100
    
    def _count_stable_discs(self, my_mask: int, op_mask: int) -> int:
        """안정한 돌의 개수 계산"""
        stable_count = 0
        occupied = my_mask | op_mask
        
        for idx in BitBoard._iter_bits(my_mask):
            x, y = divmod(idx, 8)
            if self._is_stable_disc(x, y, my_mask, occupied):
                stable_count += 1
        
        return stable_count
    
    def _is_stable_disc(self, x: int, y: int, my_mask: int, occupied: int) -> bool:
        """특정 돌이 안정한지 확인"""
        # 코너는 항상 안정
        if (x, y) in self.corners:
            return True
        
        # 변에 있고 양쪽이 안정하면 안정
        if (x, y) in self.edges:
            return self._is_edge_stable(x, y, my_mask, occupied)
        
        # 8방향 모두에서 안정한지 확인 (단순화된 버전)
        stable_directions = 0
        for dx, dy in self.directions:
            if self._is_direction_stable(x, y, dx, dy, my_mask, occupied):
                stable_directions += 1
        
        return stable_directions >= 6  # 대부분 방향에서 안정하면 안정한 것으로 간주
    
    def _evaluate_edge_control(self, bb: BitBoard, ai_color: int) -> float:
        """변 컨트롤 평가"""
        my_mask = bb.black if ai_color == BLACK else bb.white
        op_mask = bb.white if ai_color == BLACK else bb.black
        
        score = 0
        # 각 변별로 컨트롤 평가
        edges_data = [
            [(0, i) for i in range(8)],  # 상단
            [(7, i) for i in range(8)],  # 하단
            [(i, 0) for i in range(8)],  # 좌측
            [(i, 7) for i in range(8)]   # 우측
        ]
        
        for edge in edges_data:
            my_edge_count = 0
            op_edge_count = 0
            
            for x, y in edge:
                pos = 1 << ((x << 3) | y)
                if my_mask & pos:
                    my_edge_count += 1
                elif op_mask & pos:
                    op_edge_count += 1
            
            # 변을 많이 컨트롤할수록 좋음
            score += (my_edge_count - op_edge_count) * 25
        
        return score
    
    def _evaluate_parity(self, empty_count: int, ai_color: int) -> float:
        """패리티 평가 - 마지막에 둘 수 있는 플레이어가 유리"""
        # 홀수면 흑이 마지막, 짝수면 백이 마지막
        last_player = BLACK if empty_count % 2 == 1 else WHITE
        return 200 if ai_color == last_player else -200
    
    # 유틸리티 메서드들
    def _can_move_to(self, bb: BitBoard, pos: tuple, color: int) -> bool:
        """특정 위치에 특정 색이 둘 수 있는지 확인"""
        valid_moves = bb.get_valid_moves_bitboard(color)
        return pos in valid_moves
    
    def _get_adjacent_corner(self, x: int, y: int) -> int:
        """C 스퀘어와 인접한 코너의 비트마스크 반환"""
        corner_map = {
            (0, 1): (0, 0), (1, 0): (0, 0),  # 좌상단 코너
            (0, 6): (0, 7), (1, 7): (0, 7),  # 우상단 코너  
            (6, 0): (7, 0), (7, 1): (7, 0),  # 좌하단 코너
            (6, 7): (7, 7), (7, 6): (7, 7)   # 우하단 코너
        }
        
        if (x, y) in corner_map:
            cx, cy = corner_map[(x, y)]
            return 1 << ((cx << 3) | cy)
        return 0
    
    def _is_edge_stable(self, x: int, y: int, my_mask: int, occupied: int) -> bool:
        """변의 돌이 안정한지 확인 (단순화된 버전)"""
        # 실제로는 더 복잡한 로직이 필요하지만 여기서는 단순화
        return True  # 변의 돌은 일단 안정한 것으로 간주
    
    def _is_direction_stable(self, x: int, y: int, dx: int, dy: int, my_mask: int, occupied: int) -> bool:
        """특정 방향에서 안정한지 확인"""
        # 한 방향으로 쭉 가면서 내 돌이거나 경계에 닿으면 안정
        nx, ny = x + dx, y + dy
        while 0 <= nx < 8 and 0 <= ny < 8:
            pos = 1 << ((nx << 3) | ny)
            if not (occupied & pos):  # 빈칸이면 불안정
                return False
            if not (my_mask & pos):  # 상대 돌이면 불안정
                return False
            nx += dx
            ny += dy
        return True
    
    # 게임 단계별 가중치 조정
    def _get_position_weight(self, stage: str) -> float:
        weights = {"opening": 0.8, "midgame": 1.2, "endgame": 0.6}
        return weights.get(stage, 1.0)
    
    def _get_mobility_weight(self, stage: str) -> float:
        weights = {"opening": 1.5, "midgame": 2.0, "endgame": 0.8}
        return weights.get(stage, 1.0)
    
    def _get_frontier_weight(self, stage: str) -> float:
        weights = {"opening": 2.0, "midgame": 1.8, "endgame": 0.5}
        return weights.get(stage, 1.0)
    
    def _get_corner_weight(self, stage: str) -> float:
        weights = {"opening": 1.0, "midgame": 1.5, "endgame": 2.0}
        return weights.get(stage, 1.0)
    
    def _get_stability_weight(self, stage: str) -> float:
        weights = {"opening": 0.5, "midgame": 1.0, "endgame": 1.5}
        return weights.get(stage, 1.0)
    
    def _get_edge_weight(self, stage: str) -> float:
        weights = {"opening": 0.8, "midgame": 1.2, "endgame": 1.0}
        return weights.get(stage, 1.0)


class SearchEngine:
    def __init__(self, evaluator: Evaluator, tt_size: int, time_limit: float):
        self.evaluator = evaluator
        self.tt: Dict[int, TTEntry] = {}
        self.tt_age = 0
        self.max_tt_size = tt_size
        self.killer_moves: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.history_table = np.zeros((8, 8), dtype=np.int32)
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.time_limit = time_limit

    @staticmethod
    def bit_pos_hash(bb: BitBoard) -> int:
        return int(getattr(bb, 'hash_base', 0))

    @staticmethod
    def bit_hash(bb: BitBoard, side_to_move: int) -> int:
        base = np.uint64(SearchEngine.bit_pos_hash(bb))
        return int(base ^ (ZOBRIST_TURN if side_to_move == BLACK else np.uint64(0)))

    def _static_move_value_bb(self, move: Tuple[int, int]) -> int:
        x, y = move
        if move in CORNERS:
            return 1000
        if move in X_SQUARES:
            return -200
        if move in C_SQUARES:
            return -200
        if x == 0 or x == 7 or y == 0 or y == 7:
            return 100
        return 0

    def enhanced_move_ordering_bb(
        self, ai_color: int, bb: BitBoard, moves: List[Tuple[int, int]], depth: int, prev_best: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[int, int]]:
        if not moves:
            return []
        move_scores: List[Tuple[int, int, Tuple[int, int]]] = []
        tt_entry = self.tt.get(self.bit_hash(bb, ai_color))
        tt_move = tt_entry.move if tt_entry else None
        total_history = max(1, int(np.sum(self.history_table)))
        own = bb.black if ai_color == BLACK else bb.white
        opp = bb.white if ai_color == BLACK else bb.black

        for i, move in enumerate(moves):
            x, y = move
            s = 0
            if prev_best and move == prev_best:
                s += 10_000
            if tt_move and move == tt_move:
                s += 8_000
            killers = self.killer_moves.get(depth, [])
            if move in killers:
                s += 4_000 + (2 - killers.index(move)) * 1000
            s += (int(self.history_table[x][y]) * 2000) // total_history
            s += self._static_move_value_bb(move)

            idx = (x << 3) | y
            flips = bb._flip_mask_for_move(own, opp, 1 << idx)
            s += (flips.bit_count()) * 50

            move_scores.append((s, i, move))

        move_scores.sort(reverse=True)
        return [m for _, _, m in move_scores]

    def alpha_beta_bb(
        self,
        ai_color: int,
        bb: BitBoard,
        depth: int,
        alpha: int,
        beta: int,
        side_to_move: int,
        start_time: float,
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        self.nodes_searched += 1
        # Time guard: return static if near budget end
        if (self.nodes_searched & 0xFFF) == 0:
            if time.time() - start_time > self.time_limit * 0.95:
                return self.evaluator.evaluate_bitboard(bb, ai_color), None

        key = self.bit_hash(bb, side_to_move)
        tt_entry = self.tt.get(key)
        tt_move: Optional[Tuple[int, int]] = None
        if tt_entry and tt_entry.depth >= depth and self.bit_pos_hash(bb) == tt_entry.pos_hash:
            self.tt_hits += 1
            if tt_entry.node_type == 'exact':
                return tt_entry.score, tt_entry.move
            if tt_entry.node_type == 'lowerbound' and tt_entry.score >= beta:
                return tt_entry.score, tt_entry.move
            if tt_entry.node_type == 'upperbound' and tt_entry.score <= alpha:
                return tt_entry.score, tt_entry.move
            tt_move = tt_entry.move

        moves = bb.get_valid_moves_bitboard(side_to_move)

        if depth == 0 or not moves:
            if not moves:
                opp = opponent(side_to_move)
                if not bb.get_valid_moves_bitboard(opp):
                    bcnt = BitBoard.popcount(bb.black)
                    wcnt = BitBoard.popcount(bb.white)
                    diff = (bcnt - wcnt) if ai_color == BLACK else (wcnt - bcnt)
                    return MATE_SCORE * (1 if diff > 0 else -1 if diff < 0 else 0), None
                return self.alpha_beta_bb(ai_color, bb, depth, alpha, beta, opp, start_time)
            return self.evaluator.evaluate_bitboard(bb, ai_color), None

        ordered = self.enhanced_move_ordering_bb(ai_color, bb, moves, depth, tt_move)
        best_move = ordered[0]
        best_score = -INF_SCORE
        original_alpha = alpha
        opp_color = opponent(side_to_move)

        for i, move in enumerate(ordered):
            x, y = move
            next_bb = bb.apply_move_bitboard(x, y, side_to_move)
            if next_bb is None:
                continue

            if i == 0:
                score, _ = self.alpha_beta_bb(ai_color, next_bb, depth - 1, -beta, -alpha, opp_color, start_time)
                score = -score
            else:
                score, _ = self.alpha_beta_bb(ai_color, next_bb, depth - 1, -alpha - 1, -alpha, opp_color, start_time)
                score = -score
                if alpha < score < beta:
                    score, _ = self.alpha_beta_bb(ai_color, next_bb, depth - 1, -beta, -score, opp_color, start_time)
                    score = -score

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

            if alpha >= beta:
                # cutoff bookkeeping
                self.cutoffs += 1
                kl = self.killer_moves[depth]
                if move not in kl:
                    if len(kl) >= 2:
                        kl.pop(0)
                    kl.append(move)
                hx, hy = move
                self.history_table[hx][hy] += depth * depth
                break

        node_type = 'exact'
        if best_score <= original_alpha:
            node_type = 'upperbound'
        elif best_score >= beta:
            node_type = 'lowerbound'
        if not math.isfinite(best_score):
            best_score = MATE_SCORE if best_score > 0 else -MATE_SCORE

        if len(self.tt) >= self.max_tt_size:
            try:
                evict_key = min(self.tt, key=lambda k: (self.tt[k].age, self.tt[k].depth))
                self.tt.pop(evict_key, None)
            except ValueError:
                pass

        self.tt[key] = TTEntry(
            score=int(best_score),
            move=best_move,
            depth=depth,
            node_type=node_type,
            age=self.tt_age,
            pos_hash=self.bit_pos_hash(bb),
        )
        return int(best_score), best_move

    def _root_search(
        self,
        bb: BitBoard,
        ai_color: int,
        depth: int,
        start_time: float,
        root_moves: Optional[List[Tuple[int, int]]] = None,
        prev_best: Optional[Tuple[int, int]] = None,
        aspiration: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        # Order root moves
        moves = bb.get_valid_moves_bitboard(ai_color)
        if root_moves is not None:
            allowed = set(root_moves)
            moves = [m for m in moves if m in allowed]
        if not moves:
            return self.evaluator.evaluate_bitboard(bb, ai_color), None

        ordered = self.enhanced_move_ordering_bb(ai_color, bb, moves, depth, prev_best)
        alpha, beta = (-INF_SCORE, INF_SCORE) if not aspiration else aspiration
        opp = opponent(ai_color)

        best_move = ordered[0]
        best_score = -INF_SCORE
        original_alpha = alpha

        for i, move in enumerate(ordered):
            x, y = move
            next_bb = bb.apply_move_bitboard(x, y, ai_color)
            if next_bb is None:
                continue

            if i == 0:
                score, _ = self.alpha_beta_bb(ai_color, next_bb, depth - 1, -beta, -alpha, opp, start_time)
                score = -score
            else:
                score, _ = self.alpha_beta_bb(ai_color, next_bb, depth - 1, -alpha - 1, -alpha, opp, start_time)
                score = -score
                if alpha < score < beta:
                    score, _ = self.alpha_beta_bb(ai_color, next_bb, depth - 1, -beta, -score, opp, start_time)
                    score = -score

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        # store PV root to TT as well (respect TT size cap)
        key = self.bit_hash(bb, ai_color)
        node_type = 'exact'
        if best_score <= original_alpha:
            node_type = 'upperbound'
        elif best_score >= beta:
            node_type = 'lowerbound'
        if len(self.tt) >= self.max_tt_size:
            try:
                evict_key = min(self.tt, key=lambda k: (self.tt[k].age, self.tt[k].depth))
                self.tt.pop(evict_key, None)
            except ValueError:
                pass
        self.tt[key] = TTEntry(
            score=int(best_score),
            move=best_move,
            depth=depth,
            node_type=node_type,
            age=self.tt_age,
            pos_hash=self.bit_pos_hash(bb),
        )
        return int(best_score), best_move

    def get_best_move(
        self,
        board: object,
        ai_color: int,
        max_depth: int,
        root_moves: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        start_time = time.time()
        bb = BitBoard(board)
        moves = bb.get_valid_moves_bitboard(ai_color)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        best_move: Optional[Tuple[int, int]] = None
        prev_score = 0
        aspiration = 50

        for depth in range(1, max_depth + 1):
            try:
                self.nodes_searched = 0
                self.tt_hits = 0
                self.cutoffs = 0
                self.tt_age += 1
                depth_start = time.time()

                if depth >= 4 and best_move:
                    alpha = prev_score - aspiration
                    beta = prev_score + aspiration
                    score, move = self._root_search(bb, ai_color, depth, start_time, root_moves, best_move, (alpha, beta))
                    if score <= alpha:
                        score, move = self._root_search(bb, ai_color, depth, start_time, root_moves, best_move, (-INF_SCORE, beta))
                    elif score >= beta:
                        score, move = self._root_search(bb, ai_color, depth, start_time, root_moves, best_move, (alpha, INF_SCORE))
                    aspiration = min(100, aspiration + 25)
                else:
                    score, move = self._root_search(bb, ai_color, depth, start_time, root_moves, best_move, None)

                if move:
                    best_move = move
                    prev_score = score

                depth_time = time.time() - depth_start
                logging.info(f"[Search] Depth {depth}: {depth_time:.3f}s, nodes={self.nodes_searched}, TT={self.tt_hits}, Score={score}, Move={move}")

                if math.isfinite(score) and abs(score) >= MATE_SCORE:
                    logging.info("Terminal score - stopping")
                    break
                if time.time() - start_time > self.time_limit * 0.8:
                    logging.info("Time budget nearly used - stop")
                    break
            except Exception as e:
                logging.error(f"[Search] Error at depth {depth}: {e}")
                break

        return best_move


class UltraAdvancedAI:
    """Corner-first Othello AI with 2-ply corner exposure avoidance."""

    def __init__(self, color: int, difficulty: str = 'hard', time_limit: float = 10.0) -> None:
        self.color = color
        self.difficulty = difficulty

        # Defaults
        self.max_tt_size = 2**20
        self.time_limit = float(time_limit)

        # Pattern / positional weights (kept minimal but adjustable)
        self._init_pos_weights()

        # Difficulty presets
        if difficulty == 'easy':
            self.max_depth = 6
        elif difficulty == 'medium':
            self.max_depth = 8
        else:
            self.max_depth = 10

        # Two-ply corner avoidance config
        self.two_ply_check_empties = 20
        self.two_ply_opp_limit = 6
        self.two_ply_our_limit = 6

        # Corner maps for fast checks
        self.X_TO_CORNER = {(1, 1): (0, 0), (1, 6): (0, 7), (6, 1): (7, 0), (6, 6): (7, 7)}
        self.C_TO_CORNER = {
            (0, 1): (0, 0), (1, 0): (0, 0),
            (0, 6): (0, 7), (1, 7): (0, 7),
            (7, 1): (7, 0), (6, 0): (7, 0),
            (7, 6): (7, 7), (6, 7): (7, 7),
        }

        # Config overrides
        cfg = load_config()
        self.config = cfg
        self.max_tt_size = cfg.get('tt_size', self.max_tt_size)
        self.time_limit = cfg.get('time_limit', self.time_limit)
        if 'difficulties' in cfg and isinstance(cfg['difficulties'], dict):
            d = cfg['difficulties'].get(self.difficulty, {})
            self.max_depth = d.get('max_depth', self.max_depth)

        # Engine
        self.evaluator = Evaluator()
        self.search_engine = SearchEngine(self.evaluator, self.max_tt_size, self.time_limit)

        # Small opening book & endgame cache hooks (placeholders)
        self.opening_book: Dict[int, Tuple[int, int]] = {}
        self.endgame_cache: Dict[int, Tuple[int, Optional[Tuple[int, int]]]] = {}

    # ---- position weights tune ----
    def _init_pos_weights(self) -> None:
        for c in CORNERS:
            adjust_position_weight(c, 500)
        for x in X_SQUARES:
            adjust_position_weight(x, -160, stages=('early', 'mid'))
        for c in C_SQUARES:
            adjust_position_weight(c, -100, stages=('mid',))

    # ---- bitboard helpers for corner safety ----
    @staticmethod
    def _has_corner_move_bb(bb: BitBoard, side: int) -> bool:
        return any(m in CORNERS for m in bb.get_valid_moves_bitboard(side))

    def _is_corner_exposing_move_bb(self, bb: BitBoard, move: Tuple[int, int], side: int) -> bool:
        if move in CORNERS:
            return False
        # Immediate corner for opponent after our move?
        nb = bb.apply_move_bitboard(move[0], move[1], side)
        if nb is None:
            return False
        return self._has_corner_move_bb(nb, opponent(side))

    @staticmethod
    def _flip_count_for(bb: BitBoard, move: Tuple[int, int], side: int) -> int:
        own = bb.black if side == BLACK else bb.white
        opp = bb.white if side == BLACK else bb.black
        idx = (move[0] << 3) | move[1]
        flips = bb._flip_mask_for_move(own, opp, 1 << idx)
        return flips.bit_count()

    def _exposes_corner_in_two_bb(
        self,
        bb: BitBoard,
        move: Tuple[int, int],
        side: int,
        opp_limit: int,
        our_limit: int,
    ) -> bool:
        """Returns True if opponent can force an immediate corner in 2 plies after this move."""
        if move in CORNERS:
            return False
        nb = bb.apply_move_bitboard(move[0], move[1], side)
        if nb is None:
            return False
        opp = opponent(side)

        opp_moves = nb.get_valid_moves_bitboard(opp)
        if any(m in CORNERS for m in opp_moves):
            return True  # already gives them a corner next move

        # Prioritize opponent replies: proximity to empty corner + flip count
        def opp_priority(m: Tuple[int, int]) -> int:
            s = 0
            # bonus if X/C near an empty corner
            if m in self.X_TO_CORNER:
                cx, cy = self.X_TO_CORNER[m]
                if ((nb.black | nb.white) >> ((cx << 3) | cy)) & 1 == 0:
                    s += 500
            if m in self.C_TO_CORNER:
                cx, cy = self.C_TO_CORNER[m]
                if ((nb.black | nb.white) >> ((cx << 3) | cy)) & 1 == 0:
                    s += 200
            s += self._flip_count_for(nb, m, opp) * 10
            return s

        opp_moves = sorted(opp_moves, key=opp_priority, reverse=True)[:opp_limit]
        for om in opp_moves:
            nb2 = nb.apply_move_bitboard(om[0], om[1], opp)
            if nb2 is None:
                continue
            our_replies = nb2.get_valid_moves_bitboard(side)
            if not our_replies:
                # pass; if they then have a corner, it's effectively a trap
                if self._has_corner_move_bb(nb2, opp):
                    return True
                continue

            def our_priority(m: Tuple[int, int]) -> int:
                return self._flip_count_for(nb2, m, side)

            parried = False
            for rm in sorted(our_replies, key=our_priority, reverse=True)[:our_limit]:
                nb3 = nb2.apply_move_bitboard(rm[0], rm[1], side)
                if nb3 is None:
                    continue
                if not self._has_corner_move_bb(nb3, opp):
                    parried = True
                    break
            if not parried:
                return True
        return False

    # ---- public API ----
    def get_move(self, board: object) -> Optional[Tuple[int, int]]:
        bb = BitBoard(board)
        side = self.color
        moves = bb.get_valid_moves_bitboard(side)
        if not moves:
            return None

        # Corner-first: instant take
        for m in moves:
            if m in CORNERS:
                logging.info(f"[Corner-Safe AI] Corner available → taking {m}")
                return m

        empties = 64 - BitBoard.popcount(bb.black | bb.white)
        # 1-ply corner exposure filtering
        safe1 = [m for m in moves if not self._is_corner_exposing_move_bb(bb, m, side)]
        candidates = safe1 if safe1 else moves

        # 2-ply trap filtering when lots of empties
        if empties > self.two_ply_check_empties and candidates:
            safer2 = [
                m for m in candidates
                if not self._exposes_corner_in_two_bb(bb, m, side, self.two_ply_opp_limit, self.two_ply_our_limit)
            ]
            if safer2:
                candidates = safer2

        # Let the engine search, restricted to root candidates
        move = self.search_engine.get_best_move(board, side, self.max_depth, root_moves=candidates)
        return move

    # ---- standardized AI interface ----
    def set_difficulty(self, level: str) -> None:
        """Set difficulty profile and adjust max depth accordingly."""
        self.difficulty = level
        # Default mapping in case config lacks profile
        default_map = {'easy': 6, 'medium': 8, 'hard': 10}
        depth = default_map.get(level, self.max_depth)
        dcfg = self.config.get('difficulties', {})
        if isinstance(dcfg, dict):
            depth = dcfg.get(level, {}).get('max_depth', depth)
        self.max_depth = int(depth)

    def set_time_limit(self, seconds: float) -> None:
        """Update time budget and propagate to the search engine."""
        try:
            self.time_limit = float(seconds)
        except Exception:
            return
        # Keep config in sync for callers that re-read it later
        self.config['time_limit'] = self.time_limit
        self.search_engine.time_limit = self.time_limit
