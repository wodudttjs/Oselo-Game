# 플레이어 색상 정의
EMPTY, BLACK, WHITE = 0, 1, 2  # 숫자로 변경하여 일관성 확보

# 게임 단계별 평가 가중치


# 전략적으로 중요한 위치
X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]
C_SQUARES = [(0, 1), (1, 0), (0, 6), (6, 0), (7, 1), (6, 7), (7, 6), (1, 7)]
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
EDGES = [(i, 0) for i in range(8)] + [(i, 7) for i in range(8)] + [(0, i) for i in range(8)] + [(7, i) for i in range(8)]

# 오프닝북 (기초 수 순서)
OPENING_BOOK = {
    (BLACK, 3, 3, WHITE, 3, 4, BLACK, 4, 3, WHITE, 4, 4): [(2, 3), (3, 2), (4, 5), (5, 4)],
    (BLACK, 3, 3, WHITE, 3, 4, BLACK, 4, 3, WHITE, 4, 4, BLACK, 2, 3): [(3, 2), (5, 4)],
    (BLACK, 3, 3, WHITE, 3, 4, BLACK, 4, 3, WHITE, 4, 4, BLACK, 3, 2): [(2, 3), (4, 5)],
}

# 상대 색상 구하기 함수
def opponent(color):
    return BLACK if color == WHITE else WHITE