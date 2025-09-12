# 플레이어 색상 정의
EMPTY, BLACK, WHITE = 0, 1, 2  # 숫자로 변경하여 일관성 확보

# 게임 단계별 평가 가중치
EARLY_WEIGHTS = [
    [150, -40, 20,  5,  5, 20, -40, 150],
    [-40, -80, -5, -5, -5, -5, -80, -40],
    [20,   -5, 15,  3,  3, 15,  -5,  20],
    [5,    -5,  3,  0,  0,  3,  -5,   5],
    [5,    -5,  3,  0,  0,  3,  -5,   5],
    [20,   -5, 15,  3,  3, 15,  -5,  20],
    [-40, -80, -5, -5, -5, -5, -80, -40],
    [150, -40, 20,  5,  5, 20, -40, 150],
]

MID_WEIGHTS = [
    [120, -30, 25, 10, 10, 25, -30, 120],
    [-30, -60,  0,  0,  0,  0, -60, -30],
    [25,    0, 20, 10, 10, 20,   0,  25],
    [10,    0, 10,  5,  5, 10,   0,  10],
    [10,    0, 10,  5,  5, 10,   0,  10],
    [25,    0, 20, 10, 10, 20,   0,  25],
    [-30, -60,  0,  0,  0,  0, -60, -30],
    [120, -30, 25, 10, 10, 25, -30, 120],
]


LATE_WEIGHTS = [
    [200, 50,  50,  30,  30, 50,  50, 200],
    [50,  10,  20,  15,  15, 20,  10,  50],
    [50,  20,  30,  25,  25, 30,  20,  50],
    [30,  15,  25,  10,  10, 25,  15,  30],
    [30,  15,  25,  10,  10, 25,  15,  30],
    [50,  20,  30,  25,  25, 30,  20,  50],
    [50,  10,  20,  15,  15, 20,  10,  50],
    [200, 50,  50,  30,  30, 50,  50, 200],
]
def adjust_position_weight(pos, value, stages=('early', 'mid', 'late')):
    """특정 좌표의 가중치를 수정"""
    x, y = pos
    if 'early' in stages:
        EARLY_WEIGHTS[x][y] = value
        EARLY_WEIGHTS[7 - x][y] = value
        EARLY_WEIGHTS[x][7 - y] = value
        EARLY_WEIGHTS[7 - x][7 - y] = value
    if 'mid' in stages:
        MID_WEIGHTS[x][y] = value
        MID_WEIGHTS[7 - x][y] = value
        MID_WEIGHTS[x][7 - y] = value
        MID_WEIGHTS[7 - x][7 - y] = value
    if 'late' in stages:
        LATE_WEIGHTS[x][y] = value
        LATE_WEIGHTS[7 - x][y] = value
        LATE_WEIGHTS[x][7 - y] = value
        LATE_WEIGHTS[7 - x][7 - y] = value

# 전략적으로 중요한 위치
X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]
C_SQUARES = [(0, 1), (1, 0), (0, 6), (6, 0), (7, 1), (6, 7), (7, 6), (1, 7)]
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
EDGES = [(i, 0) for i in range(8)] + [(i, 7) for i in range(8)] + [(0, i) for i in range(8)] + [(7, i) for i in range(8)]




# 상대 색상 구하기 함수
def opponent(color):
    return BLACK if color == WHITE else WHITE