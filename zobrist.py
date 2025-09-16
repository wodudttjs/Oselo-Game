import numpy as np

# Deterministic Zobrist tables for reproducibility
_rng = np.random.default_rng(123456789)

# Zobrist random numbers for [row][col][piece], where piece in {0: EMPTY, 1: BLACK, 2: WHITE}
ZOBRIST_TABLE = _rng.integers(1, 2**63, size=(8, 8, 3), dtype=np.uint64)

# Side-to-move toggle value (XOR this to include turn in the key)
ZOBRIST_TURN = np.uint64(_rng.integers(1, 2**63, dtype=np.uint64))

