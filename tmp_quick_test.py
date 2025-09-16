from board import Board
from ai import UltraAdvancedAI

b = Board()
ai = UltraAdvancedAI(color=2, difficulty='easy', time_limit=0.5)
mv = ai.get_move(b)
print('AI move:', mv)

