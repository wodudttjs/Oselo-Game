Othello (Reversi) with Tkinter GUI and Advanced AI

Overview
- Tkinter-based GUI (`main.py`, `gui.py`).
- Core game logic (`board.py`, `constants.py`).
- Heuristic AI with alpha–beta and extras (`ai.py`).

Requirements
- Python 3.9+ with Tkinter included (Windows/Mac installers usually include it).
- `numpy` for AI internals.

Setup (Windows PowerShell)
1. Create a virtual environment:
   - `python -m venv .venv`
2. Activate it:
   - `.venv\Scripts\Activate`
3. Install dependencies:
   - `pip install -r requirements.txt`

Run
- `python main.py`

Notes
- The app prompts for your color (Black goes first).
- If you get a Tkinter error, install a Python build that includes Tk support.
- The AI can take longer in late game on hard difficulty.

