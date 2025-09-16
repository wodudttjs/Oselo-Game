#!/usr/bin/env python3
"""
🏆 ULTIMATE OTHELLO AI - THE UNBEATABLE CHAMPION 🏆

This is the ultimate version of our Othello AI project, featuring:

🚀 ULTRA STRONG AI:
- Perfect endgame solver (16+ empties)
- 18-ply deep search with aspiration windows
- 5M position transposition table
- Advanced pattern recognition
- Tournament-level opening book
- Late Move Reduction (LMR)
- Multi-layered evaluation function

⚡ EGAROUCID-STYLE AI:
- Inspired by world champion Egaroucid
- Multi-ProbCut pruning
- Iterative deepening
- Killer moves and history heuristic
- Game-phase adaptive evaluation

🎯 ADVANCED AI:
- Classic strong AI with modern techniques
- Alpha-beta pruning
- Move ordering optimization
- Strategic position evaluation

Created by: Your AI Development Team
Version: Ultimate Edition v1.0
Date: 2024

Ready to face the ultimate challenge? Good luck! 💀
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

def check_dependencies():
    """Check if all required files are present"""
    required_files = [
        'constants.py',
        'board.py', 
        'ai.py',
        'egaroucid_ai.py',
        'ultra_strong_ai.py',
        'ultimate_gui.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        error_msg = "❌ Missing required files:\n\n"
        for file in missing_files:
            error_msg += f"• {file}\n"
        error_msg += "\nPlease ensure all files are in the same directory."
        
        messagebox.showerror("Missing Files", error_msg)
        return False
    
    return True

def show_welcome_message():
    """Show epic welcome message"""
    welcome_msg = """
🏆 WELCOME TO ULTIMATE OTHELLO AI 🏆

You are about to face the most advanced Othello AI ever created!

🚀 ULTRA STRONG AI Features:
✓ Perfect endgame solver
✓ 18-ply deep search  
✓ 5 million position database
✓ Advanced pattern recognition
✓ Tournament opening book
✓ Multi-layered evaluation

⚠️ WARNING: This AI is EXTREMELY difficult to beat!

Are you ready for the ultimate challenge?
"""
    
    response = messagebox.askyesno("🎮 ULTIMATE CHALLENGE", welcome_msg)
    return response

def main():
    """Main entry point"""
    print("🏆 STARTING ULTIMATE OTHELLO AI...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # Import GUI (after dependency check)
        from ultimate_gui import UltimateOthelloGUI
        
        # Create main window
        root = tk.Tk()
        
        # Show welcome message
        if not show_welcome_message():
            print("👋 Challenge declined. Maybe next time!")
            sys.exit(0)
        
        # Configure window
        root.configure(bg="#0f0f23")
        
        # Set icon (if available)
        try:
            # You can add an icon file here
            # root.iconbitmap("othello_icon.ico")
            pass
        except:
            pass
        
        # Create and run the application
        print("🎮 Launching Ultimate Othello AI...")
        app = UltimateOthelloGUI(root)
        
        print("🚀 Game interface ready!")
        print("💡 Tips for playing against Ultra AI:")
        print("   • Focus on corner control")
        print("   • Limit opponent mobility") 
        print("   • Avoid X-squares near empty corners")
        print("   • Think several moves ahead")
        print("   • Good luck... you'll need it! 😈")
        print("=" * 50)
        
        # Start the GUI event loop
        root.mainloop()
        
    except ImportError as e:
        error_msg = f"❌ Import Error: {e}\n\n"
        error_msg += "Please ensure all Python files are in the same directory\n"
        error_msg += "and that there are no syntax errors in the code."
        messagebox.showerror("Import Error", error_msg)
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"❌ Unexpected Error: {e}\n\n"
        error_msg += "Please check the console for more details."
        messagebox.showerror("Error", error_msg)
        print(f"Error details: {e}")
        sys.exit(1)
    
    print("👋 Thanks for playing Ultimate Othello AI!")

if __name__ == "__main__":
    main()