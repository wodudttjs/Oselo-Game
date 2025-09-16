import sys
try:
    import ai
    from ai import UltraAdvancedAI
    print("OK: UltraAdvancedAI imported")
except Exception as e:
    print("IMPORT_ERROR:", repr(e))
    sys.exit(1)

