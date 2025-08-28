"""
calculator.py — Single-file Scientific Calculator with a Tkinter UI

- Clean, intuitive GUI (Tkinter)
- Basic ops: +, -, ×, ÷, //, %, **, parentheses
- Advanced: sqrt, cbrt, root(x,n), exp, logs, trig (deg/rad), inverse trig,
  hyperbolics, factorial, gcd, lcm, rounding and more
- Safe AST-based evaluation (no eval/exec), strict whitelist of nodes/names
- Helpful errors (division by zero, domain errors, syntax, etc.)
- Input validation and result formatting with configurable significant digits
- Extensible: add constants & functions in CalculatorEngine.build_env()
- PEP 8 compliant, standard library only (works on Python 3.10+)

Shortcuts:
  Enter            evaluate
  Esc              clear
  Ctrl+Backspace   clear entry
  Backspace        delete char
"""
