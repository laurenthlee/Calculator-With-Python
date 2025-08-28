#!/usr/bin/env python3
"""
Scientific Calculator + Converters (Tkinter)

- Light/Dark • scalable UI • live preview • history • memory • 2nd layer
- Converters: Length, Mass, Area, Volume, Speed, Temperature, Time, Data,
  Numeral System, BMI, Discount, Finance, Date difference, Currency (manual)
- Safe AST evaluator (no eval/exec), degree/radian, precision control

Input guards
- Unified sanitization for keyboard, buttons, paste
- Auto '*' before functions/constants/'(' and also before digits/'.' if they
  follow a value (e.g., ans4 -> ans*4, )3 -> )*3, pi.5 -> pi*0.5)
- Keep '*' ≤ '**'; disallow '//' floor-div; replace operator runs
- Only unary '-' allowed at start or just after '('
- No more ')' than '('; ',' only inside parentheses with a left value
- Expression & preview length limits; empty preview '= 0.00'
"""

from __future__ import annotations

import ast
import math
import datetime as _dt
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import font as tkfont
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

# ============================= Core Engine ==================================

Number = Union[int, float]

class CalculationError(Exception): pass

def is_int_like(x: float, tol: float = 1e-12) -> bool:
    if not math.isfinite(x): return False
    n = round(x); return abs(x - n) <= tol

@dataclass
class Settings:
    angle_mode: str = "rad"   # "rad" or "deg"
    precision: int = 12
    def validate(self) -> None:
        if self.angle_mode not in {"rad", "deg"}:
            raise ValueError("angle_mode must be 'rad' or 'deg'")
        if not (1 <= int(self.precision) <= 15):
            raise ValueError("precision must be 1..15")

class SafeEvaluator:
    _ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    _ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

    def __init__(self, env: Dict[str, Any]) -> None:
        self.env = env

    def eval_expr(self, expr: str) -> Number:
        if not expr or not expr.strip():
            raise CalculationError("Empty expression.")
        if len(expr) > 2000:
            raise CalculationError("Expression too long (limit: 2000 chars).")
        try:
            node = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise CalculationError(f"Syntax error: {exc.msg}") from None
        value = self._eval(node.body)
        if isinstance(value, (bool, complex)) or not isinstance(value, (int, float)):
            raise CalculationError("Unsupported result type.")
        return value

    def _eval(self, node: ast.AST) -> Number:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)): return node.value
            raise CalculationError("Only numeric literals are allowed.")

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self._ALLOWED_UNARYOPS):
            v = self._eval(node.operand)
            return +v if isinstance(node.op, ast.UAdd) else -v

        if isinstance(node, ast.BinOp) and isinstance(node.op, self._ALLOWED_BINOPS):
            l, r = self._eval(node.left), self._eval(node.right)
            try:
                if isinstance(node.op, ast.Add):       return l + r
                if isinstance(node.op, ast.Sub):       return l - r
                if isinstance(node.op, ast.Mult):      return l * r
                if isinstance(node.op, ast.Div):       return l / r
                if isinstance(node.op, ast.FloorDiv):  return l // r
                if isinstance(node.op, ast.Mod):       return l % r
                if isinstance(node.op, ast.Pow):       return l ** r
            except ZeroDivisionError: raise CalculationError("Division by zero.")
            except OverflowError:     raise CalculationError("Overflow during computation.")
            except ValueError as exc: raise CalculationError(str(exc))

        if isinstance(node, ast.Call):
            name = self._get_call_name(node.func)
            func = self.env.get(name)
            if not callable(func): raise CalculationError(f"Unknown function: {name}.")
            if node.keywords: raise CalculationError("Keyword arguments are not supported.")
            args = [self._eval(a) for a in node.args]
            try: return func(*args)
            except ZeroDivisionError: raise CalculationError("Division by zero.")
            except TypeError as exc:   raise CalculationError(f"{name} usage error: {exc}.")
            except ValueError as exc:  raise CalculationError(f"{name} domain error: {exc}.")

        if isinstance(node, ast.Name):
            val = self.env.get(node.id)
            if val is None: raise CalculationError(f"Unknown name: {node.id}.")
            if callable(val): raise CalculationError(f"'{node.id}' is a function; call it like {node.id}(...).")
            return val

        raise CalculationError("Unsupported or unsafe expression construct.")

    @staticmethod
    def _get_call_name(func_node: ast.AST) -> str:
        if isinstance(func_node, ast.Name): return func_node.id
        raise CalculationError("Only simple function calls are allowed (e.g., sin(x)).")

class CalculatorEngine:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings(); self.settings.validate()
        self.last_result: float = 0.0

    def evaluate(self, expr: str) -> Number:
        evaluator = SafeEvaluator(self.build_env())
        result = evaluator.eval_expr(expr)
        if isinstance(result, float) and result == 0.0: result = 0.0
        self.last_result = float(result)
        return result

    def format_number(self, x: Number) -> str:
        xf = float(x)
        if math.isnan(xf):  return "nan"
        if math.isinf(xf):  return "inf" if xf > 0 else "-inf"
        if is_int_like(xf): return str(int(round(xf)))
        n = int(self.settings.precision)
        return f"{xf:.{n}g}"

    def build_env(self) -> Dict[str, Any]:
        s = self.settings
        consts: Dict[str, float] = {"pi": math.pi, "tau": math.tau, "e": math.e,
                                    "phi": (1 + 5 ** 0.5) / 2, "inf": math.inf, "ans": self.last_result}

        def _to_rad(x: float) -> float:  return math.radians(x) if s.angle_mode == "deg" else x
        def _from_rad(x: float) -> float: return math.degrees(x) if s.angle_mode == "deg" else x

        def sqrt(x: float) -> float:
            if x < 0: raise ValueError("sqrt requires x >= 0.")
            return math.sqrt(x)

        def cbrt(x: float) -> float: return math.copysign(abs(x)**(1/3), x)

        def root(x: float, n: float) -> float:
            if n == 0: raise ValueError("n-th root undefined for n = 0.")
            if x < 0:
                if is_int_like(n) and int(round(n)) % 2 == 1: return -((-x) ** (1.0 / float(n)))
                raise ValueError("root(x,n<0) needs odd integer n.")
            return x ** (1.0 / float(n))

        def yroot(y: float, x: float) -> float: return root(x, y)

        def log(x: float, base: Optional[float] = None) -> float:
            if x <= 0: raise ValueError("log requires x > 0.")
            if base is None: return math.log(x)
            if base <= 0 or base == 1.0: raise ValueError("base must be >0 and !=1.")
            return math.log(x, base)

        def ln(x: float) -> float: return log(x)
        def lg(x: float) -> float: return math.log10(x)
        def ten_pow(x: float) -> float: return 10.0 ** x

        def sin(x: float) -> float:  return math.sin(_to_rad(x))
        def cos(x: float) -> float:  return math.cos(_to_rad(x))
        def tan(x: float) -> float:  return math.tan(_to_rad(x))
        def asin(x: float) -> float: return _from_rad(math.asin(x))
        def acos(x: float) -> float: return _from_rad(math.acos(x))
        def atan(x: float) -> float: return _from_rad(math.atan(x))
        def atan2(y: float, x: float) -> float: return _from_rad(math.atan2(y, x))

        def sec(x: float) -> float:
            c = cos(x)
            if c == 0: raise ValueError("sec undefined for cos(x)=0.")
            return 1.0 / c

        def csc(x: float) -> float:
            s_ = sin(x)
            if s_ == 0: raise ValueError("csc undefined for sin(x)=0.")
            return 1.0 / s_

        def cot(x: float) -> float:
            t = tan(x)
            if t == 0: raise ValueError("cot undefined for tan(x)=0.")
            return 1.0 / t

        def fact(n: float) -> int:
            if not is_int_like(n) or n < 0:
                raise ValueError("factorial needs non-negative integer.")
            return math.factorial(int(round(n)))

        def npr(n: float, r: float) -> int:
            if not (is_int_like(n) and is_int_like(r)): raise ValueError("npr requires integers.")
            n_i, r_i = int(round(n)), int(round(r))
            if r_i < 0 or n_i < 0 or r_i > n_i: raise ValueError("require 0 ≤ r ≤ n.")
            return math.perm(n_i, r_i)

        def ncr(n: float, r: float) -> int:
            if not (is_int_like(n) and is_int_like(r)): raise ValueError("ncr requires integers.")
            n_i, r_i = int(round(n)), int(round(r))
            if r_i < 0 or n_i < 0 or r_i > n_i: raise ValueError("require 0 ≤ r ≤ n.")
            return math.comb(n_i, r_i)

        def avg(*args: float) -> float:
            if not args: raise ValueError("avg requires at least one value.")
            return sum(args) / float(len(args))

        def clip(x: float, lo: float, hi: float) -> float:
            if lo > hi: lo, hi = hi, lo
            return max(lo, min(hi, x))

        def sign(x: float) -> float: return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        def inv(x: float) -> float:
            if x == 0: raise ValueError("division by zero.")
            return 1.0 / x
        def sqr(x: float) -> float: return x * x
        def cube(x: float) -> float: return x * x * x
        def pct(x: float) -> float: return x / 100.0
        def mod(a: float, b: float) -> float: return a % b

        env: Dict[str, Any] = dict(consts)
        env.update({
            "pow": pow, "sqrt": sqrt, "cbrt": cbrt, "root": root, "yroot": yroot, "exp": math.exp,
            "sqr": sqr, "cube": cube, "ten_pow": ten_pow,
            "log": log, "ln": ln, "lg": lg, "log10": math.log10, "log2": math.log2,
            "sin": sin, "cos": cos, "tan": tan, "asin": asin, "acos": acos, "atan": atan, "atan2": atan2,
            "sec": sec, "csc": csc, "cot": cot,
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
            "min": min, "max": max, "sum": sum, "avg": avg, "clip": clip, "sign": sign,
            "inv": inv, "pct": pct, "mod": mod,
            "factorial": fact, "fact": fact, "gcd": math.gcd, "lcm": math.lcm, "npr": npr, "ncr": ncr,
            "radians": math.radians, "degrees": math.degrees, "deg2rad": math.radians, "rad2deg": math.degrees,
        })
        return env

# ============================ Small UI helpers ==============================

def _hex_to_rgb(h: str) -> tuple[int,int,int]:
    h = h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
def _rgb_to_hex(r:int,g:int,b:int) -> str: return f"#{r:02x}{g:02x}{b:02x}"
def _mix(c1:str,c2:str,t:float)->str:
    r1,g1,b1=_hex_to_rgb(c1); r2,g2,b2=_hex_to_rgb(c2)
    r=round(r1+(r2-r1)*t); g=round(g1+(g2-g1)*t); b=round(b1+(b2-b1)*t)
    return _rgb_to_hex(r,g,b)

class Tooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget=widget; self.text=text; self.tip=None
        if text: widget.bind("<Enter>", self._show, add="+"); widget.bind("<Leave>", self._hide, add="+")
    def _show(self,_e=None)->None:
        if self.tip or not self.text: return
        try:
            master=self.widget.winfo_toplevel()
            self.tip=tk.Toplevel(master); self.tip.wm_overrideredirect(True)
            x=self.widget.winfo_rootx()+self.widget.winfo_width()//2
            y=self.widget.winfo_rooty()+self.widget.winfo_height()+8
            self.tip.wm_geometry(f"+{x}+{y}")
            tk.Label(self.tip,text=self.text,bg="#111",fg="#fff",
                     padx=6,pady=3,relief="solid",bd=0,font=("Segoe UI",9)).pack()
        except Exception: self.tip=None
    def _hide(self,_e=None)->None:
        if self.tip: self.tip.destroy(); self.tip=None

# ============================= Converter Tab ================================

class ConverterTab(tk.Frame):
    def __init__(self, master: tk.Misc, palette_getter: Callable[[], dict],
                 fonts_getter: Callable[[], dict]) -> None:
        super().__init__(master)
        self._pg = palette_getter
        self._fg = fonts_getter
        self._build()

    LINEAR_UNITS: Dict[str, Dict[str, float]] = {
        "Length": {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "inch": 0.0254, "ft": 0.3048, "yd": 0.9144, "mile": 1609.344},
        "Mass": {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "lb": 0.45359237, "oz": 0.028349523125, "ton": 1000.0},
        "Area": {"m²": 1.0, "km²": 1e6, "cm²": 1e-4, "mm²": 1e-6, "ft²": 0.09290304, "in²": 0.00064516, "yd²": 0.83612736,
                 "acre": 4046.8564224, "hectare": 10000.0},
        "Volume": {"m³": 1.0, "L": 0.001, "mL": 1e-6, "cm³": 1e-6, "mm³": 1e-9, "gal (US)": 0.003785411784,
                   "qt (US)": 0.000946352946, "pt (US)": 0.000473176473, "cup (US)": 0.0002365882365, "fl oz (US)": 2.95735295625e-5},
        "Speed": {"m/s": 1.0, "km/h": 1000.0/3600.0, "mph": 0.44704, "knot": 0.514444},
        "Time": {"sec": 1.0, "min": 60.0, "hour": 3600.0, "day": 86400.0, "week": 604800.0},
        "Data": {"B": 1.0, "KB": 1024.0, "MB": 1024.0**2, "GB": 1024.0**3, "TB": 1024.0**4,
                 "bit": 1.0/8.0, "Kb": 1024.0/8.0, "Mb": (1024.0**2)/8.0, "Gb": (1024.0**3)/8.0, "Tb": (1024.0**4)/8.0},
    }

    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        left = tk.Frame(self, bd=0)
        left.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 8))
        tk.Label(left, text="Categories", font=self._fg()["ui_bold"]).pack(anchor="w")
        self.cat_list = tk.Listbox(left, height=16, activestyle="none", font=self._fg()["ui"])
        self.cat_list.pack(fill="y", expand=False)
        cats = list(self.LINEAR_UNITS.keys()) + ["Temperature", "Numeral System", "BMI", "Discount",
                                                 "Finance", "Date", "Currency (manual)"]
        for c in cats: self.cat_list.insert("end", c)
        self.cat_list.bind("<<ListboxSelect>>", self._on_select)
        self.cat_list.selection_set(0)

        self.right_area = tk.Frame(self, bd=0)
        self.right_area.grid(row=0, column=1, sticky="nsew")
        self.active_panel: Optional[tk.Frame] = None
        self._show_panel("Length")
        self.after(10, self._apply_palette)

    def _apply_palette(self) -> None:
        p = self._pg(); self.configure(bg=p["bg"])
        for w in self.winfo_children():
            if isinstance(w, tk.Frame):
                w.configure(bg=p["bg"])
                for ww in w.winfo_children():
                    if isinstance(ww, tk.Listbox):
                        ww.configure(bg=p["panel"], fg=p["fg"], selectbackground=p["btn_active"],
                                     highlightthickness=0, bd=0)
                    elif isinstance(ww, tk.Label):
                        ww.configure(bg=p["bg"], fg=p["fg"])
        if self.active_panel:
            for w in self.active_panel.winfo_children():
                if isinstance(w, tk.Label): w.configure(bg=p["bg"], fg=p["fg"])
                if isinstance(w, tk.Entry): w.configure(bg=p["entry_bg"], fg=p["fg"], insertbackground=p["fg"])
                if isinstance(w, tk.OptionMenu): w.configure(bg=p["panel"], fg=p["fg"], activebackground=p["btn_active"])

    def _on_select(self, _e=None) -> None:
        sel = self.cat_list.curselection()
        if not sel: return
        self._show_panel(self.cat_list.get(sel[0]))

    def _clear_panel(self) -> None:
        if self.active_panel is not None:
            self.active_panel.destroy(); self.active_panel=None

    def _show_panel(self, name: str) -> None:
        self._clear_panel()
        if name in self.LINEAR_UNITS:
            self.active_panel = self._panel_linear(name, self.LINEAR_UNITS[name])
        elif name == "Temperature":
            self.active_panel = self._panel_temperature()
        elif name == "Numeral System":
            self.active_panel = self._panel_numeral()
        elif name == "BMI":
            self.active_panel = self._panel_bmi()
        elif name == "Discount":
            self.active_panel = self._panel_discount()
        elif name == "Finance":
            self.active_panel = self._panel_finance()
        elif name == "Date":
            self.active_panel = self._panel_date()
        else:
            self.active_panel = self._panel_currency()
        self.active_panel.grid(row=0, column=1, sticky="nsew")
        self._apply_palette()

    # ---- Panels (same as earlier; omitted comments to keep length down)

    def _panel_linear(self, title: str, units: Dict[str, float]) -> tk.Frame:
        p = tk.Frame(self.right_area)
        tk.Label(p, text=f"{title} Converter", font=self._fg()["ui_bold"]).grid(row=0, column=0, columnspan=4, sticky="w")
        tk.Label(p, text="Value").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Label(p, text="From").grid(row=2, column=0, sticky="w")
        tk.Label(p, text="To").grid(row=3, column=0, sticky="w")
        val_var = tk.StringVar()
        keys = list(units.keys())
        from_var = tk.StringVar(value=keys[0])
        to_var = tk.StringVar(value=keys[1 if len(keys) > 1 else 0])
        out_var = tk.StringVar()
        e = tk.Entry(p, textvariable=val_var, width=20, font=self._fg()["mono"])
        e.grid(row=1, column=1, sticky="w", padx=(6, 0)); e.focus_set()

        def menu(var: tk.StringVar) -> tk.OptionMenu:
            m = tk.OptionMenu(p, var, *units.keys()); m.config(width=12, font=self._fg()["ui"]); return m

        menu(from_var).grid(row=2, column=1, sticky="w", padx=(6, 0))
        menu(to_var).grid(row=3, column=1, sticky="w", padx=(6, 0))
        tk.Label(p, text="Result").grid(row=4, column=0, sticky="w", pady=(10, 0))
        tk.Entry(p, textvariable=out_var, width=28, state="readonly",
                 readonlybackground="#ddd", font=self._fg()["mono"]).grid(row=4, column=1, sticky="w", padx=(6, 0))

        def convert(*_a):
            try:
                x = float(val_var.get())
                out_var.set(f"{x * units[from_var.get()] / units[to_var.get()]:.10g}")
            except Exception:
                out_var.set("")
        for v in (val_var, from_var, to_var): v.trace_add("write", lambda *_: convert())
        convert(); return p

    def _panel_temperature(self) -> tk.Frame:
        p = tk.Frame(self.right_area)
        tk.Label(p, text="Temperature Converter", font=self._fg()["ui_bold"]).grid(row=0, column=0, columnspan=4, sticky="w")
        tk.Label(p, text="Value").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Label(p, text="From").grid(row=2, column=0, sticky="w")
        tk.Label(p, text="To").grid(row=3, column=0, sticky="w")
        units = ["Celsius", "Fahrenheit", "Kelvin"]
        val = tk.StringVar(); from_v = tk.StringVar(value=units[0]); to_v = tk.StringVar(value=units[1]); out = tk.StringVar()
        tk.Entry(p, textvariable=val, width=20, font=self._fg()["mono"]).grid(row=1, column=1, sticky="w", padx=(6, 0))
        tk.OptionMenu(p, from_v, *units).grid(row=2, column=1, sticky="w", padx=(6, 0))
        tk.OptionMenu(p, to_v, *units).grid(row=3, column=1, sticky="w", padx=(6, 0))
        tk.Label(p, text="Result").grid(row=4, column=0, sticky="w", pady=(10, 0))
        tk.Entry(p, textvariable=out, width=28, state="readonly",
                 readonlybackground="#ddd", font=self._fg()["mono"]).grid(row=4, column=1, sticky="w")

        def to_c(x: float, u: str) -> float:
            return x if u == "Celsius" else (x - 32) * 5/9 if u == "Fahrenheit" else x - 273.15
        def from_c(c: float, u: str) -> float:
            return c if u == "Celsius" else c * 9/5 + 32 if u == "Fahrenheit" else c + 273.15

        def convert(*_a):
            try: out.set(f"{from_c(to_c(float(val.get()), from_v.get()), to_v.get()):.8g}")
            except Exception: out.set("")
        for v in (val, from_v, to_v): v.trace_add("write", lambda *_: convert())
        convert(); return p

    def _panel_numeral(self) -> tk.Frame:
        p = tk.Frame(self.right_area)
        tk.Label(p, text="Numeral System Converter", font=self._fg()["ui_bold"]).grid(row=0, column=0, columnspan=4, sticky="w")
        tk.Label(p, text="Value").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Label(p, text="From base").grid(row=2, column=0, sticky="w")
        tk.Label(p, text="To base").grid(row=3, column=0, sticky="w")
        val, out = tk.StringVar(), tk.StringVar()
        from_b, to_b = tk.StringVar(value="10"), tk.StringVar(value="16")
        tk.Entry(p, textvariable=val, width=24, font=self._fg()["mono"]).grid(row=1, column=1, sticky="w", padx=(6, 0))
        bases = ["2","3","5","7","8","10","12","16","20","36"]
        tk.OptionMenu(p, from_b, *bases).grid(row=2, column=1, sticky="w", padx=(6, 0))
        tk.OptionMenu(p, to_b, *bases).grid(row=3, column=1, sticky="w", padx=(6, 0))
        tk.Label(p, text="Result").grid(row=4, column=0, sticky="w", pady=(10, 0))
        tk.Entry(p, textvariable=out, width=28, state="readonly",
                 readonlybackground="#ddd", font=self._fg()["mono"]).grid(row=4, column=1, sticky="w")

        def convert(*_a):
            try:
                n = int(val.get().strip(), base=int(from_b.get()))
                digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                if n == 0: out.set("0"); return
                sign = "-" if n < 0 else ""; n = abs(n); b = int(to_b.get()); r=[]
                while n: n,m = divmod(n,b); r.append(digits[m])
                out.set(sign + "".join(reversed(r)))
            except Exception: out.set("")
        for v in (val, from_b, to_b): v.trace_add("write", lambda *_: convert())
        convert(); return p

    def _panel_bmi(self) -> tk.Frame:
        p = tk.Frame(self.right_area)
        tk.Label(p, text="BMI Calculator", font=self._fg()["ui_bold"]).grid(row=0, column=0, columnspan=2, sticky="w")
        tk.Label(p, text="Height (cm)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        h = tk.StringVar(); tk.Entry(p, textvariable=h, width=12).grid(row=1, column=1, sticky="w")
        tk.Label(p, text="Weight (kg)").grid(row=2, column=0, sticky="w")
        w = tk.StringVar(); tk.Entry(p, textvariable=w, width=12).grid(row=2, column=1, sticky="w")
        out = tk.StringVar(); tk.Label(p, textvariable=out, font=self._fg()["ui"]).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        def calc(*_a):
            try:
                hh = float(h.get())/100.0; ww = float(w.get()); bmi = ww/(hh*hh)
                cat = "Underweight" if bmi<18.5 else ("Normal" if bmi<25 else ("Overweight" if bmi<30 else "Obese"))
                out.set(f"BMI: {bmi:.2f}  ({cat})")
            except Exception: out.set("")
        for v in (h, w): v.trace_add("write", lambda *_: calc())
        return p

    def _panel_discount(self) -> tk.Frame:
        p = tk.Frame(self.right_area)
        tk.Label(p, text="Discount Calculator", font=self._fg()["ui_bold"]).grid(row=0, column=0, columnspan=2, sticky="w")
        tk.Label(p, text="Price").grid(row=1, column=0, sticky="w", pady=(8, 0))
        price = tk.StringVar(); tk.Entry(p, textvariable=price, width=12).grid(row=1, column=1, sticky="w")
        tk.Label(p, text="Discount %").grid(row=2, column=0, sticky="w")
        pct = tk.StringVar(); tk.Entry(p, textvariable=pct, width=12).grid(row=2, column=1, sticky="w")
        out = tk.StringVar(); tk.Label(p, textvariable=out, font=self._fg()["ui"]).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        def calc(*_a):
            try:
                p0 = float(price.get()); d = float(pct.get())/100.0
                out.set(f"Final: {p0*(1-d):.2f}   Saved: {(p0 - p0*(1-d)):.2f}")
            except Exception: out.set("")
        for v in (price, pct): v.trace_add("write", lambda *_: calc())
        return p

    def _panel_finance(self) -> tk.Frame:
        p = tk.Frame(self.right_area)
        tk.Label(p, text="Finance — Simple & Compound Interest", font=self._fg()["ui_bold"]).grid(row=0, column=0, columnspan=2, sticky="w")
        labels = ["Principal", "Rate % (annual)", "Years", "Compounds/year (n)"]
        vs = [tk.StringVar() for _ in range(4)]
        for i, (lab, var) in enumerate(zip(labels, vs), start=1):
            tk.Label(p, text=lab).grid(row=i, column=0, sticky="w", pady=(6 if i==1 else 2, 0))
            tk.Entry(p, textvariable=var, width=14).grid(row=i, column=1, sticky="w")
        out = tk.StringVar(); tk.Label(p, textvariable=out, font=self._fg()["ui"]).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))

        def calc(*_a):
            try:
                P=float(vs[0].get()); r=float(vs[1].get())/100.0; t=float(vs[2].get())
                n=int(float(vs[3].get())) if vs[3].get() else 1
                out.set(f"Simple: {P*(1+r*t):.2f}   Compound: {P*((1+r/n)**(n*t)):.2f}")
            except Exception: out.set("")
        for v in vs: v.trace_add("write", lambda *_: calc())
        return p

    def _panel_date(self) -> tk.Frame:
        p = tk.Frame(self.right_area)
        tk.Label(p, text="Date Difference", font=self._fg()["ui_bold"]).grid(row=0, column=0, columnspan=2, sticky="w")
        tk.Label(p, text="Start (YYYY-MM-DD)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Label(p, text="End   (YYYY-MM-DD)").grid(row=2, column=0, sticky="w")
        s, e, out = tk.StringVar(), tk.StringVar(), tk.StringVar()
        tk.Entry(p, textvariable=s, width=14).grid(row=1, column=1, sticky="w")
        tk.Entry(p, textvariable=e, width=14).grid(row=2, column=1, sticky="w")
        tk.Label(p, textvariable=out, font=self._fg()["ui"]).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        def calc(*_a):
            try:
                d1=_dt.datetime.strptime(s.get(), "%Y-%m-%d").date()
                d2=_dt.datetime.strptime(e.get(), "%Y-%m-%d").date()
                out.set(f"{(d2-d1).days} day(s)")
            except Exception: out.set("")
        for v in (s, e): v.trace_add("write", lambda *_: calc())
        return p

    def _panel_currency(self) -> tk.Frame:
        p = tk.Frame(self.right_area)
        tk.Label(p, text="Currency (manual rate)", font=self._fg()["ui_bold"]).grid(row=0, column=0, columnspan=3, sticky="w")
        tk.Label(p, text="Amount").grid(row=1, column=0, sticky="w", pady=(8, 0))
        amt = tk.StringVar(); tk.Entry(p, textvariable=amt, width=14).grid(row=1, column=1, sticky="w")
        tk.Label(p, text="Rate: 1 From = ? To").grid(row=2, column=0, sticky="w")
        rate = tk.StringVar(); tk.Entry(p, textvariable=rate, width=14).grid(row=2, column=1, sticky="w")
        tk.Label(p, text="Result").grid(row=3, column=0, sticky="w", pady=(8, 0))
        out = tk.StringVar()
        tk.Entry(p, textvariable=out, width=22, state="readonly",
                 readonlybackground="#ddd", font=self._fg()["mono"]).grid(row=3, column=1, sticky="w")
        tk.Label(p, text="Tip: enter rate you get elsewhere (e.g., 1 USD = 83.25 INR → rate=83.25)")\
            .grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))

        def calc(*_a):
            try: out.set(f"{float(amt.get())*float(rate.get()):.6g}")
            except Exception: out.set("")
        for v in (amt, rate): v.trace_add("write", lambda *_: calc())
        return p

# =============================== Pretty UI ==================================

@dataclass
class Palette:
    name:str; bg:str; panel:str; entry_bg:str; fg:str; subtle:str; btn_bg:str; btn_active:str; accent:str; border:str

LIGHT = Palette("light","#F6F7FB","#FFFFFF","#FFFFFF","#1F2937","#6B7280","#EEF1F7","#E5E7EB","#4F46E5","#E5E7EB")
DARK  = Palette("dark" ,"#0F172A","#111827","#0B1220","#E5E7EB","#9CA3AF","#1F2937","#334155","#60A5FA","#1F2937")

class CalculatorApp(tk.Tk):
    MAX_EXPR_LEN = 160
    MAX_PREVIEW_LEN = 140

    _ALLOWED_CHARS = set("0123456789.+-*/()^,%")
    _OPS = set("+-*/^,%")

    _FUNC_TOKENS = {
        "sin(", "cos(", "tan(", "asin(", "acos(", "atan(", "atan2(",
        "lg(", "ln(", "sqrt(", "cbrt(", "yroot(", "ten_pow(", "inv(", "sqr(", "cube(", "pct(", "mod(", "sec(", "csc(", "cot("
    }
    _CONST_TOKENS = {"pi", "e", "ans"}

    @staticmethod
    def _is_op_char(ch: str) -> bool: return ch in "+-*/^,%"

    def __init__(self) -> None:
        super().__init__()
        self.title("Scientific Calculator"); self.minsize(800, 600)
        self.engine = CalculatorEngine(Settings(angle_mode="rad", precision=12))
        self.memory: float = 0.0
        self.palette: Palette = LIGHT; self._second=False; self._ui_scale=1.0

        self._init_fonts(); self._build_ui(); self._apply_palette()
        self.bind("<Return>", lambda e: self.evaluate())
        self.bind("<KP_Enter>", lambda e: self.evaluate())
        self.bind("<Escape>", lambda e: self.clear_all())
        self.bind("<Control-BackSpace>", lambda e: self.clear_all())
        self.after(80, lambda: self.entry.focus_set())

    # Fonts & scale
    def _init_fonts(self)->None:
        def choose(*names:str)->str:
            avail=set(tkfont.families())
            for n in names:
                if n in avail: return n
            return "Segoe UI"
        self.fonts={"title":tkfont.Font(family=choose("Segoe UI Variable","Segoe UI Semibold"), size=16),
                    "ui":tkfont.Font(family=choose("Segoe UI","Arial"), size=11),
                    "ui_bold":tkfont.Font(family=choose("Segoe UI Semibold","Segoe UI","Arial"), size=11, weight="bold"),
                    "mono":tkfont.Font(family=choose("Consolas","Courier New"), size=18)}
    def _apply_scale(self)->None:
        for name,f in self.fonts.items():
            base = 16 if name=="title" else 18 if name=="mono" else 11
            f.configure(size=int(round(base*self._ui_scale)))

    # UI
    def _build_ui(self)->None:
        self.root_frame=tk.Frame(self,bd=0); self.root_frame.pack(fill="both",expand=True,padx=12,pady=12)
        self.nb=ttk.Notebook(self.root_frame); self.nb.pack(fill="both",expand=True)
        self.calc_tab=tk.Frame(self.nb,bd=0); self.nb.add(self.calc_tab,text="Calculator")

        self.topbar=tk.Frame(self.calc_tab); self.topbar.pack(fill="x")
        self.title_label=tk.Label(self.topbar,text="Scientific Calculator",font=self.fonts["title"],anchor="w"); self.title_label.pack(side="left")
        self.mode_label=tk.Label(self.topbar,text=self._mode_text(),font=self.fonts["ui"]); self.mode_label.pack(side="left",padx=(10,0))
        self.theme_btn=tk.Button(self.topbar,text="🌙",width=3,relief="flat",command=self.toggle_theme); self.theme_btn.pack(side="right")

        self.entry_panel=tk.Frame(self.calc_tab,bd=1); self.entry_panel.pack(fill="x",pady=(10,6))
        self.entry_var=tk.StringVar()
        self.entry=tk.Entry(self.entry_panel,textvariable=self.entry_var,font=self.fonts["mono"],relief="flat",bd=0)
        self.entry.pack(fill="x",ipady=10,padx=8,pady=(10,4))
        self.entry.bind("<Key>", self._on_key)
        self.entry.bind("<KeyRelease>", lambda e: self._update_preview())
        self.entry.bind("<<Paste>>", self._on_paste, add="+")
        self.entry.bind("<Control-v>", self._on_paste, add="+")
        self.entry.bind("<Command-v>", self._on_paste, add="+")
        self.preview_var=tk.StringVar()
        self.preview=tk.Label(self.entry_panel,textvariable=self.preview_var,anchor="e",font=self.fonts["ui"])
        self.preview.pack(fill="x",padx=12,pady=(0,10))

        self.status=tk.Label(self.calc_tab,text=self._status_text(),anchor="w",font=self.fonts["ui"])
        self.status.pack(fill="x",pady=(0,8))

        self.area=tk.Frame(self.calc_tab); self.area.pack(fill="both",expand=True)
        self.left_box=tk.Frame(self.area); self.left_box.pack(side="left",fill="both",expand=True)
        self.right_box=tk.Frame(self.area,width=240); self.right_box.pack(side="right",fill="y",padx=(10,0))

        self.grid_frame=tk.Frame(self.left_box); self.grid_frame.pack(fill="both",expand=True)
        for c in range(6): self.grid_frame.grid_columnconfigure(c, weight=1)
        for r in range(7): self.grid_frame.grid_rowconfigure(r, weight=1)

        self.buttons: List[tk.Button] = []
        self._second_map: Dict[tk.Button, tuple] = {}

        def add_btn(row:int,col:int,primary_text:str,action:Callable[[],None],
                    secondary_text:Optional[str]=None, action2:Optional[Callable[[],None]]=None,
                    accent:bool=False, tip:str="")->None:
            b=tk.Button(self.grid_frame,text=primary_text,relief="flat",bd=0,
                        font=self.fonts["ui_bold"] if accent else self.fonts["ui"], command=action)
            b.grid(row=row,column=col,sticky="nsew",padx=4,pady=4,ipady=8)
            if tip: Tooltip(b, tip)
            if accent: b._accent=True  # type: ignore[attr-defined]
            self._style_button(b); self.buttons.append(b)
            if secondary_text and action2: self._second_map[b]=(primary_text,action,secondary_text,action2)

        def ins(s:str)->Callable[[],None]: return lambda: self.insert_text(s)

        # Row 0
        add_btn(0,0,"MC",self.mem_clear); add_btn(0,1,"MR",self.mem_recall)
        add_btn(0,2,"M+",self.mem_add);  add_btn(0,3,"M−",self.mem_sub)
        add_btn(0,4,"deg/rad",self.toggle_mode,tip="Toggle angle units")
        add_btn(0,5,"C",self.clear_all,tip="Clear entry")

        # Row 1
        add_btn(1,0,"2nd",self.toggle_second,tip="Show alternate functions")
        add_btn(1,1,"sin",ins("sin("),"asin",ins("asin("),tip="Sine / arcsine")
        add_btn(1,2,"cos",ins("cos("),"acos",ins("acos("),tip="Cosine / arccos")
        add_btn(1,3,"tan",ins("tan("),"atan",ins("atan("),tip="Tangent / arctan")
        add_btn(1,4,"lg",ins("lg("),"10ˣ",ins("ten_pow("),tip="log10 / 10^x")
        add_btn(1,5,"←",self.backspace,"±",self.toggle_sign,tip="Backspace / sign")

        # Row 2
        add_btn(2,0,"xʸ",ins("**"),"y√x",ins("yroot("),tip="Power / y-root")
        add_btn(2,1,"ln",ins("ln(")); add_btn(2,2,"(",ins("(")); add_btn(2,3,")",ins(")"))
        add_btn(2,4,"√",ins("sqrt("),"∛",ins("cbrt("))
        add_btn(2,5,"%",ins("pct("),"mod",ins("mod("),tip="Percent / modulo")

        # Row 3
        add_btn(3,0,"x!",ins("fact("),tip="Factorial")
        add_btn(3,1,"7",ins("7")); add_btn(3,2,"8",ins("8")); add_btn(3,3,"9",ins("9"))
        add_btn(3,4,"÷",ins("/"))
        add_btn(3,5,"sec",ins("sec("),"inv",ins("inv("),tip="secant / reciprocal")

        # Row 4
        add_btn(4,0,"ANS",ins("ans"),"sqr",ins("sqr("),tip="Last result / square")
        add_btn(4,1,"4",ins("4")); add_btn(4,2,"5",ins("5")); add_btn(4,3,"6",ins("6"))
        add_btn(4,4,"×",ins("*"))
        add_btn(4,5,"csc",ins("csc("),"cube",ins("cube("),tip="cosecant / cube")

        # Row 5
        add_btn(5,0,"π",ins("pi")); add_btn(5,1,"1",ins("1")); add_btn(5,2,"2",ins("2")); add_btn(5,3,"3",ins("3"))
        add_btn(5,4,"−",ins("-"))
        add_btn(5,5,"cot",ins("cot("),"atan2",ins("atan2("),tip="cotangent / atan2(y,x)")

        # Row 6
        add_btn(6,0,"e",ins("e"),"ANS",ins("ans")); add_btn(6,1,"0",ins("0"))
        add_btn(6,2,".",ins(".")); add_btn(6,3,",",ins(","))
        add_btn(6,4,"=",self.evaluate,accent=True,tip="Evaluate")
        add_btn(6,5,"+",ins("+"))

        # History
        self.hist_header=tk.Label(self.right_box,text="History",font=self.fonts["ui_bold"]); self.hist_header.pack(anchor="w",pady=(0,4))
        self.hist_container=tk.Frame(self.right_box,bd=1); self.hist_container.pack(fill="y")
        self.history_list=tk.Listbox(self.hist_container,height=18,activestyle="none",selectmode="browse",
                                     bd=0,highlightthickness=0,font=self.fonts["mono"])
        self.history_list.pack(side="left",fill="y")
        self.hist_scroll=tk.Scrollbar(self.hist_container,orient="vertical",command=self.history_list.yview)
        self.hist_scroll.pack(side="right",fill="y")
        self.history_list.config(yscrollcommand=self.hist_scroll.set)
        self.history_list.bind("<Double-1>", self._history_use_result)
        self.history_list.bind("<Button-3>", self._history_copy_popup)
        self.clear_hist_btn=tk.Button(self.right_box,text="🧹 Clear History",relief="flat",
                                      command=lambda: self.history_list.delete(0,"end"))
        self.clear_hist_btn.pack(pady=(6,0)); self._style_button(self.clear_hist_btn)

        # Converter tab + menus
        self.conv_tab=ConverterTab(self.nb,self._palette_dict,self._fonts_dict)
        self.nb.add(self.conv_tab,text="Converter")

        menubar=tk.Menu(self); settings_menu=tk.Menu(menubar,tearoff=0)
        mode_menu=tk.Menu(settings_menu,tearoff=0)
        mode_menu.add_radiobutton(label="Radians",command=lambda:self.set_mode("rad"))
        mode_menu.add_radiobutton(label="Degrees",command=lambda:self.set_mode("deg"))
        settings_menu.add_cascade(label="Angle Mode",menu=mode_menu)

        prec_menu=tk.Menu(settings_menu,tearoff=0)
        for n in (6,8,10,12,14,15): prec_menu.add_radiobutton(label=str(n),command=lambda n=n:self.set_precision(n))
        settings_menu.add_cascade(label="Precision (sig. digits)",menu=prec_menu)

        scale_menu=tk.Menu(settings_menu,tearoff=0)
        for pct in (100,112,125,150,175): scale_menu.add_radiobutton(label=f"{pct}%",command=lambda p=pct:self.set_scale(p/100.0))
        settings_menu.add_cascade(label="UI Scale",menu=scale_menu)

        settings_menu.add_separator()
        settings_menu.add_command(label="Clear Entry",command=self.clear_all,accelerator="Esc")
        settings_menu.add_command(label="Toggle Theme",command=self.toggle_theme)
        menubar.add_cascade(label="Settings",menu=settings_menu)

        help_menu=tk.Menu(menubar,tearoff=0)
        help_menu.add_command(label="Functions Reference",command=self.show_functions_help)
        help_menu.add_command(label="Shortcuts",command=self.show_shortcuts)
        help_menu.add_separator(); help_menu.add_command(label="About",command=self.show_about)
        self.config(menu=menubar)

    # Theming
    def _palette_dict(self)->dict:
        p=self.palette
        return dict(bg=p.bg,panel=p.panel,entry_bg=p.entry_bg,fg=p.fg,subtle=p.subtle,btn_bg=p.btn_bg,btn_active=p.btn_active,accent=p.accent,border=p.border)
    def _fonts_dict(self)->dict: return dict(title=self.fonts["title"],ui=self.fonts["ui"],ui_bold=self.fonts["ui_bold"],mono=self.fonts["mono"])

    def _style_button(self,b:tk.Button)->None:
        colors=self._palette_dict()
        if getattr(b,"_accent",False):
            b.configure(bg=colors["accent"],fg="white",activebackground=colors["accent"],activeforeground="white")
            hov=_mix(colors["accent"],"#ffffff",0.08); b.bind("<Enter>",lambda _e:b.configure(bg=hov)); b.bind("<Leave>",lambda _e:b.configure(bg=colors["accent"]))
        else:
            b.configure(bg=colors["btn_bg"],fg=colors["fg"],activebackground=colors["btn_active"],activeforeground=colors["fg"])
            hov=_mix(colors["btn_bg"],colors["btn_active"],0.6); b.bind("<Enter>",lambda _e:b.configure(bg=hov)); b.bind("<Leave>",lambda _e:b.configure(bg=colors["btn_bg"]))

    def _apply_palette(self)->None:
        self._apply_scale(); p=self._palette_dict()
        self.configure(bg=p["bg"])
        for w in (self.calc_tab,self.topbar,self.entry_panel,self.area,self.left_box,self.right_box,self.grid_frame):
            w.configure(bg=p["bg"])
        self.title_label.configure(bg=p["bg"],fg=p["fg"])
        self.mode_label.configure(bg=p["bg"],fg=p["subtle"])
        self.theme_btn.configure(bg=p["panel"],fg=p["fg"],activebackground=p["btn_active"],activeforeground=p["fg"])
        self.entry_panel.configure(bg=p["panel"],highlightbackground=p["border"],highlightcolor=p["border"],highlightthickness=1)
        self.entry.configure(bg=p["entry_bg"],fg=p["fg"],insertbackground=p["fg"])
        self.preview.configure(bg=p["panel"],fg=p["subtle"])
        self.status.configure(bg=p["bg"],fg=p["subtle"])
        self.grid_frame.configure(bg=p["panel"],highlightbackground=p["border"],highlightcolor=p["border"],highlightthickness=1)
        for b in self.buttons+[self.clear_hist_btn]: self._style_button(b)
        self.hist_header.configure(bg=p["bg"],fg=p["fg"])
        self.hist_container.configure(bg=p["panel"],highlightbackground=p["border"],highlightcolor=p["border"],highlightthickness=1)
        self.history_list.configure(bg=p["panel"],fg=p["fg"],selectbackground=p["btn_active"],selectforeground=p["fg"])
        self.conv_tab._apply_palette()

    def toggle_theme(self)->None:
        self.palette = DARK if self.palette is LIGHT else LIGHT
        self.theme_btn.configure(text=("🌙" if self.palette is LIGHT else "☀️"))
        self._apply_palette()
    def set_scale(self,s:float)->None: self._ui_scale=max(0.9,min(2.0,s)); self._apply_palette()

    # ----------------------------- Sanitize helpers ------------------------
    def _prev_char(self, s: str, pos: int) -> str: return s[pos-1] if pos>0 else ""
    def _unclosed_parens(self, s: str) -> int:
        opens=0
        for ch in s:
            if ch=="(": opens+=1
            elif ch==")" and opens>0: opens-=1
        return opens

    def _ends_with_name(self, s: str) -> bool:
        tail3 = s[-3:] if len(s)>=3 else s
        tail2 = s[-2:] if len(s)>=2 else s
        return tail3.endswith("ans") or tail2.endswith("pi") or s.endswith("e")

    def _is_value_before(self, s: str, pos: int) -> bool:
        if pos<=0: return False
        c=s[pos-1]
        if c.isdigit() or c==")": return True
        return self._ends_with_name(s[:pos])

    def _trim_preview(self, txt: str) -> str:
        return txt if len(txt)<=self.MAX_PREVIEW_LEN else txt[:self.MAX_PREVIEW_LEN-1]+"…"

    # ----------------------------- Actions ---------------------------------
    def insert_text(self, text: Optional[str]) -> None:
        if not text: return
        s = self.entry_var.get(); pos = self.entry.index(tk.INSERT)
        before, after = s[:pos], s[pos:]

        if text == "^": text = "**"

        # auto '*' before functions/constants/'('
        if text in self._FUNC_TOKENS or text in self._CONST_TOKENS or text == "(":
            if self._is_value_before(s,pos): before += "*"

        # auto '*' before DIGITS or '.' when they directly follow a value/name or ')'
        if (text and text[0].isdigit()) or text == ".":
            if text == ".":
                # if starting a number like ".5", make it "0.5" unless already in digits
                if pos==0 or self._is_op_char(self._prev_char(s,pos)) or self._prev_char(s,pos)=="(":
                    before += "0"
                elif self._ends_with_name(s[:pos]) or self._prev_char(s,pos)==")":
                    before += "*0"
            else:  # a digit
                if self._ends_with_name(s[:pos]) or self._prev_char(s,pos)==")":
                    before += "*"

        # operators & punctuation normalization
        if text in {"+","-","*","/","%","^",",",")","**"}:
            if text == "^": text = "**"
            # limit '*' to '**'
            if text == "*" and pos>=2 and s[pos-1]=="*" and s[pos-2]=="*": self.bell(); return
            # disallow '//' (floor div)
            if text == "/" and pos>0 and s[pos-1]=="/": self.bell(); return

            prev=self._prev_char(s,pos)
            if prev and self._is_op_char(prev):
                # allow unary '-' after '(' or at start; otherwise replace previous
                if not ((prev=="(" or pos==0) and text=="-"):
                    before = s[:pos-1]

            # disallow binary op at start/after '(' except unary '-'
            if (pos==0 or self._prev_char(before,len(before))=="(") and text not in ("-","("):
                if text in {"+","*","/","%",",",")","**"}: self.bell(); return

        # ')' rules
        if text==")":
            if self._unclosed_parens(s[:pos])<=0: self.bell(); return
            if pos>0 and self._is_op_char(s[pos-1]): self.bell(); return

        # ',' rules
        if text==",":
            if self._unclosed_parens(s[:pos])<=0: self.bell(); return
            if pos==0 or self._is_op_char(s[pos-1]) or s[pos-1]=="(": self.bell(); return

        new_s = before + text + after
        if len(new_s) > self.MAX_EXPR_LEN: self.bell(); return

        self.entry_var.set(new_s)
        self.entry.icursor(len(before)+len(text))
        self.entry.focus_set()
        self._update_preview()

    def evaluate(self) -> None:
        expr = self.entry_var.get()
        missing = self._unclosed_parens(expr)
        if missing>0: expr = expr + (")"*missing)
        try:
            result = self.engine.evaluate(expr)
        except CalculationError as exc:
            self.bell(); messagebox.showerror("Error", str(exc), parent=self); return
        formatted = self.engine.format_number(result)
        self.entry_var.set(formatted)
        self._update_status()
        self._add_history(expr, formatted)
        self._update_preview()

    def _update_preview(self) -> None:
        expr = self.entry_var.get().strip()
        if not expr: self.preview_var.set("= 0.00"); return
        try:
            val = self.engine.evaluate(expr)
            self.preview_var.set(self._trim_preview("= " + self.engine.format_number(val)))
        except Exception:
            self.preview_var.set("")

    def clear_all(self) -> None:
        self.entry_var.set(""); self._update_status(); self._update_preview()

    def backspace(self) -> None:
        pos = self.entry.index(tk.INSERT)
        if pos>0: self.entry.delete(pos-1)
        self.entry.focus_set(); self._update_preview()

    def toggle_sign(self) -> None:
        s = self.entry_var.get().strip()
        if not s: self.entry_var.set("-"); return
        try:
            val=float(s); self.entry_var.set(self.engine.format_number(-val))
        except Exception:
            if s.startswith("-(") and s.endswith(")"): self.entry_var.set(s[2:-1])
            else: self.entry_var.set(f"-({s})")
        self._update_preview()

    def toggle_mode(self)->None:
        self.engine.settings.angle_mode = "deg" if self.engine.settings.angle_mode=="rad" else "rad"
        self._update_status()
    def set_mode(self,mode:str)->None: self.engine.settings.angle_mode=mode; self._update_status()
    def set_precision(self,n:int)->None: self.engine.settings.precision=int(n); self._update_status(); self._update_preview()
    def toggle_second(self)->None:
        self._second = not self._second
        for btn,(t1,a1,t2,a2) in self._second_map.items():
            btn.configure(text=(t2 if self._second else t1), command=(a2 if self._second else a1))

    # Memory
    def mem_clear(self)->None: self.memory=0.0; self._update_status()
    def mem_recall(self)->None: self.insert_text(self.engine.format_number(self.memory))
    def mem_add(self)->None: self.memory += self.engine.last_result; self._update_status()
    def mem_sub(self)->None: self.memory -= self.engine.last_result; self._update_status()

    # Keyboard & paste
    def _filter_text(self, text: str) -> str:
        out=[]
        for ch in text:
            if ch not in self._ALLOWED_CHARS: continue
            if ch=="*":
                if len(out)>=2 and out[-1]=="*" and out[-2]=="*": continue
                out.append("*"); continue
            if self._is_op_char(ch):
                if out and self._is_op_char(out[-1]): out[-1]=ch
                else: out.append(ch)
                continue
            out.append(ch)
        return "".join(out)

    def _on_paste(self,_e=None):
        try: txt=self.clipboard_get()
        except tk.TclError: return "break"
        clean=self._filter_text(txt)
        if clean: self.insert_text(clean)
        return "break"

    def _on_key(self, event: tk.Event):
        if event.keysym=="asciicircum": self.insert_text("**"); return "break"
        if event.keysym in ("Return","KP_Enter"): return "break"
        if event.keysym in ("BackSpace","Delete","Left","Right","Home","End","Tab"): return
        ch = event.char or ""
        if not ch: return
        if ch.isalpha(): return "break"
        if ch not in self._ALLOWED_CHARS: return "break"
        self.insert_text(ch)   # route through same sanitizer
        return "break"

    # Helpers
    def _status_text(self)->str:
        s=self.engine.settings; mem=self.engine.format_number(self.memory)
        return f"Angle: {s.angle_mode.upper()} • Precision: {s.precision} sig. digits • ANS: {self.engine.format_number(self.engine.last_result)} • MEM: {mem}"
    def _mode_text(self)->str: return f"Mode: {self.engine.settings.angle_mode.upper()}"
    def _update_status(self)->None:
        self.status.config(text=self._status_text()); self.mode_label.config(text=self._mode_text())
    def _add_history(self, expr: str, result_str: str) -> None:
        self.history_list.insert("end", f"{expr}  =  {result_str}"); self.history_list.see("end")
    def _history_use_result(self,_e=None)->None:
        sel=self.history_list.curselection()
        if not sel: return
        item=self.history_list.get(sel[0])
        if "  =  " in item:
            result=item.split("  =  ",1)[1]
            self.entry_var.set(result); self.entry.icursor("end"); self.entry.focus_set(); self._update_preview()
    def _history_copy_popup(self,e:tk.Event)->None:
        sel=self.history_list.curselection()
        if not sel: return
        item=self.history_list.get(sel[0]); m=tk.Menu(self,tearoff=0)
        m.add_command(label="Copy line",command=lambda:self._copy(item))
        if "  =  " in item:
            expr,res=item.split("  =  ",1)
            m.add_command(label="Copy result",command=lambda:self._copy(res))
            m.add_command(label="Copy expression",command=lambda:self._copy(expr))
        m.tk_popup(e.x_root,e.y_root)
    def _copy(self,text:str)->None:
        self.clipboard_clear(); self.clipboard_append(text)

    # About/help
    def show_about(self)->None:
        messagebox.showinfo("About",
            "Scientific Calculator + Converters (Tkinter)\n"
            "Safe AST • 2nd functions • Live preview • History • Memory • Light/Dark • Scalable UI\n"
            "Standard library only.", parent=self)
    def show_shortcuts(self)->None:
        messagebox.showinfo("Shortcuts",
            "Enter / Num Enter  : Evaluate\n"
            "Esc / Ctrl+Backspace: Clear entry\n"
            "Backspace           : Delete char\n"
            "Double-click history: Paste result\n"
            "2nd                 : Toggle alternate functions", parent=self)
    def show_functions_help(self)->None:
        env=self.engine.build_env()
        funcs=sorted(k for k,v in env.items() if callable(v))
        consts=sorted((k,v) for k,v in env.items() if not callable(v))
        lines=["Functions:\n  " + ", ".join(funcs), "\nConstants:"]
        lines += [f"  {k} = {self.engine.format_number(v)}" for k,v in consts]
        messagebox.showinfo("Functions & Constants","\n".join(lines),parent=self)

# ============================= Entrypoint ===================================

def main()->int:
    try:
        app=CalculatorApp(); app.mainloop(); return 0
    except Exception as exc:
        messagebox.showerror("Fatal Error", str(exc)); return 1

if __name__=="__main__":
    raise SystemExit(main())
