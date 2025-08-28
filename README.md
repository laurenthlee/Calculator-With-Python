## UI Tour
- **Top bar**: title, current angle mode (DEG/RAD), theme toggle.
- **Entry panel**: main input box + gray live preview line underneath.
- **Keypad grid**: primary functions; press 2nd to switch to the alternate set.
- **Right panel (History)**: shows expression = result. Double-click a line to paste its result back into the entry. “🧹 Clear History” to wipe it.
- **Menu** → **Settings**:
- **Angle** Mode: Radians or Degrees
- **Precision**: 6–15 significant digits
- **UI Scale**: 100–175%
- **Clear Entry**, **Toggle Theme**
- **Tabs**: Calculator and Converter (left list of categories, right active panel).

---

## Shortcuts
- **Enter / Numpad Enter** — Evaluate
- **Esc / Ctrl+Backspace** — Clear entry
- **Backspace — Delete** one char
- **Double-click History** — Paste result into entry
- **2nd** — Toggle alternate functions

---
## ✨ User Interface
<img width="600" height="800" alt="image" src="https://github.com/user-attachments/assets/84f8f496-b0b3-4f76-a388-78e1e4858784" />
<img width="600" height="800" alt="image" src="https://github.com/user-attachments/assets/afbb0e09-b44f-4c2d-aa24-2f2c63af078a" />
<img width="600" height="800" alt="image" src="https://github.com/user-attachments/assets/a3da7ff9-3b1c-4f5d-a85c-83750ad85514" />

---

## ✨ Highlights

- Clean light/dark theme, hover effects, scalable UI (100–175%).
- Live result **preview while typing**.
- Memory keys & history panel (double-click to reuse results).
- **2nd** layer buttons (asin/acos/atan, y√x, 10^x, ∛, mod, inv, sqr, cube, atan2).
- **Converters:** Length, Mass, Area, Volume, Speed, Temperature, Time, Data,
  Numeral System, BMI, Discount, Finance, Date difference, Currency (manual rate).
- Degree/Radian toggle, precision (significant digits), UI scale.
- Hardened input: unified sanitization for keyboard, buttons and paste.

---

## 🧰 Requirements

- **Python 3.8+** (3.10+ recommended)
- Tkinter (ships with python.org builds and most Linux distros)
  - Ubuntu/Debian: `sudo apt install python3-tk`
  - Fedora: `sudo dnf install python3-tkinter`
  - Arch: `sudo pacman -S tk`

No other packages are needed.

---

## 🚀 Run

```bash
python calculator.py
