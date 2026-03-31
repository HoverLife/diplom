@echo off
setlocal

if not exist .venv (
  python -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

pyinstaller --noconfirm --clean --onefile --windowed --name AuctionPdfParser app.py

echo.
echo Build complete. EXE: dist\AuctionPdfParser.exe
pause
