# PDF Auction Parser (Windows GUI app + CLI)

Минималистичное приложение для Windows: перетащите один или несколько PDF в окно и получите готовый Excel с лотами.

## Что умеет
- Drag & Drop PDF (основной сценарий) + кнопка выбора файлов.
- Извлечение текста из PDF (PyMuPDF) и OCR fallback (pytesseract) при пустом тексте.
- Парсинг лотов по строкам в Excel (1 лот = 1 строка).
- Сохранение картинок из PDF в `output_images/<pdf_name>/` (опционально).
- Лог ошибок в `parsing_errors.csv`.
- CLI-режим для автоматизации.

## Установка (разработка)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Установите OCR-движок:
- Windows: установите Tesseract OCR и добавьте его в PATH.

## Запуск GUI
```bash
python app.py
```

## Сборка полноценного `.exe` для Windows
### Вариант 1 (автоматически)
Запустите:
```bat
build_windows_exe.bat
```
Готовый файл: `dist\AuctionPdfParser.exe`.

### Вариант 2 (вручную)
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name AuctionPdfParser app.py
```

## CLI режим
```bash
python app.py --cli C:\path\to\folder -o result.xlsx
```

Опции:
- `--force-ocr` — OCR для всех страниц.
- `--no-save-images` — не извлекать изображения.
- `--run-tests` — встроенные тесты.

## Результаты
- `result.xlsx` — таблица по лотам.
- `parsing_errors.csv` — ошибки парсинга.
- `output_images/...` — извлечённые изображения.

## TODO
- Нормализация брендов через расширяемый словарь.
- Дедупликация лотов по `SerialNumber`.
- Аналитика цен (бакеты, перцентили).
- Экспорт в CSV + Power BI friendly формат.
- Визуализации (гистограмма цен, heatmap адресов).
