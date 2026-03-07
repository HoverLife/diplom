import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import pandas as pd
import pytesseract
from dateutil import parser as dt_parser
from PIL import Image

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except Exception:
    QtCore = QtGui = QtWidgets = None

KNOWN_BRANDS = {
    "aoc": "AOC",
    "samsung": "Samsung",
    "lg": "LG",
    "dell": "Dell",
    "hp": "HP",
    "acer": "Acer",
    "philips": "Philips",
    "benq": "BenQ",
    "asus": "ASUS",
    "lenovo": "Lenovo",
    "viewsonic": "ViewSonic",
    "sony": "Sony",
    "iiyama": "iiyama",
}

COLUMN_ORDER = [
    "SourceFile",
    "SourcePage",
    "ProcessedAt",
    "LotPosition",
    "AuctionCity",
    "AuctionEndDateTime",
    "ItemType",
    "Brand",
    "Model",
    "Diagonal",
    "SerialNumber",
    "StartPriceRUB",
    "PickupAddress",
    "CityPickup",
    "VideoInputs",
    "PowerType",
    "PowerIncluded",
    "MatrixIssues",
    "Defect1",
    "Defect2",
    "Photo1",
    "Photo2",
    "Photo3",
    "HasImages",
    "RawText",
    "ExtractedFeatures",
    "Keywords",
    "EstimatedCondition",
    "Notes",
]

KEY_TO_COLUMN = {
    "видеовходы": "VideoInputs",
    "тип питания": "PowerType",
    "сетевой шнур или блок питания в наличии": "PowerIncluded",
    "затемнения и полосы на матрице": "MatrixIssues",
    "диагональ": "Diagonal",
    "дефект1": "Defect1",
    "дефект2": "Defect2",
}


@dataclass
class ParseResult:
    rows: List[Dict]
    errors: List[Dict]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_lots(page_text: str) -> List[str]:
    text = "\n" + page_text.strip()
    starts = [m.start() for m in re.finditer(r"(?m)^\s*\d+\s*-{2,}\s*", text)]
    if not starts:
        starts = [m.start() for m in re.finditer(r"(?m)^\s*\d+\s+", text)]
    if not starts:
        return [page_text]

    blocks: List[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
    return blocks


def normalize_diagonal(value: str) -> str:
    match = re.search(r"(\d{1,2}(?:[\.,]\d{1,2})?)\s*['\"]?", value)
    if not match:
        return value.strip()
    return f"{match.group(1).replace(',', '.')}\""


def parse_int_price(lines: List[str]) -> Optional[int]:
    for line in lines:
        m = re.match(r"^\s*([\d\s]{2,12})\s*(?:руб|р\.|RUB)?\s*$", line, flags=re.IGNORECASE)
        if not m:
            continue
        value = re.sub(r"\s+", "", m.group(1))
        if value.isdigit():
            return int(value)
    return None


def parse_datetime_from_header(text: str) -> Optional[datetime]:
    m = re.search(r"(\d{1,2}[./]\d{1,2}[./]\d{2,4}(?:\s+\d{1,2}:\d{2})?)", text)
    if not m:
        return None
    try:
        return dt_parser.parse(m.group(1), dayfirst=True)
    except Exception:
        return None


def parse_head(block: str) -> Tuple[str, str, str, str, str]:
    city = ""
    item_type = ""
    brand = ""
    model = ""
    diagonal = ""

    city_m = re.search(r"\(([^)]+)\)", block)
    if city_m:
        city = city_m.group(1).strip()

    head_region = block[city_m.end():] if city_m else block
    end_sn = re.search(r"(?i)\bs/n\s*:", head_region)
    if end_sn:
        head_text = head_region[: end_sn.start()].strip()
    else:
        head_text = re.split(r"-{2,}", head_region, maxsplit=1)[0].strip()

    tokens = head_text.split()
    if tokens:
        item_type = tokens[0]

    for i, token in enumerate(tokens[1:], start=1):
        canon = KNOWN_BRANDS.get(token.lower().strip(".,"))
        if canon:
            brand = canon
            if i + 1 < len(tokens):
                model = tokens[i + 1]
                diagonal = normalize_diagonal(" ".join(tokens[i + 2 :])) if i + 2 < len(tokens) else ""
            break

    if not brand and len(tokens) > 1:
        model = " ".join(tokens[1:])

    if not diagonal:
        dm = re.search(r"(\d{1,2}(?:[\.,]\d{1,2})?)\s*['\"]", head_text)
        if dm:
            diagonal = normalize_diagonal(dm.group(0))

    return city, item_type, brand, model, diagonal


def parse_serial(block: str) -> str:
    m = re.search(r"(?i)s/n\s*:\s*([A-Za-z0-9\-]+)", block)
    return m.group(1).strip() if m else ""


def parse_address(block: str) -> str:
    m = re.search(r"(г\.\s*[А-ЯЁA-Z][^\n]+)", block)
    return m.group(1).strip() if m else ""


def parse_city_from_address(address: str) -> str:
    m = re.search(r"г\.\s*([А-ЯЁA-Zа-яa-z\-]+)", address)
    return m.group(1).strip() if m else ""


def estimate_condition(row: Dict) -> str:
    txt = " ".join(
        [
            row.get("Defect1", ""),
            row.get("Defect2", ""),
            row.get("MatrixIssues", ""),
            row.get("RawText", ""),
        ]
    ).lower()
    bad_markers = ["не работает", "трещ", "бит", "мертв", "полос"]
    fair_markers = ["следы эксплуатации", "царап", "потерт", "потёрт"]
    if any(x in txt for x in bad_markers):
        return "Bad"
    if any(x in txt for x in fair_markers):
        return "Fair"
    return "Good"


def derive_keywords(row: Dict) -> str:
    chunks = [row.get("Brand", ""), row.get("Model", ""), row.get("VideoInputs", ""), row.get("Defect1", ""), row.get("Defect2", "")]
    out: List[str] = []
    for chunk in chunks:
        for token in re.split(r"[,/\s]+", chunk):
            token = token.strip()
            if token and len(token) > 2 and token.lower() not in {x.lower() for x in out}:
                out.append(token)
    return ", ".join(out[:12])


def parse_attributes(attrs_zone: str) -> Tuple[Dict[str, str], Dict[str, str], str]:
    normalized_zone = attrs_zone.replace("VGA/DVI", "VGA__DVI")
    attr_parts = [part.strip() for part in normalized_zone.split("/") if part.strip()]

    mapped: Dict[str, str] = {}
    extra: Dict[str, str] = {}
    free_defect = ""

    for part in attr_parts:
        part = part.replace("VGA__DVI", "VGA/DVI")
        m = re.match(r"([А-Яа-яA-Za-z0-9\-\s]+)\s*-\s*(.+)", part)
        if not m:
            if "дефект" in part.lower() or "следы эксплуатации" in part.lower():
                free_defect = part.splitlines()[0].strip(" .")
            continue
        key = m.group(1).strip().lower()
        value = m.group(2).strip(" .")
        col = KEY_TO_COLUMN.get(key)
        if col:
            mapped[col] = normalize_diagonal(value) if col == "Diagonal" else value
        else:
            extra[m.group(1).strip()] = value

    return mapped, extra, free_defect


def parse_block(block: str, source_file: str, source_page: int, auction_dt: Optional[datetime], photos: List[str]) -> Tuple[Dict, List[Dict]]:
    row = {key: "" for key in COLUMN_ORDER}
    row.update(
        {
            "SourceFile": source_file,
            "SourcePage": source_page,
            "ProcessedAt": datetime.now().isoformat(timespec="seconds"),
            "AuctionEndDateTime": auction_dt,
            "RawText": block,
            "HasImages": bool(photos),
        }
    )

    for i, path in enumerate(photos[:3], start=1):
        row[f"Photo{i}"] = path

    errors: List[Dict] = []

    pos = re.match(r"^\s*(\d+)", block)
    row["LotPosition"] = int(pos.group(1)) if pos else None

    city, item_type, brand, model, diagonal = parse_head(block)
    row["AuctionCity"] = city
    row["ItemType"] = item_type
    row["Brand"] = brand
    row["Model"] = model
    row["Diagonal"] = diagonal
    row["SerialNumber"] = parse_serial(block)

    row["PickupAddress"] = parse_address(block)
    row["CityPickup"] = parse_city_from_address(row["PickupAddress"])

    markers = re.split(r"-{2,}", block, maxsplit=2)
    attrs_zone = markers[2] if len(markers) > 2 else block

    mapped, extras, free_defect = parse_attributes(attrs_zone)
    for key, value in mapped.items():
        row[key] = value

    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    row["StartPriceRUB"] = parse_int_price(lines)
    if row["StartPriceRUB"] is None:
        errors.append(
            {
                "SourceFile": source_file,
                "LotPosition": row["LotPosition"],
                "ErrorMessage": "Price not parsed",
                "RawText": block,
            }
        )

    if not row.get("Defect1"):
        row["Defect1"] = free_defect

    row["ExtractedFeatures"] = json.dumps(extras, ensure_ascii=False)
    row["Keywords"] = derive_keywords(row)
    row["EstimatedCondition"] = estimate_condition(row)
    return row, errors


def extract_images(pdf_path: Path, images_root: Path) -> Dict[int, List[str]]:
    by_page: Dict[int, List[str]] = {}
    out_dir = images_root / pdf_path.stem
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            images = page.get_images(full=True)
            page_list: List[str] = []
            if images:
                out_dir.mkdir(parents=True, exist_ok=True)
            for i, image_ref in enumerate(images, start=1):
                xref = image_ref[0]
                data = doc.extract_image(xref)
                ext = data.get("ext", "png")
                file_name = f"page{page_num:03d}_{i:02d}.{ext}"
                output_file = out_dir / file_name
                output_file.write_bytes(data["image"])
                page_list.append(str(output_file.relative_to(images_root.parent)))
            by_page[page_num] = page_list
    return by_page


def extract_text_from_pdf(pdf_path: Path, force_ocr: bool = False, ocr_lang: str = "rus+eng") -> Tuple[List[Tuple[int, str]], bool]:
    pages: List[Tuple[int, str]] = []
    ocr_used = False

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = "" if force_ocr else (page.get_text("text") or "")
            if not text.strip():
                pix = page.get_pixmap(dpi=250)
                mode = "RGBA" if pix.alpha else "RGB"
                image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                try:
                    text = pytesseract.image_to_string(image, lang=ocr_lang)
                    ocr_used = True
                except Exception as exc:
                    text = ""
                    ocr_used = True
                    pages.append((page_num, normalize_text(text)))
                    continue
            pages.append((page_num, normalize_text(text)))

    return pages, ocr_used


def process_pdfs(pdf_files: List[Path], output_excel: Path, save_images: bool = True, force_ocr: bool = False) -> ParseResult:
    rows: List[Dict] = []
    errors: List[Dict] = []
    images_root = output_excel.parent / "output_images"

    for pdf in pdf_files:
        page_images = extract_images(pdf, images_root) if save_images else {}
        pages, _ocr_used = extract_text_from_pdf(pdf, force_ocr=force_ocr)

        auction_dt = parse_datetime_from_header(pages[0][1]) if pages else None

        for page_num, text in pages:
            if not text.strip():
                errors.append({"SourceFile": pdf.name, "LotPosition": None, "ErrorMessage": "Empty page text", "RawText": ""})
                continue
            for block in split_lots(text):
                if len(block.strip()) < 8:
                    continue
                row, row_errors = parse_block(block, pdf.name, page_num, auction_dt, page_images.get(page_num, []))
                rows.append(row)
                errors.extend(row_errors)

    df = pd.DataFrame(rows)
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = ""
    df = df[COLUMN_ORDER]
    df["StartPriceRUB"] = pd.to_numeric(df["StartPriceRUB"], errors="coerce")
    df["AuctionEndDateTime"] = pd.to_datetime(df["AuctionEndDateTime"], errors="coerce")

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Lots")
        ws = writer.book["Lots"]
        for idx, col in enumerate(df.columns, start=1):
            col_letter = ws.cell(row=1, column=idx).column_letter
            if df.empty:
                width = 14
            else:
                width = min(60, max(12, int(df[col].astype(str).str.len().quantile(0.9))))
            ws.column_dimensions[col_letter].width = width

    err_file = output_excel.parent / "parsing_errors.csv"
    if errors:
        with open(err_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["SourceFile", "LotPosition", "ErrorMessage", "RawText"])
            writer.writeheader()
            writer.writerows(errors)

    return ParseResult(rows=rows, errors=errors)


def test_on_sample_text() -> bool:
    sample = (
        "1--- (Москва) Монитор AOC M2060SWD2 19,5' s/n: GLXK5HA053007 --- "
        "Диагональ-19,5' / Затемнения и полосы на матрице-Нет / Видеовходы-VGA/DVI / "
        "Тип питания-Кабель / Сетевой шнур или блок питания в наличии - Да/Следы эксплуатации.\n"
        "700\n"
        "г. Москва, шоссе Энтузиастов д.14\n"
    )
    row, errs = parse_block(sample, "example.pdf", 1, None, [])
    checks = [
        row["LotPosition"] == 1,
        row["AuctionCity"] == "Москва",
        row["Brand"] == "AOC",
        row["Model"] == "M2060SWD2",
        row["Diagonal"] == '19.5"',
        row["SerialNumber"] == "GLXK5HA053007",
        row["StartPriceRUB"] == 700,
        row["VideoInputs"] == "VGA/DVI",
        row["PowerType"] == "Кабель",
        row["PowerIncluded"] == "Да",
        row["Defect1"].startswith("Следы эксплуатации"),
        len(errs) == 0,
    ]
    return all(checks)


def test_ocr_fallback_switch() -> bool:
    # Проверяем, что OCR path включается при пустом тексте (без вызова реального tesseract).
    class DummyPage:
        alpha = False

        def get_text(self, _mode):
            return ""

        def get_pixmap(self, dpi=250):
            class DummyPix:
                alpha = False
                width = 2
                height = 2
                samples = b"\x00\x00\x00" * 4

            return DummyPix()

    class DummyDoc:
        def __enter__(self):
            return [DummyPage()]

        def __exit__(self, exc_type, exc, tb):
            return False

    old_open = fitz.open
    old_ocr = pytesseract.image_to_string
    try:
        fitz.open = lambda _path: DummyDoc()
        pytesseract.image_to_string = lambda _img, lang="rus+eng": "ocr text"
        pages, used = extract_text_from_pdf(Path("dummy.pdf"))
        return used is True and pages[0][1] == "ocr text"
    finally:
        fitz.open = old_open
        pytesseract.image_to_string = old_ocr


def run_sample_tests() -> int:
    tests = [
        ("sample lot parsing", test_on_sample_text),
        ("ocr fallback switch", test_ocr_fallback_switch),
    ]
    passed = 0
    print("Running sample parser tests...")
    for name, fn in tests:
        ok = fn()
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if ok:
            passed += 1
    print(f"Tests passed: {passed}/{len(tests)}")
    return 0 if passed == len(tests) else 1


if QtWidgets is not None:
    class DropArea(QtWidgets.QFrame):
        files_dropped = QtCore.pyqtSignal(list)

        def __init__(self):
            super().__init__()
            self.setAcceptDrops(True)
            self.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.setStyleSheet("QFrame { border: 2px dashed #7a7a7a; border-radius: 8px; background: #fafafa; }")
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            label = QtWidgets.QLabel("Перетащите PDF сюда")
            label.setAlignment(QtCore.Qt.AlignCenter)
            font = QtGui.QFont()
            font.setPointSize(10)
            label.setFont(font)
            layout.addWidget(label)

        def dragEnterEvent(self, event):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

        def dropEvent(self, event):
            files: List[str] = []
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.suffix.lower() == ".pdf":
                    files.append(str(p))
            if files:
                self.files_dropped.emit(files)


    class Worker(QtCore.QThread):
        finished_ok = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)

        def __init__(self, files: List[Path], output: Path, force_ocr: bool, save_images: bool):
            super().__init__()
            self.files = files
            self.output = output
            self.force_ocr = force_ocr
            self.save_images = save_images

        def run(self):
            try:
                result = process_pdfs(self.files, self.output, save_images=self.save_images, force_ocr=self.force_ocr)
                self.finished_ok.emit(result)
            except Exception as exc:
                self.failed.emit(str(exc))


    class MainWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("PDF Auction Parser")
            self.setMinimumSize(560, 340)
            self.resize(620, 380)

            self.files: List[Path] = []
            self.worker: Optional[Worker] = None

            root = QtWidgets.QVBoxLayout(self)
            root.setSpacing(10)

            title = QtWidgets.QLabel("Парсер аукционных PDF")
            title_font = QtGui.QFont()
            title_font.setPointSize(12)
            title_font.setBold(True)
            title.setFont(title_font)

            self.info = QtWidgets.QLabel("Файлы не выбраны")
            self.drop_area = DropArea()
            self.drop_area.files_dropped.connect(self.set_files)

            self.progress = QtWidgets.QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

            btn_choose = QtWidgets.QPushButton("Выбрать PDF")
            btn_start = QtWidgets.QPushButton("Обработать")
            btn_choose.clicked.connect(self.choose_files)
            btn_start.clicked.connect(self.start_processing)

            self.chk_ocr = QtWidgets.QCheckBox("Принудительный OCR")
            self.chk_images = QtWidgets.QCheckBox("Сохранять изображения")
            self.chk_images.setChecked(True)

            controls = QtWidgets.QHBoxLayout()
            controls.addWidget(btn_choose)
            controls.addWidget(btn_start)
            controls.addStretch(1)
            controls.addWidget(self.chk_ocr)
            controls.addWidget(self.chk_images)

            root.addWidget(title)
            root.addWidget(self.info)
            root.addWidget(self.drop_area, 1)
            root.addWidget(self.progress)
            root.addLayout(controls)

        def set_files(self, files: List[str]):
            self.files = [Path(f) for f in files if f.lower().endswith(".pdf")]
            self.info.setText(f"Выбрано PDF: {len(self.files)}")
            # Основной сценарий: сразу предложить сохранить и запустить обработку
            self.start_processing(auto_from_drop=True)

        def choose_files(self):
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Выберите PDF", "", "PDF Files (*.pdf)")
            if files:
                self.files = [Path(f) for f in files]
                self.info.setText(f"Выбрано PDF: {len(self.files)}")

        def start_processing(self, auto_from_drop: bool = False):
            if not self.files:
                QtWidgets.QMessageBox.warning(self, "Нет файлов", "Выберите или перетащите PDF файлы.")
                return

            default_name = f"parsed_lots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить Excel", default_name, "Excel (*.xlsx)")
            if not out:
                if auto_from_drop:
                    self.info.setText("Обработка отменена пользователем")
                return

            self.progress.setValue(5)
            self.setEnabled(False)
            self.worker = Worker(self.files, Path(out), self.chk_ocr.isChecked(), self.chk_images.isChecked())
            self.worker.finished_ok.connect(self.on_done)
            self.worker.failed.connect(self.on_error)
            self.worker.start()

        def on_done(self, result: ParseResult):
            self.progress.setValue(100)
            self.setEnabled(True)
            QtWidgets.QMessageBox.information(
                self,
                "Готово",
                f"Лотов обработано: {len(result.rows)}\nОшибок: {len(result.errors)}",
            )

        def on_error(self, message: str):
            self.progress.setValue(0)
            self.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "Ошибка", message)


def collect_pdf_inputs(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob("*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            files.append(p)
    return files


def cli_mode(args) -> int:
    pdf_files = collect_pdf_inputs(args.inputs)
    if not pdf_files:
        print("PDF files not found")
        return 1
    result = process_pdfs(pdf_files, Path(args.output), save_images=args.save_images, force_ocr=args.force_ocr)
    print(f"Done. Lots: {len(result.rows)}, errors: {len(result.errors)}, output: {args.output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Windows-ready PDF auction parser (GUI + CLI)")
    parser.add_argument("inputs", nargs="*", help="PDF files or folders for CLI mode")
    parser.add_argument("-o", "--output", default="parsed_lots.xlsx", help="Output xlsx (CLI mode)")
    parser.add_argument("--cli", action="store_true", help="Run CLI mode")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR on all pages")
    parser.add_argument("--no-save-images", action="store_false", dest="save_images", help="Disable image extraction")
    parser.add_argument("--run-tests", action="store_true", help="Run built-in parser tests")
    args = parser.parse_args()

    if args.run_tests:
        raise SystemExit(run_sample_tests())

    if args.cli or args.inputs:
        raise SystemExit(cli_mode(args))

    if QtWidgets is None:
        print("PyQt5 not available. Install dependencies or run in CLI mode.")
        raise SystemExit(1)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
