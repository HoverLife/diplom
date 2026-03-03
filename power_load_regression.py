#!/usr/bin/env python3
"""Обучение и запуск модели регрессии для прогнозирования электрических нагрузок.

Поддерживаемые форматы входа:
- .csv (разделитель определяется автоматически)
- .xlsx/.xls

Ожидается, что в файле есть столбец даты/времени и измерения мощности (kw/kvar)
по нескольким точкам (например, 6 точек фиксации).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


DEFAULT_LAGS = [1, 2, 48]

def _is_unnamed_label(value: str) -> bool:
    v = str(value).strip().lower()
    return v.startswith("unnamed:") or v in {"nan", "none", ""}


def _flatten_multiindex_columns(columns: pd.MultiIndex) -> List[str]:
    levels = [pd.Series(columns.get_level_values(i)).astype(str) for i in range(columns.nlevels)]

    # Для merged-ячеек Excel pandas часто создает Unnamed: ... — протягиваем соседние заголовки
    for i in range(len(levels)):
        lvl = levels[i].copy()
        lvl = lvl.mask(lvl.map(_is_unnamed_label))
        lvl = lvl.ffill().fillna("")
        levels[i] = lvl

    flat_cols: List[str] = []
    for col_idx in range(len(columns)):
        parts = []
        for lvl in levels:
            token = str(lvl.iloc[col_idx]).strip()
            if token and token not in parts:
                parts.append(token)

        if parts:
            flat_cols.append(" ".join(parts))
        else:
            flat_cols.append(f"column_{col_idx}")

    return flat_cols


def _normalize_singlelevel_columns(columns: List[str]) -> List[str]:
    normalized: List[str] = []
    prev = ""
    for idx, col in enumerate(columns):
        raw = str(col).strip()
        if _is_unnamed_label(raw):
            new_name = prev if prev else f"column_{idx}"
        else:
            new_name = raw
        normalized.append(new_name)
        prev = new_name
    return normalized




def _make_unique(columns: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for col in columns:
        base = col if col else "column"
        cnt = seen.get(base, 0)
        if cnt == 0:
            unique_name = base
        else:
            unique_name = f"{base}_{cnt}"
        seen[base] = cnt + 1
        unique.append(unique_name)
    return unique




def _is_generic_measurement_name(name: str) -> bool:
    s = str(name).strip().lower()
    if s.startswith("column_"):
        return True
    if s.startswith("time_"):
        suffix = s.split("time_", 1)[1]
        return suffix.isdigit()
    return False


def _rename_generic_measurement_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Переименовывает технические имена (column_i/time_i) в человеко-понятные."""
    out = df.copy()
    generic_cols = [c for c in out.columns if c != "timestamp" and _is_generic_measurement_name(c)]
    if not generic_cols:
        return out

    rename_map: Dict[str, str] = {}

    # Эвристика для типового случая: пары активной/реактивной мощности
    if len(generic_cols) % 2 == 0:
        point_id = 1
        for idx, col in enumerate(generic_cols, start=1):
            suffix = "kw" if idx % 2 == 1 else "kvar"
            rename_map[col] = f"measurement_point_{point_id:02d}_{suffix}"
            if suffix == "kvar":
                point_id += 1
    else:
        for idx, col in enumerate(generic_cols, start=1):
            rename_map[col] = f"measurement_feature_{idx:02d}"

    out = out.rename(columns=rename_map)
    out.columns = _make_unique([str(c) for c in out.columns])
    return out


def read_dataset(path: str) -> pd.DataFrame:
    """Читает CSV/Excel и пытается аккуратно нормализовать заголовки."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        # Для файлов после сводной таблицы нередко полезно читать multi-header
        try:
            df = pd.read_excel(file_path, header=[0, 1])
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = _make_unique(_flatten_multiindex_columns(df.columns))
        except Exception:
            df = pd.read_excel(file_path)
    else:
        # sep=None => python engine сам определяет разделитель
        df = pd.read_csv(file_path, sep=None, engine="python")

    # Если попался неявный multi-index через автогенерированные колонки
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = _make_unique(_flatten_multiindex_columns(df.columns))
    else:
        df.columns = _make_unique(_normalize_singlelevel_columns(list(df.columns)))

    # Удаляем полностью пустые столбцы/строки
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    # Попытка найти столбец времени
    datetime_candidates = [
        c
        for c in df.columns
        if any(token in str(c).lower() for token in ["date", "time", "дата", "время", "названия строк"])
    ]

    if datetime_candidates:
        dt_col = datetime_candidates[0]
    else:
        # Частый случай: первый столбец - это timestamp
        dt_col = df.columns[0]

    df = df.rename(columns={dt_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Чистим числовые столбцы (на случай пробелов, запятых и т.д.)
    for col in df.columns:
        if col == "timestamp":
            continue
        if df[col].dtype == object:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.replace("\u00a0", "", regex=False)
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    # Оставляем только timestamp + числовые столбцы
    numeric_cols = [c for c in df.columns if c == "timestamp" or pd.api.types.is_numeric_dtype(df[c])]
    df = df[numeric_cols].copy()

    # Если из Excel пришли технические заголовки, даем им человеко-понятные имена
    df = _rename_generic_measurement_columns(df)

    if len(df.columns) < 3:
        raise ValueError(
            "После очистки в датасете слишком мало числовых столбцов. "
            "Проверьте формат входного файла (ожидаются kw/kvar по точкам)."
        )

    return df


def find_kw_columns(df: pd.DataFrame) -> List[str]:
    """Ищет столбцы активной мощности kw."""
    kw_cols = [c for c in df.columns if c != "timestamp" and "kw" in c.lower() and "kvar" not in c.lower()]
    if not kw_cols:
        # fallback: считаем все числовые, кроме timestamp
        kw_cols = [c for c in df.columns if c != "timestamp"]
    return kw_cols


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute
    out["dow"] = out["timestamp"].dt.dayofweek

    # Циклические признаки
    out["hour_sin"] = np.sin(2 * np.pi * (out["hour"] + out["minute"] / 60.0) / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * (out["hour"] + out["minute"] / 60.0) / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    return out


def build_supervised_frame(df: pd.DataFrame, horizon_steps: int, lags: List[int]) -> Tuple[pd.DataFrame, str]:
    """Готовит таблицу для задачи: предсказать total_kw через horizon_steps шагов."""
    out = df.copy()
    kw_cols = find_kw_columns(out)

    out["total_kw"] = out[kw_cols].sum(axis=1)
    out = add_time_features(out)

    for lag in lags:
        out[f"total_kw_lag_{lag}"] = out["total_kw"].shift(lag)

    target_col = f"target_total_kw_t_plus_{horizon_steps}"
    out[target_col] = out["total_kw"].shift(-horizon_steps)

    out = out.dropna().reset_index(drop=True)
    return out, target_col


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.clip(np.abs(y_true), 1e-6, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"MAE": float(mae), "RMSE": rmse, "MAPE_percent": mape}


def train_models(
    data: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[dict, dict, list]:
    feature_cols = [c for c in data.columns if c not in {"timestamp", target_col}]

    split_idx = int(len(data) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(data):
        raise ValueError("Некорректный test_size. Должен быть в диапазоне (0, 1).")

    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    x_train = train_df[feature_cols]
    y_train = train_df[target_col]
    x_test = test_df[feature_cols]
    y_test = test_df[target_col]

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    fitted = {}
    scores = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        fitted[name] = model
        scores[name] = calc_metrics(y_test.values, pred)

    return fitted, scores, feature_cols


def save_artifacts(
    output_dir: str,
    best_model_name: str,
    model,
    feature_cols: List[str],
    scores: Dict[str, Dict[str, float]],
    horizon_steps: int,
    lags: List[int],
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    meta_path = out_dir / "metadata.json"

    joblib.dump(model, model_path)

    meta = {
        "best_model": best_model_name,
        "feature_columns": feature_cols,
        "scores": scores,
        "horizon_steps": horizon_steps,
        "lags": lags,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def train_command(args: argparse.Namespace):
    df = read_dataset(args.input)
    supervised_df, target_col = build_supervised_frame(df, args.horizon_steps, args.lags)

    fitted_models, scores, feature_cols = train_models(
        supervised_df,
        target_col=target_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Лучшая модель по MAE
    best_model_name = min(scores.keys(), key=lambda name: scores[name]["MAE"])
    best_model = fitted_models[best_model_name]

    save_artifacts(
        output_dir=args.output_dir,
        best_model_name=best_model_name,
        model=best_model,
        feature_cols=feature_cols,
        scores=scores,
        horizon_steps=args.horizon_steps,
        lags=args.lags,
    )

    print("=== Метрики на тестовой выборке ===")
    for model_name, metrics in scores.items():
        print(f"[{model_name}] MAE={metrics['MAE']:.3f} | RMSE={metrics['RMSE']:.3f} | MAPE={metrics['MAPE_percent']:.3f}%")
    print(f"Лучшая модель: {best_model_name}")
    print(f"Артефакты сохранены в: {args.output_dir}")


def predict_command(args: argparse.Namespace):
    model = joblib.load(Path(args.model_dir) / "model.joblib")
    meta = json.loads((Path(args.model_dir) / "metadata.json").read_text(encoding="utf-8"))

    df = read_dataset(args.input)
    supervised_df, target_col = build_supervised_frame(df, meta["horizon_steps"], meta["lags"])

    feature_cols = meta["feature_columns"]
    missing = set(feature_cols) - set(supervised_df.columns)
    if missing:
        raise ValueError(f"Во входных данных не хватает признаков: {missing}")

    x = supervised_df[feature_cols]
    supervised_df["prediction_total_kw"] = model.predict(x)

    cols = ["timestamp", "total_kw", target_col, "prediction_total_kw"]
    result = supervised_df[cols].copy()
    result.to_csv(args.output, index=False)
    print(f"Прогнозы сохранены в: {args.output}")


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Прогнозирование электрических нагрузок (регрессия)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Обучение модели")
    train_parser.add_argument("--input", required=True, help="Путь к CSV/XLSX датасету")
    train_parser.add_argument("--output-dir", default="artifacts", help="Директория для сохранения модели")
    train_parser.add_argument("--horizon-steps", type=int, default=1, help="Горизонт прогноза в шагах (1 шаг = 30 минут)")
    train_parser.add_argument("--lags", nargs="+", type=int, default=DEFAULT_LAGS, help="Лаги total_kw для признаков")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Доля тестовой части (по времени)")
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.set_defaults(func=train_command)

    predict_parser = subparsers.add_parser("predict", help="Запуск прогноза по сохраненной модели")
    predict_parser.add_argument("--input", required=True, help="Путь к CSV/XLSX датасету")
    predict_parser.add_argument("--model-dir", default="artifacts", help="Папка с model.joblib и metadata.json")
    predict_parser.add_argument("--output", default="predictions.csv", help="Файл для сохранения прогноза")
    predict_parser.set_defaults(func=predict_command)

    return parser


def main():
    parser = make_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
