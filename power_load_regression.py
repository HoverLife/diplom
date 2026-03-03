#!/usr/bin/env python3
"""Обучение и запуск модели регрессии для прогнозирования электрических нагрузок."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_LAGS = [1, 2, 48]
DEFAULT_ROLLING_WINDOWS = [4, 48]


# -------------------------
# Утилиты для очистки имён
# -------------------------
def _slugify(s: str) -> str:
    """Сделать строку безопасной для имени столбца."""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"[\/\|\(\)\[\]\:;,]+", " ", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"__+", "_", s)
    s = re.sub(r"[^0-9A-Za-zА-Яа-я_\-]", "", s)
    return s.strip("_") or "col"


def _detect_measure_token(token: str) -> str | None:
    """Распознать kw/kvar (с учетом русских вариантов)."""
    if token is None:
        return None
    t = str(token).lower()
    if "kw" in t:
        return "kw"
    if "kvar" in t or "квар" in t or "кvar" in t:
        return "kvar"
    if "сумма" in t and "kw" in t:
        return "kw"
    if "сумма" in t and "kvar" in t:
        return "kvar"
    return None


def _make_unique(columns: list[str]) -> list[str]:
    seen = {}
    unique = []
    for col in columns:
        base = col if col else "column"
        cnt = seen.get(base, 0)
        unique_name = base if cnt == 0 else f"{base}_{cnt}"
        seen[base] = cnt + 1
        unique.append(unique_name)
    return unique


# ---------------------------------------
# Чтение файла с хорошей интерпретацией
# ---------------------------------------
def read_dataset_better(path: str) -> pd.DataFrame:
    """Улучшенное чтение и нормализация заголовков."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = None
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path, header=[0, 1])
        else:
            raise ValueError("skip")
    except Exception:
        try:
            if path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")

    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        levels = [pd.Series(df.columns.get_level_values(i)).astype(str) for i in range(df.columns.nlevels)]
        for i in range(len(levels)):
            lvl = levels[i].mask(
                levels[i].str.strip().str.lower().map(
                    lambda v: v.startswith("unnamed:") if isinstance(v, str) else False
                )
            )
            levels[i] = lvl.ffill().fillna("")

        for col_idx in range(len(df.columns)):
            parts = [str(levels[i].iloc[col_idx]).strip() for i in range(len(levels))]
            top = next((p for p in parts if p and not _detect_measure_token(p)), "")

            measure = None
            for p in parts:
                m = _detect_measure_token(p)
                if m:
                    measure = m
                    break
            if measure is None:
                for p in parts:
                    if p and ("kw" in p.lower() or "kvar" in p.lower()):
                        measure = "kw" if "kw" in p.lower() else "kvar"
                        break

            name = _slugify(top) if top else _slugify(next((p for p in parts if p), f"col{col_idx}"))
            if measure:
                name = f"{name}__{measure}"
            cols.append(name)

        df.columns = _make_unique(cols)
    else:
        raw_cols = list(df.columns)
        norm = []
        prev = ""
        for idx, col in enumerate(raw_cols):
            raw = str(col).strip()
            if raw == "" or raw.lower().startswith("unnamed"):
                name = prev if prev else f"column_{idx}"
            else:
                name = raw
            name = _slugify(name)
            norm.append(name)
            prev = name
        df.columns = _make_unique(norm)

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    dt_candidates = [
        c
        for c in df.columns
        if any(token in str(c).lower() for token in ["date", "time", "дата", "время", "timestamp", "ts", "nazvaniya"])
    ]
    dt_col = dt_candidates[0] if dt_candidates else df.columns[0]
    df = df.rename(columns={dt_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for col in df.columns:
        if col == "timestamp":
            continue
        if df[col].dtype == object:
            cleaned = (
                df[col].astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace("\u00a0", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    numeric_cols = [c for c in df.columns if c == "timestamp" or pd.api.types.is_numeric_dtype(df[c])]
    df = df[numeric_cols]

    if len(df.columns) < 3:
        raise ValueError("После очистки слишком мало признаков. Проверьте формат выгрузки.")

    return df


# ---------------------------
# Поиск kw-колонок (улучшенный)
# ---------------------------
def find_kw_columns_better(df: pd.DataFrame) -> list[str]:
    cand = [c for c in df.columns if c != "timestamp" and "__kw" in c]
    if not cand:
        cand = [c for c in df.columns if c != "timestamp" and re.search(r"\bkw\b", c, flags=re.I)]
    if not cand:
        cand = [c for c in df.columns if c != "timestamp"]
    return cand


# ---------------------------------
# Построение признаков (feature eng)
# ---------------------------------
def build_features_better(
    df: pd.DataFrame,
    horizon_steps: int,
    lags: list[int],
    rolling_windows: list[int],
) -> Tuple[pd.DataFrame, List[str], str, dict]:
    out = df.copy()
    kw_cols = find_kw_columns_better(out)

    kw_only = [c for c in kw_cols if "__kw" in c or re.search(r"\bkw\b", c, flags=re.I)]
    kvar_cols = [c for c in kw_cols if "__kvar" in c or re.search(r"\bkvar\b", c, flags=re.I)]
    used_kw_cols = kw_only if kw_only else [c for c in out.columns if c != "timestamp"]

    out["total_kw"] = out[used_kw_cols].sum(axis=1)

    for c in used_kw_cols:
        share_col = f"{c}__share"
        out[share_col] = np.where(out["total_kw"].abs() > 0, out[c] / out["total_kw"], 0.0)

    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute
    out["dow"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * (out["hour"] + out["minute"] / 60) / 24)
    out["hour_cos"] = np.cos(2 * np.pi * (out["hour"] + out["minute"] / 60) / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)

    for lag in lags:
        out[f"total_kw_lag_{lag}"] = out["total_kw"].shift(lag)

    for w in rolling_windows:
        out[f"total_kw_roll_mean_{w}"] = out["total_kw"].shift(1).rolling(w).mean()
        out[f"total_kw_roll_std_{w}"] = out["total_kw"].shift(1).rolling(w).std()

    target_col = f"target_total_kw_t_plus_{horizon_steps}"
    out[target_col] = out["total_kw"].shift(-horizon_steps)

    out = out.dropna().reset_index(drop=True)

    feature_cols = [c for c in out.columns if c not in ["timestamp", target_col]]
    meta = {
        "kw_cols": used_kw_cols,
        "kvar_cols": kvar_cols,
        "orig_columns": list(df.columns),
    }
    return out, feature_cols, target_col, meta


def plot_total_with_roll(df: pd.DataFrame, roll_window: int = 48, show_legend: bool = True):
    """Построить total_kw и его скользящее среднее для интерпретации."""
    if "total_kw" not in df.columns:
        raise ValueError("В df нет колонки total_kw. Постройте признаки сначала.")
    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["total_kw"], label="total_kw")
    if f"total_kw_roll_mean_{roll_window}" in df.columns:
        plt.plot(df["timestamp"], df[f"total_kw_roll_mean_{roll_window}"], label=f"roll_mean_{roll_window}")
    plt.xlabel("timestamp")
    plt.ylabel("kW")
    plt.title("Total kW и скользящее среднее")
    if show_legend:
        plt.legend()
    plt.tight_layout()
    plt.show()


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

    fitted, scores = {}, {}
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
    rolling_windows: List[int],
    build_meta: dict,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")

    meta = {
        "best_model": best_model_name,
        "feature_columns": feature_cols,
        "scores": scores,
        "horizon_steps": horizon_steps,
        "lags": lags,
        "rolling_windows": rolling_windows,
        "build_meta": build_meta,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def train_command(args: argparse.Namespace):
    df = read_dataset_better(args.input)
    supervised_df, _, target_col, build_meta = build_features_better(
        df,
        horizon_steps=args.horizon_steps,
        lags=args.lags,
        rolling_windows=args.rolling_windows,
    )

    fitted_models, scores, feature_cols = train_models(
        supervised_df,
        target_col=target_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

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
        rolling_windows=args.rolling_windows,
        build_meta=build_meta,
    )

    print("=== Метрики на тестовой выборке ===")
    for model_name, metrics in scores.items():
        print(f"[{model_name}] MAE={metrics['MAE']:.3f} | RMSE={metrics['RMSE']:.3f} | MAPE={metrics['MAPE_percent']:.3f}%")
    print(f"Лучшая модель: {best_model_name}")
    print(f"Артефакты сохранены в: {args.output_dir}")


def predict_command(args: argparse.Namespace):
    model = joblib.load(Path(args.model_dir) / "model.joblib")
    meta = json.loads((Path(args.model_dir) / "metadata.json").read_text(encoding="utf-8"))

    df = read_dataset_better(args.input)
    supervised_df, _, target_col, _ = build_features_better(
        df,
        horizon_steps=meta["horizon_steps"],
        lags=meta["lags"],
        rolling_windows=meta.get("rolling_windows", DEFAULT_ROLLING_WINDOWS),
    )

    feature_cols = meta["feature_columns"]
    missing = set(feature_cols) - set(supervised_df.columns)
    if missing:
        raise ValueError(f"Во входных данных не хватает признаков: {missing}")

    x = supervised_df[feature_cols]
    supervised_df["prediction_total_kw"] = model.predict(x)

    cols = ["timestamp", "total_kw", target_col, "prediction_total_kw"]
    supervised_df[cols].to_csv(args.output, index=False)
    print(f"Прогнозы сохранены в: {args.output}")


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Прогнозирование электрических нагрузок (регрессия)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Обучение модели")
    train_parser.add_argument("--input", required=True, help="Путь к CSV/XLSX датасету")
    train_parser.add_argument("--output-dir", default="artifacts", help="Директория для сохранения модели")
    train_parser.add_argument("--horizon-steps", type=int, default=1, help="Горизонт прогноза в шагах (1 шаг = 30 минут)")
    train_parser.add_argument("--lags", nargs="+", type=int, default=DEFAULT_LAGS, help="Лаги total_kw для признаков")
    train_parser.add_argument(
        "--rolling-windows",
        nargs="+",
        type=int,
        default=DEFAULT_ROLLING_WINDOWS,
        help="Окна для rolling-статистик total_kw",
    )
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
