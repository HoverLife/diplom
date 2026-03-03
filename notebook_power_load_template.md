# Шаблон для Jupyter Notebook: прогноз электрической нагрузки (регрессия)

Ниже готовые ячейки, которые можно вставить в ваш `ipynb`.

---

## Cell 1 — Импорты

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
```

---

## Cell 2 — Настройки

```python
# Путь к вашему датасету
DATA_PATH = "data.xlsx"   # или "data.csv"

# 1 шаг = 30 минут в вашем датасете
HORIZON_STEPS = 1          # прогноз на следующий шаг
LAGS = [1, 2, 48]          # 30 мин, 1 час, сутки
ROLLING_WINDOWS = [4, 48]  # 2 часа и сутки

# Размер финального holdout (последние 20% по времени)
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

---

## Cell 3 — Функции загрузки и очистки

```python
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Утилиты для очистки имён
# -------------------------
def _slugify(s: str) -> str:
    """Сделать строку безопасной для имени столбца: убрать лишние символы, заменить пробелы на _"""
    if s is None:
        return ""
    s = str(s)
    s = s.strip()
    s = re.sub(r"[\/\|\(\)\[\]\:;,]+", " ", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"__+", "_", s)
    s = re.sub(r"[^0-9A-Za-zА-Яа-я_\-]", "", s)
    return s.strip("_") or "col"


def _detect_measure_token(token: str) -> str | None:
    """Попытаться распознать, это kw или kvar (или None)."""
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


def read_dataset_better(path: str) -> pd.DataFrame:
    """Улучшенное чтение с интерпретацией multi-header и Unnamed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = None
    try:
        df = pd.read_excel(path, header=[0, 1])
    except Exception:
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")

    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        levels = [pd.Series(df.columns.get_level_values(i)).astype(str) for i in range(df.columns.nlevels)]
        for i in range(len(levels)):
            lvl = levels[i].mask(levels[i].str.strip().str.lower().map(lambda v: v.startswith("unnamed:") if isinstance(v, str) else False))
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
        c for c in df.columns
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
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    numeric_cols = [c for c in df.columns if c == "timestamp" or pd.api.types.is_numeric_dtype(df[c])]
    df = df[numeric_cols]

    if len(df.columns) < 3:
        raise ValueError("После очистки слишком мало признаков. Проверьте формат выгрузки.")

    return df
```

---

## Cell 4 — Feature engineering (обновлённый)

```python
def find_kw_columns_better(df: pd.DataFrame) -> list[str]:
    cand = [c for c in df.columns if c != "timestamp" and "__kw" in c]
    if not cand:
        cand = [c for c in df.columns if c != "timestamp" and re.search(r"\bkw\b", c, flags=re.I)]
    if not cand:
        cand = [c for c in df.columns if c != "timestamp"]
    return cand


def build_features_better(df: pd.DataFrame, horizon_steps: int, lags: list[int], rolling_windows: list[int]):
    out = df.copy()
    kw_cols = find_kw_columns_better(out)

    kw_only = [c for c in kw_cols if "__kw" in c or re.search(r"\bkw\b", c, flags=re.I)]
    kvar_cols = [c for c in kw_cols if "__kvar" in c or re.search(r"\bkvar\b", c, flags=re.I)]
    used_kw_cols = kw_only if kw_only else [c for c in out.columns if c != "timestamp"]

    out["total_kw"] = out[used_kw_cols].sum(axis=1)

    for c in used_kw_cols:
        out[f"{c}__share"] = np.where(out["total_kw"].abs() > 0, out[c] / out["total_kw"], 0.0)

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
        "orig_columns": list(df.columns)
    }
    return out, feature_cols, target_col, meta


def plot_total_with_roll(df: pd.DataFrame, roll_window: int = 48, show_legend: bool = True):
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
```

---

## Cell 5 — Метрики и оценка (TimeSeriesSplit + holdout)

```python
def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE_%": float(mape)}


def evaluate_with_tscv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        m = regression_metrics(y_val, pred)
        m["fold"] = fold
        rows.append(m)

    fold_df = pd.DataFrame(rows)
    avg = fold_df[["MAE", "RMSE", "MAPE_%"]].mean().to_dict()
    return fold_df, avg
```

---

## Cell 6 — Подготовка данных

```python
df_raw = read_dataset_better(DATA_PATH)
df_model, feature_cols, target_col, meta = build_features_better(
    df_raw,
    horizon_steps=HORIZON_STEPS,
    lags=LAGS,
    rolling_windows=ROLLING_WINDOWS,
)

# Хронологический split на финальный test
split_idx = int(len(df_model) * (1 - TEST_SIZE))
train_df = df_model.iloc[:split_idx].copy()
test_df = df_model.iloc[split_idx:].copy()

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("Train size:", len(train_df))
print("Test size:", len(test_df))
print("Features:", len(feature_cols))
print("Используемые kw-колонки:", meta["kw_cols"])
```

---

## Cell 7 — Сравнение моделей на кросс-валидации

```python
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

cv_summary = {}
for name, model in models.items():
    fold_df, avg = evaluate_with_tscv(model, X_train, y_train, n_splits=5)
    cv_summary[name] = avg
    print(f"\n{name} — средние CV-метрики")
    print(pd.Series(avg))

cv_table = pd.DataFrame(cv_summary).T.sort_values("MAE")
cv_table
```

---

## Cell 8 — Финальная проверка на отложенном test

```python
results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    m = regression_metrics(y_test, pred_test)
    m["model"] = name
    results.append(m)
    fitted_models[name] = model

test_table = pd.DataFrame(results).set_index("model").sort_values("MAE")
test_table
```

---

## Cell 9 — Выбор лучшей модели и сохранение

```python
best_model_name = test_table.index[0]
best_model = fitted_models[best_model_name]

print("Лучшая модель:", best_model_name)
print(test_table.loc[best_model_name])

# Сохраняем модель + метаданные
import joblib

ARTIFACTS_DIR = Path("artifacts_notebook")
ARTIFACTS_DIR.mkdir(exist_ok=True)

joblib.dump(best_model, ARTIFACTS_DIR / "model.joblib")

metadata = {
    "best_model": best_model_name,
    "feature_cols": feature_cols,
    "target_col": target_col,
    "horizon_steps": HORIZON_STEPS,
    "lags": LAGS,
    "rolling_windows": ROLLING_WINDOWS,
    "cv_metrics": cv_table.to_dict(orient="index"),
    "test_metrics": test_table.to_dict(orient="index"),
}
(ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Сохранено в: {ARTIFACTS_DIR.resolve()}")
```

---

## Cell 10 — Прогнозы на test и экспорт

```python
test_pred = best_model.predict(X_test)

pred_df = pd.DataFrame({
    "timestamp": test_df["timestamp"].values,
    "actual_total_kw_t_plus_h": y_test.values,
    "pred_total_kw_t_plus_h": test_pred,
})

pred_df["abs_error"] = (pred_df["actual_total_kw_t_plus_h"] - pred_df["pred_total_kw_t_plus_h"]).abs()
pred_df["ape_%"] = pred_df["abs_error"] / np.clip(pred_df["actual_total_kw_t_plus_h"].abs(), 1e-6, None) * 100

pred_df.to_csv("predictions_test.csv", index=False)
pred_df.head()
```

---

## Как оценивать модели «правильнее»

1. **Обязательно хронологическое разделение**, без перемешивания.
2. **TimeSeriesSplit** для выбора модели и гиперпараметров на train-части.
3. **Отдельный финальный holdout** (последний кусок ряда), на котором вы показываете итоговые метрики в дипломе.
4. Основная метрика — **MAE** (интерпретируема в kW), дополнительно RMSE и MAPE.
5. Для объяснимости:
   - для `LinearRegression` показывайте коэффициенты;
   - для `RandomForest` — `feature_importances_`.

---

## Cell 11 — График факта и прогноза (главный для защиты)

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

plt.figure(figsize=(16, 6))
plt.plot(pred_df["timestamp"], pred_df["actual_total_kw_t_plus_h"], label="Факт", linewidth=2)
plt.plot(pred_df["timestamp"], pred_df["pred_total_kw_t_plus_h"], label="Прогноз", linewidth=2, alpha=0.9)
plt.title("Прогноз суммарной активной мощности на test-периоде")
plt.xlabel("Время")
plt.ylabel("Мощность, kW")
plt.legend()
plt.tight_layout()
plt.savefig("fig_01_actual_vs_pred.png", dpi=200)
plt.show()
```

---

## Cell 12 — Ошибка во времени и распределение ошибок

```python
pred_df["error"] = pred_df["actual_total_kw_t_plus_h"] - pred_df["pred_total_kw_t_plus_h"]

fig, axes = plt.subplots(1, 2, figsize=(18, 5))

axes[0].plot(pred_df["timestamp"], pred_df["error"], color="tab:red")
axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_title("Ошибка прогноза во времени (actual - pred)")
axes[0].set_xlabel("Время")
axes[0].set_ylabel("Ошибка, kW")

sns.histplot(pred_df["abs_error"], bins=40, kde=True, ax=axes[1], color="tab:blue")
axes[1].set_title("Распределение абсолютной ошибки")
axes[1].set_xlabel("|Ошибка|, kW")

plt.tight_layout()
plt.savefig("fig_02_error_analysis.png", dpi=200)
plt.show()
```

---

## Cell 13 — Сравнение моделей по метрикам

```python
plot_metrics = test_table[["MAE", "RMSE", "MAPE_%"]].copy()
plot_metrics.plot(kind="bar", figsize=(10, 5), rot=0)
plt.title("Сравнение моделей на финальном test")
plt.ylabel("Значение метрики")
plt.tight_layout()
plt.savefig("fig_03_models_metrics.png", dpi=200)
plt.show()
```

---

## Cell 14 — Интерпретация моделей

```python
# 1) Важность признаков RandomForest
if "RandomForest" in fitted_models:
    rf = fitted_models["RandomForest"]
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 7))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Top-20 важностей признаков (RandomForest)")
    plt.xlabel("Feature importance")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.savefig("fig_04_rf_importance.png", dpi=200)
    plt.show()

# 2) Коэффициенты LinearRegression
if "LinearRegression" in fitted_models:
    lr = fitted_models["LinearRegression"]
    coefs = pd.Series(lr.coef_, index=feature_cols).sort_values()

    top_neg = coefs.head(10)
    top_pos = coefs.tail(10)
    coef_plot = pd.concat([top_neg, top_pos])

    plt.figure(figsize=(10, 7))
    sns.barplot(x=coef_plot.values, y=coef_plot.index)
    plt.title("Ключевые коэффициенты LinearRegression")
    plt.xlabel("Коэффициент")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.savefig("fig_05_lr_coefficients.png", dpi=200)
    plt.show()
```

---

## Cell 15 — Тепловая карта ошибок (час × день недели)

```python
heat_df = pred_df.copy()
heat_df["hour"] = pd.to_datetime(heat_df["timestamp"]).dt.hour
heat_df["dow"] = pd.to_datetime(heat_df["timestamp"]).dt.dayofweek

pivot = heat_df.pivot_table(
    index="dow",
    columns="hour",
    values="abs_error",
    aggfunc="mean"
)

plt.figure(figsize=(14, 5))
sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label": "Средняя |ошибка|, kW"})
plt.title("Тепловая карта ошибок: день недели × час")
plt.xlabel("Час")
plt.ylabel("День недели (0=Пн)")
plt.tight_layout()
plt.savefig("fig_06_error_heatmap.png", dpi=200)
plt.show()
```

---

## Cell 16 — Автоматическая таблица для вставки в диплом

```python
summary = {
    "best_model": best_model_name,
    "best_MAE_kW": float(test_table.loc[best_model_name, "MAE"]),
    "best_RMSE_kW": float(test_table.loc[best_model_name, "RMSE"]),
    "best_MAPE_percent": float(test_table.loc[best_model_name, "MAPE_%"]),
    "train_size": int(len(train_df)),
    "test_size": int(len(test_df)),
    "horizon_steps": int(HORIZON_STEPS),
    "n_features": int(len(feature_cols)),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv("table_01_model_summary.csv", index=False)
summary_df
```

> После выполнения Cells 11–16 в папке проекта будут готовые PNG-графики и CSV-таблица, которые можно напрямую вставлять в презентацию и диплом.
