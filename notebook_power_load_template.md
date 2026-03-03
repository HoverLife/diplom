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
def read_dataset(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        # Часто после выгрузок из сводных таблиц заголовок многоуровневый
        try:
            df = pd.read_excel(path, header=[0, 1])
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    " ".join([str(x).strip() for x in col if str(x) != "nan"]).strip()
                    for col in df.columns
                ]
        except Exception:
            df = pd.read_excel(path)
    else:
        # sep=None: pandas сам определит разделитель
        df = pd.read_csv(path, sep=None, engine="python")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x).strip() for x in col if str(x) != "nan"]).strip()
            for col in df.columns
        ]

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    # Поиск колонки времени
    dt_candidates = [
        c for c in df.columns
        if any(token in str(c).lower() for token in ["date", "time", "дата", "время", "названия строк"])
    ]
    dt_col = dt_candidates[0] if dt_candidates else df.columns[0]

    df = df.rename(columns={dt_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Чистка чисел
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


def find_kw_columns(df: pd.DataFrame):
    kw_cols = [c for c in df.columns if c != "timestamp" and "kw" in c.lower() and "kvar" not in c.lower()]
    if not kw_cols:
        # fallback: если в выгрузке нет явных 'kw' в имени
        kw_cols = [c for c in df.columns if c != "timestamp"]
    return kw_cols
```

---

## Cell 4 — Feature engineering

```python
def build_features(df: pd.DataFrame, horizon_steps: int, lags: list[int], rolling_windows: list[int]):
    out = df.copy()

    kw_cols = find_kw_columns(out)
    out["total_kw"] = out[kw_cols].sum(axis=1)

    # Календарные признаки
    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute
    out["dow"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    # Циклическое кодирование
    out["hour_sin"] = np.sin(2 * np.pi * (out["hour"] + out["minute"] / 60) / 24)
    out["hour_cos"] = np.cos(2 * np.pi * (out["hour"] + out["minute"] / 60) / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)

    # Лаги total_kw
    for lag in lags:
        out[f"total_kw_lag_{lag}"] = out["total_kw"].shift(lag)

    # Скользящие статистики только по прошлому (shift(1) убирает утечку)
    for w in rolling_windows:
        out[f"total_kw_roll_mean_{w}"] = out["total_kw"].shift(1).rolling(w).mean()
        out[f"total_kw_roll_std_{w}"] = out["total_kw"].shift(1).rolling(w).std()

    target_col = f"target_total_kw_t_plus_{horizon_steps}"
    out[target_col] = out["total_kw"].shift(-horizon_steps)

    out = out.dropna().reset_index(drop=True)

    feature_cols = [c for c in out.columns if c not in ["timestamp", target_col]]
    return out, feature_cols, target_col
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
df_raw = read_dataset(DATA_PATH)
df_model, feature_cols, target_col = build_features(
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
