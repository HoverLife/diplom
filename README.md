# diplom

Скрипт для задачи регрессии: прогноз общего активного энергопотребления `total_kw`
на горизонте `t + h` (по умолчанию `h=1` шаг = 30 минут).

## Почему не логистическая регрессия
Логистическая регрессия применяется для **классификации**, а у вас задача **регрессии**
(непрерывная величина нагрузки в kW). Поэтому в коде используются:
- `LinearRegression` — максимально объяснимая базовая модель.
- `RandomForestRegressor` — более гибкая модель, но все еще интерпретируемая на уровне важности признаков.

## Что делает пайплайн
1. Загружает CSV/XLSX.
2. Нормализует заголовки, парсит timestamp.
3. Ищет столбцы `kw` и считает:
   - `total_kw = сумма kw по всем точкам фиксации`.
4. Добавляет признаки времени (час, день недели, синус/косинус).
5. Добавляет лаги `total_kw` (`1`, `2`, `48` по умолчанию).
6. Формирует target: `total_kw` через `h` шагов вперед.
7. Делит выборку по времени (без перемешивания).
8. Обучает 2 модели, сравнивает MAE/RMSE/MAPE и сохраняет лучшую.

## Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Обучение
```bash
python power_load_regression.py train \
  --input data.xlsx \
  --output-dir artifacts \
  --horizon-steps 1 \
  --lags 1 2 48 \
  --test-size 0.2
```

## Прогноз
```bash
python power_load_regression.py predict \
  --input data.xlsx \
  --model-dir artifacts \
  --output predictions.csv
```

## Выходные артефакты
- `artifacts/model.joblib` — лучшая модель.
- `artifacts/metadata.json` — список признаков, метрики, параметры горизонта и лагов.
- `predictions.csv` — `timestamp`, фактический `total_kw`, target и прогноз.
