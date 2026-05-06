# TABD — TFT-анализ сети АЗС Татнефть

Магистратура ТАБД, Задание 6: обучение Temporal Fusion Transformer на данных
сети АЗС, прогнозирование продаж топлива и сопутствующих товаров,
интерактивный дашборд.

---

## Требования

- **Windows 10/11** + Anaconda
- **NVIDIA GPU** с CUDA (тестировано на RTX 4060 Ti, CUDA 12.4)
- ~15 ГБ свободного места

---

## Установка (один раз)

```powershell
cd D:\project\TABD
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

Скрипт создаст conda-окружение `tabd_tft` (Python 3.11), установит
PyTorch 2.5.1 + CUDA 12.4 и все зависимости (~15-20 минут).

---

## Расположение данных

Положите CSV-файлы датасета (`detailed_data.csv`, `5stations_data.csv`,
`stations_metadata.csv`, `5stations_metadata.csv`) в одно из двух мест:

1. **Папка `data/` в корне проекта** (по умолчанию).
2. **Любая другая папка** — укажите путь через переменную окружения:

```powershell
$env:TABD_DATA_DIR = "D:\path\to\csv"
```

После первого обучения данные кэшируются в `models/data_cache.parquet`,
и для запуска дашборда CSV-файлы больше не требуются.

---

## Ручной запуск

В PowerShell задайте путь к Python из conda-окружения (один раз на сессию):

```powershell
$py = "C:\Users\Admin\anaconda3\envs\tabd_tft\python.exe"
cd D:\project\TABD
```

### 1. Обучение модели (~4-8 часов на полном датасете)

```powershell
& $py run.py --mode train
```

Создаст:
- `models/tft-epoch=XX-val_loss=YY.YYYY.ckpt` — лучший чекпоинт
- `models/last.ckpt` — последний чекпоинт (для дообучения)
- `models/training_meta.json` — метаданные (путь к лучшему чекпоинту)
- `models/data_cache.parquet` — кэш предобработанных данных

Быстрый тест на 5 АЗС:

```powershell
& $py run.py --mode train --quick
```

### 2. Генерация прогнозов (нужна обученная модель)

```powershell
& $py run.py --mode predict
```

Использует **лучший чекпоинт** из `models/training_meta.json` и кэш
`models/data_cache.parquet`. Создаёт CSV-файлы прогнозов в
`outputs/forecasts/` для всех 9 целевых переменных
(P10 / медиана / P90).

Использовать конкретный чекпоинт:

```powershell
& $py run.py --mode predict --checkpoint "models\tft-epoch=04-val_loss=39.8722.ckpt"
```

### 3. Запуск дашборда

```powershell
& $py run.py --mode dashboard
```

Открыть в браузере: **http://localhost:8050**

Дашборд читает данные из parquet-кэша и прогнозы из `outputs/forecasts/`
(если они есть). Без обученной модели — отображается аналитика по сырым
данным; с моделью — добавляются графики прогнозов и важности факторов.

### 4. Полный pipeline

```powershell
& $py run.py --mode all
```

Эквивалент `train` → `predict` → `dashboard`.

---

## Использование частично обученной модели

Если в `models/` уже лежит чекпоинт `.ckpt` от предыдущего обучения:

1. **Прогнозы** — будут сгенерированы из лучшего чекпоинта:
   ```powershell
   & $py run.py --mode predict
   ```
2. **Дашборд** — покажет графики прогнозов из `outputs/forecasts/`:
   ```powershell
   & $py run.py --mode dashboard
   ```
3. **Дообучение** — запустить обучение заново; ckpt-файлы сохраняются
   автоматически и `last.ckpt` можно использовать как стартовую точку
   (Lightning умеет резюмировать обучение через `trainer.fit(ckpt_path=...)`).

---

## Структура проекта

```
TABD/
├── src/
│   ├── config.py        # пути, гиперпараметры, целевые переменные
│   ├── data_loader.py   # загрузка CSV, feature engineering, TimeSeriesDataSet
│   ├── train.py         # обучение TFT с GPU, чекпоинты, early stopping
│   ├── predict.py       # инференс, квантильные прогнозы, важность факторов
│   └── dashboard.py     # Dash-дашборд (5 вкладок)
├── data/                # CSV-файлы (положить сюда или указать TABD_DATA_DIR)
├── models/              # чекпоинты + parquet-кэш
├── outputs/forecasts/   # CSV с прогнозами по каждому таргету
├── logs/                # CSVLogger логи обучения
├── run.py               # главная точка входа
├── run.ps1              # обёртка PowerShell
├── setup_env.ps1        # установка окружения
└── requirements.txt
```

---

## Основные настройки (`src/config.py`)

```python
USE_5_STATIONS = False          # True для быстрого теста на 5 АЗС
MAX_PREDICTION_LENGTH = 24      # горизонт прогноза (часов)
MAX_ENCODER_LENGTH = 7 * 24    # окно контекста (1 неделя)
BATCH_SIZE = 64
MAX_EPOCHS = 30                 # с early stopping (patience=8)
HIDDEN_SIZE = 64                # увеличить до 128 для лучшего качества
LEARNING_RATE = 3e-3
```

---

## Документация по показателям и дашборду

Подробное описание всех 9 целевых переменных, типов прогнозов
(P10 / медиана / P90), вкладок дашборда и интерактивных графиков —
в отдельном файле: **[DOCS.md](DOCS.md)**.

---

## Известные особенности

- **Кэш данных в parquet** — `models/data_cache.parquet` создаётся при `train`
  и используется при `predict` и `dashboard` (обходит Windows DLL-конфликт
  между CUDA и pandas при чтении больших CSV).
- **numpy 1.26.x обязателен** — numpy 2.x несовместим с
  pytorch-forecasting 1.7.
- Лучший checkpoint автоматически записывается в
  `models/training_meta.json` — именно его подхватывает `predict`.

---

## Полезное

```powershell
# Активировать окружение в текущем PS
conda activate tabd_tft

# Проверить GPU
& $py -c "import torch; print(torch.cuda.get_device_name(0))"

# Логи обучения
ls logs\tft\version_*\metrics.csv
```

---

## Стек

- **PyTorch 2.5.1** + CUDA 12.4
- **pytorch-forecasting 1.7.0** (TFT, MultiLoss + QuantileLoss)
- **Lightning 2.6** (тренер с callbacks)
- **Dash 4.1** + **Plotly 6.7** + **dash-bootstrap-components** (тёмная тема)
- **pandas 3.0** + **numpy 1.26**
