# TABD — TFT-анализ сети АЗС Татнефть

Магистратура ТАБД, Задание 6: обучение Temporal Fusion Transformer на данных
сети АЗС, прогнозирование продаж топлива и сопутствующих товаров,
интерактивный дашборд.

---

## Требования

- **Windows 10/11** + Anaconda
- **NVIDIA GPU** с CUDA (тестировано на RTX 4060 Ti, CUDA 12.4)
- ~15 ГБ свободного места на диске
- Данные: `C:\Users\Admin\Desktop\ОбучениеМагистратура\ТАБД\Задание 6\`

---

## Установка (один раз)

```powershell
cd D:\project\TABD
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

Скрипт создаст conda-окружение `tabd_tft` (Python 3.11), установит
PyTorch 2.5.1 + CUDA 12.4 и все зависимости (~15-20 минут).

---

## Запуск

### Через `run.py` (рекомендуется)

```powershell
$py = "C:\Users\Admin\anaconda3\envs\tabd_tft\python.exe"

# Полный pipeline: обучение → прогнозы → дашборд
& $py run.py --mode all

# Только обучение (~4-8 часов на 25 АЗС, или ~1-2 ч с --quick)
& $py run.py --mode train

# Только генерация прогнозов (требует обученную модель)
& $py run.py --mode predict

# Только дашборд (без модели — показывает данные и аналитику)
& $py run.py --mode dashboard

# Быстрый тест на 5 АЗС
& $py run.py --mode all --quick
```

Дашборд открывается на **http://localhost:8050**

---

## Структура проекта

```
TABD/
├── src/
│   ├── config.py        # пути, гиперпараметры TFT, целевые переменные
│   ├── data_loader.py   # загрузка CSV, feature engineering, TimeSeriesDataSet
│   ├── train.py         # обучение TFT с GPU, чекпоинты, early stopping
│   ├── predict.py       # инференс, квантильные прогнозы, важность факторов
│   └── dashboard.py     # Dash-дашборд (5 вкладок)
├── data/                # README — данные читаются из исходной папки
├── models/              # сохранённые чекпоинты + parquet-кэш данных
├── outputs/forecasts/   # CSV с прогнозами по каждому таргету
├── logs/                # CSVLogger логи обучения
├── run.py               # главный entry point
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

## Целевые переменные (9)

Модель прогнозирует одновременно:

- `total_fuel_sales` — общие продажи топлива (л/ч)
- `sales_AI92`, `sales_AI95`, `sales_AI98` — продажи бензина по маркам
- `sales_DT_EURO`, `sales_DT_TANEKO`, `sales_DT_SUMMER`, `sales_DT_WINTER` — дизель
- `shop_total_revenue` — выручка магазина (руб/ч)

Для каждого таргета — квантильный прогноз (P10 / медиана / P90).

---

## Дашборд — вкладки

1. **Обзор сети** — KPI, динамика продаж, структура трафика, heatmap по часам
2. **Анализ АЗС** — выбор станции, паттерны (час/день/погода)
3. **Прогнозы TFT** — факт vs прогноз с интервалом неопределённости
4. **Факторный анализ** — важность признаков, эффект акций, конкурентные цены
5. **Рекомендации** — автоматические инсайты на основе модели

---

## Известные особенности

- **Кэш данных в parquet** — `models/data_cache.parquet` создаётся при `train`
  и используется при `predict` (обходит Windows DLL-конфликт между CUDA
  и pandas при чтении больших CSV)
- **numpy 1.26.x обязателен** — numpy 2.x несовместим с pytorch-forecasting 1.7
- Модель сохраняется автоматически (best + last). Лучший checkpoint
  записывается в `models/training_meta.json`

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
