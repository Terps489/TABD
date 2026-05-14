import os
from pathlib import Path

# ── Пути ───────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Путь к исходным CSV-данным. Настраивается через переменную окружения TABD_DATA_DIR.
# По умолчанию — папка ./data в корне проекта.
DATA_DIR = Path(os.environ.get("TABD_DATA_DIR", PROJECT_DIR / "data"))

MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
LOGS_DIR = PROJECT_DIR / "logs"

DETAILED_DATA = DATA_DIR / "detailed_data.csv"
STATIONS_META = DATA_DIR / "stations_metadata.csv"
FIVE_STATIONS_DATA = DATA_DIR / "5stations_data.csv"
FIVE_STATIONS_META = DATA_DIR / "5stations_metadata.csv"
DATA_CACHE = MODELS_DIR / "data_cache.parquet"  # кэш предобработанных данных

# ── Режим ──────────────────────────────────────────────────────────────────────
# True = 5 АЗС (быстрый тест ~5 мин), False = 25 АЗС (полный, ~30-60 мин)
USE_5_STATIONS = False

# ── Гиперпараметры TFT ─────────────────────────────────────────────────────────
MAX_PREDICTION_LENGTH = 24       # горизонт прогноза (часов вперёд)
MAX_ENCODER_LENGTH = 7 * 24     # окно контекста: 1 неделя
BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 3e-3
HIDDEN_SIZE = 64
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 32
GRADIENT_CLIP_VAL = 0.1

# Валидация: последние 7 дней данных
VAL_SPLIT_HOURS = 7 * 24

# ── Целевые переменные ─────────────────────────────────────────────────────────
TARGETS = [
    "total_fuel_sales",
    "sales_AI92",
    "sales_AI95",
    "sales_AI98",
    "sales_DT_EURO",
    "sales_DT_TANEKO",
    "sales_DT_SUMMER",
    "sales_DT_WINTER",
    "shop_total_revenue",
]

# ── Дашборд ────────────────────────────────────────────────────────────────────
DASHBOARD_HOST = os.environ.get("TABD_DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.environ.get("TABD_DASHBOARD_PORT", 8050))
DASHBOARD_DEBUG = False

# График прогноза (вкладка «Прогнозы TFT») — продвинутые настройки.
# Сколько часов факта рисовать слева от "сейчас". Удобно поставить
# вплоть до MAX_ENCODER_LENGTH (168), если хочется видеть всю
# историю, которую видит модель.
FORECAST_CHART_HISTORY_HOURS = 48
# Сколько часов прогноза рисовать справа. <=24 — берётся из готовых CSV
# (training-time валидация, быстро). >24 — запускается итеративный rollout
# через predict.forecast_extended; точность падает с горизонтом.
FORECAST_CHART_FUTURE_HOURS = 24
