from pathlib import Path

# ── Пути ───────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\Admin\Desktop\ОбучениеМагистратура\ТАБД\Задание 6")
PROJECT_DIR = Path(r"D:\project\TABD")
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
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = False
