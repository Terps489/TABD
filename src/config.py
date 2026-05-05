from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\Admin\Desktop\ОбучениеМагистратура\ТАБД\Задание 6")
PROJECT_DIR = Path(r"D:\project\TABD")
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
LOGS_DIR = PROJECT_DIR / "logs"

DETAILED_DATA = DATA_DIR / "detailed_data.csv"
STATIONS_META = DATA_DIR / "stations_metadata.csv"
FIVE_STATIONS_DATA = DATA_DIR / "5stations_data.csv"
FIVE_STATIONS_META = DATA_DIR / "5stations_metadata.csv"
DATA_CACHE = MODELS_DIR / "data_cache.parquet"  # preprocessed data cache

# ── Mode ───────────────────────────────────────────────────────────────────────
# True = 5 stations (fast test ~5 min), False = 25 stations (full, ~30-60 min)
USE_5_STATIONS = False

# ── TFT Hyperparameters ────────────────────────────────────────────────────────
MAX_PREDICTION_LENGTH = 24       # hours ahead to forecast
MAX_ENCODER_LENGTH = 7 * 24     # context window: 1 week
BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 3e-3
HIDDEN_SIZE = 64
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 32
GRADIENT_CLIP_VAL = 0.1

# Validation: last 7 days of data
VAL_SPLIT_HOURS = 7 * 24

# ── Target variables ───────────────────────────────────────────────────────────
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

# ── Dashboard ──────────────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = False
