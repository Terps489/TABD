"""Загрузка данных и подготовка TimeSeriesDataSet для TFT."""
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

from src.config import (
    DETAILED_DATA, FIVE_STATIONS_DATA,
    MAX_PREDICTION_LENGTH, MAX_ENCODER_LENGTH,
    USE_5_STATIONS, BATCH_SIZE, VAL_SPLIT_HOURS, TARGETS
)

# ── Списки признаков ───────────────────────────────────────────────────────────
STATIC_CATEGORICALS = ["road_type", "direction", "settlement_size", "station_name"]

STATIC_REALS = [
    "distance_to_city_km", "total_pumps", "shop_area_m2",
    "competitors_within_5km", "corporate_customer_ratio",
    "staff_engagement_score", "customer_loyalty_score",
    "has_car_wash", "has_cafe", "has_shop",
]

TIME_VARYING_KNOWN_CATEGORICALS = ["season"]

TIME_VARYING_KNOWN_REALS = [
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_holiday", "is_rush_hour", "is_night",
]

TIME_VARYING_UNKNOWN_REALS = [
    "temperature", "precipitation_mm", "visibility_km", "wind_speed_ms",
    "is_snow", "is_rain", "is_fog",
    "traffic_Passengers_cars", "traffic_Truck_short", "traffic_Truck",
    "traffic_Truck_long", "traffic_Transporter", "total_traffic",
    "promotion_fuel_active", "promotion_shop_active", "promotion_cafe_active",
    "ad_active",
    "competitor_price_AI92", "competitor_price_AI95", "competitor_price_DT",
    "price_AI92", "price_AI95", "price_DT_EURO",
]


def load_raw(use_5_stations: bool | None = None) -> pd.DataFrame:
    """Загрузить CSV и вернуть исходный DataFrame."""
    if use_5_stations is None:
        use_5_stations = USE_5_STATIONS
    path = FIVE_STATIONS_DATA if use_5_stations else DETAILED_DATA
    print(f"Загрузка: {path.name}  ({'5 АЗС' if use_5_stations else '25 АЗС'})")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"  Строк: {len(df):,}  |  Колонок: {df.shape[1]}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering и очистка данных."""
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # Целочисленный временной индекс (часов с начала)
    t0 = df["timestamp"].min()
    df["time_idx"] = ((df["timestamp"] - t0).dt.total_seconds() / 3600).astype(int)

    # Циклическое кодирование времени (избегает разрыва на границе суток/недели)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Бинарные колонки -> float
    bool_cols = [
        "is_snow", "is_rain", "is_fog", "is_weekend", "is_holiday",
        "is_rush_hour", "is_night", "has_car_wash", "has_cafe", "has_shop",
        "promotion_fuel_active", "promotion_shop_active", "promotion_cafe_active",
        "ad_active",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Строковые категориальные признаки
    for col in STATIC_CATEGORICALS + TIME_VARYING_KNOWN_CATEGORICALS:
        df[col] = df[col].astype(str)

    # ID группы должен быть строкой
    df["station_id"] = df["station_id"].astype(str)

    # Заполнение цен конкурентов медианой по станции
    for col in ["competitor_price_AI92", "competitor_price_AI95", "competitor_price_DT"]:
        if col in df.columns:
            df[col] = df.groupby("station_id")[col].transform(lambda s: s.fillna(s.median()))

    # Целевые переменные не могут быть отрицательными
    for col in TARGETS:
        df[col] = df[col].fillna(0).clip(lower=0)

    return df


def create_datasets(df: pd.DataFrame):
    """Возвращает (training_dataset, validation_dataset, train_loader, val_loader)."""
    cutoff = df["time_idx"].max() - VAL_SPLIT_HOURS

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= cutoff].copy(),
        time_idx="time_idx",
        target=TARGETS,
        group_ids=["station_id"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=STATIC_CATEGORICALS,
        static_reals=STATIC_REALS,
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS + TARGETS,
        target_normalizer=MultiNormalizer(
            [GroupNormalizer(groups=["station_id"], transformation="softplus") for _ in TARGETS]
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )

    train_loader = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=False
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE * 2, num_workers=0, pin_memory=True
    )

    print(f"Обучающая выборка: {len(training):,}  |  Валидация: {len(validation):,}")
    return training, validation, train_loader, val_loader
