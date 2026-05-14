"""Генерация полностью синтетического датасета АЗС за 2024 год.

Создаёт два CSV в формате, который ждёт `src/data_loader.py`:
- detailed_data.csv (5 АЗС × 8784 ч = 43 920 строк, 2024 — високосный)
- stations_metadata.csv

В данные заложены реалистичные паттерны:
- суточная сезонность (пики 7-9 и 17-19, ночной провал)
- недельная сезонность (выходные)
- годовая сезонность (температура; DT_SUMMER только тёплые месяцы, DT_WINTER только холодные)
- влияние погоды/трафика/акций/праздников на продажи
- статические характеристики станции (тип дороги, размер магазина, ...)

Использование:
    python scripts/make_synthetic.py
    python scripts/make_synthetic.py --stations 8 --out data/synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


RU_HOLIDAYS_2024 = [
    "2024-01-01", "2024-01-02", "2024-01-07",   # Новый год + Рождество
    "2024-02-23",                                # День защитника
    "2024-03-08",                                # 8 марта
    "2024-05-01", "2024-05-09",                  # Праздник весны + Победы
    "2024-06-12",                                # День России
    "2024-11-04",                                # День народного единства
    "2024-12-31",                                # Новый год
]


ROAD_TYPES = ["highway", "city", "regional"]
DIRECTIONS = ["north", "south", "east", "west"]
SETTLEMENT_SIZES = ["small", "medium", "large"]
WEATHER_CONDITIONS = ["clear", "cloudy", "rain", "snow", "fog"]


def _seasonal_temperature(day_of_year: np.ndarray, base: float, amp: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Годовой косинус: минимум в январе, максимум в июле + шум."""
    seasonal = base + amp * -np.cos(2 * np.pi * (day_of_year - 15) / 366)
    return seasonal + rng.normal(0, 3, size=day_of_year.shape)


def _build_station_meta(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for sid in range(n):
        rows.append({
            "station_id": sid,
            "station_name": f"Station_{sid + 1:02d}",
            "road_type": rng.choice(ROAD_TYPES, p=[0.5, 0.3, 0.2]),
            "direction": rng.choice(DIRECTIONS),
            "settlement_size": rng.choice(SETTLEMENT_SIZES, p=[0.3, 0.4, 0.3]),
            "distance_to_city_km": float(rng.uniform(2, 50)),
            "total_pumps": int(rng.integers(4, 13)),
            "shop_area_m2": float(rng.uniform(30, 180)),
            "competitors_within_5km": int(rng.integers(0, 6)),
            "corporate_customer_ratio": float(rng.uniform(0.05, 0.45)),
            "staff_engagement_score": float(rng.uniform(0.4, 0.95)),
            "customer_loyalty_score": float(rng.uniform(0.4, 0.9)),
            "has_car_wash": int(rng.random() < 0.6),
            "has_cafe": int(rng.random() < 0.45),
            "has_shop": 1,  # магазин есть всегда — в нём считаем shop_total_revenue
        })
    return pd.DataFrame(rows)


def _generate_one_station(
    meta_row: pd.Series,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(timestamps)
    hour = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    day = timestamps.day.to_numpy()
    month = timestamps.month.to_numpy()
    doy = timestamps.dayofyear.to_numpy()

    is_weekend = (dow >= 5).astype(float)
    is_rush_hour = np.isin(hour, [7, 8, 9, 17, 18, 19]).astype(float)
    is_night = ((hour < 6) | (hour >= 22)).astype(float)
    is_holiday = np.isin(
        timestamps.strftime("%Y-%m-%d"), RU_HOLIDAYS_2024
    ).astype(float)

    # Погода: температура зависит от дня года, осадки — стохастика
    temperature = _seasonal_temperature(doy, base=6.0, amp=18.0, rng=rng)
    precipitation_mm = np.clip(rng.gamma(0.4, 0.8, n) - 0.3, 0, None)
    wind_speed_ms = np.clip(rng.normal(4, 2, n), 0, None)
    visibility_km = np.clip(20 - precipitation_mm * 2 + rng.normal(0, 1, n),
                            0.2, 25)

    is_snow = ((temperature < 0) & (precipitation_mm > 0.1)).astype(float)
    is_rain = ((temperature >= 0) & (precipitation_mm > 0.1)).astype(float)
    is_fog = (visibility_km < 2).astype(float)

    weather_condition = np.full(n, "clear", dtype=object)
    weather_condition[is_rain == 1] = "rain"
    weather_condition[is_snow == 1] = "snow"
    weather_condition[is_fog == 1] = "fog"
    cloudy_mask = (weather_condition == "clear") & (rng.random(n) < 0.3)
    weather_condition[cloudy_mask] = "cloudy"

    # Базовая часовая интенсивность трафика
    hour_curve = np.array([
        0.25, 0.18, 0.15, 0.15, 0.22, 0.45,    # 0-5
        0.85, 1.40, 1.55, 1.20, 0.95, 1.00,    # 6-11
        1.05, 1.00, 0.95, 1.05, 1.30, 1.55,    # 12-17
        1.45, 1.20, 0.95, 0.75, 0.55, 0.35,    # 18-23
    ])
    week_mult = np.where(dow >= 5, 0.85, 1.0)  # выходные чуть тише в трассе
    holiday_mult = np.where(is_holiday == 1, 0.7, 1.0)
    road_mult = {"highway": 1.4, "city": 1.0, "regional": 0.7}[
        meta_row["road_type"]
    ]
    settlement_mult = {"small": 0.7, "medium": 1.0, "large": 1.4}[
        meta_row["settlement_size"]
    ]

    base = (hour_curve[hour] * week_mult * holiday_mult
            * road_mult * settlement_mult)

    # Трафик по типам
    cars = np.clip(base * 60 * (1 + rng.normal(0, 0.12, n)), 0, None)
    truck_short = np.clip(base * 10 * (1 + rng.normal(0, 0.2, n)), 0, None)
    truck = np.clip(base * 7 * (1 + rng.normal(0, 0.2, n)), 0, None)
    truck_long = np.clip(base * 5 * road_mult * 0.6
                          * (1 + rng.normal(0, 0.25, n)), 0, None)
    transporter = np.clip(base * 3 * (1 + rng.normal(0, 0.25, n)), 0, None)
    total_traffic = cars + truck_short + truck + truck_long + transporter

    # Промо и реклама — пуассон-процесс
    promo_fuel = (rng.random(n) < 0.05).astype(float)
    promo_shop = (rng.random(n) < 0.06).astype(float)
    promo_cafe = (rng.random(n) < 0.04).astype(float)
    ad_active = (rng.random(n) < 0.10).astype(float)

    # Цены (рубли)
    price_AI92 = 51 + rng.normal(0, 0.7, n).cumsum() * 0.01
    price_AI95 = price_AI92 + 2.2 + rng.normal(0, 0.1, n)
    price_AI98 = price_AI95 + 4.0 + rng.normal(0, 0.15, n)
    price_DT_EURO = 60 + rng.normal(0, 0.7, n).cumsum() * 0.01
    competitor_price_AI92 = price_AI92 + rng.normal(0, 1.0, n)
    competitor_price_AI95 = price_AI95 + rng.normal(0, 1.0, n)
    competitor_price_DT = price_DT_EURO + rng.normal(0, 1.0, n)

    # Топливо: конверсия трафика в литры
    conv_passenger = 35.0   # средний чек легковушки, литров
    conv_truck = 90.0
    sales_fuel_pool = (
        cars * conv_passenger * 0.08 +
        (truck_short + truck) * conv_truck * 0.10 +
        (truck_long + transporter) * conv_truck * 0.15
    )

    # Эффект промо/рекламы/конкурентных цен/погоды
    promo_effect = 1.0 + 0.18 * promo_fuel + 0.06 * ad_active
    price_gap = (competitor_price_AI95 - price_AI95)  # +ve = мы дешевле
    price_effect = 1.0 + np.clip(price_gap / 10.0, -0.15, 0.15)
    weather_drag = 1.0 - 0.08 * is_snow - 0.04 * is_rain - 0.03 * is_fog
    customer_lift = 0.8 + 0.4 * float(meta_row["customer_loyalty_score"])

    sales_fuel_pool = (sales_fuel_pool * promo_effect * price_effect
                       * weather_drag * customer_lift)
    sales_fuel_pool *= (1 + rng.normal(0, 0.08, n))
    sales_fuel_pool = np.clip(sales_fuel_pool, 0, None)

    # Структура: бензины ~70%, дизель ~30%
    share_AI92 = 0.27
    share_AI95 = 0.38
    share_AI98 = 0.05
    share_DT = 0.30
    sales_AI92 = sales_fuel_pool * share_AI92
    sales_AI95 = sales_fuel_pool * share_AI95
    sales_AI98 = sales_fuel_pool * share_AI98

    # DT: разделяем между TANEKO/EURO/SUMMER/WINTER по сезону
    dt_total = sales_fuel_pool * share_DT
    # Зимний дизель — холодные месяцы; летний — тёплые
    is_summer = np.isin(month, [5, 6, 7, 8, 9]).astype(float)
    is_winter = np.isin(month, [11, 12, 1, 2, 3]).astype(float)
    sales_DT_SUMMER = dt_total * 0.30 * is_summer
    sales_DT_WINTER = dt_total * 0.30 * is_winter
    # Остаток между EURO и TANEKO
    dt_rest = dt_total - sales_DT_SUMMER - sales_DT_WINTER
    sales_DT_EURO = dt_rest * 0.55
    sales_DT_TANEKO = dt_rest * 0.45

    total_fuel_sales = (sales_AI92 + sales_AI95 + sales_AI98 +
                        sales_DT_EURO + sales_DT_TANEKO +
                        sales_DT_SUMMER + sales_DT_WINTER)

    # Магазин: коррелирует с трафиком и наличием кафе
    shop_base = total_traffic * 8.0 * (1 + 0.35 * meta_row["has_cafe"])
    shop_base *= (1.0 + 0.22 * promo_shop + 0.05 * ad_active)
    shop_base *= (0.85 + 0.30 * meta_row["customer_loyalty_score"])
    shop_drinks = shop_base * 0.30 * (1 + rng.normal(0, 0.1, n))
    shop_snacks = shop_base * 0.22 * (1 + rng.normal(0, 0.1, n))
    shop_auto = shop_base * 0.10 * (1 + rng.normal(0, 0.15, n))
    shop_coffee = shop_base * 0.25 * (1 + rng.normal(0, 0.12, n)) \
        * (1 + 0.3 * meta_row["has_cafe"])
    shop_tobacco = shop_base * 0.13 * (1 + rng.normal(0, 0.12, n))
    shop_total_revenue = (shop_drinks + shop_snacks + shop_auto
                          + shop_coffee + shop_tobacco)
    shop_total_revenue = np.clip(shop_total_revenue, 0, None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "station_id": meta_row["station_id"],
        "station_name": meta_row["station_name"],
        "road_type": meta_row["road_type"],
        "direction": meta_row["direction"],
        "settlement_size": meta_row["settlement_size"],
        "distance_to_city_km": meta_row["distance_to_city_km"],
        "total_pumps": meta_row["total_pumps"],
        "shop_area_m2": meta_row["shop_area_m2"],
        "competitors_within_5km": meta_row["competitors_within_5km"],
        "corporate_customer_ratio": meta_row["corporate_customer_ratio"],
        "staff_engagement_score": meta_row["staff_engagement_score"],
        "customer_loyalty_score": meta_row["customer_loyalty_score"],
        "has_car_wash": meta_row["has_car_wash"],
        "has_cafe": meta_row["has_cafe"],
        "has_shop": meta_row["has_shop"],
        "hour": hour,
        "day_of_week": dow,
        "day": day,
        "month": month,
        "season": pd.Series(month).map(
            {12: "winter", 1: "winter", 2: "winter",
             3: "spring", 4: "spring", 5: "spring",
             6: "summer", 7: "summer", 8: "summer",
             9: "autumn", 10: "autumn", 11: "autumn"}
        ).values,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_rush_hour": is_rush_hour,
        "is_night": is_night,
        "temperature": temperature.round(1),
        "precipitation_mm": precipitation_mm.round(2),
        "visibility_km": visibility_km.round(2),
        "wind_speed_ms": wind_speed_ms.round(2),
        "is_snow": is_snow,
        "is_rain": is_rain,
        "is_fog": is_fog,
        "weather_condition": weather_condition,
        "traffic_Passengers_cars": cars.round(1),
        "traffic_Truck_short": truck_short.round(1),
        "traffic_Truck": truck.round(1),
        "traffic_Truck_long": truck_long.round(1),
        "traffic_Transporter": transporter.round(1),
        "total_traffic": total_traffic.round(1),
        "promotion_fuel_active": promo_fuel,
        "promotion_shop_active": promo_shop,
        "promotion_cafe_active": promo_cafe,
        "ad_active": ad_active,
        "price_AI92": price_AI92.round(2),
        "price_AI95": price_AI95.round(2),
        "price_AI98": price_AI98.round(2),
        "price_DT_EURO": price_DT_EURO.round(2),
        "competitor_price_AI92": competitor_price_AI92.round(2),
        "competitor_price_AI95": competitor_price_AI95.round(2),
        "competitor_price_DT": competitor_price_DT.round(2),
        "total_fuel_sales": total_fuel_sales.round(2),
        "sales_AI92": sales_AI92.round(2),
        "sales_AI95": sales_AI95.round(2),
        "sales_AI98": sales_AI98.round(2),
        "sales_DT_EURO": sales_DT_EURO.round(2),
        "sales_DT_TANEKO": sales_DT_TANEKO.round(2),
        "sales_DT_SUMMER": sales_DT_SUMMER.round(2),
        "sales_DT_WINTER": sales_DT_WINTER.round(2),
        "shop_напитки": shop_drinks.round(1),
        "shop_закуски": shop_snacks.round(1),
        "shop_автотовары": shop_auto.round(1),
        "shop_кофе": shop_coffee.round(1),
        "shop_табак": shop_tobacco.round(1),
        "shop_total_revenue": shop_total_revenue.round(2),
    })
    return df


def make_synthetic(out_dir: Path, n_stations: int = 5,
                    year: int = 2024, seed: int = 42) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start=f"{year}-01-01 00:00", end=f"{year}-12-31 23:00", freq="h"
    )
    meta = _build_station_meta(n_stations, rng)

    pieces = []
    for _, row in meta.iterrows():
        df_st = _generate_one_station(row, timestamps, rng)
        pieces.append(df_st)
        print(f"  Station {row['station_name']}: {len(df_st)} строк")
    data = pd.concat(pieces, ignore_index=True)

    data_path = out_dir / "detailed_data.csv"
    meta_path = out_dir / "stations_metadata.csv"
    data.to_csv(data_path, index=False)
    meta.to_csv(meta_path, index=False)
    # Дубли под именами, которые ждёт --quick
    (out_dir / "5stations_data.csv").write_bytes(data_path.read_bytes())
    (out_dir / "5stations_metadata.csv").write_bytes(meta_path.read_bytes())

    print(f"\nСохранено: {data_path}  ({len(data):,} строк, "
          f"{data['station_id'].nunique()} АЗС, "
          f"{data.shape[1]} колонок)")
    print(f"Сохранено: {meta_path}")
    return data_path, meta_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--stations", type=int, default=5)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    make_synthetic(Path(args.out), args.stations, args.year, args.seed)


if __name__ == "__main__":
    main()
