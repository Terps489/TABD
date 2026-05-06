"""Загрузка обученного TFT, генерация прогнозов и рекомендаций."""
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer

from src.config import MODELS_DIR, OUTPUTS_DIR, DATA_CACHE, TARGETS
from src.data_loader import create_datasets

warnings.filterwarnings("ignore")


def load_model(checkpoint_path: str | Path | None = None) -> TemporalFusionTransformer:
    """Загрузить TFT из чекпоинта. Если путь не указан — найдёт лучший автоматически."""
    if checkpoint_path is None:
        meta_file = MODELS_DIR / "training_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            checkpoint_path = meta["best_checkpoint"]
        else:
            # Найти последний чекпоинт
            ckpts = sorted(MODELS_DIR.glob("tft-*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"Чекпоинт не найден в {MODELS_DIR}. Сначала запустите обучение.")
            checkpoint_path = ckpts[-1]

    print(f"Загрузка модели: {checkpoint_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    return model


def predict(
    use_5_stations: bool | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Запустить инференс на валидационной выборке.
    Возвращает dict: {имя_таргета -> DataFrame с forecast_p10, forecast_median, forecast_p90}
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    forecasts_dir = OUTPUTS_DIR / "forecasts"
    forecasts_dir.mkdir(exist_ok=True)

    # Загрузка из parquet-кэша (обходит Windows DLL-конфликт CUDA + чтение CSV)
    if not DATA_CACHE.exists():
        raise FileNotFoundError(
            f"Кэш данных не найден: {DATA_CACHE}\n"
            "Сначала запустите обучение: python run.py --mode train"
        )
    print(f"Загрузка кэша данных из {DATA_CACHE.name}...")
    df = pd.read_parquet(DATA_CACHE)

    training, validation, _, val_loader = create_datasets(df)

    model = load_model(checkpoint_path)
    model.eval()

    print("Генерация прогнозов (mode=quantiles)...")
    # Режим quantiles возвращает list[Tensor] для multi-target
    preds_obj = model.predict(val_loader, mode="quantiles", return_x=True)
    preds = preds_obj.output
    x = preds_obj.x

    # Соответствие сэмпл -> station_id (через index валидационной выборки)
    index_df = validation.x_to_index(x)
    sample_station_ids = index_df["station_id"].astype(str).tolist()

    # preds — list[Tensor] для multi-target
    if not isinstance(preds, (list, tuple)):
        preds = [preds[..., i] for i in range(len(TARGETS))]

    results = {}
    for i, target in enumerate(TARGETS):
        tp = preds[i].cpu().numpy()  # (n_samples, n_quantiles, pred_len) или (n_samples, pred_len, n_quantiles)

        # Определяем ось квантилей: дефолтный QuantileLoss даёт 7 квантилей [0.02,0.1,0.25,0.5,0.75,0.9,0.98]
        if tp.ndim == 3 and tp.shape[1] == 7:
            # (n_samples, n_quantiles, pred_len)
            p10, median, p90 = tp[:, 1, :], tp[:, 3, :], tp[:, 5, :]
        elif tp.ndim == 3 and tp.shape[2] == 7:
            # (n_samples, pred_len, n_quantiles)
            p10, median, p90 = tp[:, :, 1], tp[:, :, 3], tp[:, :, 5]
        else:
            # На всякий случай: используем как есть
            p10 = median = p90 = tp

        df_out = pd.DataFrame({
            "station_id": sample_station_ids,
            "forecast_p10": p10.mean(axis=-1),
            "forecast_median": median.mean(axis=-1),
            "forecast_p90": p90.mean(axis=-1),
        })
        results[target] = df_out
        df_out.to_csv(forecasts_dir / f"{target}.csv", index=False)
        print(f"  Сохранено: {target}.csv")

    # Важность признаков (корреляция факторов с total_fuel_sales)
    _save_feature_importance(df, forecasts_dir)

    print(f"\nПрогнозы сохранены в {forecasts_dir}")
    return results


def _save_feature_importance(df: pd.DataFrame, out_dir: Path):
    """Важность признаков через абсолютную корреляцию с total_fuel_sales.

    Multi-target TFT в pytorch-forecasting 1.7 не поддерживает interpret_output,
    поэтому используется простой и надёжный корреляционный подход — он
    отражает линейную связь каждого фактора с целевой переменной.
    """
    try:
        print("Расчёт важности признаков (корреляция)...")
        candidate_features = [
            "total_traffic", "traffic_Passengers_cars", "traffic_Truck",
            "traffic_Truck_long", "traffic_Truck_short", "traffic_Transporter",
            "temperature", "precipitation_mm", "wind_speed_ms", "visibility_km",
            "is_snow", "is_rain", "is_fog",
            "promotion_fuel_active", "promotion_shop_active", "ad_active",
            "competitor_price_AI92", "competitor_price_AI95", "competitor_price_DT",
            "price_AI92", "price_AI95",
            "is_weekend", "is_holiday", "is_rush_hour", "is_night",
            "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
            "corporate_customer_ratio", "staff_engagement_score",
            "customer_loyalty_score", "competitors_within_5km",
        ]
        target = "total_fuel_sales"
        importance = {"encoder": {}}
        for col in candidate_features:
            if col in df.columns:
                corr = df[col].corr(df[target])
                if pd.notna(corr):
                    importance["encoder"][col] = abs(float(corr))

        (out_dir / "feature_importance.json").write_text(
            json.dumps(importance, indent=2, ensure_ascii=False)
        )
        print(f"Важность признаков сохранена ({len(importance['encoder'])} факторов).")
    except Exception as e:
        print(f"Не удалось рассчитать важность признаков: {e}")


def generate_recommendations(importance_path: Path | None = None) -> list[str]:
    """Текстовые рекомендации на основе важности признаков."""
    if importance_path is None:
        importance_path = OUTPUTS_DIR / "forecasts" / "feature_importance.json"

    if not importance_path.exists():
        return ["Запустите predict.py для генерации рекомендаций."]

    data = json.loads(importance_path.read_text())
    encoder = data.get("encoder", {})

    top_factors = sorted(encoder.items(), key=lambda x: x[1], reverse=True)[:5]
    recs = []

    factor_map = {
        "total_traffic": "трафик — наиболее важный фактор продаж. Усильте маркетинг в часы пик (7-9, 17-19).",
        "temperature": "температура существенно влияет на продажи. Учитывайте сезонность при планировании запасов.",
        "promotion_fuel_active": "акции на топливо значимо увеличивают продажи. Рекомендуется регулярное проведение акций.",
        "ad_active": "реклама даёт заметный эффект. Увеличьте бюджет на активные каналы.",
        "competitor_price_AI92": "цена конкурентов влияет на спрос. Мониторьте и оперативно корректируйте цены.",
        "is_weekend": "выходные дни показывают иной паттерн продаж. Планируйте персонал и запасы отдельно.",
        "hour_sin": "время суток критично. Оптимизируйте режим работы колонок по часам.",
    }

    for factor, importance in top_factors:
        text = factor_map.get(factor, f"{factor} — важный фактор (вес {importance:.3f}).")
        recs.append(f"• {text}")

    if not recs:
        recs = ["Недостаточно данных для генерации рекомендаций. Запустите обучение и predict."]

    return recs


if __name__ == "__main__":
    predict()
