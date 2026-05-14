"""Загрузка обученного TFT, генерация прогнозов и рекомендаций."""
import json
import math
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.config import MODELS_DIR, OUTPUTS_DIR, DATA_CACHE, TARGETS
from src.data_loader import create_datasets

warnings.filterwarnings("ignore")

# Кэш ресурсов для on-demand прогнозов (дашборд):
# модель и training-датасет грузятся один раз, результаты прогноза — по таргету.
_MODEL_CACHE: TemporalFusionTransformer | None = None
_TRAINING_CACHE: TimeSeriesDataSet | None = None
_DF_CACHE: pd.DataFrame | None = None
_FORECAST_CACHE: dict[str, tuple[int, pd.DataFrame]] = {}


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
        # Приводим к форме (n_samples, pred_len) для каждого квантиля.
        if tp.ndim == 3 and tp.shape[1] == 7:
            p10, median, p90 = tp[:, 1, :], tp[:, 3, :], tp[:, 5, :]
        elif tp.ndim == 3 and tp.shape[2] == 7:
            p10, median, p90 = tp[:, :, 1], tp[:, :, 3], tp[:, :, 5]
        else:
            p10 = median = p90 = tp.reshape(tp.shape[0], -1)

        # Long-формат: одна строка на (станция, шаг прогноза)
        n_samples, pred_len = p10.shape
        rows = []
        for s in range(n_samples):
            for h in range(pred_len):
                rows.append({
                    "station_id": sample_station_ids[s],
                    "step": h,
                    "forecast_p10": float(p10[s, h]),
                    "forecast_median": float(median[s, h]),
                    "forecast_p90": float(p90[s, h]),
                })
        df_out = pd.DataFrame(rows)
        results[target] = df_out
        df_out.to_csv(forecasts_dir / f"{target}.csv", index=False)
        print(f"  Сохранено: {target}.csv ({n_samples} станций x {pred_len} часов)")

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


def _get_inference_resources():
    """Лениво подгружает модель + training-датасет + parquet-кэш. Один раз за процесс."""
    global _MODEL_CACHE, _TRAINING_CACHE, _DF_CACHE
    if _MODEL_CACHE is None or _TRAINING_CACHE is None or _DF_CACHE is None:
        if not DATA_CACHE.exists():
            raise FileNotFoundError(f"Нет {DATA_CACHE}. Сначала запустите обучение.")
        df = pd.read_parquet(DATA_CACHE)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        training, _, _, _ = create_datasets(df)
        _DF_CACHE = df
        _TRAINING_CACHE = training
        _MODEL_CACHE = load_model()
    return _MODEL_CACHE, _TRAINING_CACHE, _DF_CACHE


def _month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "autumn"


def _synthesize_future(df_full: pd.DataFrame, n_hours: int) -> pd.DataFrame:
    """Достроить df_full будущими n_hours часами на станцию.

    Шаблон: последние 168 часов истории каждой станции, реплицируется по модулю 168.
    Это сохраняет недельную сезонность для time_varying_unknown_reals
    (трафик, погода, цены) — на каждый будущий час берётся значение того же
    часа недели из прошлой недели.

    Временные поля (timestamp, hour_sin/cos, season и т.п.) пересчитываются.
    Целевые переменные обнуляются — их заполнит итеративный прогноз.
    """
    last_idx = int(df_full["time_idx"].max())
    last_ts = df_full["timestamp"].max()

    pieces = [df_full]
    for sid, df_st in df_full.groupby("station_id", sort=False):
        df_st = df_st.sort_values("time_idx").reset_index(drop=True)
        template = df_st.tail(168).reset_index(drop=True)
        if len(template) == 0:
            continue
        # Реплицируем шаблон по модулю длины (на случай <168)
        idxs = np.arange(n_hours) % len(template)
        block = template.iloc[idxs].reset_index(drop=True).copy()

        new_ts = last_ts + pd.to_timedelta(np.arange(1, n_hours + 1), unit="h")
        block["timestamp"] = new_ts.values
        block["time_idx"] = last_idx + np.arange(1, n_hours + 1)

        # Базовые временные поля (если есть в df)
        if "hour" in block.columns:
            block["hour"] = new_ts.hour
        if "day_of_week" in block.columns:
            block["day_of_week"] = new_ts.dayofweek
        if "day" in block.columns:
            block["day"] = new_ts.day
        if "month" in block.columns:
            block["month"] = new_ts.month

        # Циклические признаки (известные модели)
        block["hour_sin"] = np.sin(2 * np.pi * new_ts.hour / 24)
        block["hour_cos"] = np.cos(2 * np.pi * new_ts.hour / 24)
        block["day_sin"] = np.sin(2 * np.pi * new_ts.dayofweek / 7)
        block["day_cos"] = np.cos(2 * np.pi * new_ts.dayofweek / 7)
        block["month_sin"] = np.sin(2 * np.pi * new_ts.month / 12)
        block["month_cos"] = np.cos(2 * np.pi * new_ts.month / 12)

        block["is_weekend"] = (new_ts.dayofweek >= 5).astype(float)
        block["is_rush_hour"] = new_ts.hour.isin([7, 8, 9, 17, 18, 19]).astype(float)
        block["is_night"] = ((new_ts.hour < 6) | (new_ts.hour >= 22)).astype(float)

        # Сезон от месяца (категориальный)
        if "season" in block.columns:
            block["season"] = [_month_to_season(int(m)) for m in new_ts.month]
            block["season"] = block["season"].astype(str)

        # Обнуляем целевые — заполнит итеративный rollout
        for t in TARGETS:
            if t in block.columns:
                block[t] = 0.0

        pieces.append(block)

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values(["station_id", "time_idx"]).reset_index(drop=True)
    return out


def _extract_quantiles(tp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Из тензора квантилей (n_samples, n_q, pred_len) или (n_samples, pred_len, n_q)
    вернуть (P10, медиана, P90) формы (n_samples, pred_len).
    Дефолтный QuantileLoss — 7 квантилей, индексы 1/3/5 = P10/median/P90."""
    if tp.ndim == 3 and tp.shape[1] == 7:
        return tp[:, 1, :], tp[:, 3, :], tp[:, 5, :]
    if tp.ndim == 3 and tp.shape[2] == 7:
        return tp[:, :, 1], tp[:, :, 3], tp[:, :, 5]
    flat = tp.reshape(tp.shape[0], -1)
    return flat, flat, flat


def _iterative_forecast(target_name: str, horizon_hours: int) -> pd.DataFrame:
    """Итеративный rollout: модель шагает по 24ч, предсказания подставляются
    в энкодер как «история» для следующего шага.

    Возвращает long-DataFrame: [station_id, hour_ahead, timestamp, p10, median, p90].
    Содержит ВСЕ таргеты, но возвращается срез по `target_name`.
    """
    model, training, df_orig = _get_inference_resources()
    target_idx = TARGETS.index(target_name)

    last_idx = int(df_orig["time_idx"].max())
    last_ts = df_orig["timestamp"].max()

    # Один раз достраиваем весь будущий горизонт; заполнять будем итеративно
    df_ext = _synthesize_future(df_orig, horizon_hours)

    n_iter = math.ceil(horizon_hours / 24)
    results: list[dict] = []

    for it in range(n_iter):
        block_last_idx = last_idx + (it + 1) * 24
        df_iter = df_ext[df_ext["time_idx"] <= block_last_idx].copy()

        pred_ds = TimeSeriesDataSet.from_dataset(
            training, df_iter, predict=True, stop_randomization=True
        )
        pred_loader = pred_ds.to_dataloader(train=False, batch_size=128, num_workers=0)

        print(f"  rollout {it + 1}/{n_iter} (до часа +{(it + 1) * 24})...")
        preds_obj = model.predict(pred_loader, mode="quantiles", return_x=True)
        preds = preds_obj.output
        x = preds_obj.x
        index_df = pred_ds.x_to_index(x)
        sample_station_ids = index_df["station_id"].astype(str).tolist()

        if not isinstance(preds, (list, tuple)):
            preds = [preds[..., i] for i in range(len(TARGETS))]

        # Подставляем медианы ВСЕХ таргетов в df_ext (на следующую итерацию)
        # и сохраняем p10/median/p90 запрошенного таргета в результат.
        for ti, t in enumerate(TARGETS):
            tp = preds[ti].cpu().numpy()
            _, median_arr, _ = _extract_quantiles(tp)
            # median_arr: (n_samples, 24)
            for s_i, sid in enumerate(sample_station_ids):
                start = block_last_idx - 24 + 1
                # batched assign через mask по диапазону time_idx
                mask = (
                    (df_ext["station_id"] == sid)
                    & (df_ext["time_idx"] >= start)
                    & (df_ext["time_idx"] <= block_last_idx)
                )
                df_ext.loc[mask, t] = median_arr[s_i, :]

        # Сохраняем требуемый таргет с квантилями в результат
        tp_req = preds[target_idx].cpu().numpy()
        p10, med, p90 = _extract_quantiles(tp_req)
        for s_i, sid in enumerate(sample_station_ids):
            for h in range(med.shape[1]):
                abs_idx = block_last_idx - 24 + 1 + h
                hour_ahead = abs_idx - last_idx
                if hour_ahead > horizon_hours:
                    continue
                ts = last_ts + pd.Timedelta(hours=int(hour_ahead))
                results.append({
                    "station_id": sid,
                    "hour_ahead": int(hour_ahead),
                    "timestamp": ts,
                    "p10": float(p10[s_i, h]),
                    "median": float(med[s_i, h]),
                    "p90": float(p90[s_i, h]),
                })

    return pd.DataFrame(results)


def forecast_extended(target_name: str, horizon_hours: int) -> pd.DataFrame:
    """Прогноз на любой горизонт. Для 24ч читает из кэша CSV (быстро),
    для большего горизонта запускает итеративный rollout.

    Возвращает: [station_id, hour_ahead, timestamp, p10, median, p90].
    """
    if target_name not in TARGETS:
        raise ValueError(f"Неизвестный таргет: {target_name}")

    # Внутрипроцессный кэш по таргету (горизонт >= запрошенного — отдаём срез)
    cached = _FORECAST_CACHE.get(target_name)
    if cached is not None:
        cached_h, df_cached = cached
        if cached_h >= horizon_hours:
            return df_cached[df_cached["hour_ahead"] <= horizon_hours].copy()

    # Быстрый путь: 24ч — берём из готовых CSV
    if horizon_hours <= 24:
        csv_path = OUTPUTS_DIR / "forecasts" / f"{target_name}.csv"
        if csv_path.exists():
            df_fc = pd.read_csv(csv_path, dtype={"station_id": str})
            # Найдём last_ts через parquet-кэш
            if not DATA_CACHE.exists():
                raise FileNotFoundError(f"Нет {DATA_CACHE}.")
            df_meta = pd.read_parquet(DATA_CACHE, columns=["timestamp"])
            last_ts = pd.to_datetime(df_meta["timestamp"]).max()
            df_fc["hour_ahead"] = df_fc["step"].astype(int) + 1
            df_fc["timestamp"] = last_ts + pd.to_timedelta(df_fc["hour_ahead"], unit="h")
            out = df_fc.rename(columns={
                "forecast_p10": "p10",
                "forecast_median": "median",
                "forecast_p90": "p90",
            })[["station_id", "hour_ahead", "timestamp", "p10", "median", "p90"]]
            out = out[out["hour_ahead"] <= horizon_hours]
            return out

    # Длинный путь: итеративный rollout
    print(f"\nИтеративный прогноз: {target_name}, горизонт {horizon_hours} ч")
    df_out = _iterative_forecast(target_name, horizon_hours)
    _FORECAST_CACHE[target_name] = (horizon_hours, df_out)
    return df_out


if __name__ == "__main__":
    predict()
