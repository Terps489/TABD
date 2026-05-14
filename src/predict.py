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
from src.baselines import (
    build_validation_actual, load_tft_predictions, run_baselines,
)
from src.metrics import compute_metrics_long, save_metrics, write_summary

warnings.filterwarnings("ignore")

# Кэш ресурсов для on-demand прогнозов (дашборд):
# модель и training-датасет грузятся один раз, rollout-результаты — по таргету.
_MODEL_CACHE: TemporalFusionTransformer | None = None
_TRAINING_CACHE: TimeSeriesDataSet | None = None
_DF_CACHE: pd.DataFrame | None = None
# key=target, value=(longest_horizon_computed, long-df). Хранит ТОЛЬКО rollout-результаты;
# 24ч-путь использует training-time CSV и не кэшируется (численно другая дорожка).
_FORECAST_CACHE: dict[str, tuple[int, pd.DataFrame]] = {}


def _extract_quantiles(tp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Из тензора квантилей (n_samples, n_q, pred_len) или (n_samples, pred_len, n_q)
    вернуть (P10, медиана, P90) формы (n_samples, pred_len).
    Дефолтный QuantileLoss даёт 7 квантилей, индексы 1/3/5 = P10/median/P90."""
    if tp.ndim == 3 and tp.shape[1] == 7:
        return tp[:, 1, :], tp[:, 3, :], tp[:, 5, :]
    if tp.ndim == 3 and tp.shape[2] == 7:
        return tp[:, :, 1], tp[:, :, 3], tp[:, :, 5]
    flat = tp.reshape(tp.shape[0], -1)
    return flat, flat, flat


def _split_multitarget_preds(preds) -> list:
    """В pytorch-forecasting 1.7 multi-target predict возвращает list[Tensor]; на single-target
    собираем то же из последней оси для единого API."""
    if isinstance(preds, (list, tuple)):
        return list(preds)
    return [preds[..., i] for i in range(len(TARGETS))]


def load_model(checkpoint_path: str | Path | None = None) -> TemporalFusionTransformer:
    """Загрузить TFT из чекпоинта. Если путь не указан — найдёт лучший автоматически."""
    if checkpoint_path is None:
        meta_file = MODELS_DIR / "training_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            checkpoint_path = meta["best_checkpoint"]
        else:
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
    """Запустить инференс на валидационной выборке и сохранить CSV с прогнозами на 24ч.
    Возвращает dict: {имя_таргета -> DataFrame с forecast_p10/median/p90}."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    forecasts_dir = OUTPUTS_DIR / "forecasts"
    forecasts_dir.mkdir(exist_ok=True)

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
    preds_obj = model.predict(val_loader, mode="quantiles", return_x=True)
    preds = _split_multitarget_preds(preds_obj.output)
    sample_station_ids = validation.x_to_index(preds_obj.x)["station_id"].astype(str).tolist()

    results = {}
    for i, target in enumerate(TARGETS):
        p10, median, p90 = _extract_quantiles(preds[i].cpu().numpy())
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

    _save_feature_importance(df, forecasts_dir)

    print(f"\nПрогнозы сохранены в {forecasts_dir}")

    try:
        evaluate_all()
    except Exception as e:
        print(f"Предупреждение: расчёт метрик не удался ({e}).")

    return results


def evaluate_all() -> pd.DataFrame:
    """Считает метрики TFT (по медиане) и baseline-моделей на одной и той же
    24-часовой валидационной выборке. Возвращает long-DataFrame со всеми
    строками и пишет outputs/metrics/metrics.csv + summary.json."""
    print("\nРасчёт метрик качества...")
    actual = build_validation_actual()
    print(f"  Валидационная выборка: {len(actual)} строк")

    all_rows: list[pd.DataFrame] = []

    try:
        tft_pred = load_tft_predictions()
        all_rows.append(compute_metrics_long(
            model_name="TFT", actual=actual, predicted=tft_pred,
        ))
        print(f"  TFT: метрики посчитаны")
    except FileNotFoundError as e:
        print(f"  TFT: пропуск ({e})")

    baseline_preds = run_baselines()
    for name, preds in baseline_preds.items():
        all_rows.append(compute_metrics_long(
            model_name=name, actual=actual, predicted=preds,
        ))
        print(f"  {name}: метрики посчитаны")

    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not df_all.empty:
        csv_path = save_metrics(df_all)
        json_path = write_summary(df_all)
        print(f"  Сохранено: {csv_path.name}, {json_path.name}")
    return df_all


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
        idxs = np.arange(n_hours) % len(template)
        block = template.iloc[idxs].reset_index(drop=True).copy()

        new_ts = last_ts + pd.to_timedelta(np.arange(1, n_hours + 1), unit="h")
        block["timestamp"] = new_ts.values
        block["time_idx"] = last_idx + np.arange(1, n_hours + 1)

        if "hour" in block.columns:
            block["hour"] = new_ts.hour
        if "day_of_week" in block.columns:
            block["day_of_week"] = new_ts.dayofweek
        if "day" in block.columns:
            block["day"] = new_ts.day
        if "month" in block.columns:
            block["month"] = new_ts.month

        block["hour_sin"] = np.sin(2 * np.pi * new_ts.hour / 24)
        block["hour_cos"] = np.cos(2 * np.pi * new_ts.hour / 24)
        block["day_sin"] = np.sin(2 * np.pi * new_ts.dayofweek / 7)
        block["day_cos"] = np.cos(2 * np.pi * new_ts.dayofweek / 7)
        block["month_sin"] = np.sin(2 * np.pi * new_ts.month / 12)
        block["month_cos"] = np.cos(2 * np.pi * new_ts.month / 12)

        block["is_weekend"] = (new_ts.dayofweek >= 5).astype(float)
        block["is_rush_hour"] = new_ts.hour.isin([7, 8, 9, 17, 18, 19]).astype(float)
        block["is_night"] = ((new_ts.hour < 6) | (new_ts.hour >= 22)).astype(float)

        if "season" in block.columns:
            block["season"] = [_month_to_season(int(m)) for m in new_ts.month]
            block["season"] = block["season"].astype(str)

        for t in TARGETS:
            if t in block.columns:
                block[t] = 0.0

        pieces.append(block)

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values(["station_id", "time_idx"]).reset_index(drop=True)
    return out


def _iterative_forecast(target_name: str, horizon_hours: int) -> pd.DataFrame:
    """Итеративный rollout: модель шагает по 24ч, медианы подставляются
    в энкодер как «история» для следующего шага.

    Возвращает long-DataFrame: [station_id, hour_ahead, timestamp, p10, median, p90]
    для запрошенного таргета.
    """
    model, training, df_orig = _get_inference_resources()
    target_idx = TARGETS.index(target_name)

    last_idx = int(df_orig["time_idx"].max())
    last_ts = df_orig["timestamp"].max()

    df_ext = _synthesize_future(df_orig, horizon_hours).reset_index(drop=True)

    # MultiIndex (station_id, time_idx) → позиция в df_ext: даёт O(1)-lookup для
    # векторного присваивания предсказанных медиан вместо boolean-mask на каждую станцию.
    df_ext_idx = pd.MultiIndex.from_arrays(
        [df_ext["station_id"].astype(str).values, df_ext["time_idx"].values]
    )

    n_iter = math.ceil(horizon_hours / 24)
    records: list[dict] = []

    for it in range(n_iter):
        block_last_idx = last_idx + (it + 1) * 24
        df_iter = df_ext[df_ext["time_idx"] <= block_last_idx]

        pred_ds = TimeSeriesDataSet.from_dataset(
            training, df_iter, predict=True, stop_randomization=True
        )
        pred_loader = pred_ds.to_dataloader(train=False, batch_size=128, num_workers=0)

        print(f"  rollout {it + 1}/{n_iter} (до часа +{(it + 1) * 24})...")
        preds_obj = model.predict(pred_loader, mode="quantiles", return_x=True)
        preds = _split_multitarget_preds(preds_obj.output)
        sample_station_ids = pred_ds.x_to_index(preds_obj.x)["station_id"].astype(str).tolist()

        # Позиции 25×24=600 строк прогноза в df_ext: первая ось — станции в порядке батча,
        # вторая — часы в окне. Совпадает с .ravel() от тензора (n_samples, 24).
        start = block_last_idx - 23
        time_idx_range = np.arange(start, block_last_idx + 1)
        keys = [(sid, idx) for sid in sample_station_ids for idx in time_idx_range]
        positions = df_ext_idx.get_indexer(keys)

        req_p10 = req_med = req_p90 = None
        for ti, t in enumerate(TARGETS):
            p10, median, p90 = _extract_quantiles(preds[ti].cpu().numpy())
            df_ext.iloc[positions, df_ext.columns.get_loc(t)] = median.ravel()
            if ti == target_idx:
                req_p10, req_med, req_p90 = p10, median, p90

        for s_i, sid in enumerate(sample_station_ids):
            for h in range(req_med.shape[1]):
                hour_ahead = (start + h) - last_idx
                if hour_ahead > horizon_hours:
                    continue
                records.append({
                    "station_id": sid,
                    "hour_ahead": int(hour_ahead),
                    "timestamp": last_ts + pd.Timedelta(hours=int(hour_ahead)),
                    "p10": float(req_p10[s_i, h]),
                    "median": float(req_med[s_i, h]),
                    "p90": float(req_p90[s_i, h]),
                })

    return pd.DataFrame(records)


def forecast_extended(target_name: str, horizon_hours: int) -> pd.DataFrame:
    """Прогноз на любой горизонт. Для ≤24ч читает training-time CSV (быстро),
    для большего горизонта — итеративный rollout.

    Возвращает: [station_id, hour_ahead, timestamp, p10, median, p90].
    """
    if target_name not in TARGETS:
        raise ValueError(f"Неизвестный таргет: {target_name}")

    # Быстрый путь: 24ч — это training-time валидация из CSV. Не кэшируем:
    # rollout-результаты численно отличаются (медиана подаётся обратно в энкодер),
    # поэтому смешивать их в одном кэше было бы неверно.
    if horizon_hours <= 24:
        csv_path = OUTPUTS_DIR / "forecasts" / f"{target_name}.csv"
        if csv_path.exists():
            df_fc = pd.read_csv(csv_path, dtype={"station_id": str})
            if not DATA_CACHE.exists():
                raise FileNotFoundError(f"Нет {DATA_CACHE}.")
            last_ts = pd.to_datetime(
                pd.read_parquet(DATA_CACHE, columns=["timestamp"])["timestamp"]
            ).max()
            df_fc["hour_ahead"] = df_fc["step"].astype(int) + 1
            df_fc["timestamp"] = last_ts + pd.to_timedelta(df_fc["hour_ahead"], unit="h")
            out = df_fc.rename(columns={
                "forecast_p10": "p10",
                "forecast_median": "median",
                "forecast_p90": "p90",
            })[["station_id", "hour_ahead", "timestamp", "p10", "median", "p90"]]
            return out[out["hour_ahead"] <= horizon_hours]

    # Длинный путь: кэшируем самый длинный посчитанный горизонт; более короткие — срезаем.
    cached = _FORECAST_CACHE.get(target_name)
    if cached is not None and cached[0] >= horizon_hours:
        return cached[1][cached[1]["hour_ahead"] <= horizon_hours].copy()

    print(f"\nИтеративный прогноз: {target_name}, горизонт {horizon_hours} ч")
    df_out = _iterative_forecast(target_name, horizon_hours)
    _FORECAST_CACHE[target_name] = (horizon_hours, df_out)
    return df_out


if __name__ == "__main__":
    predict()
