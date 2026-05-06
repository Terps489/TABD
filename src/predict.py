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
    # Режим quantiles возвращает list[Tensor] для multi-target, форма (n_samples, n_quantiles, pred_len)
    preds = model.predict(val_loader, mode="quantiles", return_x=False)

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
            "forecast_p10": p10.mean(axis=-1),
            "forecast_median": median.mean(axis=-1),
            "forecast_p90": p90.mean(axis=-1),
        })
        results[target] = df_out
        df_out.to_csv(forecasts_dir / f"{target}.csv", index=False)
        print(f"  Сохранено: {target}.csv")

    # Важность признаков
    _save_feature_importance(model, training, forecasts_dir)

    print(f"\nПрогнозы сохранены в {forecasts_dir}")
    return results


def _save_feature_importance(model, training, out_dir: Path, n_samples: int = 500):
    """Рассчитать важность признаков на подмножестве (а не на всей train-выборке).

    Полная train-выборка (~215K сэмплов) с mode='raw' слишком тяжёлая
    и часто зависает. 500 сэмплов дают репрезентативный результат за <1 мин.
    """
    try:
        from torch.utils.data import Subset
        import random

        random.seed(42)
        subset_size = min(n_samples, len(training))
        indices = random.sample(range(len(training)), subset_size)
        subset = Subset(training, indices)

        loader = torch.utils.data.DataLoader(
            subset, batch_size=64, num_workers=0, collate_fn=training._collate_fn
        )

        print(f"Расчёт важности признаков на {subset_size} сэмплах...")
        interp = model.interpret_output(
            model.predict(loader, mode="raw", return_x=True).output,
            reduction="mean",
        )
        importance = {}
        if "encoder_variables" in interp:
            importance["encoder"] = {
                k: float(v) for k, v in zip(
                    interp["encoder_variables"]["labels"],
                    interp["encoder_variables"]["values"]
                )
            }
        if "decoder_variables" in interp:
            importance["decoder"] = {
                k: float(v) for k, v in zip(
                    interp["decoder_variables"]["labels"],
                    interp["decoder_variables"]["values"]
                )
            }
        (out_dir / "feature_importance.json").write_text(
            json.dumps(importance, indent=2, ensure_ascii=False)
        )
        print("Важность признаков сохранена.")
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
