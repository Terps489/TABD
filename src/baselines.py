"""Простые baseline-модели для сравнения с TFT.

- `naive_yesterday`:    ŷ(t) = y(t-24)   — повтор того же часа предыдущего дня
- `seasonal_naive_week`: ŷ(t) = y(t-168) — повтор того же часа неделю назад

Прогнозы строятся на той же 24-часовой валидационной выборке, что и TFT
(последние MAX_PREDICTION_LENGTH часов на каждую станцию). Сохраняем в том же
long-формате, что и TFT-прогнозы из `outputs/forecasts/`, чтобы можно было
прогнать через `src.metrics.compute_metrics_long` без адаптации.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_CACHE, MAX_PREDICTION_LENGTH, OUTPUTS_DIR, TARGETS,
)


BASELINES_DIR = OUTPUTS_DIR / "baselines"


BASELINE_LAGS = {
    "naive_yesterday": 24,
    "seasonal_naive_week": 168,
}


def _load_df() -> pd.DataFrame:
    if not DATA_CACHE.exists():
        raise FileNotFoundError(
            f"Кэш данных не найден: {DATA_CACHE}. Запустите обучение."
        )
    df = pd.read_parquet(DATA_CACHE)
    df["station_id"] = df["station_id"].astype(str)
    return df.sort_values(["station_id", "time_idx"]).reset_index(drop=True)


def build_validation_actual(
    df: pd.DataFrame | None = None,
    horizon: int = MAX_PREDICTION_LENGTH,
) -> pd.DataFrame:
    """Фактические значения таргетов в валидационном окне (last horizon часов
    на каждую станцию). Формат: [station_id, step, <TARGETS...>]."""
    if df is None:
        df = _load_df()
    pieces = []
    for sid, grp in df.groupby("station_id", sort=True):
        tail = grp.tail(horizon).reset_index(drop=True)
        if len(tail) < horizon:
            continue
        block = tail[TARGETS].copy()
        block.insert(0, "station_id", sid)
        block.insert(1, "step", np.arange(len(block)))
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)


def build_baseline_predictions(
    name: str,
    df: pd.DataFrame | None = None,
    horizon: int = MAX_PREDICTION_LENGTH,
) -> pd.DataFrame:
    """Long-DataFrame прогнозов baseline для последних `horizon` часов.

    Формат: [station_id, step, <TARGETS...>] — совпадает с `build_validation_actual`.
    """
    if name not in BASELINE_LAGS:
        raise ValueError(
            f"Неизвестный baseline: {name}. Доступно: {list(BASELINE_LAGS)}"
        )
    if df is None:
        df = _load_df()

    lag = BASELINE_LAGS[name]
    pieces = []
    for sid, grp in df.groupby("station_id", sort=True):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        if n < horizon + lag:
            continue
        pred_slice = grp.iloc[n - horizon - lag: n - lag][TARGETS].to_numpy()
        block = pd.DataFrame(pred_slice, columns=list(TARGETS))
        block.insert(0, "station_id", sid)
        block.insert(1, "step", np.arange(horizon))
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)


def load_tft_predictions(horizon: int = MAX_PREDICTION_LENGTH) -> pd.DataFrame:
    """Собирает TFT-прогноз (медиану) из per-target CSV в общий long-DF."""
    forecasts_dir = OUTPUTS_DIR / "forecasts"
    if not forecasts_dir.exists():
        raise FileNotFoundError(
            f"Папка {forecasts_dir} не найдена. Запустите `predict`."
        )

    parts: list[pd.DataFrame] = []
    for i, target in enumerate(TARGETS):
        path = forecasts_dir / f"{target}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Нет {path}. Запустите `predict`.")
        df = pd.read_csv(path, dtype={"station_id": str})
        df = df[df["step"] < horizon][["station_id", "step", "forecast_median"]]
        df = df.rename(columns={"forecast_median": target})
        if i == 0:
            parts.append(df)
        else:
            parts.append(df[[target]].reset_index(drop=True))
    base = parts[0].reset_index(drop=True)
    for extra in parts[1:]:
        base = pd.concat([base, extra], axis=1)
    return base


def save_baseline_predictions(name: str, df: pd.DataFrame) -> Path:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def run_baselines(horizon: int = MAX_PREDICTION_LENGTH) -> dict[str, pd.DataFrame]:
    """Прогнать все baseline-модели и сохранить их прогнозы."""
    df = _load_df()
    out: dict[str, pd.DataFrame] = {}
    for name in BASELINE_LAGS:
        preds = build_baseline_predictions(name, df, horizon)
        save_baseline_predictions(name, preds)
        out[name] = preds
        print(f"  Baseline {name}: {len(preds)} строк сохранено")
    return out
