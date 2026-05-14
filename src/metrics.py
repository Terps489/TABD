"""Метрики качества прогнозов: MAE, RMSE, MAPE, SMAPE.

Считает per-target × per-station и агрегаты для TFT + baseline-моделей.
Артефакты складываются в outputs/metrics/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, TARGETS


METRICS_DIR = OUTPUTS_DIR / "metrics"


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """MAPE с защитой от нулей: знаменатель не меньше eps.

    eps = 1 (литры/рубли в час) — на наших масштабах продаж это незаметная
    корректировка, но избавляет от взрыва на сэмплах с y_true ≈ 0.
    """
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    """Симметричный MAPE: 200·|y-ŷ| / (|y|+|ŷ|+eps). Ограничен 0..200%."""
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def _row(
    *, model: str, target: str, station_id: str | None,
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict:
    return {
        "model": model,
        "target": target,
        "station_id": station_id if station_id is not None else "ALL",
        "n": int(y_true.size),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def compute_metrics_long(
    *,
    model_name: str,
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    targets: Iterable[str] = TARGETS,
) -> pd.DataFrame:
    """Считает per-station и aggregate метрики.

    Ожидаемый формат `actual` и `predicted`: long-DataFrame с колонками
    `station_id` (str), `step` (0..H-1) и значениями по таргетам в одноимённых
    колонках. Обе таблицы должны иметь одинаковый набор (station_id, step).
    """
    a = actual.copy()
    p = predicted.copy()
    a["station_id"] = a["station_id"].astype(str)
    p["station_id"] = p["station_id"].astype(str)

    merged = a.merge(p, on=["station_id", "step"], suffixes=("_true", "_pred"))
    if merged.empty:
        return pd.DataFrame(columns=[
            "model", "target", "station_id", "n", "mae", "rmse", "mape", "smape"
        ])

    rows: list[dict] = []
    for target in targets:
        t_true = f"{target}_true"
        t_pred = f"{target}_pred"
        if t_true not in merged.columns or t_pred not in merged.columns:
            continue
        for sid, grp in merged.groupby("station_id", sort=True):
            rows.append(_row(
                model=model_name, target=target, station_id=sid,
                y_true=grp[t_true].to_numpy(),
                y_pred=grp[t_pred].to_numpy(),
            ))
        rows.append(_row(
            model=model_name, target=target, station_id=None,
            y_true=merged[t_true].to_numpy(),
            y_pred=merged[t_pred].to_numpy(),
        ))
    return pd.DataFrame(rows)


def save_metrics(df_long: pd.DataFrame, filename: str = "metrics.csv") -> Path:
    """Сохранить metric-таблицу. Если файл уже есть и содержит другие модели —
    обновляем строки текущей модели, остальные не трогаем."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / filename
    if path.exists():
        prev = pd.read_csv(path, dtype={"station_id": str})
        models_now = set(df_long["model"].unique())
        prev = prev[~prev["model"].isin(models_now)]
        combined = pd.concat([prev, df_long], ignore_index=True)
    else:
        combined = df_long
    combined.to_csv(path, index=False)
    return path


def write_summary(df_long: pd.DataFrame, filename: str = "summary.json") -> Path:
    """Агрегированная сводка по моделям и таргетам (только ALL-строки)."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / filename
    agg = df_long[df_long["station_id"] == "ALL"]
    summary: dict[str, dict] = {}
    for model, g_model in agg.groupby("model", sort=True):
        per_target = {
            row["target"]: {
                "mae": row["mae"], "rmse": row["rmse"],
                "mape": row["mape"], "smape": row["smape"],
                "n": int(row["n"]),
            }
            for _, row in g_model.iterrows()
        }
        summary[model] = {
            "per_target": per_target,
            "macro_avg": {
                "mae": float(g_model["mae"].mean()),
                "rmse": float(g_model["rmse"].mean()),
                "mape": float(g_model["mape"].mean()),
                "smape": float(g_model["smape"].mean()),
            },
        }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return path


def load_metrics(filename: str = "metrics.csv") -> pd.DataFrame:
    path = METRICS_DIR / filename
    if not path.exists():
        return pd.DataFrame(columns=[
            "model", "target", "station_id", "n", "mae", "rmse", "mape", "smape"
        ])
    return pd.read_csv(path, dtype={"station_id": str})
