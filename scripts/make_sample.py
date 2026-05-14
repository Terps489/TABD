"""Сгенерировать маленький обезличенный sample dataset для git.

Берёт 2 АЗС × 14 дней (672 строки) из полного detailed_data.csv и:
- обезличивает имена станций → Station_A / Station_B
- сохраняет в data/sample/{sample_data.csv, sample_metadata.csv}

Использование (из корня проекта):
    python scripts/make_sample.py
    python scripts/make_sample.py --stations 2 --days 14
    python scripts/make_sample.py --source "C:/path/to/data"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Делаем модуль запускаемым из корня проекта без `python -m`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, PROJECT_DIR  # noqa: E402


def make_sample(
    source_dir: Path,
    out_dir: Path,
    n_stations: int = 2,
    n_days: int = 14,
) -> tuple[Path, Path]:
    detailed_path = source_dir / "detailed_data.csv"
    meta_path = source_dir / "stations_metadata.csv"
    if not detailed_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Не нашёл detailed_data.csv / stations_metadata.csv в {source_dir}.\n"
            f"Передайте --source путь к папке с CSV."
        )

    print(f"Источник: {source_dir}")
    df = pd.read_csv(detailed_path, parse_dates=["timestamp"])
    meta = pd.read_csv(meta_path)

    # Берём первые N станций по station_id, чтобы выборка была детерминирована.
    station_ids = sorted(df["station_id"].unique())[:n_stations]
    n_hours = n_days * 24

    # Из каждой выбранной станции — первые n_hours часов хронологически.
    pieces = []
    for sid in station_ids:
        grp = df[df["station_id"] == sid].sort_values("timestamp").head(n_hours)
        pieces.append(grp)
    sample = pd.concat(pieces, ignore_index=True)

    meta_sample = meta[meta["station_id"].isin(station_ids)].copy()

    # Обезличиваем: Station_A, Station_B, ...
    name_map = {sid: f"Station_{chr(ord('A') + i)}"
                for i, sid in enumerate(station_ids)}
    sample["station_name"] = sample["station_id"].map(name_map)
    meta_sample["station_name"] = meta_sample["station_id"].map(name_map)

    # Убираем потенциально-личные поля из метаданных (если они есть).
    for col in ("address", "phone", "manager", "owner_name"):
        if col in meta_sample.columns:
            meta_sample = meta_sample.drop(columns=[col])

    out_dir.mkdir(parents=True, exist_ok=True)
    # Имена совпадают с теми, что ждёт data_loader.py — `TABD_DATA_DIR=data/sample`
    # сразу подхватит этот sample как detailed_data.csv (а в режиме --quick — как
    # 5stations_data.csv, тот же файл под обоими именами).
    data_out = out_dir / "detailed_data.csv"
    meta_out = out_dir / "stations_metadata.csv"
    sample.to_csv(data_out, index=False)
    meta_sample.to_csv(meta_out, index=False)
    (out_dir / "5stations_data.csv").write_bytes(data_out.read_bytes())
    (out_dir / "5stations_metadata.csv").write_bytes(meta_out.read_bytes())
    print(f"Сохранено: {data_out}  ({len(sample)} строк, "
          f"{sample['station_id'].nunique()} АЗС)")
    print(f"Сохранено: {meta_out}  ({len(meta_sample)} строк)")
    return data_out, meta_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default=str(DATA_DIR),
                    help="Папка с detailed_data.csv (по умолчанию — TABD_DATA_DIR)")
    ap.add_argument("--out", type=str,
                    default=str(PROJECT_DIR / "data" / "sample"),
                    help="Куда сохранить sample")
    ap.add_argument("--stations", type=int, default=2)
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    make_sample(Path(args.source), Path(args.out), args.stations, args.days)


if __name__ == "__main__":
    main()
