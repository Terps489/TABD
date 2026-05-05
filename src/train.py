"""TFT training with checkpointing and early stopping."""
import json
import warnings
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss

from src.config import (
    MODELS_DIR, LOGS_DIR, DATA_CACHE,
    MAX_EPOCHS, LEARNING_RATE, HIDDEN_SIZE, ATTENTION_HEAD_SIZE,
    DROPOUT, HIDDEN_CONTINUOUS_SIZE, GRADIENT_CLIP_VAL, TARGETS
)
from src.data_loader import load_raw, preprocess, create_datasets

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")  # RTX 4060 Ti Tensor Cores


def build_model(training_dataset) -> TemporalFusionTransformer:
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        loss=MultiLoss([QuantileLoss() for _ in TARGETS]),
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
    )
    print(f"TFT parameters: {tft.size() / 1e3:.1f}k")
    return tft


def train(use_5_stations: bool | None = None) -> Path:
    """Full training pipeline. Returns path to best checkpoint."""
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # Data — read CSV BEFORE any GPU ops, then cache for predict
    df = preprocess(load_raw(use_5_stations))
    df.to_parquet(DATA_CACHE, index=False)
    print(f"Data cached: {DATA_CACHE}")
    training, _, train_loader, val_loader = create_datasets(df)

    # Model
    tft = build_model(training)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=MODELS_DIR,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=8, mode="min", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        logger=CSVLogger(save_dir=str(LOGS_DIR), name="tft"),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    print(f"\nTraining on: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = Path(checkpoint_cb.best_model_path)
    print(f"\nBest checkpoint: {best_ckpt}")

    # Save metadata alongside checkpoint
    meta = {
        "best_checkpoint": str(best_ckpt),
        "n_stations": df["station_id"].nunique(),
        "targets": TARGETS,
        "val_loss": float(checkpoint_cb.best_model_score),
    }
    (MODELS_DIR / "training_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return best_ckpt


if __name__ == "__main__":
    train()
