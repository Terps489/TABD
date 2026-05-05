"""
Main entry point for the TABD TFT project.

Usage:
    python run.py --mode train           # Train TFT model
    python run.py --mode predict         # Generate forecasts (needs trained model)
    python run.py --mode dashboard       # Start dashboard (data only, no model required)
    python run.py --mode all             # Train → predict → dashboard
    python run.py --mode train --quick   # Quick test on 5 stations
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="TABD TFT Pipeline")
    parser.add_argument(
        "--mode", choices=["train", "predict", "dashboard", "all"],
        default="dashboard", help="Pipeline mode"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use 5-station subset for quick testing"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (for predict mode)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        import src.config as cfg
        cfg.USE_5_STATIONS = True
        print("[Quick mode] Using 5-station subset.")

    if args.mode in ("train", "all"):
        print("\n" + "="*50)
        print(" STEP 1: Training TFT model")
        print("="*50)
        from src.train import train
        ckpt = train(use_5_stations=args.quick)
        print(f"\nModel saved: {ckpt}")

    if args.mode in ("predict", "all"):
        print("\n" + "="*50)
        print(" STEP 2: Generating forecasts")
        print("="*50)
        from src.predict import predict
        predict(use_5_stations=args.quick, checkpoint_path=args.checkpoint)

    if args.mode in ("dashboard", "all"):
        print("\n" + "="*50)
        print(" STEP 3: Starting dashboard")
        print("="*50)
        from src.dashboard import run_dashboard
        run_dashboard()


if __name__ == "__main__":
    main()
