"""
Главная точка входа для проекта TABD TFT.

Использование:
    python run.py --mode train           # Обучить модель TFT
    python run.py --mode predict         # Сгенерировать прогнозы (нужна обученная модель)
    python run.py --mode evaluate        # Только метрики (TFT + baselines), без перегенерации прогнозов
    python run.py --mode dashboard       # Запустить дашборд (только данные, без модели)
    python run.py --mode all             # Обучение → прогнозы → дашборд
    python run.py --mode train --quick   # Быстрый тест на 5 АЗС
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="TABD TFT Pipeline")
    parser.add_argument(
        "--mode", choices=["train", "predict", "evaluate", "dashboard", "all"],
        default="dashboard", help="Режим работы pipeline"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Использовать срез из 5 АЗС для быстрого тестирования"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Путь к чекпоинту модели (для режима predict)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        import src.config as cfg
        cfg.USE_5_STATIONS = True
        print("[Быстрый режим] Используется срез из 5 АЗС.")

    if args.mode in ("train", "all"):
        print("\n" + "="*50)
        print(" ШАГ 1: Обучение модели TFT")
        print("="*50)
        from src.train import train
        ckpt = train(use_5_stations=args.quick)
        print(f"\nМодель сохранена: {ckpt}")

    if args.mode in ("predict", "all"):
        print("\n" + "="*50)
        print(" ШАГ 2: Генерация прогнозов")
        print("="*50)
        from src.predict import predict
        predict(use_5_stations=args.quick, checkpoint_path=args.checkpoint)

    if args.mode == "evaluate":
        print("\n" + "="*50)
        print(" Расчёт метрик качества")
        print("="*50)
        from src.predict import evaluate_all
        evaluate_all()

    if args.mode in ("dashboard", "all"):
        print("\n" + "="*50)
        print(" ШАГ 3: Запуск дашборда")
        print("="*50)
        from src.dashboard import run_dashboard
        run_dashboard()


if __name__ == "__main__":
    main()
